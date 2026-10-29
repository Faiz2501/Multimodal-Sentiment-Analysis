import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from mmsa.data import UtteranceDataset, collate_batch, load_split
from mmsa.model import SpeechTextFusion


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(frame, tokenizer, config, shuffle):
    dataset = UtteranceDataset(
        frame, config['sample_rate'], config['max_audio_seconds'],
        augment=shuffle and config.get('audio_augmentation', False),
    )
    return DataLoader(
        dataset, batch_size=config['batch_size'], shuffle=shuffle, num_workers=0,
        collate_fn=lambda rows: collate_batch(rows, tokenizer, config['max_text_tokens']),
    )


def move(batch, device):
    lengths = torch.tensor([len(item) for item in batch['audio']], device=device, dtype=torch.long)
    audio = pad_sequence([item.to(device) for item in batch['audio']], batch_first=True)
    audio = audio / audio.abs().amax(1, keepdim=True).clamp(min=1.0)
    return {
        'input_ids': batch['input_ids'].to(device),
        'attention_mask': batch['attention_mask'].to(device),
        'audio_values': audio,
        'audio_lengths': lengths,
        'sentiment': batch['sentiment'].to(device),
        'emotions': batch['emotions'].to(device),
        'sarcasm': batch['sarcasm'].to(device),
    }


def sentiment_targets(values):
    return torch.where(values < -0.5, 0, torch.where(values > 0.5, 2, 1)).long()


def loss_fn(predictions, batch, config, device):
    targets = sentiment_targets(batch['sentiment'])
    class_weights = torch.tensor(config['sentiment_class_weights'], device=device, dtype=torch.float32)
    class_loss = F.cross_entropy(predictions['sentiment_logits'].float(), targets, weight=class_weights)
    regression_loss = F.smooth_l1_loss(predictions['sentiment'].float(), batch['sentiment'] / 3.0)
    emotion_targets = batch['emotions']
    emotion_mask = torch.isfinite(emotion_targets)
    pos_weight = torch.tensor(config['emotion_pos_weight'], device=device, dtype=torch.float32)
    if emotion_mask.any():
        emotion_loss = F.binary_cross_entropy_with_logits(
            predictions['emotions'].float()[emotion_mask], emotion_targets[emotion_mask],
            pos_weight=pos_weight.expand_as(emotion_targets)[emotion_mask],
        )
    else:
        emotion_loss = predictions['emotions'].sum() * 0
    if config.get('sarcasm_enabled', False):
        sarcasm_target = batch['sarcasm']
        sarcasm_mask = torch.isfinite(sarcasm_target)
        sarcasm_loss = F.binary_cross_entropy_with_logits(
            predictions['sarcasm'].float()[sarcasm_mask], sarcasm_target[sarcasm_mask]
        ) if sarcasm_mask.any() else predictions['sarcasm'].sum() * 0
    else:
        sarcasm_loss = predictions['sarcasm'].sum() * 0
    return class_loss + float(config.get('regression_loss_weight', 0.25)) * regression_loss + 0.5 * emotion_loss + 0.25 * sarcasm_loss


@torch.no_grad()
def validate(model, loader, device, config):
    model.eval()
    losses, targets, predicted, scores = [], [], [], []
    emotion_targets, emotion_predictions = [], []
    use_amp = device.type == 'cuda' and config.get('mixed_precision', True)
    for raw_batch in loader:
        batch = move(raw_batch, device)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            output = model(batch['input_ids'], batch['attention_mask'], batch['audio_values'], batch['audio_lengths'])
            loss = loss_fn(output, batch, config, device)
        losses.append(float(loss.item()))
        targets.extend(sentiment_targets(batch['sentiment']).cpu().tolist())
        predicted.extend(output['sentiment_logits'].argmax(-1).cpu().tolist())
        scores.extend((output['sentiment'].float().cpu().clamp(-1, 1) * 3).tolist())
        known = torch.isfinite(batch['emotions']).all(1)
        if known.any():
            emotion_targets.extend(batch['emotions'][known].argmax(-1).cpu().tolist())
            emotion_predictions.extend(output['emotions'][known].argmax(-1).cpu().tolist())
    original_scores = np.asarray([[-3.0, 0.0, 3.0][target] for target in targets])
    return {
        'loss': float(np.mean(losses)),
        'sentiment_accuracy': float(accuracy_score(targets, predicted)),
        'sentiment_macro_f1': float(f1_score(targets, predicted, average='macro', zero_division=0)),
        'sentiment_mae': float(mean_absolute_error(original_scores, scores)),
        'emotion_macro_f1': float(f1_score(emotion_targets, emotion_predictions, average='macro', zero_division=0)) if emotion_targets else 0.0,
    }


def derive_training_weights(frame, config):
    sentiment = frame['sentiment'].astype(float).to_numpy()
    labels = np.where(sentiment < -0.5, 0, np.where(sentiment > 0.5, 2, 1))
    counts = np.bincount(labels, minlength=3).astype(float)
    weights = np.sqrt(counts.sum() / (3 * np.maximum(counts, 1)))
    config['sentiment_class_weights'] = (weights / weights.mean()).tolist()
    emotion_columns = [f"emotion_{name}" for name in config['emotion_labels']]
    positives = frame[emotion_columns].fillna(0).to_numpy(dtype=float).sum(0)
    negatives = len(frame) - positives
    config['emotion_pos_weight'] = np.clip(np.sqrt(negatives / np.maximum(positives, 1)), 1.0, 5.0).tolist()
    print('Train sentiment counts (negative, neutral, positive):', counts.astype(int).tolist())
    print('Sentiment class weights:', [round(value, 3) for value in config['sentiment_class_weights']])
    print('Emotion positive weights:', [round(value, 3) for value in config['emotion_pos_weight']])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--config', default='configs/default.json')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    seed_all(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}; text blocks unfrozen: {config.get("unfreeze_text_layers", 0)}; audio blocks unfrozen: {config.get("unfreeze_audio_layers", 0)}')

    training_frame = load_split(args.manifest, 'train')
    validation_frame = load_split(args.manifest, 'validation')
    derive_training_weights(training_frame, config)
    tokenizer = AutoTokenizer.from_pretrained(config['text_model'])
    model = SpeechTextFusion(config).to(device)
    start_epoch, best_f1, best_mae, stale_epochs = 0, -1.0, float('inf'), 0
    if args.resume:
        checkpoint = torch.load(output_dir / 'latest.pt', map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint['best_f1']
        best_mae = checkpoint['best_mae']
        stale_epochs = checkpoint.get('stale_epochs', 0)

    train_loader = make_loader(training_frame, tokenizer, config, True)
    validation_loader = make_loader(validation_frame, tokenizer, config, False)
    encoder_parameters, head_parameters = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (encoder_parameters if name.startswith(('text_encoder.', 'audio_encoder.')) else head_parameters).append(parameter)
    optimizer = torch.optim.AdamW([
        {'params': encoder_parameters, 'lr': float(config.get('encoder_learning_rate', 1e-5))},
        {'params': head_parameters, 'lr': float(config['learning_rate'])},
    ], weight_decay=float(config['weight_decay']))
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
    use_amp = device.type == 'cuda' and config.get('mixed_precision', True)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    accumulation = max(1, int(config.get('gradient_accumulation_steps', 1)))
    patience = int(config.get('early_stopping_patience', 3))

    for epoch in range(start_epoch, int(config['epochs'])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for step, raw_batch in enumerate(tqdm(train_loader, desc=f'epoch {epoch + 1}')):
            batch = move(raw_batch, device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                predictions = model(batch['input_ids'], batch['attention_mask'], batch['audio_values'], batch['audio_lengths'])
                loss = loss_fn(predictions, batch, config, device)
            scaler.scale(loss / accumulation).backward()
            total_loss += float(loss.detach().item())
            if (step + 1) % accumulation == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        metrics = validate(model, validation_loader, device, config)
        metrics['train_loss'] = total_loss / max(1, len(train_loader))
        print({'epoch': epoch + 1, **metrics})
        current_f1 = metrics['sentiment_macro_f1']
        improved = current_f1 > best_f1 or (np.isclose(current_f1, best_f1) and metrics['sentiment_mae'] < best_mae)
        if improved:
            best_f1, best_mae, stale_epochs = current_f1, metrics['sentiment_mae'], 0
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            tokenizer.save_pretrained(output_dir / 'tokenizer')
            (output_dir / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
            (output_dir / 'labels.json').write_text(json.dumps({
                'sentiment_classes': ['negative', 'neutral', 'positive'],
                'emotions': config['emotion_labels'],
                'sarcasm': ['not_sarcastic', 'sarcastic'],
            }, indent=2), encoding='utf-8')
            (output_dir / 'best_validation.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        else:
            stale_epochs += 1
        torch.save({
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'epoch': epoch, 'best_f1': best_f1, 'best_mae': best_mae,
            'stale_epochs': stale_epochs, 'config': config,
        }, output_dir / 'latest.pt')
        if stale_epochs >= patience:
            print(f'Early stopping after {epoch + 1} epochs; validation macro-F1 did not improve for {patience} epochs.')
            break
    print(f'Training complete. Best validation macro-F1={best_f1:.4f}; artifact: {output_dir.resolve()}')


if __name__ == '__main__':
    main()
