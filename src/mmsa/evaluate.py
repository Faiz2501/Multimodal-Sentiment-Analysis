import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from mmsa.data import EMOTIONS, UtteranceDataset, collate_batch, load_split
from mmsa.model import SpeechTextFusion


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--split', default='test')
    args = parser.parse_args()
    root = Path(args.model_dir)
    config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(root / 'tokenizer')
    model = SpeechTextFusion(config).to(device)
    model.load_state_dict(torch.load(root / 'best_model.pt', map_location=device, weights_only=True))
    model.eval()
    dataset = UtteranceDataset(load_split(args.manifest, args.split), config['sample_rate'], config['max_audio_seconds'])
    loader = DataLoader(dataset, batch_size=config['batch_size'], collate_fn=lambda rows: collate_batch(rows, tokenizer, config['max_text_tokens']))
    sentiment_targets, sentiment_predictions, scores = [], [], []
    emotion_targets, emotion_predictions = [], []
    use_amp = device.type == 'cuda' and config.get('mixed_precision', True)
    for batch in loader:
        audio = pad_sequence(batch['audio'], batch_first=True).to(device)
        audio = audio / audio.abs().amax(1, keepdim=True).clamp(min=1.0)
        lengths = torch.tensor([len(item) for item in batch['audio']], device=device)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            output = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), audio, lengths)
        target = batch['sentiment']
        sentiment_targets.extend(torch.where(target < -0.5, 0, torch.where(target > 0.5, 2, 1)).tolist())
        sentiment_predictions.extend(output['sentiment_logits'].argmax(-1).cpu().tolist())
        scores.extend((output['sentiment'].float().cpu().clamp(-1, 1) * 3).tolist())
        known = torch.isfinite(batch['emotions']).all(1)
        if known.any():
            emotion_targets.extend(batch['emotions'][known].argmax(-1).tolist())
            emotion_predictions.extend(output['emotions'][known].argmax(-1).cpu().tolist())
    actual_scores = np.asarray([[-3.0, 0.0, 3.0][label] for label in sentiment_targets])
    report = {
        'sentiment_accuracy': float(accuracy_score(sentiment_targets, sentiment_predictions)),
        'sentiment_macro_f1': float(f1_score(sentiment_targets, sentiment_predictions, average='macro', zero_division=0)),
        'sentiment_mae': float(mean_absolute_error(actual_scores, scores)),
        'emotion_accuracy': float(accuracy_score(emotion_targets, emotion_predictions)),
        'emotion_macro_f1': float(f1_score(emotion_targets, emotion_predictions, average='macro', zero_division=0)),
        'test_examples': len(sentiment_targets),
    }
    print(json.dumps(report, indent=2))
    print('Sentiment confusion matrix (negative, neutral, positive):\n', confusion_matrix(sentiment_targets, sentiment_predictions, labels=[0, 1, 2]))
    print('Sentiment report:\n', classification_report(sentiment_targets, sentiment_predictions, labels=[0, 1, 2], target_names=['negative', 'neutral', 'positive'], zero_division=0))
    print('Emotion report:\n', classification_report(emotion_targets, emotion_predictions, labels=list(range(len(config['emotion_labels']))), target_names=config['emotion_labels'], zero_division=0))
    (root / f'{args.split}_metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
