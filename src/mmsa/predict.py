import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from mmsa.data import read_audio
from mmsa.model import SpeechTextFusion


class Predictor:
    def __init__(self, model_dir, device=None):
        self.root = Path(model_dir)
        self.config = json.loads((self.root / 'config.json').read_text(encoding='utf-8'))
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.tokenizer = AutoTokenizer.from_pretrained(self.root / 'tokenizer')
        self.model = SpeechTextFusion(self.config).to(self.device)
        self.model.load_state_dict(torch.load(self.root / 'best_model.pt', map_location=self.device, weights_only=True))
        self.model.eval()

    @torch.no_grad()
    def predict(self, audio_path, text):
        encoded = self.tokenizer(text or '', return_tensors='pt', truncation=True, max_length=self.config['max_text_tokens'])
        audio = torch.from_numpy(read_audio(audio_path, self.config['sample_rate'], self.config['max_audio_seconds'])).unsqueeze(0).to(self.device)
        audio = audio / audio.abs().amax(1, keepdim=True).clamp(min=1.0)
        use_amp = self.device.type == 'cuda' and self.config.get('mixed_precision', True)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            output = self.model(encoded['input_ids'].to(self.device), encoded['attention_mask'].to(self.device), audio, torch.tensor([audio.shape[1]], device=self.device))
        score = float(output['sentiment'].float().cpu().item() * 3)
        classes = ['negative', 'neutral', 'positive']
        sentiment = classes[int(output['sentiment_logits'].argmax(-1).cpu().item())]
        emotions = torch.sigmoid(output['emotions'].float())[0].cpu().tolist()
        sarcasm = float(torch.sigmoid(output['sarcasm'].float())[0].cpu())
        return {
            'sentiment_score': max(-3.0, min(3.0, score)),
            'sentiment': sentiment,
            'emotions': dict(zip(self.config['emotion_labels'], emotions)),
            'sarcasm_probability': sarcasm if self.config.get('sarcasm_enabled', False) else None,
        }
