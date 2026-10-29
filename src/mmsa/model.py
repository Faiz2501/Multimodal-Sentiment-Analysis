import torch
from torch import nn
from transformers import AutoModel, Wav2Vec2Model


class SpeechTextFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(config['text_model'])
        self.audio_encoder = Wav2Vec2Model.from_pretrained(config['audio_model'])
        # MELD has short clips, so Wav2Vec2 pretraining masks can exceed sequence length.
        self.audio_encoder.config.mask_time_prob = 0.0
        self.audio_encoder.config.mask_feature_prob = 0.0

        # Fine-tune only the final encoder blocks to adapt to sentiment without
        # destabilizing all pretrained representations on this relatively small set.
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.audio_encoder.parameters():
            parameter.requires_grad = False
        text_count = int(config.get('unfreeze_text_layers', 0))
        audio_count = int(config.get('unfreeze_audio_layers', 0))
        if text_count:
            for layer in self.text_encoder.transformer.layer[-text_count:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
        if audio_count:
            for layer in self.audio_encoder.encoder.layers[-audio_count:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
            for parameter in self.audio_encoder.encoder.layer_norm.parameters():
                parameter.requires_grad = True

        dimension = int(config.get('hidden_dim', 256))
        dropout = float(config.get('dropout', 0.3))
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, dimension),
            nn.LayerNorm(dimension), nn.GELU(), nn.Dropout(dropout),
        )
        self.audio_projection = nn.Sequential(
            nn.Linear(self.audio_encoder.config.hidden_size, dimension),
            nn.LayerNorm(dimension), nn.GELU(), nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(nn.Linear(dimension * 2, dimension), nn.Sigmoid())
        self.fusion = nn.Sequential(
            nn.Linear(dimension * 4, dimension * 2), nn.LayerNorm(dimension * 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(dimension * 2, dimension),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.sentiment = nn.Linear(dimension, 1)
        self.sentiment_logits = nn.Linear(dimension, 3)
        self.emotions = nn.Linear(dimension, len(config.get('emotion_labels', [
            'happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral'
        ])))
        self.sarcasm = nn.Linear(dimension, 1)

    def forward(self, input_ids, attention_mask, audio_values, audio_lengths=None):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_mask = attention_mask.unsqueeze(-1).to(text_outputs.dtype)
        text_vector = (text_outputs * text_mask).sum(1) / text_mask.sum(1).clamp(min=1)

        audio_outputs = self.audio_encoder(input_values=audio_values).last_hidden_state
        if audio_lengths is not None:
            frame_lengths = audio_lengths
            for kernel, stride in zip(self.audio_encoder.config.conv_kernel, self.audio_encoder.config.conv_stride):
                frame_lengths = torch.div(frame_lengths - kernel, stride, rounding_mode='floor') + 1
            frame_ids = torch.arange(audio_outputs.shape[1], device=audio_outputs.device).unsqueeze(0)
            audio_mask = (frame_ids < frame_lengths.unsqueeze(1)).unsqueeze(-1).to(audio_outputs.dtype)
            audio_vector = (audio_outputs * audio_mask).sum(1) / audio_mask.sum(1).clamp(min=1)
        else:
            audio_vector = audio_outputs.mean(1)

        text_vector = self.text_projection(text_vector)
        audio_vector = self.audio_projection(audio_vector)
        gate = self.gate(torch.cat([text_vector, audio_vector], -1))
        fused = self.fusion(torch.cat([
            gate * text_vector, (1 - gate) * audio_vector,
            torch.abs(text_vector - audio_vector), text_vector * audio_vector,
        ], -1))
        return {
            'sentiment': self.sentiment(fused).squeeze(-1),
            'sentiment_logits': self.sentiment_logits(fused),
            'emotions': self.emotions(fused),
            'sarcasm': self.sarcasm(fused).squeeze(-1),
        }
