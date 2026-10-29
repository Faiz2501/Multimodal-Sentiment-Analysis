"""Local optional Whisper transcription; no audio is sent to an API."""
import librosa
import numpy as np
import torch
from transformers import pipeline
from mmsa.data import read_audio

class LocalTranscriber:
    def __init__(self,model_id='openai/whisper-tiny'):
        device=0 if torch.cuda.is_available() else -1
        self.pipe=pipeline('automatic-speech-recognition',model=model_id,device=device)
    def transcribe(self,audio_path,sample_rate=16000,max_seconds=60):
        audio=read_audio(audio_path,sample_rate,max_seconds)
        result=self.pipe({'raw':audio,'sampling_rate':sample_rate},chunk_length_s=20)
        return result['text'].strip()
