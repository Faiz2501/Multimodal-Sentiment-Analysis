import numpy as np
import librosa
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset
EMOTIONS=['happiness','sadness','anger','fear','disgust','surprise','neutral']
def read_audio(path,sample_rate=16000,max_seconds=12.0):
    audio,sr=sf.read(str(path),dtype='float32',always_2d=True); audio=audio.mean(axis=1)
    if sr!=sample_rate: audio=librosa.resample(audio,orig_sr=sr,target_sr=sample_rate)
    audio=audio[:int(sample_rate*max_seconds)]
    if len(audio)==0: audio=np.zeros(sample_rate//2,dtype=np.float32)
    peak=np.max(np.abs(audio))
    if peak>1e-6: audio=audio/max(1.0,peak)
    return audio.astype(np.float32)
def number(v):
    try:
        n=float(v); return n if np.isfinite(n) else float('nan')
    except (TypeError,ValueError): return float('nan')
class UtteranceDataset(Dataset):
    def __init__(self,frame,sample_rate=16000,max_seconds=12.0,augment=False): self.frame=frame.reset_index(drop=True); self.sample_rate=sample_rate; self.max_seconds=max_seconds; self.augment=augment
    def __len__(self): return len(self.frame)
    def __getitem__(self,index):
        row=self.frame.iloc[index]; audio=read_audio(row.audio_path,self.sample_rate,self.max_seconds)
        if self.augment:
            if np.random.random()<.5: audio*=np.random.uniform(.65,1.25)
            if np.random.random()<.45: audio+=np.random.normal(0,np.random.uniform(.001,.012),len(audio)).astype(np.float32)
            if np.random.random()<.25: audio=librosa.effects.time_stretch(audio,rate=np.random.uniform(.92,1.08)).astype(np.float32)[:int(self.sample_rate*self.max_seconds)]
        labels={k:number(row.get('emotion_'+k)) for k in EMOTIONS}; labels['sarcasm']=number(row.get('sarcasm'))
        return {'sample_id':str(row.sample_id),'text':str(row.text),'audio':torch.from_numpy(audio),'sentiment':torch.tensor(float(row.sentiment),dtype=torch.float32),'emotions':torch.tensor([labels[k] for k in EMOTIONS],dtype=torch.float32),'sarcasm':torch.tensor(labels['sarcasm'],dtype=torch.float32)}
def collate_batch(rows,tokenizer,max_text_tokens):
    enc=tokenizer([r['text'] for r in rows],padding=True,truncation=True,max_length=max_text_tokens,return_tensors='pt')
    return {'sample_ids':[r['sample_id'] for r in rows],'input_ids':enc['input_ids'],'attention_mask':enc['attention_mask'],'audio':[r['audio'] for r in rows],'sentiment':torch.stack([r['sentiment'] for r in rows]),'emotions':torch.stack([r['emotions'] for r in rows]),'sarcasm':torch.stack([r['sarcasm'] for r in rows])}
def load_split(manifest,split):
    frame=pd.read_csv(manifest)
    if 'split' not in frame: raise ValueError('Manifest needs split column')
    frame=frame[frame.split==split]
    if frame.empty: raise ValueError(f'No rows for split={split}')
    return frame
