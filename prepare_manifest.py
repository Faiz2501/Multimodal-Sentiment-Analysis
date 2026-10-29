import argparse
from pathlib import Path
import pandas as pd
REQUIRED=['sample_id','audio_path','text','sentiment','speaker_id','group_id','split']
OPTIONAL=['emotion_happiness','emotion_sadness','emotion_anger','emotion_fear','emotion_disgust','emotion_surprise','emotion_neutral','sarcasm']
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    source=Path(a.input).resolve(); df=pd.read_csv(source); missing=sorted(set(REQUIRED)-set(df.columns))
    if missing: raise SystemExit('Missing required columns: '+', '.join(missing))
    for col in OPTIONAL:
        if col not in df: df[col]=float('nan')
    df['audio_path']=df.audio_path.map(lambda p:str((source.parent/str(p)).resolve()) if not Path(str(p)).is_absolute() else str(Path(str(p)).resolve()))
    df['text']=df.text.fillna('').astype(str); df['sentiment']=pd.to_numeric(df.sentiment,errors='coerce'); df=df.dropna(subset=['sample_id','sentiment','speaker_id','group_id','split'])
    df['split']=df.split.astype(str).str.lower().replace({'val':'validation','valid':'validation','dev':'validation'})
    if not set(df.split).issubset({'train','validation','test'}): raise SystemExit('split must be train, validation, or test')
    if ((df.sentiment < -3)|(df.sentiment > 3)).any(): raise SystemExit('sentiment scores must be in [-3,3]')
    for col in OPTIONAL:
        df[col]=pd.to_numeric(df[col],errors='coerce')
        if len(df[col].dropna().loc[lambda x:~x.isin([0,1])]): raise SystemExit(f'{col} must be binary 0/1 or blank')
    overlap=df.groupby('group_id').split.nunique()
    if (overlap>1).any(): raise SystemExit(f'Group leakage: {(overlap>1).sum()} groups cross splits')
    missing_audio=[p for p in df.audio_path if not Path(p).is_file()]
    if missing_audio: raise SystemExit(f'Audio file not found (first): {missing_audio[0]}')
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); df.to_csv(out,index=False); print(f'Wrote {len(df)} rows to {out}\n{df.groupby("split").size()}')
if __name__=='__main__': main()
