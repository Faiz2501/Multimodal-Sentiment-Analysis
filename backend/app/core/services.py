from functools import lru_cache
from pathlib import Path
import os
import torch
from mmsa.predict import Predictor
from mmsa.transcribe import LocalTranscriber

DEFAULT_MODEL_DIR=Path(__file__).resolve().parents[3]/"artifacts"/"final"
MODEL_DIR=Path(os.getenv("MSA_MODEL_DIR",str(DEFAULT_MODEL_DIR))).resolve()
ASR_MODEL=os.getenv("MSA_ASR_MODEL","openai/whisper-tiny")

@lru_cache(maxsize=1)
def predictor():
    required=[MODEL_DIR/"best_model.pt",MODEL_DIR/"config.json",MODEL_DIR/"tokenizer"/"tokenizer.json"]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"Trained model bundle is incomplete under {MODEL_DIR}")
    return Predictor(MODEL_DIR)

@lru_cache(maxsize=1)
def transcriber():
    return LocalTranscriber(ASR_MODEL)

def status():
    required=[MODEL_DIR/"best_model.pt",MODEL_DIR/"config.json",MODEL_DIR/"tokenizer"/"tokenizer.json"]
    ready=all(path.is_file() for path in required)
    return {"status":"ok","model_ready":ready,
            "device":"cuda" if torch.cuda.is_available() else "cpu",
            "asr_model":ASR_MODEL}
