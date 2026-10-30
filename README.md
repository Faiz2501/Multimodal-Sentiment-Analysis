# Multimodal Sentiment Analysis

Speech and text sentiment analysis project with a multimodal model, training/evaluation pipeline, FastAPI backend, and React frontend.

## Project layout

- `src/mmsa/` — model, data loading, training, evaluation, inference, and transcription
- `backend/` — FastAPI application for health, transcription, and sentiment analysis
- `frontend/` — React + TypeScript user interface
- `configs/` — model and training configuration
- `scripts/` — MELD acquisition preparation and manifest utilities
- `notebooks/` — Colab workflow notes
- `artifacts/final/` — local model checkpoints; model weights are excluded from Git by default
- `data/` — local datasets and derived features; excluded from Git

## Model and dataset

The model combines DistilBERT text representations and Wav2Vec2 speech representations with learned gated fusion. It predicts sentiment and seven emotion classes. Training augmentation applies gain, noise, and small speed changes to training audio only. Transcription uses a locally run Whisper model.

The training pipeline targets the official MELD dataset, which provides aligned audio, transcripts, three sentiment labels, seven emotion labels, and train/dev/test dialogue splits. MELD consists of scripted television dialogue. It does not establish accent robustness or sarcasm detection; these need separate licensed, representative data and dedicated evaluation. Keep raw dataset clips out of GitHub.

## Trained model snapshot

The saved improved checkpoint was trained on the official MELD train split, selected using the validation split, and evaluated once on the held-out official test split (2,610 utterances). Recorded results: sentiment accuracy **69.20%**, sentiment macro-F1 **66.45%**, sentiment MAE **1.325** on the configured −3 to +3 score scale; seven-way emotion accuracy **61.07%** and macro-F1 **35.43%**. The emotion task is substantially harder for rare labels (fear and disgust were not reliably detected). These are dataset-specific results, not guarantees for natural conversations, different accents, recording conditions, or sarcasm.

The model predicts sentiment and emotions only. The sarcasm output is disabled because MELD does not provide a reliable sarcasm target for this training setup. Whisper transcription runs locally; the supplied transcript can also be edited before analysis.

## Install

Python 3.10+ and Node.js 20+ are recommended. Training is intended for a CUDA GPU runtime such as Google Colab; this PC's Radeon RX 5500M is not supported by the current PyTorch CUDA build.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
cd frontend
npm install
```

## Prepare the data

Download MELD from [the official MELD page](https://affective-meld.github.io/), extract it under `data/raw/meld/`, and put the official train/dev/test sentiment annotation CSVs in `data/raw/meld/annotations/`.

```powershell
$env:PYTHONPATH = "$PWD/src"
python scripts/prepare_meld.py --meld-root data/raw/meld/MELD.Raw --annotations data/raw/meld/annotations --output data/processed/meld
python prepare_manifest.py --input data/processed/meld/manifest.csv --output data/processed/manifest.csv
```

MELD's official splits are dialogue-disjoint. Speakers can occur across splits, so do not describe this as speaker-independent evaluation.

## Train and evaluate

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m mmsa.train --manifest data/processed/manifest.csv --output-dir artifacts/final --config configs/default.json
python -m mmsa.evaluate --manifest data/processed/manifest.csv --model-dir artifacts/final
```

Training saves the best validation checkpoint, a resumable checkpoint, model configuration, tokenizer, label map, and validation metrics. Use the held-out test set once for final reporting. Report sentiment MAE/macro-F1, emotion macro-F1 and per-class results, and modality ablations. Evaluate audio corruption robustness separately.

## Run the application

The frontend and API are connected through the Vite `/api` proxy in development. Build the frontend with `npm run build`; when `frontend/dist` exists, FastAPI serves the production UI and API from the same local server.

Start the backend in one terminal:

```powershell
$env:PYTHONPATH = "$PWD/src"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8001
```

Start the frontend in another terminal:

```powershell
cd frontend
npm run dev
```

Open the Vite URL printed by the frontend during development. For a single-server production-style local run, build the frontend and open `http://127.0.0.1:8001`. Audio uploads are decoded locally by the backend and are not persisted. Place the trained bundle in `artifacts/final/`; the health endpoint reports when the weights, config, and tokenizer are ready.

### Checkpoint availability

The application source and compact-checkpoint loader are in this repository, but the trained checkpoint files are not. Until the model bundle is exported from Colab and installed under `artifacts/final/`, `/api/health` reports `model_ready: false` and `/api/analyze` returns HTTP 503. The Colab runtime currently has no saved copy of the export, and free GPU access is temporarily unavailable, so the weights still need to be regenerated or recovered from a downloaded bundle before this app can perform inference.

## License and publication

Review MELD's current terms and cite its authors in the final-year report. This repository contains the implementation only; do not commit MELD files, user recordings, credentials, or large trained weights. Publish model weights separately only after reviewing dataset terms and confirming redistribution rights.

