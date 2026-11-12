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

The released checkpoint was trained on MELD's official train split, selected using validation data, and evaluated on the held-out official test split (2,610 utterances). Results: sentiment accuracy **69.46%**, sentiment macro-F1 **65.70%**, sentiment MAE **1.417** on the configured −3 to +3 score scale; seven-way emotion accuracy **60.92%** and macro-F1 **35.80%**. Fear and disgust remain difficult classes. These MELD results do not establish performance on natural conversations, different accents, recording conditions, or sarcasm.

The model predicts sentiment and emotions only; sarcasm output is disabled because MELD does not provide a reliable sarcasm target for this training setup. Whisper transcription runs locally, and the supplied transcript can be edited before analysis.

### Download the trained model

Download the [v1.0.0 model bundle](https://github.com/Faiz2501/Multimodal-Sentiment-Analysis/releases/download/v1.0.0/multimodal-sentiment-analysis-model.zip) and extract it into `artifacts/final/`:

```powershell
New-Item -ItemType Directory -Force artifacts/final | Out-Null
Invoke-WebRequest "https://github.com/Faiz2501/Multimodal-Sentiment-Analysis/releases/download/v1.0.0/multimodal-sentiment-analysis-model.zip" -OutFile "$env:TEMP\mmsa-model.zip"
Expand-Archive "$env:TEMP\mmsa-model.zip" -DestinationPath artifacts/final -Force
```

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

The source tree includes the compact-checkpoint loader. Download and extract the release bundle above before running the application. The checkpoint contains fine-tuned layers and fusion heads; the matching pretrained DistilBERT and Wav2Vec2 encoder weights download from Hugging Face during first initialization. The first model startup therefore requires an internet connection. Set `HF_HOME` to a writable cache directory if your default Hugging Face cache is not writable.

## Dataset Credits

https://affective-meld.github.io/
