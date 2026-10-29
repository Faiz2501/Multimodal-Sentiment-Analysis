# Dataset preparation

No raw recordings are included in the deliverable. The project uses the official MELD release for its current speech-text training path because it contains paired utterance clips, transcripts, three sentiment classes, seven emotion labels, and official train/dev/test dialogue splits. Download from the [official MELD page](https://affective-meld.github.io/) and retain it locally. Do not publish the extracted TV clips or include them in a public repository.

MELD is scripted television dialogue. It is useful for a reproducible academic benchmark, but does not establish performance on spontaneous speech, broad accents, or sarcasm. CMU-MOSEI remains a useful follow-on sentiment-intensity benchmark; its standard SDK release provides processed features while raw source videos are not redistributed.

## Convert official MELD release

Extract `MELD.Raw.tar.gz` to a local folder. Download the three annotation CSVs from the official [`declare-lab/MELD` repository](https://github.com/declare-lab/MELD/tree/master/data/MELD) into `data/raw/meld/annotations/`, then run:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_meld.py --meld-root data\raw\meld\MELD.Raw --annotations data\raw\meld\annotations --output data\processed\meld
.\.venv\Scripts\python.exe prepare_manifest.py --input data\processed\meld\manifest.csv --output data\processed\manifest.csv
```

The converter extracts 16 kHz mono WAV files from the utterance videos, maps sentiment to -3/0/+3 for the regression head, maps Joy to Happiness, and emits seven one-hot emotion targets. Missing clips are skipped and should be noted when comparing split counts.

## Manifest

Required columns: `sample_id,audio_path,text,sentiment,speaker_id,group_id,split`. Dialogue group IDs must not cross splits. MELD reuses characters/speakers across official splits, so these splits are group-disjoint by dialogue but not speaker-disjoint. Optional emotion columns are `emotion_happiness`, `emotion_sadness`, `emotion_anger`, `emotion_fear`, `emotion_disgust`, `emotion_surprise`, `emotion_neutral`; `sarcasm` remains blank until separate sarcasm data is prepared.
