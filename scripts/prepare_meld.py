"""Convert official MELD splits into project utterance WAVs and manifest."""
import argparse
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import pandas as pd

SPLITS = {'train': 'train', 'dev': 'validation', 'test': 'test'}
MEDIA_DIRS = {
    'train': 'train_splits',
    'dev': 'dev_splits_complete',
    'test': 'output_repeated_splits_test',
}
EMOTIONS = ['happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral']
SENTIMENT = {'negative': -3.0, 'neutral': 0.0, 'positive': 3.0}
EMOTION = {
    'joy': 'happiness', 'happy': 'happiness', 'sadness': 'sadness', 'anger': 'anger',
    'fear': 'fear', 'disgust': 'disgust', 'surprise': 'surprise', 'neutral': 'neutral',
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--meld-root', required=True, help='Extracted MELD.Raw folder')
    ap.add_argument('--annotations', required=True, help='Folder containing train/dev/test_sent_emo.csv')
    ap.add_argument('--output', required=True, help='Output folder for WAV files and manifest')
    args = ap.parse_args()
    root, ann, out = Path(args.meld_root), Path(args.annotations), Path(args.output)
    audio_dir = out / 'audio'
    audio_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    rows = []
    missing_clips = 0
    unreadable_clips = 0
    invalid_labels = 0

    for csv_split, project_split in SPLITS.items():
        csv_path = ann / f'{csv_split}_sent_emo.csv'
        frame = pd.read_csv(csv_path)
        media_dir = root / MEDIA_DIRS[csv_split]
        if not media_dir.is_dir():
            raise SystemExit(f'Missing media directory: {media_dir}')

        for row in frame.itertuples(index=False):
            filename = f'dia{int(row.Dialogue_ID)}_utt{int(row.Utterance_ID)}.mp4'
            video = media_dir / filename
            if not video.is_file():
                missing_clips += 1
                continue

            audio = audio_dir / f'{csv_split}_{video.stem}.wav'
            if not audio.is_file() or audio.stat().st_size < 44:
                audio.unlink(missing_ok=True)
                command = [ffmpeg, '-y', '-v', 'error', '-i', str(video), '-vn', '-ac', '1',
                           '-ar', '16000', '-sample_fmt', 's16', str(audio)]
                try:
                    subprocess.run(command, check=True, capture_output=True, text=True)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                    audio.unlink(missing_ok=True)
                    unreadable_clips += 1
                    if unreadable_clips <= 20:
                        reason = (exc.stderr or str(exc)).strip().splitlines()[-1]
                        print(f'Skipping unreadable clip {video.name}: {reason}', file=sys.stderr)
                    continue
                if not audio.is_file() or audio.stat().st_size < 44:
                    audio.unlink(missing_ok=True)
                    unreadable_clips += 1
                    continue

            sentiment = str(row.Sentiment).strip().lower()
            emotion = str(row.Emotion).strip().lower()
            if sentiment not in SENTIMENT or emotion not in EMOTION:
                invalid_labels += 1
                continue
            label = EMOTION[emotion]
            targets = {f'emotion_{name}': int(name == label) for name in EMOTIONS}
            rows.append({
                'sample_id': f'{csv_split}_{video.stem}',
                'audio_path': str(audio.resolve()),
                'text': str(row.Utterance),
                'sentiment': SENTIMENT[sentiment],
                'sentiment_label': sentiment,
                'speaker_id': str(row.Speaker),
                'group_id': f'{csv_split}_dialogue_{int(row.Dialogue_ID)}',
                'split': project_split,
                'sarcasm': None,
                **targets,
            })

    if not rows:
        raise SystemExit('No usable MELD utterances were prepared; check the archive and annotations.')
    manifest = out / 'manifest.csv'
    frame = pd.DataFrame(rows)
    frame.to_csv(manifest, index=False)
    print(f'Prepared {len(rows)} utterances: {manifest.resolve()}')
    print(frame.groupby('split').size().to_string())
    print(f'Skipped missing clips: {missing_clips}; unreadable clips: {unreadable_clips}; invalid labels: {invalid_labels}')
    print('The official splits are dialogue-disjoint; speakers may appear in multiple splits.')


if __name__ == '__main__':
    main()
