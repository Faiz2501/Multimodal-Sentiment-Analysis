from pathlib import Path
import subprocess
import tempfile
from contextlib import contextmanager
import imageio_ffmpeg

MAX_UPLOAD_BYTES=50*1024*1024
MAX_AUDIO_SECONDS=12

@contextmanager
def as_wav(upload_bytes: bytes, filename: str):
    """Decode a browser recording or common audio file to temporary 16 kHz mono WAV."""
    if len(upload_bytes)>MAX_UPLOAD_BYTES:
        raise ValueError("Audio upload exceeds the 50 MB limit")
    suffix=Path(filename or "recording.wav").suffix.lower()
    if suffix not in {".wav",".flac",".mp3",".m4a",".mp4",".webm",".ogg",".aac"}:
        suffix=".bin"
    with tempfile.TemporaryDirectory(prefix="msa-audio-") as folder:
        source=Path(folder)/("input"+suffix); output=Path(folder)/"audio.wav"
        source.write_bytes(upload_bytes)
        command=[imageio_ffmpeg.get_ffmpeg_exe(),"-y","-v","error","-i",str(source),"-t",str(MAX_AUDIO_SECONDS),"-vn","-ac","1","-ar","16000","-c:a","pcm_s16le",str(output)]
        try:
            subprocess.run(command,check=True,capture_output=True,timeout=90)
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:
            raise ValueError("Could not decode this audio. Try a WAV, FLAC, MP3, M4A, or WebM recording under 60 seconds.") from exc
        if not output.exists() or output.stat().st_size<44:
            raise ValueError("The uploaded file contains no decodable audio")
        yield output
