# Colab training workflow

Use a free Colab GPU runtime when available. Upload the source project (not MELD audio) to the notebook, install the Python requirements, download MELD from its official source directly into the runtime, extract annotations/audio, create the processed manifest, and run `mmsa.train` with `PYTHONPATH=src`. Save checkpoints to Google Drive only if you choose to connect Drive; otherwise download the resulting `artifacts/final` directory before the runtime ends. Verify `torch.cuda.is_available()` before training. Colab GPU availability and session duration are not guaranteed.

For the reproducible project-specific commands, use the root README. Keep all dataset and checkpoints out of a public GitHub repository unless redistribution terms have been reviewed.
