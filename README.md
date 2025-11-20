# Smart Audio Trimmer

A utility for intelligent trimming of audio files using the Whisper model, synchronizing trimming of pairs of files: original and diarized.

---

## Description

The script loads audio files from the `input_audio` folder, uses Whisper to transcribe and detect speech segments, then trims audio files to a minimum and maximum duration (default 60 and 120 seconds).

Key feature — if a pair of audio files exists (e.g., original and diarized with suffix `__diarized`), the script trims both files synchronously to maintain equal length.

---

## Installation

1. Create and activate a virtual environment:
python3 -m venv venv
source venv/bin/activate # Linux/macOS
venv\Scripts\activate.bat # Windows cmd


2. Install dependencies:
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org pydub openai-whisper numpy audioop-lts


3. Install [ffmpeg](https://ffmpeg.org/download.html) (required by pydub):

Linux (Ubuntu/Debian)

sudo apt install ffmpeg
macOS

brew install ffmpeg


3. Trimmed files will appear in the `output_audio` folder.

4. Logs with trimming info are saved under `output_audio/logs`.

---

## Configuration

- `MIN_DURATION` — minimum length for saved audio (seconds, default 60)
- `MAX_DURATION` — maximum audio length (default 120)
- `MODEL_SIZE` — Whisper model size: tiny, base, small, medium, large (default base)

These can be configured in the `main()` function of `smart_audio_trim.py`.

---

## Features

- Intelligent trimming based on speech transcription preserving sentence integrity.
- Synchronized trimming of paired original and diarized audio files.
- Support for trimming individual files without pairs.
- Silence trimming support can be further enhanced.

---

## Requirements

- Python 3.12 or 3.13
- ffmpeg
- Python libraries: pydub, whisper, numpy, audioop-lts

---


