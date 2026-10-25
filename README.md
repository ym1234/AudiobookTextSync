# AudiobookTextSync
Hey! I'll write a detailed README later but for now:

## Installation
```bash
uv sync
# or
uv sync --extra cuda # for NVIDIA GPUs
```
Also needs ffmpeg and a C compiler installed.

## Usage
The command line interface is currently in flux while I try to figure out a good way to structure it but for now you can use
```bash
uv run ats sync --text fst.epub --text snd.epub --text etc.epub --audio [any number of audio files] --output-dir subs/
```
