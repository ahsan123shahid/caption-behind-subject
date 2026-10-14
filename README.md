# Caption Behind the Subject

A modular, backend video processing pipeline that renders stylized captions **behind the subject** (`background → text layer → segmented subject on top → optional front subtitles`), creating a depth / 3D look.

Built for **Roosh Tech / bolti.ai** task requirements.

---

## Features
- **3D Depth Occlusion**: Text sits behind the subject so the person's head/body naturally occludes it.
- **Auto-Head Tracking**: Automatically tracks the subject's crown position so text aligns with the person and bottom edges naturally touch/tuck behind the subject.
- **2 Ready-to-Use Templates**:
  - `dynamic` — Bold & energetic: Heavy condensed uppercase font (`Anton`), vibrant yellow/orange pop animation with bounce overshoot.
  - `luxe` — Elegant & premium: Refined semi-bold sans (`Montserrat`), cool cyan/teal accent with smooth fade & slide.
- **Swappable Segmentation**: Modular architecture supports `mediapipe` (fast CPU), `yolo` (YOLOv8 segmentation), `rembg`, and `dummy`.
- **Zero-Lag Speech Sync**: Exact speech alignment powered by OpenAI Whisper word timestamps.
- **Data-Driven**: Fully configured via JSON files in `behind_caption/templates/`. Add new templates without touching code.

---

## Project Structure
```text
behind_caption/
├── cli.py             # CLI entrypoint (--video, --srt, --template, --segmenter)
├── transcriber.py     # Auto Speech-to-Text via Whisper AI & Hero Keyword extraction
├── bolti.py           # bolti.ai / Kaizynn project JSON -> SRT (keeps emphasis marks)
├── pipeline.py        # Frame-by-frame compositing loop, FFmpeg stream pipe
├── renderer.py        # Text layout engine, auto-fitting, pop/slide animations
├── segmentation.py    # Swappable segmenter registry (mediapipe, yolo, rembg)
├── captions.py        # SRT parser with millisecond precision
└── templates/         # JSON style configurations
    ├── dynamic.json   # Template 1: Dynamic (Bold / Energetic)
    └── luxe.json      # Template 2: Luxe (Elegant / Premium)
```

---

## Installation

1. **Install requirements** (Python 3.10+):
```bash
pip install -r requirements.txt
```
*(Requires `ffmpeg` installed on your system PATH).*

### Optional: Cloud Acceleration with Groq API
For near-instant speech transcription (<2s) with Whisper-Large-v3, set your Groq key (or add to `.env`):
```bash
export GROQ_API_KEY="your_groq_key"       # Linux/macOS
set GROQ_API_KEY=your_groq_key            # Windows
```
*(If no API key is set, the system automatically falls back to local Whisper).*

---

## Usage

### 1. Automatic Run (No SRT Needed — Auto Whisper AI)
Simply pass the video without `--srt`. Whisper AI will automatically transcribe speech, compute exact word timings, tag hero keywords, and render the final video:
```bash
python -m behind_caption myvideo.mp4 --template dynamic --out output_dynamic.mp4
```

### 2. Custom SRT Run
If you already have a transcribed `.srt` file:
```bash
python -m behind_caption myvideo.mp4 --srt samples/Video.srt --template dynamic --out output_dynamic.mp4
```

### 3. bolti.ai / Kaizynn JSON Transcript Run
Already have the project JSON exported from bolti.ai / Kaizynn (`{"project": {"transcript": [...], "clips": [...]}}`)? Feed it straight in — word timings are preserved, the preset's caption clips become the cue boundaries, and every word flagged `emphasis: true` is wrapped in `*asterisks*` as a Hero Keyword:
```bash
python -m behind_caption myvideo.mp4 --srt samples/bolti_transcript.json --template dynamic --out output_dynamic.mp4
```
Or only export the SRT:
```bash
python -m behind_caption.bolti samples/bolti_transcript.json -o samples/bolti_transcript.srt
```
If the payload has no `clips` array, cues are re-chunked with the auto-transcriber rhythm (`--words-per-cue`, `--max-duration`); add `--plain` to skip the `*hero*` marks.

### 4. Render Luxe Template
```bash
python -m behind_caption myvideo.mp4 --template luxe --out output_luxe.mp4
```

### 5. Render All Templates in One Command
```bash
python -m behind_caption myvideo.mp4 --template all --out outputs/
```

### Options
- `--template`: `dynamic`, `luxe`, or `all` (default: `dynamic`)
- `--segmenter`: `mediapipe`, `yolo`, `rembg` (default: `mediapipe`)
- `--smooth`: Temporal mask smoothing factor (default: `0.5`)
- `--crf`: Video quality factor (default: `18`)

In the SRT file, wrap a keyword in `*asterisks*` (e.g. `yaar *1 MILLION* followers`) to designate it as the hero keyword behind the subject. Any remaining words appear cleanly as front subtitle text at the bottom.
