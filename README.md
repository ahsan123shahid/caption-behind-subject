# Caption Behind the Subject Video Pipeline

A high-performance, modular backend video processing pipeline that creates the **"Caption Behind the Subject"** 3D depth effect for vertical talking-head reels and shorts. 

Built for **Roosh Tech / bolti.ai**.

---

## 🌟 Overview & Approach

For each video frame, the pipeline executes the following compositing order:
1. **Per-Frame Person Segmentation:** Uses Google MediaPipe Tasks Image Segmenter (`selfie_segmenter.tflite`) with edge feathering (Gaussian alpha matting) to isolate the foreground subject cleanly without jagged edges.
2. **Dynamic Caption Typography:** Renders animated typography layers synced with SRT / transcript timestamps.
3. **Occlusion Compositing:**
   $$\text{Final Frame} = \text{Person} \times \text{Mask} + (\text{Background} + \text{Text Layer}) \times (1 - \text{Mask})$$
   Because the segmented subject is composited on top, the text naturally sits behind the speaker's head/shoulders.
4. **Universal Export:** Merges the original audio track and exports production-ready H.264 MP4 videos using FFmpeg.

---

## 🎨 Supported Templates

### 1. `dynamic` ("Estate Dynamic")
- **Style:** Heavy condensed bold uppercase typography (Impact / Arial Black).
- **Aesthetic:** High-energy vibrant orange (`#FF5500`) + clean white with drop shadow.
- **Badge Element:** Highlighted keyword enclosed in a filled pill badge.
- **Animation:** Kinetic word pop / scale effect.

### 2. `luxe` ("Luxe Premium")
- **Style:** Refined semi-bold sans typography with tracking.
- **Aesthetic:** Cool ocean teal / cyan (`#00E1CC`) + crisp white.
- **Context Element:** Delicate contextual phrase aligned upper-left.
- **Animation:** Smooth fade-in and subtle upward slide easing.

---

## 📂 Project Architecture

The architecture is strictly decoupled and modular, making it effortless to swap segmentation models or add new templates without touching the core engine:

```text
task_caption/
├── configs/
│   ├── dynamic.json          # Style parameters for Estate Dynamic
│   └── luxe.json             # Style parameters for Luxe
├── models/
│   └── selfie_segmenter.tflite # Downloaded automatically
├── segmenters/
│   ├── __init__.py           # Factory (easily plug in RVM, Rembg, SAM2)
│   ├── base.py               # Abstract BaseSegmenter interface
│   └── mediapipe_segmenter.py# MediaPipe Tasks implementation
├── templates/
│   ├── __init__.py           # Template factory & config loader
│   ├── base.py               # BaseTemplate interface
│   ├── utils.py              # Typography, pills, shadows, font loading
│   ├── dynamic_template.py   # Dynamic template renderer
│   └── luxe_template.py      # Luxe template renderer
├── subtitle_parser.py        # SRT & transcript parser with timing
├── pipeline.py               # Core compositing engine & FFmpeg integration
├── main.py                   # Production CLI runner
├── create_sample_demo.py     # Sample talking-head video & SRT generator
└── requirements.txt          # Dependencies
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- FFmpeg installed and available on system `PATH`

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/your-username/caption-behind-subject.git
cd caption-behind-subject

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🎬 How to Run

### Run on your video:
```bash
# Dynamic Style
python main.py --video your_video.mp4 --srt captions.srt --template dynamic --output output_dynamic.mp4

# Luxe Style
python main.py --video your_video.mp4 --srt captions.srt --template luxe --output output_luxe.mp4
```

### Run Quick Verification Demo (Offline):
If you don't have a sample video ready, run the built-in generator:
```bash
# 1. Generate sample talking-head video and SRT
python create_sample_demo.py

# 2. Render both styles
python main.py --video sample_video.mp4 --srt sample_captions.srt --template dynamic --output demo_dynamic.mp4
python main.py --video sample_video.mp4 --srt sample_captions.srt --template luxe --output demo_luxe.mp4
```

---

## 🔧 CLI Options

| Argument | Shorthand | Description | Default |
|---|---|---|---|
| `--video` | `-v` | Path to talking-head MP4 video | *(Required)* |
| `--srt` | `-s` | Path to subtitles SRT file | *(Required)* |
| `--template`| `-t` | Style name: `dynamic` or `luxe` | `dynamic` |
| `--output` | `-o` | Output file path | `<input>_behind_<template>.mp4` |
| `--segmenter`| | Segmentation model (`mediapipe`) | `mediapipe` |
| `--max_frames`| | Limit frame count for quick testing | `None` (full video) |

---

## 💡 Extensibility (GPU / RVM Ready)

To add **Robust Video Matting (RVM)** or GPU-accelerated segmentation, simply inherit from `BaseSegmenter`:
```python
from segmenters.base import BaseSegmenter

class RVMSegmenter(BaseSegmenter):
    def segment(self, frame_bgr):
        # Your GPU inference code here
        return alpha_mask
```
Register it in `segmenters/__init__.py` and pass `--segmenter rvm`. No changes needed in templates or rendering logic!
