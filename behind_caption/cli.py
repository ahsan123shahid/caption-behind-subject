import argparse
from pathlib import Path
from .pipeline import run
from .segmentation import REGISTRY
from .templates import list_templates


def main():
    ap = argparse.ArgumentParser(description="Caption-behind-subject renderer")
    ap.add_argument("video")
    ap.add_argument("--srt", default=None, help="path to .srt file (optional; auto-transcribed via Whisper AI if omitted)")
    ap.add_argument("--template", default="dynamic", help=f"one of {list_templates()} or 'all'")
    ap.add_argument("--out", default="out", help="output .mp4 (single template) or directory (all)")
    ap.add_argument("--segmenter", default="mediapipe", choices=list(REGISTRY))
    ap.add_argument("--smooth", type=float, default=0.15, help="temporal smoothing (default: 0.15 to avoid ghosting lag)")
    ap.add_argument("--hero-interval", type=int, default=3, help="pace behind-the-subject hero words every N lines (default: 3)")
    a = ap.parse_args()

    # Automatic Speech-to-Text via Whisper AI if --srt is not provided
    if not a.srt:
        from .transcriber import transcribe_video
        auto_srt = str(Path(a.video).with_suffix(".auto.srt"))
        print(f"[auto-caption] No --srt provided. Transcribing speech from '{a.video}' via Whisper AI...")
        a.srt = transcribe_video(a.video, output_srt=auto_srt)
        print(f"[auto-caption] Using auto-generated SRT: {a.srt}")

    if a.template == "all":
        Path(a.out).mkdir(parents=True, exist_ok=True)
        for t in list_templates():
            run(a.video, a.srt, t, str(Path(a.out) / f"{Path(a.video).stem}_{t}.mp4"), a.segmenter, a.smooth, hero_interval=a.hero_interval)
    else:
        run(a.video, a.srt, a.template, a.out if a.out.endswith(".mp4") else a.out + ".mp4", a.segmenter, a.smooth, hero_interval=a.hero_interval)


if __name__ == "__main__":
    main()
