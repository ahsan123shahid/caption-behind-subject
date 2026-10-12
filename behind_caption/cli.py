import argparse
from pathlib import Path
from .pipeline import run
from .segmentation import REGISTRY
from .templates import list_templates


def main():
    ap = argparse.ArgumentParser(description="Caption-behind-subject renderer")
    ap.add_argument("video")
    ap.add_argument("--srt", required=True)
    ap.add_argument("--template", default="dynamic", help=f"one of {list_templates()} or 'all'")
    ap.add_argument("--out", default="out", help="output .mp4 (single template) or directory (all)")
    ap.add_argument("--segmenter", default="mediapipe", choices=list(REGISTRY))
    ap.add_argument("--smooth", type=float, default=0.5)
    a = ap.parse_args()
    if a.template == "all":
        Path(a.out).mkdir(parents=True, exist_ok=True)
        for t in list_templates():
            run(a.video, a.srt, t, str(Path(a.out) / f"{Path(a.video).stem}_{t}.mp4"), a.segmenter, a.smooth)
    else:
        run(a.video, a.srt, a.template, a.out if a.out.endswith(".mp4") else a.out + ".mp4", a.segmenter, a.smooth)


if __name__ == "__main__":
    main()
