import subprocess
import cv2
import numpy as np
from .captions import parse_srt
from .renderer import CaptionRenderer
from .segmentation import get_segmenter
from .templates import get_template


import shutil


def _get_ffmpeg_bin() -> str:
    bin_path = shutil.which("ffmpeg")
    if bin_path:
        return bin_path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _over(base: np.ndarray, layer) -> np.ndarray:
    """Alpha-composite an RGBA PIL layer (RGB order) onto a BGR float frame."""
    arr = np.asarray(layer, dtype=np.float32)
    a = arr[..., 3:4] / 255.0
    return base * (1 - a) + arr[..., 2::-1] * a


def run(video: str, srt: str, template: str, out: str, segmenter: str = "mediapipe",
        smooth: float = 0.5, crf: int = 18):
    tpl, cues, seg = get_template(template), parse_srt(srt), get_segmenter(segmenter)
    cap = cv2.VideoCapture(video)
    W, H = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    rend = CaptionRenderer(tpl, W, H)
    ffmpeg_bin = _get_ffmpeg_bin()

    ff = subprocess.Popen([
        ffmpeg_bin, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
        "-i", video, "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-c:a", "copy", "-shortest", out], stdin=subprocess.PIPE)

    prev, i = None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        i += 1
        cue = next((c for c in cues if c.start <= t < c.end), None)
        if cue is None:                      # no caption -> skip segmentation (fast path)
            ff.stdin.write(frame.tobytes())
            prev = None
            continue

        alpha = seg.mask(frame)
        if prev is not None and smooth > 0:  # light temporal smoothing vs. edge flicker
            alpha = smooth * prev + (1 - smooth) * alpha
        prev = alpha

        behind, front = rend.render(cue, t)
        f = frame.astype(np.float32)
        comp = _over(f, behind)                       # background -> text
        m = alpha[..., None]
        comp = comp * (1 - m) + f * m                 # person on top
        comp = _over(comp, front)                     # optional front text
        ff.stdin.write(np.clip(comp, 0, 255).astype(np.uint8).tobytes())

    cap.release()
    ff.stdin.close()
    ff.wait()
    print(f"[{template}] done -> {out}")
