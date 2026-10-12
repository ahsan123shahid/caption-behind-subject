from .base import BaseSegmenter
from .mediapipe_segmenter import MediaPipeSegmenter

def get_segmenter(name: str = "mediapipe", **kwargs) -> BaseSegmenter:
    """
    Factory function to easily swap segmentation backends:
    - 'mediapipe': MediaPipe Tasks Selfie Segmenter (lightweight, CPU/GPU)
    - Future plug-ins: 'rvm' (Robust Video Matting), 'rembg', 'sam2'
    """
    name_lower = name.lower()
    if name_lower == "mediapipe":
        return MediaPipeSegmenter(**kwargs)
    else:
        raise ValueError(f"Unknown segmenter backend '{name}'. Available: ['mediapipe']")

__all__ = ["BaseSegmenter", "MediaPipeSegmenter", "get_segmenter"]
