"""Swappable person segmentation. Add a GPU model (e.g. RVM) by registering a new class."""
from abc import ABC, abstractmethod
import os
import urllib.request
import cv2
import numpy as np

REGISTRY: dict[str, type] = {}


def register(name):
    def deco(cls):
        REGISTRY[name] = cls
        return cls
    return deco


def get_segmenter(name: str) -> "Segmenter":
    if name not in REGISTRY:
        raise SystemExit(f"Unknown segmenter '{name}'. Available: {', '.join(REGISTRY)}")
    return REGISTRY[name]()


class Segmenter(ABC):
    @abstractmethod
    def mask(self, bgr: np.ndarray) -> np.ndarray:
        """Return float32 alpha matte HxW in [0,1] (1 = person)."""


def _soften(m: np.ndarray, lo=0.35, hi=0.75) -> np.ndarray:
    m = np.clip((m - lo) / (hi - lo), 0, 1)
    return m * m * (3 - 2 * m)  # smoothstep -> cleaner edges


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"


@register("mediapipe")
class MediaPipeSegmenter(Segmenter):
    def __init__(self):
        import mediapipe as mp
        if hasattr(mp, 'solutions'):
            self.mode = "legacy"
            self.seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        else:
            self.mode = "tasks"
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, "selfie_segmenter.tflite")

            if not os.path.exists(model_path):
                print(f"[Info] Downloading MediaPipe model to {model_path}...")
                urllib.request.urlretrieve(MODEL_URL, model_path)

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.ImageSegmenterOptions(
                base_options=base_options,
                output_confidence_masks=True
            )
            self.segmenter = vision.ImageSegmenter.create_from_options(options)

    def mask(self, bgr):
        import mediapipe as mp
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self.mode == "legacy":
            res = self.seg.process(rgb)
            return _soften(res.segmentation_mask.astype(np.float32))

        # MediaPipe Tasks API
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.segmenter.segment(mp_image)
        if result.confidence_masks:
            raw_mask = result.confidence_masks[0].numpy_view()
            if raw_mask.ndim == 3 and raw_mask.shape[-1] == 1:
                raw_mask = raw_mask.squeeze(-1)
            return _soften(raw_mask.astype(np.float32))

        h, w = bgr.shape[:2]
        return np.ones((h, w), dtype=np.float32)


@register("rembg")
class RembgSegmenter(Segmenter):
    def __init__(self):
        from rembg import new_session
        self.session = new_session("u2net_human_seg")

    def mask(self, bgr):
        from rembg import remove
        m = remove(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), only_mask=True, session=self.session)
        return np.asarray(m, dtype=np.float32) / 255.0


@register("dummy")
class DummySegmenter(Segmenter):
    """Centered ellipse; for tests/CI without models."""
    def mask(self, bgr):
        h, w = bgr.shape[:2]
        m = np.zeros((h, w), np.float32)
        cv2.ellipse(m, (w // 2, int(h * .6)), (int(w * .18), int(h * .45)), 0, 0, 360, 1.0, -1)
        return cv2.GaussianBlur(m, (0, 0), 3)
