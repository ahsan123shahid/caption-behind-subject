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


def _clean_matte(m: np.ndarray, thresh: float = 0.22) -> np.ndarray:
    """
    Produces clean, solid foreground matte with smooth anti-aliased edge:
    1. Lower threshold (0.22) captures hats, accessories, and fine hair without dropouts.
    2. Morphological closing seals small internal mask holes.
    3. Slight dilation ensures subject silhouette cleanly covers the behind text boundary.
    4. Sub-pixel Gaussian anti-aliasing eliminates harsh pixelation and prevents color bleeding.
    """
    binary = (m > thresh).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    matte = cv2.GaussianBlur(dilated.astype(np.float32), (5, 5), 1.2)
    return np.clip(matte, 0.0, 1.0)


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
            return _clean_matte(res.segmentation_mask.astype(np.float32))

        # MediaPipe Tasks API
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.segmenter.segment(mp_image)
        if result.confidence_masks:
            raw_mask = result.confidence_masks[0].numpy_view()
            if raw_mask.ndim == 3 and raw_mask.shape[-1] == 1:
                raw_mask = raw_mask.squeeze(-1)
            return _clean_matte(raw_mask.astype(np.float32))

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


@register("yolo")
class YOLOSegmenter(Segmenter):
    """Ultralytics YOLOv8-seg person segmenter for ultra-sharp cutouts matching Whitestair production stack."""
    def __init__(self, model_name: str = "yolov8n-seg.pt", conf: float = 0.35):
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        self.conf = conf

    def mask(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        # Predict person class (class 0 = person)
        res = self.model.predict(bgr, conf=self.conf, classes=[0], verbose=False)
        if res and len(res) > 0 and res[0].masks is not None:
            m = np.zeros((h, w), dtype=np.float32)
            for mask_data in res[0].masks.data:
                m_np = mask_data.cpu().numpy()
                if m_np.shape != (h, w):
                    m_np = cv2.resize(m_np, (w, h), interpolation=cv2.INTER_LINEAR)
                m = np.maximum(m, m_np)
            return cv2.GaussianBlur(m, (3, 3), 0)
        return np.ones((h, w), dtype=np.float32)

