import os
import cv2
import numpy as np
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from .base import BaseSegmenter

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"

class MediaPipeSegmenter(BaseSegmenter):
    """
    Person segmenter using Google MediaPipe Tasks ImageSegmenter (selfie_segmenter.tflite).
    Fast, highly optimized CPU/GPU execution with feathering for clean compositing.
    """
    def __init__(self, model_path: str = None, threshold: float = 0.5, blur_radius: int = 7):
        self.threshold = threshold
        self.blur_radius = blur_radius if blur_radius % 2 != 0 else blur_radius + 1

        if model_path is None:
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

    def segment(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Accepts BGR frame from OpenCV, returns smooth alpha mask (0.0 to 1.0).
        1.0 represents person foreground, 0.0 represents background.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        result = self.segmenter.segment(mp_image)
        if result.confidence_masks:
            # First mask is person confidence
            confidence = result.confidence_masks[0].numpy_view()
            binary_mask = (confidence > self.threshold).astype(np.float32)
            
            # Feather edges for smooth compositing without jagged halo
            if self.blur_radius > 1:
                binary_mask = cv2.GaussianBlur(binary_mask, (self.blur_radius, self.blur_radius), 0)
            
            return np.clip(binary_mask, 0.0, 1.0)

        # Fallback if no mask detected
        h, w = frame_bgr.shape[:2]
        return np.ones((h, w), dtype=np.float32)

    def close(self):
        if hasattr(self, 'segmenter') and self.segmenter:
            self.segmenter.close()
