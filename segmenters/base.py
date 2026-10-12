from abc import ABC, abstractmethod
import numpy as np

class BaseSegmenter(ABC):
    """
    Abstract base class for person segmentation models.
    Designed so that MediaPipe, rembg, Robust Video Matting (RVM), 
    or any GPU-accelerated model can be plugged in seamlessly.
    """

    @abstractmethod
    def segment(self, frame: np.ndarray) -> np.ndarray:
        """
        Takes an RGB/BGR video frame (H, W, 3).
        Returns a float32 mask (H, W) between 0.0 (background) and 1.0 (foreground person).
        """
        pass

    def close(self):
        """Release any model or GPU resources."""
        pass
