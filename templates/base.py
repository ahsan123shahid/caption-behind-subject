from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np

class BaseTemplate(ABC):
    """
    Abstract base class for caption template styles.
    Renders caption layers (both background-layer and foreground-overlay if any).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def render_behind_layer(
        self, 
        frame_shape: tuple, 
        current_time: float, 
        active_captions: List[Any]
    ) -> np.ndarray:
        """
        Renders the text layer that sits BEHIND the person (in front of background).
        Returns a BGRA image (H, W, 4) with transparent background.
        """
        pass

    def render_overlay_layer(
        self, 
        frame_shape: tuple, 
        current_time: float, 
        active_captions: List[Any]
    ) -> np.ndarray:
        """
        Optional: Renders text that sits in front of the person (if template has foreground elements).
        Returns a BGRA image (H, W, 4). Default is completely transparent.
        """
        h, w = frame_shape[:2]
        return np.zeros((h, w, 4), dtype=np.uint8)
