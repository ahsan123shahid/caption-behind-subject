from PIL import Image, ImageDraw
import numpy as np
import math
from typing import Dict, Any, List
from .base import BaseTemplate
from .utils import get_system_font, draw_text_with_effects

class LuxeTemplate(BaseTemplate):
    """
    Template 2: 'Luxe' (Elegant / Premium)
    - Refined semi-bold sans font (Segoe UI / Helvetica).
    - Cool accent: Teal / Cyan gradient (#00E1CC / #008080) + Pure White.
    - Subtle fade and smooth upward slide animation.
    - Large keyword sits behind the subject.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.font_name = self.config.get("font_family", "segoeuib")
        self.primary_color = tuple(self.config.get("primary_color", [255, 255, 255])) + (255,)
        self.accent_color = tuple(self.config.get("accent_color", [0, 220, 200])) + (255,)

    def render_behind_layer(
        self, 
        frame_shape: tuple, 
        current_time: float, 
        active_captions: List[Any]
    ) -> np.ndarray:
        h, w = frame_shape[:2]
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        if not active_captions:
            return np.array(img)

        cap = active_captions[-1]
        text = cap.text.strip()
        words = text.split()
        if not words:
            return np.array(img)

        # Smooth slide and fade animation
        progress = min(1.0, max(0.0, (current_time - cap.start) / max(0.01, (cap.end - cap.start))))
        # Ease out cubic
        ease_progress = 1.0 - math.pow(1.0 - min(1.0, progress * 2.5), 3)
        slide_offset_y = int((1.0 - ease_progress) * 20)
        alpha = int(min(255, ease_progress * 255))

        accent_color_alpha = self.accent_color[:3] + (alpha,)

        # Identify main keyword (longest or last word)
        keyword = words[-1].capitalize() if len(words) > 1 else words[0].capitalize()

        main_font_size = int(w * 0.22)  # Large, elegant
        main_font = get_system_font(self.font_name, main_font_size)

        bbox = main_font.getbbox(keyword)
        kw_w = bbox[2] - bbox[0]
        
        # Position around upper center / behind shoulders
        cx = w // 2 - (kw_w // 2)
        cy = int(h * 0.28) + slide_offset_y

        draw_text_with_effects(
            draw, (cx, cy), keyword, main_font,
            fill_color=accent_color_alpha,
            stroke_color=(10, 40, 50, alpha), stroke_width=max(1, int(w * 0.005)),
            shadow_offset=(3, 6),
            shadow_color=(0, 0, 0, int(alpha * 0.6))
        )

        return np.array(img)

    def render_overlay_layer(
        self, 
        frame_shape: tuple, 
        current_time: float, 
        active_captions: List[Any]
    ) -> np.ndarray:
        """
        Renders the elegant cursive / delicate context phrase (e.g. 'greeted by this').
        """
        h, w = frame_shape[:2]
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        if not active_captions:
            return np.array(img)

        cap = active_captions[-1]
        words = cap.text.split()
        if len(words) > 1:
            phrase = " ".join(words[:-1]).lower()
            font = get_system_font("segoeui", int(w * 0.065))
            
            # Draw above keyword towards the left
            draw_text_with_effects(
                draw, (int(w * 0.10), int(h * 0.23)), phrase, font,
                fill_color=self.primary_color,
                stroke_color=(0, 0, 0, 160), stroke_width=1,
                shadow_offset=(2, 3)
            )

        return np.array(img)
