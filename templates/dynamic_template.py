from PIL import Image, ImageDraw
import numpy as np
import math
from typing import Dict, Any, List
from .base import BaseTemplate
from .utils import get_system_font, draw_text_with_effects, draw_pill_keyword

class DynamicTemplate(BaseTemplate):
    """
    Template 1: 'Dynamic' (Estate Dynamic)
    - Heavy condensed uppercase font (Impact / Arial Black style).
    - Bold vibrant accent (Orange #FF5500) + Clean White.
    - Highlighted keyword in a filled pill badge.
    - Sits BEHIND the subject creating 3D depth occluded by head/body.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.font_name = self.config.get("font_family", "impact")
        self.primary_color = tuple(self.config.get("primary_color", [255, 255, 255])) + (255,)
        self.accent_color = tuple(self.config.get("accent_color", [255, 85, 0])) + (255,)
        self.pill_color = (25, 25, 25, 240)
        self.pill_text_color = (255, 255, 255, 255)

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

        # Use the latest active caption
        cap = active_captions[-1]
        text = cap.text.strip().upper()
        words = text.split()
        if not words:
            return np.array(img)

        # Animation progress (0.0 to 1.0)
        progress = min(1.0, max(0.0, (current_time - cap.start) / max(0.01, (cap.end - cap.start))))
        # Word pop scale effect
        pop_scale = 1.0 + 0.12 * math.sin(min(1.0, progress * 4.0) * math.pi)

        # Dynamic layout from reference:
        # If multiple words, break into:
        # Top word (Large bold behind), Middle pill keyword, Bottom word (Large bold behind)
        main_font_size = int(w * 0.16 * pop_scale)
        pill_font_size = int(w * 0.045)
        
        main_font = get_system_font(self.font_name, main_font_size)
        pill_font = get_system_font("arialbd", pill_font_size)

        center_x = w // 2
        # Position around upper-chest / head level (0.28 to 0.40 height)
        base_y = int(h * 0.30)

        if len(words) == 1:
            # Single big keyword
            word = words[0]
            bbox = main_font.getbbox(word)
            tw = bbox[2] - bbox[0]
            tx = center_x - tw // 2
            ty = base_y
            draw_text_with_effects(
                draw, (tx, ty), word, main_font, 
                fill_color=self.accent_color,
                stroke_color=(0, 0, 0, 220), stroke_width=max(2, int(w * 0.006)),
                shadow_offset=(4, 6)
            )
        elif len(words) == 2:
            # Two words (e.g. "REAL" on top, "ESTATE" on bottom)
            w1, w2 = words[0], words[1]
            bbox1 = main_font.getbbox(w1)
            tw1 = bbox1[2] - bbox1[0]
            draw_text_with_effects(
                draw, (center_x - tw1 // 2, base_y - int(h * 0.08)), w1, main_font,
                fill_color=self.accent_color,
                stroke_color=(0, 0, 0, 220), stroke_width=max(2, int(w * 0.006)),
                shadow_offset=(4, 6)
            )
            bbox2 = main_font.getbbox(w2)
            tw2 = bbox2[2] - bbox2[0]
            draw_text_with_effects(
                draw, (center_x - tw2 // 2, base_y + int(h * 0.06)), w2, main_font,
                fill_color=self.accent_color,
                stroke_color=(0, 0, 0, 220), stroke_width=max(2, int(w * 0.006)),
                shadow_offset=(4, 6)
            )
        else:
            # 3+ words: Top large, Middle pill, Bottom large
            w_top = words[0]
            w_pill = words[1]
            w_bottom = " ".join(words[2:])

            bbox_top = main_font.getbbox(w_top)
            draw_text_with_effects(
                draw, (center_x - (bbox_top[2] - bbox_top[0]) // 2, base_y - int(h * 0.09)), w_top, main_font,
                fill_color=self.accent_color,
                stroke_color=(0, 0, 0, 220), stroke_width=max(2, int(w * 0.006)),
                shadow_offset=(4, 6)
            )

            bbox_bot = main_font.getbbox(w_bottom)
            draw_text_with_effects(
                draw, (center_x - (bbox_bot[2] - bbox_bot[0]) // 2, base_y + int(h * 0.08)), w_bottom, main_font,
                fill_color=self.accent_color,
                stroke_color=(0, 0, 0, 220), stroke_width=max(2, int(w * 0.006)),
                shadow_offset=(4, 6)
            )

            # Center badge/pill
            draw_pill_keyword(
                draw, (center_x, base_y + int(h * 0.005)), w_pill.lower(), pill_font,
                bg_color=self.pill_color, text_color=self.pill_text_color,
                padding_x=int(w * 0.03), padding_y=int(h * 0.008), radius=14
            )

        return np.array(img)

    def render_overlay_layer(
        self, 
        frame_shape: tuple, 
        current_time: float, 
        active_captions: List[Any]
    ) -> np.ndarray:
        """
        Renders small contextual words on top (like 'you can' and 'add' in reference).
        """
        h, w = frame_shape[:2]
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        if not active_captions:
            return np.array(img)

        cap = active_captions[-1]
        words = cap.text.split()
        if len(words) >= 4:
            small_font = get_system_font("arial", int(w * 0.045))
            # Left helper text
            draw_text_with_effects(
                draw, (int(w * 0.12), int(h * 0.18)), words[0].lower(), small_font,
                fill_color=self.primary_color, stroke_color=(0, 0, 0, 180), stroke_width=2
            )
            # Right helper text
            draw_text_with_effects(
                draw, (int(w * 0.70), int(h * 0.18)), words[1].lower(), small_font,
                fill_color=self.primary_color, stroke_color=(0, 0, 0, 180), stroke_width=2
            )

        return np.array(img)
