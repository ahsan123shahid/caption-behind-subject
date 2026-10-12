from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from typing import Tuple, Optional

def get_system_font(font_name: str, size: int):
    """
    Tries to load font from Windows Fonts directory or fallbacks.
    """
    win_fonts = os.environ.get("WINDIR", "C:\\Windows") + "\\Fonts\\"
    font_candidates = [
        win_fonts + f"{font_name}.ttf",
        win_fonts + f"{font_name.lower()}.ttf",
        win_fonts + f"{font_name}.otf",
        win_fonts + "impact.ttf",
        win_fonts + "arialbd.ttf",
        win_fonts + "arial.ttf",
        win_fonts + "segoeui.ttf",
        win_fonts + "segoeuib.ttf"
    ]
    for path in font_candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    try:
        return ImageFont.truetype(font_name, size)
    except Exception:
        return ImageFont.load_default()

def draw_text_with_effects(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill_color: Tuple[int, int, int, int] = (255, 255, 255, 255),
    stroke_color: Optional[Tuple[int, int, int, int]] = (0, 0, 0, 255),
    stroke_width: int = 0,
    shadow_offset: Optional[Tuple[int, int]] = None,
    shadow_color: Tuple[int, int, int, int] = (0, 0, 0, 160)
):
    """
    Renders text with optional stroke and drop shadow.
    """
    x, y = xy
    if shadow_offset and shadow_offset != (0, 0):
        sx, sy = shadow_offset
        draw.text((x + sx, y + sy), text, font=font, fill=shadow_color)
    
    if stroke_width > 0 and stroke_color is not None:
        draw.text((x, y), text, font=font, fill=fill_color, stroke_width=stroke_width, stroke_fill=stroke_color)
    else:
        draw.text((x, y), text, font=font, fill=fill_color)

def draw_pill_keyword(
    draw: ImageDraw.ImageDraw,
    center_xy: Tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    bg_color: Tuple[int, int, int, int] = (20, 20, 20, 230),
    text_color: Tuple[int, int, int, int] = (255, 255, 255, 255),
    padding_x: int = 24,
    padding_y: int = 10,
    radius: int = 16
):
    """
    Draws a rounded pill badge with centered keyword (like in Estate Dynamic template).
    """
    cx, cy = center_xy
    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x0 = cx - (text_w // 2) - padding_x
    y0 = cy - (text_h // 2) - padding_y
    x1 = cx + (text_w // 2) + padding_x
    y1 = cy + (text_h // 2) + padding_y

    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=bg_color)
    # Draw centered text inside pill
    draw.text((cx - text_w // 2, cy - text_h // 2 - bbox[1]), text, font=font, fill=text_color)
