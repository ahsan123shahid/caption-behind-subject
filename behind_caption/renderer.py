"""Renders caption text into two RGBA layers: 'behind' (under person) and 'front'."""
import re
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from .templates import ROOT
from .captions import Cue


def _rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (255,)


def _clean(w):
    return re.sub(r"[^\w']", "", w)


@lru_cache(maxsize=128)
def _font(path, size):
    try:
        return ImageFont.truetype(str(ROOT / path), size)
    except OSError:
        return ImageFont.load_default(size)


def _ease_out_back(x, s=1.4):
    x = min(max(x, 0), 1) - 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2


def _ease_out(x):
    x = min(max(x, 0), 1) - 1
    return 1 + x ** 3


class CaptionRenderer:
    def __init__(self, tpl: dict, W: int, H: int):
        self.t, self.W, self.H = tpl, W, H
        self._fit_cache: dict = {}
        self._img_cache: dict = {}

    def _fit(self, text: str, font_path: str, size_ratio: float, max_w_ratio: float):
        key = (text, font_path, size_ratio, max_w_ratio)
        if key in self._fit_cache:
            return self._fit_cache[key]
        size = int(size_ratio * self.H)
        max_w = int(max_w_ratio * self.W)
        while size > 10:
            w = _font(font_path, size).getlength(text)
            if w <= max_w:
                break
            size = int(size * 0.94)
        self._fit_cache[key] = size
        return size

    def _draw_bold_text(self, text: str, size: int, cfg: dict):
        key = (text, size, id(cfg))
        if key in self._img_cache:
            return self._img_cache[key]

        f = _font(cfg["font"], size)
        st = cfg.get("stroke")
        sw = int(st.get("width", 0.02) * size) if st else 0
        bb = f.getbbox(text, stroke_width=sw)
        tw = bb[2] - bb[0]
        th = bb[3] - bb[1]
        pad = sw + 8

        img = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)

        # Drop shadow
        sh = cfg.get("shadow")
        if sh:
            ox = int(sh.get("ox", 0.02) * size)
            oy = int(sh.get("oy", 0.04) * size)
            scolor = _rgb(sh.get("color", "#000000"))
            s_alpha = int(255 * sh.get("alpha", 0.7))
            d.text((pad - bb[0] + ox, pad - bb[1] + oy), text, font=f,
                   fill=(scolor[0], scolor[1], scolor[2], s_alpha),
                   stroke_width=sw, stroke_fill=(0, 0, 0, s_alpha))

        # Main text fill & stroke
        fill_color = _rgb(cfg.get("color", "#FFFFFF"))
        stroke_color = _rgb(st.get("color", "#000000")) if st else None
        d.text((pad - bb[0], pad - bb[1]), text, font=f, fill=fill_color,
               stroke_width=sw, stroke_fill=stroke_color)

        # Vertical Gradient fill
        grad = cfg.get("gradient")
        if grad:
            mask = Image.new("L", img.size, 0)
            ImageDraw.Draw(mask).text((pad - bb[0], pad - bb[1]), text, font=f, fill=255)
            g = Image.linear_gradient("L").resize(img.size)
            col = Image.composite(Image.new("RGBA", img.size, _rgb(grad[1])),
                                  Image.new("RGBA", img.size, _rgb(grad[0])), g)
            img.paste(col, (0, 0), mask)

        self._img_cache[key] = img
        return img

    def _draw_pill(self, text: str, size: int):
        f = _font(self.t.get("pill_font", "fonts/PlusJakartaSans-800-normal.ttf"), size)
        pill_cfg = self.t.get("pill", {})
        pad_x = int(pill_cfg.get("pad_x", 18))
        pad_y = int(pill_cfg.get("pad_y", 7))
        radius = int(pill_cfg.get("radius", 14))

        bb = f.getbbox(text)
        tw = bb[2] - bb[0]
        th = bb[3] - bb[1]

        w = tw + 2 * pad_x
        h = th + 2 * pad_y
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)

        # Dark pill container
        bg_col = _rgb(pill_cfg.get("color", "#181818"))
        d.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=bg_col)

        # White centered text
        tx_col = _rgb(pill_cfg.get("text_color", "#FFFFFF"))
        d.text((pad_x - bb[0], pad_y - bb[1]), text, font=f, fill=tx_col)
        return img

    def _anim_values(self, t_in: float, to_end: float):
        a = self.t.get("animation", {})
        dur = a.get("duration", 0.15)
        out_dur = a.get("out_duration", 0.10)
        p = min(1.0, max(0.0, t_in / dur))

        if a.get("type") == "pop":
            scale = 0.85 + 0.15 * _ease_out_back(p)
            alpha = min(1.0, p * 2.5)
            dy = 0.0
        else:
            e = _ease_out(p)
            scale = 1.0
            alpha = e
            dy = (1.0 - e) * 20

        if to_end < out_dur:
            alpha *= max(0.0, to_end / out_dur)

        return scale, alpha, dy

    def _paste(self, layer, img, cx, cy, scale=1.0, alpha=1.0):
        if alpha <= 0.02:
            return
        if abs(scale - 1.0) > 0.02:
            nw = max(1, int(img.width * scale))
            nh = max(1, int(img.height * scale))
            img = img.resize((nw, nh), Image.BILINEAR)
        if alpha < 0.98:
            img = img.copy()
            img.putalpha(img.getchannel("A").point(lambda v: int(v * alpha)))
        layer.alpha_composite(img, (int(cx - img.width / 2), int(cy - img.height / 2)))

    def render(self, cue: Cue, t: float, head_x: float = None, head_y: float = None):
        """
        Renders Hero Text Pop layout (matching Bolti.ai / Submagic / reference style):
        - Behind Subject: Giant Bold Hero Keyword popped behind the head/upper body
        - Front Overlay: Clean support subtitle text at bottom
        """
        behind_layer = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        front_layer = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))

        t_in = max(0.0, t - cue.start)
        to_end = max(0.0, cue.end - t)
        scale, alpha, dy = self._anim_values(t_in, to_end)

        raw_text = cue.text.strip()
        if not raw_text:
            return behind_layer, front_layer

        K = self.t.get("keyword", {})
        S = self.t.get("support", {})

        # Extract forced hero keyword from *...* markers
        forced = [f.strip() for f in re.findall(r'\*([^*]+)\*', raw_text) if f.strip()]
        plain_text = re.sub(r'\*[^*]+\*', '', raw_text).strip()
        plain_text = re.sub(r'\s+', ' ', plain_text)

        if forced:
            hero_kw = forced[0]
            supp_text = plain_text
        else:
            words = raw_text.split()
            if len(words) <= 1:
                hero_kw = raw_text
                supp_text = ""
            else:
                # Pick longest significant word as hero keyword
                hero_kw = max(words, key=lambda w: len(_clean(w)))
                supp_text = " ".join([w for w in words if w != hero_kw])

        if K.get("uppercase", True):
            hero_kw = hero_kw.upper()
        if S.get("uppercase", False):
            supp_text = supp_text.upper()

        # 1. Render Hero Keyword Behind Head
        if hero_kw:
            k_size = self._fit(hero_kw, K["font"], K.get("size_ratio", 0.22), K.get("max_width_ratio", 0.94))
            kimg = self._draw_bold_text(hero_kw, k_size, K)

            # Center Y: positioned so bottom edge just touches/overlaps the head crown
            if head_y is not None:
                ky = int(head_y - (kimg.height * 0.32)) + dy
            else:
                ky = int(K.get("y_ratio", 0.18) * self.H) + dy

            # Center X:
            # If text is wide (> 60% of frame), center on frame so it fills screen symmetrically
            # Otherwise track speaker's head position
            if kimg.width > int(0.60 * self.W) or head_x is None:
                kx = self.W // 2
            else:
                margin = kimg.width // 2 + 20
                kx = max(margin, min(self.W - margin, int(head_x)))

            self._paste(behind_layer, kimg, kx, ky, scale, alpha)

        # 2. Render Support Subtitle in Front at Bottom
        if supp_text:
            s_size = self._fit(supp_text, S["font"], S.get("size_ratio", 0.048), S.get("max_width_ratio", 0.85))
            simg = self._draw_bold_text(supp_text, s_size, S)
            sy = int(S.get("y_ratio", 0.90) * self.H)
            sx = self.W // 2
            self._paste(front_layer, simg, sx, sy, 1.0, alpha)

        return behind_layer, front_layer

