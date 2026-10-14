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

    def _draw_karaoke_line(self, words: list[str], active_idx: int, font_path: str, size: int,
                           active_color: str, inactive_color: str = "#FFFFFF", stroke_color: str = "#000000"):
        key = ("karaoke", tuple(words), active_idx, font_path, size, active_color, inactive_color, stroke_color)
        if key in self._img_cache:
            return self._img_cache[key]

        f_normal = _font(font_path, size)
        f_active = _font(font_path, int(size * 1.10))
        sw = max(2, int(size * 0.055))

        word_imgs = []
        max_h = 0
        for i, w in enumerate(words):
            is_act = (i == active_idx)
            f = f_active if is_act else f_normal
            col = _rgb(active_color) if is_act else _rgb(inactive_color)
            st_col = _rgb(stroke_color)

            bb = f.getbbox(w, stroke_width=sw)
            w_w = bb[2] - bb[0]
            w_h = bb[3] - bb[1]
            pad = sw + 8

            img = Image.new("RGBA", (w_w + 2 * pad, w_h + 2 * pad), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)

            # Drop shadow for high contrast over any background
            ox = int(0.025 * size)
            oy = int(0.04 * size)
            d.text((pad - bb[0] + ox, pad - bb[1] + oy), w, font=f,
                   fill=(0, 0, 0, 190), stroke_width=sw, stroke_fill=(0, 0, 0, 190))
            # Text fill and stroke
            d.text((pad - bb[0], pad - bb[1]), w, font=f,
                   fill=col, stroke_width=sw, stroke_fill=st_col)

            word_imgs.append(img)
            max_h = max(max_h, img.height)

        space_w = int(size * 0.32)
        total_w = sum(img.width for img in word_imgs) + space_w * max(0, len(words) - 1)

        line_img = Image.new("RGBA", (total_w, max_h + 10), (0, 0, 0, 0))
        cur_x = 0
        for img in word_imgs:
            y_pos = (max_h - img.height) // 2
            line_img.alpha_composite(img, (cur_x, y_pos))
            cur_x += img.width + space_w

        self._img_cache[key] = line_img
        return line_img

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

        # 1. RENDER HERO KEYWORD BEHIND THE SUBJECT
        if cue.is_hero and cue.hero_text:
            hero_kw = cue.hero_text.upper() if K.get("uppercase", True) else cue.hero_text
            k_size = self._fit(hero_kw, K["font"], K.get("size_ratio", 0.22), K.get("max_width_ratio", 0.94))
            kimg = self._draw_bold_text(hero_kw, k_size, K)

            # Center Y: positioned around upper-center so subject naturally occludes the middle of it
            if head_y is not None:
                ky = int(head_y + 0.06 * self.H) + dy
            else:
                ky = int(K.get("y_ratio", 0.25) * self.H) + dy

            # Center X:
            if kimg.width > int(0.60 * self.W) or head_x is None:
                kx = self.W // 2
            else:
                margin = kimg.width // 2 + 20
                kx = max(margin, min(self.W - margin, int(head_x)))

            self._paste(behind_layer, kimg, kx, ky, scale, alpha)
            # Front layer remains empty when Hero word pops behind

        # 2. RENDER NORMAL SUBTITLE IN FRONT AT BOTTOM (VIRAL REELS / KARAOKE POP STYLE)
        else:
            front_text = cue.front_text or raw_text
            front_text = re.sub(r'\*([^*]+)\*', r'\1', front_text).strip()
            if S.get("uppercase", False):
                front_text = front_text.upper()

            if front_text:
                words = front_text.split()
                # Compute active word timing based on speech progress
                dur = max(0.15, cue.end - cue.start)
                t_rel = min(dur, max(0.0, t - cue.start))
                weights = [max(2, len(re.sub(r'[^\w]', '', w))) for w in words]
                total_weight = sum(weights) or 1
                accum = 0.0
                active_idx = 0
                for w_i, w_weight in enumerate(weights):
                    w_dur = (w_weight / total_weight) * dur
                    if accum <= t_rel <= (accum + w_dur):
                        active_idx = w_i
                        break
                    accum += w_dur
                else:
                    active_idx = len(words) - 1

                s_size = self._fit(front_text, S["font"], S.get("size_ratio", 0.056), S.get("max_width_ratio", 0.88))
                active_col = S.get("active_color", K.get("color", "#FFE600"))
                inactive_col = S.get("color", "#FFFFFF")
                stroke_col = S.get("stroke", {}).get("color", "#000000")

                simg = self._draw_karaoke_line(
                    words, active_idx, S["font"], s_size,
                    active_col, inactive_col, stroke_col
                )
                sy = int(S.get("y_ratio", 0.82) * self.H)
                sx = self.W // 2
                self._paste(front_layer, simg, sx, sy, 1.0, alpha)
                # Behind layer remains empty when normal subtitle shows in front

        return behind_layer, front_layer

