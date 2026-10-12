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
        Renders Whitestair-style multi-layer layout:
        - Behind Subject: Large Upper Keyword (Anton Orange) + Center Pill + Large Lower Keyword (Anton Orange)
        - Front Overlay: Support context words (White text)
        """
        behind_layer = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        front_layer = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))

        t_in = max(0.0, t - cue.start)
        to_end = max(0.0, cue.end - t)
        scale, alpha, dy = self._anim_values(t_in, to_end)

        words = cue.text.strip().split()
        if not words:
            return behind_layer, front_layer

        K = self.t.get("keyword", {})
        S = self.t.get("support", {})

        center_x = int(head_x) if head_x is not None else (self.W // 2)
        base_head_y = head_y if head_y is not None else int(0.24 * self.H)

        # Estate Dynamic Layout (Matching Whitestair)
        if self.t.get("layout") == "estate_dynamic":
            raw_text = cue.text.strip()
            forced = [f.strip() for f in re.findall(r'\*([^*]+)\*', raw_text) if f.strip()]
            plain_text = re.sub(r'\*[^*]+\*', '', raw_text)
            plain = [p.strip() for p in plain_text.split() if p.strip()]

            if len(forced) >= 2:
                top_kw = forced[0].upper()
                bot_kw = forced[1].upper()
                pill_word = plain[0].lower() if plain else "real"
            elif len(forced) == 1:
                top_kw = forced[0].upper()
                if len(plain) >= 2:
                    pill_word = plain[0].lower()
                    bot_kw = " ".join(plain[1:]).upper()
                elif len(plain) == 1:
                    pill_word = plain[0].lower()
                    bot_kw = ""
                else:
                    pill_word = ""
                    bot_kw = ""
            else:
                # Distribute words: 1st word top, 2nd pill, remaining bottom
                if len(plain) == 1:
                    top_kw, pill_word, bot_kw = plain[0].upper(), "", ""
                elif len(plain) == 2:
                    top_kw, pill_word, bot_kw = plain[0].upper(), plain[1].lower(), ""
                else:
                    top_kw = plain[0].upper()
                    pill_word = plain[1].lower()
                    bot_kw = " ".join(plain[2:]).upper()

            # 1. Render Top Bold Word (Behind Head)
            if top_kw:
                size_top = self._fit(top_kw, K["font"], K.get("size_ratio", 0.13), K.get("max_width_ratio", 0.88))
                img_top = self._draw_bold_text(top_kw, size_top, K)
                pos_y = base_head_y - int(0.03 * self.H) + dy
                self._paste(behind_layer, img_top, center_x, pos_y, scale, alpha)

            # 2. Render Center Pill Badge
            if pill_word:
                pill_size = int(0.036 * self.H)
                img_pill = self._draw_pill(pill_word, pill_size)
                pos_y = base_head_y + int(0.11 * self.H) + dy
                self._paste(behind_layer, img_pill, center_x, pos_y, scale, alpha)

            # 3. Render Bottom Bold Word (Behind Chest/Shoulders)
            if bot_kw:
                size_bot = self._fit(bot_kw, K["font"], K.get("size_ratio", 0.13), K.get("max_width_ratio", 0.88))
                img_bot = self._draw_bold_text(bot_kw, size_bot, K)
                pos_y = base_head_y + int(0.20 * self.H) + dy
                self._paste(behind_layer, img_bot, center_x, pos_y, scale, alpha)

            # 4. Context words on top-left and top-right (front overlay)
            if len(plain) >= 2:
                ctx_left = plain[0].lower()
                ctx_right = plain[1].lower() if len(plain) > 1 else ""
                s_size = int(S.get("size_ratio", 0.045) * self.H)
                img_l = self._draw_bold_text(ctx_left, s_size, S)
                pos_l_x = max(int(img_l.width // 2 + 10), center_x - int(0.26 * self.W))
                self._paste(front_layer, img_l, pos_l_x, base_head_y - int(0.12 * self.H) + dy, 1.0, alpha)

                if ctx_right:
                    img_r = self._draw_bold_text(ctx_right, s_size, S)
                    pos_r_x = min(int(self.W - img_r.width // 2 - 10), center_x + int(0.26 * self.W))
                    self._paste(front_layer, img_r, pos_r_x, base_head_y - int(0.12 * self.H) + dy, 1.0, alpha)

            return behind_layer, front_layer

        # Fallback for other single-keyword templates (luxe, editorial, etc.)
        raw_kw = words[-1]
        kw_text = raw_kw.upper() if K.get("uppercase") else raw_kw
        k_size = self._fit(kw_text, K["font"], K.get("size_ratio", 0.16), K.get("max_width_ratio", 0.85))
        kimg = self._draw_bold_text(kw_text, k_size, K)
        ky = base_head_y + dy
        self._paste(behind_layer, kimg, center_x, ky, scale, alpha)

        if len(words) > 1:
            supp_text = " ".join(words[:-1])
            s_size = self._fit(supp_text, S["font"], S.get("size_ratio", 0.05), S.get("max_width_ratio", 0.80))
            simg = self._draw_bold_text(supp_text, s_size, S)
            sy = ky - kimg.height // 2 - int(0.04 * self.H)
            self._paste(front_layer, simg, center_x, sy, 1.0, alpha)

        return behind_layer, front_layer
