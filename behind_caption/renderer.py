"""Renders caption text into two RGBA layers: 'behind' (under person) and 'front'."""
import re
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont
from .templates import ROOT
from .captions import Cue


def _rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (255,)


def _clean(w):
    return re.sub(r"[^\w']", "", w)


@lru_cache(maxsize=64)
def _font(path, size):
    try:
        return ImageFont.truetype(str(ROOT / path), size)
    except OSError:
        return ImageFont.load_default(size)  # fallback if fonts not downloaded


def _ease_out_back(x, s=1.7):
    x = min(max(x, 0), 1) - 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2


def _ease_out(x):
    x = min(max(x, 0), 1)
    return 1 - (1 - x) ** 3


class _Prepared:
    def __init__(self, cue: Cue):
        raw = cue.text.split()
        forced = [i for i, w in enumerate(raw) if w.startswith("*") and w.endswith("*") and len(w) > 2]
        words = [w.strip("*") for w in raw]
        self.kw = forced[0] if forced else max(range(len(words)), key=lambda i: len(_clean(words[i])))
        self.words, self.cue = words, cue
        n = len(words)
        span = (cue.end - cue.start) * 0.7
        self.times = [cue.start + span * i / n for i in range(n)]


class CaptionRenderer:
    def __init__(self, tpl: dict, W: int, H: int):
        self.t, self.W, self.H = tpl, W, H
        self._prep: dict[int, _Prepared] = {}
        self._fit_cache: dict = {}
        self._img_cache: dict = {}

    def _fit(self, key, text, cfg):
        """Largest font size (<= size_ratio*H) so text fits max_width_ratio*W."""
        if key in self._fit_cache:
            return self._fit_cache[key]
        size = int(cfg["size_ratio"] * self.H)
        pill = cfg.get("pill")
        while size > 8:
            w = _font(cfg["font"], size).getlength(text) * (1 + (2 * pill["pad_x"] if pill else 0))
            if w <= cfg["max_width_ratio"] * self.W:
                break
            size = int(size * 0.94)
        self._fit_cache[key] = size
        return size

    def _text_img(self, text, size, kind):
        key = (text, size, kind)
        if key in self._img_cache:
            return self._img_cache[key]
        cfg = self.t[kind]
        f = _font(cfg["font"], size)
        st = cfg.get("stroke")
        sw = int(st["width"] * size) if st else 0
        pill = cfg.get("pill")
        bb = f.getbbox(text, stroke_width=sw)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        px = int(pill["pad_x"] * size) if pill else sw + 2
        py = int(pill["pad_y"] * size) if pill else sw + 2
        img = Image.new("RGBA", (tw + 2 * px, th + 2 * py), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        if pill:
            d.rounded_rectangle([0, 0, img.width - 1, img.height - 1],
                                radius=int(pill["radius"] * size), fill=_rgb(pill["color"]))
        sh = cfg.get("shadow")
        if sh:
            ox = int(sh.get("ox", 0.03) * size)
            oy = int(sh.get("oy", 0.05) * size)
            scolor = _rgb(sh.get("color", "#000000"))
            s_alpha = int(255 * sh.get("alpha", 0.7))
            d.text((px - bb[0] + ox, py - bb[1] + oy), text, font=f, fill=(scolor[0], scolor[1], scolor[2], s_alpha),
                   stroke_width=sw, stroke_fill=(0, 0, 0, s_alpha) if st else None)

        d.text((px - bb[0], py - bb[1]), text, font=f, fill=_rgb(cfg["color"]),
               stroke_width=sw, stroke_fill=_rgb(st["color"]) if st else None)
        grad = cfg.get("gradient")  # [top_color, bottom_color] vertical fill
        if grad and not pill:
            mask = Image.new("L", img.size, 0)
            ImageDraw.Draw(mask).text((px - bb[0], py - bb[1]), text, font=f, fill=255)
            g = Image.linear_gradient("L").resize(img.size)
            col = Image.composite(Image.new("RGBA", img.size, _rgb(grad[1])),
                                  Image.new("RGBA", img.size, _rgb(grad[0])), g)
            img.paste(col, (0, 0), mask)
        if len(self._img_cache) > 300:
            self._img_cache.clear()
        self._img_cache[key] = img
        return img

    def _anim(self, t_in, t_to_end):
        a = self.t["animation"]
        od = a.get("out_duration", 0.15)
        p = t_in / a["duration"]
        if a["type"] == "pop":
            scale, alpha, dy = 0.5 + 0.5 * _ease_out_back(p), min(p * 3, 1), 0.0
        else:  # fade_slide
            e = _ease_out(p)
            scale, alpha, dy = 1.0, e, (1 - e) * a.get("slide_ratio", 0.04) * self.H
        if t_to_end < od:
            alpha *= max(t_to_end / od, 0)
        return scale, alpha, dy

    def _paste(self, layer, img, cx, cy, scale, alpha):
        if alpha <= 0.01:
            return
        if abs(scale - 1) > 0.01:
            img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BILINEAR)
        if alpha < 0.99:
            img = img.copy()
            img.putalpha(img.getchannel("A").point(lambda v: int(v * alpha)))
        layer.alpha_composite(img, (int(cx - img.width / 2), int(cy - img.height / 2)))

    def render(self, cue: Cue, t: float, head_y: float = None):
        """Return (behind_layer, front_layer) as RGBA PIL images."""
        p = self._prep.setdefault(id(cue), _Prepared(cue))
        K, S = self.t["keyword"], self.t["support"]
        wbw = self.t["animation"].get("word_by_word", False)
        layers = {True: Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0)),
                  False: Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))}
        to_end = cue.end - t

        kw_text = p.words[p.kw]
        kw_text = kw_text.upper() if K.get("uppercase") else kw_text
        kw_in = p.times[p.kw] if wbw else cue.start
        kimg = self._text_img(kw_text, self._fit(("k", id(cue)), kw_text, K), "keyword")
        ky = head_y if head_y is not None else (K.get("y_ratio", 0.19) * self.H)
        if t >= kw_in:
            s, a, dy = self._anim(t - kw_in, to_end)
            self._paste(layers[K["behind"]], kimg, self.W / 2, ky + dy, s, a)

        others = [(i, w) for i, w in enumerate(p.words) if i != p.kw]
        shown = [w for i, w in others if (not wbw) or t >= p.times[i]]
        if shown:
            up = S.get("uppercase")
            full = " ".join(w for _, w in others)
            ssize = self._fit(("s", id(cue)), full.upper() if up else full, S)
            txt = " ".join(shown)
            simg = self._text_img(txt.upper() if up else txt, ssize, "support")
            first_in = p.times[others[0][0]] if wbw else cue.start
            s, a, dy = self._anim(t - first_in, to_end)
            sy = ky - kimg.height / 2 - S["gap_ratio"] * self.H - simg.height / 2
            self._paste(layers[S["behind"]], simg, self.W / 2, sy + dy, 1.0 if wbw else s, a)
        return layers[True], layers[False]
