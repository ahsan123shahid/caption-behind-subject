"""Bolti.ai / Kaizynn project JSON -> SRT exporter.

Understands the full project payload that bolti.ai exports:

    {"project": {"transcript": [{"id": "seg_..", "words": [...]}],
                 "clips": [...], ...}}

Rules:
* Cue boundaries come from the `ca-split` caption clips (`start` + `duration`)
  when the payload ships them. Otherwise the words are re-chunked with the same
  rhythm rules the auto-transcriber uses (`words_per_cue` / `max_duration` /
  pause gap).
* Exact word timestamps are preserved; `textLatn` is preferred over `text`.
* Words flagged `emphasis: true` are the Hero Keywords. The strongest
  emphasized run inside each cue is wrapped in `*asterisks*` so
  `captions.prepare_cues` promotes it to the behind-the-subject hero line.

CLI:
    python -m behind_caption.bolti project.json [-o project.srt]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .captions import COMMON_STOPWORDS

#: Shortest hero candidate `captions.prepare_cues` will accept.
MIN_HERO_LEN = 4

#: Default cue rhythm (mirrors transcriber.transcribe_video defaults).
DEFAULT_WORDS_PER_CUE = 4
DEFAULT_MAX_DURATION = 2.0
DEFAULT_PAUSE_GAP = 0.60
DEFAULT_MIN_CUE_DURATION = 0.60

_TRAILING_PUNCT = re.compile(r"^(.*?)([\W_]+)$", re.UNICODE)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _num(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt_ts(seconds: float) -> str:
    total_ms = int(round(max(seconds, 0.0) * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    hours, rem = divmod(total_s, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def project_from(data) -> dict:
    """Unwrap `{"project": {...}}` (or pass a bare project dict through)."""
    if isinstance(data, dict) and isinstance(data.get("project"), dict):
        return data["project"]
    return data


def load_project(path) -> dict:
    """Read a bolti/kaizynn `.json` export and return the project dict."""
    raw = Path(path).read_text(encoding="utf-8-sig")
    return project_from(json.loads(raw))


def flatten_words(project: dict) -> list[dict]:
    """Flatten `transcript[].words[]` into time-ordered word records."""
    words = []
    for seg in project.get("transcript") or []:
        for w in seg.get("words") or []:
            text = (w.get("textLatn") or w.get("text") or "").strip()
            if not text:
                continue
            start = _num(w.get("start"))
            end = max(_num(w.get("end"), start), start)
            words.append({
                "id": w.get("id"),
                "text": text,
                "start": start,
                "end": end,
                "emphasis": bool(w.get("emphasis")),
                "importance": _num(w.get("importanceScore")),
                "tier": w.get("emphasisTier"),
            })
    words.sort(key=lambda w: (w["start"], w["end"]))
    return words


# --------------------------------------------------------------------------- #
# cue text (hero keyword marking)
# --------------------------------------------------------------------------- #
def _emphasis_runs(words: list[dict]) -> list[tuple[int, int]]:
    """Index ranges of maximal runs of consecutive emphasized words."""
    runs, i = [], 0
    while i < len(words):
        if words[i]["emphasis"]:
            j = i
            while j + 1 < len(words) and words[j + 1]["emphasis"]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def _wrap_run(run: list[dict]) -> str | None:
    """Wrap an emphasized run in `*ASTERISKS*`, or None if it is not hero-worthy."""
    raw = " ".join(w["text"] for w in run).strip()
    inner, suffix = raw, ""
    m = _TRAILING_PUNCT.match(raw)
    if m and (m.group(1) or "").strip():
        inner, suffix = m.group(1).rstrip(), m.group(2)

    # Same acceptance test prepare_cues() applies to a forced hero candidate.
    if len(inner) < MIN_HERO_LEN or inner.lower() in COMMON_STOPWORDS:
        return None
    return f"*{inner.upper()}*{suffix}"


def build_cue_text(words: list[dict], mark_emphasis: bool = True) -> str:
    """Join a cue's words, wrapping the best emphasized run as the hero keyword."""
    if not words:
        return ""
    if not mark_emphasis:
        return re.sub(r"\s+", " ", " ".join(w["text"] for w in words)).strip()

    best = None
    best_key = None
    for (i, j) in _emphasis_runs(words):
        run = words[i:j + 1]
        wrapped = _wrap_run(run)
        if wrapped is None:
            continue
        key = (max(w["importance"] for w in run), j - i)
        if best_key is None or key > best_key:
            best_key, best = key, (i, j, wrapped)

    if best is None:
        return re.sub(r"\s+", " ", " ".join(w["text"] for w in words)).strip()

    i, j, wrapped = best
    parts = [w["text"] for w in words[:i]] + [wrapped] + [w["text"] for w in words[j + 1:]]
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


# --------------------------------------------------------------------------- #
# cue assembly
# --------------------------------------------------------------------------- #
def _chunk_words(
    words: list[dict],
    words_per_cue: int = DEFAULT_WORDS_PER_CUE,
    max_duration: float = DEFAULT_MAX_DURATION,
    pause_gap: float = DEFAULT_PAUSE_GAP,
) -> list[list[dict]]:
    """Group words into rhythmic cues (same rules as the auto-transcriber)."""
    cues, curr, chunk_start = [], [], 0.0
    for w in words:
        if not curr:
            curr, chunk_start = [w], w["start"]
            continue
        duration = w["end"] - chunk_start
        pause = w["start"] - curr[-1]["end"]
        if (len(curr) >= words_per_cue or duration >= max_duration or pause > pause_gap) and (
            len(curr) >= 2 or duration >= 0.9
        ):
            cues.append(curr)
            curr, chunk_start = [w], w["start"]
        else:
            curr.append(w)

    if curr:
        if len(curr) == 1 and cues:
            cues[-1].extend(curr)
        else:
            cues.append(curr)
    return cues


def _cues_from_clips(
    project: dict,
    words: list[dict],
    chunk_opts: dict,
) -> list[dict]:
    """Build cues from the preset's caption clips, recovering any orphan words."""
    by_id = {w["id"]: w for w in words if w.get("id")}
    used: set = set()
    cues = []

    for clip in project.get("clips") or []:
        if clip.get("kind") is not None and clip.get("kind") != "caption":
            continue
        clip_words = []
        for wid in clip.get("wordIds") or []:
            w = by_id.get(wid)
            if w is None:
                continue
            clip_words.append(w)
            used.add(wid)
        if not clip_words:
            continue

        start = _num(clip.get("start"), clip_words[0]["start"])
        end = start + _num(clip.get("duration"))
        end = max(end, clip_words[-1]["end"])
        cues.append({"start": start, "end": end, "words": clip_words})

    # Words the clipper never referenced still deserve to be captioned.
    orphans = [w for w in words if w.get("id") not in used]
    for chunk in _chunk_words(orphans, **chunk_opts):
        cues.append({
            "start": chunk[0]["start"],
            "end": max(chunk[-1]["end"], chunk[0]["start"] + DEFAULT_MIN_CUE_DURATION),
            "words": chunk,
        })
    return cues


def _finalize(cues: list[dict], mark_emphasis: bool = True) -> list[tuple]:
    """Sort, remove overlaps and render each cue's text."""
    cues = sorted(cues, key=lambda c: (c["start"], c["end"]))
    out = []
    for idx, cue in enumerate(cues):
        text = build_cue_text(cue["words"], mark_emphasis=mark_emphasis)
        if not text:
            continue
        start = cue["start"]
        end = cue["end"]
        if idx + 1 < len(cues):
            nxt = cues[idx + 1]["start"]
            if nxt > start:
                end = min(end, nxt)
        if end <= start:
            end = start + 0.30
        out.append((start, end, text))
    return out


def build_cues(
    data,
    words_per_cue: int = DEFAULT_WORDS_PER_CUE,
    max_duration: float = DEFAULT_MAX_DURATION,
    pause_gap: float = DEFAULT_PAUSE_GAP,
    mark_emphasis: bool = True,
) -> list[tuple]:
    """Return `[(start, end, text), ...]` for a bolti project (path or dict)."""
    project = load_project(data) if isinstance(data, (str, Path)) else project_from(data)
    words = flatten_words(project)
    if not words:
        raise ValueError("No word-level transcript found in project JSON.")

    if project.get("clips"):
        cues = _cues_from_clips(
            project, words,
            {"words_per_cue": words_per_cue, "max_duration": max_duration, "pause_gap": pause_gap},
        )
    else:
        cues = []
        for chunk in _chunk_words(words, words_per_cue, max_duration, pause_gap):
            cues.append({
                "start": chunk[0]["start"],
                "end": max(chunk[-1]["end"], chunk[0]["start"] + DEFAULT_MIN_CUE_DURATION),
                "words": chunk,
            })
    return _finalize(cues, mark_emphasis=mark_emphasis)


def cues_to_srt(cues: list[tuple]) -> str:
    blocks = [
        f"{i}\n{_fmt_ts(start)} --> {_fmt_ts(end)}\n{text}"
        for i, (start, end, text) in enumerate(cues, 1)
    ]
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def project_to_srt(data, output=None, **kwargs) -> str:
    """Convert project data to SRT text, writing it when `output` is given."""
    srt = cues_to_srt(build_cues(data, **kwargs))
    if output is None:
        return srt
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(srt, encoding="utf-8")
    return str(out)


def convert(json_path, output=None, **kwargs) -> str:
    """`bolti project.json` -> `.srt`. Returns the written SRT path (or text)."""
    default_out = str(Path(json_path).with_suffix(".srt"))
    return project_to_srt(load_project(json_path), output or default_out, **kwargs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m behind_caption.bolti",
                                 description="Convert a bolti.ai project JSON into an SRT with *hero keywords*.")
    ap.add_argument("json", help="path to the bolti/kaizynn project .json export")
    ap.add_argument("-o", "--out", default=None, help="output .srt path (default: alongside the json)")
    ap.add_argument("--words-per-cue", type=int, default=DEFAULT_WORDS_PER_CUE,
                    help="max words per cue when the payload has no caption clips (default: 4)")
    ap.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION,
                    help="max cue length in seconds when chunking (default: 2.0)")
    ap.add_argument("--plain", action="store_true", help="omit *hero keyword* marks")
    a = ap.parse_args(argv)

    out = convert(
        a.json,
        a.out,
        words_per_cue=a.words_per_cue,
        max_duration=a.max_duration,
        mark_emphasis=not a.plain,
    )
    print(f"[bolti-json] Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
