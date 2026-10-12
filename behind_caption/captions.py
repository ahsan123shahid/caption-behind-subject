"""Minimal SRT parser. Wrap a word in *asterisks* to force it as the keyword."""
import re
from dataclasses import dataclass

_TS = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")


@dataclass
class Cue:
    start: float
    end: float
    text: str


def _sec(s: str) -> float:
    h, m, sec, frac = _TS.search(s).groups()
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(frac) / 10 ** len(frac)


def parse_srt(path: str) -> list[Cue]:
    raw = open(path, encoding="utf-8-sig").read().replace("\r", "")
    cues = []
    for block in re.split(r"\n\s*\n", raw.strip()):
        lines = block.split("\n")
        idx = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if idx is None:
            continue
        a, b = lines[idx].split("-->")
        text = " ".join(l.strip() for l in lines[idx + 1:] if l.strip())
        if text:
            cues.append(Cue(_sec(a), _sec(b), text))
    return sorted(cues, key=lambda c: c.start)
