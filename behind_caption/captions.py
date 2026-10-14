"""Minimal SRT parser. Wrap a word in *asterisks* to force it as the keyword."""
import re
from dataclasses import dataclass

_TS = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")


@dataclass
class Cue:
    start: float
    end: float
    text: str
    is_hero: bool = False
    hero_text: str = ""
    front_text: str = ""


COMMON_STOPWORDS = {
    # English stopwords
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "is", "are", "am", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "and", "or", "but", "if",
    "this", "that", "these", "those", "it", "its", "you", "your", "he", "she", "we", "they", "me",
    "my", "mine", "him", "her", "us", "them", "not", "just", "about", "from", "by", "as",
    # Roman Urdu / Hindi stopwords & common auxiliary verbs / pronouns
    "hai", "hain", "hoon", "ho", "tha", "thi", "the", "ka", "ki", "ke", "ko", "se", "me", "mein",
    "par", "pe", "aur", "ya", "yeh", "ye", "woh", "wo", "kar", "karo", "raha", "rahi", "rahe",
    "gaya", "gayi", "gaye", "to", "toh", "bhi", "hi", "kya", "kyun", "kab", "kahan", "kaise",
    "ap", "aap", "hum", "main", "maine", "meri", "mera", "mere", "degi", "wali", "wale", "wala",
    "bhai", "sab", "tu", "agar", "aane", "lagata", "lagate", "lagati", "lagaye", "lagaya",
    "hoga", "hogi", "honge", "karta", "karti", "karte", "kartein", "banae", "banaye", "nahi",
    "nahin", "mat", "let", "tab", "kahin", "shabd", "saath", "hota", "hoti", "hote", "kuch",
    "sirf", "ab", "jab", "phir", "kyon", "kise", "unka", "unki", "unke", "iska", "iski", "iske"
}


def prepare_cues(cues: list[Cue], hero_interval: int = 3) -> list[Cue]:
    """
    Intelligently paces cues between clean Front Subtitles and Behind-the-Subject Hero Keywords.
    
    Rules:
    - Standard Front Subtitles show by default (70-80% of lines).
    - Hero Keyword pops behind the subject only every 2-4 lines (paced), highlighting impactful words.
    - Mutual Exclusivity:
      - If it's a Hero Cue: only the Hero keyword appears behind the subject (front is empty).
      - If it's a Front Subtitle Cue: the clean subtitle appears in front (behind is empty).
    """
    last_hero_idx = -hero_interval

    for idx, cue in enumerate(cues):
        forced = [f.strip() for f in re.findall(r'\*([^*]+)\*', cue.text) if f.strip()]
        # Strip asterisks for clean front display
        plain_text = re.sub(r'\*([^*]+)\*', r'\1', cue.text).strip()
        plain_text = re.sub(r'\s+', ' ', plain_text)
        cue.front_text = plain_text

        is_hero_cue = False
        hero_kw = ""

        if forced:
            # If user marked words with *...*, pace them with minimum hero_interval gap (e.g. 3 cues)
            candidate = forced[0]
            if (idx - last_hero_idx) >= hero_interval and candidate.lower() not in COMMON_STOPWORDS and len(candidate) >= 4:
                hero_kw = candidate
                is_hero_cue = True
        else:
            # Automatic pacing: every `hero_interval` cues
            if (idx - last_hero_idx) >= hero_interval:
                words = [re.sub(r'[^\w]', '', w) for w in plain_text.split()]
                valid_words = [w for w in words if w.lower() not in COMMON_STOPWORDS and len(w) >= 4]
                if valid_words:
                    hero_kw = max(valid_words, key=len)
                    is_hero_cue = True

        if is_hero_cue and hero_kw:
            cue.is_hero = True
            cue.hero_text = hero_kw.upper()
            last_hero_idx = idx
        else:
            cue.is_hero = False
            cue.hero_text = ""

    return cues


def _sec(s: str) -> float:
    h, m, sec, frac = _TS.search(s).groups()
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(frac) / 10 ** len(frac)


def parse_srt(path: str, hero_interval: int = 3) -> list[Cue]:
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
    sorted_cues = sorted(cues, key=lambda c: c.start)
    return prepare_cues(sorted_cues, hero_interval=hero_interval)
