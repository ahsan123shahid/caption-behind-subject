import re
from typing import List, Dict, Any, Optional

def parse_time_to_seconds(time_str: str) -> float:
    """Converts '00:01:23,456' or '00:01:23.456' to seconds."""
    time_str = time_str.strip().replace(',', '.')
    parts = time_str.split(':')
    if len(parts) == 3:
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        minutes = float(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    return float(parts[0])

class CaptionItem:
    def __init__(self, start: float, end: float, text: str, words: Optional[List[Dict[str, Any]]] = None):
        self.start = start
        self.end = end
        self.text = text.strip()
        self.words = words or []

    def __repr__(self):
        return f"<CaptionItem {self.start:.2f}s - {self.end:.2f}s: '{self.text}'>"

def parse_srt(file_path: str) -> List[CaptionItem]:
    """
    Parses a standard .srt subtitle file into a list of CaptionItem objects.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    entries = re.split(r'\n\s*\n', content.strip())
    items = []

    for entry in entries:
        lines = [line.strip() for line in entry.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        
        # Check if first line is number or time
        time_line_idx = 1 if lines[0].isdigit() else 0
        if time_line_idx >= len(lines):
            continue

        time_line = lines[time_line_idx]
        if '-->' not in time_line:
            continue

        start_str, end_str = time_line.split('-->')
        start_sec = parse_time_to_seconds(start_str)
        end_sec = parse_time_to_seconds(end_str)

        text_lines = lines[time_line_idx + 1:]
        text = " ".join(text_lines)

        # Generate synthetic word timestamps if word-level isn't explicitly provided
        words_list = text.split()
        word_items = []
        if words_list:
            total_duration = max(0.1, end_sec - start_sec)
            word_duration = total_duration / len(words_list)
            for i, w in enumerate(words_list):
                w_start = start_sec + i * word_duration
                w_end = w_start + word_duration
                word_items.append({
                    "word": w,
                    "start": w_start,
                    "end": w_end
                })

        items.append(CaptionItem(start=start_sec, end=end_sec, text=text, words=word_items))

    return items
