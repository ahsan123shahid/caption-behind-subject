"""Automatic Speech-to-Text Transcription with Whisper AI & Hero Keyword Tagging."""
from pathlib import Path
import re
import whisper


def _format_srt_time(seconds: float) -> str:
    millis = int(round((seconds - int(seconds)) * 1000))
    secs = int(seconds) % 60
    mins = (int(seconds) // 60) % 60
    hours = int(seconds) // 3600
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def transcribe_video(
    video_path: str,
    output_srt: str = None,
    model_name: str = "base",
    words_per_cue: int = 3,
    max_duration: float = 1.6
) -> str:
    """
    Transcribes audio from a video using Whisper AI, groups words into snappy subtitle
    cues with exact timestamps, and auto-tags a prominent hero keyword (*KEYWORD*) for 3D behind-subject effect.
    """
    video = Path(video_path)
    if not output_srt:
        output_srt = str(video.with_suffix(".srt"))

    print(f"[auto-caption] Loading Whisper '{model_name}' model...")
    model = whisper.load_model(model_name)

    print(f"[auto-caption] Transcribing audio from '{video.name}'...")
    result = model.transcribe(str(video), word_timestamps=True)

    # Flatten all words with timestamps
    all_words = []
    for seg in result.get("segments", []):
        if "words" in seg and seg["words"]:
            for w in seg["words"]:
                word_str = w["word"].strip()
                if word_str:
                    all_words.append({
                        "word": word_str,
                        "start": float(w["start"]),
                        "end": float(w["end"])
                    })
        else:
            text = seg.get("text", "").strip()
            if text:
                all_words.append({
                    "word": text,
                    "start": float(seg["start"]),
                    "end": float(seg["end"])
                })

    if not all_words:
        raise RuntimeError("No speech detected in video audio.")

    # Group words into short, rhythmic subtitle cues
    cues = []
    curr_chunk = []
    chunk_start = 0.0

    for w in all_words:
        if not curr_chunk:
            curr_chunk.append(w)
            chunk_start = w["start"]
        else:
            dur = w["end"] - chunk_start
            pause = w["start"] - curr_chunk[-1]["end"]
            if len(curr_chunk) >= words_per_cue or dur >= max_duration or pause > 0.45:
                cues.append(curr_chunk)
                curr_chunk = [w]
                chunk_start = w["start"]
            else:
                curr_chunk.append(w)

    if curr_chunk:
        cues.append(curr_chunk)

    # Convert word chunks to SRT lines with auto-detected Hero Keyword
    srt_lines = []
    for idx, chunk in enumerate(cues, 1):
        c_start = chunk[0]["start"]
        c_end = max(chunk[-1]["end"], c_start + 0.6)
        
        words = [w["word"] for w in chunk]
        clean_words = [re.sub(r"[^\w]", "", w) for w in words]
        best_idx = 0
        max_len = 0
        for i, cw in enumerate(clean_words):
            if len(cw) > max_len:
                max_len = len(cw)
                best_idx = i

        tagged_words = []
        for i, w in enumerate(words):
            if i == best_idx and len(clean_words[i]) >= 3:
                clean = clean_words[i].upper()
                tagged_words.append(f"*{clean}*")
            else:
                tagged_words.append(w)

        cue_text = " ".join(tagged_words)
        srt_lines.append(f"{idx}\n{_format_srt_time(c_start)} --> {_format_srt_time(c_end)}\n{cue_text}\n")

    out_path = Path(output_srt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(srt_lines), encoding="utf-8")
    print(f"[auto-caption] Successfully generated {len(cues)} auto-synced captions -> {output_srt}")
    return str(out_path)
