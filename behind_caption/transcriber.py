"""Fast & Accurate Speech-to-Text via Groq API (Whisper-Large-v3 + Qwen Transliteration)
with automatic fallback to local Whisper.
"""
import os
import re
import subprocess
import tempfile
from pathlib import Path
import requests

def _load_groq_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key.strip()
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("GROQ_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


GROQ_API_KEY = _load_groq_key()
GROQ_AUDIO_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


def _format_srt_time(seconds: float) -> str:
    millis = int(round((seconds - int(seconds)) * 1000))
    secs = int(seconds) % 60
    mins = (int(seconds) // 60) % 60
    hours = int(seconds) // 3600
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def _has_indic(text: str) -> bool:
    """Returns True if text contains Arabic/Urdu or Devanagari characters."""
    return bool(re.search(r'[\u0600-\u06FF\u0900-\u097F]', text))


def _transliterate_with_groq_llm(words_list: list, api_key: str) -> list:
    """
    Translates a list of words containing Urdu/Hindi into natural Roman Urdu / Hinglish
    using Groq's high-speed Qwen model while preserving exact word-by-word alignment.
    """
    joined_text = " ".join([w["word"] for w in words_list])
    if not _has_indic(joined_text):
        return words_list

    prompt = (
        "You are an expert transliterator for social media video captions. "
        "Convert the following Urdu / Hindi transcript into natural Roman Urdu / Hinglish (Latin English alphabet only). "
        "Rules:\n"
        "1. Do NOT translate meanings. Keep the exact words spoken, just write them phonetically in English letters (Roman Urdu).\n"
        "2. Keep English words like teeth, whitening, products, price, smile, actual, brush as English.\n"
        "3. Return ONLY the transliterated words separated by spaces. No explanations, no punctuation other than commas."
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "qwen/qwen3.8-27b",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": joined_text}
        ],
        "temperature": 0.1
    }

    try:
        r = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=10)
        if r.status_code == 200:
            roman_text = r.json()["choices"][0]["message"]["content"].strip()
            roman_tokens = roman_text.split()
            # If token lengths match or closely match, map them back to timestamps
            if abs(len(roman_tokens) - len(words_list)) <= max(2, len(words_list) // 4):
                aligned = []
                for i in range(len(words_list)):
                    word_label = roman_tokens[i] if i < len(roman_tokens) else words_list[i]["word"]
                    aligned.append({
                        "word": word_label,
                        "start": words_list[i]["start"],
                        "end": words_list[i]["end"]
                    })
                return aligned
    except Exception as e:
        print(f"[groq-llm] Notice: Transliteration fallback: {e}")

    # Fallback to local phonetic mapper if LLM alignment differs
    from .transcriber_fallback import to_roman
    for w in words_list:
        w["word"] = to_roman(w["word"])
    return words_list


def _transcribe_groq(video_path: Path, api_key: str) -> list:
    """Extracts audio and calls Groq Whisper-Large-v3 API."""
    print(f"[groq-api] Transcribing '{video_path.name}' via Groq Whisper-Large-v3...")

    # Extract temporary compressed audio (16kHz mono mp3/wav) for rapid upload
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_audio:
        tmp_audio_path = tmp_audio.name

    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ar", "16000", "-ac", "1", "-b:a", "64k",
            tmp_audio_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        headers = {"Authorization": f"Bearer {api_key}"}
        with open(tmp_audio_path, "rb") as f:
            files = {"file": (Path(tmp_audio_path).name, f, "audio/mp3")}
            data = {
                "model": "whisper-large-v3",
                "response_format": "verbose_json",
                "timestamp_granularities[]": "word"
            }
            r = requests.post(GROQ_AUDIO_URL, headers=headers, files=files, data=data, timeout=30)

        if r.status_code != 200:
            raise RuntimeError(f"Groq API error ({r.status_code}): {r.text}")

        res = r.json()
        raw_words = res.get("words", [])
        if not raw_words:
            raise RuntimeError("No words detected in audio.")

        print(f"[groq-api] Transcription complete in <2s! ({len(raw_words)} words, detected: {res.get('language')})")

        words = []
        for w in raw_words:
            w_str = w.get("word", "").strip()
            if w_str:
                words.append({
                    "word": w_str,
                    "start": float(w.get("start", 0)),
                    "end": float(w.get("end", 0))
                })

        # If South Asian speech detected, transliterate into natural Roman Urdu / Hinglish
        if _has_indic(" ".join([w["word"] for w in words[:20]])):
            print("[groq-api] Converting to natural Roman Urdu / Hinglish via Groq Qwen...")
            words = _transliterate_with_groq_llm(words, api_key)

        return words

    finally:
        if os.path.exists(tmp_audio_path):
            try:
                os.remove(tmp_audio_path)
            except Exception:
                pass


def _transcribe_local(video_path: Path, model_name: str = "base") -> list:
    """Fallback to local CPU Whisper if Groq API is unavailable."""
    import whisper
    print(f"[local-whisper] Running local Whisper ({model_name})...")
    model = whisper.load_model(model_name)
    result = model.transcribe(str(video_path), word_timestamps=True)

    words = []
    for seg in result.get("segments", []):
        if "words" in seg and seg["words"]:
            for w in seg["words"]:
                w_str = w["word"].strip()
                if w_str:
                    words.append({
                        "word": w_str,
                        "start": float(w["start"]),
                        "end": float(w["end"])
                    })
        else:
            for w in seg.get("text", "").split():
                words.append({
                    "word": w,
                    "start": float(seg["start"]),
                    "end": float(seg["end"])
                })
    return words


def transcribe_video(
    video_path: str,
    output_srt: str = None,
    words_per_cue: int = 3,
    max_duration: float = 1.6
) -> str:
    """
    Main transcription entrypoint:
    Uses Groq Whisper-Large-v3 API + Qwen Roman Urdu transliteration (<2s).
    Falls back gracefully to local Whisper if offline.
    """
    video = Path(video_path)
    if not output_srt:
        output_srt = str(video.with_suffix(".auto.srt"))

    # Try Groq API first
    all_words = None
    if GROQ_API_KEY:
        try:
            all_words = _transcribe_groq(video, GROQ_API_KEY)
        except Exception as e:
            print(f"[groq-api] Warning: API request failed ({e}). Falling back to local Whisper...")

    if not all_words:
        all_words = _transcribe_local(video)

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
    from .captions import COMMON_STOPWORDS
    last_tagged_idx = -3
    srt_lines = []
    for idx, chunk in enumerate(cues, 1):
        c_start = chunk[0]["start"]
        c_end = max(chunk[-1]["end"], c_start + 0.6)

        words = [w["word"] for w in chunk]
        clean_words = [re.sub(r"[^\w]", "", w) for w in words]

        should_tag = (idx - last_tagged_idx) >= 3
        best_i = -1
        if should_tag:
            candidates = [
                (i, len(cw)) for i, cw in enumerate(clean_words)
                if cw.lower() not in COMMON_STOPWORDS and len(cw) >= 4
            ]
            if candidates:
                best_i = max(candidates, key=lambda x: x[1])[0]
                last_tagged_idx = idx

        tagged_words = []
        for i, w in enumerate(words):
            if i == best_i:
                clean = clean_words[i].upper()
                tagged_words.append(f"*{clean}*")
            else:
                tagged_words.append(w)

        cue_text = " ".join(tagged_words)
        srt_lines.append(f"{idx}\n{_format_srt_time(c_start)} --> {_format_srt_time(c_end)}\n{cue_text}\n")

    out_path = Path(output_srt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(srt_lines), encoding="utf-8")
    print(f"[auto-caption] Successfully generated {len(cues)} auto-synced Hinglish captions -> {output_srt}")
    return str(out_path)
