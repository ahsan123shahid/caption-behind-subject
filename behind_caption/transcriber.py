"""Automatic Speech-to-Text Transcription with Whisper AI in Roman Urdu / Hinglish."""
from pathlib import Path
import re
import whisper

# Neutral prompting for Whisper without misleading narrative context
DEFAULT_PROMPT = "Roman Urdu, Hinglish, English speech transcript."

# Common Urdu vocabulary to clean Roman Urdu
URDU_WORD_MAP = {
    'نہیں': 'nahi', 'ایک': 'ek', 'کیا': 'kya', 'ہے': 'hai', 'ہیں': 'hain',
    'آپ': 'aap', 'کا': 'ka', 'کی': 'ki', 'کے': 'ke', 'سے': 'se',
    'میں': 'mein', 'کو': 'ko', 'نہ': 'na', 'تھا': 'tha', 'تھی': 'thi',
    'تھے': 'the', 'گا': 'ga', 'گی': 'gi', 'گے': 'ge', 'یہ': 'yeh',
    'وہ': 'woh', 'پر': 'par', 'اور': 'aur', 'بھی': 'bhi', 'تو': 'to',
    'کر': 'kar', 'رہا': 'raha', 'رہی': 'rahi', 'رہے': 'rahe', 'ہوں': 'hoon',
    'ہو': 'ho', 'بھائی': 'bhai', 'یار': 'yaar', 'اس': 'is', 'ان': 'un',
    'اب': 'ab', 'جب': 'jab', 'تب': 'tab', 'کب': 'kab', 'سب': 'sab',
    'ساتھ': 'saath', 'بات': 'baat', 'کام': 'kaam', 'لوگ': 'log', 'ویڈیو': 'video',
    'بڑا': 'bada', 'بڑی': 'badi', 'بڑے': 'bade', 'بہت': 'bohot', 'زیادہ': 'zyada',
    'کم': 'kam', 'اچھا': 'acha', 'اچھی': 'achi', 'اچھے': 'ache', 'اپنا': 'apna',
    'سکے': 'sake', 'سکتے': 'sakte', 'سکتی': 'sakti', 'ہزاروں': 'hazaron',
    'تقریبا': 'taqreeban', 'تکری': 'taqreeban', 'بن': 'ban', 'پہل': 'pehal',
    'دکھا': 'dikha', 'تکھا': 'dikha', 'چاہتے': 'chahte', 'پروڈکٹ': 'product',
    'پروڈکٹس': 'products', 'ٹائم': 'time', 'پرائس': 'price', 'اندر': 'andar',
    'دے': 'de', 'لینے': 'lene', 'لیں': 'lein', 'چلو': 'chalo', 'دیکھتے': 'dekhte'
}

URDU_CHAR_MAP = {
    'ا': 'a', 'آ': 'aa', 'ب': 'b', 'پ': 'p', 'ت': 't', 'ٹ': 't', 'ث': 's',
    'ج': 'j', 'چ': 'ch', 'ح': 'h', 'خ': 'kh', 'د': 'd', 'ڈ': 'd', 'ذ': 'z',
    'ر': 'r', 'ڑ': 'r', 'ز': 'z', 'ژ': 'zh', 'س': 's', 'ش': 'sh', 'ص': 's',
    'ض': 'z', 'ط': 't', 'ظ': 'z', 'ع': 'a', 'غ': 'gh', 'ف': 'f', 'ق': 'q',
    'ک': 'k', 'گ': 'g', 'ل': 'l', 'م': 'm', 'ن': 'n', 'ں': 'n', 'و': 'o',
    'ہ': 'h', 'ھ': 'h', 'ء': '', 'ی': 'i', 'ے': 'e', 'ۂ': 'h', 'ۃ': 't',
    'ئ': 'i', 'ؤ': 'o'
}

DEVA_CHAR_MAP = {
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo', 'ऋ': 'ri',
    'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'क': 'k', 'ख': 'kh', 'ग': 'g',
    'घ': 'gh', 'ङ': 'ng', 'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n', 'त': 't', 'थ': 'th',
    'द': 'd', 'ध': 'dh', 'न': 'n', 'प': 'p', 'फ': 'f', 'ब': 'b', 'भ': 'bh',
    'म': 'm', 'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh', 'ष': 'sh',
    'स': 's', 'ह': 'h', 'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo',
    'ृ': 'ri', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', '्': '', 'ं': 'n',
    'ँ': 'n', 'ः': 'h', '़': '', '०': '0', '१': '1', '२': '2', '३': '3',
    '४': '4', '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
}


def to_roman(text: str) -> str:
    """Converts Urdu or Hindi script to clean Latin Roman Urdu / Hinglish."""
    if not text:
        return text
    # Check if text contains non-latin Indic / Arabic characters
    has_indic = bool(re.search(r'[\u0600-\u06FF\u0900-\u097F]', text))
    if not has_indic:
        return text

    words = text.split()
    roman_words = []
    for w in words:
        clean_w = re.sub(r'[^\w]', '', w)
        if clean_w in URDU_WORD_MAP:
            roman_words.append(URDU_WORD_MAP[clean_w])
            continue

        chars = []
        for ch in w:
            if ch in URDU_CHAR_MAP:
                chars.append(URDU_CHAR_MAP[ch])
            elif ch in DEVA_CHAR_MAP:
                chars.append(DEVA_CHAR_MAP[ch])
            else:
                chars.append(ch)
        roman_words.append(''.join(chars))

    res = ' '.join(roman_words)
    return re.sub(r'\s+', ' ', res).strip()


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
    Transcribes video audio with Whisper AI, auto-converts Urdu/Hindi to Roman Urdu / Hinglish,
    creates snappy subtitle cues, and auto-tags hero keywords (*KEYWORD*) for 3D behind-subject effect.
    """
    video = Path(video_path)
    if not output_srt:
        output_srt = str(video.with_suffix(".auto.srt"))

    print(f"[auto-caption] Loading Whisper '{model_name}' model...")
    model = whisper.load_model(model_name)

    print(f"[auto-caption] Transcribing speech from '{video.name}'...")
    result = model.transcribe(
        str(video),
        initial_prompt=DEFAULT_PROMPT,
        word_timestamps=True
    )

    detected_lang = result.get("language", "en")
    print(f"[auto-caption] Detected speech language: {detected_lang.upper()}")

    # Flatten all words with timestamps and transliterate to Roman Urdu if needed
    all_words = []
    for seg in result.get("segments", []):
        if "words" in seg and seg["words"]:
            for w in seg["words"]:
                word_str = w["word"].strip()
                if word_str:
                    # Transliterate to Roman script if in Urdu / Hindi
                    roman_w = to_roman(word_str)
                    if roman_w:
                        all_words.append({
                            "word": roman_w,
                            "start": float(w["start"]),
                            "end": float(w["end"])
                        })
        else:
            text = seg.get("text", "").strip()
            if text:
                roman_text = to_roman(text)
                for w in roman_text.split():
                    all_words.append({
                        "word": w,
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
    print(f"[auto-caption] Successfully generated {len(cues)} auto-synced Roman/Hinglish captions -> {output_srt}")
    return str(out_path)
