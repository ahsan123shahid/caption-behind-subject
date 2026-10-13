import re

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
    'پروڈکٹس': 'products', 'ٹائم': 'time', 'پرائس': 'price', 'اندر': 'andar'
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
    'म': 'm', 'य': 'y', 'ر': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh', 'ष': 'sh',
    'स': 's', 'ह': 'h', 'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo',
    'ृ': 'ri', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', '्': '', 'ं': 'n',
    'ँ': 'n', 'ः': 'h', '़': '', '०': '0', '१': '1', '२': '2', '३': '3',
    '४': '4', '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
}


def to_roman(text: str) -> str:
    """Converts Urdu or Hindi script to clean Latin Roman Urdu / Hinglish."""
    if not text:
        return text
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
