"""
Tầng 1 — Logic xử lý: áp dụng dictionaries.py vào một câu,
trả về corrected_text + danh sách errors kèm offset.
"""
import re
from .dictionaries import TEENCODE_MAP, CONFUSION_INITIAL, TELEX_ERRORS
from .wordlist import get_wordlist

_TELEX_LOOKUP = {}
_WORDLIST = get_wordlist()
for correct_word, wrong_variants in TELEX_ERRORS.items():
    for wrong in wrong_variants:
        _TELEX_LOOKUP[wrong] = correct_word

_WORD_PATTERN = re.compile(r"\w+")

# Cụm teencode dài nhất trong TEENCODE_MAP có bao nhiêu từ — tính động,
# để nếu sau này thêm key 3 từ, code tự thích ứng không cần sửa tay.
_MAX_NGRAM = max(len(k.split()) for k in TEENCODE_MAP)
# Sắp key dài trước (2 ký tự: ch, tr, gi, ng) để tránh 'ng...' bị
# nhận nhầm thành 'n' + phần còn lại có chữ 'g' đứng đầu.
_CONFUSION_KEYS_BY_LEN = sorted(CONFUSION_INITIAL.keys(), key=len, reverse=True)

def _tokenize(text: str):
    """Trả về list các token dạng (start, end, text_gốc)."""
    return [(m.start(), m.end(), m.group()) for m in _WORD_PATTERN.finditer(text)]


def _split_initial(word: str):
    """Tách âm đầu (nếu nằm trong CONFUSION_INITIAL) khỏi phần còn lại.
    Trả về None nếu từ không bắt đầu bằng âm đầu nào trong danh sách."""
    for prefix in _CONFUSION_KEYS_BY_LEN:
        if word.startswith(prefix):
            return prefix, word[len(prefix):]
    return None
def _try_confusion_correction(word_lower: str):
    """Chỉ sửa khi word_lower KHÔNG có trong wordlist, và có ĐÚNG MỘT
    biến thể hoán đổi âm đầu hợp lệ. Nhiều hơn 1 -> quá mơ hồ, bỏ qua
    (an toàn hơn là đoán bừa)."""
    if word_lower in _WORDLIST:
        return None  # từ gốc đã đúng -> không đụng vào, dù trùng confusion set
    split = _split_initial(word_lower)
    if not split:
        return None
    prefix, rest = split
    candidates = [p + rest for p in CONFUSION_INITIAL[prefix]]
    valid = [c for c in candidates if c in _WORDLIST]
    return valid[0] if len(valid) == 1 else None
def _preserve_case(original: str, correction: str) -> str:
    """Nếu từ/cụm gốc viết hoa chữ đầu, giữ hoa chữ đầu ở bản sửa.
    Đây là heuristic đơn giản — không xử lý trường hợp ALL CAPS
    hay hoa ở giữa cụm, coi là giới hạn chấp nhận được ở bước này."""
    if original[:1].isupper() and correction:
        return correction[0].upper() + correction[1:]
    return correction


def correct_sentence(text: str) -> dict:
    tokens = _tokenize(text)
    errors = []
    i = 0
    while i < len(tokens):
        matched = False
        # Thử cụm dài nhất trước (greedy longest-match), ngắn dần về 1.
        for n in range(min(_MAX_NGRAM, len(tokens) - i), 0, -1):
            group = tokens[i:i + n]
            # Chỉ ghép cụm nếu các token liền kề cách nhau đúng 1 dấu cách
            # trong văn bản gốc — tránh ghép nhầm qua dấu câu/xuống dòng.
            if n > 1:
                contiguous = all(
                    text[group[k][1]:group[k + 1][0]] == " "
                    for k in range(len(group) - 1)
                )
                if not contiguous:
                    continue

            phrase_original = text[group[0][0]:group[-1][1]]
            phrase_lower = phrase_original.lower()

            correction, error_type = None, None
            if phrase_lower in TEENCODE_MAP:
                correction, error_type = TEENCODE_MAP[phrase_lower], "E05"
            elif n == 1 and phrase_lower in _TELEX_LOOKUP:
                correction, error_type = _TELEX_LOOKUP[phrase_lower], "E03"
            elif n == 1:
                confusion_fix = _try_confusion_correction(phrase_lower)
                if confusion_fix:
                    correction, error_type = confusion_fix, "E02"
            if correction and correction != phrase_lower:
                correction = _preserve_case(phrase_original, correction)
                errors.append({
                    "offset_start": group[0][0],
                    "offset_end": group[-1][1],
                    "type": error_type,
                    "original": phrase_original,
                    "corrected": correction,
                    "explanation": _explain(error_type),
                })
                i += n
                matched = True
                break

        if not matched:
            i += 1

    return {
        "corrected_text": _apply_corrections(text, errors),
        "errors": errors,
    }


def _explain(error_type: str) -> str:
    return {
        "E05": "Từ/cụm viết tắt kiểu teencode",
        "E03": "Lỗi gõ Telex/VNI",
        "E02": "Nhầm âm đầu (s/x, ch/tr, d/gi/r, l/n, n/ng)",
    }.get(error_type, "")


def _apply_corrections(text: str, errors: list) -> str:
    if not errors:
        return text
    result = []
    last_end = 0
    for err in sorted(errors, key=lambda e: e["offset_start"]):
        result.append(text[last_end:err["offset_start"]])
        result.append(err["corrected"])
        last_end = err["offset_end"]
    result.append(text[last_end:])
    return "".join(result)