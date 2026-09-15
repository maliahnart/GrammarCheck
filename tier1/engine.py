"""
Tầng 1 — Logic xử lý: áp dụng dictionaries.py vào một câu,
trả về corrected_text + danh sách errors kèm offset.
"""
import re
from .dictionaries import TEENCODE_MAP, CONFUSION_INITIAL, TELEX_ERRORS

# Đảo ngược TELEX_ERRORS ("đúng" -> [sai]) thành lookup thuận chiều
# ("sai" -> "đúng"), giống chiều của TEENCODE_MAP. Build 1 lần lúc
# import module, không build lại mỗi lần gọi correct_sentence().
_TELEX_LOOKUP = {}
for correct_word, wrong_variants in TELEX_ERRORS.items():
    for wrong in wrong_variants:
        _TELEX_LOOKUP[wrong] = correct_word

# \w trong Python 3 regex mặc định Unicode-aware, tự nhận diện chữ
# có dấu tiếng Việt (không cần khai báo range Unicode thủ công).
_WORD_PATTERN = re.compile(r"\w+")


def correct_sentence(text: str) -> dict:
    """Sửa lỗi Tầng 1 cho một câu. Trả về dict theo schema errors[]
    đã định nghĩa trong SRS mục 5.4/9.1."""
    errors = []

    for match in _WORD_PATTERN.finditer(text):
        word = match.group()
        word_lower = word.lower()
        start, end = match.start(), match.end()

        correction, error_type = None, None
        if word_lower in TEENCODE_MAP:
            correction, error_type = TEENCODE_MAP[word_lower], "E05"
        elif word_lower in _TELEX_LOOKUP:
            correction, error_type = _TELEX_LOOKUP[word_lower], "E03"

        if correction and correction != word_lower:
            errors.append({
                "offset_start": start,
                "offset_end": end,
                "type": error_type,
                "original": word,
                "corrected": correction,
                "explanation": _explain(error_type),
            })

    return {
        "corrected_text": _apply_corrections(text, errors),
        "errors": errors,
    }


def _explain(error_type: str) -> str:
    return {
        "E05": "Từ viết tắt kiểu teencode",
        "E03": "Lỗi gõ Telex/VNI",
    }.get(error_type, "")


def _apply_corrections(text: str, errors: list) -> str:
    """Dựng lại corrected_text MỘT LẦN từ offset của câu gốc —
    không sửa trực tiếp trong lúc duyệt (xem giải thích ở trên)."""
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