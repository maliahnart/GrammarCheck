"""Tiện ích xử lý văn bản tiếng Việt dùng chung cho toàn bộ dự án.

Gồm: chuẩn hoá Unicode, tách token, bỏ dấu, danh sách viết tắt / từ cảm thán,
và các hàm so khớp dùng cho sinh lỗi, mô hình và bộ bảo toàn văn phong.
"""
from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Chuẩn hoá & tách token
# ---------------------------------------------------------------------------
# Token "từ": chữ/số (có thể nối bằng - hoặc '), còn lại là cụm ký tự dấu câu / emoticon.
_EMOTICON_TOKEN = r"(?:[:;=][-']?[)(\]\[DPpOo3/\|*v]+|\^_?\^|<3+|-_-|T_T)(?!\w)"
_TOKEN_RE = re.compile(_EMOTICON_TOKEN + r"|\w+(?:['’-]\w+)*|[^\w\s]+", re.UNICODE)
_WORD_RE = re.compile(r"^\w+(?:['’-]\w+)*$", re.UNICODE)

TONE_MARKS = {"́", "̀", "̉", "̃", "̣"}  # sắc, huyền, hỏi, ngã, nặng
VOWEL_MODS = {"̂", "̆", "̛"}  # mũ (â ê ô), trăng (ă), móc (ơ ư)


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", nfc(text)).strip()


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(nfc(text))


def tokenize_with_spans(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def is_word(tok: str) -> bool:
    return bool(_WORD_RE.match(tok)) and not tok.isdigit()


def eval_form(text: str) -> str:
    """Dạng chuẩn hoá thống nhất dùng khi chấm điểm: NFC + tách dấu câu + 1 khoảng trắng."""
    return " ".join(tokenize(text))


def remove_diacritics(text: str) -> str:
    s = unicodedata.normalize("NFD", text)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")


def remove_tone(syl: str) -> str:
    """Chỉ bỏ dấu thanh, giữ dấu mũ/móc/trăng (hồng -> hông, được -> đươc)."""
    s = unicodedata.normalize("NFD", syl)
    return nfc("".join(c for c in s if c not in TONE_MARKS))


def has_diacritic(text: str) -> bool:
    return remove_diacritics(text) != text


def match_case(src: str, new: str) -> str:
    """Áp kiểu viết hoa của src lên new."""
    if not src or not new:
        return new
    if src.isupper() and len(src) > 1:
        return new.upper()
    if src[0].isupper():
        return new[0].upper() + new[1:]
    return new


def norm_key(text: str) -> str:
    """Khoá gần-trùng: chữ thường, bỏ dấu, chỉ giữ chữ số. Dùng để chống rò rỉ train/test."""
    s = remove_diacritics(nfc(text).lower())
    return re.sub(r"[^a-z0-9]+", "", s)


# ---------------------------------------------------------------------------
# Viết tắt phổ biến (teencode) -> dạng chuẩn
# ---------------------------------------------------------------------------
ABBREVIATIONS: dict[str, str] = {
    "e": "em", "a": "anh", "c": "chị", "ko": "không", "k": "không", "hok": "không",
    "kh": "không", "khg": "không", "kg": "không", "hem": "không",
    "mik": "mình", "mk": "mình", "mn": "mọi người", "mng": "mọi người",
    "dc": "được", "đc": "được", "dk": "được", "đk": "được",
    "j": "gì", "gi": "gì", "vs": "với", "ng": "người", "nc": "nước",
    "bh": "bây giờ", "bjo": "bây giờ", "r": "rồi", "cx": "cũng", "cg": "cũng",
    "bít": "biết", "bik": "biết", "bn": "bao nhiêu", "v": "vậy", "z": "vậy",
    "ntn": "như thế nào", "lm": "làm", "trc": "trước", "ns": "nói", "nch": "nói chuyện",
    "nhìu": "nhiều", "wa": "quá", "qá": "quá", "ik": "đi", "thik": "thích", "iu": "yêu",
    "ad": "admin", "sn": "sinh nhật", "tks": "cảm ơn", "thanks": "cảm ơn",
}
# Viết tắt đa nghĩa: chỉ mở rộng khi có ngữ cảnh rõ / không nằm trong tập dùng để sinh lỗi.
AMBIGUOUS_ABBREVIATIONS = {"a", "c", "r", "v", "gi", "ng", "k", "ad", "thanks"}

# Ngược lại: dạng chuẩn -> các dạng viết tắt (để sinh lỗi). Chỉ dùng dạng 1 âm tiết.
ABBREV_NOISE: dict[str, list[str]] = {}
for _ab, _full in ABBREVIATIONS.items():
    if " " not in _full and _ab not in {"gi", "ad", "thanks", "tks"}:
        ABBREV_NOISE.setdefault(_full, []).append(_ab)

# ---------------------------------------------------------------------------
# Thành phần văn phong cần bảo toàn
# ---------------------------------------------------------------------------
INTERJECTIONS = {
    "hihi", "haha", "hehe", "huhu", "hix", "hic", "hjhj", "hoho", "ahihi", "kaka", "keke",
    "kkk", "kk", "hiuhiu", "hmm", "hm", "uhm", "ừm", "ừ", "ờ", "ơ", "ôi", "ui", "trời",
    "wow", "oke", "ok", "okela", "okie", "yeah", "hả", "hở", "ha", "nha", "nhé", "nhỉ",
    "nè", "á", "ạ", "à", "ơi", "nhá", "hen", "nghen", "đó", "đấy", "thôi", "luôn",
}
EMOTICON_RE = re.compile(
    r"^(?:[:;=xX8][-']?[)(\]\[DPpOo3/\\|*]+|[)(]+:|\^\^|\^_\^|<3+|=+\)+|:+\)+|:v|-_-|T_T)$"
)
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿\U0001F000-\U0001F2FF]")
URL_RE = re.compile(r"^(?:https?://|www\.)\S+$|^\S+@\S+\.\S+$")


def is_interjection(tok: str) -> bool:
    low = tok.lower()
    if low in INTERJECTIONS:
        return True
    # haha, hahaha, hihihi, kkkk ...
    return bool(re.fullmatch(r"(?:ha|hi|he|hu|ho|hj|hì|há|hí|hê){2,}|k{2,}", low))


def is_emoticon(tok: str) -> bool:
    return bool(EMOTICON_RE.match(tok)) or bool(EMOJI_RE.search(tok))


def is_protected(tok: str) -> bool:
    """Token mang văn phong/biểu cảm, không được tự ý sửa hoặc xoá."""
    return is_interjection(tok) or is_emoticon(tok) or bool(URL_RE.match(tok))


# ---------------------------------------------------------------------------
# Quy tắc ngữ pháp đơn giản (dùng chung cho sinh lỗi, ML baseline và bộ bảo toàn)
# ---------------------------------------------------------------------------
# Hư từ hay bị lặp do gõ nhầm: "của của" -> "của". Không gồm từ láy hợp lệ (ngày ngày, từ từ).
DUPLICABLE_FUNCTION_WORDS = {
    "của", "là", "và", "các", "những", "được", "có", "đã", "sẽ", "đang", "trong", "với",
    "cho", "để", "thì", "mà", "nên", "nhưng", "một", "này", "khi", "về", "từ", "tại",
}
# Từ láy hợp lệ dù trùng với hư từ ở trên ("từ từ") -> quy tắc sửa không được xoá.
# (Danh sách sinh lỗi giữ nguyên để tái lập đúng dữ liệu đã tạo.)
VALID_REDUPLICATIONS = {"từ"}
# Quan hệ từ thừa đứng ngay sau một quan hệ từ khác: "nhưng nên" -> "nhưng".
REDUNDANT_AFTER = {"nhưng": {"nên"}, "nên": {"nhưng"}, "vì": {"do"}, "tuy": {"dù"}}


def edit_distance(a, b) -> int:
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.distance(a, b)
