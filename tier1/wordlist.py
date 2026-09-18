"""
Tầng 1 — Wordlist tiếng Việt hợp lệ, dùng để xác nhận một từ có
tồn tại trong tiếng Việt hay không (phục vụ confusion-set correction).
Nguồn: Hồ Ngọc Đức's Free Vietnamese Dictionary Project, qua
https://github.com/duyet/vietnamese-wordlist (Viet74K.txt), GPL.
"""
import os

_WORDLIST_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Viet74K.txt")
_wordlist_cache = None


def get_wordlist() -> set:
    """Load wordlist 1 lần duy nhất (lazy), cache lại cho các lần gọi sau."""
    global _wordlist_cache
    if _wordlist_cache is None:
        with open(_WORDLIST_PATH, encoding="utf-8") as f:
            _wordlist_cache = {line.strip().lower() for line in f if line.strip()}
    return _wordlist_cache