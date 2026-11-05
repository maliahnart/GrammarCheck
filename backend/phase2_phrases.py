"""Optional phrase suggestions over original text and offsets."""

import re
from functools import lru_cache
from unicodedata import normalize

from backend.schemas import Change
from backend.service import document_spans, normalize_with_offsets
from experimental_core.engine import intersects
from experimental_phase2.engine import hard_spans
from vn_text import match_case, remove_diacritics


FOOTBALL_CUES = (
    "trọng tài", "cầu thủ", "việt vị", "vạch vôi", "khung thành",
    "bàn thắng", "thủ môn", "cú sút", "bóng đá", "phạt góc",
)
# Prefer a full collocation over an overlapping shorter entry.
FOOTBALL_PHRASES = (
    "bóng đã qua vạch vôi", "trọng tài biên", "phất cờ", "việt vị", "vạch vôi",
)
COMMON_PHRASES = ("thậm chí cũng",)


@lru_cache
def phrase_pattern(phrase):
    words = [
        "(?:" + re.escape(word) + "|" + re.escape(remove_diacritics(word)) + ")"
        for word in phrase.split()
    ]
    return re.compile(r"(?<!\w)" + r"[ \t]+".join(words) + r"(?!\w)", re.IGNORECASE)


def football_cues(text):
    return {phrase for phrase in FOOTBALL_CUES if phrase_pattern(phrase).search(text)}


def phrase_suggestions(text, *, previous="", following="", protected_terms=()):
    normalized, boundaries = normalize_with_offsets(text)
    locks = hard_spans(normalized, [normalize("NFC", term) for term in protected_terms])
    surrounding = football_cues(normalize("NFC", previous)) | football_cues(normalize("NFC", following))
    suggestions = []
    # The original document model can split one sentence into multiple chunks.
    for start, end in document_spans(normalized, limit=len(normalized) + 1):
        sentence = normalized[start:end]
        football = len(football_cues(sentence) | surrounding) >= 2
        phrases = COMMON_PHRASES + (FOOTBALL_PHRASES if football else ())
        occupied = list(locks)
        for phrase in phrases:
            for match in phrase_pattern(phrase).finditer(sentence):
                left, right = start + match.start(), start + match.end()
                if intersects(left, right, occupied):
                    continue
                occupied.append((left, right))
                # Keep the user's casing and the spacing between syllables.
                parts = re.split(r"([ \t]+)", match.group())
                parts[::2] = [
                    match_case(source, target)
                    for source, target in zip(parts[::2], phrase.split())
                ]
                replacement = "".join(parts)
                if replacement == match.group():
                    continue
                original_left, original_right = boundaries[left], boundaries[right]
                suggestions.append(Change(
                    id=f"phase2-phrase-{original_left}-{original_right}",
                    start=original_left, end=original_right,
                    source=text[original_left:original_right], replacement=replacement,
                    kind="diacritic", explanation="Gợi ý khôi phục dấu theo cụm từ",
                    accepted=False, reason="Đối chiếu cả cụm từ trước khi áp dụng.",
                ))
    return sorted(suggestions, key=lambda edit: edit.start)
