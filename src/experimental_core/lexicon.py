"""Read-only word/phrase index. Dictionary membership is evidence, not correctness."""
from __future__ import annotations

import unicodedata
from collections import defaultdict
from pathlib import Path

from vn_text import remove_diacritics


class PhraseLexicon:
    def __init__(self, entries, max_syllables=4):
        self.max_syllables = max_syllables
        self.entries = frozenset(
            tuple(unicodedata.normalize("NFC", entry).lower().split())
            for entry in entries if entry.strip()
        )
        index = defaultdict(list)
        for words in sorted(self.entries):
            if 1 <= len(words) <= max_syllables and all(word.isalpha() for word in words):
                index[tuple(remove_diacritics(word) for word in words)].append(words)
        self.index = dict(index)

    @classmethod
    def load(cls, path: str | Path, max_syllables=4):
        with open(path, encoding="utf-8") as handle:
            return cls(handle, max_syllables)

    def variants(self, words):
        return self.index.get(tuple(remove_diacritics(word.lower()) for word in words), ())

    def contains(self, words):
        return tuple(word.lower() for word in words) in self.entries

    def coverage(self, tokens):
        """Maximum non-overlapping multi-syllable coverage, without counting punctuation."""
        dp = [0] * (len(tokens) + 1)
        for i in range(len(tokens)):
            dp[i + 1] = max(dp[i + 1], dp[i])
            for size in range(2, self.max_syllables + 1):
                end = i + size
                if end <= len(tokens) and self.contains(tokens[i:end]):
                    dp[end] = max(dp[end], dp[i] + size - 1)
        return dp[-1]
