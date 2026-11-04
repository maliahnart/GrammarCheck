"""Phase 1: optional ML + phrase lattice, without modifying the legacy corrector.

Scores are ranking features, NOT calibrated probabilities. Whole-sentence hypotheses
remain atomic; their individual edits must not be mixed with another hypothesis.
"""
from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from preservation import edits_as_dicts, preserve
from vn_text import (edit_distance, has_diacritic, is_protected, is_word,
                     match_case, remove_diacritics, tokenize, tokenize_with_spans)

from .lexicon import PhraseLexicon

ROOT = Path(__file__).resolve().parents[2]
LOCK_PATTERN = re.compile(
    r"https?://\S+|www\.\S+|[\w.+-]+@[\w.-]+\.[\w-]+|#[\w-]+|\b\d[\d.,:/%+-]*\b"
)


@dataclass(frozen=True)
class Policy:
    realword_margin: float = 2.0
    sentence_margin: float = 0.05


def normalize_with_offsets(source):
    normalized = unicodedata.normalize("NFC", source)
    boundaries = {0: 0}
    for end in range(1, len(source) + 1):
        boundaries[len(unicodedata.normalize("NFC", source[:end]))] = end
    return normalized, boundaries


def protected_spans(text, terms=()):
    spans = [(match.start(), match.end()) for match in LOCK_PATTERN.finditer(text)]
    spans.extend((start, end) for token, start, end in tokenize_with_spans(text) if is_protected(token))
    for term in terms:
        if not term.strip() or len(term) > 100:
            raise ValueError("Mỗi thuật ngữ bảo vệ cần có từ 1 đến 100 ký tự.")
        spans.extend((m.start(), m.end()) for m in re.finditer(re.escape(term), text, re.IGNORECASE))
    return spans


def intersects(start, end, locks):
    return any((start < right and end > left) or (start == end and left < start < right)
               for left, right in locks)


def apply_edits(source, edits):
    """Apply accepted edits to immutable original offsets; preserve untouched bytes/chars."""
    pieces, cursor = [], 0
    for edit in sorted(edits, key=lambda e: (e["start"], e["end"])):
        if not edit["accepted"]:
            continue
        start, end = edit["start"], edit["end"]
        if start < cursor or source[start:end] != edit["source"]:
            raise ValueError("Overlapping edits or stale original offsets.")
        pieces.extend((source[cursor:start], edit["replacement"]))
        cursor = end
    return "".join(pieces) + source[cursor:]


class ExperimentalCorrector:
    def __init__(self, model, lexicon, config=None):
        self.model, self.lexicon = model, lexicon
        self.config = config or {}
        self.width = int(self.config.get("beam_width", 12))
        self.max_candidates = int(self.config.get("max_candidates", 12))
        self.edge_limit = int(self.config.get("max_variants_per_edge", 8))
        self.channel_weight = float(self.config.get("channel_weight", 1.5))
        self.phrase_weight = float(self.config.get("phrase_weight", 0.25))
        if min(self.width, self.max_candidates, self.edge_limit) < 1:
            raise ValueError("Search limits must be positive.")

    @classmethod
    def from_project(cls, config_path=None):
        from inference import get_model
        path = Path(config_path) if config_path else ROOT / "configs/experimental_phase1.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        if config["model"] != "ml":
            raise ValueError("Phase 1 implements the ML adapter only.")
        lexicon = PhraseLexicon.load(ROOT / config["wordlist"], config["max_phrase_syllables"])
        return cls(get_model("ml"), lexicon, config)

    def route(self, source):
        words = [t for t in tokenize(source) if t.isalpha() and not is_protected(t)]
        ratio = sum(has_diacritic(t) for t in words) / max(1, len(words))
        # Low accent ratio alone never unlocks an entire sentence.
        unknown_restorable = any(t.lower() not in self.model.vocab and self.lexicon.variants([t])
                                 for t in words)
        mode = "normal"
        if len(words) >= 3 and ratio < 0.2:
            mode = "suspected_undiacritized" if unknown_restorable else "uncertain"
        return {"mode": mode, "diacritic_ratio": ratio,
                "unknown_restorable": bool(unknown_restorable)}

    def _lattice(self, seed, locks, normalize_abbrev):
        spans = tokenize_with_spans(seed)
        if not spans:
            return [seed]
        tokens = [t for t, _, _ in spans]
        probabilities = [p for _, p in self.model.detect(seed)]
        restore = self.route(seed)["mode"] == "suspected_undiacritized"
        suspect = [restore or p >= self.model.threshold for p in probabilities]
        # (incremental LM score, rendered prefix, last emitted token)
        beams = defaultdict(list)
        beams[0] = [(0.0, seed[:spans[0][1]], "<s>")]
        for i in range(len(tokens)):
            unique = {}
            for state in beams.pop(i, []):
                if state[1] not in unique or state[0] > unique[state[1]][0]:
                    unique[state[1]] = state
            states = sorted(unique.values(), key=lambda s: (-s[0], s[1]))[:self.width]
            if not states:
                continue
            edges = [(i + 1, tokens[i], (tokens[i].lower(),), 0.0, 0.0)]
            for size in range(1, self.lexicon.max_syllables + 1):
                end = i + size
                if end > len(tokens):
                    break
                part = tokens[i:end]
                start_char, end_char = spans[i][1], spans[end - 1][2]
                original = seed[start_char:end_char]
                if (not all(t.isalpha() for t in part) or "\n" in original or "\r" in original
                        or intersects(start_char, end_char, locks)):
                    break
                variants = self.lexicon.variants(part)
                ranked = sorted(variants, key=lambda words: (
                    -sum(math.log1p(self.model.lm.uni.get(w, 0)) for w in words), words))
                for words in ranked[:self.edge_limit]:
                    if not any(suspect[i:end]) and tuple(t.lower() for t in part) != words:
                        continue
                    if not normalize_abbrev:
                        from vn_text import ABBREVIATIONS
                        if any(t.lower() in ABBREVIATIONS for t in part):
                            continue
                    replacements = [match_case(old, new) for old, new in zip(part, words)]
                    rendered = replacements[0]
                    for k in range(1, size):
                        rendered += seed[spans[i + k - 1][2]:spans[i + k][1]] + replacements[k]
                    changed = sum(a.lower() != b for a, b in zip(part, words))
                    edges.append((end, rendered, words, 0.5 * changed, float(size - 1)))
            separator = seed[spans[i - 1][2]:spans[i][1]] if i else ""
            for score, prefix, last in states:
                for end, rendered, words, channel, phrase in edges:
                    value, previous = score, last
                    for word in words:
                        value += self.model.lm.logp(previous, word)
                        previous = word
                    value += self.phrase_weight * phrase - self.channel_weight * channel
                    beams[end].append((value, prefix + separator + rendered, previous))
            # Keep memory bounded even when many edges jump to the same node.
            for end in range(i + 1, min(len(tokens), i + self.lexicon.max_syllables) + 1):
                if len(beams[end]) > self.width * self.edge_limit:
                    beams[end] = sorted(beams[end], key=lambda s: (-s[0], s[1]))[:self.width * 2]
        endings = sorted(beams[len(tokens)], key=lambda s: (
            -(s[0] + self.model.lm.logp(s[2], "</s>")), s[1]))
        return [prefix + seed[spans[-1][2]:] for _, prefix, _ in endings[:self.width]]

    def _score(self, source, candidate, use_lexicon=True):
        tokens = [t.lower() for t in tokenize(candidate)]
        last, value = "<s>", 0.0
        for token in tokens:
            value += self.model.lm.logp(last, token)
            last = token
        value += self.model.lm.logp(last, "</s>")
        # Accent/case-free distance measures typing change; restoring accents still costs 0.5.
        a, b = [t.lower() for t in tokenize(source)], tokens
        change = edit_distance([remove_diacritics(t) for t in a], [remove_diacritics(t) for t in b])
        if len(a) == len(b):
            change += 0.5 * sum(x != y and remove_diacritics(x) == remove_diacritics(y)
                                for x, y in zip(a, b))
        value -= self.channel_weight * change
        if use_lexicon:
            value += self.phrase_weight * self.lexicon.coverage(tokens)
        return value / max(1, len(tokens))

    def _annotate(self, source, candidate, locks, normalize_abbrev):
        judged = preserve(source, candidate, normalize_abbrev=normalize_abbrev)
        spans = tokenize_with_spans(source)
        edits = []
        for raw in edits_as_dicts(judged["edits"]):
            start, end = raw["start"], raw["end"]
            old, new = tokenize(source[start:end]), tokenize(raw["replacement"])
            realword = (len(old) == len(new) == 1 and old[0].lower() != new[0].lower()
                        and is_word(old[0]) and old[0].lower() in self.model.vocab
                        and new[0].lower() in self.model.vocab
                        and raw["kind"] != "grammar")
            left = next((t.lower() for t, _, stop in reversed(spans) if stop <= start), "<s>")
            right = next((t.lower() for t, begin, _ in spans if begin >= end), "</s>")
            gain = 0.0
            if realword:
                gain = (self.model.lm.logp(left, new[0].lower()) + self.model.lm.logp(new[0].lower(), right)
                        - self.model.lm.logp(left, old[0].lower()) - self.model.lm.logp(old[0].lower(), right))
            raw.update(source=source[start:end], base_accepted=raw["accepted"],
                       realword=realword, local_lm_gain=gain,
                       locked=intersects(start, end, locks))
            edits.append(raw)
        return edits

    def analyze(self, text, *, normalize_abbrev=True, protected_terms=(), use_lexicon=True):
        if len(text) > 2000 or len(tokenize(text)) > 256:
            raise ValueError("Phase 1 accepts at most 2,000 characters and 256 tokens; split longer documents explicitly.")
        if not text.strip():
            raise ValueError("Vui lòng nhập văn bản có nội dung.")
        started = time.perf_counter()
        source, boundaries = normalize_with_offsets(text)
        terms = [unicodedata.normalize("NFC", term) for term in protected_terms]
        locks = protected_spans(source, terms)
        raw = self.model.correct_batch([source])[0]
        baseline_edits = self._annotate(source, raw, locks, normalize_abbrev)
        for edit in baseline_edits:
            edit["accepted"] = edit["base_accepted"] and not edit["locked"]
        baseline = apply_edits(source, baseline_edits)
        pool = [(baseline, "model_preserved"), (source, "keep_original"), (raw, "model_raw")]
        if use_lexicon:
            pool.extend((value, "source_lattice") for value in self._lattice(source, locks, normalize_abbrev))
            # A second hypothesis seed, never an irreversible first-pass correction.
            if baseline != source:
                baseline_locks = protected_spans(baseline, terms)
                pool.extend((value, "model_lattice") for value in self._lattice(baseline, baseline_locks, normalize_abbrev))
        unique = {}
        for value, origin in pool:
            unique.setdefault(value, {"text": value, "origins": [], "score": self._score(source, value, use_lexicon)})
            unique[value]["origins"].append(origin)
        ranked, seen = [], set()
        for candidate in sorted(unique.values(), key=lambda item: -item["score"]):
            key = " ".join(tokenize(candidate["text"])).casefold()
            if key not in seen:
                ranked.append(candidate)
                seen.add(key)
        candidates = ranked[:self.max_candidates]
        if not any(c["text"] == source for c in candidates):
            candidates.append(unique[source])
        for candidate in candidates:
            candidate["edits"] = self._annotate(source, candidate["text"], locks, normalize_abbrev)
        return {"input": text, "normalized": source, "boundaries": boundaries,
                "raw_output": raw, "legacy_preserved": preserve(source, raw, normalize_abbrev=normalize_abbrev)["text"],
                "route": self.route(source), "candidates": candidates, "original_score": unique[source]["score"],
                "seconds": time.perf_counter() - started, "use_lexicon": use_lexicon}

    @staticmethod
    def decide(analysis, policy=Policy()):
        best = analysis["candidates"][0]
        edits, blocked = [], []
        sentence_gain = best["score"] - analysis["original_score"]
        boundaries = analysis["boundaries"]
        # JSON cache keys are strings; in-memory keys are integers.
        boundary = lambda n: boundaries.get(n, boundaries.get(str(n)))
        source = analysis["input"]
        for raw in best["edits"]:
            edit = dict(raw)
            allowed = edit["base_accepted"] and not edit["locked"]
            reason = edit["reason"]
            if edit["locked"]:
                allowed, reason = False, "Giữ nguyên vùng được bảo vệ."
            elif edit["realword"] and edit["local_lm_gain"] < policy.realword_margin:
                allowed, reason = False, "Chưa đủ bằng chứng ngữ cảnh cho việc đổi từ hợp lệ."
            elif edit["kind"] not in {"casing", "punctuation", "grammar", "abbreviation"} and sentence_gain < policy.sentence_margin:
                allowed, reason = False, "Chênh lệch xếp hạng với bản gốc còn thấp."
            start, end = boundary(edit["start"]), boundary(edit["end"])
            replacement = edit["replacement"]
            if not replacement and end > start:
                while end < len(source) and source[end] in " \t":
                    end += 1
            if start == end and replacement:
                if start and source[start - 1].isalnum() and replacement[0].isalnum():
                    replacement = " " + replacement
                if end < len(source) and source[end].isalnum() and replacement[-1].isalnum():
                    replacement += " "
            edit.update(start=start, end=end, source=source[start:end], replacement=replacement,
                        accepted=allowed, reason=reason)
            edits.append(edit)
            if not allowed:
                blocked.append(edit)
        output = apply_edits(source, edits)
        hypotheses, seen = [], set()
        for candidate in analysis["candidates"]:
            key = " ".join(tokenize(candidate["text"])).casefold()
            if key in seen:
                continue
            seen.add(key)
            hypotheses.append({"id": f"hypothesis-{len(hypotheses)}", "text": candidate["text"],
                               "score": candidate["score"], "origins": candidate["origins"]})
            if len(hypotheses) == 3:
                break
        return {"input": source, "output": output, "best_candidate": best["text"],
                "raw_output": analysis["raw_output"], "candidates": hypotheses,
                "keep_original": source, "edits": edits, "suggestions": blocked,
                "route": analysis["route"], "policy": asdict(policy),
                "model": "ml", "pipeline": "experimental_phase1",
                "offset_unit": "unicode_code_point", "score_type": "uncalibrated_ranking_score",
                "seconds": analysis["seconds"]}

    def correct(self, text, *, policy=Policy(), **options):
        return self.decide(self.analyze(text, **options), policy)
