"""Stable ML corrections plus train-only lexical context reranking (opt-in).

No-context automatic output starts from the ML proposal, not a winning dictionary
rewrite. Context may promote an accent-restoration alternative under explicit gates.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from experimental_core.engine import (ExperimentalCorrector, LOCK_PATTERN, apply_edits,
                                      intersects, normalize_with_offsets, protected_spans)
from experimental_core.lexicon import PhraseLexicon
from vn_text import eval_form, is_protected, remove_diacritics, tokenize, tokenize_with_spans

from .context import ContextIndex

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Phase2Policy:
    realword_margin: float = 2.0
    context_weight: float = 16.0
    promotion_margin: float = 0.01


def hard_spans(source, terms):
    spans = [(m.start(), m.end()) for m in LOCK_PATTERN.finditer(source)]
    for term in terms:
        if not term.strip() or len(term) > 100:
            raise ValueError("Mỗi thuật ngữ bảo vệ cần có từ 1 đến 100 ký tự.")
        spans.extend((m.start(), m.end()) for m in re.finditer(re.escape(term), source, re.IGNORECASE))
    return spans


def accents_only(first, second):
    a, b = tokenize(first), tokenize(second)
    return len(a) == len(b) and all(remove_diacritics(x).casefold() == remove_diacritics(y).casefold()
                                    for x, y in zip(a, b))


class ContextLM:
    """Search-time lexical bonus only. The trained detector/model stay unchanged."""
    def __init__(self, base, profile, weight):
        self.base, self.profile, self.weight = base, profile, weight
        self.uni, self.bi = base.uni, base.bi

    @lru_cache(maxsize=2048)
    def support(self, token):
        return self.profile.score(token)

    def logp(self, left, right):
        return self.base.logp(left, right) + self.weight * self.support(right)


class SearchAdapter:
    def __init__(self, model, profile, weight):
        self.original = model
        self.lm = ContextLM(model.lm, profile, weight)

    def __getattr__(self, name):
        return getattr(self.original, name)


class ContextSearch(ExperimentalCorrector):
    def route(self, source):
        result = super().route(source)
        if result["diacritic_ratio"] < 0.2:
            result = {**result, "mode": "suspected_undiacritized"}
        return result


class Phase2Corrector:
    def __init__(self, model, lexicon, config=None, context_index=None):
        self.model, self.lexicon = model, lexicon
        self.config = config or {}
        self.base = ExperimentalCorrector(model, lexicon, self.config.get("search", {}))
        self.context_index = context_index

    @classmethod
    def from_project(cls, use_context=True):
        from inference import get_model
        config = json.loads((ROOT / "configs/experimental_phase2.json").read_text())
        index = None
        if use_context:
            cache = ROOT / "outputs/experimental/phase2/context_index.joblib"
            if not cache.exists():
                raise FileNotFoundError("Build the phase 2 train-only context index first: scripts/evaluate_phase2.py --stage build")
            index = ContextIndex.load(cache)
            corpus_hash = hashlib.sha256((ROOT / config["corpus"]).read_bytes()).hexdigest()
            if index.metadata["corpus_sha256"] != corpus_hash:
                raise ValueError("Context index does not match the current train corpus.")
        return cls(get_model("ml"), PhraseLexicon.load(ROOT / config["wordlist"]), config, index)

    def _annotate(self, source, candidate, hard, normalize_abbrev):
        soft = [(a, b) for token, a, b in tokenize_with_spans(source) if is_protected(token)]
        first_word = next((a for token, a, _ in tokenize_with_spans(source) if token.isalpha()), -1)
        edits = self.base._annotate(source, candidate, hard + soft, normalize_abbrev)
        for edit in edits:
            initial_case = (edit["kind"] == "casing" and edit["start"] == first_word
                            and edit["source"].isalpha()
                            and edit["source"].casefold() == edit["replacement"].casefold())
            edit["locked"] = (intersects(edit["start"], edit["end"], hard)
                              or (intersects(edit["start"], edit["end"], soft) and not initial_case))
            if edit["kind"] == "deletion" and edit["base_accepted"]:
                # Legacy preservation accepts word deletion only for its explicit grammar rules.
                edit["legacy_kind"] = "deletion"
                edit["kind"] = "grammar"
                edit["explanation"] = "Xóa từ thừa theo quy tắc ngữ pháp hiện có"
        return edits

    @staticmethod
    def _render(analysis, annotated, policy, contextual_spans=frozenset()):
        boundaries, source = analysis["boundaries"], analysis["input"]
        get_boundary = lambda n: boundaries.get(n, boundaries.get(str(n)))
        edits = []
        for proposal in annotated:
            edit = dict(proposal)
            allowed = edit["base_accepted"] and not edit["locked"]
            reason = edit["reason"]
            can_restore = ((edit["start"], edit["end"]) in contextual_spans and edit["kind"] == "diacritic"
                           and accents_only(edit["source"], edit["replacement"]))
            if edit["locked"]:
                allowed, reason = False, "Giữ nguyên vùng được bảo vệ."
            elif can_restore:
                allowed, reason = True, "Phương án phục hồi dấu được hỗ trợ bởi ngữ cảnh lân cận; cần đọc lại."
            elif edit["realword"] and edit["local_lm_gain"] < policy.realword_margin:
                allowed, reason = False, "Chưa đủ bằng chứng để tự đổi một từ hợp lệ."
            start, end = get_boundary(edit["start"]), get_boundary(edit["end"])
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
        return apply_edits(source, edits), edits

    @staticmethod
    def _merge_with_anchor(source, candidate_edits, anchor_edits, supported):
        """Preserve ML repairs unless context supports changing that same source span."""
        accepted_anchors = [dict(e) for e in anchor_edits if e["accepted"]]
        merged = []
        for proposal in candidate_edits:
            key = (proposal["start"], proposal["end"])
            conflicts = [a for a in accepted_anchors if (
                a["start"] < proposal["end"] and a["end"] > proposal["start"])
                or (a["start"] == a["end"] == proposal["start"] == proposal["end"])]
            if conflicts and key not in supported:
                # The anchor will be inserted below with its unchanged original offsets.
                continue
            edit = dict(proposal)
            if not conflicts and key not in supported:
                edit["accepted"] = False
                edit["reason"] = "Ngữ cảnh chưa hỗ trợ riêng phép đổi này; giữ phương án ML."
            merged.append(edit)
        for anchor in accepted_anchors:
            if any(e["accepted"] and ((e["start"] < anchor["end"] and e["end"] > anchor["start"])
                   or (e["start"] == e["end"] == anchor["start"] == anchor["end"])) for e in merged):
                continue
            merged.append(anchor)
        # Non-applied suggestions may overlap; accepted edits may not.
        return apply_edits(source, merged), merged

    def analyze(self, text, *, previous="", following="", normalize_abbrev=True, protected_terms=(), use_context=True):
        started = time.perf_counter()
        if not isinstance(previous, str) or not isinstance(following, str):
            raise ValueError("Neighboring contexts must be strings.")
        if len(previous) > 4000 or len(following) > 4000:
            raise ValueError("Each neighboring context is limited to 4,000 characters.")
        normalized_terms = [unicodedata.normalize("NFC", term) for term in protected_terms]
        initial = self.base.analyze(text, normalize_abbrev=normalize_abbrev, protected_terms=normalized_terms)
        source = initial["normalized"]
        hard = hard_spans(source, normalized_terms)
        anchor_edits = self._annotate(source, initial["raw_output"], hard, normalize_abbrev)
        pool = {c["text"]: list(c["origins"]) for c in initial["candidates"]}
        profile = None
        if use_context and (previous.strip() or following.strip()):
            if self.context_index is None:
                raise ValueError("Context was supplied but no train-only context index was loaded.")
            profile = self.context_index.profile(previous, following)
            if profile.available:
                # Search adds lexical evidence; it never fits on the current question or its answer.
                proxy = SearchAdapter(self.model, profile, max(self.config.get("context_weight_grid", [16.0])))
                search = ContextSearch(proxy, self.lexicon, self.config.get("search", {}))
                extra = search.analyze(text, normalize_abbrev=normalize_abbrev, protected_terms=normalized_terms)
                for candidate in extra["candidates"]:
                    pool.setdefault(candidate["text"], []).append("context_search")
        pool.setdefault(source, ["keep_original"])
        pool.setdefault(initial["raw_output"], ["model_raw"])
        candidates = []
        for value, origins in pool.items():
            candidates.append({"text": value, "origins": origins,
                               "local_score": self.base._score(source, value),
                               "context_score": profile.score(value) if profile else 0.0,
                               "edits": self._annotate(source, value, hard, normalize_abbrev)})
        return {"input": text, "normalized": source, "boundaries": initial["boundaries"],
                "raw_output": initial["raw_output"], "legacy_preserved": initial["legacy_preserved"],
                "anchor_edits": anchor_edits, "candidates": candidates,
                "_context_profile": profile,
                "context": profile.describe() if profile else {"available": False, "reason": "No neighboring context used."},
                "route": initial["route"], "seconds": time.perf_counter() - started}

    def rank_choices(self, text, choices, *, previous="", following="", weight=16.0):
        """Diagnostic candidate ranking only. Never called with gold choices in inference."""
        profile = self.context_index.profile(previous, following) if self.context_index else None
        return sorted([{"text": value, "local_score": self.base._score(text, value),
                        "context_score": profile.score(value) if profile else 0.0,
                        "score": self.base._score(text, value) + weight * (profile.score(value) if profile else 0.0)}
                       for value in choices], key=lambda c: (-c["score"], c["text"]))

    def decide(self, analysis, policy=Phase2Policy()):
        anchor, anchor_edits = self._render(analysis, analysis["anchor_edits"], policy)
        context_available = analysis["context"]["available"] and policy.context_weight > 0
        ranked = analysis["candidates"]
        # Only low-accent restoration can override the stable ML output in this phase.
        restore = analysis["route"]["diacritic_ratio"] < 0.2
        profile = analysis.get("_context_profile")
        anchor_context = profile.score(anchor) if profile else 0.0
        rendered = []
        for candidate in ranked:
            context_gain = candidate["context_score"] - anchor_context
            permit = (context_available and restore and context_gain >= policy.promotion_margin
                      and accents_only(analysis["normalized"], candidate["text"]))
            supported, original_supported = set(), set()
            if permit:
                for proposal in candidate["edits"]:
                    if proposal["kind"] != "diacritic" or proposal["locked"]:
                        continue
                    competing = next((e["replacement"] for e in analysis["anchor_edits"]
                                      if e["start"] == proposal["start"] and e["end"] == proposal["end"]
                                      and e["base_accepted"] and not e["locked"]), proposal["source"])
                    if profile.score(proposal["replacement"]) - profile.score(competing) >= policy.promotion_margin:
                        supported.add((proposal["start"], proposal["end"]))
                        boundary = lambda n: analysis["boundaries"].get(n, analysis["boundaries"].get(str(n)))
                        original_supported.add((boundary(proposal["start"]), boundary(proposal["end"])))
            value, edits = self._render(analysis, candidate["edits"], policy, contextual_spans=supported)
            if permit:
                value, edits = self._merge_with_anchor(analysis["input"], edits, anchor_edits, original_supported)
            safe_context = profile.score(value) if profile else 0.0
            safe_local = self.base._score(analysis["normalized"], value)
            rendered.append({**candidate, "safe_text": value, "safe_edits": edits,
                             "context_score": safe_context, "local_score": safe_local,
                             "context_gain": safe_context - anchor_context,
                             "permit_context": permit and safe_context - anchor_context >= policy.promotion_margin,
                             "score": safe_local + policy.context_weight * safe_context})
        rendered.sort(key=lambda c: (-c["score"], c["safe_text"]))
        best = rendered[0]
        promoted = bool(best["permit_context"] and best["safe_text"] != anchor)
        output, edits = (best["safe_text"], best["safe_edits"]) if promoted else (anchor, anchor_edits)
        hypotheses, seen = [], set()
        for candidate in rendered:
            key = eval_form(candidate["safe_text"]).casefold()
            if key in seen:
                continue
            seen.add(key)
            hypotheses.append({"id": f"phase2-hypothesis-{len(hypotheses)}", "text": candidate["safe_text"],
                               "score": candidate["score"], "local_score": candidate["local_score"],
                               "context_score": candidate["context_score"], "origins": candidate["origins"],
                               "edits": candidate["safe_edits"], "apply_as_atomic_hypothesis": True})
            if len(hypotheses) == 3:
                break
        return {"input": analysis["input"], "output": output, "raw_output": analysis["raw_output"],
                "best_candidate": best["safe_text"], "keep_original": analysis["input"],
                "edits": edits, "suggestions": [e for e in edits if not e["accepted"]],
                "candidates": hypotheses, "context": analysis["context"], "context_promoted": promoted,
                "policy": asdict(policy), "pipeline": "experimental_phase2", "model": "ml",
                "offset_unit": "unicode_code_point", "score_type": "uncalibrated_lexical_ranking",
                "seconds": analysis["seconds"]}

    def correct(self, text, *, policy=Phase2Policy(), **options):
        return self.decide(self.analyze(text, **options), policy)
