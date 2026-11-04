"""Behavioral checks for the opt-in experiment; no model downloads or retraining."""
import sys
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experimental_core.engine import ExperimentalCorrector, Policy, apply_edits, protected_spans
from experimental_core.lexicon import PhraseLexicon
from vn_text import tokenize


class TinyLM:
    uni = {"học": 100, "luật": 100, "trọng": 50, "tài": 50, "tai": 50,
           "biến": 50, "mai": 50, "mãi": 50}
    bi = {("<s>", "mãi"): 10}

    def logp(self, left, right):
        return -0.2 if right in {"học", "luật", "mãi"} else -4.0


class TinyModel:
    threshold = 0.45
    lm = TinyLM()
    vocab = {word: 2 for word in "anh em ra xem đi mai mãi học luật trọng tài biên trong tai biến".split()}

    def correct_batch(self, texts):
        return [text.replace("hoc", "học").replace("Mai", "Mãi").replace("OpenAI", "OpenAi")
                for text in texts]

    def detect(self, text):
        return [(word, 0.9 if word in {"hoc", "luat"} else 0.0) for word in tokenize(text)]


@pytest.fixture
def engine(monkeypatch):
    import preservation
    model = TinyModel()
    monkeypatch.setattr(preservation, "load_lm", lambda: model.lm)
    monkeypatch.setattr(preservation, "load_vocab", lambda: set(model.vocab))
    lexicon = PhraseLexicon(["học", "luật", "trọng tài", "tai biến", "biên", "trong"])
    return ExperimentalCorrector(model, lexicon)


def test_phrase_index_combines_entries_and_has_no_closed_vocabulary_assumption():
    lexicon = PhraseLexicon(["trọng tài", "biên", "tai biến", "một cụm dài hơn bốn âm tiết"])
    assert ("trọng", "tài") in lexicon.variants(["trong", "tai"])
    assert ("tai", "biến") in lexicon.variants(["tai", "bien"])
    assert not lexicon.contains(["trọng", "tài", "biên"])
    assert lexicon.coverage(["trọng", "tài", "biên"]) == 1


def test_lattice_can_compose_a_phrase_that_is_not_a_dictionary_entry(engine):
    paths = engine._lattice("trong tai bien sai luat", [], True)
    assert any("trọng tài biên" in path for path in paths)


def test_low_accent_ratio_does_not_unlock_a_correct_sentence(engine):
    assert engine.route("Anh em ra xem.")["mode"] == "uncertain"
    assert engine.correct("Anh em ra xem.")["output"] == "Anh em ra xem."


def test_original_layout_and_decomposed_unicode_offsets(engine):
    text = unicodedata.normalize("NFD", "🙂 Em đi hoc.\r\n\r\n\tEm  đi hoc.  ")
    result = engine.correct(text, policy=Policy(0.0, 0.0))
    assert result["output"] == text.replace("hoc", "học")
    assert all(text[e["start"]:e["end"]] == e["source"] for e in result["edits"])
    assert result["keep_original"] == text


def test_hard_locks_block_even_a_model_casing_edit(engine):
    text = "OpenAI hoc. https://hoc.vn #DH123 test@hoc.vn 12.50"
    result = engine.correct(text, protected_terms=["OpenAI"], policy=Policy(0.0, 0.0))
    assert "OpenAI" in result["output"] and "OpenAi" not in result["output"]
    assert "https://hoc.vn #DH123 test@hoc.vn 12.50" in result["output"]


def test_expressive_and_user_protected_spans_are_preserved(engine):
    text = "hoc hihi :)) 🙂"
    result = engine.correct(text, protected_terms=["hoc"], policy=Policy(0.0, 0.0))
    assert result["output"] == text


def test_keep_original_candidate_and_top_three_have_explicit_score_type(engine):
    analysis = engine.analyze("Em đi hoc.")
    assert any(c["text"] == analysis["normalized"] for c in analysis["candidates"])
    result = engine.decide(analysis)
    assert len(result["candidates"]) <= 3
    assert result["score_type"] == "uncalibrated_ranking_score"
    keys = [" ".join(tokenize(c["text"])).casefold() for c in result["candidates"]]
    assert len(keys) == len(set(keys))


def test_strict_policy_can_abstain_on_a_high_ranked_real_word_replacement(engine):
    analysis = engine.analyze("Mai đi học.")
    result = engine.decide(analysis, Policy(realword_margin=100.0, sentence_margin=0.0))
    assert result["output"] == "Mai đi học."
    assert result["best_candidate"].startswith("Mãi")
    assert any(not e["accepted"] and e["realword"] for e in result["edits"])


def test_no_abbreviation_expansion_when_disabled(engine):
    engine.model.correct_batch = lambda texts: [t.replace("ko", "không") for t in texts]
    assert engine.correct("Em ko đi.", normalize_abbrev=False)["output"] == "Em ko đi."


def test_apply_edits_rejects_stale_or_overlapping_offsets():
    with pytest.raises(ValueError):
        apply_edits("abc", [{"start": 0, "end": 1, "source": "z", "replacement": "x", "accepted": True}])
    with pytest.raises(ValueError):
        apply_edits("abc", [
            {"start": 0, "end": 2, "source": "ab", "replacement": "x", "accepted": True},
            {"start": 1, "end": 3, "source": "bc", "replacement": "y", "accepted": True}])


def test_overlong_input_is_rejected_instead_of_silently_truncated(engine):
    with pytest.raises(ValueError):
        engine.correct("x" * 2001)


def test_phrase_search_does_not_cross_paragraph_boundaries(engine):
    assert all("trọng\ntài" not in value for value in engine._lattice("trong\ntai", [], True))


def test_policy_can_be_applied_to_a_serialized_analysis(engine):
    import json
    analysis = engine.analyze("Em đi hoc.")
    assert engine.decide(json.loads(json.dumps(analysis))) == engine.decide(analysis)


def test_case_only_original_remains_available_without_duplicate_display_candidates(engine):
    engine.model.correct_batch = lambda texts: [text.capitalize() for text in texts]
    analysis = engine.analyze("em")
    assert any(c["text"] == "em" for c in analysis["candidates"])
    result = engine.decide(analysis)
    assert result["keep_original"] == "em"
    assert len(result["candidates"]) == 1
