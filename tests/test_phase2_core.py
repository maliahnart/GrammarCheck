import sys
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experimental_core.engine import apply_edits
from experimental_core.lexicon import PhraseLexicon
from experimental_phase2.context import ContextIndex
from experimental_phase2.engine import Phase2Corrector, Phase2Policy
from vn_text import eval_form, tokenize


class ToyLM:
    uni = {w: 20 for w in "tôi đi học trời mưa vì nên nhưng đã hoàn thành em thi thì mai mãi trọng tài biên trong tai biến".split()}
    bi = {("<s>", "mãi"): 2}

    def logp(self, left, right):
        return -0.2 if right in {"học", "mãi", "thì"} else -4.0


class ToyModel:
    lm = ToyLM()
    vocab = {w: 20 for w in "tôi đi học trời mưa vì nên nhưng đã hoàn thành em thi thì mai mãi trọng tài biên trong tai biến".split()}
    threshold = 0.45

    def correct_batch(self, texts):
        return [t.replace("hoc", "học").replace("nhưng chuyến", "nên chuyến")
                 .replace("đã đã ", "đã ").replace("haha", "Haha").replace("OpenAI", "OpenAi")
                 .replace("Mai", "Mãi") for t in texts]

    def detect(self, text):
        return [(t, 0.9 if t in {"hoc", "bien"} else 0.0) for t in tokenize(text)]


@pytest.fixture
def engine(monkeypatch):
    import preservation
    model = ToyModel()
    monkeypatch.setattr(preservation, "load_lm", lambda: model.lm)
    monkeypatch.setattr(preservation, "load_vocab", lambda: set(model.vocab))
    return Phase2Corrector(model, PhraseLexicon(["học", "trọng tài", "biên", "tai biến", "trong"]))


def test_rule_correction_survives_sentence_ranking(engine):
    text = "Vì thời tiết xấu nhưng chuyến bay bị hoãn lại."
    result = engine.correct(text, policy=Phase2Policy(0, 0))
    assert result["output"] == text.replace("nhưng", "nên")


def test_approved_grammar_deletion_is_not_rejected_by_a_sentence_margin(engine):
    result = engine.correct("Tôi đã đã hoàn thành báo cáo.")
    assert result["output"] == "Tôi đã hoàn thành báo cáo."
    assert any(e["kind"] == "grammar" and e["accepted"] for e in result["edits"])


def test_expression_can_be_capitalized_at_start_but_user_lock_wins(engine):
    source = "haha, vui ghê :))"
    assert engine.correct(source)["output"] == "Haha, vui ghê :))"
    assert engine.correct(source, protected_terms=["haha"])["output"] == source


def test_hard_locks_apply_to_every_returned_hypothesis(engine):
    source = "OpenAI hoc. https://hoc.vn #DH123"
    result = engine.correct(source, protected_terms=["OpenAI"], policy=Phase2Policy(0, 0))
    assert result["output"] == source.replace("hoc.", "học.", 1)
    for candidate in result["candidates"]:
        assert "OpenAI" in candidate["text"] and "OpenAi" not in candidate["text"]
        assert "https://hoc.vn #DH123" in candidate["text"]
        assert apply_edits(source, candidate["edits"]) == candidate["text"]


def test_nfd_and_layout_remain_on_original_offsets(engine):
    source = unicodedata.normalize("NFD", "🙂 Tôi đi hoc.\r\n\r\n\tEm  đi hoc.  ")
    result = engine.correct(source, policy=Phase2Policy(0, 0))
    assert result["output"] == source.replace("hoc", "học")
    assert all(source[e["start"]:e["end"]] == e["source"] for e in result["edits"])


def test_realword_gate_is_separate_from_candidate_ranking(engine):
    result = engine.correct("Mai đi học.", policy=Phase2Policy(100, 0))
    assert result["output"] == "Mai đi học."
    assert not result["context_promoted"]


def test_no_neighbors_means_no_context_promotion(engine):
    result = engine.correct("trong tai bien", policy=Phase2Policy(0, 64))
    assert not result["context"]["available"]
    assert not result["context_promoted"]


def test_context_requires_loaded_index_and_has_bounded_input(engine):
    with pytest.raises(ValueError, match="no train-only"):
        engine.correct("Tôi đi học.", previous="Trời mưa.")
    with pytest.raises(ValueError, match="4,000"):
        engine.correct("Tôi đi học.", previous="x" * 4001)


@pytest.fixture
def context_index(tmp_path):
    corpus = tmp_path / "train_only.txt"
    corpus.write_text("\n".join([
        "Trọng tài điều khiển trận đấu bóng đá trên sân.",
        "Trọng tài biên làm việc tại sân bóng trong trận đấu.",
        "Tai biến cần cấp cứu và bác sĩ điều trị bệnh viện.",
        "Bác sĩ chữa tai biến cho bệnh nhân tại bệnh viện."]), encoding="utf-8")
    return ContextIndex.build(corpus, {"fit_sources": [str(corpus)], "label_files_used": []},
                              {"min_df": 1, "max_features": 1000, "neighbors": 2, "minimum_similarity": 0.05})


def test_context_swap_changes_ranking_without_keyword_case_rules(engine, context_index):
    engine.context_index = context_index
    choices = ["trọng tài biên", "trong tai biến"]
    sport = engine.rank_choices("trong tai bien", choices, previous="Trận đấu bóng đá trên sân.", weight=64)
    medicine = engine.rank_choices("trong tai bien", choices, previous="Bác sĩ cấp cứu tại bệnh viện.", weight=64)
    assert sport[0]["text"] == "trọng tài biên"
    assert medicine[0]["text"] == "trong tai biến"
    assert engine.rank_choices("trong tai bien", choices[::-1], previous="Trận đấu bóng đá trên sân.", weight=64) == sport


def test_empty_or_unknown_context_abstains(context_index):
    assert not context_index.profile().available
    assert not context_index.profile("qxzvub kkjjx", "").available
    assert context_index.metadata["label_files_used"] == []


def test_next_context_optional_and_normalized(context_index):
    text = "Bác sĩ cấp cứu tại bệnh viện."
    a = context_index.profile("", text)
    b = context_index.profile("", unicodedata.normalize("NFD", text))
    assert a.available and b.available
    assert a.score("tai biến") == pytest.approx(b.score("tai biến"))


def test_context_merge_does_not_replace_an_unrelated_ml_repair():
    source = "Toi ban."
    anchor = [{"start": 0, "end": 3, "source": "Toi", "replacement": "Tôi", "accepted": True}]
    proposed = [
        {"start": 0, "end": 3, "source": "Toi", "replacement": "Tối", "accepted": True},
        {"start": 4, "end": 7, "source": "ban", "replacement": "bán", "accepted": True}]
    output, edits = Phase2Corrector._merge_with_anchor(source, proposed, anchor, {(4, 7)})
    assert output == "Tôi bán."
    assert apply_edits(source, edits) == output


def test_context_queries_never_fit_the_vectorizer(context_index, monkeypatch):
    before = dict(context_index.vectorizer.vocabulary_)

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("Inference must not fit on user or evaluation context")

    monkeypatch.setattr(context_index.vectorizer, "fit", unexpected_fit)
    monkeypatch.setattr(context_index.vectorizer, "fit_transform", unexpected_fit)
    assert context_index.profile("Trận đấu bóng đá trên sân.").available
    assert context_index.vectorizer.vocabulary_ == before


def test_context_probes_are_split_by_ambiguity_group_and_document():
    import json
    rows = [[json.loads(line) for line in (ROOT / f"data/experimental/phase2/context_{split}.jsonl").read_text().splitlines()]
            for split in ("dev", "test")]
    assert not ({r["ambiguity_group"] for r in rows[0]} & {r["ambiguity_group"] for r in rows[1]})
    assert not ({r["document_id"] for r in rows[0]} & {r["document_id"] for r in rows[1]})
    assert not ({eval_form(r["input"]) for r in rows[0]} & {eval_form(r["input"]) for r in rows[1]})
    for split in rows:
        grouped = {}
        for row in split:
            assert row["target"] in row["choices"]
            grouped.setdefault(row["ambiguity_group"], []).append(row)
        assert all(len(pair) == 2 and pair[0]["input"] == pair[1]["input"]
                   and pair[0]["target"] != pair[1]["target"] for pair in grouped.values())
