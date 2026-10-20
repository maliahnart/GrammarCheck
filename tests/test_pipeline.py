"""Kiểm thử pipeline. Chạy: python -m pytest tests -q
Kiểm thử Transformer cần tải checkpoint (~0.9GB) -> chỉ chạy khi đặt RUN_TRANSFORMER=1."""
import json
import os
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate import compute_metrics  # noqa: E402
from generate_noise import make_noisy  # noqa: E402
from preservation import preserve  # noqa: E402
from vn_text import eval_form, norm_key, remove_diacritics, tokenize  # noqa: E402

ML_PATH = ROOT / "models/ml_baseline.pkl"
BILSTM_PATH = ROOT / "models/bilstm_attention.pt"


# ----------------------------- tiện ích văn bản ----------------------------
def test_tokenize_keeps_emoticons():
    assert tokenize("vui quá :)) haha :D") == ["vui", "quá", ":))", "haha", ":D"]
    assert tokenize("Hà Nội, 12:30.") == ["Hà", "Nội", ",", "12", ":", "30", "."]


def test_remove_diacritics():
    assert remove_diacritics("Đường phố Hà Nội") == "Duong pho Ha Noi"


# ----------------------------- dữ liệu ------------------------------------
def test_noise_changes_sentence_and_reports_types():
    rng = random.Random(1)
    s = "Đội tuyển Việt Nam đã giành chiến thắng thuyết phục."
    for _ in range(20):
        noisy, types = make_noisy(s, rng)
        assert noisy != s and types
        assert set(types) <= {"diacritic", "typo", "phonetic", "abbreviation", "grammar", "casing", "punctuation"}


@pytest.mark.skipif(not (ROOT / "data/test.jsonl").exists(), reason="chưa chạy prepare_data.py")
def test_no_leakage_between_train_and_test():
    def keys(name):
        return {norm_key(json.loads(line)["target"]) for line in open(ROOT / f"data/{name}.jsonl", encoding="utf-8")}
    train = keys("train")
    assert not (train & keys("test"))
    assert not (train & keys("val"))


# ----------------------------- chỉ số -------------------------------------
def test_metrics_identity_and_perfect():
    recs = [{"input": "toi di hoc", "target": "Tôi đi học.", "error_type": "spelling", "style": "formal"},
            {"input": "Trời đẹp.", "target": "Trời đẹp.", "error_type": "none", "style": "formal"}]
    perfect = compute_metrics(recs, [r["target"] for r in recs])
    assert perfect["exact_match"] == 1.0 and perfect["cer_after"] == 0 and perfect["wer_after"] == 0
    assert perfect["sent_precision"] == 1.0 and perfect["clean_kept_rate"] == 1.0
    ident = compute_metrics(recs, [r["input"] for r in recs])
    assert ident["exact_match"] == 0.5 and ident["sent_recall"] == 0.0
    assert ident["wer_after"] == ident["wer_before"]


# ----------------------------- bảo toàn văn phong -------------------------
def test_preserve_keeps_expressive_tokens():
    r = preserve("Bạn ăn cơm chưa? Mình đói quá rồi hihi :))", "Bạn ăn cơm chưa? Mình đói quá rồi khi")
    assert r["text"] == "Bạn ăn cơm chưa? Mình đói quá rồi hihi :))"
    assert r["suggestions"] and not any(e.accepted for e in r["edits"])


def test_preserve_allows_many_fixes_in_one_sentence():
    src = "Kinh tế viet nam dang phat trien nhanh trong nhung nam gan day."
    out = "Kinh tế Việt Nam đang phát triển nhanh trong những năm gần đây."
    r = preserve(src, out)
    assert eval_form(r["text"]) == eval_form(out)
    assert all(e.accepted for e in r["edits"])


def test_preserve_blocks_rewrite():
    src = "hôm nay e đi học có vui không"
    out = "Hôm nay em đến trường có cảm thấy vui vẻ không"
    r = preserve(src, out)
    assert "đến trường" not in r["text"] and "cảm thấy" not in r["text"]
    assert r["text"].startswith("Hôm nay em đi học")


def test_preserve_abbreviation_toggle():
    src, out = "ko sao đâu, mik hiểu mà", "không sao đâu, mình hiểu mà"
    assert preserve(src, out, normalize_abbrev=True)["text"] == out
    assert preserve(src, out, normalize_abbrev=False)["text"] == src
    assert preserve(src, out, normalize_abbrev=False, strict=False)["text"] == src


def test_preserve_grammar_deletion_allowed_but_not_random_deletion():
    r = preserve("Tuy trời mưa nhưng nên tôi vẫn đi học.", "Tuy trời mưa nhưng tôi vẫn đi học.")
    assert r["text"] == "Tuy trời mưa nhưng tôi vẫn đi học."
    r = preserve("Tôi rất thích đọc sách.", "Tôi thích đọc sách.")
    assert r["text"] == "Tôi rất thích đọc sách."


# ----------------------------- mô hình ------------------------------------
@pytest.mark.skipif(not ML_PATH.exists(), reason="chưa train ML baseline")
def test_ml_baseline_corrects():
    from baseline_ml import MLCorrector
    m = MLCorrector.load(ML_PATH)
    assert m.correct("Tuy trời mưa nhưng nên tôi vẫn đi học.") == "Tuy trời mưa nhưng tôi vẫn đi học."
    assert "đang" in m.correct("Kinh tế dang phát triển nhanh.")
    clean = "Hội nghị sẽ được tổ chức tại Hà Nội vào tuần tới."
    assert m.correct(clean) == clean


@pytest.mark.skipif(not BILSTM_PATH.exists(), reason="chưa train BiLSTM")
def test_bilstm_runs():
    from bilstm_attention import BiLSTMCorrector
    m = BiLSTMCorrector.load(BILSTM_PATH, device="cpu")
    outs = m.correct_batch(["Hôm nay tôi đi hoc.", "Trời đẹp quá."])
    assert len(outs) == 2 and all(isinstance(o, str) and o for o in outs)


@pytest.mark.skipif(os.environ.get("RUN_TRANSFORMER") != "1", reason="đặt RUN_TRANSFORMER=1 để chạy")
def test_transformer_and_inference_api():
    from inference import correct_text
    r = correct_text("Kinh tế viet nam dang phat trien.", model="transformer")
    assert "phát triển" in r["output"]
    r = correct_text("Mình đói quá rồi hihi ^^", model="transformer")
    assert "hihi" in r["output"] and "^^" in r["output"]
