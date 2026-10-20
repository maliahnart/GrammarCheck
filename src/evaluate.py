"""Các chỉ số đánh giá. Mọi chuỗi được đưa về `eval_form` (NFC, tách dấu câu, 1 khoảng trắng)
trước khi so sánh để cả 3 mô hình được chấm cùng một cách tiền xử lý.

- Phát hiện lỗi CẤP CÂU: câu có lỗi (input != target) vs mô hình có sửa (pred != input).
- Phát hiện lỗi CẤP TOKEN: token nguồn có lỗi (căn chỉnh input->target khác) vs token nguồn bị
  mô hình thay đổi (căn chỉnh input->pred khác). Lỗi dạng "thiếu token" (ví dụ thiếu dấu chấm cuối
  câu) không có token nguồn nên không tính ở cấp token.
- Sửa lỗi: CER, WER (trước/sau), Exact Match, và F0.5 cấp phép sửa (edit) kiểu ERRANT đơn giản.
- Bảo toàn: % câu đúng được giữ nguyên, số câu đúng bị sửa thừa, số token biểu cảm bị mất/đổi.
"""
from __future__ import annotations

import difflib
from collections import Counter, defaultdict

from vn_text import edit_distance, eval_form, is_protected, tokenize


def _prf(tp, fp, fn, beta=1.0):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    b2 = beta * beta
    f = (1 + b2) * p * r / (b2 * p + r) if p + r else 0.0
    return p, r, f


def _edits(src: list[str], hyp: list[str]) -> set[tuple]:
    sm = difflib.SequenceMatcher(a=src, b=hyp, autojunk=False)
    return {(i1, i2, " ".join(hyp[j1:j2])) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"}


def _changed_src(src: list[str], hyp: list[str]) -> set[int]:
    sm = difflib.SequenceMatcher(a=src, b=hyp, autojunk=False)
    out = set()
    for op, i1, i2, _, _ in sm.get_opcodes():
        if op in ("replace", "delete"):
            out.update(range(i1, i2))
    return out


def protected_lost(src: str, pred: str) -> int:
    """Số token biểu cảm (hihi, :)), emoji...) có trong input nhưng biến mất/đổi trong output."""
    a = Counter(t.lower() for t in tokenize(src) if is_protected(t) and len(t) > 1 and t.lower() not in
                {"nha", "nhé", "ạ", "à", "á", "ơi", "đó", "đấy", "thôi", "luôn", "ok"})
    b = Counter(t.lower() for t in tokenize(pred))
    return sum(max(0, c - b.get(t, 0)) for t, c in a.items())


def compute_metrics(records: list[dict], preds: list[str]) -> dict:
    assert len(records) == len(preds)
    s_tp = s_fp = s_fn = s_tn = 0
    t_tp = t_fp = t_fn = 0
    e_tp = e_fp = e_fn = 0
    char_err_before = char_err_after = n_chars = 0
    word_err_before = word_err_after = n_words = 0
    exact = 0
    clean_total = clean_kept = 0
    lost_expr = 0
    by_type = defaultdict(lambda: [0, 0])
    by_style = defaultdict(lambda: [0, 0])
    for r, p in zip(records, preds):
        src, tgt, hyp = eval_form(r["input"]), eval_form(r["target"]), eval_form(p)
        st, tt, ht = src.split(), tgt.split(), hyp.split()
        has_err, changed = src != tgt, hyp != src
        # cấp câu
        if has_err and changed:
            s_tp += 1
        elif not has_err and changed:
            s_fp += 1
        elif has_err and not changed:
            s_fn += 1
        else:
            s_tn += 1
        # cấp token
        gold_idx, pred_idx = _changed_src(st, tt), _changed_src(st, ht)
        t_tp += len(gold_idx & pred_idx)
        t_fp += len(pred_idx - gold_idx)
        t_fn += len(gold_idx - pred_idx)
        # cấp phép sửa
        ge, pe = _edits(st, tt), _edits(st, ht)
        e_tp += len(ge & pe)
        e_fp += len(pe - ge)
        e_fn += len(ge - pe)
        # CER / WER
        char_err_before += edit_distance(src, tgt)
        char_err_after += edit_distance(hyp, tgt)
        n_chars += max(1, len(tgt))
        word_err_before += edit_distance(st, tt)
        word_err_after += edit_distance(ht, tt)
        n_words += max(1, len(tt))
        em = hyp == tgt
        exact += em
        by_type[r.get("error_type", "?")][0] += em
        by_type[r.get("error_type", "?")][1] += 1
        by_style[r.get("style", "?")][0] += em
        by_style[r.get("style", "?")][1] += 1
        if not has_err:
            clean_total += 1
            clean_kept += not changed
        lost_expr += protected_lost(r["input"], p)

    sp, sr, sf = _prf(s_tp, s_fp, s_fn)
    tp_, tr_, tf_ = _prf(t_tp, t_fp, t_fn)
    ep, er, ef05 = _prf(e_tp, e_fp, e_fn, beta=0.5)
    n = len(records)
    return {
        "n": n,
        "sent_precision": sp, "sent_recall": sr, "sent_f1": sf,
        "tok_precision": tp_, "tok_recall": tr_, "tok_f1": tf_,
        "edit_precision": ep, "edit_recall": er, "edit_f05": ef05,
        "cer_before": char_err_before / n_chars, "cer_after": char_err_after / n_chars,
        "wer_before": word_err_before / n_words, "wer_after": word_err_after / n_words,
        "exact_match": exact / n,
        "exact_match_input": sum(eval_form(r["input"]) == eval_form(r["target"]) for r in records) / n,
        "clean_sentences": clean_total,
        "clean_kept_rate": clean_kept / clean_total if clean_total else None,
        "overcorrected_clean": clean_total - clean_kept,
        "expressive_tokens_lost": lost_expr,
        "exact_by_error_type": {k: v[0] / v[1] for k, v in sorted(by_type.items())},
        "count_by_error_type": {k: v[1] for k, v in sorted(by_type.items())},
        "exact_by_style": {k: v[0] / v[1] for k, v in sorted(by_style.items())},
    }
