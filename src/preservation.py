"""Hạn chế sửa quá mức (overcorrection) và bảo toàn văn phong gốc.

Ý tưởng: căn chỉnh câu gốc và câu do mô hình sinh ra ở cấp token, tách thành từng phép sửa
(edit) nhỏ, rồi xét RIÊNG từng phép sửa bằng các quy tắc kiểm tra. Phép sửa hợp lệ được giữ,
phép sửa đáng ngờ bị hoàn tác và chuyển thành "gợi ý". Vì xét riêng từng phép, câu có nhiều lỗi
vẫn được sửa đầy đủ; chỉ những thay đổi không cần thiết bị loại.

Quy tắc chính:
  1. Không sửa/xoá token biểu cảm: hihi, haha, :)), ^^, emoji, URL...
  2. Khi tắt "chuẩn hoá viết tắt": giữ nguyên ko, mik, dc...
  3. Từ gốc là từ hợp lệ trong từ điển -> chỉ chấp nhận thay đổi gần (dấu, nhầm phụ âm, 1-2 ký tự).
  4. Từ gốc không có trong từ điển (lỗi gõ) -> chấp nhận nếu phương án sửa đủ gần chữ gốc.
  5. Không chấp nhận chèn thêm từ mới hoặc xoá từ, trừ dấu câu và các lỗi ngữ pháp có quy tắc
     (lặp hư từ "của của", quan hệ từ thừa "nhưng nên").
  6. Không tự hạ chữ hoa của câu gốc (tên riêng, viết hoa có chủ ý).
  7. Kiểm tra độ tin cậy: khi một TỪ HỢP LỆ bị đổi thành một từ hợp lệ khác (vd "Mai" -> "Mãi"),
     chỉ chấp nhận nếu bigram LM (học từ corpus sạch) cho điểm ngữ cảnh của từ mới cao hơn từ cũ.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from vn_text import (ABBREVIATIONS, DUPLICABLE_FUNCTION_WORDS, REDUNDANT_AFTER, VALID_REDUPLICATIONS,
                     edit_distance, is_protected, is_word, remove_diacritics, tokenize_with_spans)

ROOT = Path(__file__).resolve().parents[1]
_VOCAB: set[str] | None = None
_LM = None

_INITIAL_GROUPS = [{"s", "x"}, {"ch", "tr"}, {"d", "gi", "r"}, {"l", "n"}]
_FINAL_GROUPS = [{"c", "t"}, {"n", "ng", "nh"}]

EXPLAIN = {
    "casing": "Viết hoa",
    "punctuation": "Dấu câu",
    "abbreviation": "Viết tắt -> dạng đầy đủ",
    "diacritic": "Thiếu hoặc sai dấu",
    "phonetic": "Nhầm phụ âm (s/x, ch/tr, d/gi/r, l/n, c/t, n/ng) / sai từ theo ngữ cảnh",
    "grammar": "Lỗi ngữ pháp (lặp từ, quan hệ từ thừa/sai)",
    "spacing": "Tách/ghép từ",
    "spelling": "Lỗi gõ phím / chính tả",
    "insertion": "Thêm từ",
    "deletion": "Xoá từ",
    "rewrite": "Thay từ khác (viết lại)",
}


def load_vocab(path: Path | None = None) -> set[str]:
    """Từ điển âm tiết hợp lệ (tạo khi train ML baseline: models/vocab.txt)."""
    global _VOCAB
    if _VOCAB is None:
        p = path or ROOT / "models/vocab.txt"
        _VOCAB = {line.split("\t")[0] for line in open(p, encoding="utf-8")} if p.exists() else set()
    return _VOCAB


def load_lm():
    """Bigram LM lấy từ ML baseline đã train (models/ml_baseline.pkl). Không có thì trả None."""
    global _LM
    if _LM is None:
        p = ROOT / "models/ml_baseline.pkl"
        if p.exists():
            from baseline_ml import MLCorrector
            _LM = MLCorrector.load(p).lm
        else:
            _LM = False
    return _LM or None


@dataclass
class Edit:
    src: str
    out: str
    kind: str
    accepted: bool
    reason: str = ""

    @property
    def explanation(self) -> str:
        return EXPLAIN.get(self.kind, self.kind)


def _is_punct(tok: str) -> bool:
    return not any(ch.isalnum() for ch in tok) and not is_protected(tok)


def _phonetic_pair(a: str, b: str) -> bool:
    a, b = remove_diacritics(a.lower()), remove_diacritics(b.lower())
    for grp in _INITIAL_GROUPS:
        for x in grp:
            for y in grp:
                if x != y and a.startswith(x) and b.startswith(y) and a[len(x):] == b[len(y):]:
                    return True
    for grp in _FINAL_GROUPS:
        for x in grp:
            for y in grp:
                if x != y and a.endswith(x) and b.endswith(y) and a[: -len(x)] == b[: -len(y)]:
                    return True
    return False


def classify(src: list[str], out: list[str]) -> str:
    s, o = " ".join(src), " ".join(out)
    if not src:
        return "punctuation" if all(_is_punct(t) for t in out) else "insertion"
    if not out:
        return "punctuation" if all(_is_punct(t) for t in src) else "deletion"
    if all(_is_punct(t) for t in src + out):
        return "punctuation"
    if s.lower() == o.lower():
        return "casing"
    if len(src) == 1 and ABBREVIATIONS.get(s.lower()) == o.lower():
        return "abbreviation"
    if len(src) == len(out) and remove_diacritics(s.lower()) == remove_diacritics(o.lower()):
        return "diacritic"
    if len(src) == 1 and len(out) == 1 and _phonetic_pair(s, o):
        return "phonetic"
    if s.replace(" ", "").lower() == o.replace(" ", "").lower():
        return "spacing"
    if {s.lower(), o.lower()} == {"nhưng", "nên"}:
        return "grammar"
    return "spelling"


def _lm_prefers(lm, left: str, right: str, old: str, new: str, margin: float) -> bool:
    """LM ủng hộ từ mới? Nếu từ mới có bigram quan sát được với ngữ cảnh: chỉ cần điểm tăng.
    Nếu không có bằng chứng bigram (điểm chỉ dựa vào backoff, kém tin cậy): cần tăng > margin."""
    old, new = old.lower(), new.lower()

    def score(w):
        return lm.logp(left, w) + lm.logp(w, right)
    evidence = (left, new) in lm.bi or (new, right) in lm.bi
    return score(new) - score(old) > (0.0 if evidence else margin)


def _judge(src: list[str], out: list[str], nbr_src: tuple[str, str], kind: str, vocab: set[str],
           normalize_abbrev: bool, ctx: tuple[str, str] = ("<s>", "</s>"), lm=None,
           lm_margin: float = 0.0) -> tuple[bool, str]:
    """Quyết định chấp nhận một phép sửa. Trả về (chấp nhận?, lý do nếu từ chối)."""
    if any(is_protected(t) for t in src) and kind != "casing":
        return False, "Giữ nguyên từ biểu cảm / emoticon"
    if kind == "casing":
        if any(a != b and a[:1].isupper() and not b[:1].isupper() for a, b in zip(src, out)):
            return False, "Không hạ chữ hoa của câu gốc"
        return True, ""
    if kind == "punctuation":
        return True, ""
    low_src = [t.lower() for t in src]
    if any(t in ABBREVIATIONS for t in low_src) and not normalize_abbrev:
        return False, "Đã tắt chuẩn hoá viết tắt"
    if kind == "abbreviation":
        return True, ""
    if kind == "deletion":
        prev_src, next_src = nbr_src
        w = low_src[0] if len(src) == 1 else None
        if w and (w in DUPLICABLE_FUNCTION_WORDS and w not in VALID_REDUPLICATIONS and w in (prev_src, next_src)
                  or w in REDUNDANT_AFTER.get(prev_src, ())):
            return True, ""
        return False, "Không tự ý xoá từ"
    if kind == "insertion":
        return False, "Không tự ý thêm từ mới"
    real_word = (len(src) == 1 and len(out) == 1 and bool(vocab) and low_src[0] in vocab
                 and out[0].lower() in vocab and is_word(src[0]))
    if real_word and lm is not None and kind != "grammar":
        if not _lm_prefers(lm, ctx[0], ctx[1], src[0], out[0], lm_margin):
            return False, "Từ gốc hợp lệ, ngữ cảnh không ủng hộ phương án sửa - chỉ gợi ý"
    if kind in ("diacritic", "phonetic", "grammar", "spacing"):
        return True, ""
    # Còn lại: lỗi gõ hoặc thay từ.
    s_join = remove_diacritics("".join(low_src))
    o_join = remove_diacritics("".join(t.lower() for t in out))
    dist = edit_distance(s_join, o_join)
    all_valid = all((t in vocab) or not is_word(t) for t in low_src) if vocab else False
    if len(src) == 1 and len(out) == 1:
        if all_valid:
            ok = dist <= 1 or (dist <= 2 and len(s_join) >= 6)
        else:
            ok = dist <= max(2, len(s_join) // 2)
    else:
        # nhiều token: chỉ chấp nhận khi nguồn có từ sai chính tả và tổng thay đổi nhỏ
        ok = (not all_valid) and dist <= max(2, len(s_join) // 3)
    return (True, "") if ok else (False, "Thay đổi lớn so với từ gốc - chỉ đưa ra gợi ý")


def _align(st: list[str], ot: list[str]) -> list[tuple[int, int, int, int]]:
    """Căn chỉnh token theo khoá không dấu + chữ thường (để "me" khớp "mẹ"), trả về các đơn vị
    (i1, i2, j1, j2). Đoạn thay thế cùng độ dài được tách 1-1; dấu câu ở biên được tách riêng."""
    def key(t):
        return remove_diacritics(t.lower())
    sm = difflib.SequenceMatcher(a=[key(t) for t in st], b=[key(t) for t in ot], autojunk=False)
    units = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal" or (op == "replace" and i2 - i1 == j2 - j1):
            units += [(i1 + k, i1 + k + 1, j1 + k, j1 + k + 1) for k in range(i2 - i1)]
            continue
        tail = []
        if any(is_protected(t) for t in st[i1:i2]):  # không tách quanh emoticon (":D" -> "gớm :")
            units.append((i1, i2, j1, j2))
            continue
        # Tách dấu câu thêm/bớt ở cuối và đầu đoạn để không gộp vào phép sửa từ.
        while j2 > j1 and i2 - i1 != j2 - j1 and _is_punct(ot[j2 - 1]):
            tail.insert(0, (i2, i2, j2 - 1, j2))
            j2 -= 1
        while i2 > i1 and i2 - i1 != j2 - j1 and _is_punct(st[i2 - 1]):
            tail.insert(0, (i2 - 1, i2, j2, j2))
            i2 -= 1
        if i2 - i1 == j2 - j1:
            units += [(i1 + k, i1 + k + 1, j1 + k, j1 + k + 1) for k in range(i2 - i1)]
        elif i2 > i1 or j2 > j1:
            units += _split_abbrev(st, ot, i1, i2, j1, j2) or [(i1, i2, j1, j2)]
        units += tail
    return units


def _split_abbrev(st, ot, i1, i2, j1, j2):
    """Tách đoạn thay thế không cân ("bh mik" -> "Bây giờ mình") thành từng phép sửa 1 token nguồn
    -> k token đích, dựa vào bảng viết tắt. Trả về None nếu không tách được trọn vẹn."""
    units, j = [], j1
    for i in range(i1, i2):
        exp = ABBREVIATIONS.get(st[i].lower(), "").split()
        n = len(exp) if exp and [t.lower() for t in ot[j:j + len(exp)]] == exp else 1
        if j + n > j2:
            return None
        units.append((i, i + 1, j, j + n))
        j += n
    return units if j == j2 else None


def preserve(src_text: str, out_text: str, normalize_abbrev: bool = True, vocab: set[str] | None = None,
             use_lm: bool = True, lm_margin: float = 0.0, strict: bool = True) -> dict:
    """So sánh câu gốc và câu mô hình sửa; giữ các phép sửa hợp lệ, hoàn tác phép sửa đáng ngờ.

    strict=False: tắt mọi quy tắc bảo toàn (giữ nguyên output mô hình), chỉ còn tuỳ chọn viết tắt.

    Trả về {"text": câu kết quả, "edits": [Edit...], "suggestions": [Edit bị từ chối]}."""
    vocab = load_vocab() if vocab is None else vocab
    lm = load_lm() if use_lm else None
    S = tokenize_with_spans(src_text)
    O = tokenize_with_spans(out_text)
    st, ot = [t for t, _, _ in S], [t for t, _, _ in O]
    units = _align(st, ot)

    pieces: list[tuple[str, str]] = []  # (khoảng trắng trước, token)
    edits: list[Edit] = []

    def emit(side, a, b):
        toks, text = (S, src_text) if side == "src" else (O, out_text)
        for k in range(a, b):
            ws = text[toks[k - 1][2]:toks[k][1]] if k > 0 else " "
            pieces.append((ws, toks[k][0]))

    for i1, i2, j1, j2 in units:
        src, out = st[i1:i2], ot[j1:j2]
        if src == out:
            emit("out", j1, j2)
            continue
        kind = classify(src, out)
        nbr_src = (st[i1 - 1].lower() if i1 > 0 else "", st[i2].lower() if i2 < len(st) else "")
        # Ngữ cảnh trái/phải lấy từ câu đã sửa (các từ lân cận đã được sửa).
        ctx = (ot[j1 - 1].lower() if j1 > 0 else "<s>", ot[j2].lower() if j2 < len(ot) else "</s>")
        if strict:
            ok, reason = _judge(src, out, nbr_src, kind, vocab, normalize_abbrev, ctx, lm, lm_margin)
        elif not normalize_abbrev and any(t.lower() in ABBREVIATIONS for t in src):
            ok, reason = False, "Đã tắt chuẩn hoá viết tắt"
        else:
            ok, reason = True, ""
        edits.append(Edit(" ".join(src), " ".join(out), kind, ok, reason))
        emit("out", j1, j2) if ok else emit("src", i1, i2)

    text = re.sub(r"[ \t]{2,}", " ","".join(ws + t for ws, t in pieces)).strip()
    return {"text": text, "edits": edits, "suggestions": [e for e in edits if not e.accepted]}


def edits_as_dicts(edits: list[Edit]) -> list[dict]:
    return [dict(asdict(e), explanation=e.explanation) for e in edits]
