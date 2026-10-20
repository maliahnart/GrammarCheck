"""Mô hình 1 - ML Baseline.

Pipeline 3 bước:
  1. Phát hiện lỗi cấp token: TF-IDF n-gram ký tự + đặc trưng từ điển/ngữ cảnh -> Logistic Regression.
  2. Sinh ứng viên cho token bị nghi lỗi: từ điển âm tiết + biến thể dấu + edit distance 1
     + nhầm lẫn s/x, ch/tr, d/gi/r... + bảng viết tắt.
  3. Chọn ứng viên theo ngữ cảnh: bigram LM (absolute discounting) + chi phí kênh, beam search.
Cộng thêm các quy tắc: lặp hư từ, quan hệ từ thừa, viết hoa đầu câu, dấu câu cuối câu.
"""
from __future__ import annotations

import difflib
import math
import pickle
import re
from collections import Counter, defaultdict

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from vn_text import (ABBREVIATIONS, DUPLICABLE_FUNCTION_WORDS, REDUNDANT_AFTER, VALID_REDUPLICATIONS,
                     has_diacritic, is_protected, is_word, match_case, remove_diacritics, tokenize,
                     tokenize_with_spans)

LETTERS = "abcdefghijklmnopqrstuvwxyz"
INITIAL_SWAPS = [("tr", "ch"), ("ch", "tr"), ("gi", "d"), ("d", "gi"), ("r", "d"), ("d", "r"), ("r", "gi"),
                 ("gi", "r"), ("s", "x"), ("x", "s"), ("l", "n"), ("n", "l")]
FINAL_SWAPS = [("ng", "n"), ("n", "ng"), ("c", "t"), ("t", "c"), ("nh", "n"), ("n", "nh")]
QUESTION_END = {"không", "chưa", "nhỉ", "hả", "chứ", "sao", "gì", "nào", "đâu", "à"}
INFORMAL_MARKERS = {"em", "mình", "tớ", "cậu", "mày", "tao", "bạn", "ơi", "nha", "nhé", "nhỉ", "ạ", "á", "hả",
                    "luôn", "quá", "ghê", "lắm"}


# ---------------------------------------------------------------------------
class BigramLM:
    """Bigram LM với absolute discounting, nội suy với unigram (add-one)."""

    def __init__(self, d: float = 0.75, min_bigram: int = 2):
        self.d, self.min_bigram = d, min_bigram

    def fit(self, sentences):
        uni, bi = Counter(), Counter()
        for s in sentences:
            toks = ["<s>"] + [t.lower() for t in tokenize(s)] + ["</s>"]
            uni.update(toks)
            bi.update(zip(toks[:-1], toks[1:]))
        self.uni = dict(uni)
        self.total = sum(uni.values())
        self.V = len(uni) + 1
        self.bi = {k: v for k, v in bi.items() if v >= self.min_bigram}
        followers = Counter(k[0] for k in self.bi)
        self.n1plus = dict(followers)
        return self

    def p_uni(self, w):
        return (self.uni.get(w, 0) + 1) / (self.total + self.V)

    def logp(self, prev, w):
        cp = self.uni.get(prev, 0)
        pu = self.p_uni(w)
        if cp == 0:
            return math.log(pu)
        c = self.bi.get((prev, w), 0)
        lam = self.d * self.n1plus.get(prev, 0) / cp
        return math.log(max(c - self.d, 0) / cp + max(lam, 1e-3) * pu)


# ---------------------------------------------------------------------------
class MLCorrector:
    def __init__(self, top_k: int = 25, beam: int = 8):
        self.top_k, self.beam = top_k, beam
        self.threshold = 0.5
        self.w_channel = 1.5

    # ------------------------ từ điển & ứng viên -------------------------
    def build_lexicon(self, lm_sentences, min_count: int = 2):
        self.lm = BigramLM().fit(lm_sentences)
        self.vocab = {w: c for w, c in self.lm.uni.items() if c >= min_count and is_word(w)}
        idx = defaultdict(list)
        for w, c in self.vocab.items():
            idx[remove_diacritics(w)].append((w, c))
        self.index = {k: [w for w, _ in sorted(v, key=lambda x: -x[1])] for k, v in idx.items()}
        # Truecasing: từ hầu như luôn viết hoa khi đứng giữa câu (Việt, Nam, Hà, Nội...).
        cap, tot = Counter(), Counter()
        for s in lm_sentences:
            for t in tokenize(s)[1:]:
                if is_word(t):
                    tot[t.lower()] += 1
                    cap[t.lower()] += t[0].isupper()
        self.truecase = {w for w, n in tot.items() if n >= 20 and cap[w] / n >= 0.95}

    def in_dict(self, w: str) -> bool:
        return w.lower() in self.vocab

    def candidates(self, tok: str) -> list[tuple[list[str], float]]:
        """Trả về [(chuỗi token thay thế, chi phí kênh)], đã sắp theo chi phí rồi tần suất."""
        t = tok.lower()
        if not is_word(t):
            return []
        cands: dict[tuple, float] = {}

        def add(words, cost):
            key = tuple(words)
            if key != (t,) and (key not in cands or cost < cands[key]):
                cands[key] = cost

        if t in ABBREVIATIONS:
            add(ABBREVIATIONS[t].split(), 0.2)
        base = remove_diacritics(t)
        no_diac = not has_diacritic(t)
        for w in self.index.get(base, [])[: self.top_k]:
            add([w], 0.5 if no_diac else 1.0)
        # Edit distance 1 trên dạng không dấu.
        short_penalty = 1.5 if len(base) <= 2 else 0.0  # token rất ngắn: sửa 1 ký tự rất dễ sai
        for e in _edits1(base):
            for w in self.index.get(e, [])[:5]:
                add([w], 2.5 + short_penalty + (0 if no_diac or remove_diacritics(w) == w else 0.3))
        # Nhầm lẫn phụ âm đầu / cuối.
        for a, b in INITIAL_SWAPS:
            if t.startswith(a) and not t.startswith("đ"):
                for w in self.index.get(remove_diacritics(b + t[len(a):]), [])[:8]:
                    add([w], 1.8)
        for a, b in FINAL_SWAPS:
            if t.endswith(a):
                for w in self.index.get(remove_diacritics(t[: -len(a)] + b), [])[:8]:
                    add([w], 1.8)
        ranked = sorted(cands.items(), key=lambda kv: (kv[1], -self.vocab.get(kv[0][0], 0)))
        return [(list(k), v) for k, v in ranked[: self.top_k]]

    # ------------------------ đặc trưng phát hiện ------------------------
    def _numeric_features(self, toks_low: list[str], i: int) -> list[float]:
        t = toks_low[i]
        prev = toks_low[i - 1] if i > 0 else "<s>"
        nxt = toks_low[i + 1] if i + 1 < len(toks_low) else "</s>"
        s_t = self.lm.logp(prev, t) + self.lm.logp(t, nxt)
        best_gain = -10.0
        n_c = 0
        for words, cost in self.candidates(t)[:12]:
            n_c += 1
            s = self.lm.logp(prev, words[0]) + self.lm.logp(words[-1], nxt) - cost
            best_gain = max(best_gain, s - s_t)
        freq = self.vocab.get(t, 0)
        return [float(self.in_dict(t)), math.log1p(freq) / 10, float(t in ABBREVIATIONS),
                float(has_diacritic(t)), min(len(t), 10) / 10, max(min(best_gain, 15), -10) / 10,
                s_t / 20, n_c / 12, float(is_protected(t))]

    def _token_examples(self, text: str):
        toks = tokenize(text)
        low = [t.lower() for t in toks]
        idxs = [i for i, t in enumerate(toks) if is_word(t)]
        return toks, low, idxs

    def featurize(self, items):
        """items: list[(toks_low, i)] -> ma trận đặc trưng."""
        strs = [f"<{tl[i]}>" for tl, i in items]
        num = np.array([self._numeric_features(tl, i) for tl, i in items], dtype=np.float32)
        return hstack([self.tfidf.transform(strs), csr_matrix(num * 3.0)]).tocsr()

    # ------------------------------ huấn luyện ---------------------------
    def fit_detector(self, pairs, max_tokens: int = 400_000, rng=None):
        items, labels = [], []
        for inp, tgt in pairs:
            toks, low, idxs = self._token_examples(inp)
            lab = token_labels(low, [t.lower() for t in tokenize(tgt)])
            for i in idxs:
                items.append((low, i))
                labels.append(lab[i])
            if len(items) >= max_tokens:
                break
        self.tfidf = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=3, sublinear_tf=True)
        self.tfidf.fit([f"<{tl[i]}>" for tl, i in items])
        X = self.featurize(items)
        y = np.array(labels)
        self.clf = LogisticRegression(max_iter=2000, C=2.0, class_weight={0: 1.0, 1: 2.0})
        self.clf.fit(X, y)
        return {"n_tokens": len(y), "pos_rate": float(y.mean())}

    # ------------------------------ suy luận -----------------------------
    def detect(self, text: str) -> list[tuple[str, float]]:
        toks, low, idxs = self._token_examples(text)
        probs = [0.0] * len(toks)
        if idxs:
            p = self.clf.predict_proba(self.featurize([(low, i) for i in idxs]))[:, 1]
            for i, pi in zip(idxs, p):
                probs[i] = 0.0 if is_protected(toks[i]) else float(pi)
        return list(zip(toks, probs))

    def correct(self, text: str) -> str:
        spans = tokenize_with_spans(text)
        if not spans:
            return text
        toks = [t for t, _, _ in spans]
        low = [t.lower() for t in toks]
        probs = [p for _, p in self.detect(text)]
        options = []
        for i, t in enumerate(low):
            opts = [([t], 0.0)]
            if probs[i] >= self.threshold:
                opts += self.candidates(t)
            options.append(opts)
        best = self._beam(options)
        # Ghép lại văn bản, giữ nguyên khoảng trắng gốc và kiểu viết hoa.
        new_toks = []
        for (orig, _, _), words in zip(spans, best):
            new_toks.append(" ".join(words) if words == [orig.lower()] else match_case(orig, " ".join(words)))
        new_toks = [orig if nt.lower() == orig.lower() else nt for (orig, _, _), nt in zip(spans, new_toks)]
        # Truecasing cho token đã được sửa hoặc toàn chữ thường (tên riêng phổ biến).
        new_toks = [nt[0].upper() + nt[1:] if k > 0 and len(nt) > 1 and nt.lower() in self.truecase and nt.islower()
                    else nt
                    for k, nt in enumerate(new_toks)]
        out_toks, keep = apply_grammar_rules([t for t in new_toks])
        pieces = []
        for k, ((_, s, e), nt) in enumerate(zip(spans, out_toks)):
            if not keep[k]:
                continue
            ws = text[spans[k - 1][2]:s] if k > 0 else ""
            pieces.append((ws if pieces else "") + nt)
        return fix_casing_punct("".join(pieces))

    def correct_batch(self, texts: list[str], batch_size: int = 0) -> list[str]:
        return [self.correct(t) for t in texts]

    def _beam(self, options):
        beams = [(0.0, "<s>", [])]
        for opts in options:
            new = []
            for score, last, seq in beams:
                for words, cost in opts:
                    s = score - self.w_channel * cost
                    prev = last
                    for w in words:
                        s += self.lm.logp(prev, w)
                        prev = w
                    new.append((s, prev, seq + [words]))
            new.sort(key=lambda x: -x[0])
            beams = new[: self.beam]
        beams = [(s + self.lm.logp(last, "</s>"), last, seq) for s, last, seq in beams]
        return max(beams, key=lambda x: x[0])[2]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path) -> "MLCorrector":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
def _edits1(w: str) -> set[str]:
    splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits if b for c in LETTERS]
    inserts = [a + c + b for a, b in splits for c in LETTERS]
    return set(deletes + transposes + replaces + inserts)


def token_labels(src_low: list[str], tgt_low: list[str]) -> list[int]:
    """Nhãn lỗi cho từng token nguồn (1 = token này cần sửa/xoá) dựa trên căn chỉnh với đích."""
    lab = [0] * len(src_low)
    sm = difflib.SequenceMatcher(a=src_low, b=tgt_low, autojunk=False)
    for op, i1, i2, _, _ in sm.get_opcodes():
        if op in ("replace", "delete"):
            for i in range(i1, i2):
                lab[i] = 1
    return lab


def apply_grammar_rules(toks: list[str]) -> tuple[list[str], list[bool]]:
    """Quy tắc ngữ pháp đơn giản. Trả về (token sau sửa, cờ giữ token)."""
    toks = list(toks)
    keep = [True] * len(toks)
    low = [t.lower() for t in toks]
    for i in range(1, len(toks)):
        prev = next((j for j in range(i - 1, -1, -1) if keep[j]), None)
        if prev is None:
            continue
        if low[i] == low[prev] and low[i] in DUPLICABLE_FUNCTION_WORDS and low[i] not in VALID_REDUPLICATIONS:
            keep[i] = False  # "của của" -> "của"
        elif low[i] in REDUNDANT_AFTER.get(low[prev], ()):
            keep[i] = False  # "nhưng nên" -> "nhưng"
    # "Vì ... nhưng ..." -> "Vì ... nên ..."
    if "vì" in low and "nhưng" in low and "nên" not in low:
        vi, nh = low.index("vì"), low.index("nhưng")
        if vi < nh and keep[nh]:
            toks[nh] = match_case(toks[nh], "nên")
    return toks, keep


def fix_casing_punct(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    text = re.sub(r"(^|[.!?]\s+)([^\W\d_])", lambda m: m.group(1) + m.group(2).upper(), text)
    toks = tokenize(text)
    last = toks[-1] if toks else ""
    if is_word(last) and not is_protected(last):
        low = {t.lower() for t in toks}
        if last.lower() in QUESTION_END and ("có" in low or last.lower() in {"chưa", "nhỉ", "hả", "chứ"}):
            text += "?"
        elif len(toks) >= 6 and not (low & INFORMAL_MARKERS) and not any(t.lower() in ABBREVIATIONS for t in toks):
            text += "."
    return text
