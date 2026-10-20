"""Chuẩn bị dữ liệu: lọc câu sạch, chia tập không rò rỉ, sinh lỗi và ghi JSONL.

Đầu ra (thư mục data/):
  train.jsonl, val.jsonl, test.jsonl  - cặp (input lỗi, target sạch) giả lập
  test_vsec.jsonl                     - lỗi do người gõ thật (VSEC), chỉ dùng để test
  lm_corpus.txt                       - câu sạch phía train dùng cho từ điển / n-gram LM
  data_stats.json                     - thống kê
(test_real.jsonl là bộ câu tự soạn thủ công, không do script này tạo.)
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from generate_noise import make_noisy
from vn_text import has_diacritic, norm_key, normalize_spaces, tokenize

ROOT = Path(__file__).resolve().parents[1]

FAMILY = {"diacritic": "spelling", "typo": "spelling", "phonetic": "spelling",
          "abbreviation": "abbreviation", "grammar": "grammar", "casing": "casing_punct",
          "punctuation": "casing_punct"}


def is_good_sentence(s: str, cfg: dict) -> bool:
    words = s.split()
    if not (cfg["min_words"] <= len(words) <= cfg["max_words"]) or len(s) > cfg["max_chars"]:
        return False
    if re.search(r"[=|{}\[\]<>*#_@\\~^]|https?:|\.\.\.", s) or s[0] in "-–•(\"'":
        return False
    if not s[0].isupper() or s[-1] not in ".!?":
        return False
    letters = [w for w in words if re.search(r"[^\W\d_]", w)]
    viet = [w for w in letters if has_diacritic(w)]
    # Loại câu chủ yếu là tiếng nước ngoài / danh sách tên riêng.
    if len(viet) < 0.4 * len(letters):
        return False
    caps = sum(1 for w in letters[1:] if w[0].isupper())
    return caps <= 0.4 * max(1, len(letters) - 1)


def load_clean(path: Path, cfg: dict) -> list[str]:
    out, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = normalize_spaces(json.loads(line)["target"])
            k = norm_key(s)
            if k in seen or not is_good_sentence(s, cfg):
                continue
            seen.add(k)
            out.append(s)
    return out


def shingles(s: str, n: int = 5) -> set[int]:
    toks = [t.lower() for t in tokenize(s)]
    return {hash(" ".join(toks[i:i + n])) for i in range(max(1, len(toks) - n + 1))}


# ---------------------- câu hội thoại sinh từ mẫu -------------------------
SUBJ = ["em", "anh", "chị", "mình", "tớ", "bạn", "cậu", "tụi mình", "bọn em", "mẹ", "bố", "con"]
ACTS = ["đi học", "ăn cơm", "xem phim", "làm bài tập", "đi chơi", "về nhà", "ngủ sớm", "đọc sách",
        "đi làm", "nấu ăn", "đi siêu thị", "học tiếng Anh", "chơi bóng đá", "đi du lịch", "gọi điện",
        "uống cà phê", "đi bơi", "tập thể dục", "dọn nhà", "đi khám bệnh", "học bài", "đi mua sắm"]
TIMES = ["hôm nay", "tối nay", "ngày mai", "hôm qua", "cuối tuần này", "chiều nay", "sáng mai",
         "tuần sau", "lúc nãy", "bây giờ"]
INTERJ = ["hihi", "haha", "hehe", ":))", ":D", "huhu", "^^", "hihi :))", "kkk"]
FEEL = ["vui quá", "mệt quá", "buồn ghê", "thích lắm", "chán thật", "đói bụng rồi", "hồi hộp quá"]
PATTERNS = [
    "{T} {S} {A} có vui không?",
    "{T} {S} {A} không?",
    "{S} {A} {T} rồi, {F}.",
    "{T} {S} không {A} được, {F}.",
    "{S} ơi, {T} mình {A} nhé.",
    "{S} đã {A} chưa?",
    "Cảm ơn {S} nhiều nha.",
    "{T} {S} được {A} nên {F}.",
    "Sao {S} không {A} với mình?",
    "{S} muốn {A} lắm nhưng không có thời gian.",
]


def build_informal(n: int, rng: random.Random) -> list[str]:
    """Sinh n câu gốc hội thoại khác nhau (chưa gắn từ cảm thán)."""
    out = set()
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        p = rng.choice(PATTERNS)
        s = p.format(S=rng.choice(SUBJ), A=rng.choice(ACTS), T=rng.choice(TIMES), F=rng.choice(FEEL))
        out.add(s[0].upper() + s[1:])
    res = sorted(out)
    rng.shuffle(res)
    return res


def decorate(bases: list[str], rng: random.Random) -> list[str]:
    """Gắn từ cảm thán / emoticon sau khi đã chia tập, để biến thể không vượt ranh giới tập."""
    return [b + " " + rng.choice(INTERJ) if rng.random() < 0.45 else b for b in bases]


# ---------------------------------------------------------------------------
def make_records(sentences: list[tuple[str, str]], rng: random.Random, clean_ratio: float,
                 split: str) -> list[dict]:
    recs = []
    for i, (s, style) in enumerate(sentences):
        if rng.random() < clean_ratio:
            noisy, types = s, []
        else:
            noisy, types = make_noisy(s, rng, style)
        fams = sorted({FAMILY[t] for t in types})
        etype = "none" if not fams else (fams[0] if len(fams) == 1 else "mixed")
        recs.append({"id": f"{split}-{i}", "input": noisy, "target": s, "error_type": etype,
                     "error_types": types, "style": style, "source": "synthetic"})
    return recs


def prepare_vsec(path: Path, n: int, clean_ratio: float, rng: random.Random) -> list[dict]:
    df = pd.read_parquet(path)
    idx = list(range(len(df)))
    rng.shuffle(idx)
    n_clean = int(n * clean_ratio)
    recs = []
    for j, i in enumerate(idx[:n]):
        row = df.iloc[i]
        tgt = normalize_spaces(row["corrected_text"])
        if j < n_clean:  # câu đã sửa đúng -> kiểm tra sửa thừa
            inp, etype = tgt, "none"
        else:
            inp, etype = normalize_spaces(row["text"]), "spelling"
        recs.append({"id": f"vsec-{i}", "input": inp, "target": tgt, "error_type": etype,
                     "error_types": [] if etype == "none" else ["human"], "style": "formal",
                     "source": "vsec"})
    rng.shuffle(recs)
    return recs


def write_jsonl(path: Path, recs: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/config.yaml"))
    args = ap.parse_args()
    cfg_all = yaml.safe_load(open(args.config, encoding="utf-8"))
    cfg = cfg_all["data"]
    rng = random.Random(cfg_all["seed"])
    out = ROOT / cfg["out_dir"]

    print("Đang đọc câu sạch phía train ...")
    train_pool = load_clean(ROOT / cfg["clean_train_file"], cfg)
    print("  câu hợp lệ:", len(train_pool))
    heldout = load_clean(ROOT / cfg["clean_heldout_file"], cfg)
    print("  câu held-out hợp lệ:", len(heldout))

    rng.shuffle(train_pool)
    train_keys = {norm_key(s) for s in train_pool}
    pair_sents = train_pool[: cfg["n_train_sentences"]]
    lm_sents = train_pool[: cfg["n_lm_sentences"]]

    # Lọc held-out: không trùng khoá chuẩn hoá và không chia sẻ >=50% 5-gram với câu train.
    train_sh = set()
    for s in pair_sents:
        train_sh |= shingles(s)
    filtered = []
    for s in heldout:
        if norm_key(s) in train_keys:
            continue
        sh = shingles(s)
        if len(sh & train_sh) >= 0.5 * len(sh):
            continue
        filtered.append(s)
    print("  held-out sau khi lọc gần trùng:", len(filtered))
    rng.shuffle(filtered)
    val_s = filtered[: cfg["n_val"]]
    test_s = filtered[cfg["n_val"]: cfg["n_val"] + cfg["n_test"]]

    informal = build_informal(cfg["n_informal_templates"], rng)
    n_inf = len(informal)
    inf_train, inf_val, inf_test = (decorate(informal[: int(0.8 * n_inf)], rng),
                                    decorate(informal[int(0.8 * n_inf): int(0.9 * n_inf)], rng),
                                    decorate(informal[int(0.9 * n_inf):], rng))

    splits = {
        "train": [(s, "formal") for s in pair_sents] + [(s, "informal") for s in inf_train],
        "val": [(s, "formal") for s in val_s] + [(s, "informal") for s in inf_val],
        "test": [(s, "formal") for s in test_s] + [(s, "informal") for s in inf_test],
    }
    stats = {}
    for name, sents in splits.items():
        rng.shuffle(sents)
        recs = make_records(sents, rng, cfg["clean_ratio"], name)
        write_jsonl(out / f"{name}.jsonl", recs)
        stats[name] = {"n": len(recs), "error_type": dict(Counter(r["error_type"] for r in recs)),
                       "style": dict(Counter(r["style"] for r in recs))}
        print(name, stats[name])

    # Kiểm tra chống rò rỉ.
    keys = {n: {norm_key(r["target"]) for r in json_iter(out / f"{n}.jsonl")} for n in splits}
    stats["leak_train_test"] = len(keys["train"] & keys["test"])
    stats["leak_train_val"] = len(keys["train"] & keys["val"])
    print("Rò rỉ train∩test:", stats["leak_train_test"], "train∩val:", stats["leak_train_val"])

    with open(out / "lm_corpus.txt", "w", encoding="utf-8") as f:
        for s in lm_sents + inf_train:
            f.write(s + "\n")
    stats["lm_sentences"] = len(lm_sents) + len(inf_train)

    vsec = prepare_vsec(ROOT / cfg["vsec_file"], cfg["n_vsec_test"], cfg["vsec_clean_ratio"], rng)
    write_jsonl(out / "test_vsec.jsonl", vsec)
    stats["test_vsec"] = {"n": len(vsec), "error_type": dict(Counter(r["error_type"] for r in vsec))}
    print("test_vsec", stats["test_vsec"])
    json.dump(stats, open(out / "data_stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def json_iter(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


if __name__ == "__main__":
    main()
