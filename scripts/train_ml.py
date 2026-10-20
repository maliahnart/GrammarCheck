"""Huấn luyện ML Baseline: xây từ điển + bigram LM, train Logistic Regression phát hiện lỗi,
tinh chỉnh ngưỡng phát hiện và trọng số chi phí kênh trên tập Val (theo WER sau sửa)."""
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baseline_ml import MLCorrector  # noqa: E402
from evaluate import compute_metrics  # noqa: E402


def read_jsonl(p):
    return [json.loads(line) for line in open(p, encoding="utf-8")]


def main():
    cfg = yaml.safe_load(open(ROOT / "configs/config.yaml", encoding="utf-8"))
    mcfg = cfg["ml"]
    t0 = time.time()
    lm_sents = [line.strip() for line in open(ROOT / "data/lm_corpus.txt", encoding="utf-8")]
    model = MLCorrector(top_k=mcfg["candidate_top_k"], beam=mcfg["beam_size"])
    model.build_lexicon(lm_sents)
    print(f"Từ điển: {len(model.vocab)} âm tiết, bigram: {len(model.lm.bi)} ({time.time() - t0:.0f}s)")

    train = read_jsonl(ROOT / "data/train.jsonl")
    info = model.fit_detector([(r["input"], r["target"]) for r in train], max_tokens=mcfg["max_train_tokens"])
    print("Detector:", info, f"({time.time() - t0:.0f}s)")

    val = read_jsonl(ROOT / "data/val.jsonl")[:400]
    best = None
    log = []
    for thr in (0.45, 0.6, 0.75):
        for w in (1.2, 2.0, 3.0):
            model.threshold, model.w_channel = thr, w
            m = compute_metrics(val, [model.correct(r["input"]) for r in val])
            log.append({"threshold": thr, "w_channel": w, "wer_after": m["wer_after"],
                        "exact_match": m["exact_match"], "clean_kept_rate": m["clean_kept_rate"]})
            print(log[-1])
            if best is None or m["wer_after"] < best[0]:
                best = (m["wer_after"], thr, w)
    model.threshold, model.w_channel = best[1], best[2]
    print("Chọn threshold=%.2f w_channel=%.1f" % (best[1], best[2]))
    out = ROOT / mcfg["model_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    # Từ điển âm tiết hợp lệ dùng chung cho bộ bảo toàn văn phong (preservation.py).
    with open(ROOT / "models/vocab.txt", "w", encoding="utf-8") as f:
        for w, c in sorted(model.vocab.items(), key=lambda x: -x[1]):
            f.write(f"{w}\t{c}\n")
    json.dump({"detector": info, "tuning": log, "chosen": {"threshold": best[1], "w_channel": best[2]},
               "train_seconds": time.time() - t0},
              open(ROOT / "outputs/metrics/ml_training.json", "w", encoding="utf-8"), indent=2)
    print(f"Đã lưu {out} ({out.stat().st_size / 1e6:.1f} MB), tổng {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
