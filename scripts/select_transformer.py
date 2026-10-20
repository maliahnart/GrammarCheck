"""Thí nghiệm chọn checkpoint Transformer: chạy 2 ứng viên trên tập Val (giả lập),
200 câu VSEC đầu (lỗi thật) và bộ câu thủ công; KHÔNG dùng tập Test giả lập để chọn."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate import compute_metrics  # noqa: E402
from pretrained_model import TransformerCorrector  # noqa: E402

CANDIDATES = ["bmd1905/vietnamese-correction-v2", "nrl-ai/vn-spell-correction-base"]
KEYS = ["wer_after", "cer_after", "exact_match", "clean_kept_rate", "expressive_tokens_lost"]


def read_jsonl(p):
    return [json.loads(line) for line in open(p, encoding="utf-8")]


def main():
    sets = {"val": read_jsonl(ROOT / "data/val.jsonl")[:300],
            "vsec_200": read_jsonl(ROOT / "data/test_vsec.jsonl")[:200],
            "real": read_jsonl(ROOT / "data/test_real.jsonl")}
    report = {}
    for name in CANDIDATES:
        t0 = time.time()
        model = TransformerCorrector(name)
        report[name] = {"load_seconds": time.time() - t0, "params_M": model.n_params / 1e6}
        for sname, recs in sets.items():
            t0 = time.time()
            preds = model.correct_batch([r["input"] for r in recs])
            m = compute_metrics(recs, preds)
            report[name][sname] = {k: m[k] for k in KEYS} | {"sec_per_sent": (time.time() - t0) / len(recs)}
            print(name, sname, json.dumps(report[name][sname]))
            if sname == "real":
                report[name]["real_examples"] = [{"input": r["input"], "pred": p, "target": r["target"]}
                                                 for r, p in zip(recs, preds)]
        del model
    json.dump(report, open(ROOT / "outputs/metrics/transformer_selection.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
