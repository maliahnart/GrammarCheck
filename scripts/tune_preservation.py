"""Tinh chỉnh ngưỡng lm_margin của lớp bảo toàn trên tập Val (không dùng Test).
So sánh: không có lớp bảo toàn, bảo toàn không dùng LM, và bảo toàn + LM với các margin khác nhau."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate import compute_metrics  # noqa: E402
from inference import get_model  # noqa: E402
from preservation import preserve  # noqa: E402

KEYS = ["wer_after", "exact_match", "clean_kept_rate", "expressive_tokens_lost", "edit_f05"]


def main():
    val = [json.loads(line) for line in open(ROOT / "data/val.jsonl", encoding="utf-8")]
    report = {}
    for name in ("ml", "transformer"):
        raws = get_model(name).correct_batch([r["input"] for r in val])
        rows = {"raw": compute_metrics(val, raws)}
        rows["preserve_no_lm"] = compute_metrics(val, [preserve(r["input"], p, use_lm=False)["text"]
                                                       for r, p in zip(val, raws)])
        for margin in (0.0, 1.0, 2.0, 3.0, 5.0):
            rows[f"preserve_lm_m{margin}"] = compute_metrics(
                val, [preserve(r["input"], p, lm_margin=margin)["text"] for r, p in zip(val, raws)])
        report[name] = {k: {m: v[m] for m in KEYS} for k, v in rows.items()}
        for k, v in report[name].items():
            print(name, f"{k:<20}", {m: round(x, 4) for m, x in v.items()}, flush=True)
    json.dump(report, open(ROOT / "outputs/metrics/preservation_tuning.json", "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
