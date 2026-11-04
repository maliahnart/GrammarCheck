"""Re-evaluate the current local checkpoint without overwriting the historical report."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from backend.schemas import CheckRequest
from backend.service import check_document
from evaluate import compute_metrics
from inference import correct_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", nargs="+", default=["test_real"], choices=["test", "test_real", "test_vsec"])
    args = parser.parse_args()
    snapshot = {"model": "ml", "checkpoint_sha256": hashlib.sha256((ROOT / "models/ml_baseline.pkl").read_bytes()).hexdigest(),
                "note": "Current checkpoint; product uses per-sentence inference and keeps original layout. Independent of historical report.",
                "sets": {}}
    for name in args.sets:
        records = [json.loads(line) for line in (ROOT / f"data/{name}.jsonl").read_text().splitlines()]
        predictions = correct_texts([r["input"] for r in records], model="ml", normalize_abbrev=True)
        product = [check_document(CheckRequest(text=r["input"], model="ml", normalize_abbrev=True)).output for r in records]
        snapshot["sets"][name] = {"legacy_pipeline": compute_metrics(records, [r["output"] for r in predictions]),
                                 "product_document_pipeline": compute_metrics(records, product)}
        print(name, "done", flush=True)
    output = ROOT / "outputs/metrics/product_ml_evaluation.json"
    output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
