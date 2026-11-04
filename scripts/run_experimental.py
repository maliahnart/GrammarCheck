"""Explicit entry point for phase 1. The original inference/API defaults are untouched."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from experimental_core.engine import ExperimentalCorrector, Policy


def main():
    parser = argparse.ArgumentParser(description="Opt-in GrammarCheck ML experiment")
    parser.add_argument("text")
    parser.add_argument("--no-abbrev", action="store_true")
    parser.add_argument("--protect", action="append", default=[])
    parser.add_argument("--no-lexicon", action="store_true")
    parser.add_argument("--policy", type=Path, default=ROOT / "outputs/experimental/phase1/selected_policy.json")
    args = parser.parse_args()
    snapshot = json.loads(args.policy.read_text()) if args.policy.exists() else None
    if snapshot:
        for relative, expected in snapshot["source_hashes"].items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                parser.error("Policy không khớp mã/config hiện tại; chạy lại đánh giá dev trước khi dùng.")
        for relative, key in [("models/ml_baseline.pkl", "checkpoint_sha256"),
                              ("data/Viet74K.txt", "wordlist_sha256")]:
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != snapshot[key]:
                parser.error("Checkpoint hoặc wordlist đã thay đổi sau khi chọn policy.")
    policy = Policy(**snapshot["policy"]) if snapshot else Policy()
    engine = ExperimentalCorrector.from_project()
    result = engine.correct(args.text, policy=policy, normalize_abbrev=not args.no_abbrev,
                            protected_terms=args.protect, use_lexicon=not args.no_lexicon)
    result["policy_source"] = str(args.policy) if args.policy.exists() else "uncalibrated_default"
    result["checkpoint_sha256"] = hashlib.sha256((ROOT / "models/ml_baseline.pkl").read_bytes()).hexdigest()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
