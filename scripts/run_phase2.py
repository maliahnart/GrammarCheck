"""Separate CLI for phase 2. Does not change any existing API or command default."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from experimental_phase2.engine import Phase2Corrector, Phase2Policy


def main():
    parser = argparse.ArgumentParser(description="GrammarCheck phase 2 lexical context experiment")
    parser.add_argument("text")
    parser.add_argument("--previous", default="")
    parser.add_argument("--next", dest="following", default="")
    parser.add_argument("--protect", action="append", default=[])
    parser.add_argument("--no-abbrev", action="store_true")
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--policy", type=Path, default=ROOT / "outputs/experimental/phase2/selected_policy.json")
    args = parser.parse_args()
    snapshot = json.loads(args.policy.read_text()) if args.policy.exists() else None
    if snapshot:
        for relative, expected in snapshot["source_hashes"].items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                parser.error("Phase 2 code/config changed after selection; run --stage dev again.")
        for relative, field in [("models/ml_baseline.pkl", "checkpoint_sha256"),
                                ("data/Viet74K.txt", "wordlist_sha256"),
                                ("data/lm_corpus.txt", "corpus_sha256")]:
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != snapshot[field]:
                parser.error("Model, corpus or wordlist changed after policy selection.")
    use_context = not args.no_context and bool(args.previous or args.following)
    engine = Phase2Corrector.from_project(use_context=use_context)
    if snapshot and use_context:
        index = ROOT / "outputs/experimental/phase2/context_index.joblib"
        if hashlib.sha256(index.read_bytes()).hexdigest() != snapshot["index_sha256"]:
            parser.error("Context index differs from the one evaluated with this policy.")
    policy = Phase2Policy(**snapshot["policy"]) if snapshot else Phase2Policy()
    result = engine.correct(args.text, policy=policy, previous=args.previous, following=args.following,
                            use_context=use_context, protected_terms=args.protect,
                            normalize_abbrev=not args.no_abbrev)
    result["policy_source"] = str(args.policy) if snapshot else "uncalibrated_default"
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
