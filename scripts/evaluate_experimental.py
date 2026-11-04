"""Dev selection -> freeze -> test, with independent artifacts and immutable old files.

No training is performed. VSEC sentence text is never exported in new artifacts.
The authored diagnostic is a development regression set, NOT an independent test.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import random
import statistics
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate import compute_metrics
from experimental_core.engine import ExperimentalCorrector, Policy, apply_edits
from vn_text import eval_form, is_word, norm_key, tokenize

CONFIG_PATH = ROOT / "configs/experimental_phase1.json"
DEFAULT_OUTPUT = ROOT / "outputs/experimental/phase1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    paths = [CONFIG_PATH, Path(__file__).resolve(),
             *sorted((ROOT / "src/experimental_core").glob("*.py"))]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def read_records(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        row.setdefault("error_type", row.get("category", "unknown"))
    return rows


def immutable_check(out):
    manifest = json.loads((out / "original_manifest.json").read_text())
    changed = [p for p, expected in manifest.items() if not (ROOT / p).exists() or sha(ROOT / p) != expected]
    if changed:
        raise RuntimeError(f"Original files changed since experiment started: {changed}")
    return {"files_checked": len(manifest), "all_unchanged": True}


def metrics(records, predictions):
    result = compute_metrics(records, predictions)
    result["raw_exact_match"] = sum(r["target"] == p for r, p in zip(records, predictions)) / len(records)
    return result


def analyze_rows(engine, records, use_lexicon=True):
    analyses = []
    for index, row in enumerate(records, 1):
        analyses.append(engine.analyze(row["input"], normalize_abbrev=row.get("normalize_abbrev", True),
                                       protected_terms=row.get("protected_terms", []), use_lexicon=use_lexicon))
        if index % 100 == 0:
            print(f"  {'lexicon' if use_lexicon else 'no_lexicon'}: {index}/{len(records)}", flush=True)
    return analyses


def policy_metrics(records, analyses, policy):
    return metrics(records, [ExperimentalCorrector.decide(a, policy)["output"] for a in analyses])


def choose_policy(config, records, analyses):
    rows = []
    grid = config["policy_grid"]
    for realword, sentence in itertools.product(grid["realword_margin"], grid["sentence_margin"]):
        policy = Policy(realword, sentence)
        result = policy_metrics(records, analyses, policy)
        rows.append({"policy": asdict(policy), "metrics": result})
    chosen = max(rows, key=lambda row: (row["metrics"]["edit_f05"], row["metrics"]["clean_kept_rate"],
                                     -row["metrics"]["wer_after"], row["policy"]["realword_margin"],
                                     row["policy"]["sentence_margin"]))
    return chosen, rows


def prediction_rows(records, analyses, policy, include_text=True):
    output, stats = [], Counter()
    times = []
    for row, analysis in zip(records, analyses):
        result = ExperimentalCorrector.decide(analysis, policy)
        legacy = analysis["legacy_preserved"]
        final = result["output"]
        gold = eval_form(row["target"])
        preserved = not any(e["accepted"] and e["locked"] for e in result["edits"])
        assert preserved
        assert apply_edits(row["input"], result["edits"]) == final
        stats["protected_span_violations"] += not preserved
        stats["legacy_only_exact"] += eval_form(legacy) == gold and eval_form(final) != gold
        stats["experimental_only_exact"] += eval_form(legacy) != gold and eval_form(final) == gold
        stats["changed_vs_legacy"] += eval_form(legacy) != eval_form(final)
        stats["accepted_edits"] += sum(e["accepted"] for e in result["edits"])
        stats["blocked_edits"] += sum(not e["accepted"] for e in result["edits"])
        stats["proposed_edits"] += len(result["edits"])
        stats["candidate_pool_gold"] += any(eval_form(c["text"]) == gold for c in analysis["candidates"])
        stats["top3_gold"] += any(eval_form(c["text"]) == gold for c in result["candidates"])
        stats["keep_original_is_gold"] += eval_form(row["input"]) == gold
        times.append(analysis["seconds"])
        item = {"id": row["id"], "legacy_exact": eval_form(legacy) == gold,
                "experimental_exact": eval_form(final) == gold,
                "top3_contains_gold": any(eval_form(c["text"]) == gold for c in result["candidates"]),
                "route": result["route"]["mode"], "accepted_edits": sum(e["accepted"] for e in result["edits"])}
        if include_text:
            item.update(input=row["input"], target=row["target"], legacy=legacy, output=final,
                        best_candidate=result["best_candidate"], candidates=result["candidates"], edits=result["edits"])
        output.append(item)
    details = dict(stats)
    details["candidate_pool_gold_rate"] = stats["candidate_pool_gold"] / len(records)
    details["top3_gold_rate"] = stats["top3_gold"] / len(records)
    details["auto_acceptance_rate"] = stats["accepted_edits"] / max(1, stats["proposed_edits"])
    details["latency_ms"] = {"p50": statistics.median(times) * 1000,
                             "p95": sorted(times)[max(0, int(0.95 * len(times)) - 1)] * 1000,
                             "mean": statistics.mean(times) * 1000}
    return output, details


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def train_prefix_sample(engine, count=400):
    """Sample only the prefix consumed by the LR detector; not the full train corpus."""
    prefix, tokens = [], 0
    for line in (ROOT / "data/train.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        prefix.append(row)
        tokens += sum(is_word(t) for t in tokenize(row["input"]))
        if tokens >= 400000:
            break
    sample = random.Random(42).sample(prefix, min(count, len(prefix)))
    return sample, {"population_records": len(prefix), "population_tokens": tokens,
                    "sample_size": len(sample), "seed": 42,
                    "purpose": "Descriptive in-sample score, not used in selection and not proof of no overfitting."}


def overlap_audit():
    train_keys = {norm_key(json.loads(line)["target"])
                  for line in (ROOT / "data/train.jsonl").open(encoding="utf-8")}
    result = {"method": "Intersection of normalized clean-target keys (lowercase, unaccented, alphanumeric).",
              "limitations": "Does not detect all paraphrases, source-document overlap or pretrained-model contamination.",
              "train_unique_keys": len(train_keys), "sets": {}}
    for name, path in [("val", "data/val.jsonl"), ("test", "data/test.jsonl"),
                       ("test_real", "data/test_real.jsonl"), ("test_vsec", "data/test_vsec.jsonl"),
                       ("diagnostic", "data/experimental/phase1_diagnostic.jsonl")]:
        rows = read_records(ROOT / path)
        keys = [norm_key(row["target"]) for row in rows]
        result["sets"][name] = {"n": len(rows), "train_overlap_records": sum(k in train_keys for k in keys),
                                  "duplicate_keys_within_set": len(keys) - len(set(keys))}
    return result


def run_dev(engine, config, out):
    records = read_records(ROOT / "data/val.jsonl")
    print(f"Validation: {len(records)} records; tuning uses this set only.", flush=True)
    analyses = analyze_rows(engine, records)
    no_lexicon = analyze_rows(engine, records, False)
    chosen, grid = choose_policy(config, records, analyses)
    no_lexicon_chosen, no_lexicon_grid = choose_policy(config, records, no_lexicon)
    policy = Policy(**chosen["policy"])
    manifest = {"version": config["version"], "policy": chosen["policy"],
                "no_lexicon_policy": no_lexicon_chosen["policy"], "selected_on": "data/val.jsonl",
                "n_validation": len(records), "selection_rule": config["selection"],
                "source_hashes": source_hashes(), "checkpoint_sha256": sha(ROOT / "models/ml_baseline.pkl"),
                "wordlist_sha256": sha(ROOT / config["wordlist"]),
                "validation_sha256": sha(ROOT / "data/val.jsonl"),
                "frozen_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(out / "selected_policy.json", manifest)
    rows, details = prediction_rows(records, analyses, policy)
    write_rows(out / "validation_predictions.jsonl", rows)
    validation = {"legacy": metrics(records, [a["legacy_preserved"] for a in analyses]),
                  "experimental": chosen["metrics"], "no_lexicon": no_lexicon_chosen["metrics"],
                  "details": details, "grid": grid, "no_lexicon_grid": no_lexicon_grid}
    sample, sample_info = train_prefix_sample(engine)
    print("In-sample diagnostic: 400 records from detector fitting prefix.", flush=True)
    train_analyses = analyze_rows(engine, sample)
    train_rows, train_details = prediction_rows(sample, train_analyses, policy)
    write_rows(out / "train_sample_predictions.jsonl", train_rows)
    write_json(out / "development.json", {"validation": validation, "train_sample": {
        "provenance": sample_info, "legacy": metrics(sample, [a["legacy_preserved"] for a in train_analyses]),
        "experimental": policy_metrics(sample, train_analyses, policy), "details": train_details}})
    print("Frozen policy:", chosen["policy"], "val F0.5:", chosen["metrics"]["edit_f05"], flush=True)


def run_test(engine, config, out):
    selected = json.loads((out / "selected_policy.json").read_text())
    if selected["source_hashes"] != source_hashes():
        raise RuntimeError("Experiment code/config changed after dev selection. Run --stage dev again before testing.")
    if selected["checkpoint_sha256"] != sha(ROOT / "models/ml_baseline.pkl"):
        raise RuntimeError("Checkpoint changed after selection.")
    if selected["wordlist_sha256"] != sha(ROOT / config["wordlist"]):
        raise RuntimeError("Wordlist changed after selection.")
    policy = Policy(**selected["policy"])
    no_lexicon_policy = Policy(**selected["no_lexicon_policy"])
    summary = {"phase": 1, "trained_weights": False, "added_training_records": 0,
               "runtime": {"python": sys.version, "platform": platform.platform()},
               "frozen_selection": selected, "sets": {}, "limitations": [
                   "Existing test sets have been inspected in earlier project development; results are exploratory.",
                   "Diagnostic cases are authored development regressions, not an independent benchmark.",
                   "No learned cross-sentence reranker, BiLSTM n-best or Transformer n-best in phase 1.",
                   "Top-3 and oracle figures are whole-sentence candidate coverage, not correction accuracy.",
                   "Latency is a single sequential CPU run including search/guard; no concurrency or repeated timing.",
                   "No claim that no exact train overlap proves absence of memorization."]}
    summary["overlap_audit"] = overlap_audit()
    for name, path in [("test", ROOT / "data/test.jsonl"), ("test_real", ROOT / "data/test_real.jsonl"),
                       ("test_vsec", ROOT / "data/test_vsec.jsonl"),
                       ("diagnostic", ROOT / "data/experimental/phase1_diagnostic.jsonl")]:
        records = read_records(path)
        print(f"Evaluation {name}: {len(records)}", flush=True)
        analyses = analyze_rows(engine, records)
        ablation = analyze_rows(engine, records, False)
        predictions, details = prediction_rows(records, analyses, policy, include_text=name != "test_vsec")
        write_rows(out / f"predictions_{name}.jsonl", predictions)
        summary["sets"][name] = {
            "n": len(records), "input_sha256": sha(path),
            "legacy": metrics(records, [a["legacy_preserved"] for a in analyses]),
            "experimental": policy_metrics(records, analyses, policy),
            "no_lexicon": policy_metrics(records, ablation, no_lexicon_policy),
            "no_lexicon_same_policy": policy_metrics(records, ablation, policy),
            "details": details}
        print(name, {key: round(summary["sets"][name][key]["wer_after"], 5)
                     for key in ("legacy", "experimental", "no_lexicon")}, flush=True)
        write_json(out / "evaluation.json", summary)
    summary["original_files"] = immutable_check(out)
    write_json(out / "evaluation.json", summary)
    print("Original files unchanged:", summary["original_files"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["dev", "test", "all"], default="all")
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text())
    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
    immutable_check(DEFAULT_OUTPUT)
    engine = ExperimentalCorrector.from_project()
    # Warm up caches, without labels or using this result for selection.
    engine.correct("Tôi đi học.")
    if args.stage in {"dev", "all"}:
        run_dev(engine, config, DEFAULT_OUTPUT)
    if args.stage in {"test", "all"}:
        run_test(engine, config, DEFAULT_OUTPUT)


if __name__ == "__main__":
    main()
