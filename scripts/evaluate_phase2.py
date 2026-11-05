"""Phase 2: fit train-only retrieval -> dev selection -> frozen exploratory tests."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate import compute_metrics
from experimental_core.engine import apply_edits
from experimental_phase2.context import ContextIndex
from experimental_phase2.engine import Phase2Corrector, Phase2Policy
from vn_text import eval_form

OUT = ROOT / "outputs/experimental/phase2"
CONFIG = ROOT / "configs/experimental_phase2.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        row.setdefault("error_type", row.get("category", "context"))
        row.setdefault("style", "formal")
    return rows


def immutable_check():
    previous = json.loads((OUT / "previous_manifest.json").read_text())
    changed = [path for path, expected in previous["files"].items()
               if not (ROOT / path).exists() or sha(ROOT / path) != expected]
    if changed:
        raise RuntimeError(f"Previous pipeline/phase 1 files changed: {changed}")
    return {"parent_commit": previous["head"], "files_checked": len(previous["files"]), "all_unchanged": True}


def check_data_plan():
    plan = json.loads((OUT / "dataset_plan.json").read_text())
    for relative, expected in plan["fixture_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError("Context fixtures changed after the dataset plan was fixed.")
    return plan


def source_hashes():
    paths = [CONFIG, Path(__file__).resolve(), ROOT / "scripts/run_phase2.py",
             *sorted((ROOT / "src/experimental_core").glob("*.py")),
             *sorted((ROOT / "src/experimental_phase2").glob("*.py"))]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def build_index(config):
    start = time.perf_counter()
    metadata = {"corpus_path": config["corpus"], "corpus_sha256": sha(ROOT / config["corpus"]),
                "fit_sources": [config["corpus"]], "label_files_used": []}
    index = ContextIndex.build(ROOT / config["corpus"], metadata, config["index"])
    index.save(OUT / "context_index.joblib")
    write_json(OUT / "context_index_metadata.json", {**index.metadata,
               "cache_sha256": sha(OUT / "context_index.joblib"), "build_seconds": time.perf_counter() - start})


def analyze_rows(engine, rows):
    analyses = []
    for index, row in enumerate(rows, 1):
        analyses.append(engine.analyze(row["input"], normalize_abbrev=row.get("normalize_abbrev", True),
                                       protected_terms=row.get("protected_terms", [])))
        if index % 100 == 0:
            print(f"  {index}/{len(rows)}", flush=True)
    return analyses


def metrics(rows, outputs):
    result = compute_metrics(rows, outputs)
    result["raw_exact_match"] = sum(row["target"] == out for row, out in zip(rows, outputs)) / len(rows)
    return result


def probe_ranking(engine, rows, weight):
    predictions = []
    for row in rows:
        ranked = engine.rank_choices(row["input"], row["choices"], previous=row["previous"],
                                     following=row["next"], weight=weight)
        predictions.append({"id": row["id"], "group": row["ambiguity_group"],
                            "predicted": ranked[0]["text"], "target": row["target"],
                            "correct": ranked[0]["text"] == row["target"], "ranked": ranked})
    by_group = {}
    for prediction in predictions:
        by_group.setdefault(prediction["group"], []).append(prediction["correct"])
    return {"n": len(rows), "accuracy": sum(p["correct"] for p in predictions) / len(rows),
            "contrast_pairs_both_correct": sum(all(values) for values in by_group.values()),
            "n_contrast_pairs": len(by_group), "predictions": predictions}


def run_dev(engine, config):
    rows = read_rows(ROOT / "data/val.jsonl")
    print("Development: original val, 800 records.", flush=True)
    analyses = analyze_rows(engine, rows)
    policies = []
    for margin in config["realword_margin_grid"]:
        policy = Phase2Policy(realword_margin=margin, context_weight=0)
        # With no neighbors, automatic output is exactly this stable anchor.
        outputs = [engine._render(a, a["anchor_edits"], policy)[0] for a in analyses]
        policies.append({"margin": margin, "metrics": metrics(rows, outputs)})
    best = max(policies, key=lambda p: (p["metrics"]["edit_f05"], p["metrics"]["clean_kept_rate"],
                                       -p["metrics"]["wer_after"], p["margin"]))
    context_rows = read_rows(ROOT / "data/experimental/phase2/context_dev.jsonl")
    grid = [{"weight": weight, "ranking": probe_ranking(engine, context_rows, weight)}
            for weight in config["context_weight_grid"]]
    best_context = max(grid, key=lambda p: (p["ranking"]["accuracy"], -p["weight"]))
    policy = Phase2Policy(best["margin"], best_context["weight"], config["context_promotion_margin"])
    selected = {"policy": asdict(policy), "source_hashes": source_hashes(),
                "checkpoint_sha256": sha(ROOT / "models/ml_baseline.pkl"),
                "wordlist_sha256": sha(ROOT / config["wordlist"]), "corpus_sha256": sha(ROOT / config["corpus"]),
                "index_sha256": sha(OUT / "context_index.joblib"),
                "validation_sha256": sha(ROOT / "data/val.jsonl"), "dataset_plan": check_data_plan(),
                "selection_rule": config["selection"], "frozen_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(OUT / "selected_policy.json", selected)
    write_json(OUT / "development.json", {"legacy": metrics(rows, [a["legacy_preserved"] for a in analyses]),
               "phase2": best["metrics"], "margin_grid": policies, "context_grid": grid,
               "note": "The synthetic context dev set selects only the ranking weight, not model weights."})
    print("Frozen phase 2 policy:", asdict(policy), flush=True)


def verify_frozen(config):
    selected = json.loads((OUT / "selected_policy.json").read_text())
    if selected["source_hashes"] != source_hashes():
        raise RuntimeError("Code/config changed after dev selection; repeat dev before testing.")
    for path, field in [(ROOT / "models/ml_baseline.pkl", "checkpoint_sha256"),
                        (ROOT / config["wordlist"], "wordlist_sha256"),
                        (ROOT / config["corpus"], "corpus_sha256"),
                        (OUT / "context_index.joblib", "index_sha256")]:
        if sha(path) != selected[field]:
            raise RuntimeError(f"Artifact changed after dev selection: {path}")
    check_data_plan()
    return selected


def run_test(engine, config):
    selected = verify_frozen(config)
    policy = Phase2Policy(**selected["policy"])
    phase1 = json.loads((ROOT / "outputs/experimental/phase1/evaluation.json").read_text())
    summary = {"phase": 2, "policy": selected, "original_files": immutable_check(),
               "supervised_training_records_added": 0, "baseline_checkpoint_retrained": False,
               "context_index": json.loads((OUT / "context_index_metadata.json").read_text()),
               "runtime": {"python": sys.version, "platform": platform.platform()}, "sets": {},
               "limitations": ["Existing test sets were inspected during phase 1; this is exploratory regression analysis.",
                               "Context probes are synthetic, authored by the implementer, small, and not independent human evaluation.",
                               "Candidate-supplied probes isolate ranking; they do not measure end-to-end correction.",
                               "TF-IDF context support is lexical association, not calibrated semantic understanding.",
                               "No new learning curve or proof of absence of memorization."]}
    for name, path in [("test", "data/test.jsonl"), ("test_real", "data/test_real.jsonl"),
                       ("test_vsec", "data/test_vsec.jsonl"), ("diagnostic", "data/experimental/phase1_diagnostic.jsonl")]:
        print(f"Regression evaluation {name}", flush=True)
        rows = read_rows(ROOT / path)
        assert sha(ROOT / path) == phase1["sets"][name]["input_sha256"]
        analyses = analyze_rows(engine, rows)
        results = [engine.decide(a, policy) for a in analyses]
        details, latencies = [], []
        for row, analysis, result in zip(rows, analyses, results):
            assert apply_edits(row["input"], result["edits"]) == result["output"]
            assert not any(e["accepted"] and e["locked"] for e in result["edits"])
            for candidate in result["candidates"]:
                assert apply_edits(row["input"], candidate["edits"]) == candidate["text"]
                assert not any(e["accepted"] and e["locked"] for e in candidate["edits"])
            item = {"id": row["id"], "legacy_exact": eval_form(analysis["legacy_preserved"]) == eval_form(row["target"]),
                    "phase2_exact": eval_form(result["output"]) == eval_form(row["target"])}
            if name != "test_vsec":
                item.update(input=row["input"], target=row["target"], legacy=analysis["legacy_preserved"],
                            output=result["output"], candidates=[c["text"] for c in result["candidates"]])
            details.append(item)
            latencies.append(analysis["seconds"] * 1000)
        (OUT / f"predictions_{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in details))
        summary["sets"][name] = {"n": len(rows), "input_sha256": sha(ROOT / path),
            "legacy": metrics(rows, [a["legacy_preserved"] for a in analyses]),
            "phase1": phase1["sets"][name]["experimental"],
            "phase2": metrics(rows, [r["output"] for r in results]),
            "protected_span_violations": 0,
            "latency_ms": {"p50_analysis": statistics.median(latencies),
                           "p95_analysis": sorted(latencies)[max(0, int(.95 * len(latencies)) - 1)]}}
        print(name, {k: round(summary["sets"][name][k]["wer_after"], 5) for k in ("legacy", "phase1", "phase2")}, flush=True)
        write_json(OUT / "evaluation.json", summary)
    # Only now score the held-out ambiguity groups. Choices never enter the inference path.
    probes = read_rows(ROOT / "data/experimental/phase2/context_test.jsonl")
    print("Held-out context groups:", len(probes), flush=True)
    with_context = probe_ranking(engine, probes, policy.context_weight)
    without_context = probe_ranking(engine, probes, 0)
    end_to_end = []
    for row in probes:
        result = engine.correct(row["input"], previous=row["previous"], following=row["next"], policy=policy)
        without = engine.correct(row["input"], policy=policy)
        end_to_end.append({"id": row["id"], "document_id": row["document_id"], "target": row["target"],
                           "output": result["output"], "no_context_output": without["output"],
                           "correct": eval_form(result["output"]) == eval_form(row["target"]),
                           "no_context_correct": eval_form(without["output"]) == eval_form(row["target"]),
                           "context_promoted": result["context_promoted"], "context": result["context"],
                           "candidates": [c["text"] for c in result["candidates"]]})
    summary["context_test"] = {"ranking_with_context": with_context, "ranking_without_context": without_context,
                               "end_to_end": end_to_end,
                               "end_to_end_accuracy": sum(r["correct"] for r in end_to_end) / len(end_to_end),
                               "end_to_end_no_context_accuracy": sum(r["no_context_correct"] for r in end_to_end) / len(end_to_end)}
    summary["original_files"] = immutable_check()
    write_json(OUT / "evaluation.json", summary)
    print("Context ranking accuracy:", without_context["accuracy"], "->", with_context["accuracy"], flush=True)
    print("End-to-end context accuracy:", summary["context_test"]["end_to_end_no_context_accuracy"], "->", summary["context_test"]["end_to_end_accuracy"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["build", "dev", "test", "all"], default="all")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    immutable_check()
    check_data_plan()
    if args.stage == "build" or (args.stage == "all" and not (OUT / "context_index.joblib").exists()):
        build_index(config)
    if args.stage == "build":
        return
    engine = Phase2Corrector.from_project()
    engine.correct("Tôi đi học.")
    if args.stage in {"dev", "all"}:
        run_dev(engine, config)
    if args.stage in {"test", "all"}:
        run_test(engine, config)


if __name__ == "__main__":
    main()
