"""Đánh giá 3 mô hình (thô và + lớp bảo toàn văn phong) trên cùng các tập Test.

Tập Test:
  test       - giả lập (câu sạch từ split val của nrl-ai, lỗi do generate_noise.py sinh ra)
  test_vsec  - lỗi do người gõ thật (VSEC), 800 câu lỗi + 200 câu đúng
  test_real  - 95 câu tự soạn thủ công: trang trọng, hội thoại, teencode, câu đúng có biểu cảm

Đầu ra:
  outputs/predictions/<model>__<set>.jsonl
  outputs/metrics/all_metrics.json, results_<set>.csv, latency.json, model_size.json

Chạy:  python scripts/evaluate_all.py            (toàn bộ)
       python scripts/evaluate_all.py --models ml --sets test_real
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate import compute_metrics  # noqa: E402
from inference import MODEL_LABELS, get_model, load_config  # noqa: E402
from preservation import preserve  # noqa: E402

SETS = ["test", "test_vsec", "test_real"]
MODELS = ["ml", "bilstm", "transformer"]
MAIN_KEYS = ["sent_precision", "sent_recall", "sent_f1", "tok_precision", "tok_recall", "tok_f1",
             "edit_f05", "cer_before", "cer_after", "wer_before", "wer_after", "exact_match",
             "clean_kept_rate", "overcorrected_clean", "expressive_tokens_lost"]


def read_jsonl(p):
    return [json.loads(line) for line in open(p, encoding="utf-8")]


def measure_latency(name: str, texts: list[str], device: str) -> float:
    """Thời gian trung bình mỗi câu (ms) khi xử lý từng câu một (batch = 1), gồm cả lớp bảo toàn."""
    model = get_model(name, device)
    model.correct_batch(texts[:3])  # khởi động (warm-up)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for t in texts:
        out = model.correct_batch([t])[0]
        preserve(t, out)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / len(texts) * 1000


def model_size(name: str, cfg: dict) -> dict:
    if name == "ml":
        p = ROOT / cfg["ml"]["model_path"]
        m = get_model("ml")
        return {"file_mb": p.stat().st_size / 1e6, "params": None,
                "note": f"{len(m.vocab)} âm tiết, {len(m.lm.bi)} bigram, LR {m.clf.coef_.size} trọng số"}
    if name == "bilstm":
        p = ROOT / cfg["bilstm"]["model_path"]
        ck = torch.load(p, map_location="cpu", weights_only=False)
        return {"file_mb": p.stat().st_size / 1e6, "params": ck["n_params"], "best_epoch": ck["epoch"]}
    from huggingface_hub import HfApi
    info = HfApi().model_info(cfg["transformer"]["model_name"], files_metadata=True)
    size = sum(s.size or 0 for s in info.siblings if s.rfilename.endswith((".safetensors", ".bin")))
    return {"file_mb": size / 1e6, "params": get_model("transformer").n_params}


def run(models, sets, device):
    cfg = load_config()
    data = {s: read_jsonl(ROOT / f"data/{s}.jsonl") for s in sets}
    out_pred = ROOT / "outputs/predictions"
    out_met = ROOT / "outputs/metrics"
    out_pred.mkdir(parents=True, exist_ok=True)
    out_met.mkdir(parents=True, exist_ok=True)
    all_path = out_met / "all_metrics.json"
    results = json.load(open(all_path, encoding="utf-8")) if all_path.exists() else {}
    for name in models:
        model = get_model(name, device)
        for s in sets:
            recs = data[s]
            t0 = time.time()
            raws = model.correct_batch([r["input"] for r in recs])
            t_model = time.time() - t0
            finals = [preserve(r["input"], p)["text"] for r, p in zip(recs, raws)]
            t_total = time.time() - t0
            with open(out_pred / f"{name}__{s}.jsonl", "w", encoding="utf-8") as f:
                for r, p, q in zip(recs, raws, finals):
                    f.write(json.dumps({"id": r["id"], "input": r["input"], "target": r["target"], "raw": p,
                                        "final": q, "error_type": r["error_type"], "style": r["style"],
                                        "category": r.get("category")}, ensure_ascii=False) + "\n")
            for variant, preds in (("raw", raws), ("preserve", finals)):
                m = compute_metrics(recs, preds)
                m["batch_seconds_per_sent"] = (t_model if variant == "raw" else t_total) / len(recs)
                results.setdefault(s, {}).setdefault(name, {})[variant] = m
                print(f"[{s}] {name:<11} {variant:<8} WER {m['wer_before']:.4f}->{m['wer_after']:.4f} "
                      f"CER {m['cer_after']:.4f} EM {m['exact_match']:.3f} keep {m['clean_kept_rate']} "
                      f"lost {m['expressive_tokens_lost']}", flush=True)
            json.dump(results, open(all_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    for s in results:
        with open(out_met / f"results_{s}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "variant"] + MAIN_KEYS)
            for name, variants in results[s].items():
                for variant, m in variants.items():
                    w.writerow([MODEL_LABELS[name], variant] + [m[k] for k in MAIN_KEYS])


def run_latency(models, cfg, device):
    test = [r["input"] for r in read_jsonl(ROOT / "data/test.jsonl")]
    path = ROOT / "outputs/metrics/latency.json"
    lat = json.load(open(path, encoding="utf-8")) if path.exists() else {}
    for name in models:
        lat.setdefault(name, {})
        if device == "cuda" and name != "ml":
            lat[name]["gpu_ms"] = measure_latency(name, test[: cfg["eval"]["latency_samples"]], "cuda")
        lat[name]["cpu_ms"] = measure_latency(name, test[: cfg["eval"]["latency_samples_cpu"]], "cpu")
        print(name, lat[name], flush=True)
        json.dump(lat, open(path, "w", encoding="utf-8"), indent=2)
    sizes = {name: model_size(name, cfg) for name in models}
    path = ROOT / "outputs/metrics/model_size.json"
    old = json.load(open(path, encoding="utf-8")) if path.exists() else {}
    json.dump(old | sizes, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(sizes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--sets", nargs="+", default=SETS, choices=SETS)
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--latency-only", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config()
    if not args.latency_only:
        run(args.models, args.sets, device)
    if not args.skip_latency:
        run_latency(args.models, cfg, device)


if __name__ == "__main__":
    main()
