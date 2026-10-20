"""Huấn luyện BiLSTM + Attention trên data/train.jsonl, chọn checkpoint tốt nhất theo CER trên Val."""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bilstm_attention import train  # noqa: E402


def read_pairs(p):
    rows = [json.loads(line) for line in open(p, encoding="utf-8")]
    return [(r["input"], r["target"]) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="chỉ dùng N cặp train (chạy thử nhanh)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs/config.yaml", encoding="utf-8"))
    bcfg = dict(cfg["bilstm"])
    if args.epochs:
        bcfg["epochs"] = args.epochs
    train_pairs = read_pairs(ROOT / "data/train.jsonl")[: args.limit]
    val_pairs = read_pairs(ROOT / "data/val.jsonl")[:400]
    out = ROOT / bcfg["model_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    hist = train(train_pairs, val_pairs, bcfg, out, device=args.device, seed=cfg["seed"])
    json.dump(hist, open(ROOT / "outputs/metrics/bilstm_training.json", "w", encoding="utf-8"), indent=2)
    print("Đã lưu", out)


if __name__ == "__main__":
    main()
