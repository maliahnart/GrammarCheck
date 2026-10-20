"""API suy luận thống nhất cho 3 mô hình + lớp bảo toàn văn phong.

Dùng trong code:
    from inference import correct_text
    correct_text("hôm nay e đi hoc", model="transformer")

Dùng dòng lệnh:
    python src/inference.py --model ml "hom nay e di hoc"
    python src/inference.py --model transformer --no-preserve --no-abbrev "ko bít nữa"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preservation import edits_as_dicts, preserve  # noqa: E402

MODEL_LABELS = {
    "ml": "ML Baseline (TF-IDF + LogReg + n-gram)",
    "bilstm": "BiLSTM + Attention (char Seq2Seq)",
    "transformer": "Pre-trained Transformer (ViT5, nrl-ai)",
}
_CACHE: dict = {}


def load_config() -> dict:
    return yaml.safe_load(open(ROOT / "configs/config.yaml", encoding="utf-8"))


def get_model(name: str, device: str | None = None):
    key = (name, device)
    if key in _CACHE:
        return _CACHE[key]
    cfg = load_config()
    if name == "ml":
        from baseline_ml import MLCorrector
        model = MLCorrector.load(ROOT / cfg["ml"]["model_path"])
    elif name == "bilstm":
        from bilstm_attention import BiLSTMCorrector
        model = BiLSTMCorrector.load(ROOT / cfg["bilstm"]["model_path"], device=device)
    elif name == "transformer":
        from pretrained_model import TransformerCorrector
        t = cfg["transformer"]
        model = TransformerCorrector(t["model_name"], device=device, max_length=t["max_length"],
                                     num_beams=t["num_beams"])
    else:
        raise ValueError(f"Mô hình không hợp lệ: {name}")
    _CACHE[key] = model
    return model


def correct_texts(texts: list[str], model: str = "transformer", preserve_style: bool = True,
                  normalize_abbrev: bool = True, device: str | None = None) -> list[dict]:
    m = get_model(model, device)
    t0 = time.time()
    raws = m.correct_batch(texts)
    elapsed = time.time() - t0
    results = []
    for src, raw in zip(texts, raws):
        res = preserve(src, raw, normalize_abbrev=normalize_abbrev, strict=preserve_style)
        results.append({"input": src, "raw_output": raw, "output": res["text"],
                        "edits": edits_as_dicts(res["edits"]),
                        "suggestions": edits_as_dicts(res["suggestions"]),
                        "model": model, "seconds": elapsed / max(1, len(texts))})
    return results


def correct_text(text: str, model: str = "transformer", preserve_style: bool = True,
                 normalize_abbrev: bool = True, device: str | None = None) -> dict:
    return correct_texts([text], model, preserve_style, normalize_abbrev, device)[0]


def main():
    ap = argparse.ArgumentParser(description="Sửa lỗi chính tả & ngữ pháp tiếng Việt")
    ap.add_argument("text", nargs="+")
    ap.add_argument("--model", choices=list(MODEL_LABELS), default="transformer")
    ap.add_argument("--no-preserve", action="store_true", help="tắt lớp bảo toàn văn phong")
    ap.add_argument("--no-abbrev", action="store_true", help="không chuẩn hoá viết tắt (ko, mik, dc...)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    for r in correct_texts([" ".join(args.text)], args.model, not args.no_preserve, not args.no_abbrev,
                           args.device):
        print("Gốc     :", r["input"])
        print("Mô hình :", r["raw_output"])
        print("Kết quả :", r["output"])
        for e in r["edits"]:
            mark = "✓" if e["accepted"] else "✗ (gợi ý)"
            print(f"  {mark} '{e['src']}' -> '{e['out']}'  [{e['explanation']}] {e['reason']}")


if __name__ == "__main__":
    main()
