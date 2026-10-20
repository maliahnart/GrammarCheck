"""Vẽ biểu đồ từ số liệu ĐO THỰC TẾ trong outputs/metrics/*.json -> outputs/figures/*.png."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MET, FIG = ROOT / "outputs/metrics", ROOT / "outputs/figures"
MODELS = ["ml", "bilstm", "transformer"]
NAMES = {"ml": "ML Baseline", "bilstm": "BiLSTM + Attention", "transformer": "Transformer (ViT5)"}
COLORS = {"ml": "#2a78d6", "bilstm": "#eb6834", "transformer": "#1baf7a"}
SET_NAMES = {"test": "Test giả lập", "test_vsec": "VSEC (lỗi thật)", "test_real": "Bộ thủ công"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})


def _style(ax, ylabel):
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def grouped(metric, ylabel, fname, title, pct=True):
    res = json.load(open(MET / "all_metrics.json", encoding="utf-8"))
    sets = [s for s in SET_NAMES if s in res]
    fig, axes = plt.subplots(1, len(sets), figsize=(4.2 * len(sets), 3.8), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, sets):
        x = np.arange(len(MODELS))
        w = 0.36
        before = res[s][MODELS[0]]["raw"].get(metric.replace("_after", "_before"))
        for k, var in enumerate(("raw", "preserve")):
            vals = [res[s][m][var][metric] if m in res[s] else np.nan for m in MODELS]
            bars = ax.bar(x + (k - 0.5) * w, vals, w - 0.03, color=[COLORS[m] for m in MODELS],
                          hatch="///" if var == "raw" else None, edgecolor="white", linewidth=0.6,
                          alpha=0.55 if var == "raw" else 1.0)
            for b, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width() / 2, v, f"{v * 100:.1f}" if pct else f"{v:.0f}",
                            ha="center", va="bottom", fontsize=7.5, color=TEXT)
        if before is not None and metric.endswith("_after"):
            ax.axhline(before, color=MUTED, linestyle="--", linewidth=1)
            ax.text(len(MODELS) - 0.5, before, "trước khi sửa", ha="right", va="bottom", fontsize=8, color=MUTED)
        ax.set_xticks(x, [NAMES[m].replace(" + ", "\n+ ").replace(" (", "\n(") for m in MODELS], fontsize=8.5)
        ax.set_title(SET_NAMES[s], fontsize=10.5, color=TEXT)
        _style(ax, ylabel if ax is axes[0] else "")
        if pct:
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor="#9a9994", hatch="///", alpha=0.55, label="Output thô của mô hình"),
                        Patch(facecolor="#5f5e5a", label="+ Lớp bảo toàn văn phong")],
               loc="upper right", frameon=False, fontsize=8.5, ncol=2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / fname)
    plt.close(fig)


def by_error_type():
    res = json.load(open(MET / "all_metrics.json", encoding="utf-8"))["test"]
    types = [t for t in res[MODELS[0]]["preserve"]["exact_by_error_type"]]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(types))
    w = 0.26
    for k, m in enumerate(MODELS):
        if m not in res:
            continue
        vals = [res[m]["preserve"]["exact_by_error_type"][t] for t in types]
        ax.bar(x + (k - 1) * w, vals, w - 0.03, color=COLORS[m], label=NAMES[m])
    counts = res[MODELS[0]]["preserve"]["count_by_error_type"]
    ax.set_xticks(x, [f"{t}\n(n={counts[t]})" for t in types])
    _style(ax, "Exact Match")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.set_title("Exact Match theo loại lỗi - Test giả lập (+ bảo toàn)", loc="left", fontsize=11, pad=24)
    fig.tight_layout()
    fig.savefig(FIG / "exact_by_error_type.png")
    plt.close(fig)


def latency():
    lat = json.load(open(MET / "latency.json", encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    rows = [(m, d, lat[m][d]) for m in MODELS if m in lat for d in ("cpu_ms", "gpu_ms") if d in lat[m]]
    y = np.arange(len(rows))
    ax.barh(y, [r[2] for r in rows], color=[COLORS[r[0]] for r in rows], height=0.6)
    for yi, r in zip(y, rows):
        ax.text(r[2] * 1.08, yi, f"{r[2]:.1f} ms", va="center", fontsize=8.5, color=TEXT)
    ax.set_yticks(y, [f"{NAMES[r[0]]} - {'GPU' if r[1] == 'gpu_ms' else 'CPU'}" for r in rows], fontsize=8.5)
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("ms / câu (batch = 1, thang log)")
    ax.set_title("Thời gian xử lý trung bình mỗi câu", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "latency.png")
    plt.close(fig)


def bilstm_curve():
    hist = json.load(open(MET / "bilstm_training.json", encoding="utf-8"))
    ep = [h["epoch"] for h in hist]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    axes[0].plot(ep, [h["train_loss"] for h in hist], color=COLORS["bilstm"], linewidth=2, marker="o", ms=4)
    _style(axes[0], "Cross-entropy (train)")
    axes[0].set_title("Loss huấn luyện", loc="left", fontsize=10.5)
    axes[1].plot(ep, [h["val_cer"] for h in hist], color=COLORS["bilstm"], linewidth=2, marker="o", ms=4)
    _style(axes[1], "CER trên Val")
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.1f}%"))
    axes[1].set_title("CER trên tập Val (400 câu)", loc="left", fontsize=10.5)
    for ax in axes:
        ax.set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(FIG / "bilstm_training.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    grouped("wer_after", "WER sau sửa", "wer_after.png", "WER sau khi sửa (thấp hơn là tốt hơn)")
    grouped("exact_match", "Exact Match", "exact_match.png", "Tỉ lệ câu sửa trùng đáp án (cao hơn là tốt hơn)")
    grouped("clean_kept_rate", "Câu đúng được giữ nguyên", "clean_kept.png",
            "Bảo toàn: tỉ lệ câu vốn đúng được giữ nguyên")
    grouped("expressive_tokens_lost", "Số token", "expressive_lost.png",
            "Số từ biểu cảm / emoticon bị mất hoặc bị đổi", pct=False)
    by_error_type()
    if (MET / "latency.json").exists():
        latency()
    if (MET / "bilstm_training.json").exists():
        bilstm_curve()
    print("Đã lưu biểu đồ vào", FIG)


if __name__ == "__main__":
    main()
