"""Giao diện demo (Gradio): sửa lỗi chính tả & ngữ pháp tiếng Việt, bảo toàn văn phong.

Chạy:  python app.py   rồi mở http://127.0.0.1:7860
"""
import difflib
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from inference import MODEL_LABELS, correct_text  # noqa: E402
from vn_text import tokenize_with_spans  # noqa: E402

LABEL_TO_KEY = {v: k for k, v in MODEL_LABELS.items()}
EXAMPLES = [
    ["hôm nay e đi hoc có vui không, hihi"],
    ["Đội tuyển Việt Nam đã dành chiến thắng trong trận chung kết."],
    ["Tuy trời mưa nhưng nên tôi vẫn đi học."],
    ["Kinh tế viet nam dang phat trien nhanh trong nhung nam gan day."],
    ["ko bít nữa, để mik hỏi lại đã :))"],
    ["Cảm ơn bạn nhiều nha, mình rất thích món quà này ^^"],
    ["Hôm qua tôi đã gặp lại người bạn củ thời đại học."],
]


def diff_highlight(src: str, out: str):
    """Danh sách (đoạn văn bản, nhãn) cho HighlightedText: '-' bị xoá, '+' được thêm."""
    S, O = tokenize_with_spans(src), tokenize_with_spans(out)
    sm = difflib.SequenceMatcher(a=[t for t, _, _ in S], b=[t for t, _, _ in O], autojunk=False)
    res = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            res.append((" ".join(t for t, _, _ in O[j1:j2]) + " ", None))
            continue
        if i2 > i1:
            res.append((" ".join(t for t, _, _ in S[i1:i2]) + " ", "-"))
        if j2 > j1:
            res.append((" ".join(t for t, _, _ in O[j1:j2]) + " ", "+"))
    return res


def run(text, model_label, preserve_style, normalize_abbrev):
    text = (text or "").strip()
    if not text:
        return "", [], [], "", "Vui lòng nhập văn bản."
    try:
        r = correct_text(text, LABEL_TO_KEY[model_label], preserve_style, normalize_abbrev)
    except FileNotFoundError as e:
        return "", [], [], "", f"Chưa có checkpoint cho mô hình này: {e}. Hãy chạy script huấn luyện tương ứng."
    rows = [[e["src"] or "∅", e["out"] or "∅", e["kind"], e["explanation"],
             "Đã sửa" if e["accepted"] else f"Chỉ gợi ý - {e['reason']}"] for e in r["edits"]]
    n_ok = sum(e["accepted"] for e in r["edits"])
    status = (f"{n_ok} chỗ đã sửa, {len(r['suggestions'])} gợi ý chưa áp dụng · {r['seconds'] * 1000:.0f} ms"
              if r["edits"] else f"Không phát hiện lỗi - giữ nguyên câu gốc · {r['seconds'] * 1000:.0f} ms")
    return r["output"], diff_highlight(text, r["output"]), rows, r["raw_output"], status


with gr.Blocks(title="Sửa lỗi tiếng Việt") as demo:
    gr.Markdown("## Phát hiện & sửa lỗi chính tả, ngữ pháp tiếng Việt\n"
                "Chỉ sửa lỗi cần thiết, giữ nguyên văn phong (từ cảm thán, emoticon, cách xưng hô).")
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="Văn bản gốc", lines=4, placeholder="Nhập câu tiếng Việt...")
            model = gr.Dropdown(list(MODEL_LABELS.values()), value=MODEL_LABELS["transformer"], label="Mô hình")
            with gr.Row():
                preserve_cb = gr.Checkbox(value=True, label="Bảo toàn văn phong (hạn chế sửa quá mức)")
                abbrev_cb = gr.Checkbox(value=True, label="Chuẩn hoá viết tắt (ko, mik, dc...)")
            btn = gr.Button("Kiểm tra và sửa lỗi", variant="primary")
            gr.Examples(EXAMPLES, inputs=[inp])
        with gr.Column():
            out = gr.Textbox(label="Văn bản sau khi sửa", lines=4)
            status = gr.Markdown()
            hl = gr.HighlightedText(label="Vị trí thay đổi (đỏ: gốc, xanh: sửa)", combine_adjacent=False,
                                    show_legend=True, color_map={"-": "red", "+": "green"})
    edits = gr.Dataframe(headers=["Gốc", "Sửa thành", "Loại lỗi", "Giải thích", "Trạng thái"],
                         label="Chi tiết các thay đổi", wrap=True)
    with gr.Accordion("Output thô của mô hình (trước lớp bảo toàn văn phong)", open=False):
        raw = gr.Textbox(show_label=False)
    btn.click(run, [inp, model, preserve_cb, abbrev_cb], [out, hl, edits, raw, status])
    inp.submit(run, [inp, model, preserve_cb, abbrev_cb], [out, hl, edits, raw, status])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
