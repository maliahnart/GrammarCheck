"""Render a factual Markdown assessment from saved experiment artifacts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experimental/phase1"


def pct(value):
    return f"{100 * value:.2f}%"


def main():
    report = json.loads((OUT / "evaluation.json").read_text())
    development = json.loads((OUT / "development.json").read_text())
    lines = ["# Đánh giá hướng phát triển lõi đợt 1", "",
             "Đây là so sánh nhánh ML thử nghiệm với pipeline ML hiện tại, dùng cùng checkpoint và dữ liệu. "
             "Không huấn luyện lại và không thêm bản ghi vào train. Các tập test đã được quan sát trong quá trình phát triển trước, "
             "nên kết quả mang tính khám phá, chưa phải xác nhận trên dữ liệu độc lập mới.", "",
             "## Tính toàn vẹn", "",
             f"Kiểm tra {report['original_files']['files_checked']} tệp gốc: "
             f"tất cả không thay đổi = {report['original_files']['all_unchanged']}. "
             "Mã suy luận cũ, API, checkpoint, dữ liệu và số liệu lịch sử giữ nguyên. Các tệp mới nằm ở các đường dẫn riêng.", "",
             "## Chọn ngưỡng trên validation", "",
             f"Policy đầy đủ: `{report['frozen_selection']['policy']}`. "
             f"Policy ablation không Viet74K: `{report['frozen_selection']['no_lexicon_policy']}`. "
             "Chọn trên 800 câu validation, đóng băng hash trước khi test. Chọn tốt trên val không tự chứng minh chất lượng thực tế.", "",
             "| Pipeline | WER val ↓ | Exact Match val ↑ | Edit F0.5 val ↑ | Giữ câu đúng val ↑ |",
             "|---|---:|---:|---:|---:|"]
    for label, key in [("ML cũ hiện tại", "legacy"), ("Thử nghiệm đầy đủ", "experimental"), ("Không Viet74K", "no_lexicon")]:
        m = development["validation"][key]
        lines.append(f"| {label} | {pct(m['wer_after'])} | {pct(m['exact_match'])} | {pct(m['edit_f05'])} | {pct(m['clean_kept_rate'])} |")
    lines.extend(["", "## Đánh giá trên các tập đã có", "",
                  "| Tập | Pipeline | WER ↓ | Exact Match ↑ | Edit F0.5 ↑ | Giữ câu đúng ↑ |",
                  "|---|---|---:|---:|---:|---:|"])
    labels = {"test": "Giả lập 1653", "test_real": "Thủ công 95", "test_vsec": "VSEC 1000", "diagnostic": "Hồi quy phát triển 24"}
    for name, values in report["sets"].items():
        for label, key in [("ML cũ", "legacy"), ("Thử nghiệm", "experimental"), ("Không Viet74K, ngưỡng riêng", "no_lexicon"),
                           ("Không Viet74K, cùng ngưỡng", "no_lexicon_same_policy")]:
            m = values[key]
            lines.append(f"| {labels[name]} | {label} | {pct(m['wer_after'])} | {pct(m['exact_match'])} | {pct(m['edit_f05'])} | {pct(m['clean_kept_rate'])} |")
    lines.extend(["", "## Khác biệt so với ML cũ", "",
                  "| Tập | Chỉ bản mới đúng cả câu | Chỉ bản cũ đúng cả câu | Câu đổi kết quả | Pool chứa đáp án | Top-3 chứa đáp án | p50 / p95 ms |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for name, values in report["sets"].items():
        d = values["details"]
        lines.append(f"| {labels[name]} | {d['experimental_only_exact']} | {d['legacy_only_exact']} | {d['changed_vs_legacy']} | "
                     f"{pct(d['candidate_pool_gold_rate'])} | {pct(d['top3_gold_rate'])} | "
                     f"{d['latency_ms']['p50']:.1f} / {d['latency_ms']['p95']:.1f} |")
    lines.extend(["", "Top-3 ở đây là giả thuyết toàn câu, không phải Top-3 cho từng vùng sửa. "
                  "Pool chứa đáp án là khả năng tìm thấy của bộ sinh, không phải độ chính xác tự sửa. "
                  "Thời gian là một lượt CPU, chưa có đối chứng tốc độ lặp lại trong cùng giao thức.", "",
                  "## Nhận định từ số liệu", ""])
    for name in ("test", "test_vsec", "test_real"):
        values = report["sets"][name]
        old, new = values["legacy"], values["experimental"]
        wer_delta = 100 * (new["wer_after"] - old["wer_after"])
        em_delta = 100 * (new["exact_match"] - old["exact_match"])
        keep_delta = 100 * (new["clean_kept_rate"] - old["clean_kept_rate"])
        lines.append(f"- {labels[name]}: WER thay đổi {wer_delta:+.2f} điểm phần trăm; "
                     f"Exact Match {em_delta:+.2f}; giữ câu đúng {keep_delta:+.2f}.")
    lines.extend(["", "### Tác dụng riêng của Viet74K", ""])
    for name in ("test", "test_vsec", "test_real"):
        values = report["sets"][name]
        full, ablated = values["experimental"], values["no_lexicon_same_policy"]
        lines.append(f"- {labels[name]}, cùng policy: thêm Viet74K làm WER thay đổi "
                     f"{100 * (full['wer_after'] - ablated['wer_after']):+.3f} điểm phần trăm và Exact Match "
                     f"{100 * (full['exact_match'] - ablated['exact_match']):+.3f} điểm phần trăm.")
    lines.extend(["", "Không quy toàn bộ cải thiện của bản mới cho Viet74K: bản mới còn thay xếp hạng, "
                  "bảo vệ vùng và quyết định áp dụng. So với ablation mới là phép kiểm tra đóng góp của từ điển trong cấu hình này.", "",
                  "## Ví dụ hồi quy cần giải quyết ở đợt sau", "",
                  "Các ví dụ dưới đây được đọc sau khi cố định cấu hình và chạy test. Không dùng chúng để sửa ngưỡng rồi báo lại như một test độc lập.", "",
                  "| ID thủ công | Đầu vào | ML cũ | Thử nghiệm |",
                  "|---|---|---|---|"])
    manual = [json.loads(line) for line in (OUT / "predictions_test_real.jsonl").read_text().splitlines()]
    for row in manual:
        if row["legacy_exact"] and not row["experimental_exact"]:
            cells = [row[key].replace("|", "\\|").replace("\n", " ") for key in ("id", "input", "legacy", "output")]
            lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "Có cả trường hợp phát sinh lỗi từ hợp lệ (thi → thì, bao → báo), bỏ sót sửa quan hệ từ, "
                  "và giữ chữ thường ở từ biểu cảm trong khi nhãn yêu cầu viết hoa đầu câu. Cần phân biệt sửa sai nội dung với "
                  "khác quy ước viết hoa khi phân tích Exact Match. Không kết luận LM hiểu ý nghĩa chỉ vì điểm xếp hạng cao.", "",
                  "## Chẩn đoán khả năng khái quát hóa", "",
                  "| Tập | ML cũ WER / Exact Match | Thử nghiệm WER / Exact Match |",
                  "|---|---:|---:|"])
    for name, values in [("400 câu trong tiền tố train detector", development["train_sample"]),
                         ("800 câu validation", development["validation"])]:
        a, b = values["legacy"], values["experimental"]
        lines.append(f"| {name} | {pct(a['wer_after'])} / {pct(a['exact_match'])} | {pct(b['wer_after'])} / {pct(b['exact_match'])} |")
    lines.extend(["", "Các tập khác độ khó và phân phối. Chênh lệch này không chứng minh mô hình học vẹt hoặc không học vẹt. "
                  "Đợt 1 không train lại nên không có learning curve mới. Cần tập lỗi người thật và ngữ cảnh độc lập ở đợt sau.", "",
                  "| Tập | Số bản ghi | Trùng khóa target với train | Khóa target lặp trong tập |",
                  "|---|---:|---:|---:|"])
    for name, values in report["overlap_audit"]["sets"].items():
        lines.append(f"| {name} | {values['n']} | {values['train_overlap_records']} | {values['duplicate_keys_within_set']} |")
    lines.extend(["", report["overlap_audit"]["limitations"], "",
                  "## Quyết định sau đợt 1", "",
                  "Giữ nhánh mới ở trạng thái opt-in. Không thay pipeline mặc định hoặc checkpoint. "
                  "Đợt tiếp theo nên phân tích các câu chỉ bản cũ đúng, đánh giá ứng viên nhập nhằng và thêm dữ liệu ngữ cảnh "
                  "trước khi tích hợp giao diện hoặc huấn luyện lại. Chỉ số tốt trên các tập hiện có chưa đủ để tự động đưa vào sản phẩm.", "",
                  "24 câu diagnostic là bộ phát triển/hồi quy nhỏ, gồm ví dụ đã thảo luận và một số câu quen thuộc trong dự án; "
                  "không dùng chúng để tuyên bố chất lượng tiếng Việt tổng quát. VSEC không được xuất lại câu ở các artifact mới.", "",
                  "Các tệp máy đọc: `evaluation.json`, `development.json`, `selected_policy.json`, "
                  "`predictions_*.jsonl` và `original_manifest.json` cùng thư mục."])
    (OUT / "assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "assessment.md")


if __name__ == "__main__":
    main()
