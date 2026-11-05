"""Write the phase 2 assessment from frozen evaluation artifacts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experimental/phase2"


def pct(value):
    return f"{value * 100:.2f}%"


def main():
    evaluation = json.loads((OUT / "evaluation.json").read_text())
    dev = json.loads((OUT / "development.json").read_text())
    context = evaluation["context_test"]
    lines = ["# Đánh giá hướng phát triển lõi đợt 2", "",
        "Đợt 2 bổ sung xếp hạng theo ngữ cảnh bằng TF-IDF từ corpus train và tách quyết định sửa tự động khỏi thứ hạng từ điển. "
        "Pipeline cũ và đợt 1 giữ nguyên. Không huấn luyện lại checkpoint ML/BiLSTM/Transformer; không thêm cặp vào train.", "",
        "## Tính toàn vẹn và lựa chọn tham số", "",
        f"Đã đối chiếu {evaluation['original_files']['files_checked']} tệp trước đợt 2, tất cả không đổi. "
        f"Commit gốc của đợt này là `{evaluation['original_files']['parent_commit']}`.", "",
        f"Policy được chọn trên dev và cố định trước test: `{evaluation['policy']['policy']}`. "
        "Ngưỡng thay từ dùng 800 câu val cũ; trọng số ngữ cảnh dùng 8 probe dev. Chỉ mục fit trên 211.396 câu corpus train, không dùng nhãn probe.", "",
        "| Tập validation | WER ↓ | Exact Match ↑ | Edit F0.5 ↑ | Giữ câu đúng ↑ |",
        "|---|---:|---:|---:|---:|"]
    for label, key in [("ML cũ", "legacy"), ("Đợt 2", "phase2")]:
        m = dev[key]
        lines.append(f"| {label} | {pct(m['wer_after'])} | {pct(m['exact_match'])} | {pct(m['edit_f05'])} | {pct(m['clean_kept_rate'])} |")
    lines.extend(["", "## Hồi quy trên dữ liệu đã có", "",
        "Các tập này đã được xem trong phát triển trước đây. Đây là phân tích hồi quy khám phá, không phải xác nhận test mù mới. "
        "Cột đợt 1 lấy từ artifact cố định cùng checkpoint/dữ liệu; baseline và đợt 2 được chạy lại.", "",
        "| Tập | Pipeline | WER ↓ | Exact Match ↑ | Edit F0.5 ↑ | Giữ câu đúng ↑ |",
        "|---|---|---:|---:|---:|---:|"])
    labels = {"test": "Giả lập 1653", "test_real": "Thủ công 95", "test_vsec": "VSEC 1000", "diagnostic": "Hồi quy 24"}
    for name, values in evaluation["sets"].items():
        for label, key in [("ML cũ", "legacy"), ("Đợt 1", "phase1"), ("Đợt 2", "phase2")]:
            m = values[key]
            lines.append(f"| {labels[name]} | {label} | {pct(m['wer_after'])} | {pct(m['exact_match'])} | {pct(m['edit_f05'])} | {pct(m['clean_kept_rate'])} |")
    lines.extend(["", "### Thay đổi so với đợt 1", ""])
    for name in ("test", "test_vsec", "test_real"):
        old, new = evaluation["sets"][name]["phase1"], evaluation["sets"][name]["phase2"]
        lines.append(f"- {labels[name]}: WER {100 * (new['wer_after'] - old['wer_after']):+.2f} điểm phần trăm; "
                     f"Exact Match {100 * (new['exact_match'] - old['exact_match']):+.2f}; "
                     f"edit F0.5 {100 * (new['edit_f05'] - old['edit_f05']):+.2f}.")
    lines.extend(["", "## Xếp hạng với ngữ cảnh", "",
        "20 probe tổng hợp được soạn riêng: 8 dev thuộc 4 nhóm, 12 test thuộc 6 nhóm không giao nhau. "
        "Đây chưa phải dữ liệu lỗi thật gán nhãn độc lập. Các cặp giữ cùng câu hiện tại nhưng đổi ngữ cảnh và đáp án.", "",
        "| Phép đo | Không ngữ cảnh | Có ngữ cảnh |",
        "|---|---:|---:|",
        f"| Xếp hạng trong 2 lựa chọn cho sẵn, test 12 câu | {pct(context['ranking_without_context']['accuracy'])} | {pct(context['ranking_with_context']['accuracy'])} |",
        f"| Cặp tương phản đúng cả hai phía | {context['ranking_without_context']['contrast_pairs_both_correct']}/6 | {context['ranking_with_context']['contrast_pairs_both_correct']}/6 |",
        f"| Sửa đầu cuối, không đưa lựa chọn cho mô hình | {pct(context['end_to_end_no_context_accuracy'])} | {pct(context['end_to_end_accuracy'])} |", "",
        "Phép chấm lựa chọn cho sẵn chỉ kiểm tra bộ xếp hạng và không được gọi là độ chính xác sửa lỗi đầu cuối. "
        "Sửa đầu cuối còn phụ thuộc vào việc tự sinh đúng ứng viên và các bước bảo toàn. Điểm TF-IDF là liên kết từ vựng, không phải xác suất hiểu đúng ý định.", "",
        "| ID test | Đáp án | Không ngữ cảnh | Có ngữ cảnh | Ngữ cảnh được tự áp dụng |",
        "|---|---|---|---|---|"])
    for row in context["end_to_end"]:
        cells = [row["id"], row["target"], row["no_context_output"], row["output"], str(row["context_promoted"])]
        lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    lines.extend(["", "## Hồi quy đã biết từ đợt 1", "",
        "Các trường hợp này đã được dùng phân tích lỗi để thiết kế đợt 2; không dùng chúng làm bằng chứng độc lập về khái quát hóa.", "",
        "| ID thủ công | Đợt 1 | Đợt 2 | Đáp án |",
        "|---|---|---|---|"])
    before = {row["id"]: row for row in (json.loads(line) for line in
              (ROOT / "outputs/experimental/phase1/predictions_test_real.jsonl").read_text().splitlines())}
    after = [json.loads(line) for line in (OUT / "predictions_test_real.jsonl").read_text().splitlines()]
    for row in after:
        old = before[row["id"]]
        if old["legacy_exact"] and not old["experimental_exact"]:
            cells = [row["id"], old["output"], row["output"], row["target"]]
            lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    lines.extend(["", "## Giới hạn và quyết định", "",
        "Giữ đợt 2 ở chế độ opt-in. Kết quả tốt hơn trên bộ đã biết không chứng minh hệ thống hiểu ngữ cảnh nói chung hoặc không học vẹt. "
        "Ngữ cảnh chỉ được fit từ corpus train, nhưng các trọng số và quy tắc vẫn có thể thiên lệch theo dev. "
        "Bộ ngữ cảnh nhỏ, tổng hợp và cùng người triển khai soạn; cần dữ liệu tài liệu thật được gán nhãn độc lập để xác nhận.", "",
        "Tự sửa không ngữ cảnh vẫn có thể giữ các lỗi của baseline. Tự phục hồi dấu theo ngữ cảnh có thể sai khi hàng xóm không liên quan hoặc retrieval thiếu bằng chứng. "
        "Mọi lựa chọn trả về giữ vùng bảo vệ và vị trí gốc, nhưng chưa được bảo đảm đúng ngữ nghĩa.", "",
        "Chưa có n-best BiLSTM/ViT5, Top-3 theo vùng trong ứng dụng, learning curve mới hoặc kiểm định thống kê. "
        "Các tệp đánh giá máy đọc, policy, kế hoạch dữ liệu và manifest nằm cùng thư mục; cache TF-IDF được tạo lại bằng `--stage build` và không đưa vào commit."])
    verification_path = OUT / "verification.json"
    if verification_path.exists():
        verified = json.loads(verification_path.read_text())
        tests = verified["python_tests"]
        lines.extend(["", "## Kiểm thử", "",
            f"{tests['passed']} kiểm thử Python đạt, {tests['skipped']} bỏ qua (Transformer thật chưa được bật). "
            "Đã chạy CLI đợt 1 và đợt 2, xác minh CLI từ chối policy sai checkpoint, "
            "và kiểm tra baseline chấm lại trùng các chỉ số baseline đã lưu ở đợt 1. Chi tiết ở `verification.json`."])
    (OUT / "assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "assessment.md")


if __name__ == "__main__":
    main()
