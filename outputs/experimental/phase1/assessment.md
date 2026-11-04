# Đánh giá hướng phát triển lõi đợt 1

Đây là so sánh nhánh ML thử nghiệm với pipeline ML hiện tại, dùng cùng checkpoint và dữ liệu. Không huấn luyện lại và không thêm bản ghi vào train. Các tập test đã được quan sát trong quá trình phát triển trước, nên kết quả mang tính khám phá, chưa phải xác nhận trên dữ liệu độc lập mới.

## Tính toàn vẹn

Kiểm tra 78 tệp gốc: tất cả không thay đổi = True. Mã suy luận cũ, API, checkpoint, dữ liệu và số liệu lịch sử giữ nguyên. Các tệp mới nằm ở các đường dẫn riêng.

## Chọn ngưỡng trên validation

Policy đầy đủ: `{'realword_margin': 0.0, 'sentence_margin': 0.15}`. Policy ablation không Viet74K: `{'realword_margin': 0.0, 'sentence_margin': 0.0}`. Chọn trên 800 câu validation, đóng băng hash trước khi test. Chọn tốt trên val không tự chứng minh chất lượng thực tế.

| Pipeline | WER val ↓ | Exact Match val ↑ | Edit F0.5 val ↑ | Giữ câu đúng val ↑ |
|---|---:|---:|---:|---:|
| ML cũ hiện tại | 4.33% | 50.00% | 70.55% | 73.75% |
| Thử nghiệm đầy đủ | 3.89% | 58.25% | 71.95% | 90.00% |
| Không Viet74K | 3.88% | 58.25% | 72.17% | 90.00% |

## Đánh giá trên các tập đã có

| Tập | Pipeline | WER ↓ | Exact Match ↑ | Edit F0.5 ↑ | Giữ câu đúng ↑ |
|---|---|---:|---:|---:|---:|
| Giả lập 1653 | ML cũ | 4.01% | 51.42% | 71.09% | 78.55% |
| Giả lập 1653 | Thử nghiệm | 3.96% | 53.78% | 70.60% | 86.51% |
| Giả lập 1653 | Không Viet74K, ngưỡng riêng | 3.96% | 53.72% | 70.61% | 83.74% |
| Giả lập 1653 | Không Viet74K, cùng ngưỡng | 3.96% | 54.08% | 70.85% | 86.51% |
| Thủ công 95 | ML cũ | 2.47% | 72.63% | 78.77% | 77.50% |
| Thủ công 95 | Thử nghiệm | 3.49% | 63.16% | 72.77% | 77.50% |
| Thủ công 95 | Không Viet74K, ngưỡng riêng | 2.81% | 68.42% | 77.01% | 77.50% |
| Thủ công 95 | Không Viet74K, cùng ngưỡng | 2.90% | 67.37% | 76.35% | 82.50% |
| VSEC 1000 | ML cũ | 2.05% | 53.80% | 61.12% | 81.00% |
| VSEC 1000 | Thử nghiệm | 2.07% | 53.40% | 61.33% | 85.50% |
| VSEC 1000 | Không Viet74K, ngưỡng riêng | 2.05% | 53.80% | 61.17% | 81.00% |
| VSEC 1000 | Không Viet74K, cùng ngưỡng | 2.03% | 53.80% | 61.94% | 85.50% |
| Hồi quy phát triển 24 | ML cũ | 3.83% | 75.00% | 58.82% | 64.71% |
| Hồi quy phát triển 24 | Thử nghiệm | 3.28% | 79.17% | 66.67% | 70.59% |
| Hồi quy phát triển 24 | Không Viet74K, ngưỡng riêng | 2.73% | 79.17% | 66.67% | 70.59% |
| Hồi quy phát triển 24 | Không Viet74K, cùng ngưỡng | 2.73% | 79.17% | 66.67% | 70.59% |

## Khác biệt so với ML cũ

| Tập | Chỉ bản mới đúng cả câu | Chỉ bản cũ đúng cả câu | Câu đổi kết quả | Pool chứa đáp án | Top-3 chứa đáp án | p50 / p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| Giả lập 1653 | 68 | 29 | 246 | 63.76% | 61.34% | 17.1 / 32.7 |
| Thủ công 95 | 1 | 10 | 13 | 80.00% | 75.79% | 9.2 / 14.2 |
| VSEC 1000 | 12 | 16 | 76 | 64.40% | 60.20% | 21.8 / 43.7 |
| Hồi quy phát triển 24 | 1 | 0 | 2 | 100.00% | 95.83% | 6.2 / 8.2 |

Top-3 ở đây là giả thuyết toàn câu, không phải Top-3 cho từng vùng sửa. Pool chứa đáp án là khả năng tìm thấy của bộ sinh, không phải độ chính xác tự sửa. Thời gian là một lượt CPU, chưa có đối chứng tốc độ lặp lại trong cùng giao thức.

## Nhận định từ số liệu

- Giả lập 1653: WER thay đổi -0.05 điểm phần trăm; Exact Match +2.36; giữ câu đúng +7.96.
- VSEC 1000: WER thay đổi +0.02 điểm phần trăm; Exact Match -0.40; giữ câu đúng +4.50.
- Thủ công 95: WER thay đổi +1.02 điểm phần trăm; Exact Match -9.47; giữ câu đúng +0.00.

### Tác dụng riêng của Viet74K

- Giả lập 1653, cùng policy: thêm Viet74K làm WER thay đổi -0.003 điểm phần trăm và Exact Match -0.302 điểm phần trăm.
- VSEC 1000, cùng policy: thêm Viet74K làm WER thay đổi +0.039 điểm phần trăm và Exact Match -0.400 điểm phần trăm.
- Thủ công 95, cùng policy: thêm Viet74K làm WER thay đổi +0.596 điểm phần trăm và Exact Match -4.211 điểm phần trăm.

Không quy toàn bộ cải thiện của bản mới cho Viet74K: bản mới còn thay xếp hạng, bảo vệ vùng và quyết định áp dụng. So với ablation mới là phép kiểm tra đóng góp của từ điển trong cấu hình này.

## Ví dụ hồi quy cần giải quyết ở đợt sau

Các ví dụ dưới đây được đọc sau khi cố định cấu hình và chạy test. Không dùng chúng để sửa ngưỡng rồi báo lại như một test độc lập.

| ID thủ công | Đầu vào | ML cũ | Thử nghiệm |
|---|---|---|---|
| real-11 | Vì thời tiết xấu nhưng chuyến bay bị hoãn lại. | Vì thời tiết xấu nên chuyến bay bị hoãn lại. | Vì thời tiết xấu nhưng chuyến bay bị hoãn lại. |
| real-36 | haha, hôm qua bọn mình đi chơi vui lắm luôn :)) | Haha, hôm qua bọn mình đi chơi vui lắm luôn :)) | haha, hôm qua bọn mình đi chơi vui lắm luôn :)) |
| real-40 | trời ơi, mệt quá đi mất huhu | Trời ơi, mệt quá đi mất huhu | trời ơi, mệt quá đi mất huhu |
| real-48 | Tớ vừa thi xong, cảm giác nhẹ nhỏm hẳn hihi | Tớ vừa thi xong, cảm giác nhẹ nhõm hẳn hihi | Tớ vừa thì xong, cảm giác nhẹ nhõm hẳn hihi |
| real-54 | ko bít nữa, để mik hỏi lại đã | Không biết nữa, để mình hỏi lại đã | Không biết nữa, để minh hỏi lại đã |
| real-55 | tối nay đi ăn ko, mik bao | Tối nay đi ăn không, mình bao | Tối nay đi ăn không, mình báo |
| real-61 | mn ơi, mai lớp mình được nghỉ học nha | Mọi người ơi, mai lớp mình được nghỉ học nha | Mọi người ơi, mái lớp mình được nghỉ học nha |
| real-71 | Vì bạn ấy chăm chỉ nhưng bạn ấy đạt điểm cao. | Vì bạn ấy chăm chỉ nên bạn ấy đạt điểm cao. | Vì bạn ấy chăm chỉ nhưng bạn ấy đạt điểm cao. |
| real-73 | Tôi đã đã hoàn thành báo cáo vào tối qua. | Tôi đã hoàn thành báo cáo vào tối qua. | Tôi đã đã hoàn thành báo cáo vào tối qua. |
| real-78 | Nó cười hì hì rồi chạy biến đi mất. | Nó cười hì hì rồi chạy biến đi mất. | Nó cười hi hi rồi chạy biến đi mất. |

Có cả trường hợp phát sinh lỗi từ hợp lệ (thi → thì, bao → báo), bỏ sót sửa quan hệ từ, và giữ chữ thường ở từ biểu cảm trong khi nhãn yêu cầu viết hoa đầu câu. Cần phân biệt sửa sai nội dung với khác quy ước viết hoa khi phân tích Exact Match. Không kết luận LM hiểu ý nghĩa chỉ vì điểm xếp hạng cao.

## Chẩn đoán khả năng khái quát hóa

| Tập | ML cũ WER / Exact Match | Thử nghiệm WER / Exact Match |
|---|---:|---:|
| 400 câu trong tiền tố train detector | 2.77% / 61.50% | 3.03% / 60.50% |
| 800 câu validation | 4.33% / 50.00% | 3.89% / 58.25% |

Các tập khác độ khó và phân phối. Chênh lệch này không chứng minh mô hình học vẹt hoặc không học vẹt. Đợt 1 không train lại nên không có learning curve mới. Cần tập lỗi người thật và ngữ cảnh độc lập ở đợt sau.

| Tập | Số bản ghi | Trùng khóa target với train | Khóa target lặp trong tập |
|---|---:|---:|---:|
| val | 800 | 0 | 0 |
| test | 1653 | 0 | 0 |
| test_real | 95 | 0 | 0 |
| test_vsec | 1000 | 0 | 0 |
| diagnostic | 24 | 0 | 0 |

Does not detect all paraphrases, source-document overlap or pretrained-model contamination.

## Quyết định sau đợt 1

Giữ nhánh mới ở trạng thái opt-in. Không thay pipeline mặc định hoặc checkpoint. Đợt tiếp theo nên phân tích các câu chỉ bản cũ đúng, đánh giá ứng viên nhập nhằng và thêm dữ liệu ngữ cảnh trước khi tích hợp giao diện hoặc huấn luyện lại. Chỉ số tốt trên các tập hiện có chưa đủ để tự động đưa vào sản phẩm.

24 câu diagnostic là bộ phát triển/hồi quy nhỏ, gồm ví dụ đã thảo luận và một số câu quen thuộc trong dự án; không dùng chúng để tuyên bố chất lượng tiếng Việt tổng quát. VSEC không được xuất lại câu ở các artifact mới.

Các tệp máy đọc: `evaluation.json`, `development.json`, `selected_policy.json`, `predictions_*.jsonl` và `original_manifest.json` cùng thư mục.
