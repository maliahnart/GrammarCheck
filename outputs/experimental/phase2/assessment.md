# Đánh giá hướng phát triển lõi đợt 2

Đợt 2 bổ sung xếp hạng theo ngữ cảnh bằng TF-IDF từ corpus train và tách quyết định sửa tự động khỏi thứ hạng từ điển. Pipeline cũ và đợt 1 giữ nguyên. Không huấn luyện lại checkpoint ML/BiLSTM/Transformer; không thêm cặp vào train.

## Tính toàn vẹn và lựa chọn tham số

Đã đối chiếu 186 tệp trước đợt 2, tất cả không đổi. Commit gốc của đợt này là `c5578073f23f1a7c722588f68a99f32e0f6197b9`.

Policy được chọn trên dev và cố định trước test: `{'realword_margin': 0.0, 'context_weight': 64.0, 'promotion_margin': 0.01}`. Ngưỡng thay từ dùng 800 câu val cũ; trọng số ngữ cảnh dùng 8 probe dev. Chỉ mục fit trên 211.396 câu corpus train, không dùng nhãn probe.

| Tập validation | WER ↓ | Exact Match ↑ | Edit F0.5 ↑ | Giữ câu đúng ↑ |
|---|---:|---:|---:|---:|
| ML cũ | 4.33% | 50.00% | 70.55% | 73.75% |
| Đợt 2 | 3.88% | 58.25% | 72.20% | 90.00% |

## Hồi quy trên dữ liệu đã có

Các tập này đã được xem trong phát triển trước đây. Đây là phân tích hồi quy khám phá, không phải xác nhận test mù mới. Cột đợt 1 lấy từ artifact cố định cùng checkpoint/dữ liệu; baseline và đợt 2 được chạy lại.

| Tập | Pipeline | WER ↓ | Exact Match ↑ | Edit F0.5 ↑ | Giữ câu đúng ↑ |
|---|---|---:|---:|---:|---:|
| Giả lập 1653 | ML cũ | 4.01% | 51.42% | 71.09% | 78.55% |
| Giả lập 1653 | Đợt 1 | 3.96% | 53.78% | 70.60% | 86.51% |
| Giả lập 1653 | Đợt 2 | 3.96% | 53.72% | 70.57% | 83.74% |
| Thủ công 95 | ML cũ | 2.47% | 72.63% | 78.77% | 77.50% |
| Thủ công 95 | Đợt 1 | 3.49% | 63.16% | 72.77% | 77.50% |
| Thủ công 95 | Đợt 2 | 2.47% | 72.63% | 78.77% | 77.50% |
| VSEC 1000 | ML cũ | 2.05% | 53.80% | 61.12% | 81.00% |
| VSEC 1000 | Đợt 1 | 2.07% | 53.40% | 61.33% | 85.50% |
| VSEC 1000 | Đợt 2 | 2.05% | 53.80% | 61.17% | 81.00% |
| Hồi quy 24 | ML cũ | 3.83% | 75.00% | 58.82% | 64.71% |
| Hồi quy 24 | Đợt 1 | 3.28% | 79.17% | 66.67% | 70.59% |
| Hồi quy 24 | Đợt 2 | 2.73% | 79.17% | 66.67% | 70.59% |

### Thay đổi so với đợt 1

- Giả lập 1653: WER +0.00 điểm phần trăm; Exact Match -0.06; edit F0.5 -0.03.
- VSEC 1000: WER -0.02 điểm phần trăm; Exact Match +0.40; edit F0.5 -0.15.
- Thủ công 95: WER -1.02 điểm phần trăm; Exact Match +9.47; edit F0.5 +6.00.

## Xếp hạng với ngữ cảnh

20 probe tổng hợp được soạn riêng: 8 dev thuộc 4 nhóm, 12 test thuộc 6 nhóm không giao nhau. Đây chưa phải dữ liệu lỗi thật gán nhãn độc lập. Các cặp giữ cùng câu hiện tại nhưng đổi ngữ cảnh và đáp án.

| Phép đo | Không ngữ cảnh | Có ngữ cảnh |
|---|---:|---:|
| Xếp hạng trong 2 lựa chọn cho sẵn, test 12 câu | 50.00% | 58.33% |
| Cặp tương phản đúng cả hai phía | 0/6 | 1/6 |
| Sửa đầu cuối, không đưa lựa chọn cho mô hình | 8.33% | 25.00% |

Phép chấm lựa chọn cho sẵn chỉ kiểm tra bộ xếp hạng và không được gọi là độ chính xác sửa lỗi đầu cuối. Sửa đầu cuối còn phụ thuộc vào việc tự sinh đúng ứng viên và các bước bảo toàn. Điểm TF-IDF là liên kết từ vựng, không phải xác suất hiểu đúng ý định.

| ID test | Đáp án | Không ngữ cảnh | Có ngữ cảnh | Ngữ cảnh được tự áp dụng |
|---|---|---|---|---|
| test-thu-letter | Tôi xem thư. | Tôi xem thử. | Tôi xem thư. | True |
| test-thu-animal | Tôi xem thú. | Tôi xem thử. | Tôi xem thử. | False |
| test-can-scale | Tôi mua cân. | Tôi mua căn. | Tôi mua căn. | False |
| test-can-container | Tôi mua can. | Tôi mua căn. | Tôi mua căn. | False |
| test-dau-oil | Tôi cần dầu. | Tôi cần đầu. | Tới cần dầu. | True |
| test-dau-berry | Tôi cần dâu. | Tôi cần đầu. | Tới cần dầu. | True |
| test-da-stone | Đây là đá. | Đây là đá. | Đây là đá. | False |
| test-da-leather | Đây là da. | Đây là đá. | Đây là đá. | False |
| test-ma-code | Tôi xem mã. | Tôi xem ma. | Tôi xem mã. | True |
| test-ma-seedling | Tôi xem mạ. | Tôi xem ma. | Tôi xem mà. | True |
| test-vai-role | Tôi chọn vai. | Tôi cho vai. | Tôi cho vai. | False |
| test-vai-fabric | Tôi chọn vải. | Tôi cho vai. | Tôi cho vai. | False |

## Hồi quy đã biết từ đợt 1

Các trường hợp này đã được dùng phân tích lỗi để thiết kế đợt 2; không dùng chúng làm bằng chứng độc lập về khái quát hóa.

| ID thủ công | Đợt 1 | Đợt 2 | Đáp án |
|---|---|---|---|
| real-11 | Vì thời tiết xấu nhưng chuyến bay bị hoãn lại. | Vì thời tiết xấu nên chuyến bay bị hoãn lại. | Vì thời tiết xấu nên chuyến bay bị hoãn lại. |
| real-36 | haha, hôm qua bọn mình đi chơi vui lắm luôn :)) | Haha, hôm qua bọn mình đi chơi vui lắm luôn :)) | Haha, hôm qua bọn mình đi chơi vui lắm luôn :)) |
| real-40 | trời ơi, mệt quá đi mất huhu | Trời ơi, mệt quá đi mất huhu | Trời ơi, mệt quá đi mất huhu |
| real-48 | Tớ vừa thì xong, cảm giác nhẹ nhõm hẳn hihi | Tớ vừa thi xong, cảm giác nhẹ nhõm hẳn hihi | Tớ vừa thi xong, cảm giác nhẹ nhõm hẳn hihi |
| real-54 | Không biết nữa, để minh hỏi lại đã | Không biết nữa, để mình hỏi lại đã | Không biết nữa, để mình hỏi lại đã |
| real-55 | Tối nay đi ăn không, mình báo | Tối nay đi ăn không, mình bao | Tối nay đi ăn không, mình bao |
| real-61 | Mọi người ơi, mái lớp mình được nghỉ học nha | Mọi người ơi, mai lớp mình được nghỉ học nha | Mọi người ơi, mai lớp mình được nghỉ học nha |
| real-71 | Vì bạn ấy chăm chỉ nhưng bạn ấy đạt điểm cao. | Vì bạn ấy chăm chỉ nên bạn ấy đạt điểm cao. | Vì bạn ấy chăm chỉ nên bạn ấy đạt điểm cao. |
| real-73 | Tôi đã đã hoàn thành báo cáo vào tối qua. | Tôi đã hoàn thành báo cáo vào tối qua. | Tôi đã hoàn thành báo cáo vào tối qua. |
| real-78 | Nó cười hi hi rồi chạy biến đi mất. | Nó cười hì hì rồi chạy biến đi mất. | Nó cười hì hì rồi chạy biến đi mất. |

## Giới hạn và quyết định

Giữ đợt 2 ở chế độ opt-in. Kết quả tốt hơn trên bộ đã biết không chứng minh hệ thống hiểu ngữ cảnh nói chung hoặc không học vẹt. Ngữ cảnh chỉ được fit từ corpus train, nhưng các trọng số và quy tắc vẫn có thể thiên lệch theo dev. Bộ ngữ cảnh nhỏ, tổng hợp và cùng người triển khai soạn; cần dữ liệu tài liệu thật được gán nhãn độc lập để xác nhận.

Tự sửa không ngữ cảnh vẫn có thể giữ các lỗi của baseline. Tự phục hồi dấu theo ngữ cảnh có thể sai khi hàng xóm không liên quan hoặc retrieval thiếu bằng chứng. Mọi lựa chọn trả về giữ vùng bảo vệ và vị trí gốc, nhưng chưa được bảo đảm đúng ngữ nghĩa.

Chưa có n-best BiLSTM/ViT5, Top-3 theo vùng trong ứng dụng, learning curve mới hoặc kiểm định thống kê. Các tệp đánh giá máy đọc, policy, kế hoạch dữ liệu và manifest nằm cùng thư mục; cache TF-IDF được tạo lại bằng `--stage build` và không đưa vào commit.

## Kiểm thử

54 kiểm thử Python đạt, 1 bỏ qua (Transformer thật chưa được bật). Đã chạy CLI đợt 1 và đợt 2, xác minh CLI từ chối policy sai checkpoint, và kiểm tra baseline chấm lại trùng các chỉ số baseline đã lưu ở đợt 1. Chi tiết ở `verification.json`.
