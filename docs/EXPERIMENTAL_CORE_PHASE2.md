# Hướng phát triển lõi GrammarCheck đợt 2

Đợt 2 tiếp tục nhánh ML thử nghiệm, khắc phục cách chọn kết quả ở đợt 1 và bổ sung xếp hạng bằng ngữ cảnh câu trước/sau. Mã suy luận chính, ứng dụng, checkpoint ML/BiLSTM, dữ liệu gốc và toàn bộ mã/kết quả đợt 1 được giữ nguyên. Module mới nằm ở `src/experimental_phase2/` để không ảnh hưởng kiểm tra hash của đợt 1.

## Hành vi mới

1. Khi không có ngữ cảnh, đầu ra tự động bắt đầu từ các phép sửa ML đã qua bảo toàn. Một phương án từ điển được xếp cao hơn không tự thay thế phương án này. Các phép sửa từ hợp lệ vẫn chịu ngưỡng chọn trên validation.
2. Sửa quan hệ từ và xóa từ thừa đã được lớp bảo toàn cũ chấp nhận không bị loại chỉ vì điểm LM của cả câu. Xóa từ được ghi rõ là sửa ngữ pháp khi đã qua quy tắc hiện có; không mở quyền xóa từ bất kỳ.
3. Cho phép viết hoa từ biểu cảm đầu câu khi mô hình đề xuất, nhưng không vượt qua thuật ngữ bảo vệ do người dùng chỉ định. Biểu cảm ở các vị trí khác vẫn được giữ.
4. Câu trước và câu sau được dùng làm truy vấn vào chỉ mục TF-IDF từ corpus train. Các câu train gần truy vấn tạo thành biểu diễn ngữ cảnh để xếp hạng các giả thuyết. Không có bảng ánh xạ từ khóa thể thao/y tế viết tay.
5. Khi câu hiện tại ít dấu và có hỗ trợ ngữ cảnh, bộ tìm kiếm mở thêm biến thể dấu. Tự động đổi phương án chỉ khi có mức tăng hỗ trợ ngữ cảnh; từng phép đổi phải có hỗ trợ riêng so với phương án ML. Các sửa ML không liên quan được giữ lại. Đây là heuristic, không phải bảo đảm xác suất đúng.
6. Mọi giả thuyết trả về đều đã qua kiểm tra bảo vệ và có danh sách edit với vị trí gốc để áp dụng nguyên giả thuyết. Không trộn edit giữa hai giả thuyết. Các phương án đã qua bảo vệ vẫn có thể sai về ý nghĩa.

`output` là kết quả tự động. `best_candidate` là phương án được xếp hạng cao sau các kiểm tra; nó có thể chưa được tự động áp dụng. `context_promoted` cho biết ngữ cảnh có thay đầu ra ML hay không. Điểm `uncalibrated_lexical_ranking` không phải xác suất đúng.

## Giới hạn ngữ cảnh

Đây là bộ xếp hạng liên kết từ vựng, không phải mô hình hiểu ngữ nghĩa tổng quát. Chỉ mục được fit trên 211.396 câu trong `data/lm_corpus.txt`, không đọc nhãn validation hoặc test. Bộ phát hiện lỗi ML và checkpoint không được huấn luyện lại. TF-IDF học thống kê từ/cụm từ corpus train, còn trọng số trộn ngữ cảnh được chọn trên dev.

Người gọi phải cung cấp câu trước/sau khi có sẵn. Không có câu tiếp theo thì để trống; hệ thống không bịa thêm ngữ cảnh. Dòng CSV độc lập không được coi dòng bên cạnh là câu liên quan. Lệnh mới chưa tự chia tài liệu dài hoặc gắn vào extension/mobile.

## Chạy riêng

Từ thư mục dự án:

```bash
# Chỉ cần tạo lại cache nếu chưa có. Cache khoảng 25 MB không đưa vào commit.
.venv/bin/python scripts/evaluate_phase2.py --stage build

# Chạy bình thường, không dùng ngữ cảnh
.venv/bin/python scripts/run_phase2.py "Tôi đã đã hoàn thành báo cáo."

# Xếp hạng có xét câu trước/sau
.venv/bin/python scripts/run_phase2.py "Toi muon ban them." \
  --previous "Cửa hàng đang thanh lý nhiều sản phẩm." \
  --next "Khách hàng hỏi giá trước khi mua."

# Thuật ngữ và tùy chọn viết tắt giữ cách dùng riêng
.venv/bin/python scripts/run_phase2.py "OpenAI hoc." --protect OpenAI
.venv/bin/python scripts/run_phase2.py "Mình ko đi." --no-abbrev
```

`--no-context` tắt hoàn toàn việc dùng câu lân cận. Không cần cache để chạy khi không dùng ngữ cảnh. Các giới hạn vẫn là tối đa 2.000 ký tự/256 token cho câu hiện tại và 4.000 ký tự cho mỗi câu lân cận; vượt giới hạn bị từ chối, không cắt âm thầm.

Policy được lưu ở `outputs/experimental/phase2/selected_policy.json`. CLI kiểm tra hash mã, corpus, checkpoint và wordlist; khi dùng ngữ cảnh kiểm tra cả cache. Cache tạo lại có thể khác khi đổi môi trường thư viện; trong trường hợp đó cần chọn lại policy và đánh giá, không tự gán số liệu cũ cho cache mới.

## Đánh giá và dữ liệu mới

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/evaluate_phase2.py --stage dev
.venv/bin/python scripts/evaluate_phase2.py --stage test
.venv/bin/python scripts/report_phase2.py
```

- 800 câu validation cũ chọn ngưỡng thay từ hợp lệ. Không dùng kết quả test để chọn lại.
- 8 câu ngữ cảnh dev thuộc 4 nhóm nhập nhằng chọn trọng số ngữ cảnh.
- 12 câu ngữ cảnh test thuộc 6 nhóm khác được giữ riêng để chấm sau khi cố định cấu hình. ID tài liệu, nhóm nhập nhằng và câu đầu vào không giao với dev. Mỗi nhóm có cùng câu hiện tại nhưng hai ngữ cảnh hướng đến hai đáp án khác nhau.
- Tổng cộng thêm 20 câu probe đánh giá, không thêm vào train. Chúng do người triển khai soạn và là câu tổng hợp ngắn, không phải tài liệu thật do người gán nhãn độc lập. Sự tách nhóm không tự biến bộ nhỏ này thành bằng chứng chất lượng sản phẩm rộng rãi.
- Các lựa chọn `choices` chỉ dùng trong phép chấm riêng bộ xếp hạng, không truyền cho pipeline suy luận đầu cuối. Phải đọc hai kết quả riêng: xếp hạng trong tập ứng viên cho sẵn và sửa lỗi đầu cuối tự sinh ứng viên.
- Test giả lập 1.653 câu, VSEC 1.000 câu, bộ thủ công 95 câu và 24 câu hồi quy đợt 1 tiếp tục dùng làm hồi quy đã biết. Chúng đã được xem ở các đợt trước, không gọi là test mù mới.

`dataset_plan.json` chốt checksum các probe trước khi tạo chỉ mục. `previous_manifest.json` bảo vệ 186 tệp trước đợt 2. `selected_policy.json` cố định mã/cấu hình/cache trước khi chạy test. Nội dung câu VSEC không được xuất lại trong artifact mới. So sánh đợt 1 sử dụng bảng đã lưu, chỉ sau khi xác nhận checkpoint và dữ liệu tương ứng giữ nguyên.

## Học vẹt và các bước chưa làm

Không có trọng số mô hình sửa lỗi nào được học lại, nhưng các quy tắc, trọng số và dữ liệu tổng hợp vẫn có thể thiên lệch theo dev. Các kiểm thử xác minh truy vấn ngữ cảnh không fit lại vectorizer và việc đổi ngữ cảnh có thể đổi thứ hạng trong corpus thử có kiểm soát. Đây là kiểm thử chức năng; hiệu quả trên corpus thật nằm ở báo cáo đánh giá riêng.

Chưa có learning curve mới, kiểm định thống kê hoặc kết luận đã loại bỏ học vẹt. Dữ liệu đối chiếu nhỏ không đánh giá hết hội thoại, tên riêng, từ ngoại ngữ, phủ định hoặc câu cần suy luận. Chưa nâng BiLSTM/ViT5 lên n-best, chưa thêm Top-3 theo vùng trong ứng dụng và chưa thay mặc định API. Những phần đó giữ cho các đợt tiếp theo sau khi xem kết quả đợt 2.
