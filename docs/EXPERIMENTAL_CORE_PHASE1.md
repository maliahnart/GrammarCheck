# Hướng phát triển lõi GrammarCheck đợt 1

Đợt này tạo một nhánh ML thử nghiệm độc lập để kiểm tra việc sinh nhiều phương án bằng Viet74K và hạn chế sửa thiếu chắc chắn. Pipeline, API, ứng dụng, checkpoint, tập dữ liệu gốc và các bảng kết quả lịch sử không được sửa. Đây chưa phải thay thế mặc định cho sản phẩm.

## Phạm vi đã triển khai

- Chỉ mục từ và cụm Viet74K theo khóa bỏ dấu, xét tối đa 4 âm tiết.
- Tìm kiếm beam có giới hạn; các đường giữ nguyên và các cách ghép nhiều mục từ cùng cạnh tranh. Một cụm không cần tồn tại nguyên vẹn trong từ điển.
- Dùng câu gốc và phương án ML làm hai giả thuyết khởi đầu. Không chốt trước một “mỏ neo chắc chắn 99%”.
- Xếp hạng bằng bigram LM đã có, chi phí thay đổi và tín hiệu cụm từ. Điểm là điểm xếp hạng chưa hiệu chuẩn, không phải xác suất đúng.
- Tỷ lệ dấu chỉ là tín hiệu. Câu ít dấu nhưng toàn các mục từ quen thuộc không tự động mở khóa toàn câu.
- Giữ bản gốc, trả tối đa 3 giả thuyết toàn câu khác nhau sau loại trùng về chữ hoa và khoảng trắng. Bản gốc vẫn có riêng ở `keep_original`.
- Căn chỉnh đầu ra bằng lớp bảo toàn hiện có; thêm ngưỡng chênh lệch xếp hạng và điểm ngữ cảnh khi thay một từ hợp lệ bằng từ hợp lệ khác.
- Bảo vệ URL, email, hashtag/mã, số liệu, biểu cảm và thuật ngữ người dùng chỉ định. Không khẳng định đã nhận diện đúng mọi tên riêng hoặc từ tiếng Anh.
- Áp dụng phép sửa lên offset gốc bằng Unicode code point, giữ phần văn bản ngoài phép sửa, kể cả Unicode NFD, CRLF và tab.

## Những phần dành cho các đợt sau

Chưa triển khai mô hình xếp hạng ngữ nghĩa liên câu, n-best của BiLSTM/ViT5, Top-3 theo vùng sửa trong giao diện, tự sửa kéo dài ký tự, hay huấn luyện thêm. Ba phương án toàn câu không được tự ý tách rồi ghép các edit của chúng với nhau. `best_candidate` có thể khác `output`: trường thứ hai chỉ áp dụng các edit qua kiểm tra. Phương án được xếp cao không mặc nhiên được tự sửa.

Các chuỗi trong `candidates` là giả thuyết trước quyết định bảo toàn, có thể chứa phép sửa bị chặn, kể cả trong vùng bảo vệ. Không áp dụng nguyên chuỗi này lên tài liệu. Khi tích hợp lựa chọn người dùng ở đợt sau, mỗi giả thuyết cần đi qua cùng bước kiểm tra và ánh xạ offset. Hiện chưa nối các chuỗi đó vào giao diện.

Viet74K không có tần suất theo miền hoặc nhãn lỗi–đúng. Nó mở rộng tập ứng viên, không bảo đảm tìm được ý định của người viết. Ví dụ “trong tai bien” có thể vẫn xếp sai dù xuất hiện thêm phương án thể thao. Giai đoạn hiện tại không có bộ phân xử liên câu để giải quyết chắc chắn trường hợp đó.

## Cách chạy riêng

Chạy từ thư mục gốc, dùng môi trường Python đang có trong dự án:

```bash
.venv/bin/python scripts/run_experimental.py "Kinh tế viet nam dang phat trien nhanh."
.venv/bin/python scripts/run_experimental.py "OpenAI hoc." --protect OpenAI
.venv/bin/python scripts/run_experimental.py "Mình ko đi học hôm nay." --no-abbrev
```

Lệnh cũ `python src/inference.py ...`, Gradio và API không chuyển sang nhánh này. CLI mới dùng policy đã chọn trên validation ở `outputs/experimental/phase1/selected_policy.json`; khi tệp chưa có, đầu ra ghi rõ dùng mặc định chưa chọn trên dev.

Đợt 1 nhận tối đa 2.000 ký tự và 256 token mỗi yêu cầu, từ chối rõ ràng đầu vào dài hơn. Nó chưa tự chia tài liệu dài hoặc đọc câu trước/sau. Không có tải checkpoint mới hay gọi dịch vụ bên ngoài.

## Quy trình đánh giá

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/evaluate_experimental.py --stage dev
.venv/bin/python scripts/evaluate_experimental.py --stage test
.venv/bin/python scripts/report_experimental.py
```

1. `original_manifest.json` lưu checksum của 78 tệp có sẵn trước khi làm đợt 1. Script từ chối tiếp tục khi các tệp đó đã thay đổi, kể cả checkpoint và dữ liệu. Không cập nhật manifest để che một thay đổi.
2. Lưới ngưỡng được định nghĩa trong `configs/experimental_phase1.json` trước khi test. Chỉ 800 câu validation dùng để chọn theo edit F0.5, rồi tỷ lệ giữ câu đúng và WER. Ngưỡng được đóng băng cùng hash mã, cấu hình, checkpoint và wordlist.
3. Một mẫu 400 câu từ chính tiền tố dữ liệu mà Logistic Regression đã dùng được chấm mô tả. Nó không dùng chọn ngưỡng, không phải thí nghiệm learning curve.
4. Chạy lại baseline ML hiện tại và nhánh mới trên cùng 1.653 câu giả lập, 1.000 câu VSEC, 95 câu thủ công. Không so checkpoint mới với chỉ số lịch sử khác phiên bản.
5. Ablation `no_lexicon` bỏ cả ứng viên từ Viet74K và điểm thưởng cụm. Báo cả policy riêng được chọn trên dev và cùng policy với bản đầy đủ.
6. 24 câu ở `data/experimental/phase1_diagnostic.jsonl` là hồi quy do người triển khai soạn, có những ví dụ đã thảo luận/đã xuất hiện trong quá trình phát triển. Không gọi đây là bộ test độc lập. Chúng không được thêm vào train.
7. Kết quả ghi riêng dưới `outputs/experimental/phase1/`. Với VSEC, chỉ xuất ID và số liệu kiểm tra; không xuất lại nội dung câu hoặc bản sửa.

Các tham số không được thay sau khi đọc kết quả test trong cùng một lượt xác nhận. Sửa lỗi kỹ thuật sau test phải ghi rõ lý do và coi lần chạy tiếp theo là phân tích tiếp diễn, không phải test mù mới.

## Cách đọc kết quả

- WER/CER/Exact Match và edit F0.5 dùng lại hàm chấm của dự án để so sánh nhất quán. Token chấm không phải từ tiếng Việt đã được phân đoạn theo ngôn ngữ học.
- `clean_kept_rate` đo sửa thừa trên câu đúng. Đầu ra còn được kiểm tra không áp dụng edit vào vùng bảo vệ.
- `candidate_pool_gold_rate` và `top3_gold_rate` đo có cả câu đáp án trong danh sách; không phải độ chính xác sửa lỗi theo vùng hoặc xác suất người dùng chọn đúng.
- `raw_exact_match` khắt khe cả về khoảng trắng; Exact Match chính dùng chuẩn hóa như pipeline cũ. Kiểm thử riêng xác minh định dạng ngoài edit.
- Tỷ lệ chấp nhận edit là mức độ áp dụng đề xuất, không phải tỷ lệ bao phủ tất cả lỗi thật. Edit precision/recall mới phản ánh sự trùng khớp với nhãn.
- Thời gian là một lượt CPU tuần tự, gồm gọi ML, tìm kiếm và căn chỉnh; chưa phải tải sản phẩm đồng thời hoặc benchmark nhiều lượt. Số latency lịch sử không được dùng làm đối chứng tốc độ trực tiếp.

## Học vẹt và giới hạn xác nhận

Không có trọng số nào được học lại ở đợt này, nhưng các heuristic và ngưỡng vẫn có thể quá phù hợp với validation. Kiểm tra trùng khóa chỉ bắt một số dạng rò rỉ; nó không loại trừ câu diễn đạt tương tự, trùng bài nguồn hoặc việc mô hình có sẵn từng thấy dữ liệu. Test hiện tại đã được xem trong các lần phát triển trước; VSEC và bộ thủ công từng tham gia lựa chọn checkpoint Transformer. Do đó kết quả là so sánh khám phá trên bộ hiện có.

Mẫu train 400 câu khác độ khó và phân phối với validation, nên khoảng cách chỉ số không chứng minh nguyên nhân là học vẹt. Đợt sau cần dữ liệu ngữ cảnh độc lập, các mẫu lỗi chưa dùng xây quy tắc và bộ test mới chưa dùng chọn tham số. Giữ dữ liệu cũ làm hồi quy, không trộn test vào train.

## Các đợt dự kiến tiếp theo

| Đợt | Công việc | Điều kiện đánh giá |
|---|---|---|
| 2 | Phân tích lỗi đợt 1; bổ sung bộ ngữ cảnh có ID tài liệu; thiết kế xếp hạng liên câu | Giữ test mới độc lập, đo thay đổi đáp án theo ngữ cảnh |
| 3 | Adapter nhiều phương án cho ViT5 rồi BiLSTM | So lợi ích Top-3 với độ trễ, không so trực tiếp điểm thô giữa mô hình |
| 4 | Top-3 theo vùng và tích hợp API/ứng dụng bằng chế độ riêng | Kiểm tra edit phụ thuộc nhau, lựa chọn gốc, định dạng và khả năng hoàn tác |

Các đợt này là lộ trình, chưa được đánh dấu hoàn thành trong đợt 1.
