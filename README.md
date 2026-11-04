# Hệ thống phát hiện và tự động sửa lỗi chính tả, ngữ pháp tiếng Việt

Đồ án NLP: so sánh 3 hướng tiếp cận sửa lỗi tiếng Việt và một lớp **hạn chế sửa quá mức / bảo toàn văn phong**
dùng chung cho cả ba.

| # | Mô hình | Ý tưởng |
|---|---|---|
| 1 | **ML Baseline** | TF-IDF n-gram ký tự + Logistic Regression phát hiện token lỗi → từ điển + edit distance + bảng nhầm lẫn s/x, ch/tr, d/gi/r sinh ứng viên → bigram LM + beam search chọn từ hợp ngữ cảnh; quy tắc ngữ pháp, viết hoa, dấu câu |
| 2 | **BiLSTM + Attention** | Seq2Seq cấp ký tự (BiLSTM encoder 2 lớp, LSTM decoder, Luong attention), PyTorch, tự huấn luyện |
| 3 | **Pre-trained Transformer** | Checkpoint đã fine-tune `nrl-ai/vn-spell-correction-base` (ViT5-base), chỉ inference |
| + | **Lớp bảo toàn văn phong** | Căn chỉnh câu gốc ↔ câu sửa, xét riêng từng phép sửa: giữ từ biểu cảm/emoticon, không thêm/xoá từ tuỳ tiện, kiểm tra thay từ hợp lệ bằng LM; phép sửa đáng ngờ → chuyển thành gợi ý |

Kết quả thực nghiệm và phân tích: xem [report.md](report.md).

## Extension và ứng dụng iOS/Android

Đã bổ sung `backend/` (FastAPI), `apps/extension/` (Manifest V3) và `apps/mobile/` (Flutter iOS/Android),
cùng ba chế độ **kiểm tra tài liệu**, **hỗ trợ học tiếng Việt** và **chuẩn hóa dữ liệu NLP**.
Mô hình chạy tại API dùng chung; ứng dụng cần kết nối máy chủ.

Hướng dẫn chạy/cài đặt, giới hạn và kiểm thử: [docs/MULTIPLATFORM.md](docs/MULTIPLATFORM.md).

```bash
python -m pip install -r requirements-api.txt
python scripts/run_api.py
```

Extension có thể nạp bằng Load unpacked. App Flutter có sẵn dự án native cho cả iOS và Android.
Kết quả đánh giá lại checkpoint ML hiện tại được lưu riêng ở `outputs/metrics/product_ml_evaluation.json`;
báo cáo bên dưới vẫn mô tả thí nghiệm ban đầu.

## Cấu trúc

```text
configs/config.yaml          cấu hình dữ liệu, siêu tham số
data/                        train/val/test.jsonl (giả lập), test_vsec.jsonl, test_real.jsonl (thủ công)
src/vn_text.py               tiện ích tiếng Việt (tách token, bỏ dấu, viết tắt, từ cảm thán)
src/generate_noise.py        sinh lỗi có kiểm soát
src/prepare_data.py          lọc câu sạch, chia tập chống rò rỉ, sinh dữ liệu
src/baseline_ml.py           Mô hình 1
src/bilstm_attention.py      Mô hình 2
src/pretrained_model.py      Mô hình 3
src/preservation.py          lớp hạn chế sửa quá mức / bảo toàn văn phong
src/evaluate.py              các chỉ số đánh giá
src/inference.py             API + CLI suy luận thống nhất
scripts/train_ml.py, train_bilstm.py, select_transformer.py, tune_preservation.py,
        evaluate_all.py, make_figures.py
app.py                       giao diện Gradio
tests/test_pipeline.py       kiểm thử
outputs/metrics|predictions|figures   kết quả đo thực tế
tier1/, tests/test_tier1.py, data/Viet74K.txt   phiên bản rule-based trước đó (không thuộc pipeline này)
```

### Có sẵn trong repo / cần tạo lại

- **Có sẵn:** dữ liệu đã xử lý (`data/train|val|test|test_real.jsonl`, `data/lm_corpus.txt`), checkpoint
  `models/ml_baseline.pkl`, `models/bilstm_attention.pt`, `models/vocab.txt`, toàn bộ metrics, biểu đồ và
  predictions trên Test giả lập + bộ thủ công. Có thể chạy demo / đánh giá ML và BiLSTM ngay.
- **Không đưa lên:** `data/raw/` (file `nrl_train.jsonl` > 100 MB) và mọi file chứa câu VSEC
  (`data/test_vsec.jsonl`, `outputs/predictions/*__test_vsec.jsonl`) vì giấy phép CC-BY-NC-ND-4.0 của VSEC
  không cho phân phối lại bản đã chỉnh sửa. Tạo lại bằng bước 1–2 bên dưới (seed cố định nên ra đúng dữ liệu
  đã dùng). Checkpoint Transformer tự tải từ Hugging Face.

## Cài đặt

Python 3.10. Trên Windows, nếu thư mục dự án nằm sâu, nên tạo virtualenv ở **đường dẫn ngắn**
(DLL của PyTorch CUDA lỗi `WinError 206` khi đường dẫn quá dài):

```bash
python -m venv C:/venvs/vn_correction
C:/venvs/vn_correction/Scripts/python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
C:/venvs/vn_correction/Scripts/python.exe -m pip install -r requirements.txt
```

Không có GPU: bỏ dòng `--index-url` (cài torch bản CPU). Mọi mô hình đều chạy được trên CPU;
Transformer trên CPU chậm hơn (xem bảng latency trong report). Có thể chạy trên Google Colab bằng
cách upload thư mục và chạy đúng các lệnh dưới đây trong cell `!python ...`.

## Chạy từ đầu đến cuối

Các lệnh chạy từ thư mục gốc dự án, `python` là python của virtualenv.

```bash
# 1. Tải dữ liệu thô (~165 MB)
mkdir -p data/raw
curl -L -o data/raw/nrl_train.jsonl https://huggingface.co/datasets/nrl-ai/vn-spell-correction-train/resolve/main/train.jsonl
curl -L -o data/raw/nrl_val.jsonl https://huggingface.co/datasets/nrl-ai/vn-spell-correction-train/resolve/main/val.jsonl
curl -L -o data/raw/vsec.parquet https://huggingface.co/datasets/nguyenthanhasia/vsec-vietnamese-spell-correction/resolve/main/data/train-00000-of-00001.parquet

# 2. Chuẩn bị dữ liệu (≈1 phút)
python src/prepare_data.py

# 3. Huấn luyện ML Baseline (≈5 phút CPU) -> models/ml_baseline.pkl, models/vocab.txt
python scripts/train_ml.py

# 4. Huấn luyện BiLSTM + Attention (≈45 phút trên RTX 4050; thử nhanh: --epochs 1 --limit 5000)
python scripts/train_bilstm.py

# 5. (Tuỳ chọn) So sánh 2 checkpoint Transformer, tinh chỉnh lớp bảo toàn trên Val
python scripts/select_transformer.py
python scripts/tune_preservation.py

# 6. Đánh giá 3 mô hình trên cùng các tập Test + đo latency, kích thước
python scripts/evaluate_all.py

# 7. Vẽ biểu đồ
python scripts/make_figures.py

# 8. Kiểm thử (RUN_TRANSFORMER=1 để kiểm thử cả Transformer)
python -m pytest tests -q

# 9. Demo: mở http://127.0.0.1:7860
python app.py
```

Checkpoint Transformer (~0.9 GB) tự tải về cache Hugging Face ở lần chạy đầu.

## Dùng từ dòng lệnh

```bash
python src/inference.py --model transformer "hôm nay e đi hoc có vui không, hihi"
python src/inference.py --model ml --no-abbrev "ko bít nữa, để mik hỏi lại đã"
python src/inference.py --model bilstm --no-preserve "Tuy trời mưa nhưng nên tôi vẫn đi học."
```

## Nguồn dữ liệu & mô hình

- Câu sạch: phía `target` của [nrl-ai/vn-spell-correction-train](https://huggingface.co/datasets/nrl-ai/vn-spell-correction-train) (CC-BY-SA-4.0; gốc từ Wikipedia tiếng Việt và tin tức).
- Lỗi thật: [VSEC](https://arxiv.org/abs/2111.00640), bản trên HF `nguyenthanhasia/vsec-vietnamese-spell-correction` (giấy phép gốc CC-BY-NC-ND-4.0) — chỉ dùng để đánh giá, không chỉnh sửa / phân phối lại.
- Transformer: [nrl-ai/vn-spell-correction-base](https://huggingface.co/nrl-ai/vn-spell-correction-base) (Apache-2.0). Ứng viên đã thử: [bmd1905/vietnamese-correction-v2](https://huggingface.co/bmd1905/vietnamese-correction-v2) (Apache-2.0).
