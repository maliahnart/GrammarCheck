# GrammarCheck: extension và ứng dụng iOS/Android

## Phạm vi bản đầu tiên

Hai ứng dụng dùng cùng một API và ba mô hình có sẵn. ML là mặc định để chạy bằng checkpoint local; có thể chọn BiLSTM hoặc Transformer. Mô hình chạy trên máy chủ, **không chạy offline trên điện thoại**.

| Chế độ | Extension Chrome/Edge | Flutter iOS/Android |
|---|---|---|
| Tài liệu | Nhập/dán văn bản, mở TXT, lấy đoạn đã chọn trên trang, duyệt sửa, sao chép và thay đoạn chọn | Nhập/dán, mở TXT, duyệt sửa, sao chép/chia sẻ |
| Học tiếng Việt | Phân tích câu, thư viện 6 bài về chính tả/từ vựng/ngữ pháp/giao tiếp, câu hỏi có phản hồi | Cùng thư viện và câu hỏi, phân tích câu và duyệt gợi ý |
| Dữ liệu NLP | Chọn CSV/JSONL, chọn cột, tải bản chuẩn hóa và nhật ký | Chọn CSV/JSONL, chọn cột, lưu/chia sẻ bản kết quả |

Không yêu cầu tự viết bài để dùng thư viện học. Nội dung là bộ bài mẫu biên soạn trong mã nguồn, cần rà soát sư phạm trước khi dùng làm tài liệu giảng dạy. Phản hồi mô hình là gợi ý; không có chức năng chấm năng lực tiếng Việt.

Giới hạn mỗi yêu cầu: 30.000 ký tự/tài liệu; 2 MB, 300 bản ghi và tổng 100.000 ký tự/tệp dữ liệu. Bản này chưa nhập DOCX/PDF, chưa có tài khoản/đồng bộ lịch sử, chưa có hàng đợi bền vững hoặc tiếp tục tác vụ sau khi máy chủ khởi động lại.

## 1. Chạy API

Từ thư mục gốc dự án, sử dụng môi trường Python đã cài các mô hình:

```bash
python -m pip install -r requirements-api.txt
python scripts/run_api.py
```

Mặc định API chỉ lắng nghe tại `http://127.0.0.1:8000`. Tài liệu tương tác ở `http://127.0.0.1:8000/docs`. Health check cũng yêu cầu API key khi key được cấu hình.

Để điện thoại cùng Wi-Fi truy cập máy chủ phát triển:

```bash
export GRAMMARCHECK_API_KEY='thay-bang-khoa-cua-ban'
python scripts/run_api.py --host 0.0.0.0
```

Nhập `http://<IP-LAN-máy-chủ>:8000` và API key trong phần Kết nối của app. `127.0.0.1` trên điện thoại là chính điện thoại. Android Emulator thường dùng `http://10.0.2.2:8000`; iOS Simulator trên cùng Mac dùng `http://127.0.0.1:8000`.

API không lưu nội dung yêu cầu lên đĩa. Dữ liệu được gửi đến máy chủ cấu hình; HTTPS và API key cần được thiết lập khi dùng qua Internet. `GRAMMARCHECK_ORIGINS` là danh sách origin web được phép, phân cách bằng dấu phẩy. CORS không thay thế xác thực.

```bash
curl http://127.0.0.1:8000/v1/check \
  -H 'Content-Type: application/json' \
  -d '{"text":"Kinh tế dang phát triển nhanh.","model":"ml","mode":"document","preserve_style":true,"normalize_abbrev":false,"protected_terms":[]}'
```

Nếu đã cấu hình key, thêm header `X-API-Key`. Transformer tải checkpoint Hugging Face ở lần chạy đầu nếu cache chưa có; không cần để chạy ML/BiLSTM.

## 2. Cài extension để phát triển

1. Mở `chrome://extensions` hoặc `edge://extensions`.
2. Bật Developer mode → Load unpacked → chọn thư mục `apps/extension`.
3. Mở extension → Kết nối & tùy chọn → nhập API và key → Lưu kết nối.
4. Nhập văn bản hoặc chọn đoạn trên trang, mở extension và bấm **Lấy đoạn đã chọn**.
5. Kiểm tra, bật/tắt từng gợi ý rồi sao chép hoặc thay đoạn chọn trong ô nhập liệu.

`python scripts/package_extension.py` tạo `dist/grammarcheck-extension.zip` chỉ gồm tệp chạy. Giải nén rồi Load unpacked thư mục giải nén; không chọn trực tiếp ZIP.

Nút ↗ mở không gian làm việc trong tab riêng, phù hợp tài liệu dài và dữ liệu. Chức năng lấy đoạn chọn dùng trong popup trên tab chứa văn bản. Extension dùng Manifest V3, `activeTab` và `scripting`; không có script tự quét tất cả trang. Quyền kết nối máy chủ khác được yêu cầu khi lưu địa chỉ.

Cũng có thể chọn văn bản rồi nhấp chuột phải → **Kiểm tra tiếng Việt với GrammarCheck** để mở đoạn đó trong tab làm việc. Nội dung chỉ gửi khi bấm kiểm tra.

Chỉ hỗ trợ thay thế trong input text/search, textarea và contenteditable thông thường. Trang nội bộ trình duyệt và trình soạn thảo chuyên biệt như Google Docs có thể không hỗ trợ; có thể dùng sao chép/dán. Thay thế lựa chọn trong contenteditable chèn văn bản thuần. Extension kiểm tra nội dung còn khớp trước khi ghi đè.

Endpoint/tùy chọn lưu trong storage local. API key chỉ giữ trong storage session; nội dung và kết quả không được lưu lịch sử. Đóng popup có thể làm mất bản đang duyệt; dùng tab riêng cho tác vụ dài.

## 3. Chạy Flutter trên iOS/Android

Yêu cầu Flutter tương thích Dart >=3.9.2, Android SDK/JDK; với iOS cần macOS, Xcode và CocoaPods.

```bash
cd apps/mobile
flutter pub get
flutter run --dart-define=API_BASE_URL=http://127.0.0.1:8000
```

Chọn thiết bị bằng `flutter devices` và `flutter run -d <device-id>`. Có thể đổi API/key trong app, không cần sửa mã nguồn. Thiết lập không nhạy cảm được lưu; API key chỉ giữ trong bộ nhớ phiên.

```bash
flutter build apk --debug
flutter build ios --simulator --debug
```

APK: `apps/mobile/build/app/outputs/flutter-apk/app-debug.apk`.
App Simulator: `apps/mobile/build/ios/iphonesimulator/Runner.app`.

Nếu CocoaPods báo lỗi ASCII-8BIT với đường dẫn tiếng Việt, chạy lệnh build với `LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8`.

HTTP LAN chỉ được mở trong cấu hình debug; bản release yêu cầu HTTPS trong client. Android release cần cấu hình khóa ký riêng để phát hành. Với iPhone thật cần chọn Signing Team trong Xcode; dự án không cố định tài khoản Apple. Việc tạo APK/App Simulator không đồng nghĩa đã có bản trên các cửa hàng.

## 4. Hợp đồng API

- `GET /health`: trạng thái API, không tải mô hình.
- `POST /v1/check`: `text`, `model`, `mode`, `preserve_style`, `normalize_abbrev`, `protected_terms`.
- `GET /v1/lessons`: bài học, ví dụ, câu hỏi, đáp án và phản hồi tự học.
- `POST /v1/datasets/clean`: multipart `file`, `column`, tùy chọn giống kiểm tra; `protected_terms` là mảng JSON mã hóa thành chuỗi.

Kết quả kiểm tra gồm bản gốc, bản sửa mặc định, các phép sửa có ID và khoảng `[start, end)`. **Offset tính bằng Unicode code point**, JavaScript dùng `Array.from`, Dart dùng `runes` khi áp dụng. Không dùng trực tiếp UTF-16 substring với offset API. Bỏ chọn toàn bộ phép sửa phải khôi phục bản gốc chính xác.

Văn bản được tách theo câu/đoạn và giới hạn nhỏ trước suy luận; xuống dòng và khoảng trắng ngoài phép sửa được giữ từ bản gốc. Không dùng dấu câu sinh tại ranh giới chia kỹ thuật làm kết thúc câu mới. URL, email, số liệu và thuật ngữ tùy chọn được giữ; điều này cũng có thể chặn một lỗi cần sửa trong các vùng đó.

CSV/JSONL giữ nguyên cột nguồn và nhãn, thêm `_gc_output`, `_gc_edits`, `_gc_model`, `_gc_revision`, `_gc_version`, `_gc_options`. Revision là SHA-256 checkpoint local hoặc commit mô hình Transformer khi có; dòng trống không chạy suy luận được ghi `not_processed`. File đã có các cột này bị từ chối để tránh ghi đè. Không tự kết luận nhãn vẫn đúng sau chuẩn hóa: cần đánh giá tác vụ NLP đích và duyệt mẫu kết quả. Chưa có confidence score đã hiệu chuẩn.

## 5. Kiểm thử và đánh giá

```bash
python -m pytest tests -q
node --test apps/extension/core.test.mjs
cd apps/mobile
flutter analyze
flutter test
```

Kiểm thử extension trong Chromium tách biệt, gọi API ML thật:

```bash
cd apps/extension
npm ci
npx playwright install chromium
npm run test:e2e
```

Kiểm thử tự khởi động API tại cổng 8765 và đóng sau khi hoàn tất.

`RUN_TRANSFORMER=1` chạy thêm kiểm thử Transformer thật (cần checkpoint). Kiểm thử chia đoạn Transformer dùng tokenizer giả riêng để kiểm tra không cắt mất nội dung mà không tải mô hình.

Chạy lại ML hiện tại, lưu tách biệt với báo cáo cũ:

```bash
python scripts/evaluate_product.py --sets test test_real test_vsec
```

Kết quả: `outputs/metrics/product_ml_evaluation.json`, gồm hash checkpoint, pipeline cũ và pipeline tài liệu mới. Cấu hình đánh giá bật mở rộng viết tắt để so sánh với báo cáo; giao diện mặc định tắt tùy chọn này. Không coi các số đo này là kiểm chứng sư phạm, hiệu quả tác vụ NLP phía sau hoặc chất lượng tài liệu nhiều trang.

## Tài liệu nền tảng

- [Chrome: activeTab](https://developer.chrome.com/docs/extensions/develop/concepts/activeTab)
- [Flutter: gửi dữ liệu qua mạng](https://docs.flutter.dev/cookbook/networking/send-data)
- [FastAPI: CORS](https://fastapi.tiangolo.com/tutorial/cors/)
