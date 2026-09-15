"""
Tầng 1 — Dữ liệu: teencode dictionary, confusion set, telex mapping.
Tách riêng khỏi engine.py để dễ mở rộng mà không đụng logic xử lý.
"""

# Teencode phổ biến -> dạng chuẩn.
# Key luôn viết thường vì lúc so khớp trong engine sẽ lowercase câu input trước.
TEENCODE_MAP = {
    "hum ni": "hôm nay", "hnay": "hôm nay",
    "iu": "yêu", "iuu": "yêu",
    "bùn": "buồn", "bùn wá": "buồn quá",
    "dc": "được", "ko": "không", "k": "không", "hok": "không",
    "j": "gì", "z": "vậy", "vs": "với",
    "wa": "quá", "zui": "vui", "lun": "luôn",
    "mik": "mình", "trog": "trong", "đag": "đang",
    "đt": "điện thoại", "ntn": "như thế nào",
    "bh": "bây giờ", "ns": "nói",
}

# Confusion set âm đầu — mỗi key ánh xạ tới DANH SÁCH các âm dễ nhầm.
# Đây là bản đã fix bug: CONFUSION_INITIAL gốc trong SRS ban đầu có
# key 'n' bị lặp 2 lần (dict literal ghi đè), khiến lỗi nhầm l/n
# không bao giờ được phát hiện. Bản dưới gộp đúng cả 2 giá trị.
CONFUSION_INITIAL = {
    's': ['x'], 'x': ['s'],
    'ch': ['tr'], 'tr': ['ch'],
    'd': ['gi', 'r'], 'gi': ['d', 'r'], 'r': ['d', 'gi'],
    'l': ['n'], 'n': ['l', 'ng'], 'ng': ['n'],
}

# Lỗi gõ Telex/VNI phổ biến -> từ đúng.
# Key là từ ĐÚNG, value là danh sách các cách gõ sai thường gặp —
# thiết kế ngược thế này vì lúc tra cứu, ta có từ nghi ngờ (vd "khoong")
# và cần biết nó là lỗi gõ của từ nào; nhưng khi XÂY dữ liệu, liệt kê
# theo "1 từ đúng - nhiều lỗi gõ" tự nhiên và ít trùng lặp hơn.
TELEX_ERRORS = {
    'không': ['khoong', 'khongr', 'khongx', 'khongj', 'khongz'],
    'muốn': ['muoon', 'muonr', 'muonx', 'muonj'],
    'được': ['duowjc', 'duocj', 'dươcj', 'duocx'],
    'học': ['hoc', 'hocr', 'hocj'],
}