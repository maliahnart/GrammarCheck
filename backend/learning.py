"""Small, explicit learning library. Model predictions are suggestions, not lessons."""

LESSONS = [
    {
        "id": "diacritic", "title": "Dấu và nghĩa của từ", "category": "Chính tả",
        "explanation": "Dấu thanh và dấu của nguyên âm có thể làm thay đổi nghĩa. Cần đọc cả câu để chọn từ phù hợp; một chuỗi không dấu có thể có nhiều cách hiểu.",
        "examples": ["Tôi đi hoc. → Tôi đi học.", "‘ma’, ‘má’, ‘mà’, ‘mả’, ‘mã’, ‘mạ’ có nghĩa khác nhau."],
        "question": "Chọn từ phù hợp: ‘Em đang … bài.’", "choices": ["học", "hộc", "hóc"],
        "answer": 0, "feedback": "‘Học bài’ là hoạt động tiếp thu, ôn lại kiến thức.",
    },
    {
        "id": "phonetic", "title": "Phụ âm dễ nhầm", "category": "Chính tả",
        "explanation": "Một số cách phát âm vùng miền khiến s/x, ch/tr, d/gi/r dễ bị nhầm khi viết. Không thể thay đồng loạt các phụ âm; cần đối chiếu từ và ngữ cảnh.",
        "examples": ["xử lý công việc", "sử dụng máy tính", "chia sẻ niềm vui"],
        "question": "Chọn cách viết phù hợp: ‘… công việc.’", "choices": ["Sử lý", "Xử lý", "Xữ lý"],
        "answer": 1, "feedback": "Cách viết đúng trong cụm này là ‘xử lý’.",
    },
    {
        "id": "vocabulary", "title": "Phân biệt ‘dành’ và ‘giành’", "category": "Từ vựng",
        "explanation": "‘Dành’ thường chỉ để riêng cho ai hoặc việc gì. ‘Giành’ thường chỉ cố gắng đạt được, lấy về qua cạnh tranh. Hai từ đều tồn tại; lựa chọn phụ thuộc ngữ cảnh.",
        "examples": ["Tôi dành thời gian đọc sách.", "Đội tuyển giành chiến thắng."],
        "question": "Chọn từ: ‘Đội tuyển … chiến thắng.’", "choices": ["dành", "giành"],
        "answer": 1, "feedback": "Dùng ‘giành chiến thắng’; còn ‘dành thời gian’ là để thời gian cho một việc.",
    },
    {
        "id": "grammar", "title": "Quan hệ nguyên nhân – kết quả", "category": "Ngữ pháp",
        "explanation": "Cặp ‘vì … nên …’ có thể biểu thị nguyên nhân và kết quả. Cần xác định quan hệ ý nghĩa trước khi chọn từ nối; gợi ý tự động về từ nối luôn cần được đọc lại.",
        "examples": ["Vì trời mưa nên đường trơn.", "Tuy trời mưa nhưng tôi vẫn đi học."],
        "question": "Điền từ: ‘Vì chăm chỉ … Lan tiến bộ.’", "choices": ["nhưng", "nên", "hoặc"],
        "answer": 1, "feedback": "‘Nên’ nối kết quả ‘Lan tiến bộ’ với nguyên nhân ‘chăm chỉ’.",
    },
    {
        "id": "abbreviation", "title": "Viết tắt và hoàn cảnh giao tiếp", "category": "Giao tiếp",
        "explanation": "‘ko’, ‘mik’, ‘dc’ thường xuất hiện trong tin nhắn. Trong bài học, báo cáo hoặc thư trang trọng, thường nên viết đầy đủ. Viết tắt trong hội thoại không mặc nhiên là lỗi.",
        "examples": ["ko → không", "mik → mình", "dc → được (cần kiểm tra ngữ cảnh)"],
        "question": "Cách viết nào phù hợp hơn với báo cáo?", "choices": ["mik ko đồng ý", "Tôi không đồng ý."],
        "answer": 1, "feedback": "Cách viết đầy đủ phù hợp hơn với văn bản trang trọng.",
    },
    {
        "id": "casing", "title": "Viết hoa đầu câu", "category": "Chính tả",
        "explanation": "Trong văn bản thông thường, chữ cái mở đầu câu được viết hoa. Tên riêng có quy tắc riêng; hệ thống có thể chưa nhận diện đầy đủ tên riêng mới.",
        "examples": ["hôm nay trời đẹp. → Hôm nay trời đẹp."],
        "question": "Chọn cách viết đầu câu phù hợp.", "choices": ["hôm nay tôi đi học.", "Hôm nay tôi đi học."],
        "answer": 1, "feedback": "Viết hoa chữ H mở đầu câu.",
    },
]


def lesson_for(kind):
    return kind if any(item["id"] == kind for item in LESSONS) else None
