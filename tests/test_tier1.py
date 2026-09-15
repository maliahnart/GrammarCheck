import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tier1.engine import correct_sentence


def run_case(text, expected_corrected):
    result = correct_sentence(text)
    status = "PASS" if result["corrected_text"] == expected_corrected else "FAIL"
    print(f"[{status}] input   = {text!r}")
    print(f"        expected = {expected_corrected!r}")
    print(f"        got      = {result['corrected_text']!r}")
    print(f"        errors   = {result['errors']}")
    print()


if __name__ == "__main__":
    run_case("Hôm nay tôi ko đi học.", "Hôm nay tôi không đi học.")
    run_case("Mik trog nhà.", "mình trong nhà.")  # lưu ý: mất hoa đầu câu — giới hạn đã nói ở trên
    run_case("Hoc bai chua?", "học bai chua?")     # "bai", "chua" chưa có trong dict -> giữ nguyên