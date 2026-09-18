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
    run_case("Mik trog nhà.", "Mình trong nhà.")            # case cũ, giờ phải giữ hoa "Mình"
    run_case("Hoc bai chua?", "học bai chua?")
    run_case("Tôi bùn wá.", "Tôi buồn quá.")                 # test greedy longest-match
    run_case("Hum ni troi dep.", "Hôm nay troi dep.")        # test cụm 2 từ + giữ hoa "Hum"
    run_case("Anh xang nhà em chơi.", "Anh sang nhà em chơi.")   # x -> s, đúng 1 candidate hợp lệ
    run_case("Anh lói dối em.", "Anh nói dối em.")                # l -> n, chính bug đã fix ở dictionaries.py
    run_case("Anh nói dối em.", "Anh nói dối em.")                 # "nói" đã đúng -> KHÔNG được sửa thành "ngói"