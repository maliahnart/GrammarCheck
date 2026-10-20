"""Sinh lỗi giả lập có kiểm soát cho câu tiếng Việt sạch.

Mỗi hàm lỗi nhận một âm tiết (hoặc cả câu) và trả về dạng bị lỗi. Hàm
`make_noisy` kết hợp ngẫu nhiên các loại lỗi và trả về (câu lỗi, danh sách loại lỗi).

Loại lỗi:
  diacritic    - thiếu/sai dấu thanh, mất dấu mũ/móc (học -> hoc, hỏi -> hói)
  typo         - gõ nhầm phím, thiếu/thừa/đảo ký tự (học -> hco)
  phonetic     - nhầm s/x, ch/tr, d/gi/r, l/n, c/t, n/ng (giành -> dành)
  abbreviation - viết tắt (không -> ko, được -> dc)
  grammar      - lặp hư từ, quan hệ từ thừa/sai (nhưng -> nhưng nên)
  casing       - mất viết hoa đầu câu
  punctuation  - mất dấu câu cuối câu
"""
from __future__ import annotations

import random
import re
import unicodedata

from vn_text import ABBREV_NOISE, DUPLICABLE_FUNCTION_WORDS, TONE_MARKS, VOWEL_MODS, match_case, nfc, \
    remove_diacritics

KEYBOARD_NEIGHBORS = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh", "u": "yij", "i": "uok",
    "o": "ipl", "p": "ol", "a": "qsz", "s": "adwx", "d": "sfec", "f": "dgr", "g": "fht", "h": "gjy",
    "j": "hku", "k": "jli", "l": "kp", "z": "asx", "x": "zsc", "c": "xdv", "v": "cfb", "b": "vgn",
    "n": "bhm", "m": "nj",
}
INITIAL_CONFUSIONS = [("tr", "ch"), ("ch", "tr"), ("gi", "d"), ("d", "gi"), ("r", "d"), ("d", "r"),
                      ("r", "gi"), ("s", "x"), ("x", "s"), ("l", "n"), ("n", "l")]
FINAL_CONFUSIONS = [("ng", "n"), ("n", "ng"), ("c", "t"), ("t", "c"), ("nh", "n")]
TONES = sorted(TONE_MARKS)


def _split_affix(tok: str) -> tuple[str, str, str]:
    m = re.match(r"^(\W*)(.*?)(\W*)$", tok, re.UNICODE)
    return m.group(1), m.group(2), m.group(3)


# --------------------------- lỗi cấp âm tiết -------------------------------
def noise_diacritic(syl: str, rng: random.Random) -> str:
    nfd = unicodedata.normalize("NFD", syl)
    tones = [c for c in nfd if c in TONE_MARKS]
    mods = [c for c in nfd if c in VOWEL_MODS]
    choice = rng.random()
    if choice < 0.45:  # bỏ toàn bộ dấu
        return remove_diacritics(syl) if syl.lower() != "đ" else syl
    if choice < 0.75 and tones:  # đổi dấu thanh
        new = rng.choice([t for t in TONES if t != tones[0]])
        return nfc(nfd.replace(tones[0], new))
    if choice < 0.9 and tones:  # bỏ dấu thanh
        return nfc(nfd.replace(tones[0], ""))
    if mods:  # bỏ dấu mũ/móc/trăng
        return nfc(nfd.replace(mods[0], "", 1))
    return remove_diacritics(syl)


def noise_typo(syl: str, rng: random.Random) -> str:
    if len(syl) < 2:
        return syl
    chars = list(syl)
    i = rng.randrange(len(chars))
    op = rng.random()
    base = remove_diacritics(chars[i]).lower()
    if op < 0.3 and i < len(chars) - 1:  # đảo hai ký tự
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    elif op < 0.55 and len(chars) > 2:  # thiếu ký tự
        del chars[i]
    elif op < 0.8 and base in KEYBOARD_NEIGHBORS:  # gõ nhầm phím bên cạnh
        chars[i] = rng.choice(KEYBOARD_NEIGHBORS[base])
    else:  # thừa ký tự (lặp hoặc phím cạnh)
        extra = rng.choice(KEYBOARD_NEIGHBORS.get(base, base) + chars[i])
        chars.insert(i, extra)
    return "".join(chars)


def noise_phonetic(syl: str, rng: random.Random) -> str | None:
    low = syl.lower()
    options = []
    for a, b in INITIAL_CONFUSIONS:
        if low.startswith(a) and not (a == "g" and low.startswith("gi")):
            if a == "d" and low.startswith("đ"):
                continue
            options.append(b + low[len(a):])
    for a, b in FINAL_CONFUSIONS:
        if low.endswith(a) and len(low) > len(a) + 1 and not (a == "n" and low.endswith(("ng", "nh"))):
            options.append(low[: -len(a)] + b)
    if not options:
        return None
    return match_case(syl, rng.choice(options))


def noise_abbreviation(syl: str, rng: random.Random) -> str | None:
    forms = ABBREV_NOISE.get(syl.lower())
    if not forms:
        return None
    return match_case(syl, rng.choice(forms)) if syl[0].isupper() else rng.choice(forms)


# --------------------------- lỗi cấp câu -----------------------------------
def noise_grammar(words: list[str], rng: random.Random) -> list[str] | None:
    """Tạo lỗi ngữ pháp có quy tắc rõ ràng. Trả về None nếu câu không áp dụng được."""
    lows = [w.lower() for w in words]
    options = []
    if "nhưng" in lows:
        options.append("redundant")
    if "vì" in lows[:-3] and "nên" in lows:
        options.append("wrong_conj")
    dup_idx = [i for i, w in enumerate(lows) if w in DUPLICABLE_FUNCTION_WORDS]
    if dup_idx:
        options.append("dup")
    if not options:
        return None
    kind = rng.choice(options)
    out = list(words)
    if kind == "redundant":  # Tuy ... nhưng tôi -> Tuy ... nhưng nên tôi
        i = lows.index("nhưng")
        out.insert(i + 1, "nên")
    elif kind == "wrong_conj":  # Vì ... nên -> Vì ... nhưng
        start = lows.index("vì") + 1
        if "nên" not in lows[start:]:
            return None
        i = lows.index("nên", start)
        out[i] = match_case(words[i], "nhưng")
    else:  # của -> của của
        i = rng.choice(dup_idx)
        out.insert(i + 1, lows[i])
    return out


SYLLABLE_NOISES = {
    "diacritic": (noise_diacritic, 0.45),
    "typo": (noise_typo, 0.25),
    "phonetic": (noise_phonetic, 0.18),
    "abbreviation": (noise_abbreviation, 0.12),
}


def make_noisy(sentence: str, rng: random.Random, style: str = "formal") -> tuple[str, list[str]]:
    """Sinh một biến thể lỗi của câu sạch. Trả về (câu lỗi, các loại lỗi đã chèn).

    Nếu sau vài lần thử câu vẫn không đổi (câu quá ngắn / không có âm tiết phù hợp)
    thì trả về (câu gốc, []) và phía gọi coi đó là câu đúng."""
    for _ in range(5):
        text, types = _make_noisy_once(sentence, rng, style)
        if text != sentence:
            return text, types
    return sentence, []


def _make_noisy_once(sentence: str, rng: random.Random, style: str) -> tuple[str, list[str]]:
    tokens = sentence.split(" ")
    types: list[str] = []

    # 1) Khoảng 12% câu bị bỏ dấu toàn bộ (kiểu gõ không dấu: "hom nay e di hoc").
    if rng.random() < 0.12:
        tokens = [remove_diacritics(t) for t in tokens]
        types.append("diacritic")
        if style == "informal" or rng.random() < 0.3:
            tokens, _ = _apply_abbrev(tokens, rng, prob=0.6)
    else:
        # 2) Lỗi ngữ pháp (nếu áp dụng được) với xác suất 12%.
        if rng.random() < 0.12:
            g = noise_grammar(tokens, rng)
            if g is not None:
                tokens = g
                types.append("grammar")
        # 3) 1-3 lỗi âm tiết.
        n_err = rng.choices([1, 2, 3], weights=[0.5, 0.33, 0.17])[0]
        idxs = [i for i, t in enumerate(tokens) if re.search(r"[^\W\d_]{1,}", _split_affix(t)[1])]
        rng.shuffle(idxs)
        done = 0
        names = list(SYLLABLE_NOISES)
        weights = [SYLLABLE_NOISES[n][1] for n in names]
        if style == "informal":
            weights[names.index("abbreviation")] = 0.45
        for i in idxs:
            if done >= n_err:
                break
            pre, core, post = _split_affix(tokens[i])
            for _ in range(3):
                name = rng.choices(names, weights=weights)[0]
                new = SYLLABLE_NOISES[name][0](core, rng)
                if new and new != core:
                    tokens[i] = pre + new + post
                    types.append(name)
                    done += 1
                    break
    # 4) Viết hoa & dấu câu.
    text = " ".join(tokens)
    if text and text[0].isupper() and rng.random() < 0.25:
        text = text[0].lower() + text[1:]
        types.append("casing")
    if text and text[-1] in ".!?" and rng.random() < 0.2:
        text = text[:-1].rstrip()
        types.append("punctuation")
    return text, sorted(set(types), key=types.index)


def _apply_abbrev(tokens, rng, prob):
    out, changed = [], False
    for t in tokens:
        pre, core, post = _split_affix(t)
        # Sau khi bỏ dấu, tra từ điển viết tắt theo dạng không dấu không còn chính xác, nên chỉ
        # áp dụng cho các âm tiết không dấu có dạng viết tắt rõ ràng.
        cands = {"khong": "ko", "duoc": "dc", "minh": "mk", "em": "e", "gi": "j", "voi": "vs",
                 "roi": "r", "cung": "cx", "biet": "bik", "nguoi": "ng"}
        if core.lower() in cands and rng.random() < prob:
            out.append(pre + cands[core.lower()] + post)
            changed = True
        else:
            out.append(t)
    return out, changed


if __name__ == "__main__":
    r = random.Random(0)
    for s in ["Hôm nay em đi học có vui không?", "Tuy trời mưa nhưng tôi vẫn đi học.",
              "Đội tuyển Việt Nam đã giành chiến thắng thuyết phục.", "Vì trời mưa to nên trận đấu bị hoãn lại."]:
        for _ in range(3):
            print(make_noisy(s, r), "<-", s)
