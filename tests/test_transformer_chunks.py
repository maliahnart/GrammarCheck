import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pretrained_model import TransformerCorrector


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=True):
        return list(range(len(text) + 2))


def test_transformer_long_sentence_and_unbroken_word_are_not_truncated():
    model = TransformerCorrector.__new__(TransformerCorrector)
    model.max_chunk_chars, model.max_length, model.tok = 400, 32, CharacterTokenizer()
    text = "học tiếng Việt " * 30 + "x" * 90
    chunks = model._split(text)
    assert all(len(model.tok.encode(chunk)) <= model.max_length for chunk in chunks)
    assert "".join("".join(chunks).split()) == "".join(text.split())
