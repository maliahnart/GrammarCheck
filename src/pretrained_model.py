"""Mô hình 3 - Transformer seq2seq đã fine-tune sẵn cho sửa lỗi tiếng Việt (chỉ inference).

Mặc định: nrl-ai/vn-spell-correction-base (ViT5-base / T5, Apache-2.0) - chọn qua thí nghiệm.
Dự phòng:  bmd1905/vietnamese-correction-v2 (BARTpho-syllable / MBart, Apache-2.0).
Chạy được cả GPU (fp16) lẫn CPU (fp32). Checkpoint tự tải từ Hugging Face về cache lần đầu.
"""
from __future__ import annotations

import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from vn_text import nfc

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


class TransformerCorrector:
    def __init__(self, model_name: str = "nrl-ai/vn-spell-correction-base", device: str | None = None,
                 max_length: int = 256, num_beams: int = 1, max_chunk_chars: int = 400):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device).eval()
        self.max_length, self.num_beams, self.max_chunk_chars = max_length, num_beams, max_chunk_chars

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def _split(self, text: str) -> list[str]:
        if len(text) <= self.max_chunk_chars:
            return [text]
        return [s for s in _SENT_RE.split(text) if s]

    @torch.no_grad()
    def correct_batch(self, texts: list[str], batch_size: int = 16) -> list[str]:
        pieces, owner = [], []
        for k, t in enumerate(texts):
            for c in self._split(nfc(t).strip()):
                pieces.append(c)
                owner.append(k)
        outs = [""] * len(pieces)
        order = sorted(range(len(pieces)), key=lambda i: len(pieces[i]))
        for s in range(0, len(order), batch_size):
            ids = order[s:s + batch_size]
            enc = self.tok([pieces[i] for i in ids], return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_length).to(self.device)
            gen = self.model.generate(**enc, max_length=self.max_length, num_beams=self.num_beams)
            for i, g in zip(ids, self.tok.batch_decode(gen, skip_special_tokens=True)):
                outs[i] = g.strip()
        res = [[] for _ in texts]
        for o, k in zip(outs, owner):
            res[k].append(o)
        return [" ".join(r) for r in res]

    def correct(self, text: str) -> str:
        return self.correct_batch([text])[0]
