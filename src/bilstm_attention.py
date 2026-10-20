"""Mô hình 2 - Seq2Seq cấp ký tự: BiLSTM Encoder + LSTM Decoder + Luong Attention (PyTorch).

Làm việc ở cấp ký tự để xử lý được cả từ gõ sai chưa từng thấy (không có vấn đề OOV).
Huấn luyện bằng teacher forcing; suy luận bằng greedy decoding theo lô.
"""
from __future__ import annotations

import json
import functools
import math
import random
import re
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from vn_text import nfc

PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]


class CharVocab:
    def __init__(self, chars: list[str]):
        self.itos = SPECIALS + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def build(cls, texts, min_freq: int = 3):
        cnt = Counter(c for t in texts for c in t)
        return cls(sorted(c for c, n in cnt.items() if n >= min_freq))

    def encode(self, s: str) -> list[int]:
        return [self.stoi.get(c, UNK) for c in s]

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            if i == EOS:
                break
            if i > UNK:
                out.append(self.itos[i])
        return "".join(out)

    def __len__(self):
        return len(self.itos)


class Seq2SeqAttn(nn.Module):
    def __init__(self, vocab_size: int, emb_dim=128, hidden_dim=256, enc_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        dec_dim = 2 * hidden_dim
        self.src_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=enc_layers, bidirectional=True,
                               batch_first=True, dropout=dropout if enc_layers > 1 else 0.0)
        self.init_h = nn.Linear(2 * hidden_dim, dec_dim)
        self.init_c = nn.Linear(2 * hidden_dim, dec_dim)
        self.decoder = nn.LSTM(emb_dim, dec_dim, num_layers=1, batch_first=True)
        self.attn_W = nn.Linear(2 * hidden_dim, dec_dim, bias=False)  # Luong "general"
        self.combine = nn.Linear(dec_dim + 2 * hidden_dim, dec_dim)
        self.out = nn.Linear(dec_dim, vocab_size)
        self.drop = nn.Dropout(dropout)

    def encode(self, src, src_len):
        emb = self.drop(self.src_emb(src))
        packed = nn.utils.rnn.pack_padded_sequence(emb, src_len.cpu(), batch_first=True, enforce_sorted=False)
        out, (h, c) = self.encoder(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=src.size(1))
        # Ghép trạng thái cuối 2 chiều của lớp trên cùng -> trạng thái đầu cho decoder.
        h_last = torch.cat([h[-2], h[-1]], dim=-1)
        c_last = torch.cat([c[-2], c[-1]], dim=-1)
        state = (torch.tanh(self.init_h(h_last)).unsqueeze(0), self.init_c(c_last).unsqueeze(0))
        return out, self.attn_W(out), state

    def attend(self, dec_out, enc_out, enc_keys, src_mask):
        scores = torch.bmm(dec_out, enc_keys.transpose(1, 2))  # (B, T, S)
        scores = scores.masked_fill(~src_mask.unsqueeze(1), -1e4)
        attn = F.softmax(scores, dim=-1)
        ctx = torch.bmm(attn, enc_out)
        h = torch.tanh(self.combine(torch.cat([dec_out, ctx], dim=-1)))
        return self.out(self.drop(h)), attn

    def forward(self, src, src_len, tgt_in):
        enc_out, enc_keys, state = self.encode(src, src_len)
        dec_out, _ = self.decoder(self.drop(self.tgt_emb(tgt_in)), state)
        logits, _ = self.attend(dec_out, enc_out, enc_keys, src != PAD)
        return logits

    @torch.no_grad()
    def greedy(self, src, src_len, max_len: int):
        enc_out, enc_keys, state = self.encode(src, src_len)
        mask = src != PAD
        B = src.size(0)
        tok = torch.full((B, 1), SOS, dtype=torch.long, device=src.device)
        outs = []
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            dec_out, state = self.decoder(self.tgt_emb(tok), state)
            logits, _ = self.attend(dec_out, enc_out, enc_keys, mask)
            tok = logits.argmax(-1)
            outs.append(tok)
            done |= tok.squeeze(1) == EOS
            if done.all():
                break
        return torch.cat(outs, dim=1)


# ---------------------------------------------------------------------------
def _pad(seqs, value=PAD):
    L = max(len(s) for s in seqs)
    return torch.tensor([s + [value] * (L - len(s)) for s in seqs], dtype=torch.long)


def make_batches(pairs, vocab, batch_size, shuffle=True, rng=None):
    """Gom các câu có độ dài gần nhau vào cùng lô để giảm padding."""
    idx = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]))
    batches = [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]
    if shuffle:
        (rng or random).shuffle(batches)
    for b in batches:
        src = [vocab.encode(pairs[i][0]) for i in b]
        tgt = [vocab.encode(pairs[i][1]) for i in b]
        yield (_pad(src), torch.tensor([len(s) for s in src]), _pad([[SOS] + t for t in tgt]),
               _pad([t + [EOS] for t in tgt]))


_CHUNK_RE = re.compile(r"(?<=[.!?])\s+")


def _chunks(text: str, max_len: int) -> list[str]:
    """Cắt văn bản dài thành các đoạn <= max_len ký tự (theo câu, rồi theo từ)."""
    out = []
    for sent in _CHUNK_RE.split(text):
        cur = ""
        for w in sent.split(" "):
            if cur and len(cur) + 1 + len(w) > max_len:
                out.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}" if cur else w
        if cur:
            out.append(cur)
    return out or [""]


class BiLSTMCorrector:
    def __init__(self, model: Seq2SeqAttn, vocab: CharVocab, max_len: int, device: str):
        self.model, self.vocab, self.max_len, self.device = model.to(device).eval(), vocab, max_len, device

    @classmethod
    def load(cls, path, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(path, map_location=device, weights_only=False)
        vocab = CharVocab(ck["chars"])
        model = Seq2SeqAttn(len(vocab), **ck["hparams"])
        model.load_state_dict(ck["state_dict"])
        return cls(model, vocab, ck["max_len"], device)

    def correct_batch(self, texts: list[str], batch_size: int = 64) -> list[str]:
        pieces, owner = [], []
        for k, t in enumerate(texts):
            for c in _chunks(nfc(t).strip(), self.max_len):
                pieces.append(c)
                owner.append(k)
        outs = [""] * len(pieces)
        order = sorted(range(len(pieces)), key=lambda i: len(pieces[i]))
        for s in range(0, len(order), batch_size):
            ids = order[s:s + batch_size]
            enc = [self.vocab.encode(pieces[i]) or [UNK] for i in ids]
            src = _pad(enc).to(self.device)
            lens = torch.tensor([len(e) for e in enc])
            gen = self.model.greedy(src, lens, max_len=int(lens.max()) + 20)
            for i, row in zip(ids, gen.tolist()):
                outs[i] = self.vocab.decode(row)
        res = [[] for _ in texts]
        for o, k in zip(outs, owner):
            res[k].append(o)
        return [" ".join(r).strip() for r in res]

    def correct(self, text: str) -> str:
        return self.correct_batch([text])[0]


# ---------------------------------------------------------------------------
def train(train_pairs, val_pairs, cfg: dict, out_path, device=None, log_fn=functools.partial(print, flush=True),
          seed=42):
    from evaluate import compute_metrics

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    max_len = cfg["max_len"]
    train_pairs = [(a, b) for a, b in train_pairs if len(a) <= max_len and len(b) <= max_len]
    vocab = CharVocab.build([a for a, _ in train_pairs] + [b for _, b in train_pairs])
    hp = {k: cfg[k] for k in ("emb_dim", "hidden_dim", "enc_layers", "dropout")}
    model = Seq2SeqAttn(len(vocab), **hp).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log_fn(f"Vocab ký tự: {len(vocab)}, tham số: {n_params / 1e6:.2f}M, thiết bị: {device}, "
           f"số cặp train: {len(train_pairs)}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    steps_per_epoch = math.ceil(len(train_pairs) / cfg["batch_size"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps_per_epoch * cfg["epochs"],
                                                pct_start=0.1)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history, best_cer = [], float("inf")
    val_records = [{"input": a, "target": b} for a, b in val_pairs]
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        t0, tot, ntok = time.time(), 0.0, 0
        for src, src_len, tin, tout in make_batches(train_pairs, vocab, cfg["batch_size"], rng=rng):
            src, tin, tout = src.to(device), tin.to(device), tout.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(src, src_len, tin)
                loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tout.reshape(-1),
                                       ignore_index=PAD)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            n = (tout != PAD).sum().item()
            tot += loss.item() * n
            ntok += n
        corr = BiLSTMCorrector(model, vocab, max_len, device)
        preds = corr.correct_batch([a for a, _ in val_pairs])
        model.train()
        m = compute_metrics(val_records, preds)
        rec = {"epoch": ep, "train_loss": tot / ntok, "val_cer": m["cer_after"], "val_wer": m["wer_after"],
               "val_exact": m["exact_match"], "seconds": time.time() - t0}
        history.append(rec)
        log_fn(json.dumps(rec))
        if m["cer_after"] < best_cer:
            best_cer = m["cer_after"]
            torch.save({"state_dict": model.state_dict(), "chars": vocab.itos[len(SPECIALS):], "hparams": hp,
                        "max_len": max_len, "epoch": ep, "n_params": n_params}, out_path)
    return history
