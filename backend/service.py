"""Document-safe inference: sentence boundaries, original offsets and reversible edits."""
import hashlib
import re
import sys
import threading
import time
import unicodedata
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference import get_model  # noqa: E402
from preservation import preserve  # noqa: E402

from backend.learning import lesson_for
from backend.schemas import Change, CheckRequest, CheckResponse

# Cache/model execution is serialized because the model instances are shared.
MODEL_LOCK = threading.Lock()
PROTECTED = re.compile(r"https?://\S+|www\.\S+|[\w.+-]+@[\w.-]+\.[\w-]+|\b\d[\d.,:/%+-]*\b")


@lru_cache(maxsize=8)
def checkpoint_hash(path, modified, size):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_revision(name, model):
    if name in {"ml", "bilstm"}:
        path = ROOT / "models" / ("ml_baseline.pkl" if name == "ml" else "bilstm_attention.pt")
        stat = path.stat()
        return "sha256:" + checkpoint_hash(str(path), stat.st_mtime_ns, stat.st_size)
    config = getattr(getattr(model, "model", None), "config", None)
    return getattr(config, "_commit_hash", None) or "unknown"


def document_spans(text, limit=140):
    """Partition all source characters, preserving every whitespace separator."""
    spans = []
    # Paragraph breaks and sentence-ending whitespace remain outside inference.
    for match in re.finditer(r"[^\r\n]+", text):
        line_start = match.start()
        for sentence in re.finditer(r"\S[\s\S]*?(?:[.!?](?=\s|$)|$)", match.group()):
            start, end = line_start + sentence.start(), line_start + sentence.end()
            while start < end:
                while start < end and text[start].isspace():
                    start += 1
                if start == end:
                    break
                stop = min(start + limit, end)
                if stop < end:
                    boundary = max(text.rfind(" ", start, stop + 1), text.rfind("\t", start, stop + 1))
                    if boundary > start:
                        stop = boundary
                while stop > start and text[stop - 1].isspace():
                    stop -= 1
                if stop == start:
                    break
                spans.append((start, stop))
                start = stop
    return spans


def apply_changes(text, changes):
    """Offsets refer to the original Python Unicode string, not a mutated string."""
    output, cursor = [], 0
    for edit in sorted(changes, key=lambda item: (item.start, item.end)):
        if not edit.accepted:
            continue
        if edit.start < cursor or text[edit.start:edit.end] != edit.source:
            raise ValueError("Các phép sửa chồng lấn hoặc không khớp văn bản gốc.")
        output.extend((text[cursor:edit.start], edit.replacement))
        cursor = edit.end
    output.append(text[cursor:])
    return "".join(output)


def normalize_with_offsets(source):
    """Map NFC boundaries back to the exact source, including decomposed accents."""
    normalized = unicodedata.normalize("NFC", source)
    boundaries = {0: 0}
    for index in range(1, len(source) + 1):
        boundaries[len(unicodedata.normalize("NFC", source[:index]))] = index
    return normalized, boundaries


def check_document(request: CheckRequest):
    text = request.text
    if not text.strip():
        raise ValueError("Vui lòng nhập văn bản có nội dung.")
    if any(not term.strip() or len(term) > 100 for term in request.protected_terms):
        raise ValueError("Mỗi thuật ngữ bảo vệ cần có từ 1 đến 100 ký tự.")
    started = time.perf_counter()
    spans = document_spans(text)
    protected = [(m.start(), m.end()) for m in PROTECTED.finditer(text)]
    for term in request.protected_terms:
        protected.extend((m.start(), m.end()) for m in re.finditer(re.escape(term), text, re.IGNORECASE))
    changes = []
    with MODEL_LOCK:
        model = get_model(request.model)
        revision = model_revision(request.model, model)
        # Bound intermediate batches rather than retaining a whole document on GPU.
        for offset in range(0, len(spans), 32):
            group = spans[offset:offset + 32]
            normalized = [normalize_with_offsets(text[start:end]) for start, end in group]
            sources = [item[0] for item in normalized]
            predictions = model.correct_batch(sources)
            if len(predictions) != len(group):
                raise RuntimeError("Mô hình trả về thiếu đoạn văn bản.")
            for (start, end), source, prediction, (_, boundaries) in zip(group, sources, predictions, normalized):
                result = preserve(source, prediction, normalize_abbrev=request.normalize_abbrev,
                                  strict=request.preserve_style)
                for edit in result["edits"]:
                    left, right = start + boundaries[edit.start], start + boundaries[edit.end]
                    replacement = edit.replacement
                    if not replacement and right > left:
                        # Removing a word also removes its trailing separator, not a newline.
                        while right < end and text[right] in " \t":
                            right += 1
                    if left == right and replacement:
                        if left and text[left - 1].isalnum() and replacement[0].isalnum():
                            replacement = " " + replacement
                        if right < len(text) and text[right].isalnum() and replacement[-1].isalnum():
                            replacement += " "
                    accepted, reason = edit.accepted, edit.reason
                    if any((left < b and right > a) or (left == right and a < left < b) for a, b in protected):
                        accepted, reason = False, "Giữ nguyên thuật ngữ, đường dẫn hoặc số liệu."
                    # Avoid inventing sentence boundaries at an artificial word split.
                    prefix = text[:start]
                    continuation = bool(prefix.strip()) and prefix.rstrip()[-1:] not in ".!?" and not prefix.rstrip(" \t").endswith(("\r", "\n"))
                    if edit.kind == "casing" and edit.start == 0 and continuation:
                        accepted, reason = False, "Đầu đoạn chia kỹ thuật, không phải đầu câu."
                    if edit.kind == "punctuation" and edit.start == len(source) and end < len(text) and text[end] not in "\r\n":
                        accepted, reason = False, "Không thêm dấu câu tại ranh giới chia đoạn."
                    changes.append(Change(
                        id=f"edit-{len(changes)}", start=left, end=right, source=text[left:right],
                        replacement=replacement, kind=edit.kind, explanation=edit.explanation,
                        accepted=accepted, reason=reason, lesson_id=lesson_for(edit.kind),
                    ))
    return CheckResponse(input=text, output=apply_changes(text, changes), edits=changes,
                         model=request.model, model_revision=revision, mode=request.mode, seconds=time.perf_counter() - started,
                         chunks=len(spans))
