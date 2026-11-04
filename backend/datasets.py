"""In-memory CSV/JSONL normalization, retaining source columns and edit provenance."""
import csv
import io
import json

from backend.schemas import CheckRequest
from backend.service import check_document

MAX_BYTES = 2 * 1024 * 1024
MAX_ROWS = 300
MAX_CHARS = 100000
RESERVED = {"_gc_output", "_gc_edits", "_gc_model", "_gc_revision", "_gc_version", "_gc_options"}


def clean_dataset(content, filename, column, options):
    if len(content) > MAX_BYTES:
        raise ValueError("Tệp vượt quá 2 MB.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Tệp cần dùng mã hóa UTF-8.") from exc
    kind = filename.rsplit(".", 1)[-1].lower()
    if kind == "csv":
        reader = csv.DictReader(io.StringIO(text, newline=""))
        fields = reader.fieldnames
        if not fields or len(set(fields)) != len(fields):
            raise ValueError("CSV cần tiêu đề cột duy nhất.")
        if column not in fields:
            raise ValueError(f"Không tìm thấy cột '{column}'.")
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Số trường CSV không khớp tiêu đề.")
            rows.append(row)
            if len(rows) > MAX_ROWS:
                raise ValueError(f"Mỗi lần xử lý tối đa {MAX_ROWS} bản ghi.")
    elif kind == "jsonl":
        rows = []
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON không hợp lệ tại dòng {number}.") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Dòng {number} cần là một đối tượng JSON.")
            rows.append(row)
            if len(rows) > MAX_ROWS:
                raise ValueError(f"Mỗi lần xử lý tối đa {MAX_ROWS} bản ghi.")
        fields = None
    else:
        raise ValueError("Chỉ hỗ trợ CSV và JSONL.")
    if not rows:
        raise ValueError("Tệp không có bản ghi.")
    total = 0
    # Validate everything before inference; never silently skip malformed records.
    for index, row in enumerate(rows, 1):
        if RESERVED.intersection(row):
            raise ValueError("Tệp đã có cột _gc_ của lần xử lý trước. Hãy dùng lại bản gốc.")
        if column not in row or not isinstance(row[column], str):
            raise ValueError(f"Bản ghi {index}: cột '{column}' cần là chuỗi văn bản.")
        if len(row[column]) > 30000:
            raise ValueError(f"Bản ghi {index} vượt quá 30.000 ký tự.")
        total += len(row[column])
    if total > MAX_CHARS:
        raise ValueError("Tổng văn bản vượt quá 100.000 ký tự; hãy chia tệp nhỏ hơn.")
    changed = 0
    for row in rows:
        source = row[column]
        if source.strip():
            result = check_document(CheckRequest(text=source, mode="dataset", **options))
            output, edits = result.output, [edit.model_dump() for edit in result.edits]
            revision = result.model_revision
        else:
            output, edits, revision = source, [], "not_processed"
        changed += output != source
        row.update(_gc_output=output, _gc_edits=edits, _gc_model=options["model"],
                   _gc_revision=revision, _gc_version="1.0.0", _gc_options=dict(options))
    if kind == "csv":
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=fields + sorted(RESERVED))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "_gc_edits": json.dumps(row["_gc_edits"], ensure_ascii=False),
                             "_gc_options": json.dumps(row["_gc_options"], ensure_ascii=False)})
        output = "\ufeff" + buffer.getvalue()
    else:
        output = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    return {"filename": f"normalized.{kind}", "content": output, "rows": len(rows), "changed_rows": changed,
            "content_type": "text/csv" if kind == "csv" else "application/x-ndjson"}
