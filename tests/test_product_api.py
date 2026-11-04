import io
import json
import sys
import unicodedata
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import service
from backend.api import app
from backend.datasets import clean_dataset
from backend.schemas import CheckRequest


class FakeModel:
    def correct_batch(self, texts):
        return [text.replace("hoc", "học").replace("toi", "tôi").replace("OpenAI", "OpenAi") for text in texts]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("GRAMMARCHECK_API_KEY", raising=False)
    monkeypatch.setattr(service, "get_model", lambda name: FakeModel())
    with TestClient(app) as client:
        yield client


def test_document_preserves_layout_and_unicode_positions(client):
    text = "🙂 toi đi hoc.\r\n\r\n\ttoi  đi hoc.  "
    response = client.post("/v1/check", json={"text": text})
    assert response.status_code == 200
    body = response.json()
    assert body["output"] == "🙂 tôi đi học.\r\n\r\n\ttôi  đi học.  "
    for edit in body["edits"]:
        assert text[edit["start"]:edit["end"]] == edit["source"]


def test_decomposed_accents_keep_original_offsets(client):
    text = unicodedata.normalize("NFD", "Tôi đi hoc.")
    body = client.post("/v1/check", json={"text": text}).json()
    assert body["output"] == text.replace("hoc", "học")
    assert all(text[e["start"]:e["end"]] == e["source"] for e in body["edits"])


def test_protected_terms_are_kept_even_without_style_filter(client):
    body = client.post("/v1/check", json={"text": "OpenAI hoc.", "preserve_style": False,
                                          "protected_terms": ["OpenAI"]}).json()
    assert body["output"] == "OpenAI học."
    assert any(not edit["accepted"] and "thuật ngữ" in edit["reason"] for edit in body["edits"])


def test_no_content_is_lost_at_long_sentence_boundaries(client):
    text = "  " + "hoc " * 200 + "\n\n" + "x" * 400
    body = client.post("/v1/check", json={"text": text}).json()
    assert body["output"] == text.replace("hoc", "học")
    assert body["chunks"] > 3


@pytest.mark.parametrize("body", [{"text": " "}, {"text": "a" * 30001}, {"text": "x", "model": "invalid"},
                                  {"text": "x", "protected_terms": [""]}])
def test_invalid_requests(client, body):
    assert client.post("/v1/check", json=body).status_code == 422


def test_api_key_and_lessons(client, monkeypatch):
    monkeypatch.setenv("GRAMMARCHECK_API_KEY", "sample-key")
    assert client.get("/v1/lessons").status_code == 401
    response = client.get("/v1/lessons", headers={"X-API-Key": "sample-key"})
    lessons = response.json()["lessons"]
    assert {lesson["category"] for lesson in lessons} >= {"Từ vựng", "Ngữ pháp", "Giao tiếp"}
    assert all(0 <= item["answer"] < len(item["choices"]) for item in lessons)


def test_dataset_keeps_originals_and_labels(client):
    rows = [{"text": "toi đi hoc.", "label": "positive", "id": 1}, {"text": "", "label": "neutral", "id": 2}]
    data = "\n".join(json.dumps(row) for row in rows).encode()
    response = client.post("/v1/datasets/clean", files={"file": ("data.jsonl", data)}, data={"column": "text"})
    assert response.status_code == 200
    output = [json.loads(line) for line in response.json()["content"].splitlines()]
    assert output[0]["text"] == rows[0]["text"] and output[0]["label"] == "positive"
    assert output[0]["_gc_output"] == "tôi đi học."
    assert output[1]["_gc_output"] == ""
    assert output[0]["_gc_options"]["normalize_abbrev"] is False


def test_csv_quoting_and_bad_rows(client):
    data = 'text,label\r\n"toi đi hoc, vui.",yes\r\n'
    response = client.post("/v1/datasets/clean", files={"file": ("data.csv", data.encode())})
    assert response.status_code == 200
    import csv
    rows = list(csv.DictReader(io.StringIO(response.json()["content"].lstrip("\ufeff"))))
    assert rows[0]["text"] == "toi đi hoc, vui." and rows[0]["_gc_output"] == "tôi đi học, vui."
    for content in [b'{"wrong":"x"}', b'{"text":42}', b'{"text":"x", "_gc_output":"old"}']:
        assert client.post("/v1/datasets/clean", files={"file": ("data.jsonl", content)}).status_code == 422


def test_dataset_limits(client):
    data = b'{"text":"x"}\n' * 301
    assert client.post("/v1/datasets/clean", files={"file": ("data.jsonl", data)}).status_code == 422


def test_real_ml_document_smoke(monkeypatch):
    from inference import get_model
    monkeypatch.setattr(service, "get_model", get_model)
    result = service.check_document(CheckRequest(text="Tuy trời mưa nhưng nên tôi vẫn đi học.\n\nKinh tế dang phát triển nhanh."))
    assert "nhưng nên" not in result.output
    assert "đang" in result.output
    assert "\n\n" in result.output
