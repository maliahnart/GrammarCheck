import sys
import unicodedata
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.phase2_api import app


def suggest(text, **options):
    with TestClient(app) as client:
        response = client.post("/v1/experimental/phase2", json={"text": text, **options})
    assert response.status_code == 200, response.text
    return response.json()["edits"]


def test_football_phrases_work_in_another_sentence(monkeypatch):
    monkeypatch.delenv("GRAMMARCHECK_API_KEY", raising=False)
    edits = suggest("Sau cu sut sat vach voi, trong tai bien phat co bao viet vi.")
    assert {edit["replacement"] for edit in edits} == {
        "vạch vôi", "trọng tài biên", "phất cờ", "việt vị",
    }
    assert all(not edit["accepted"] for edit in edits)


def test_phrases_keep_unicode_offsets_and_respect_protection(monkeypatch):
    monkeypatch.delenv("GRAMMARCHECK_API_KEY", raising=False)
    text = "🙂 " + unicodedata.normalize("NFD", "Bóng") + "  da qua vach voi sau cu sut."
    [edit] = suggest(text)
    assert text[edit["start"]:edit["end"]] == edit["source"]
    assert edit["replacement"] == "Bóng  đã qua vạch vôi"
    assert suggest(text, protected_terms=["vach voi"]) == []


def test_non_football_and_already_accented_phrases_stay_unchanged(monkeypatch):
    monkeypatch.delenv("GRAMMARCHECK_API_KEY", raising=False)
    assert suggest("Can cap cuu ngay trong tai bien mach mau nao.") == []
    assert suggest("Trọng tài biên phất cờ báo việt vị.") == []
    assert suggest("Cầu thủ theo dõi bóng đá qua từng thời kỳ.") == []
