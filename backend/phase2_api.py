"""Original GrammarCheck API with an opt-in context suggestion endpoint."""

import json
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from backend.api import app as original_app, authorize
from backend.phase2_phrases import phrase_suggestions
from backend.schemas import Change
from backend.service import MODEL_LOCK
from scripts.evaluate_phase2 import CONFIG, verify_frozen
from experimental_phase2.engine import Phase2Corrector, Phase2Policy


app = FastAPI(title="GrammarCheck Đợt 2", dependencies=[Depends(authorize)])


class Phase2Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=2000)
    previous: str = Field(default="", max_length=4000)
    following: str = Field(default="", max_length=4000)
    normalize_abbrev: bool = False
    protected_terms: list[str] = Field(default_factory=list, max_length=100)


@lru_cache(maxsize=1)
def load_experiment():
    snapshot = verify_frozen(json.loads(CONFIG.read_text()))
    return Phase2Corrector.from_project(), Phase2Policy(**snapshot["policy"])


@app.post("/v1/experimental/phase2", response_model=dict[str, list[Change]])
def suggest(request: Phase2Request):
    try:
        edits = phrase_suggestions(
            request.text, previous=request.previous, following=request.following,
            protected_terms=request.protected_terms,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not (request.previous.strip() or request.following.strip()):
        return {"edits": edits}
    with MODEL_LOCK:
        try:
            engine, policy = load_experiment()
        except (OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(503, "Không tải được cấu hình đợt 2 đã đánh giá.") from exc
        try:
            result = engine.correct(**request.model_dump(), policy=policy)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    if not result["context_promoted"]:
        return {"edits": edits}
    # The frozen engine tags edits whose accent change earned context support.
    edits.extend(
        Change(
            id=f"phase2-{edit['start']}-{edit['end']}",
            start=edit["start"], end=edit["end"], source=edit["source"],
            replacement=edit["replacement"], kind=edit["kind"],
            explanation="Gợi ý theo ngữ cảnh · Đợt 2",
            accepted=False, reason=edit["reason"],
        )
        for edit in result["edits"]
        if edit["accepted"] and not edit["locked"]
        and edit["reason"] == "Phương án phục hồi dấu được hỗ trợ bởi ngữ cảnh lân cận; cần đọc lại."
        and not any(existing.start < edit["end"] and edit["start"] < existing.end for existing in edits)
    )
    return {"edits": edits}


app.mount("/", original_app)
