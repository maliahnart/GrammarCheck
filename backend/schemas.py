from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=30000)
    model: Literal["ml", "bilstm", "transformer"] = "ml"
    mode: Literal["document", "learn", "dataset"] = "document"
    preserve_style: bool = True
    normalize_abbrev: bool = False
    protected_terms: list[str] = Field(default_factory=list, max_length=100)


class Change(BaseModel):
    id: str
    start: int
    end: int
    source: str
    replacement: str
    kind: str
    explanation: str
    accepted: bool
    reason: str
    lesson_id: str | None = None


class CheckResponse(BaseModel):
    input: str
    output: str
    edits: list[Change]
    model: str
    model_revision: str = "unknown"
    mode: str
    seconds: float
    chunks: int
    offset_unit: str = "unicode_code_point"
    version: str = "1.0.0"
