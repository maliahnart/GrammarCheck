"""Run: python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000."""
import hmac
import json
import logging
import os
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from backend.datasets import MAX_BYTES, clean_dataset
from backend.learning import LESSONS
from backend.schemas import CheckRequest, CheckResponse
from backend.service import check_document

logger = logging.getLogger(__name__)


def authorize(x_api_key: Annotated[str | None, Header()] = None):
    expected = os.environ.get("GRAMMARCHECK_API_KEY", "")
    if expected and not hmac.compare_digest(expected, x_api_key or ""):
        raise HTTPException(401, "API key không hợp lệ.")


app = FastAPI(title="GrammarCheck · Tiếng Việt", version="1.0.0",
              description="API dùng chung cho tài liệu, học tiếng Việt và dữ liệu NLP.",
              dependencies=[Depends(authorize)])
app.add_middleware(
    CORSMiddleware,
    allow_origins=[item.strip() for item in os.environ.get("GRAMMARCHECK_ORIGINS", "http://localhost:8000").split(",")],
    allow_origin_regex=r"chrome-extension://[a-p]{32}",
    allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-API-Key"],
)


@app.exception_handler(ValueError)
async def invalid_input(request, exc):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(FileNotFoundError)
async def missing_model(request, exc):
    return JSONResponse(status_code=503, content={"detail": "Chưa có checkpoint. Hãy huấn luyện mô hình được chọn."})


@app.exception_handler(Exception)
async def inference_error(request, exc):
    # Avoid putting user text into logs or responses.
    logger.error("Request failed (%s)", type(exc).__name__)
    return JSONResponse(status_code=503, content={"detail": "Chưa xử lý được yêu cầu. Kiểm tra mô hình và cấu hình máy chủ."})


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/v1/lessons")
def lessons():
    return {"lessons": LESSONS}


@app.post("/v1/check", response_model=CheckResponse)
def check(request: CheckRequest):
    return check_document(request)


@app.post("/v1/datasets/clean")
async def dataset(
    file: Annotated[UploadFile, File()],
    column: Annotated[str, Form(max_length=100)] = "text",
    model: Annotated[str, Form()] = "ml",
    preserve_style: Annotated[bool, Form()] = True,
    normalize_abbrev: Annotated[bool, Form()] = False,
    protected_terms: Annotated[str, Form()] = "[]",
):
    try:
        data = await file.read(MAX_BYTES + 1)
    finally:
        await file.close()
    try:
        terms = json.loads(protected_terms)
        validated = CheckRequest(text="validate", model=model, protected_terms=terms,
                                 preserve_style=preserve_style, normalize_abbrev=normalize_abbrev)
    except (ValidationError, json.JSONDecodeError) as exc:
        raise HTTPException(422, "Cấu hình mô hình hoặc thuật ngữ không hợp lệ.") from exc
    options = validated.model_dump(exclude={"text", "mode"})
    return await run_in_threadpool(clean_dataset, data, file.filename or "", column, options)
