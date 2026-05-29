import os
import shutil
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

from src.rag_engine import ask_question, ensure_qdrant_collection
from src.schemas import QueryRequest, QueryResponse, TaskStatusResponse, UploadResponse
from src.worker import celery_app, process_document_task

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))

app = FastAPI(title="DocuQuery v2.0 - Enterprise RAG API")


@app.on_event("startup")
def startup_event() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ensure_qdrant_collection()


@app.post("/api/v1/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    target_path = UPLOAD_DIR / f"{uuid.uuid4()}_{Path(file.filename).name}"
    with target_path.open("wb") as output_file:
        shutil.copyfileobj(file.file, output_file)

    task = process_document_task.delay(str(target_path.resolve()))
    return UploadResponse(task_id=task.id)


@app.get("/api/v1/documents/status/{task_id}", response_model=TaskStatusResponse)
def get_document_status(task_id: str) -> TaskStatusResponse:
    task_result = celery_app.AsyncResult(task_id)

    if task_result.failed():
        return TaskStatusResponse(
            task_id=task_id,
            status=task_result.status,
            error=str(task_result.result),
        )

    result_payload = task_result.result if task_result.successful() else None
    return TaskStatusResponse(
        task_id=task_id,
        status=task_result.status,
        result=result_payload if isinstance(result_payload, dict) else None,
    )


@app.post("/api/v1/query", response_model=QueryResponse)
def query_documents(payload: QueryRequest) -> QueryResponse:
    try:
        result = ask_question(payload.query)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Query processing failed: {exc}") from exc
    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        cached=result["cached"],
    )
