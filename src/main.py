import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

from src.rag_engine import ask_question, ensure_qdrant_collection, reset_workspace
from src.schemas import QueryRequest, QueryResponse, TaskStatusResponse, UploadResponse
from src.worker import celery_app, process_document_task

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".docx", ".txt"}

@asynccontextmanager
async def lifespan(_: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ensure_qdrant_collection()
    yield


app = FastAPI(title="DocuQuery v2.0 - Enterprise RAG API", lifespan=lifespan)


def _list_uploaded_documents() -> list[str]:
    try:
        if not UPLOAD_DIR.exists():
            return []
        return sorted(
            [
                file_path.name
                for file_path in UPLOAD_DIR.iterdir()
                if file_path.is_file()
                and file_path.suffix.lower() in SUPPORTED_UPLOAD_EXTENSIONS
            ]
        )
    except OSError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to read uploaded documents: {exc}",
        ) from exc


def _delete_uploaded_documents() -> int:
    deleted_count = 0
    if not UPLOAD_DIR.exists():
        return deleted_count

    for file_path in UPLOAD_DIR.iterdir():
        if (
            not file_path.is_file()
            or file_path.suffix.lower() not in SUPPORTED_UPLOAD_EXTENSIONS
        ):
            continue
        try:
            file_path.unlink()
            deleted_count += 1
        except OSError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to delete uploaded document '{file_path.name}': {exc}",
            ) from exc

    return deleted_count


@app.post("/api/v1/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="A file is required.")

    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, DOCX, and TXT files are supported.",
        )

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


@app.get("/api/v1/documents")
def list_documents() -> dict:
    return {"documents": _list_uploaded_documents()}


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
        context=result.get("context", []),
    )


@app.delete("/api/v1/workspace/reset")
def reset_workspace_endpoint() -> dict:
    try:
        reset_result = reset_workspace()
        deleted_files = _delete_uploaded_documents()
        return {**reset_result, "deleted_files": deleted_files}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Workspace reset failed: {exc}") from exc
