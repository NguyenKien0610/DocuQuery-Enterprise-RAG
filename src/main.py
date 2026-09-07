import os
import logging
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

from src.rag_engine import (
    ask_question,
    ensure_qdrant_collection,
    register_task_workspace,
    remove_task_workspace,
    reset_workspace,
    task_belongs_to_workspace,
)
from src.schemas import QueryRequest, QueryResponse, TaskStatusResponse, UploadResponse
from src.security import RequestContextDep
from src.uploads import UploadRejected, save_validated_upload
from src.worker import celery_app, process_document_task

logger = logging.getLogger(__name__)

configured_upload_dir = Path(os.getenv("UPLOAD_DIR", "uploads"))
UPLOAD_DIR = (
    configured_upload_dir
    if configured_upload_dir.is_absolute()
    else PROJECT_ROOT / configured_upload_dir
)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
MAX_EXTRACTED_BYTES = int(os.getenv("MAX_EXTRACTED_BYTES", str(100 * 1024 * 1024)))
SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".docx", ".txt"}

@asynccontextmanager
async def lifespan(_: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ensure_qdrant_collection()
    yield


app = FastAPI(title="DocuQuery v2.0 - Enterprise RAG API", lifespan=lifespan)


def _list_uploaded_documents(workspace_id: str) -> list[str]:
    workspace_dir = UPLOAD_DIR / workspace_id
    try:
        if not workspace_dir.exists():
            return []
        return sorted(
            [
                file_path.name
                for file_path in workspace_dir.iterdir()
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
async def upload_document(
    file: Annotated[UploadFile, File()],
    context: RequestContextDep,
) -> UploadResponse:
    try:
        saved = await save_validated_upload(
            file,
            UPLOAD_DIR / context.workspace_id,
            MAX_UPLOAD_BYTES,
            MAX_EXTRACTED_BYTES,
        )
    except UploadRejected as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    except OSError as exc:
        logger.exception("Failed to store uploaded document")
        raise HTTPException(
            status_code=503,
            detail="Document storage is temporarily unavailable.",
        ) from exc

    task_id = str(uuid.uuid4())
    try:
        register_task_workspace(task_id, context.workspace_id)
        process_document_task.apply_async(
            args=[
                str(saved.path.resolve()),
                context.workspace_id,
                saved.document_id,
                saved.source_file,
            ],
            task_id=task_id,
        )
    except Exception as exc:
        saved.path.unlink(missing_ok=True)
        try:
            remove_task_workspace(task_id)
        except Exception:
            logger.exception("Failed to remove task workspace mapping")
        logger.exception("Failed to dispatch document processing task")
        raise HTTPException(
            status_code=503,
            detail="Document processing is temporarily unavailable.",
        ) from exc

    return UploadResponse(task_id=task_id, document_id=saved.document_id)


@app.get("/api/v1/documents/status/{task_id}", response_model=TaskStatusResponse)
def get_document_status(task_id: str, context: RequestContextDep) -> TaskStatusResponse:
    if not task_belongs_to_workspace(task_id, context.workspace_id):
        raise HTTPException(status_code=404, detail="Task not found.")
    task_result = celery_app.AsyncResult(task_id)

    if task_result.failed():
        return TaskStatusResponse(
            task_id=task_id,
            status=task_result.status,
            error="Document processing failed.",
        )

    result_payload = task_result.result if task_result.successful() else None
    return TaskStatusResponse(
        task_id=task_id,
        status=task_result.status,
        result=result_payload if isinstance(result_payload, dict) else None,
    )


@app.get("/api/v1/documents")
def list_documents(context: RequestContextDep) -> dict:
    return {"documents": _list_uploaded_documents(context.workspace_id)}


@app.post("/api/v1/query", response_model=QueryResponse)
def query_documents(payload: QueryRequest, context: RequestContextDep) -> QueryResponse:
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
def reset_workspace_endpoint(context: RequestContextDep) -> dict:
    try:
        reset_result = reset_workspace()
        deleted_files = _delete_uploaded_documents()
        return {**reset_result, "deleted_files": deleted_files}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Workspace reset failed: {exc}") from exc
