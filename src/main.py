import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi import Path as ApiPath

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

from src import state
from src.rag_engine import (
    WorkspaceBusyError,
    ask_question,
    delete_document,
    ensure_qdrant_collection,
    qdrant_client,
    reconcile_workspace,
    redis_client,
    register_task_workspace,
    remove_task_workspace,
    reset_workspace,
    task_belongs_to_workspace,
)
from src.schemas import QueryRequest, QueryResponse, TaskStatusResponse, UploadResponse
from src.security import RequestContextDep, credentials
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


@app.get("/health/live", include_in_schema=False)
def health_live() -> dict:
    return {"status": "alive"}


@app.get("/health/ready", include_in_schema=False)
def health_ready() -> dict:
    try:
        credentials()
        redis_client.ping()
        qdrant_client.get_collections()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Service is not ready.") from exc
    return {"status": "ready"}


def _list_uploaded_documents(workspace_id: str) -> list[str]:
    workspace_dir = UPLOAD_DIR / workspace_id
    try:
        snapshot = state.snapshot(workspace_id)
        published = [Path(doc["path"]).name for doc in snapshot["documents"]]
        if not snapshot["legacy"]:
            return sorted(published)
        if not workspace_dir.exists():
            return sorted(published)
        unpublished = {
            Path(item["path"]).resolve() for item in state.garbage(workspace_id)
        }
        return sorted(
            [
                file_path.name
                for file_path in workspace_dir.iterdir()
                if file_path.is_file()
                and file_path.suffix.lower() in SUPPORTED_UPLOAD_EXTENSIONS
                and file_path.resolve() not in unpublished
            ]
        )
    except OSError as exc:
        logger.exception("Failed to read uploaded documents")
        raise HTTPException(
            status_code=503,
            detail="Document storage is temporarily unavailable.",
        ) from exc


@app.post("/api/v1/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: Annotated[UploadFile, File()],
    context: RequestContextDep,
) -> UploadResponse:
    context.require("write")
    task_id = str(uuid.uuid4())
    try:
        state.reserve_upload(task_id, context.workspace_id, MAX_UPLOAD_BYTES)
    except state.QuotaExceededError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except WorkspaceBusyError as exc:
        raise HTTPException(status_code=409, detail="Workspace is busy.") from exc
    except Exception as exc:
        logger.exception("Failed to reserve document storage")
        raise HTTPException(
            status_code=503, detail="Document storage is temporarily unavailable."
        ) from exc
    try:
        saved = await save_validated_upload(
            file,
            UPLOAD_DIR / ".staging" / context.workspace_id,
            MAX_UPLOAD_BYTES,
            MAX_EXTRACTED_BYTES,
        )
    except UploadRejected as exc:
        state.fail_task(context.workspace_id, task_id)
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    except OSError as exc:
        state.fail_task(context.workspace_id, task_id)
        logger.exception("Failed to store uploaded document")
        raise HTTPException(
            status_code=503,
            detail="Document storage is temporarily unavailable.",
        ) from exc

    try:
        state.queue_upload(task_id, context.workspace_id, saved.path)
        register_task_workspace(task_id, context.workspace_id)
        process_document_task.apply_async(
            args=[
                str(saved.path.resolve()),
                context.workspace_id,
                saved.document_id,
                saved.source_file,
                task_id,
            ],
            task_id=task_id,
        )
    except Exception as exc:
        saved.path.unlink(missing_ok=True)
        state.fail_task(context.workspace_id, task_id)
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
    try:
        stored_task = state.task(context.workspace_id, task_id)
        if stored_task and stored_task["status"] == "failed":
            return TaskStatusResponse(
                task_id=task_id,
                status="FAILURE",
                error="Document processing failed.",
            )
        if stored_task and stored_task["status"] == "cancelled":
            return TaskStatusResponse(
                task_id=task_id,
                status="FAILURE",
                error="Task cancelled by workspace reset.",
            )
        if stored_task and stored_task["status"] == "succeeded":
            import json

            return TaskStatusResponse(
                task_id=task_id,
                status="SUCCESS",
                result=json.loads(stored_task["result"]),
            )
        if stored_task is None and not task_belongs_to_workspace(task_id, context.workspace_id):
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
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Task status lookup failed")
        raise HTTPException(
            status_code=503,
            detail="Task status is temporarily unavailable.",
        ) from exc


@app.get("/api/v1/documents")
def list_documents(context: RequestContextDep) -> dict:
    return {"documents": _list_uploaded_documents(context.workspace_id)}


@app.get("/api/v1/documents/managed")
def list_managed_documents(context: RequestContextDep) -> dict:
    try:
        return {"items": [
            {
                "document_id": doc["document_id"], "revision": doc["revision"],
                "source_file": str(doc["source_file"]).replace("\\", "/").rsplit("/", 1)[-1],
                "bytes": doc["bytes"], "chunks": doc["chunks"],
            }
            for doc in sorted(state.snapshot(context.workspace_id)["documents"], key=lambda doc: doc["source_file"])
        ]}
    except Exception as exc:
        logger.exception("Managed document listing failed")
        raise HTTPException(status_code=503, detail="Document storage is temporarily unavailable.") from exc


@app.delete("/api/v1/documents/{document_id}")
def delete_document_endpoint(
    document_id: Annotated[str, ApiPath(pattern=r"^[0-9a-f]{64}$")],
    revision: Annotated[str, Query(pattern=r"^[0-9a-f]{32}$")],
    context: RequestContextDep,
) -> dict:
    context.require("admin")
    try:
        return delete_document(context.workspace_id, document_id, revision)
    except state.DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Document not found.") from exc
    except state.DocumentConflictError as exc:
        raise HTTPException(status_code=409, detail="Document revision changed. Refresh the document list.") from exc
    except WorkspaceBusyError as exc:
        raise HTTPException(status_code=409, detail="Workspace is busy or has active uploads. Try again after processing completes.") from exc
    except Exception as exc:
        logger.exception("Document deletion failed")
        raise HTTPException(status_code=503, detail="Document deletion is temporarily unavailable.") from exc


@app.post("/api/v1/query", response_model=QueryResponse)
def query_documents(payload: QueryRequest, context: RequestContextDep) -> QueryResponse:
    try:
        if payload.history and not payload.retrieval_only:
            result = ask_question(
                payload.query, context.workspace_id, use_cache=False,
                history=[message.model_dump() for message in payload.history],
            )
        elif payload.retrieval_only:
            result = ask_question(
                payload.query, context.workspace_id, retrieval_only=True
            )
        elif payload.use_cache:
            result = ask_question(payload.query, context.workspace_id)
        else:
            result = ask_question(payload.query, context.workspace_id, use_cache=False)
    except Exception as exc:
        logger.exception("Query processing failed")
        raise HTTPException(
            status_code=503,
            detail="Query service is temporarily unavailable.",
        ) from exc
    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        cached=result["cached"],
        status=result.get("status", "generated"),
        error_code=result.get("error_code"),
        context=result.get("context", []),
    )


@app.delete("/api/v1/workspace/reset")
def reset_workspace_endpoint(context: RequestContextDep) -> dict:
    context.require("admin")
    try:
        return reset_workspace(context.workspace_id, UPLOAD_DIR)
    except WorkspaceBusyError as exc:
        raise HTTPException(status_code=409, detail="Workspace is busy.") from exc
    except Exception as exc:
        logger.exception("Workspace reset failed")
        raise HTTPException(
            status_code=503,
            detail="Workspace reset is temporarily unavailable.",
        ) from exc


@app.post("/api/v1/workspace/reconcile")
def reconcile_workspace_endpoint(context: RequestContextDep) -> dict:
    context.require("admin")
    try:
        return reconcile_workspace(context.workspace_id)
    except WorkspaceBusyError as exc:
        raise HTTPException(status_code=409, detail="Workspace is busy.") from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Workspace cleanup is temporarily unavailable."
        ) from exc
