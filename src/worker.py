import json
import os
import subprocess
import sys
from pathlib import Path

from celery import Celery
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

from src import state

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

celery_app = Celery(
    "docuquery_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
)


def run_ingestion(*arguments) -> dict:
    completed = subprocess.run(
        [sys.executable, "-m", "src.ingestion_job"],
        input=json.dumps(arguments).encode("utf-8"),
        stdout=subprocess.PIPE,
        timeout=float(os.getenv("DOCUMENT_PROCESS_TIMEOUT_SECONDS", "180")),
        check=False,
        cwd=PROJECT_ROOT,
    )
    if completed.returncode == 75:
        raise state.WorkspaceBusyError("Workspace is busy")
    if completed.returncode == 76:
        raise state.TaskCancelledError("Task was cancelled")
    if completed.returncode:
        raise RuntimeError("Document processing failed")
    return json.loads(completed.stdout)


@celery_app.task(name="process_document_task", bind=True, max_retries=5)
def process_document_task(
    self,
    file_path: str,
    workspace_id: str,
    document_id: str,
    source_file: str,
    task_id: str | None = None,
) -> dict:
    if task_id is None:
        raise state.TaskCancelledError("Legacy queued task must be uploaded again")
    path = Path(file_path).resolve()
    expected = (state.upload_root() / ".staging" / workspace_id).resolve()
    if path.parent != expected:
        raise state.TaskCancelledError("Task does not own this staging path")
    try:
        result = run_ingestion(
            file_path, workspace_id, document_id, source_file, task_id
        )
    except state.WorkspaceBusyError as exc:
        if self.request.retries < 5:
            raise self.retry(exc=exc, countdown=min(2**self.request.retries, 30))
        state.fail_task(workspace_id, task_id)
        path.unlink(missing_ok=True)
        raise
    except Exception:
        state.fail_task(workspace_id, task_id)
        path.unlink(missing_ok=True)
        raise
    path.unlink(missing_ok=True)
    return result
