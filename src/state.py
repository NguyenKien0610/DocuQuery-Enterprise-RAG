"""Authoritative local metadata; all services must share the same volume."""

import json
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout


class WorkspaceBusyError(RuntimeError):
    pass


class TaskCancelledError(RuntimeError):
    pass


class QuotaExceededError(ValueError):
    pass


class DocumentNotFoundError(LookupError):
    pass


class DocumentConflictError(RuntimeError):
    pass


def upload_root() -> Path:
    path = Path(os.getenv("UPLOAD_DIR", "uploads"))
    return path if path.is_absolute() else Path(__file__).resolve().parents[1] / path


def metadata_root() -> Path:
    root = Path(os.getenv("DOCUQUERY_STATE_DIR", str(upload_root() / ".state")))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _name(workspace: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", workspace):
        raise ValueError("Invalid workspace")
    return workspace


@contextmanager
def workspace_lock(workspace: str):
    lock = FileLock(metadata_root() / f"{_name(workspace)}.lock")
    try:
        lock.acquire(
            timeout=float(os.getenv("WORKSPACE_LOCK_BLOCKING_TIMEOUT_SECONDS", "1"))
        )
    except Timeout as exc:
        raise WorkspaceBusyError("Workspace is busy") from exc
    try:
        yield
    finally:
        lock.release()


@contextmanager
def database(workspace: str):
    connection = sqlite3.connect(
        metadata_root() / f"{_name(workspace)}.sqlite", timeout=10
    )
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA synchronous=FULL")
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS workspace (id INTEGER PRIMARY KEY, generation INTEGER, version INTEGER, legacy INTEGER);
            INSERT OR IGNORE INTO workspace VALUES (1, 0, 0, 1);
            CREATE TABLE IF NOT EXISTS documents (document_id TEXT PRIMARY KEY, revision TEXT, source_file TEXT, path TEXT, bytes INTEGER, chunks INTEGER);
            CREATE TABLE IF NOT EXISTS tasks (task_id TEXT PRIMARY KEY, generation INTEGER, status TEXT, path TEXT, reserved INTEGER, updated REAL, result TEXT);
            CREATE TABLE IF NOT EXISTS garbage (revision TEXT PRIMARY KEY, path TEXT);
            CREATE TABLE IF NOT EXISTS rate (bucket INTEGER PRIMARY KEY, count INTEGER);
            CREATE TABLE IF NOT EXISTS deleted_documents (revision TEXT PRIMARY KEY, document_id TEXT NOT NULL);
        """)
        connection.execute("BEGIN IMMEDIATE")
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def snapshot(workspace: str) -> dict:
    with database(workspace) as db:
        result = dict(db.execute("SELECT * FROM workspace").fetchone())
        result["documents"] = [
            dict(row) for row in db.execute("SELECT * FROM documents")
        ]
        result["deleted_document_ids"] = [
            row[0] for row in db.execute("SELECT DISTINCT document_id FROM deleted_documents ORDER BY document_id")
        ]
        return result


def unpublish_document(workspace: str, document_id: str, revision: str) -> dict:
    """Caller holds workspace_lock. Atomically hide a revision before physical cleanup."""
    if not re.fullmatch(r"[0-9a-f]{64}", document_id) or not re.fullmatch(r"[0-9a-f]{32}", revision):
        raise ValueError("Invalid document ID or revision")
    with database(workspace) as db:
        document = db.execute("SELECT * FROM documents WHERE document_id=?", (document_id,)).fetchone()
        version = db.execute("SELECT version FROM workspace").fetchone()[0]
        if document is None:
            deleted = db.execute(
                "SELECT 1 FROM deleted_documents WHERE revision=? AND document_id=?", (revision, document_id)
            ).fetchone()
            if deleted:
                return {"deleted": False, "corpus_version": version}
            raise DocumentNotFoundError("Document not found")
        if document["revision"] != revision:
            raise DocumentConflictError("Document revision changed")
        # ponytail: workspace-wide pending-upload guard; per-document task IDs if concurrent deletion is needed.
        if db.execute("SELECT 1 FROM tasks WHERE status IN ('uploading','queued') LIMIT 1").fetchone():
            raise WorkspaceBusyError("Workspace has active uploads")
        db.execute("INSERT INTO deleted_documents VALUES (?,?)", (revision, document_id))
        db.execute("INSERT OR REPLACE INTO garbage VALUES (?,?)", (revision, document["path"]))
        db.execute("DELETE FROM documents WHERE document_id=? AND revision=?", (document_id, revision))
        db.execute("UPDATE workspace SET version=version+1")
        return {"deleted": True, "corpus_version": version + 1}


def reserve_upload(task_id: str, workspace: str, maximum: int) -> int:
    with workspace_lock(workspace), database(workspace) as db:
        generation = db.execute("SELECT generation FROM workspace").fetchone()[0]
        used, count = db.execute(
            "SELECT COALESCE(SUM(bytes),0), COUNT(*) FROM documents"
        ).fetchone()
        reserved, pending = db.execute(
            "SELECT COALESCE(SUM(reserved),0),COUNT(*) FROM tasks WHERE reserved>0"
        ).fetchone()
        legacy_dir = upload_root() / workspace
        legacy_bytes = (
            sum(p.stat().st_size for p in legacy_dir.iterdir() if p.is_file())
            if legacy_dir.exists()
            else 0
        )
        staging_dir = upload_root() / ".staging" / workspace
        staging_bytes = (
            sum(p.stat().st_size for p in staging_dir.iterdir() if p.is_file())
            if staging_dir.exists()
            else 0
        )
        # Actual retained files include pending cleanup; quota must bound disk as well.
        if max(used, legacy_bytes) + max(reserved, staging_bytes) + maximum > int(
            os.getenv("MAX_WORKSPACE_BYTES", "268435456")
        ):
            raise QuotaExceededError("Workspace storage quota exceeded")
        if count + pending >= int(os.getenv("MAX_WORKSPACE_DOCUMENTS", "100")):
            raise QuotaExceededError("Workspace document quota exceeded")
        db.execute(
            "INSERT INTO tasks VALUES (?,?,'uploading','',?,?,NULL)",
            (task_id, generation, maximum, time.time()),
        )
        return generation


def queue_upload(task_id: str, workspace: str, path: Path) -> None:
    with workspace_lock(workspace), database(workspace) as db:
        row = db.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        generation = db.execute("SELECT generation FROM workspace").fetchone()[0]
        if (
            row is None
            or row["generation"] != generation
            or row["status"] != "uploading"
        ):
            raise TaskCancelledError("Upload was invalidated by reset")
        db.execute(
            "UPDATE tasks SET status='queued',path=?,reserved=?,updated=? WHERE task_id=?",
            (str(path.resolve()), path.stat().st_size, time.time(), task_id),
        )


def task(workspace: str, task_id: str) -> dict | None:
    with database(workspace) as db:
        row = db.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return dict(row) if row else None


def fail_task(workspace: str, task_id: str) -> None:
    with database(workspace) as db:
        db.execute("UPDATE tasks SET reserved=0 WHERE task_id=?", (task_id,))
        db.execute(
            "UPDATE tasks SET status='failed',reserved=0,updated=? WHERE task_id=? AND status NOT IN ('succeeded','cancelled')",
            (time.time(), task_id),
        )


def validate_task(workspace: str, task_id: str, path: Path) -> dict:
    current = task(workspace, task_id)
    if (
        current is None
        or current["generation"] != snapshot(workspace)["generation"]
        or current["status"] in ("cancelled", "failed", "uploading")
    ):
        raise TaskCancelledError(
            "Task no longer belongs to the active workspace generation"
        )
    if Path(current["path"]).resolve() != path.resolve():
        raise TaskCancelledError("Task path mismatch")
    return current


def record_candidate(workspace: str, revision: str, path: Path):
    with database(workspace) as db:
        db.execute("INSERT OR REPLACE INTO garbage VALUES (?,?)", (revision, str(path)))


def publish(workspace: str, document: dict, task_id: str | None) -> int:
    with database(workspace) as db:
        previous = db.execute(
            "SELECT revision,path FROM documents WHERE document_id=?",
            (document["document_id"],),
        ).fetchone()
        if previous:
            db.execute("INSERT OR REPLACE INTO garbage VALUES (?,?)", tuple(previous))
        db.execute(
            "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?)",
            tuple(
                document[key]
                for key in (
                    "document_id",
                    "revision",
                    "source_file",
                    "path",
                    "bytes",
                    "chunks",
                )
            ),
        )
        db.execute("DELETE FROM garbage WHERE revision=?", (document["revision"],))
        db.execute("UPDATE workspace SET version=version+1")
        version = db.execute("SELECT version FROM workspace").fetchone()[0]
        if task_id:
            result = {
                "status": "ingested",
                "document_id": document["document_id"],
                "source_file": document["source_file"],
                "chunks_indexed": document["chunks"],
                "corpus_version": version,
            }
            db.execute(
                "UPDATE tasks SET status='succeeded',reserved=0,result=?,updated=? WHERE task_id=?",
                (json.dumps(result), time.time(), task_id),
            )
        return version


def complete_duplicate(workspace: str, task_id: str, result: dict):
    with database(workspace) as db:
        db.execute(
            "UPDATE tasks SET status='succeeded',reserved=0,result=?,updated=? WHERE task_id=?",
            (json.dumps(result), time.time(), task_id),
        )


def reset(workspace: str, root: Path) -> int:
    with database(workspace) as db:
        for row in db.execute("SELECT revision,path FROM documents").fetchall():
            db.execute("INSERT OR REPLACE INTO garbage VALUES (?,?)", tuple(row))
        # Preserve a retryable marker for legacy vectors and files.
        db.execute(
            "INSERT OR REPLACE INTO garbage VALUES ('legacy',?)",
            (str(root / workspace),),
        )
        db.execute("DELETE FROM documents")
        db.execute(
            "UPDATE workspace SET generation=generation+1,version=version+1,legacy=0"
        )
        db.execute(
            "UPDATE tasks SET status='cancelled' WHERE status IN ('queued','uploading')"
        )
        return db.execute("SELECT version FROM workspace").fetchone()[0]


def garbage(workspace: str) -> list[dict]:
    with database(workspace) as db:
        return [dict(row) for row in db.execute("SELECT * FROM garbage")]


def forget_garbage(workspace: str, revision: str):
    with database(workspace) as db:
        db.execute("DELETE FROM garbage WHERE revision=?", (revision,))


def cleanup_tasks(workspace: str) -> int:
    """Call while holding workspace_lock; never remove a live upload's temp file."""
    cutoff = time.time() - float(os.getenv("DOCUMENT_TASK_TTL_SECONDS", "900"))
    staging = (upload_root() / ".staging" / workspace).resolve()
    removed = 0
    with database(workspace) as db:
        db.execute(
            "UPDATE tasks SET status='failed',reserved=0 WHERE status IN ('uploading','queued','cancelled') AND updated<?",
            (cutoff,),
        )
        terminal = db.execute(
            "SELECT * FROM tasks WHERE status IN ('failed','cancelled','succeeded') AND path!=''"
        ).fetchall()
        for row in terminal:
            path = Path(row["path"]).resolve()
            if path.parent != staging:
                continue
            if path.exists():
                path.unlink()
                removed += 1
            db.execute("UPDATE tasks SET reserved=0 WHERE task_id=?", (row["task_id"],))
        active = {
            row[0]
            for row in db.execute(
                "SELECT path FROM tasks WHERE status IN ('uploading','queued')"
            )
        }
        if staging.exists():
            for path in staging.iterdir():
                if (
                    path.is_file()
                    and str(path.resolve()) not in active
                    and path.stat().st_mtime < cutoff
                ):
                    path.unlink()
                    removed += 1
    return removed


def rate_limit(workspace: str) -> bool:
    bucket = int(time.time()) // 60
    with database(workspace) as db:
        db.execute("DELETE FROM rate WHERE bucket<?", (bucket,))
        db.execute(
            "INSERT INTO rate VALUES (?,1) ON CONFLICT(bucket) DO UPDATE SET count=count+1",
            (bucket,),
        )
        count = db.execute(
            "SELECT count FROM rate WHERE bucket=?", (bucket,)
        ).fetchone()[0]
        return count <= int(os.getenv("MAX_REQUESTS_PER_MINUTE", "120"))
