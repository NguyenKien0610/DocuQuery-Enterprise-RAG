import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import UploadFile

from src import rag_engine as rag
from src import state, worker
from src.uploads import save_validated_upload


@pytest.fixture
def vector_boundary(monkeypatch):
    writes = []
    monkeypatch.setattr(
        rag, "parse_isolated", lambda path: [{"text": "hello", "page_number": None}]
    )
    monkeypatch.setattr(
        rag,
        "get_embeddings",
        lambda: SimpleNamespace(
            embed_documents=lambda chunks: [[1.0, 0.0] for _ in chunks]
        ),
    )
    monkeypatch.setattr(rag, "_ensure_collection", lambda vector_size: None)
    monkeypatch.setattr(rag, "_collection_exists", lambda: False)
    monkeypatch.setattr(
        rag.qdrant_client, "upsert", lambda **kwargs: writes.append(kwargs)
    )
    monkeypatch.setattr(worker, "run_ingestion", rag.ingest_document)
    return writes


def queued_file(workspace="team-a", task_id="task-1"):
    state.reserve_upload(task_id, workspace, 1024)
    path = state.upload_root() / ".staging" / workspace / f"{task_id}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("hello", encoding="utf-8")
    state.queue_upload(task_id, workspace, path)
    return path


@pytest.mark.asyncio
async def test_duplicate_uploads_never_share_physical_file(tmp_path):
    first = await save_validated_upload(
        UploadFile(filename="same.txt", file=io.BytesIO(b"hello")), tmp_path, 100, 100
    )
    second = await save_validated_upload(
        UploadFile(filename="same.txt", file=io.BytesIO(b"hello")), tmp_path, 100, 100
    )
    assert first.document_id == second.document_id
    assert first.path != second.path
    second.path.unlink()
    assert first.path.read_bytes() == b"hello"


def test_failed_duplicate_task_cannot_delete_published_file(vector_boundary):
    path = queued_file()
    document_id = hashlib.sha256(b"hello").hexdigest()
    worker.process_document_task.run(
        str(path), "team-a", document_id, "same.txt", "task-1"
    )
    published = Path(state.snapshot("team-a")["documents"][0]["path"])
    duplicate = queued_file(task_id="task-2")
    with (
        patch.object(worker, "run_ingestion", side_effect=ValueError("failure")),
        pytest.raises(ValueError),
    ):
        worker.process_document_task.run(
            str(duplicate), "team-a", document_id, "same.txt", "task-2"
        )
    assert published.read_bytes() == b"hello"
    assert not duplicate.exists()


def test_reset_invalidates_queued_task_before_any_vector_write(vector_boundary):
    path = queued_file()
    rag.reset_workspace("team-a", state.upload_root())
    with pytest.raises(state.TaskCancelledError):
        worker.process_document_task.run(str(path), "team-a", "a", "same.txt", "task-1")
    assert vector_boundary == []
    assert not path.exists()
    assert not state.snapshot("team-a")["legacy"]


def test_reset_during_upload_rejects_late_queue(vector_boundary):
    state.reserve_upload("slow", "team-a", 1024)
    rag.reset_workspace("team-a", state.upload_root())
    with pytest.raises(state.TaskCancelledError):
        state.queue_upload("slow", "team-a", Path("unused"))


def test_metadata_failure_never_publishes_candidate(vector_boundary):
    path = queued_file()
    with (
        patch.object(state, "publish", side_effect=RuntimeError("disk full")),
        pytest.raises(RuntimeError),
    ):
        rag.ingest_document(str(path), "team-a", "a", "same.txt", "task-1")
    assert len(vector_boundary) == 1
    assert state.snapshot("team-a")["documents"] == []
    revision = vector_boundary[0]["points"][0].payload["revision"]
    assert revision not in str(rag._workspace_filter("team-a").model_dump())
    assert state.garbage("team-a")[0]["revision"] == revision
    assert rag.reconcile_workspace("team-a")["cleanup_pending"] is False


def test_reset_remains_logically_empty_when_cleanup_fails(vector_boundary, monkeypatch):
    path = queued_file()
    rag.ingest_document(str(path), "team-a", "a", "same.txt", "task-1")
    old_version = rag.get_corpus_version("team-a")
    monkeypatch.setattr(rag, "_collection_exists", lambda: True)
    monkeypatch.setattr(
        rag.qdrant_client,
        "delete",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("down")),
    )
    result = rag.reset_workspace("team-a", state.upload_root())
    assert result["cleanup_pending"] is True
    assert rag.get_corpus_version("team-a") > old_version
    assert state.snapshot("team-a")["documents"] == []
    assert "no-published-revision" in str(rag._workspace_filter("team-a").model_dump())


def test_workspace_quota_includes_pending_uploads(monkeypatch):
    monkeypatch.setenv("MAX_WORKSPACE_BYTES", "150")
    state.reserve_upload("one", "team-a", 100)
    with pytest.raises(state.QuotaExceededError):
        state.reserve_upload("two", "team-a", 100)


def test_busy_worker_retries_without_deleting_input(vector_boundary):
    from celery.exceptions import Retry

    path = queued_file()
    with (
        patch.object(worker, "run_ingestion", side_effect=state.WorkspaceBusyError()),
        patch.object(worker.process_document_task, "retry", side_effect=Retry()),
        pytest.raises(Retry),
    ):
        worker.process_document_task.run(str(path), "team-a", "a", "same.txt", "task-1")
    assert path.exists()


def test_parser_rejects_text_over_byte_budget(monkeypatch, tmp_path):
    from src.parsing import parse_isolated

    path = tmp_path / "large.txt"
    path.write_text("hello", encoding="utf-8")
    monkeypatch.setenv("MAX_PARSED_BYTES", "4")
    with pytest.raises(ValueError, match="limits"):
        parse_isolated(str(path))


def test_parser_rejects_pdf_over_page_budget(monkeypatch, tmp_path):
    from pypdf import PdfWriter

    from src.parsing import parse_isolated

    path = tmp_path / "large.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_blank_page(width=100, height=100)
    writer.write(path)
    monkeypatch.setenv("MAX_DOCUMENT_PAGES", "1")
    with pytest.raises(ValueError, match="limits"):
        parse_isolated(str(path))


def test_worker_deadline_cleans_staging_and_releases_reservation(monkeypatch):
    import subprocess

    path = queued_file()
    monkeypatch.setenv("DOCUMENT_PROCESS_TIMEOUT_SECONDS", "0.001")
    with pytest.raises(subprocess.TimeoutExpired):
        worker.process_document_task.run(str(path), "team-a", "a", "same.txt", "task-1")
    assert not path.exists()
    task = state.task("team-a", "task-1")
    assert task["status"] == "failed"
    assert task["reserved"] == 0
