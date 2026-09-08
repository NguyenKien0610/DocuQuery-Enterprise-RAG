import pytest

from src import state, worker


def test_worker_removes_uploaded_file_when_ingestion_fails(monkeypatch, tmp_path):
    path = tmp_path / ".staging" / "team-a" / "document.txt"
    path.parent.mkdir(parents=True)
    path.write_text("content", encoding="utf-8")
    monkeypatch.setattr(
        worker,
        "run_ingestion",
        lambda *args: (_ for _ in ()).throw(RuntimeError("parse failed")),
    )

    with pytest.raises(RuntimeError, match="parse failed"):
        worker.process_document_task.run(
            str(path),
            "team-a",
            "a" * 64,
            "document.txt",
            "task",
        )

    assert not path.exists()


def test_worker_cleans_staging_file_after_success(monkeypatch, tmp_path):
    path = tmp_path / ".staging" / "team-a" / "document.txt"
    path.parent.mkdir(parents=True)
    path.write_text("content", encoding="utf-8")
    expected = {"status": "ingested", "chunks_indexed": 1}
    monkeypatch.setattr(worker, "run_ingestion", lambda *args: expected)

    result = worker.process_document_task.run(
        str(path),
        "team-a",
        "a" * 64,
        "document.txt",
        "task",
    )

    assert result == expected
    assert not path.exists()


def test_legacy_task_does_not_delete_legacy_document(tmp_path):
    path = tmp_path / "legacy.txt"
    path.write_text("preserve", encoding="utf-8")
    with pytest.raises(state.TaskCancelledError):
        worker.process_document_task.run(str(path), "team-a", "a", "legacy.txt")
    assert path.read_text(encoding="utf-8") == "preserve"
