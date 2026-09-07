import pytest

from src import worker


def test_worker_removes_uploaded_file_when_ingestion_fails(monkeypatch, tmp_path):
    path = tmp_path / "document.txt"
    path.write_text("content", encoding="utf-8")
    monkeypatch.setattr(
        worker,
        "ingest_document",
        lambda *args: (_ for _ in ()).throw(RuntimeError("parse failed")),
    )

    with pytest.raises(RuntimeError, match="parse failed"):
        worker.process_document_task.run(
            str(path),
            "team-a",
            "a" * 64,
            "document.txt",
        )

    assert not path.exists()


def test_worker_keeps_uploaded_file_after_success(monkeypatch, tmp_path):
    path = tmp_path / "document.txt"
    path.write_text("content", encoding="utf-8")
    expected = {"status": "ingested", "chunks_indexed": 1}
    monkeypatch.setattr(worker, "ingest_document", lambda *args: expected)

    result = worker.process_document_task.run(
        str(path),
        "team-a",
        "a" * 64,
        "document.txt",
    )

    assert result == expected
    assert path.exists()
