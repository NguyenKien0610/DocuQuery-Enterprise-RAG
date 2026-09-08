"""Real services, isolated collection/workspaces; no model download or Gemini call."""

import hashlib
import os
import uuid
from types import SimpleNamespace

import pytest

from src import rag_engine as rag

pytestmark = pytest.mark.skipif(
    os.getenv("DOCUQUERY_INTEGRATION") != "1",
    reason="Set DOCUQUERY_INTEGRATION=1 with Redis/Qdrant running",
)


def test_real_ingestion_cache_isolation_and_reset(monkeypatch, tmp_path):
    token = uuid.uuid4().hex
    workspace = f"integration-{token}"
    other = f"other-{token}"
    collection = f"integration_{token}"
    monkeypatch.setattr(rag, "COLLECTION_NAME", collection)
    monkeypatch.setattr(
        rag,
        "get_embeddings",
        lambda: SimpleNamespace(
            embed_documents=lambda chunks: [[1.0, 0.0] for _ in chunks],
            embed_query=lambda query: [1.0, 0.0],
        ),
    )
    monkeypatch.setattr(
        rag,
        "_invoke_llm",
        lambda prompt: SimpleNamespace(content="Test answer [Source 1]"),
    )
    path = tmp_path / workspace / "doc.txt"
    path.parent.mkdir()
    path.write_text("The warranty lasts 24 months.", encoding="utf-8")
    document_id = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        rag.ingest_document(str(path), workspace, document_id, "doc.txt")
        rag.ingest_document(str(path), workspace, document_id, "doc.txt")
        assert (
            rag.qdrant_client.count(collection_name=collection, exact=True).count == 1
        )
        assert rag.ask_question("Warranty?", workspace)["cached"] is False
        assert rag.ask_question("Warranty?", workspace)["cached"] is True
        assert rag.ask_question("Warranty?", other)["status"] == "insufficient_context"
        with rag.workspace_lock(workspace), pytest.raises(rag.WorkspaceBusyError):
            rag.reset_workspace(workspace, tmp_path)
        rag.reset_workspace(workspace, tmp_path)
        assert (
            rag.ask_question("Warranty?", workspace)["status"] == "insufficient_context"
        )
    finally:
        # Only the random namespace allocated by this test is removed.
        if rag.qdrant_client.collection_exists(collection):
            rag.qdrant_client.delete_collection(collection)
        for pattern in (
            f"rag:answer:*:{workspace}:*",
            f"rag:corpus_version:{workspace}",
        ):
            for key in rag.redis_client.scan_iter(match=pattern):
                rag.redis_client.delete(key)
