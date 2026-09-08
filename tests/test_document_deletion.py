import hashlib
import os
import sqlite3
import uuid
from types import SimpleNamespace

import pytest

from src import rag_engine as rag
from src import state


def publish_document(workspace, document_id, revision):
    path = state.upload_root() / workspace / f"{revision}_notes.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("published content", encoding="utf-8")
    state.publish(workspace, {
        "document_id": document_id, "revision": revision, "source_file": "notes.txt",
        "path": str(path), "bytes": path.stat().st_size, "chunks": 1,
    }, None)
    return path


def test_delete_unpublishes_only_target_and_invalidates_cache_once(monkeypatch):
    monkeypatch.setattr(rag, "_collection_exists", lambda: False)
    first = publish_document("team-a", "a" * 64, "1" * 32)
    other = publish_document("team-a", "b" * 64, "2" * 32)
    foreign = publish_document("team-b", "a" * 64, "1" * 32)
    version = rag.get_corpus_version("team-a")
    generation = state.snapshot("team-a")["generation"]
    result = rag.delete_document("team-a", "a" * 64, "1" * 32)
    assert result["deleted"] is True
    assert result["cleanup_pending"] is False
    assert result["corpus_version"] == version + 1
    assert not first.exists()
    assert other.exists() and foreign.exists()
    assert state.snapshot("team-a")["generation"] == generation
    assert [d["document_id"] for d in state.snapshot("team-a")["documents"]] == ["b" * 64]
    replay = rag.delete_document("team-a", "a" * 64, "1" * 32)
    assert replay["deleted"] is False
    assert replay["corpus_version"] == version + 1


def test_cleanup_failure_does_not_republish_and_can_be_retried(monkeypatch):
    path = publish_document("team-a", "a" * 64, "1" * 32)
    monkeypatch.setattr(rag, "_collection_exists", lambda: True)
    monkeypatch.setattr(rag.qdrant_client, "delete", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("offline")))
    result = rag.delete_document("team-a", "a" * 64, "1" * 32)
    assert result["cleanup_pending"] is True
    assert state.snapshot("team-a")["documents"] == []
    assert path.exists()
    assert state.garbage("team-a")[0]["revision"] == "1" * 32
    deleted_filters = []
    monkeypatch.setattr(rag.qdrant_client, "delete", lambda **kwargs: deleted_filters.append(kwargs["points_selector"].filter))
    assert rag.reconcile_workspace("team-a")["cleanup_pending"] is False
    assert not path.exists()
    conditions = deleted_filters[0].must
    assert {c.key: c.match.value for c in conditions} == {"workspace_id": "team-a", "revision": "1" * 32}
    legacy = rag._workspace_filter("team-a").must[1].should[0]
    assert legacy.must_not[0].match.any == ["a" * 64]


def test_stale_delete_never_removes_reuploaded_revision(monkeypatch):
    monkeypatch.setattr(rag, "_collection_exists", lambda: False)
    publish_document("team-a", "a" * 64, "1" * 32)
    rag.delete_document("team-a", "a" * 64, "1" * 32)
    replacement = publish_document("team-a", "a" * 64, "2" * 32)
    version = rag.get_corpus_version("team-a")
    with pytest.raises(state.DocumentConflictError):
        rag.delete_document("team-a", "a" * 64, "1" * 32)
    assert replacement.exists()
    assert rag.get_corpus_version("team-a") == version


def test_active_upload_blocks_delete_without_mutation():
    path = publish_document("team-a", "a" * 64, "1" * 32)
    state.reserve_upload("pending", "team-a", 1)
    version = rag.get_corpus_version("team-a")
    with pytest.raises(state.WorkspaceBusyError):
        rag.delete_document("team-a", "a" * 64, "1" * 32)
    assert path.exists()
    assert rag.get_corpus_version("team-a") == version
    assert not state.garbage("team-a")


def test_unknown_document_is_not_deleted_from_another_workspace():
    foreign = publish_document("team-b", "a" * 64, "1" * 32)
    with pytest.raises(state.DocumentNotFoundError):
        rag.delete_document("team-a", "a" * 64, "1" * 32)
    assert foreign.exists()
    assert rag.get_corpus_version("team-a") == 0


def test_delete_transaction_rolls_back_on_metadata_failure():
    path = publish_document("team-a", "a" * 64, "1" * 32)
    version = rag.get_corpus_version("team-a")
    with state.database("team-a") as db:
        db.execute("CREATE TRIGGER reject_version BEFORE UPDATE OF version ON workspace BEGIN SELECT RAISE(ABORT, 'test failure'); END")
    with pytest.raises(sqlite3.IntegrityError):
        rag.delete_document("team-a", "a" * 64, "1" * 32)
    assert path.exists()
    snapshot = state.snapshot("team-a")
    assert len(snapshot["documents"]) == 1
    assert snapshot["deleted_document_ids"] == []
    assert snapshot["version"] == version
    assert state.garbage("team-a") == []


@pytest.mark.skipif(os.getenv("DOCUQUERY_INTEGRATION") != "1", reason="Requires Redis and Qdrant")
def test_real_delete_invalidates_cache_and_never_revives_legacy_vectors(monkeypatch, tmp_path):
    from qdrant_client import models

    token = uuid.uuid4().hex
    workspace, other = f"delete-{token}", f"other-{token}"
    collection = f"deletion_{token}"
    monkeypatch.setattr(rag, "COLLECTION_NAME", collection)

    def vector(text):
        return [1.0, 0.0] if "alpha" in text else [0.0, 1.0]

    monkeypatch.setattr(rag, "get_embeddings", lambda: SimpleNamespace(
        embed_documents=lambda chunks: [vector(text) for text in chunks], embed_query=vector,
    ))
    monkeypatch.setattr(rag, "_invoke_llm", lambda prompt: SimpleNamespace(content="Alpha answer [Source 1]"))
    paths = [tmp_path / "alpha.txt", tmp_path / "beta.txt"]
    for path in paths:
        path.write_text(path.stem + " policy", encoding="utf-8")
    doc_id = hashlib.sha256(paths[0].read_bytes()).hexdigest()
    try:
        for path in paths:
            rag.ingest_document(str(path), workspace, hashlib.sha256(path.read_bytes()).hexdigest(), path.name)
        target = next(doc for doc in state.snapshot(workspace)["documents"] if doc["document_id"] == doc_id)
        # Old unversioned copies must remain hidden even after the new revision is deleted.
        for scope in (workspace, other):
            rag.qdrant_client.upsert(collection_name=collection, wait=True, points=[models.PointStruct(
                id=str(uuid.uuid4()), vector=[1.0, 0.0], payload={
                    "workspace_id": scope, "document_id": doc_id, "source_file": "legacy.txt",
                    "chunk_index": 0, "text": "alpha legacy policy",
                },
            )])
        assert rag.ask_question("alpha?", workspace)["cached"] is False
        assert rag.ask_question("alpha?", workspace)["cached"] is True
        old_version = rag.get_corpus_version(workspace)
        result = rag.delete_document(workspace, doc_id, target["revision"])
        assert result["cleanup_pending"] is False
        assert rag.get_corpus_version(workspace) == old_version + 1
        answer = rag.ask_question("alpha?", workspace)
        assert answer["cached"] is False
        assert answer["status"] == "insufficient_context"
        assert rag.ask_question("beta?", workspace, retrieval_only=True)["context"][0]["source_file"] == "beta.txt"
        assert rag.ask_question("alpha?", other, retrieval_only=True)["context"][0]["source_file"] == "legacy.txt"
        assert rag.qdrant_client.count(collection_name=collection, exact=True).count == 3
    finally:
        if rag.qdrant_client.collection_exists(collection):
            rag.qdrant_client.delete_collection(collection)
        for scope in (workspace, other):
            for key in rag.redis_client.scan_iter(match=f"rag:answer:*:{scope}:*"):
                rag.redis_client.delete(key)
