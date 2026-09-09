import json
import subprocess
import sys
import textwrap
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from src import rag_engine


def test_backend_import_does_not_load_training_libraries():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import src.rag_engine; import sys; "
            "assert 'torch' not in sys.modules; "
            "assert 'sentence_transformers' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr


def test_import_does_not_construct_model_clients():
    code = textwrap.dedent(
        """
        import langchain_google_genai
        import langchain_huggingface

        def fail(*args, **kwargs):
            raise RuntimeError("model client constructed during import")

        langchain_huggingface.HuggingFaceEmbeddings = fail
        langchain_google_genai.ChatGoogleGenerativeAI = fail
        import src.rag_engine
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_cache_key_isolated_by_workspace_and_version():
    key = rag_engine._cache_key

    assert key("Question", "team-a", 1) != key("Question", "team-b", 1)
    assert key("Question", "team-a", 1) != key("Question", "team-a", 2)
    assert key(" Question ", "team-a", 1) == key("question", "team-a", 1)


def test_model_clients_are_lazy_and_memoized(monkeypatch):
    created = []
    rag_engine.get_embeddings.cache_clear()
    monkeypatch.setattr(
        "langchain_huggingface.HuggingFaceEmbeddings",
        lambda **kwargs: created.append(kwargs) or object(),
    )

    assert created == []
    assert rag_engine.get_embeddings() is rag_engine.get_embeddings()
    assert len(created) == 1


def test_ask_question_uses_current_corpus_version(monkeypatch):
    seen = []
    monkeypatch.setattr(rag_engine, "get_corpus_version", lambda workspace: 7)
    monkeypatch.setattr(
        rag_engine.redis_client,
        "get",
        lambda key: seen.append(key) or json.dumps({
            "answer": "cached answer [Source 1]", "status": "generated", "context": [
                {"source_file": "policy.txt", "document_id": "a", "chunk_index": 0, "text": "Evidence"},
            ],
        }),
    )

    result = rag_engine.ask_question("Question", "team-a")

    assert ":team-a:7:" in seen[0]
    assert result["cached"] is True


def test_serialized_context_omits_server_path():
    result = SimpleNamespace(
        payload={
            "source_file": "report.pdf",
            "source": "E:/private/report.pdf",
            "document_id": "a" * 64,
            "chunk_index": 0,
            "page_number": 1,
            "text": "Evidence",
        }
    )

    context = rag_engine._serialize_context_chunk(result)

    assert context["source_file"] == "report.pdf"
    assert context["document_id"] == "a" * 64
    assert "source_path" not in context


def test_deterministic_point_ids_and_version_advance(monkeypatch, tmp_path):
    batches = []
    versions = []
    monkeypatch.setattr(rag_engine, "workspace_lock", lambda workspace: nullcontext())
    monkeypatch.setattr(
        rag_engine,
        "parse_isolated",
        lambda path: [{"text": "first chunk second chunk", "page_number": 1}],
    )
    monkeypatch.setattr(
        rag_engine,
        "get_text_splitter",
        lambda: SimpleNamespace(split_text=lambda text: ["first", "second"]),
    )
    monkeypatch.setattr(
        rag_engine,
        "get_embeddings",
        lambda: SimpleNamespace(
            embed_documents=lambda chunks: [[0.1, 0.2] for chunk in chunks]
        ),
    )
    monkeypatch.setattr(rag_engine, "_ensure_collection", lambda vector_size: None)
    monkeypatch.setattr(
        rag_engine.qdrant_client,
        "upsert",
        lambda **kwargs: batches.append(kwargs),
    )
    monkeypatch.setattr(
        rag_engine,
        "advance_corpus_version",
        lambda workspace: versions.append(workspace) or len(versions),
    )
    path = tmp_path / "doc.txt"
    path.write_text("first chunk\nsecond chunk", encoding="utf-8")

    first = rag_engine.ingest_document(
        str(path),
        "team-a",
        "a" * 64,
        "doc.txt",
    )
    first_ids = [point.id for point in batches[-1]["points"]]
    second = rag_engine.ingest_document(
        str(path),
        "team-a",
        "a" * 64,
        "doc.txt",
    )
    second_ids = [point.id for point in batches[-1]["points"]]

    assert first_ids == second_ids
    assert len(batches) == 1
    assert all(batch["wait"] is True for batch in batches)
    assert first["chunks_indexed"] == second["chunks_indexed"] == 2
    assert first["document_id"] == "a" * 64
    assert rag_engine.get_corpus_version("team-a") == 1
    assert batches[0]["points"][0].payload == {
        "text": "first",
        "workspace_id": "team-a",
        "revision": batches[0]["points"][0].payload["revision"],
        "document_id": "a" * 64,
        "source_file": "doc.txt",
        "chunk_index": 0,
        "page_number": 1,
    }


def test_failed_upsert_does_not_advance_corpus_version(monkeypatch, tmp_path):
    versions = []
    monkeypatch.setattr(rag_engine, "workspace_lock", lambda workspace: nullcontext())
    monkeypatch.setattr(
        rag_engine,
        "parse_isolated",
        lambda path: [{"text": "content", "page_number": None}],
    )
    monkeypatch.setattr(
        rag_engine,
        "get_text_splitter",
        lambda: SimpleNamespace(split_text=lambda text: [text]),
    )
    monkeypatch.setattr(
        rag_engine,
        "get_embeddings",
        lambda: SimpleNamespace(embed_documents=lambda chunks: [[0.1, 0.2]]),
    )
    monkeypatch.setattr(rag_engine, "_ensure_collection", lambda vector_size: None)
    monkeypatch.setattr(
        rag_engine.qdrant_client,
        "upsert",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("qdrant down")),
    )
    monkeypatch.setattr(
        rag_engine,
        "advance_corpus_version",
        lambda workspace: versions.append(workspace),
    )

    with pytest.raises(RuntimeError, match="qdrant down"):
        rag_engine.ingest_document(
            str(tmp_path / "doc.txt"),
            "team-a",
            "a" * 64,
            "doc.txt",
        )

    assert versions == []


def test_query_filters_qdrant_by_workspace(monkeypatch):
    query_calls = []
    monkeypatch.setattr(rag_engine, "get_corpus_version", lambda workspace: 0)
    monkeypatch.setattr(rag_engine.redis_client, "get", lambda key: None)
    monkeypatch.setattr(rag_engine.redis_client, "setex", lambda key, ttl, value: True)
    monkeypatch.setattr(
        rag_engine,
        "get_embeddings",
        lambda: SimpleNamespace(embed_query=lambda query: [0.1, 0.2]),
    )
    monkeypatch.setattr(
        rag_engine.qdrant_client,
        "query_points",
        lambda **kwargs: query_calls.append(kwargs) or SimpleNamespace(points=[]),
    )
    monkeypatch.setattr(
        rag_engine,
        "_invoke_llm",
        lambda prompt: SimpleNamespace(content="No relevant information."),
    )

    rag_engine.ask_question("Question", "team-a")

    condition = query_calls[0]["query_filter"].must[0]
    assert condition.key == "workspace_id"
    assert condition.match.value == "team-a"


def test_workspace_lock_uses_scoped_key_and_releases(monkeypatch):
    with rag_engine.workspace_lock("team-a"):
        with rag_engine.workspace_lock("team-b"):
            pass
    with rag_engine.workspace_lock("team-a"):
        pass


def test_workspace_lock_rejects_busy_workspace(monkeypatch):
    monkeypatch.setenv("WORKSPACE_LOCK_BLOCKING_TIMEOUT_SECONDS", "0")
    with rag_engine.workspace_lock("team-a"):
        with (
            pytest.raises(rag_engine.WorkspaceBusyError),
            rag_engine.workspace_lock("team-a"),
        ):
            pytest.fail("Busy workspace lock was entered.")


def test_reset_deletes_only_requested_workspace(monkeypatch, tmp_path):
    team_a = tmp_path / "team-a"
    team_b = tmp_path / "team-b"
    team_a.mkdir()
    team_b.mkdir()
    (team_a / "a.txt").write_text("a", encoding="utf-8")
    (team_b / "b.txt").write_text("b", encoding="utf-8")
    delete_calls = []
    versions = []
    monkeypatch.setattr(rag_engine, "workspace_lock", lambda workspace: nullcontext())
    monkeypatch.setattr(rag_engine, "_collection_exists", lambda: True)
    monkeypatch.setattr(
        rag_engine.qdrant_client,
        "delete",
        lambda **kwargs: delete_calls.append(kwargs),
    )
    monkeypatch.setattr(
        rag_engine,
        "advance_corpus_version",
        lambda workspace: versions.append(workspace) or 4,
    )

    result = rag_engine.reset_workspace("team-a", tmp_path)

    assert list(team_a.iterdir()) == []
    assert (team_b / "b.txt").read_text(encoding="utf-8") == "b"
    assert result["deleted_files"] == 1
    assert result["corpus_version"] == 1
    selector = delete_calls[0]["points_selector"]
    assert selector.filter.must[0].key == "workspace_id"
    assert selector.filter.must[0].match.value == "team-a"
