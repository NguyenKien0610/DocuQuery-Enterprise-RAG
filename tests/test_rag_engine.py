import subprocess
import sys
import textwrap
from types import SimpleNamespace

from src import rag_engine


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
        rag_engine,
        "HuggingFaceEmbeddings",
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
        lambda key: seen.append(key) or "cached answer",
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
