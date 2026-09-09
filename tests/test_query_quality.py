import json
from types import SimpleNamespace

import pytest
import redis

from src import rag_engine as rag


@pytest.fixture
def services(monkeypatch):
    writes = []
    monkeypatch.setattr(rag.redis_client, "get", lambda key: None)
    monkeypatch.setattr(rag.redis_client, "setex", lambda *args: writes.append(args))
    monkeypatch.setattr(rag, "get_embeddings", lambda: SimpleNamespace(embed_query=lambda q: [1.0]))
    monkeypatch.setattr(rag.qdrant_client, "query_points", lambda **kw: SimpleNamespace(points=[
        SimpleNamespace(payload={"source_file": "policy.txt", "document_id": "a", "chunk_index": 0, "text": "Warranty lasts 12 months."}),
    ]))
    monkeypatch.setattr(rag, "_invoke_llm", lambda p: SimpleNamespace(content="12 months [Source 1]"))
    return writes


@pytest.mark.parametrize("operation", ["get", "setex"])
def test_cache_outage_preserves_generated_answer(services, monkeypatch, operation):
    def unavailable(*args):
        raise redis.ConnectionError("private cache hostname")
    monkeypatch.setattr(rag.redis_client, operation, unavailable)
    result = rag.ask_question("Warranty?", "team-a")
    assert result["status"] == "generated"
    assert result["answer"] == "12 months [Source 1]"
    assert result["cached"] is False
    assert result["context"][0]["source_file"] == "policy.txt"


@pytest.mark.parametrize("payload", ["not json", "[]", '{"context":null}', '{"answer":7,"context":[]}'])
def test_malformed_cache_is_a_miss(services, monkeypatch, payload):
    monkeypatch.setattr(rag.redis_client, "get", lambda key: payload)
    result = rag.ask_question("Warranty?", "team-a")
    assert result["cached"] is False
    assert result["answer"] == "12 months [Source 1]"


def test_metadata_failure_is_not_hidden_as_cache_outage(services, monkeypatch):
    def fail(*args):
        raise OSError("metadata unavailable")
    monkeypatch.setattr(rag, "get_corpus_version", fail)
    with pytest.raises(OSError):
        rag.ask_question("Warranty?", "team-a")


def test_cache_invalid_encoding_is_a_miss(services, monkeypatch):
    def undecodable(key):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")
    monkeypatch.setattr(rag.redis_client, "get", undecodable)
    result = rag.ask_question("Warranty?", "team-a")
    assert result["cached"] is False
    assert result["status"] == "generated"


def test_valid_cache_remains_a_hit(services, monkeypatch):
    context = [{"source_file": "policy.txt", "document_id": "a", "chunk_index": 0, "page_number": None, "text": "Warranty lasts 12 months."}]
    monkeypatch.setattr(rag.redis_client, "get", lambda key: json.dumps({"answer": "12 months [Source 1]", "context": context, "status": "generated"}))
    result = rag.ask_question("Warranty?", "team-a")
    assert result["cached"] is True
    assert result["context"] == context


@pytest.mark.parametrize("answer", ["12 months", "12 months [Source 2]", "12 months [Source 0]"])
def test_invalid_citations_return_evidence_not_generated_answer(services, monkeypatch, answer):
    monkeypatch.setattr(rag, "_invoke_llm", lambda p: SimpleNamespace(content=answer))
    result = rag.ask_question("Warranty?", "team-a")
    assert result["status"] == "retrieved"
    assert result["answer"] == ""
    assert result["error_code"] == "invalid_citations"
    assert result["context"][0]["text"] == "Warranty lasts 12 months."
    assert services == []


def test_invalid_cached_citations_are_not_served(services, monkeypatch):
    monkeypatch.setattr(rag.redis_client, "get", lambda k: json.dumps({
        "answer": "Wrong [Source 9]", "status": "generated", "context": [
            {"source_file": "policy.txt", "document_id": "a", "chunk_index": 0, "text": "Evidence"},
        ],
    }))
    result = rag.ask_question("Warranty?", "team-a")
    assert result["cached"] is False
    assert result["answer"] == "12 months [Source 1]"


HISTORY = [{"role": "user", "content": "Tell me about the warranty."},
           {"role": "assistant", "content": "The warranty covers defects [Source 1]."}]


def test_followup_rewrites_before_retrieval_and_bypasses_cache(services, monkeypatch):
    def forbidden(*args):
        pytest.fail("History requests must bypass shared answer cache")
    monkeypatch.setattr(rag.redis_client, "get", forbidden)
    monkeypatch.setattr(rag.redis_client, "setex", forbidden)
    prompts = []
    def provider(prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            assert "Tell me about the warranty." in prompt
            return SimpleNamespace(content='{"query":"How long is the warranty?"}')
        assert "How long is the warranty?" in prompt
        assert "The warranty covers defects" not in prompt
        return SimpleNamespace(content="12 months [Source 1]")
    monkeypatch.setattr(rag, "_invoke_llm", provider)
    embedded = []
    monkeypatch.setattr(rag, "get_embeddings", lambda: SimpleNamespace(embed_query=lambda q: embedded.append(q) or [1.0]))
    result = rag.ask_question("How long?", "team-a", history=HISTORY)
    assert embedded == ["How long is the warranty?"]
    assert result["query"] == "How long?"
    assert result["status"] == "generated"
    assert len(prompts) == 2


@pytest.mark.parametrize("rewrite", ["not json", "[]", '{"query":" "}', json.dumps({"query": "x" * 4001})])
def test_bad_rewrite_fails_safely_without_retrieving(services, monkeypatch, rewrite):
    monkeypatch.setattr(rag, "_invoke_llm", lambda p: SimpleNamespace(content=rewrite))
    monkeypatch.setattr(rag, "get_embeddings", lambda: pytest.fail("Invalid rewrite must not search"))
    result = rag.ask_question("How long?", "team-a", history=HISTORY)
    assert result["status"] == "degraded"
    assert result["error_code"] == "rewrite_unavailable"
    assert result["context"] == []
    assert services == []


def test_evidence_only_history_never_calls_model(services, monkeypatch):
    monkeypatch.setattr(rag, "_invoke_llm", lambda p: pytest.fail("Evidence-only called model"))
    result = rag.ask_question("Warranty?", "team-a", history=HISTORY, retrieval_only=True)
    assert result["status"] == "retrieved"
    assert result["answer"] == ""
