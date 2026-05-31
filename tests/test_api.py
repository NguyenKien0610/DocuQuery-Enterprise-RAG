import json
from io import BytesIO
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.main as main
import src.rag_engine as rag_engine


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "ensure_qdrant_collection", lambda: None)
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    with TestClient(main.app) as test_client:
        yield test_client


def test_upload_accepts_valid_pdf_and_returns_task_id(client, monkeypatch):
    captured = {}

    def fake_delay(file_path: str):
        captured["file_path"] = file_path
        return SimpleNamespace(id="task-valid-001")

    monkeypatch.setattr(main.process_document_task, "delay", fake_delay)

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("sample.pdf", BytesIO(b"%PDF-1.4 valid pdf payload"), "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json() == {"task_id": "task-valid-001"}
    assert captured["file_path"].endswith(".pdf")


def test_upload_accepts_empty_pdf_payload_boundary_case(client, monkeypatch):
    def fake_delay(file_path: str):
        return SimpleNamespace(id="task-empty-001")

    monkeypatch.setattr(main.process_document_task, "delay", fake_delay)

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("empty.pdf", BytesIO(b""), "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json() == {"task_id": "task-empty-001"}


def test_upload_rejects_non_pdf_extension(client):
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("image.png", BytesIO(b"not a document"), "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF, DOCX, and TXT files are supported."


def test_query_returns_cached_answer_on_cache_hit(client, monkeypatch):
    monkeypatch.setattr(
        rag_engine.redis_client,
        "get",
        lambda key: json.dumps(
            {
                "answer": "Cached answer",
                "context": [
                    {
                        "source_file": "cached-file.pdf",
                        "source_path": "E:/docs/cached-file.pdf",
                        "chunk_index": 2,
                        "page_number": 3,
                        "text": "Cached context from Redis",
                    }
                ],
            }
        ),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Cache hit should not call retrieval or LLM dependencies.")

    monkeypatch.setattr(
        rag_engine,
        "embeddings",
        SimpleNamespace(embed_query=fail_if_called),
    )
    monkeypatch.setattr(
        rag_engine,
        "qdrant_client",
        SimpleNamespace(query_points=fail_if_called),
    )

    response = client.post("/api/v1/query", json={"query": "What is cached?"})

    assert response.status_code == 200
    assert response.json() == {
        "query": "What is cached?",
        "answer": "Cached answer",
        "cached": True,
        "context": [
            {
                "source_file": "cached-file.pdf",
                "source_path": "E:/docs/cached-file.pdf",
                "chunk_index": 2,
                "page_number": 3,
                "text": "Cached context from Redis",
            }
        ],
    }


def test_query_returns_context_and_fresh_answer_on_cache_miss(client, monkeypatch):
    captured_cache = {}
    context_chunks = [
        {
            "text": "Chunk A from Qdrant",
            "source": "E:/docs/file-two.pdf",
            "chunk_index": 0,
            "page_number": 7,
        },
        {
            "text": "Chunk B from Qdrant",
            "source": "E:/docs/file-two.pdf",
            "chunk_index": 1,
            "page_number": 8,
        },
    ]

    monkeypatch.setattr(rag_engine.redis_client, "get", lambda key: None)
    monkeypatch.setattr(
        rag_engine.redis_client,
        "setex",
        lambda key, ttl, value: captured_cache.update({"key": key, "ttl": ttl, "value": value}),
    )
    monkeypatch.setattr(
        rag_engine.qdrant_client,
        "query_points",
        lambda **kwargs: SimpleNamespace(
            points=[
                SimpleNamespace(payload=context_chunks[0]),
                SimpleNamespace(payload=context_chunks[1]),
            ]
        ),
    )
    monkeypatch.setattr(
        rag_engine,
        "embeddings",
        SimpleNamespace(embed_query=lambda query: [0.1, 0.2, 0.3]),
    )
    monkeypatch.setattr(
        rag_engine,
        "_invoke_llm",
        lambda prompt: SimpleNamespace(content="Fresh generated answer"),
    )

    response = client.post("/api/v1/query", json={"query": "Explain the document"})

    assert response.status_code == 200
    assert response.json() == {
        "query": "Explain the document",
        "answer": "Fresh generated answer",
        "cached": False,
        "context": [
            {
                "source_file": "file-two.pdf",
                "source_path": "E:/docs/file-two.pdf",
                "chunk_index": 0,
                "page_number": 7,
                "text": "Chunk A from Qdrant",
            },
            {
                "source_file": "file-two.pdf",
                "source_path": "E:/docs/file-two.pdf",
                "chunk_index": 1,
                "page_number": 8,
                "text": "Chunk B from Qdrant",
            },
        ],
    }
    assert captured_cache["ttl"] == rag_engine.CACHE_TTL_SECONDS
    assert json.loads(captured_cache["value"]) == {
        "answer": "Fresh generated answer",
        "context": [
            {
                "source_file": "file-two.pdf",
                "source_path": "E:/docs/file-two.pdf",
                "chunk_index": 0,
                "page_number": 7,
                "text": "Chunk A from Qdrant",
            },
            {
                "source_file": "file-two.pdf",
                "source_path": "E:/docs/file-two.pdf",
                "chunk_index": 1,
                "page_number": 8,
                "text": "Chunk B from Qdrant",
            },
        ],
    }
