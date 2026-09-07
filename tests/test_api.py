import json
import hashlib
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

AUTH_HEADERS = {"X-API-Key": "test-api-key", "X-Workspace-ID": "team-a"}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCUQUERY_API_KEY", "test-api-key")
    monkeypatch.setattr(main, "ensure_qdrant_collection", lambda: None)
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    with TestClient(main.app) as test_client:
        yield test_client


def test_upload_accepts_valid_pdf_and_returns_task_id(client, monkeypatch):
    captured = {}

    def fake_apply_async(args, task_id):
        captured["args"] = args
        captured["task_id"] = task_id
        return SimpleNamespace(id=task_id)

    monkeypatch.setattr(main.process_document_task, "apply_async", fake_apply_async)
    monkeypatch.setattr(
        main,
        "register_task_workspace",
        lambda task_id, workspace_id: captured.update({"workspace": workspace_id}),
    )

    content = b"%PDF-1.4 valid pdf payload"
    response = client.post(
        "/api/v1/documents/upload",
        headers=AUTH_HEADERS,
        files={"file": ("sample.pdf", BytesIO(content), "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["document_id"] == hashlib.sha256(content).hexdigest()
    assert payload["task_id"] == captured["task_id"]
    assert captured["workspace"] == "team-a"
    assert captured["args"][1:] == [
        "team-a",
        payload["document_id"],
        "sample.pdf",
    ]
    assert Path(captured["args"][0]).parent.name == "team-a"


def test_upload_rejects_empty_pdf_payload(client):
    response = client.post(
        "/api/v1/documents/upload",
        headers=AUTH_HEADERS,
        files={"file": ("empty.pdf", BytesIO(b""), "application/pdf")},
    )

    assert response.status_code == 400
    assert [path for path in main.UPLOAD_DIR.rglob("*") if path.is_file()] == []


def test_upload_rejects_non_pdf_extension(client):
    response = client.post(
        "/api/v1/documents/upload",
        headers=AUTH_HEADERS,
        files={"file": ("image.png", BytesIO(b"not a document"), "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF, DOCX, and TXT files are supported."


def test_upload_rejects_oversized_file_before_dispatch(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 4)
    monkeypatch.setattr(
        main.process_document_task,
        "apply_async",
        lambda **kwargs: pytest.fail("Invalid upload reached Celery."),
    )

    response = client.post(
        "/api/v1/documents/upload",
        headers=AUTH_HEADERS,
        files={"file": ("large.txt", BytesIO(b"12345"), "text/plain")},
    )

    assert response.status_code == 413
    assert [path for path in main.UPLOAD_DIR.rglob("*") if path.is_file()] == []


def test_upload_dispatch_failure_removes_file_and_task_mapping(client, monkeypatch):
    removed_tasks = []
    monkeypatch.setattr(main, "register_task_workspace", lambda task_id, workspace: None)
    monkeypatch.setattr(main, "remove_task_workspace", removed_tasks.append)
    monkeypatch.setattr(
        main.process_document_task,
        "apply_async",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("broker secret")),
    )

    response = client.post(
        "/api/v1/documents/upload",
        headers=AUTH_HEADERS,
        files={"file": ("notes.txt", BytesIO(b"hello"), "text/plain")},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Document processing is temporarily unavailable."}
    assert len(removed_tasks) == 1
    assert [path for path in main.UPLOAD_DIR.rglob("*") if path.is_file()] == []


def test_list_documents_isolated_by_workspace(client):
    team_a = main.UPLOAD_DIR / "team-a"
    team_b = main.UPLOAD_DIR / "team-b"
    team_a.mkdir()
    team_b.mkdir()
    (team_a / "a.txt").write_text("a", encoding="utf-8")
    (team_b / "b.txt").write_text("b", encoding="utf-8")

    response = client.get("/api/v1/documents", headers=AUTH_HEADERS)

    assert response.status_code == 200
    assert response.json() == {"documents": ["a.txt"]}


def test_task_status_hidden_from_other_workspace(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "task_belongs_to_workspace",
        lambda task_id, workspace_id: False,
    )
    monkeypatch.setattr(
        main.celery_app,
        "AsyncResult",
        lambda task_id: pytest.fail("Hidden task result was accessed."),
    )

    response = client.get(
        "/api/v1/documents/status/private-task",
        headers={"X-API-Key": "test-api-key", "X-Workspace-ID": "team-b"},
    )

    assert response.status_code == 404


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

    response = client.post(
        "/api/v1/query",
        headers=AUTH_HEADERS,
        json={"query": "What is cached?"},
    )

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

    response = client.post(
        "/api/v1/query",
        headers=AUTH_HEADERS,
        json={"query": "Explain the document"},
    )

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


def test_api_rejects_missing_api_key(client):
    response = client.get(
        "/api/v1/documents",
        headers={"X-Workspace-ID": "team-a"},
    )

    assert response.status_code == 401


def test_api_rejects_wrong_api_key(client):
    response = client.get(
        "/api/v1/documents",
        headers={"X-API-Key": "wrong", "X-Workspace-ID": "team-a"},
    )

    assert response.status_code == 401


def test_api_fails_closed_without_server_key(client, monkeypatch):
    monkeypatch.delenv("DOCUQUERY_API_KEY", raising=False)

    response = client.get("/api/v1/documents", headers=AUTH_HEADERS)

    assert response.status_code == 503


def test_api_rejects_invalid_workspace(client):
    response = client.get(
        "/api/v1/documents",
        headers={"X-API-Key": "test-api-key", "X-Workspace-ID": "../team-a"},
    )

    assert response.status_code == 422


def test_query_rejects_whitespace(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "ask_question",
        lambda query: {
            "query": query,
            "answer": "Unexpected answer",
            "cached": False,
            "context": [],
        },
    )

    response = client.post(
        "/api/v1/query",
        headers=AUTH_HEADERS,
        json={"query": "   "},
    )

    assert response.status_code == 422
