# DocuQuery P0 Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the existing DocuQuery demo with authenticated workspace-scoped APIs, safe uploads, corpus-versioned exact-query caching, lazy model loading, and idempotent ingestion.

**Architecture:** A small FastAPI dependency authenticates requests and validates a workspace slug. That workspace identifier is passed explicitly through filesystem storage, Celery tasks, Redis keys, Qdrant filters, and reset operations; Redis corpus versions invalidate exact-query cache entries without scans. Existing model clients become memoized lazy getters, while document IDs and point IDs become content-derived and deterministic.

**Tech Stack:** Python 3.13, FastAPI, Pydantic, Celery, Redis, Qdrant, LangChain integrations, Streamlit, Pytest.

**Spec:** `docs/superpowers/specs/2026-09-07-p0-hardening-design.md`

## Global Constraints

- Require `X-API-Key` and a 1-64 character `X-Workspace-ID` on every `/api/v1` route.
- Workspace IDs start with an ASCII alphanumeric character and otherwise contain only ASCII letters, digits, `_`, or `-`.
- `MAX_UPLOAD_BYTES` defaults to 25 MiB and `MAX_EXTRACTED_BYTES` defaults to 100 MiB.
- `WORKSPACE_LOCK_TIMEOUT_SECONDS` defaults to 600 and `WORKSPACE_LOCK_BLOCKING_TIMEOUT_SECONDS` defaults to 1.
- Never expose absolute server paths or raw infrastructure exceptions to API clients.
- Keep the cache exact-query based and the retrieval dense-only; documentation must use those terms.
- Unit tests must run with external network disabled and without live Redis, Qdrant, Celery, Hugging Face, or Gemini.
- Do not add dependencies for functionality available in the Python standard library.

---

### Task 1: Request Security and Workspace Contract

**Files:**
- Create: `src/security.py`
- Modify: `src/main.py`
- Modify: `src/schemas.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Produces: `RequestContext(workspace_id: str)` and `require_request_context(...) -> RequestContext`.
- Produces: `RequestContextDep`, an `Annotated` FastAPI dependency used by every API route.
- Produces: a trimmed, non-empty `QueryRequest.query`.

- [ ] **Step 1: Add failing authentication, workspace, and schema tests**

Add a shared request helper and these assertions to `tests/test_api.py`:

```python
AUTH_HEADERS = {"X-API-Key": "test-api-key", "X-Workspace-ID": "team-a"}


def test_api_rejects_missing_api_key(client):
    response = client.get("/api/v1/documents", headers={"X-Workspace-ID": "team-a"})
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


def test_query_rejects_whitespace(client):
    response = client.post("/api/v1/query", headers=AUTH_HEADERS, json={"query": "   "})
    assert response.status_code == 422
```

Update the fixture to set `DOCUQUERY_API_KEY=test-api-key` and send `AUTH_HEADERS` from existing successful API tests.

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```powershell
$env:HF_HUB_OFFLINE='1'
pytest tests/test_api.py -q -p no:cacheprovider --basetemp=uploads/.pytest-p0
```

Expected: authentication tests fail because routes are open, invalid workspace is accepted, and whitespace queries are accepted.

- [ ] **Step 3: Implement the minimal request dependency and schema changes**

Create `src/security.py` with:

```python
import hmac
import os
import re
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, HTTPException

WORKSPACE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class RequestContext:
    workspace_id: str


def require_request_context(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    x_workspace_id: Annotated[str | None, Header(alias="X-Workspace-ID")] = None,
) -> RequestContext:
    configured_key = os.getenv("DOCUQUERY_API_KEY", "")
    if not configured_key:
        raise HTTPException(status_code=503, detail="API authentication is not configured.")
    if x_api_key is None or not hmac.compare_digest(x_api_key, configured_key):
        raise HTTPException(status_code=401, detail="Invalid API credentials.")
    if x_workspace_id is None or WORKSPACE_PATTERN.fullmatch(x_workspace_id) is None:
        raise HTTPException(status_code=422, detail="Invalid workspace ID.")
    return RequestContext(workspace_id=x_workspace_id)


RequestContextDep = Annotated[RequestContext, Depends(require_request_context)]
```

Add `context: RequestContextDep` to every route. Tasks 2-5 will pass `context.workspace_id` to each downstream boundary when that boundary becomes workspace-aware. Constrain and trim `QueryRequest.query` with a Pydantic field validator so whitespace-only input returns 422.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run the Task 1 command. Expected: the complete existing API suite plus the new authentication, workspace, and query validation tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/security.py src/main.py src/schemas.py tests/test_api.py
git commit -m "feat(api): protect workspace routes"
```

---

### Task 2: Safe Upload and Task Ownership

**Files:**
- Create: `src/uploads.py`
- Modify: `src/main.py`
- Modify: `src/rag_engine.py`
- Test: `tests/test_uploads.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `RequestContext.workspace_id` from Task 1.
- Produces: `SavedUpload(path: Path, document_id: str, source_file: str)`.
- Produces: `save_validated_upload(file: UploadFile, root: Path) -> SavedUpload`.
- Produces: `register_task_workspace(task_id: str, workspace_id: str)`, `remove_task_workspace(task_id: str)`, and `task_belongs_to_workspace(task_id: str, workspace_id: str) -> bool`.

- [ ] **Step 1: Add failing upload validation and cleanup tests**

Create `tests/test_uploads.py` with valid byte builders and direct async tests:

```python
import io
import zipfile

import pytest
from fastapi import UploadFile

from src.uploads import UploadRejected, save_validated_upload


def make_docx() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types />")
        archive.writestr("word/document.xml", "<document>hello</document>")
    return buffer.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "content"),
    [("empty.pdf", b""), ("fake.pdf", b"not-pdf"), ("fake.docx", b"PK-broken"), ("empty.txt", b"   ")],
)
async def test_rejects_invalid_uploads(tmp_path, name, content):
    upload = UploadFile(filename=name, file=io.BytesIO(content))
    with pytest.raises(UploadRejected):
        await save_validated_upload(upload, tmp_path, max_upload_bytes=1024, max_extracted_bytes=4096)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_same_content_has_same_document_id(tmp_path):
    first = UploadFile(filename="first.txt", file=io.BytesIO(b"hello"))
    second = UploadFile(filename="second.txt", file=io.BytesIO(b"hello"))
    one = await save_validated_upload(first, tmp_path, 1024, 4096)
    two = await save_validated_upload(second, tmp_path, 1024, 4096)
    assert one.document_id == two.document_id
```

Add API tests asserting an oversized request returns 413 without calling Celery and a mocked `apply_async` exception leaves the workspace directory empty.

Also create workspace directories containing `team-a/a.txt` and `team-b/b.txt`; assert `GET /api/v1/documents` with team A headers returns only `a.txt`. Register a task mapping for team A and assert its status endpoint returns 404 when called with team B headers.

- [ ] **Step 2: Run upload tests and confirm RED**

Run:

```powershell
pytest tests/test_uploads.py tests/test_api.py -q -p no:cacheprovider --basetemp=uploads/.pytest-p0
```

Expected: collection fails because `src.uploads` and the new upload behavior do not exist.

- [ ] **Step 3: Implement streaming validation and owned task dispatch**

Create `src/uploads.py` using only `hashlib`, `zipfile`, `dataclasses`, `pathlib`, and FastAPI's `UploadFile`. Stream in 1 MiB chunks to a dot-prefixed temporary file, enforce the byte limit before each write, validate PDF/DOCX/TXT as specified, compute the lowercase SHA-256 digest, and atomically replace the temporary path with `<document_id>_<safe_name>`.

Change `_list_uploaded_documents(workspace_id)` to read only `UPLOAD_DIR / workspace_id`. In `main.upload_document`, call `save_validated_upload` with that workspace directory, allocate `task_id = str(uuid.uuid4())`, register `rag:task_workspace:<task_id>` for 86,400 seconds, and call:

```python
process_document_task.apply_async(
    args=[str(saved.path.resolve()), context.workspace_id, saved.document_id, saved.source_file],
    task_id=task_id,
)
```

On registration or dispatch failure, remove the task mapping and exact saved path, log the exception, and return generic HTTP 503. In `get_document_status`, return 404 unless the task mapping belongs to the request workspace and return `"Document processing failed."` for failed tasks.

- [ ] **Step 4: Run upload and API tests and confirm GREEN**

Run the Task 2 command. Expected: all upload validation, cleanup, task ownership, and existing API tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/uploads.py src/main.py src/rag_engine.py tests/test_uploads.py tests/test_api.py
git commit -m "feat(upload): validate and scope documents"
```

---

### Task 3: Lazy Models and Versioned Exact-Query Cache

**Files:**
- Modify: `src/rag_engine.py`
- Test: `tests/test_rag_engine.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Produces: `get_embeddings() -> HuggingFaceEmbeddings` and `get_llm() -> ChatGoogleGenerativeAI`.
- Produces: `_cache_key(query_text: str, workspace_id: str, corpus_version: int) -> str`.
- Produces: `get_corpus_version(workspace_id: str) -> int` and `advance_corpus_version(workspace_id: str) -> int`.
- Changes: `ask_question(query_text: str, workspace_id: str) -> dict[str, Any]`.
- Produces: `ContextChunk.document_id` and removes `ContextChunk.source_path`.

- [ ] **Step 1: Add failing lazy-loading and cache-version tests**

Create `tests/test_rag_engine.py` with:

```python
from types import SimpleNamespace

from src import rag_engine


def test_cache_key_isolated_by_workspace_and_version():
    key = rag_engine._cache_key
    assert key("Question", "team-a", 1) != key("Question", "team-b", 1)
    assert key("Question", "team-a", 1) != key("Question", "team-a", 2)
    assert key(" Question ", "team-a", 1) == key("question", "team-a", 1)


def test_model_clients_are_lazy(monkeypatch):
    created = []
    rag_engine.get_embeddings.cache_clear()
    monkeypatch.setattr(rag_engine, "HuggingFaceEmbeddings", lambda **kwargs: created.append(kwargs) or object())
    assert created == []
    assert rag_engine.get_embeddings() is rag_engine.get_embeddings()
    assert len(created) == 1


def test_ask_question_uses_current_corpus_version(monkeypatch):
    seen = []
    monkeypatch.setattr(rag_engine, "get_corpus_version", lambda workspace: 7)
    monkeypatch.setattr(rag_engine.redis_client, "get", lambda key: seen.append(key) or "cached answer")
    result = rag_engine.ask_question("Question", "team-a")
    assert ":team-a:7:" in seen[0]
    assert result["cached"] is True


def test_serialized_context_omits_server_path():
    result = SimpleNamespace(payload={
        "source_file": "report.pdf",
        "source": "E:/private/report.pdf",
        "document_id": "a" * 64,
        "chunk_index": 0,
        "page_number": 1,
        "text": "Evidence",
    })
    context = rag_engine._serialize_context_chunk(result)
    assert context["source_file"] == "report.pdf"
    assert context["document_id"] == "a" * 64
    assert "source_path" not in context
```

- [ ] **Step 2: Run RAG tests and confirm RED**

Run:

```powershell
$env:HF_HUB_OFFLINE='1'
pytest tests/test_rag_engine.py -q -p no:cacheprovider --basetemp=uploads/.pytest-p0
```

Expected: tests fail because models are eager and cache functions lack workspace/version parameters.

- [ ] **Step 3: Implement lazy getters and versioned keys**

Wrap model constructors with `functools.lru_cache(maxsize=1)`. Replace every direct `embeddings` or `llm` use with `get_embeddings()` or `get_llm()`. Implement version storage as `rag:corpus_version:<workspace_id>`, defaulting a missing key to zero and using Redis `INCR` to advance it.

Build answer keys exactly as `rag:answer:<workspace_id>:<version>:<sha256>` and update `ask_question` to read the current version before cache lookup. Pass the request workspace from `main.query_documents`. Add `document_id` to `ContextChunk`, remove `source_path`, and serialize the payload's safe `source_file` directly. Remove the `error` parameter from `_fallback_answer` so raw exceptions cannot enter responses.

- [ ] **Step 4: Run RAG and API tests and confirm GREEN**

Run:

```powershell
$env:HF_HUB_OFFLINE='1'
pytest tests/test_rag_engine.py tests/test_api.py -q -p no:cacheprovider --basetemp=uploads/.pytest-p0
```

Expected: model construction occurs only through getters, versioned cache tests pass, and API tests collect without contacting Hugging Face.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/rag_engine.py tests/test_rag_engine.py tests/test_api.py
git commit -m "fix(cache): version answers by workspace"
```

---

### Task 4: Idempotent Workspace-Scoped Ingestion and Retrieval

**Files:**
- Modify: `src/rag_engine.py`
- Modify: `src/worker.py`
- Test: `tests/test_rag_engine.py`
- Test: `tests/test_worker.py`

**Interfaces:**
- Produces: `WorkspaceBusyError` and `workspace_lock(workspace_id: str)` context manager.
- Changes: `ingest_document(file_path: str, workspace_id: str, document_id: str, source_file: str) -> dict[str, Any]`.
- Changes: `process_document_task(file_path: str, workspace_id: str, document_id: str, source_file: str) -> dict`.

- [ ] **Step 1: Add failing point-ID, filter, version, and cleanup tests**

Add tests that replace the model, Qdrant, Redis, splitter, and loader boundaries with in-memory fakes, then assert:

```python
from contextlib import nullcontext


def test_deterministic_point_ids_and_version_advance(monkeypatch, tmp_path):
    batches = []
    versions = []
    monkeypatch.setattr(rag_engine, "workspace_lock", lambda workspace: nullcontext())
    monkeypatch.setattr(
        rag_engine,
        "_load_document_sections",
        lambda path: [{"text": "first chunk second chunk", "page_number": 1}],
    )
    monkeypatch.setattr(rag_engine.text_splitter, "split_text", lambda text: ["first", "second"])
    monkeypatch.setattr(
        rag_engine,
        "get_embeddings",
        lambda: SimpleNamespace(embed_documents=lambda chunks: [[0.1, 0.2] for chunk in chunks]),
    )
    monkeypatch.setattr(rag_engine, "_ensure_collection", lambda vector_size: None)
    monkeypatch.setattr(
        rag_engine.qdrant_client,
        "upsert",
        lambda **kwargs: batches.append(kwargs["points"]),
    )
    monkeypatch.setattr(
        rag_engine,
        "advance_corpus_version",
        lambda workspace: versions.append(workspace) or len(versions),
    )
    path = tmp_path / "doc.txt"
    path.write_text("first chunk\nsecond chunk", encoding="utf-8")
    first = rag_engine.ingest_document(str(path), "team-a", "a" * 64, "doc.txt")
    first_ids = [point.id for point in batches[-1]]
    second = rag_engine.ingest_document(str(path), "team-a", "a" * 64, "doc.txt")
    second_ids = [point.id for point in batches[-1]]
    assert first_ids == second_ids
    assert first["chunks_indexed"] == second["chunks_indexed"]
    assert versions == ["team-a", "team-a"]


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
```

Create `tests/test_worker.py` asserting the exact file is unlinked when `ingest_document` raises and remains when ingestion succeeds.

- [ ] **Step 2: Run ingestion tests and confirm RED**

Run:

```powershell
pytest tests/test_rag_engine.py tests/test_worker.py -q -p no:cacheprovider --basetemp=uploads/.pytest-p0
```

Expected: tests fail on old signatures, random point IDs, per-chunk writes, missing filters, and missing worker cleanup.

- [ ] **Step 3: Implement the ingestion and retrieval invariants**

Build every chunk before writing, call `get_embeddings().embed_documents(chunks)` once, validate one vector per chunk, and create deterministic IDs with:

```python
str(uuid.uuid5(uuid.NAMESPACE_URL, f"docuquery:{workspace_id}:{document_id}:{chunk_index}"))
```

Attach workspace/document/source metadata to each point and send all points in one `qdrant_client.upsert(..., wait=True)` call. Advance the corpus version only after that call returns. Add a Qdrant `Filter` with a `workspace_id` match condition to query requests.

Implement `workspace_lock` with Redis lock key `rag:workspace_lock:<workspace_id>` and the configured lease/acquisition timeouts. Raise `WorkspaceBusyError` if acquisition returns false and release only a lock acquired by the current call.

Update the Celery task signature and unlink only its exact uploaded file on exception before re-raising.

- [ ] **Step 4: Run ingestion, worker, and API tests and confirm GREEN**

Run:

```powershell
pytest tests/test_rag_engine.py tests/test_worker.py tests/test_api.py -q -p no:cacheprovider --basetemp=uploads/.pytest-p0
```

Expected: deterministic-ID, single-batch, workspace-filter, version, lock, and cleanup tests pass.

- [ ] **Step 5: Commit Task 4**

```bash
git add src/rag_engine.py src/worker.py tests/test_rag_engine.py tests/test_worker.py tests/test_api.py
git commit -m "fix(ingest): make workspace writes idempotent"
```

---

### Task 5: Isolated Reset and Safe Error Responses

**Files:**
- Modify: `src/rag_engine.py`
- Modify: `src/main.py`
- Test: `tests/test_rag_engine.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `workspace_lock` and `WorkspaceBusyError` from Task 4.
- Changes: `reset_workspace(workspace_id: str, upload_root: Path) -> dict[str, Any]`.

- [ ] **Step 1: Add failing reset-isolation and error-hygiene tests**

Add tests asserting reset sends a workspace-filtered point deletion, removes only `UPLOAD_DIR/team-a`, increments only team A's version, and preserves `UPLOAD_DIR/team-b`. Add an API test that makes `ask_question` raise `RuntimeError("secret-internal-path")` and asserts:

```python
response = client.post(
    "/api/v1/query",
    headers=AUTH_HEADERS,
    json={"query": "Question"},
)
assert response.status_code == 503
assert response.json() == {"detail": "Query service is temporarily unavailable."}
assert "secret-internal-path" not in response.text
```

Add a reset test whose fake lock refuses acquisition and assert HTTP 409 with `{"detail": "Workspace is busy."}`.

- [ ] **Step 2: Run reset tests and confirm RED**

Run:

```powershell
pytest tests/test_rag_engine.py tests/test_api.py -q -p no:cacheprovider --basetemp=uploads/.pytest-p0
```

Expected: reset still deletes the global collection/files and raw exception details remain visible.

- [ ] **Step 3: Implement filtered reset and generic API failures**

Inside `workspace_lock(workspace_id)`, delete Qdrant points with a `workspace_id` filter, delete only the resolved `upload_root / workspace_id` files, and advance that workspace's corpus version. Never delete/recreate the collection during reset. Map `WorkspaceBusyError` to 409 in the endpoint and log all unexpected exceptions with `logging.exception` before returning stable generic 503 messages.

Update failed task status to return `"Document processing failed."` without serializing the task exception.

- [ ] **Step 4: Run all backend tests and confirm GREEN**

Run:

```powershell
$env:HF_HUB_OFFLINE='1'
pytest tests -q -p no:cacheprovider --basetemp=uploads/.pytest-p0
```

Expected: all backend tests pass without external network access.

- [ ] **Step 5: Commit Task 5**

```bash
git add src/rag_engine.py src/main.py tests/test_rag_engine.py tests/test_api.py
git commit -m "fix(reset): isolate workspace deletion"
```

---

### Task 6: Frontend Headers and Accurate Documentation

**Files:**
- Modify: `frontend/app.py`
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `docs.md`
- Modify: `scripts/benchmark_docuquery.py`
- Test: `tests/test_frontend.py`

**Interfaces:**
- Consumes: `DOCUQUERY_API_KEY`, `DOCUQUERY_WORKSPACE_ID`, and optional `DOCUQUERY_API_BASE_URL`.
- Produces: `_api_headers() -> dict[str, str]`, applied to every frontend and benchmark API request.
- Produces: `_request(method: str, url: str, **kwargs) -> requests.Response`, the only frontend HTTP boundary.

- [ ] **Step 1: Add failing frontend header tests**

Create `tests/test_frontend.py`:

```python
from frontend import app


def test_api_headers_read_environment(monkeypatch):
    monkeypatch.setenv("DOCUQUERY_API_KEY", "secret")
    monkeypatch.setenv("DOCUQUERY_WORKSPACE_ID", "team-a")
    assert app._api_headers() == {
        "X-API-Key": "secret",
        "X-Workspace-ID": "team-a",
    }


def test_request_attaches_workspace_headers(monkeypatch):
    captured = {}
    monkeypatch.setenv("DOCUQUERY_API_KEY", "secret")
    monkeypatch.setenv("DOCUQUERY_WORKSPACE_ID", "team-a")
    monkeypatch.setattr(
        app.requests,
        "request",
        lambda method, url, **kwargs: captured.update(
            {"method": method, "url": url, "headers": kwargs["headers"]}
        ),
    )
    app._request("GET", "http://backend/documents")
    assert captured == {
        "method": "GET",
        "url": "http://backend/documents",
        "headers": {"X-API-Key": "secret", "X-Workspace-ID": "team-a"},
    }
```

Refactor upload, polling, listing, query, and reset helpers to call `_request`, making the second test cover their shared HTTP boundary. Add a benchmark test that calls `_request_json` with explicit API key/workspace values and asserts the same two headers reach `requests.request`.

- [ ] **Step 2: Run frontend tests and confirm RED**

Run:

```powershell
pytest tests/test_frontend.py -q -p no:cacheprovider --basetemp=uploads/.pytest-p0
```

Expected: `_api_headers` is missing and requests do not carry workspace credentials.

- [ ] **Step 3: Update clients, environment example, and documentation**

Read API base URL and credentials from environment, add `_api_headers`, and merge the returned mapping inside `_request`. Route every frontend request through that helper. Add `--api-key` and `--workspace-id` arguments to the benchmark script, defaulting from the matching environment variables, and merge those headers inside `_request_json`.

Replace `.env.example` with all required and P0-configurable values. Update README setup, API examples, diagrams, migration note, security limitations, cache terminology, and retrieval terminology. Rewrite stale `docs.md` as a concise pointer to the current architecture and P0 security boundary; remove Gemini-embedding, semantic-cache, hybrid-retrieval, persistent-without-volume, and unsupported scalability claims.

- [ ] **Step 4: Run frontend tests and documentation scans**

Run:

```powershell
pytest tests/test_frontend.py -q -p no:cacheprovider --basetemp=uploads/.pytest-p0
rg -n "semantic cache|hybrid retrieval|GoogleGenerativeAIEmbeddings|OPENAI_API_KEY|highly scalable" README.md docs.md
```

Expected: frontend tests pass and the terminology scan returns no matches.

- [ ] **Step 5: Commit Task 6**

```bash
git add frontend/app.py .env.example README.md docs.md scripts/benchmark_docuquery.py tests/test_frontend.py
git commit -m "docs(p0): align clients and architecture"
```

---

### Task 7: Full Verification and Cleanup

**Files:**
- Modify only files required to resolve findings from the commands below.

**Interfaces:**
- Consumes: all Task 1-6 deliverables.
- Produces: a clean, reproducible P0 verification result.

- [ ] **Step 1: Run the complete offline test suite**

```powershell
$env:HF_HUB_OFFLINE='1'
pytest tests -q -p no:cacheprovider --basetemp=uploads/.pytest-p0
```

Expected: all tests pass with zero failures.

- [ ] **Step 2: Run static and configuration checks**

```powershell
ruff check . --exclude uploads
mypy src frontend scripts tests --ignore-missing-imports --explicit-package-bases
python -m compileall -q src frontend scripts tests
docker compose config
git diff --check
```

Expected: every command exits zero. Fix only task-caused or existing gate-blocking findings at the narrowest responsible line, then rerun the full command set.

- [ ] **Step 3: Verify acceptance criteria directly**

Confirm the test names and outputs cover all nine acceptance criteria in the spec. Run:

```powershell
rg -n "source_path|semantic cache|hybrid retrieval|GoogleGenerativeAIEmbeddings|OPENAI_API_KEY|highly scalable" src frontend tests README.md docs.md
git status --short
```

Expected: no production/API `source_path`, no misleading documentation terms, and only intended tracked changes before the final commit.

- [ ] **Step 4: Commit verification fixes if any**

```bash
git add src frontend scripts tests README.md docs.md .env.example
git commit -m "chore(p0): pass quality gates"
```

- [ ] **Step 5: Perform final diff review**

Review `git diff HEAD~6..HEAD` for security regressions, accidental secret values, unrelated refactors, generated artifacts, and compatibility notes. Report exact test/lint/type-check results and remaining non-goals without claiming production readiness.
