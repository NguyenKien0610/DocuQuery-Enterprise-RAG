# DocuQuery P0 Hardening Design

**Date:** 2026-09-07

**Status:** Approved in chat

## Goal

Make the current DocuQuery prototype safe and truthful enough for a CV demo by fixing stale-cache behavior, isolating workspaces, protecting API operations, validating uploads, removing import-time model downloads, and making repeated ingestion idempotent.

## Scope

This change covers the existing upload, task-status, document-listing, query, and workspace-reset flows. It preserves FastAPI, Celery, Redis, Qdrant, Streamlit, local sentence-transformer embeddings, and Gemini generation.

The implementation will:

- Require `X-API-Key` on every `/api/v1` endpoint.
- Require a validated `X-Workspace-ID` on document, query, and reset requests.
- Isolate uploaded files, Qdrant points, cache entries, and reset behavior by workspace.
- Version the corpus per workspace so cached answers cannot survive corpus changes.
- Describe the cache as exact-query caching and retrieval as dense retrieval.
- Reject empty, oversized, extension-spoofed, or structurally invalid uploads before dispatch.
- Avoid returning server paths or raw exception messages to clients.
- Construct embedding and LLM clients lazily.
- Use content-derived document IDs and deterministic Qdrant point IDs.
- Serialize ingestion and reset operations with a Redis workspace lock.

## Non-goals

- JWT, OAuth, users, roles, or per-user authorization.
- Cryptographic tenant isolation. A caller holding the shared API key can choose a workspace ID.
- True semantic caching based on embedding similarity.
- Sparse/BM25 retrieval, fusion, or reranking.
- A document metadata database or per-document delete endpoint.
- Full containerization, production orchestration, observability, or load testing.
- OCR or malware scanning.

These remain follow-up work. Documentation must not imply they already exist.

## API contract

### Authentication and workspace

All `/api/v1` routes require:

```http
X-API-Key: <DOCUQUERY_API_KEY>
X-Workspace-ID: <workspace slug>
```

`X-API-Key` is compared with `hmac.compare_digest`. If the server key is not configured, protected requests fail closed with HTTP 503. A missing or incorrect client key returns HTTP 401.

Workspace IDs are 1-64 characters and may contain ASCII letters, digits, `_`, and `-`; the first character must be alphanumeric. Invalid or missing IDs return HTTP 422. The validated value is passed explicitly through the API, worker, retrieval, cache, and reset layers.

### Upload

`POST /api/v1/documents/upload` continues to accept multipart `file` and returns `task_id`, plus a content-derived `document_id`.

The API streams the upload to a workspace-specific directory while computing SHA-256 and enforcing `MAX_UPLOAD_BYTES` (default 25 MiB). It rejects empty files. After writing, it validates:

- PDF: `%PDF-` signature.
- DOCX: valid ZIP containing `[Content_Types].xml` and `word/document.xml`, with bounded declared uncompressed size.
- TXT: valid UTF-8 text containing at least one non-whitespace character.

Failed validation removes the temporary file. Failure to dispatch the Celery task removes the saved file. Successful uploads are stored below `UPLOAD_DIR/<workspace_id>/` with the document digest in the filename.

The task receives `file_path`, `workspace_id`, `document_id`, and the sanitized original filename.

The API allocates the Celery task ID, records `rag:task_workspace:<task_id>` with a 24-hour TTL, and dispatches the task with that ID. Status requests return HTTP 404 unless the recorded workspace matches the request workspace. A dispatch failure removes both the task mapping and saved file. A worker failure also removes its uploaded file.

### Query and citations

`POST /api/v1/query` validates and trims a non-empty query. Retrieval applies a Qdrant payload filter for `workspace_id`.

Citation responses include `source_file`, `document_id`, `chunk_index`, `page_number`, and `text`. They do not include `source_path`.

Client-visible failures use stable generic messages. Full exceptions are written to server logs only.

### List and reset

`GET /api/v1/documents` lists files only from the selected workspace directory.

`DELETE /api/v1/workspace/reset` acquires the same workspace lock used by ingestion, deletes only matching Qdrant points and workspace upload files, and advances that workspace's corpus version. It does not delete or recreate the global collection.

## Cache model

Redis keys use this logical structure:

```text
rag:corpus_version:<workspace_id>
rag:answer:<workspace_id>:<corpus_version>:<sha256(normalized_query)>
```

The absent version is `0`. A successful ingestion increments the version only after Qdrant accepts the complete point batch. A reset increments it only after vector and file deletion succeeds. Old answer keys are allowed to expire naturally through the existing TTL, avoiding an expensive key scan.

The cache remains exact-query caching: normalization is trim plus lowercase. Documentation and UI must use that name.

## Ingestion model

Loaders return page-aware text sections. Ingestion extracts and chunks the entire document, creates embeddings in one batch, then sends one Qdrant upsert containing all points. The configured upload limit bounds request size and memory use.

Point IDs are UUIDv5 values derived from `workspace_id`, `document_id`, and chunk index. Reprocessing identical content in the same workspace overwrites the same points instead of creating duplicates. Each payload contains `workspace_id`, `document_id`, `source_file`, `text`, `chunk_index`, and `page_number`.

Both ingestion and reset acquire `rag:workspace_lock:<workspace_id>` with finite acquisition and lease timeouts. Failure to acquire the lock produces a retryable task failure or HTTP 409 reset response. Query requests do not take the lock: a query racing with a completed corpus mutation may populate the old version, but the subsequent version increment makes that entry unreachable.

The lock lease is configured by `WORKSPACE_LOCK_TIMEOUT_SECONDS` (default 600 seconds) and acquisition waiting by `WORKSPACE_LOCK_BLOCKING_TIMEOUT_SECONDS` (default 1 second). DOCX declared uncompressed content is bounded by `MAX_EXTRACTED_BYTES` (default 100 MiB).

## Dependency lifecycle

Redis and Qdrant clients may remain cheap module-level objects because construction performs no network call. Hugging Face embeddings and the Gemini client move behind memoized getter functions. Importing `src.main` or collecting unit tests must not download a model or require a Google API key.

Application startup still verifies Qdrant collection availability. Tests replace this boundary before entering `TestClient`.

## Failure behavior

- Invalid authentication: 401 without revealing which credential field was wrong.
- Missing server API key: 503 with a configuration-safe message.
- Invalid workspace or request payload: 422.
- Invalid/empty/oversized upload: 400 or 413; no file or task remains.
- Celery dispatch failure: 503; saved file is removed.
- Workspace busy during reset: 409.
- Retrieval/generation infrastructure failure: 503 with a generic response; internal exception is logged.
- Celery task failure: task status exposes a generic failure message, not `str(exception)`.

Existing Gemini-only degraded responses may remain, but they must not include raw exception text.

## Testing strategy

All behavior changes follow red-green-refactor. Unit/API tests must run with no external network and without live Redis, Qdrant, Celery, Hugging Face, or Gemini.

Required regression coverage:

- Importing the app does not construct embedding or LLM clients.
- Missing/wrong API key is rejected and valid credentials pass.
- Missing/invalid workspace ID is rejected.
- Empty, oversized, spoofed, and malformed uploads are rejected and cleaned up.
- A dispatch failure removes the saved file.
- Identical content produces the same document and point IDs.
- Retrieval includes the workspace filter and omits source paths.
- Cache keys differ across workspaces and corpus versions.
- Successful ingestion advances the corpus version; failed ingestion does not.
- Reset deletes only the requested workspace and cannot race an acquired ingestion lock.
- Existing cache-hit, cache-miss, task status, and frontend request helpers are updated for the new headers and schemas.

A final verification run includes pytest, Ruff, Mypy, Python compilation, Docker Compose configuration validation, and a clean Git status check apart from intended source changes.

## Compatibility and rollout

The API header requirement and removal of `source_path` are intentional breaking changes. README examples and Streamlit are updated in the same change. `.env.example` gains `DOCUQUERY_API_KEY`, `DOCUQUERY_WORKSPACE_ID`, `MAX_UPLOAD_BYTES`, `MAX_EXTRACTED_BYTES`, `WORKSPACE_LOCK_TIMEOUT_SECONDS`, and `WORKSPACE_LOCK_BLOCKING_TIMEOUT_SECONDS`.

Existing vectors lack workspace metadata and are therefore invisible to the new filtered retrieval. Users must reset/re-ingest documents after upgrading. The README will state this migration step; no automatic migration is added.

## Acceptance criteria

1. The full unit suite passes without network access.
2. Repeating a question after successful ingestion cannot return the earlier corpus version's answer.
3. Two workspaces cannot retrieve, list, cache, or reset each other's data through normal API flows.
4. Unauthorized callers cannot use `/api/v1` operations.
5. Invalid uploads never reach Celery and leave no stored file.
6. Reprocessing identical content does not add duplicate Qdrant point IDs.
7. Reset and ingestion for one workspace are mutually exclusive.
8. API responses contain no absolute server path or raw infrastructure exception.
9. README and `docs.md` no longer claim semantic cache, hybrid retrieval, Gemini embeddings, or unsupported enterprise guarantees.
