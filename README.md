# DocuQuery — Workspace-Scoped RAG

DocuQuery is a CV-oriented document question-answering prototype built with FastAPI, Celery, Redis, Qdrant, local sentence-transformer embeddings, Gemini, and Streamlit.

It demonstrates asynchronous ingestion, dense vector retrieval, page-aware citations, exact-query response caching, workspace isolation, and defensive upload handling. It is not presented as a production multi-tenant platform; the security boundary and remaining limitations are documented below.

## What it demonstrates

- Asynchronous PDF, DOCX, and UTF-8 TXT ingestion through Celery.
- Local `all-MiniLM-L6-v2` embeddings, so indexing does not consume Gemini quota.
- Dense similarity search in Qdrant with workspace filters.
- Gemini answer generation using retrieved document context.
- Exact-query Redis caching keyed by workspace and corpus version.
- Automatic cache invalidation after successful ingestion or reset.
- Revision-scoped vector IDs and idempotent publication of duplicate document content.
- Source citations with safe filename, document ID, chunk index, and PDF page number.
- Upload size, signature, structure, UTF-8, and DOCX expansion checks.
- API-key protection and logical workspace isolation.

## Architecture

```mermaid
sequenceDiagram
    participant UI as Streamlit
    participant API as FastAPI
    participant R as Redis
    participant W as Celery Worker
    participant S as Local uploads + SQLite
    participant Q as Qdrant
    participant L as Gemini

    UI->>API: Upload + API key + workspace
    API->>S: Reserve workspace capacity
    API->>API: Stream, validate, hash unique staging file
    API->>R: Record task ownership
    API->>R: Enqueue task
    R->>W: Deliver task
    W->>S: Acquire OS workspace lock; validate task generation
    W->>W: Parse, chunk, embed batch
    W->>S: Journal candidate revision
    W->>Q: Upsert revision-scoped points
    W->>S: Persist file; publish revision + corpus version
    W->>S: Release lock

    UI->>API: Query + API key + workspace
    API->>S: Read corpus version
    API->>R: Read versioned exact-query cache
    alt Cache hit
        R-->>API: Cached answer and citations
    else Cache miss
        API->>Q: Dense search filtered by workspace + published revisions
        Q-->>API: Relevant chunks
        API->>L: Context + question
        L-->>API: Answer
        API->>R: Cache answer under current corpus version
    end
    API-->>UI: Answer and safe citations
```

### Cache correctness

Answers use this logical Redis key:

```text
rag:answer:v3:<workspace_id>:<corpus_version>:<threshold>:<top_k>:<sha256(normalized_query)>
```

Successful publication and reset increment the workspace's SQLite corpus version.
Old Redis answers expire through TTL but become unreachable after the version
changes. Candidate vector revisions remain hidden until SQLite publishes them.

## Security boundary

Every `/api/v1` request requires:

```http
X-API-Key: <DOCUQUERY_API_KEY>
X-Workspace-ID: <workspace-id>
```

The API key is compared in constant time and the service fails closed when no server key is configured. Workspace IDs are validated slugs and scope files, vectors, cache entries, tasks, queries, and reset operations.

The legacy key is confined to `DOCUQUERY_WORKSPACE_ID`. Optional
`DOCUQUERY_CREDENTIALS` grants static keys reader/writer/owner roles in explicit
workspaces; see `.env.example`. Owners alone may reset/reconcile. Workspace rate,
storage and document limits are enforced. JWT/OAuth, user lifecycle management,
malware scanning and cryptographic tenant isolation remain outside this demo.

## Project structure

```text
DocuQuery-Enterprise-RAG/
├── frontend/app.py
├── scripts/benchmark_docuquery.py
├── src/
│   ├── main.py
│   ├── rag_engine.py
│   ├── schemas.py
│   ├── security.py
│   ├── uploads.py
│   └── worker.py
├── tests/
├── .env.example
├── docker-compose.yml
└── requirements.txt
```

## Requirements

- Python 3.13 (the version locked and tested in CI)
- Docker Desktop or Docker Engine
- A Gemini API key

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
pip install --require-hashes --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-dev.txt
```

Copy `.env.example` to `.env` and set at least:

```env
GOOGLE_API_KEY=your_gemini_key_here
DOCUQUERY_API_KEY=replace_with_a_long_random_value
DOCUQUERY_WORKSPACE_ID=default
```

Generate a long random API key rather than using the example value.

Start the complete stack (API, worker, frontend, Redis and Qdrant):

```bash
docker compose up --build -d
```

Open `http://localhost:8501`. First startup downloads the embedding model;
API readiness may take several minutes. Data and models use named volumes.
`docker compose down` preserves these volumes; `down -v` deletes their data.
Ports bind only to localhost. If Redis port 6379 is occupied, set
`REDIS_PUBLISHED_PORT=16379` in `.env`; container connections still use 6379.
`API_PUBLISHED_PORT` and `FRONTEND_PUBLISHED_PORT` similarly override host ports
8000 and 8501 without changing service-to-service connections.

For local Python development, start only `docker compose up -d redis qdrant`,
then run the following commands. Export `.env` settings in the frontend terminal
(Streamlit does not load `.env` automatically). Set local `REDIS_PORT` to the
published port if overridden.

Start the API locally:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

Start the worker in another terminal:

```bash
celery -A src.worker.celery_app worker --loglevel=info --pool=solo
```

`--pool=solo` is the safest Windows development default.

Start the frontend:

```bash
streamlit run frontend/app.py
```

The frontend reads `DOCUQUERY_API_BASE_URL`, `DOCUQUERY_API_KEY`, and `DOCUQUERY_WORKSPACE_ID` from the environment.

### Demo controls

- **Answer generation** uses the normal RAG answer path. **Evidence only** searches
  without Gemini or the answer cache and explicitly labels that no answer was generated.
- Open the source expander to inspect literal document excerpts and available page
  numbers. `[Source N]` always refers to the original response order, including
  when multiple excerpts from one file are grouped together.
- **Download conversation (.md)** exports this browser session's questions,
  response statuses/content and citations. It does not create server-side chat
  history or include credentials/arbitrary response metadata. Source filenames
  are reduced to basenames; excerpt and chat text remain verbatim in fenced blocks.
  Review the content before sharing: uploaded documents or your questions may
  themselves contain sensitive information.

Generation failures and insufficient evidence have distinct labels, retained
when the page reruns. Evidence-only search is useful for demos without generation
quota, but a retrieved passage does not by itself establish an answer.

## API

The examples assume:

```bash
API_KEY=replace_with_a_long_random_value
WORKSPACE=default
```

### Upload

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -H "X-API-Key: $API_KEY" \
  -H "X-Workspace-ID: $WORKSPACE" \
  -F "file=@report.pdf"
```

```json
{
  "task_id": "e7b6d87c-f384-478d-b314-14e1db174718",
  "document_id": "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
}
```

Default limits are 25 MiB uploaded and 100 MiB declared uncompressed DOCX content. PDFs require a PDF signature, DOCX files require the expected ZIP members, and text files must contain non-whitespace UTF-8 text.

### Task status

```bash
curl http://localhost:8000/api/v1/documents/status/<task-id> \
  -H "X-API-Key: $API_KEY" \
  -H "X-Workspace-ID: $WORKSPACE"
```

A task ID is visible only from the workspace that created it. Worker failures return a generic message without internal paths or exception details.

Durable SQLite terminal state (success, failure, reset cancellation) is returned
before querying Redis/Celery. A workspace-local task row also establishes ownership
when the Redis mapping expires. Legacy tasks without a local row still require the
Redis workspace mapping; nonterminal broker outages remain HTTP 503.

### List documents

```bash
curl http://localhost:8000/api/v1/documents \
  -H "X-API-Key: $API_KEY" \
  -H "X-Workspace-ID: $WORKSPACE"
```

### Delete one managed document

Fetch `GET /api/v1/documents/managed` with the same headers to obtain the
`document_id` and current `revision`, then use an **owner** credential:

```bash
curl -X DELETE "http://localhost:8000/api/v1/documents/<document-id>?revision=<revision>" \
  -H "X-API-Key: $API_KEY" \
  -H "X-Workspace-ID: $WORKSPACE"
```

The sidebar also provides selection and explicit confirmation. Deletion immediately
unpublishes that revision and invalidates the workspace answer cache. A stale
revision or active upload returns HTTP 409; refresh or wait for processing to finish.
Repeating a completed deletion is a no-op. If `cleanup_pending` is true, retry
`POST /api/v1/workspace/reconcile` after storage recovers.

Legacy-only files are not offered for deletion. Tombstones keep older unversioned
vectors of a deleted managed document out of search; they do not erase those legacy
copies, backups, historical chat/exports, or responses already in flight. See
[recovery and upgrade guidance](docs/recovery.md).

### Query

Answer caching is best-effort: Redis read/write failures do not discard a usable
answer. Malformed cache entries are misses. Authentication, publication metadata
and search failures are not bypassed. New answers and cache hits must contain
canonical, in-range `[Source N]` citations; invalid citations return `retrieved`
with an empty answer and `error_code: invalid_citations`, preserving the evidence.
This checks citation syntax/indexes, **not factual support**. Old answer-cache
keys expire naturally under their existing TTL and are not reused by this version.

Optional `history` contains at most three completed user/assistant pairs (six
messages, up to 4000 characters each):

```json
{"query":"How long does it last?","history":[{"role":"user","content":"What is covered by the warranty?"},{"role":"assistant","content":"Manufacturing defects [Source 1]."}]}
```

With history, the existing model first rewrites a standalone search question,
then retrieves current published evidence. History is untrusted context, never
answer evidence, and follow-up requests bypass the shared answer cache. This adds
a model step, latency and possible provider cost; rewrite failures return
`degraded` / `rewrite_unavailable` and ask for a standalone question. The server
does not persist conversation history. The frontend sends only the three latest
answered pairs, without source metadata, truncating each message to 4000 characters.
`retrieval_only` ignores history, never invokes the model, and requires a standalone
question. Previously deleted documents remain excluded even if mentioned in history.

For chunk-level Vietnamese retrieval measurements, use
[the expanded evaluation corpus](evaluation/vietnamese/chunks/README.md).
For separate human checks of whether answers are actually supported by their
sources, use [answer grading](evaluation/answer-grading.md).

Optional `retrieval_only: true` returns evidence without Gemini or answer-cache
access. Its nonempty status is `retrieved` and `answer` is empty; ordinary queries
keep their existing behavior. See [evaluation instructions](evaluation/README.md)
for bilingual held-out questions and separate abstention/service-error metrics.

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -H "X-Workspace-ID: $WORKSPACE" \
  -d '{"query":"Summarize the document"}'
```

```json
{
  "query": "Summarize the document",
  "answer": "...",
  "cached": false,
  "status": "generated",
  "error_code": null,
  "context": [
    {
      "source_file": "report.pdf",
      "document_id": "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
      "chunk_index": 0,
      "page_number": 1,
      "text": "Relevant excerpt..."
    }
  ]
}
```

No absolute server path is returned.

`status` is `generated`, `degraded`, or `insufficient_context`. Embedding/provider
outages return safe error codes (`embedding_unavailable`, `generation_unavailable`)
and are never cached. Qdrant/Redis failures remain HTTP 503. Empty retrieval
abstains without calling Gemini. `RAG_SCORE_THRESHOLD` defaults to 0.35 (cosine);
this is a starting value, not a calibrated universal threshold. Prompts carry
source/page labels and instructions to treat documents as untrusted data.

Send `"use_cache": false` in query JSON for an uncached evaluation. Cache keys
include a pipeline revision, threshold and top-K, so P0 answers are not reused.

### Reset one workspace

```bash
curl -X DELETE http://localhost:8000/api/v1/workspace/reset \
  -H "X-API-Key: $API_KEY" \
  -H "X-Workspace-ID: $WORKSPACE"
```

Reset shares a local OS workspace lock with ingestion and deletes only that workspace's uploaded files and Qdrant points.

## Testing

Unit/API tests mock external service boundaries and do not require live Redis, Qdrant, Gemini, or Hugging Face downloads:

```bash
pytest tests -q
```

The suite covers authentication, workspace validation, upload safety and cleanup, task ownership, cache versioning, lazy model construction, deterministic vector IDs, filtered retrieval, worker cleanup, reset isolation, and safe error responses.

Set `DOCUQUERY_INTEGRATION=1` to additionally test real Redis/Qdrant ingestion,
deduplication, isolation, cache invalidation and locking. The test allocates a
random collection/workspace and cleans only that namespace. Gemini and embeddings
are deterministic test doubles, so this test does not measure retrieval quality.

`/health/live` checks the API process; `/health/ready` checks authentication
configuration and Redis/Qdrant connectivity. It does not guarantee Gemini quota
or worker availability. Compose checks the worker separately (a solo worker can
temporarily miss health pings while processing a long document).

Dependencies are defined in `pyproject.toml` and resolved in `uv.lock`. Runtime
and development requirements are hash-pinned exports. To update intentionally:

```bash
uv lock
uv export --frozen --no-dev --no-emit-project --output-file requirements.txt
uv export --frozen --no-emit-project --output-file requirements-dev.txt
```

CPU PyTorch is selected on Linux/Windows. CI installs the hashed development
export, runs tests against Redis/Qdrant, lint/type checks and a Docker build.

## Benchmark

With the API and worker running:

```bash
python scripts/benchmark_docuquery.py \
  --file path/to/document.pdf \
  --query "Summarize the document" \
  --api-key "$API_KEY" \
  --workspace-id "$WORKSPACE" \
  --reset
```

The script measures ingestion latency, first-query latency, repeated exact-query cache latency, and speedup. It is a local latency benchmark, not a retrieval-quality or concurrent-load evaluation.

It rejects degraded/insufficient answers and verifies a cold query followed by
actual cache hits. For the separate 30-question retrieval/concurrency evaluation,
see [evaluation guide](evaluation/README.md). Human faithfulness/relevance grades
are deliberately left unscored until reviewed; no fabricated quality scores are
provided.

The [Vietnamese retrieval experiments](evaluation/vietnamese/README.md) include
calibration-only model selection, frozen-revision replay, per-query error analysis
and an explicit promotion gate. Both multilingual candidates failed the recorded
gate; the default model is unchanged. These small synthetic experiments measure
retrieval, not generated-answer safety or production Vietnamese quality.

## Upgrade note

Read [storage upgrade and recovery](docs/recovery.md) before deployment. API and
worker must share a local uploads volume including SQLite metadata. Stop old
workers and back up all stores together before upgrading. Workspace-scoped legacy
vectors remain eligible until superseded/reset; vectors with no workspace remain
invisible. Do not delete old files or metadata as an automatic migration step.

## Current limitations

- Static key/workspace roles, without identity-provider integration or user lifecycle management.
- Dense retrieval without sparse search or reranking.
- Text-only PDF extraction; scanned documents need OCR before upload.
- Single-host SQLite metadata/OS locking; no distributed-host guarantee. Per-document deletion requires a managed revision and no active workspace uploads.
- No production monitoring, distributed tracing, or load-test guarantee.
- The evaluation fixture is small and synthetic; broader held-out data and human grading are needed.

## License

Licensed under the terms of [LICENSE](./LICENSE).
