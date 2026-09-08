# CV Demo Upgrade Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development for task implementation and independent review. User explicitly selected `gpt-5.6-luna`, reasoning `xhigh` for child agents.

**Goal:** Make the existing RAG demo more trustworthy and easier to demonstrate, without disguising its weak Vietnamese retrieval.

**Architecture:** Extend existing FastAPI task status, Streamlit chat and evaluation script. Reuse current authentication, immutable publication, retrieval-only API and JSON reports. No new service, framework, model download or data migration.

**Tech Stack:** Python 3.13, FastAPI, Streamlit, SQLite, existing pytest/Ruff/Mypy.

**Spec:** Acceptance requirements in this document and the scoped design presented in chat. User explicitly approved all three tasks in the current implementation turn.

## Execution ledger

- User authorizes the existing main checkout and preserving uncommitted work; no automatic commits.
- Task 1: complete, controller implemented backend/tests; two regressions reproduced before fix, 32 API tests passed.
- Task 2: complete, Luna xhigh `demo_ui` wrote initial tests then hit usage limit. Controller completed implementation and AppTest verification as previously authorized.
- Task 3: complete, controller implemented shared aggregation, language groups, byte fingerprints, labels and offline summary with tests/docs.
- Preflight: Task 1 changes status handling only, response schema preserved; Task 2 consumes existing retrieval-only schema, no backend changes; Task 3 uses the existing API schema and metric formulas. No shared implementation files or conflicting interfaces.
- Review/integration: controller reviewed changes; 141 tests pass including real Redis/Qdrant integration, 4 Streamlit AppTest scenarios, Ruff clean, Mypy clean for 25 source files. API/worker/frontend Docker builds pass. Independent Luna review remains unavailable because of usage limits; it is not claimed complete. Model changes and Vietnamese experiments remain outside this implementation.

## Global Constraints

- Work on the existing `main` checkout, as the user previously requested. Preserve all earlier uncommitted changes.
- All subagents use `gpt-5.6-luna` with `xhigh`; no child-of-child agents.
- Implementers own only their task files. No automatic commit, push, resets, model changes or destructive migration.
- Do not expose API credentials, environment contents or server filesystem paths in UI/export/report metadata.
- Existing standard query behavior, workspace grants and cache bypass must remain intact.
- Retain the historical evaluation reports unchanged. Do not tune on the held-out questions or invent answer-quality grades.
- Tests use fresh `uploads/.pytest-cv-<task>` basetemp directories, not the user's real metadata.

## Audit baseline

- Last full suite: 90 tests passed, including Redis/Qdrant integration.
- Held-out retrieval: document Recall@5 4/18, Vietnamese 0/12, English 4/6.
- Backend ignores locally failed task state before falling back to Redis/Celery.
- Existing nonterminal SQLite tasks also depend on a separately expiring Redis ownership mapping (24-hour TTL), so task visibility can disagree with durable state.
- Frontend has no retrieval-only control or conversation export; grouped citations do not visibly retain the answer's original `[Source N]` identifiers.
- Evaluator emits aggregate metrics, with no automatic language breakdown or dataset fingerprint.

### Task 1: Authoritative terminal task status

**Owner:** fresh Luna xhigh backend implementer.

**Files:** `src/main.py`, `tests/test_api.py`. Changes to `src/state.py` only if a verified audit issue requires them, with controller approval.

**Interface:** Existing GET `/api/v1/documents/status/{task_id}`, unchanged response schema. SQLite terminal status must precede external broker lookup.

- [x] Add a regression using a real local task row marked failed. Patch Redis/Celery lookups to raise if used; assert HTTP 200, `status == "FAILURE"`, a safe fixed error string and no exception details.
- [x] Verify another authorized workspace cannot read that task, and existing succeeded/cancelled tests still pass.
- [x] Treat an existing workspace-local SQLite row as ownership evidence for nonterminal tasks too; use Redis ownership only when no local row exists (legacy fallback). Test a queued task with an expired Redis mapping and a valid Celery PENDING response, plus another workspace with neither row nor mapping returning 404.
- [x] Implement the narrow terminal branch before broker ownership lookup:

```python
if stored_task and stored_task["status"] == "failed":
    return TaskStatusResponse(
        task_id=task_id, status="FAILURE", error="Document processing failed."
    )
```

- [x] Run focused tests; report RED/GREEN evidence and changed paths, without committing.

### Task 2: Evidence-first demo UI and session export

**Owner:** fresh Luna xhigh frontend implementer.

**Files:** `frontend/app.py`, `tests/test_frontend.py`; optional `tests/test_frontend_demo.py` if Streamlit AppTest cases warrant separation.

**Interfaces:** Backend already accepts `retrieval_only: bool`; returns `retrieved`, `insufficient_context`, `generated` or `degraded`, plus ordered context. Extend `query_backend(question, *, retrieval_only=False)` while preserving its four-element return tuple.

- [x] Test request payloads for ordinary and retrieval-only modes, including existing workspace/key headers.
- [x] Add a visible mode selector: answer generation versus evidence-only search. Evidence-only result must clearly say no answer was generated; never stream an empty fake answer.
- [x] Test citations from interleaved sources `[a, b, a]`; render original identifiers `[Source 1]`, `[Source 2]`, `[Source 3]` even when grouping by filename. Display page when available. Do not reinterpret score as confidence.
- [x] Add a pure Markdown session exporter and a `st.download_button`, restricted to current session messages and safe citation fields. Include query, response status, content and original source numbering. Do not persist chats on the server or export arbitrary dictionary/environment fields.

Use the seam `export_conversation(messages: list[dict]) -> str`, including empty
history and Unicode. Use text/code blocks or escaping so document content is not
treated as trusted UI markup. A concrete acceptance example:

```python
messages = [{"role": "user", "content": "Bảo hành bao lâu?", "api_key": "not-exportable"}]
exported = export_conversation(messages)
assert "Bảo hành bao lâu?" in exported
assert "not-exportable" not in exported
```

- [x] Add tests for Unicode, empty history, source numbering, retrieved/degraded/abstention labels and exclusion of extra secret-like fields.
- [x] Exercise the real Streamlit AppTest with mocked HTTP boundaries: selector changes request mode, evidence renders, download is available after a result. Use existing dependencies only.
- [x] Run focused tests and report evidence. No edits to backend or evaluator by the UI worker.

### Task 3: Auditable evaluation reports

**Owner:** fresh Luna xhigh evaluation implementer.

**Files:** `scripts/evaluate_docuquery.py`, `tests/test_evaluation.py`, `evaluation/README.md`.

**Interfaces:** Additive JSON fields `language_metrics` and `questions_sha256`, preserving existing top-level schema-v2 metric meanings and per-row data. Optional CLI `--run-label` is operator-provided metadata, not verified runtime configuration.

- [x] Add mixed-language tests where answerable and unanswerable denominators differ. Missing language maps to `unknown`; empty denominator returns null. Service failures never become correct abstentions.
- [x] Factor aggregation only as needed to reuse the existing metric formulas for language groups, not a separate scoring algorithm.
- [x] Hash the exact bytes of the loaded questions file and emit the fingerprint. Include an optional run label. Never imply the evaluator verified the embedding model or remote corpus from a caller-provided label.
- [x] Add an offline `--summarize-results PATH` option that prints aggregate and per-language retrieval/abstention/service-error metrics from recorded rows without HTTP, model loading or rewriting the report. Historical reports without language use `unknown`.
- [x] Tests prove fingerprints change when dataset bytes change; summary works without credentials/network; historical row-based reports remain readable; generated-only metrics remain null for retrieval-only runs. Aggregate-only smoke report cannot reconstruct missing rows and is explicitly rejected.
- [x] Document commands and metadata limits. Do not claim cross-run comparability from matching question hashes alone: corpus/model/threshold/config also need recording by the operator.

## Integration and review

- [x] Controller checks task diffs against the pre-existing dirty baseline; no unrelated changes are reverted or committed.
- [ ] Fresh Luna xhigh reviewer returns spec-compliance and quality verdicts for each task; fixes go back to its implementer.
- [x] Run `ruff check src frontend scripts tests` and `mypy src frontend scripts tests --ignore-missing-imports --explicit-package-bases`.
- [x] Run full pytest in a fresh basetemp; enable Redis/Qdrant integration in a random test namespace when services are available.
- [x] Controller final review and real Streamlit AppTest interaction smoke checks (HTTP boundaries mocked; no visual browser QA claimed). No generated quality score replaces the recorded baseline.

## Deferred, not silently solved

Multilingual embedding experiments need a separate calibration set, isolated new collection, consistent model/corpus provenance and fresh held-out confirmation. OCR, hybrid retrieval/reranking, per-document deletion, identity-provider integration and production observability are future slices, not requirements for this demo upgrade.

Audit follow-up: `worker.py` rejects out-of-staging task paths before its failure
cleanup, leaving reservations until reconciliation. Investigate ownership safely
before adding fail-state mutation there: a rejected message must never delete an
external file or invalidate another active task. This is tracked separately from
Task 1's status/visibility fix, not silently declared solved.
