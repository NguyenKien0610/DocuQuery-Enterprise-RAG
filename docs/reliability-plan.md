# Reliability follow-up

Authorized fixes: task-owned uploads, reset/ingestion races, cross-store recovery,
workspace permissions/resource limits, and more honest RAG evaluation.

Design: single-host API/worker share the uploads volume. Per-workspace OS file
locks replace expiring Redis leases. SQLite on that volume owns document revision,
task generation and cache version. Qdrant writes are immutable candidate revisions;
only published revisions are retrievable. Failed candidates are recorded for
reconciliation. Reset changes metadata first, invalidating pending tasks and cached
answers, then performs retryable cleanup. Legacy unversioned vectors remain visible
until explicitly superseded/reset; never delete old data during upgrade.

Credentials are configured server-side with workspace permissions. The legacy
single API key is confined to DOCUQUERY_WORKSPACE_ID, not arbitrary workspaces.
Apply bounded uploads/workspace bytes/documents, request rate and parser budgets.
No distributed/multi-host deployment claim: migrate metadata/locking together
before moving API/worker to different hosts or unreliable network filesystems.

Verify duplicate failure, queue retry, reset during upload, failed Qdrant writes,
failed metadata commit, cleanup retry, authorization, quotas, and real services.
Evaluation adds held-out Vietnamese/unanswerable questions and retrieval-only
measurement; no invented human scores or claims based on synthetic examples.

## Verification checkpoint (2026-09-08)

Full local regression suite with Redis/Qdrant enabled: **86 passed**, no skips.
Ruff and Mypy pass. Includes duplicate ownership, pending-task cancellation,
metadata publish failure, cleanup failure, workspace roles/rate limits, parser
page/text budgets and actual ingestion-subprocess timeout cleanup.

Backend imports no longer load Torch/sentence-transformers. Model SDKs and text
splitting load on demand; prompt formatting uses Python's existing string format.
Docker forwards the new settings; uv lock and hashed exports are synchronized.

Docker build and API/worker/frontend health checks passed. Public API smoke in
the initially empty `reliability-smoke` workspace uploaded `warranty.txt` twice:
both tasks succeeded (16.35s and 4.12s wall time), one published file and corpus
version 1. A real Gemini query returned `generated` with `warranty.txt` context.
Cross-workspace access returned 403. Reset advanced version to 2, deleted the one
test document with no pending cleanup; listing became empty and the repeated
query returned `insufficient_context`. These observations are functional checks,
not latency or RAG quality benchmark claims.

Upgrade/recovery steps are in `docs/recovery.md`; README architecture now matches
SQLite publication and OS locking.

After user approval, retrieval-only evaluation was implemented on the existing
query pipeline, with cache/LLM bypass and an explicit `retrieved` status. Added 24
question-held-out bilingual/unanswerable cases and separate service-error and
abstention metrics. Final full suite with real Redis/Qdrant: **90 passed**.
Ruff/Mypy pass. The Docker evaluation report is preserved in
`evaluation/results/2026-09-08-retrieval-heldout.json`: Recall@5 4/18, Vietnamese
0/12, no service errors. Evaluation functionality is verified; Vietnamese quality
is demonstrably insufficient and is not claimed fixed. Future model/threshold
experiments require separate calibration and fresh held-out confirmation.

Deployment caution: keep API and worker on the same local uploads volume including
`.state`. Stop old workers before upgrading; old queued messages without ownership
metadata are rejected and must be uploaded again. Preserve uploads, SQLite state,
Qdrant and Redis backups together; do not roll old code onto newly written data
without restoring a consistent pre-upgrade backup. OS locks are not a distributed
locking guarantee, and parser subprocess limits are not an untrusted-code sandbox.
