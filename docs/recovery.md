# Storage, upgrade and recovery

This deployment supports one host. API and worker must share the **same local
uploads volume**, including `.state` and `.staging`. Do not put SQLite or lock files
on an unreliable network filesystem or give replicas separate volumes.

## What commits a document

An upload reserves capacity and owns a unique staging file. The worker holds an
OS workspace lock, journals a candidate revision, writes Qdrant points and a
durable document copy, then publishes the revision and cache version in one SQLite
transaction. Search includes only published revisions and eligible legacy points.
An incomplete candidate is hidden even if Qdrant accepted part of the write.

Reset advances the workspace generation and cache version first. Pending uploads
from an older generation cannot publish. Physical deletion can fail independently;
`cleanup_pending: true` means the logical reset succeeded but storage reclamation
still needs a retry. It does **not** promise secure erasure of backups.

## Upgrade checklist

Per-document deletion adds a SQLite `deleted_documents` tombstone table without
rewriting existing records. Upgrade API and worker together: older readers ignore
these tombstones and can expose legacy copies of logically deleted documents.
After deletions, rollback requires the coordinated restore described below.

1. Stop incoming traffic and stop the old API and worker. Do not run old and new
   workers concurrently against these volumes.
2. Back up uploads (including hidden files), Qdrant and Redis while writers are
   stopped. Record the old image/commit and configuration. Protect backups as
   document data and credentials, and test restoring them in an isolated setup.
3. Configure workspace grants. The legacy `DOCUQUERY_API_KEY` now grants owner
   access only to `DOCUQUERY_WORKSPACE_ID`. For multiple workspaces, configure
   `DOCUQUERY_CREDENTIALS` as illustrated in `.env.example`.
4. Workspace IDs must be lowercase slugs. If an old workspace used uppercase,
   plan a controlled migration of both files and vector payloads; do not rename
   just its upload directory or assume case-insensitive isolation on Windows.
5. Start the new services. Existing unversioned vectors/files remain available;
   startup does not reset collections. Old queued messages without task ownership
   metadata are rejected; upload those documents again if needed.
6. Verify readiness, authorized listing, an upload and a query before restoring
   traffic. Test reset only in a disposable workspace.

Rollback after new writes requires restoring a consistent pre-upgrade backup of
all stores and the old image/configuration. Simply reverting code can expose
unpublished vector revisions to the old search filter. New writes after the
backup will be lost on restoration; obtain approval before restoring.

## Retry cleanup

An owner can POST `/api/v1/workspace/reconcile` with `X-API-Key` and
`X-Workspace-ID`. Retry after Qdrant/storage recovers; inspect `cleanup_pending`.
Do not delete `.state` to clear an error: it contains the publication and reset
history and deletion tombstones that prevent hidden revisions from becoming visible.

Owner-only per-document deletion journals cleanup of the exact managed revision.
It preserves other documents and rejects requests while uploads are active in the
workspace. Reconciliation retries managed-vector/file cleanup; tombstones remain
to hide older unversioned copies. This is not secure erasure of legacy copies,
backups, historical conversations, exports or responses already in flight.

Reconciliation also expires abandoned upload reservations after
`DOCUMENT_TASK_TTL_SECONDS` and removes their staging files. It is explicit,
not a background scheduler: arrange an operator call if no further ingestions
occur. Physical files awaiting cleanup still count toward storage capacity.

## Limits and observability

Use `.env.example` for upload/workspace quotas, parser page/text budgets, rate
limits and subprocess deadlines. A process timeout terminates the ingestion
child; its unpublished revision remains invisible and can be reconciled.
Parsing has bounded duration/output, but this is not a hardened document sandbox
or a strict per-process RAM cap. Restrict service access and monitor disk/RAM.

HTTP 403 indicates a missing grant, 409 a busy workspace, 413 a size/capacity
limit, 429 request throttling, and 503 unavailable infrastructure. Inspect server
logs for details without exposing document paths or provider secrets to clients.
Back up and monitor the SQLite metadata alongside the document and vector stores.
