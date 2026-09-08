# Safe per-document deletion

User authorizes functional improvements, not demo preparation. Implement on the
existing main worktree; preserve prior changes, reports, collections and .env.
No commit/push, new service/dependency, model change or actual user-document deletion.

## Contract

- GET /api/v1/documents/managed returns published IDs/revisions/names/size/chunks,
  never filesystem paths. Keep the existing filename-list endpoint unchanged.
- DELETE /api/v1/documents/{document_id}?revision=... requires owner permission.
  Validate SHA256 document ID and immutable revision. Unknown document -> 404;
  stale revision or busy workspace -> 409. A repeated completed delete is a no-op.
- Under existing workspace lock and SQLite transaction, unpublish the exact
  revision, record a deletion tombstone and cleanup journal, bump cache version.
  Keep unrelated documents and workspace generation unchanged.
- Tombstones exclude old unversioned vectors of that document even after cleanup;
  otherwise legacy fallback could resurrect deleted content. Re-upload is allowed
  with a fresh revision; a stale delete must not affect the new publication.
- Refuse deletion while any queued/uploading task exists in that workspace.
  This conservative single-host bound avoids pending uploads republishing content.
- Existing reconciliation performs retryable physical cleanup. Successful logical
  deletion may return cleanup_pending=true; it is not secure erasure of backups,
  historical chat or copies already returned by an in-flight query.
- UI provides one-document selection and explicit confirmation, uses the revision
  from the catalog, and reports permission/conflict/cleanup state without raw errors.
  Legacy-only files remain read-only through this control; no guessed ownership.

## Compatibility

Expand SQLite with one tombstone table (no rewriting existing records). All API
and worker processes must be upgraded together; old readers ignore tombstones.
Rollback after deletion requires the existing coordinated backup/restore process,
not running old code against new state. No automatic backup deletion or migration.

## Proof / execution

- [x] State/filter regression: exact revision, idempotence, stale retry, legacy
  exclusion after reconciliation, unrelated documents, active upload conflict.
- [x] API validation/permissions and safe metadata response.
- [x] UI confirmation and a real AppTest delete flow with HTTP boundary mocked.
- [x] Real Qdrant/Redis test proves deleted document absent and answer cache stale.
- [x] Full pytest, Ruff/Mypy, Docker build; document caveats and result.

Verification (2026-09-09): full pytest with DOCUQUERY_INTEGRATION=1 and Redis
port 16379: 153 passed. Focused real deletion/ingestion integration: 8 passed.
Ruff clean; Mypy clean across 26 source files; Docker API/worker/frontend builds
passed. Tests use temporary files/SQLite and UUID-scoped collections/cache keys;
no existing user documents were deleted. Main implemented and self-reviewed;
independent Luna review unavailable due to quota. No commit or push.
