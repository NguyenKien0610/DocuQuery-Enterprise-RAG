# DocuQuery Architecture Notes

The current implementation is a workspace-scoped RAG prototype. `README.md` is the operator guide; this document records the main design boundaries.

## Ingestion

1. FastAPI authenticates `X-API-Key` and validates `X-Workspace-ID`.
2. The upload is streamed to `uploads/<workspace>/`, hashed, size-limited, and validated by content.
3. The API records task ownership in Redis and dispatches a Celery task.
4. The worker acquires the workspace lock, extracts text, chunks it, creates local embeddings in one batch, and upserts deterministic point IDs into Qdrant.
5. Only after Qdrant accepts the batch does Redis increment the workspace corpus version.
6. A failed worker removes its exact uploaded file.

## Query

1. FastAPI authenticates and validates the request context.
2. Redis supplies the current workspace corpus version.
3. An exact normalized-query key is checked for that workspace and version.
4. On a miss, the query is embedded locally and Qdrant dense search is filtered by workspace.
5. Retrieved text is passed to Gemini for answer generation.
6. The answer and safe citation metadata are cached with a TTL.

## Reset

Reset and ingestion use the same Redis workspace lock. Reset deletes only the selected workspace's files and Qdrant points, then advances that workspace's corpus version. It does not delete the shared collection.

## Security model

The static API key protects the demo from anonymous access. Workspace scoping prevents accidental data mixing across normal flows, but it is not user authorization: any holder of the shared key can submit another valid workspace ID.

The API never returns absolute source paths or raw infrastructure exceptions. File handling rejects empty, oversized, extension-spoofed, structurally invalid DOCX, invalid PDF signatures, and non-UTF-8 text.

## Deferred work

- User identity, roles, organization membership, and per-workspace authorization.
- Sparse retrieval, reranking, score thresholds, and retrieval-quality evaluation.
- OCR, malware scanning, rate limiting, object storage, and document metadata persistence.
- Full-service containers, durable volumes, health probes, metrics, tracing, and concurrent load tests.
