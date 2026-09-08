# P1 implementation (original review items 9–15)

Recovered from this conversation's original review on 2026-09-07.

9. Batch embeddings and Qdrant writes: P0 already uses batch embedding and one
   deterministic upsert; preserve that behavior and regression proof.
10. Retrieval quality: configurable cosine threshold, explicit abstention,
    source/page metadata in the prompt, document-injection instructions.
    Reranking was optional in the review and remains an evaluated follow-up.
11. Explicit generated/degraded/insufficient-context status and safe error codes,
    propagated through API, cache, UI and benchmark.
12. Full Docker stack, persistent volumes, health checks, versioned images.
13. Project metadata, pinned dependencies, runtime/dev separation and CI.
14. Loader, failure and invalid configuration tests; opt-in real Redis/Qdrant
    integration tests using temporary namespaces and deterministic embeddings.
15. Reproducible 30-question fixture, Recall@K/MRR, latency percentiles,
    concurrency/failure reporting, cache assertions and human answer-quality rubric.

Work directly on main, as requested. Never report synthetic evaluation as real
production quality. Record live-service verification limitations explicitly.

## Delivery evidence

- Runtime image builds on Linux/Python 3.13 with hash-verified dependencies and
  CPU PyTorch. API, worker, frontend and Redis health probes passed; Qdrant was
  exercised by real integration tests.
- The clean Linux test suite executes 65 tests, including async upload tests and
  a real Redis/Qdrant integration test. A third-party Starlette/AnyIO deprecation
  warning remains; it is not suppressed.
- The public-API 30-question run is recorded in
  `evaluation/results/2026-09-08-smoke.json`: 8 generated, 15 quota-degraded,
  7 abstentions. Full successful-answer performance and human quality scores
  remain unverified. Threshold calibration requires held-out examples.
- Reranking remains optional/unimplemented; JWT/RBAC was not part of items 9–15.
