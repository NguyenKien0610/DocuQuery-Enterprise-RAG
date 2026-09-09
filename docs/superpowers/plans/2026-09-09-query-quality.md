# Query quality and resilience implementation plan

Approved in chat: cache failure isolation, expanded Vietnamese evaluation on real
chunks, citation validation with separate evidence grading, three-turn follow-ups.
Work on main as requested; no commit/push, new dependency, model promotion or
modification of existing user data. Historical experiment reports stay unchanged.

## Contracts and tasks

- [x] Cache: in src/rag_engine.py catch RedisError only at answer-cache read/write.
  Malformed entries are misses. Metadata/auth/search failures remain failures.
  Add failing tests in tests/test_query_quality.py, then implement and rerun.
- [x] Citation: share deterministic [Source N] index validation in src/citations.py.
  Generated answers need at least one citation and all indexes in range. Invalid
  output returns retrieved evidence, no generated answer/cache. Valid indexes do
  not prove factual support. Namespace cache to avoid trusting old answers.
- [x] Follow-ups: optional history in QueryRequest, max six messages (three pairs),
  user/assistant roles only, each <=4000 chars. UI sends prior completed pairs,
  not the current question or raw source metadata. Rewrite via existing model,
  bounded JSON standalone question; failure returns degraded, no guessed answer.
  Retrieval-only ignores history and never calls the model. History requests
  bypass answer cache; normal requests retain existing behavior. Test API, engine,
  and real Streamlit flow at the HTTP boundary.
- [x] Evaluation: expanded synthetic document-disjoint calibration/confirmation
  corpus larger than K, no-diacritic/paraphrase/cross-language/hard-negative cases.
  Use production parser/chunker/ingestion and API retrieval, not single-vector
  document experiments. Score relevant evidence spans as well as filenames.
  Add separate human grading import/summary for support versus citation syntax;
  reject missing/duplicate/invalid grades rather than fabricate quality scores.
- [ ] Verify focused RED/GREEN, full pytest, real Redis/Qdrant, Ruff/Mypy, Docker.
  Run actual local embedding retrieval baseline in disposable namespaces if
  available; report quality separately from software correctness. No paid LLM
  calls for testing; deterministic provider doubles at external boundaries.

Ownership: main owns rag_engine.py, schemas/main/frontend and their tests;
independent evaluation worker owns evaluation fixtures/scripts/tests; citation
worker owns standalone citation validation/grading modules and tests only.
Integration review must verify no raw history becomes authoritative evidence,
deleted sources remain filtered, malformed model output fails safely, and cache
errors cannot weaken ownership. Stop after these acceptance conditions pass.

Focused proof (2026-09-09): citation and evaluator tests 34 passed; cache,
history, API and integration tests include real Redis/Qdrant. The frozen
calibration baseline is `evaluation/results/2026-09-09-chunk-calibration-final.json`:
12 production chunks from eight synthetic documents, document Recall@5 0.75 / MRR
0.6042 and evidence Recall@5 0.75 / MRR 0.5833. All three unsupported questions
retrieved evidence rather than abstaining (abstention recall 0.0); this is a
quality finding, not a promotion result. The first pre-freeze baseline was
discarded rather than committed after a fixture changed; no existing report or
user document was modified.
