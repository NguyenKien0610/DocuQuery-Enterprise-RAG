# Vietnamese retrieval experiment and safe adoption

User selected Vietnamese retrieval over the prior UI bundle. Child implementers
and reviewers use gpt-5.6-luna, xhigh. Work remains on main; preserve dirty changes.

## Acceptance and scope

1. A reproducible baseline/candidate experiment selects thresholds ONLY on
   calibration questions, freezes them and then evaluates fresh test questions.
2. Use new authored Vietnamese documents/questions, not the old held-out report.
   Relevant calibration and test document sets are disjoint. These are synthetic
   fixtures, not a claim of independent real enterprise-data quality.
3. Candidate: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2;
   baseline: sentence-transformers/all-MiniLM-L6-v2. Both use existing libraries.
4. Threshold grid fixed before running: 0.2, 0.3, 0.35, 0.4, 0.5, 0.6. Select by
   mean(answerable document recall, unanswerable abstention recall), higher
   threshold breaks ties. K=5. Report both metrics separately to expose tradeoffs.
5. Never overwrite original collections, SQLite state, reports or .env. An
   opt-in multilingual deployment must use a separate collection AND uploads
   metadata directory; changing collection alone leaves stale publication rows.
6. No Gemini calls, no reranker, no frontend work, no automatic push/commit.

## Tasks and ownership

- [ ] Luna implementer: scripts/compare_embeddings.py and tests/test_compare_embeddings.py.
  CLI reads corpus/calibration/test JSON, validates disjoint questions, lazily
  loads each encoder, reports cosine retrieval, calibration choice, per-language
  metrics and SHA256 provenance. No external stores. Refuse output overwrite
  without --overwrite. Label whole-document experiment vs production chunking.
- [ ] Controller: author versioned calibration/test corpus before model execution;
  run the actual experiment, check results and preserve reports. Source docs are
  short enough for the candidate's 128-token encoder window; production chunking
  remains a separate end-to-end check.
- [ ] Fresh Luna reviewer: inspect harness/tests for leakage, honest denominators,
  output safety and errors; implementer fixes before adoption.
- [ ] After evidence: Luna deployment implementer supplies isolated opt-in
  configuration and any minimal compatible runtime seam needed for that profile.
  Do not change default model or route existing data to a new embedding space.
- [ ] Controller: run focused/full tests, inspect Docker API retrieval against
  that isolated profile if download/runtime resources permit. Report limitations
  separately from successful implementation gates.

## Promotion gate

Candidate must improve Vietnamese answerable recall over baseline with no worse
unanswerable abstention recall on the fresh test. Otherwise retain it as an
experiment and report failure; do not tune on the test or silently try model after
model against it. A passing small fixture permits an opt-in demo profile, not a
production-quality or universal Vietnamese-support claim.

## Progress

Initial: baseline historical cross-language score was Vietnamese 0/12 on English
documents. No current baseline files or reports modified. Old UI plan deferred.

Checkpoint: real offline comparison saved to
`evaluation/results/2026-09-08-vietnamese-embeddings.json`. Selected thresholds
baseline0.6/candidate0.5; test recall10/12 vs8/12, unanswerable abstention0/6 vs3/6.
Candidate fails the promotion gate; no deployment/model switch performed.
Four focused harness tests pass. Both Luna child turns hit their usage limit
before final implementation handoff and independent review. Preserve work and
await renewed Luna availability or explicit permission for a different worker.

## Follow-up protocol (fixed before E5 calibration)

User authorized the controller to continue directly after Luna's usage limit.
Keep the failed first experiment intact. Next candidate is
`intfloat/multilingual-e5-small`, using mandatory `query: ` / `passage: ` prefixes.
Compare it with the original baseline on calibration ONLY, K=5. Fixed grid:
0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.84, 0.86,
0.88, 0.9, 0.92, 0.94, 0.96, 0.98. Same balanced score and tie rule.
The previously inspected test is not fresh confirmation for this new candidate.
Before any adoption, freeze selected configuration and evaluate a newly authored
document-disjoint confirmation set, then verify the isolated production pipeline.
Calibration success alone does not authorize claiming a quality improvement.

Calibration report: `evaluation/results/2026-09-08-e5-calibration.json`.
Freeze baseline threshold 0.70 and E5 threshold 0.84, K=5 before confirmation.
Calibration answerable recall / unanswerable abstention: baseline 4/6 and 4/6;
E5 5/6 and 5/6. These are selection results, not held-out quality evidence.

Confirmation: `evaluation/results/2026-09-08-e5-confirmation.json`, using only
the three new documents in `evaluation/vietnamese/confirmation/corpus` and its
12 questions. Six answerable Vietnamese queries include two on an English
document; six questions have no supported answer. No threshold search was run
on this set. Baseline recall 1/6, abstention 6/6; E5 recall 4/6, abstention 5/6.
E5 therefore also FAILS the promotion gate. Stop model adoption here; leave
runtime defaults and existing collections/state unchanged. Both reports are
small synthetic document-level measurements, not production chunk benchmarks.
Harness verification after corrections: 9 focused tests pass, Ruff clean,
Mypy with --explicit-package-bases clean for harness and its tests.

## Autonomous follow-up: reproducibility and diagnosis

User authorized choosing and implementing next steps. Added supported
`--selection-report` confirmation replay to the existing harness, pinned revisions
and prefixes, no threshold/model/K overrides, input overwrite protection, query
and positive-calibration-document overlap guards, and a machine-readable gate.
Added pre-threshold ranking diagnostics to distinguish ranking and filtering.
Replay saved separately as `evaluation/results/2026-09-08-e5-confirmation-replay.json`:
all 24 rows exactly match original filenames, abstentions and maximum scores.
No model adoption. With 3 documents and K=5, unfiltered recall is uninformative;
both models have top-1 accuracy 4/6 and MRR 0.7778. Expand the experimental corpus
and cross-language calibration before another model/threshold experiment.

Final follow-up verification: 117 tests passed with DOCUQUERY_INTEGRATION=1
and real Redis/Qdrant (27 focused harness tests included); Ruff clean; Mypy
clean for 24 source files; git diff --check clean apart from line-ending notices.
Integration uses temporary namespaces and fake embedding/LLM boundaries; real
model quality evidence is the separate offline replay, not this test count.
No commit/push or model-default change. Luna independent review is still unavailable.
