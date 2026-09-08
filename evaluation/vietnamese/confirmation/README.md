# Frozen-threshold confirmation

Synthetic fixture authored after calibration selected baseline threshold 0.70
and multilingual-e5-small threshold 0.84. K=5. The confirmation corpus contains
only these three new documents, disjoint from the calibration corpus. All queries
are Vietnamese; the expense document is English. No real company policies/data.

Evidence: `evaluation/results/2026-09-08-e5-confirmation.json` records model
revisions, input SHA256 fingerprints, per-query sources/scores and metrics.
Thresholds come from `2026-09-08-e5-calibration.json`, never this question set.

The original offline run used `scripts.compare_embeddings` helpers. A supported
CLI now replays this experiment directly from the calibration report, without
threshold selection or model/K overrides. It pins the recorded model revision,
applies the recorded input prefixes, and rejects positive calibration documents
or calibration queries in the confirmation set. It does not establish that a
human has never inspected the test; replaying this fixture is NOT fresh evidence.

From the repository root, with project dependencies and model weights installed:

```bash
python -m scripts.compare_embeddings --corpus evaluation/vietnamese/confirmation/corpus --questions evaluation/vietnamese/confirmation/questions.json --selection-report evaluation/results/2026-09-08-e5-calibration.json --output benchmark_results_confirmation.json
```

For offline execution, set `HF_HUB_OFFLINE=1` after both pinned revisions are
cached. Missing weights fail the run rather than falling back to another model.
An existing output is rejected unless `--overwrite` is explicit; inputs cannot
be used as output even with that flag. No Gemini, Redis, Qdrant or API is needed.
All TXT documents are single vectors; no production chunking or answer generation.

Replay evidence: `evaluation/results/2026-09-08-e5-confirmation-replay.json`.
All 24 model/query rows reproduce the original retrieved filenames, abstentions
and maximum scores exactly on the cached CPU environment. Original reports were
not overwritten. New reports include a selection-report SHA256, unfiltered
ranking diagnostics and an explicit promotion-gate verdict. CLI exit 0 means
measurement completed; inspect `promotion_gate.passed` for the quality verdict.

| Model | Answerable recall@5 | Unanswerable abstention |
| --- | --- | --- |
| all-MiniLM-L6-v2 | 1/6 | 6/6 |
| multilingual-e5-small | 4/6 | 5/6 |

E5 fails the predeclared no-abstention-regression gate despite better recall.
Do not promote it or tune on this now-inspected confirmation set. Its result is
not directly comparable with historical reports using different corpora.

## What the errors actually show

- Both models rank the relevant document first for 4/6 answerable questions;
  unfiltered MRR is 0.7778 for both. The thresholded recall difference alone
  is not evidence of a ranking improvement.
- Unfiltered Recall@5 is trivially 1.0 here because K exceeds the three-document
  corpus size. Use top-1 accuracy/MRR for this diagnostic, not that recall figure.
- E5 rejects two answerable cross-language expense queries below 0.84. Their
  relevant document also does not rank first, so threshold rejection is not a
  complete explanation of the ranking weakness.
- E5 retrieves the library policy for an unsupported late-fee question. This
  is a retrieval-level missed abstention, NOT an observed hallucinated answer:
  this experiment never generated answers.
- `error_type` identifies incomplete top-K coverage (`ranking_miss`), otherwise
  evidence lost to filtering (`threshold_rejection`), or retrieval for a question
  without supported facts (`unsupported_context`). It is a diagnostic priority,
  not a claim that a query can have only one underlying problem.

Next quality experiment should expand calibration with cross-language documents
and hard same-topic negatives, then freeze configuration before authoring another
document-disjoint confirmation set with more documents than K. Do not change the
old thresholds or recycle this seen set as a new promotion test. Answerability
requires separate evidence beyond document similarity; no answer-safety claim
or deployment promotion follows from this replay.

E5 input conventions and score-range caveats are documented in the
[official model card](https://huggingface.co/intfloat/multilingual-e5-small).
