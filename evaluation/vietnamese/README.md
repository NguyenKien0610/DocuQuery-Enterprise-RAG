# Vietnamese retrieval experiment

This is an authored synthetic fixture, not real company policy. All amounts,
names and rules are fictional. It must not be represented as an independent
enterprise benchmark or a human-graded answer-quality evaluation.

The six short documents are indexed together. Calibration uses the three
`cal-*` documents as relevant sources; test questions use the three `test-*`
documents. The relevant document sets and question strings are disjoint. There
are 12 calibration questions (6 answerable, 6 unanswerable) and 18 test questions
(12 answerable, 6 unanswerable). Both sets include topically related questions
whose requested facts are absent, not just obviously unrelated queries.

Before executing encoders, the threshold grid was fixed at
`[0.2, 0.3, 0.35, 0.4, 0.5, 0.6]`, with top K 5. Each model's threshold is chosen
on calibration by the mean of answerable document recall and unanswerable
abstention recall. Ties choose the higher threshold. Test labels are never used
for threshold selection. Keep the grid and failed results in the report.

Baseline: `sentence-transformers/all-MiniLM-L6-v2`.
Candidate: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
The [candidate model card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
documents a 128-token input window. Tokenizer verification on these documents
found lengths 64, 64, 71, 73, 79 and 78 including special tokens. Whole-document
encoding is appropriate for this small experiment, not a substitute for testing
production chunking on long or scanned documents.

The historical English corpus/Vietnamese-query report is unchanged and is NOT
used for threshold selection. A result on this new Vietnamese corpus is not
numerically comparable to that historical cross-language test. Compare models
within the same saved experiment first; use an isolated production API run for
confirmation before adopting a demo profile.

No Gemini calls, Qdrant mutations or SQLite mutations are needed for the offline
experiment. Future reindexing must use a separate collection and metadata state;
equal vector dimension does not mean two models share an embedding space.

## First recorded run: candidate not promoted

Report: `../results/2026-09-08-vietnamese-embeddings.json`. Real encoders ran in
the project's Docker image; no external document or vector stores were changed.

| Model | Calibration-selected threshold | Test Recall@5 | Test unanswerable abstention recall |
|---|---:|---:|---:|
| all-MiniLM-L6-v2 | 0.6 | 10/12 (83.3%) | 0/6 (0%) |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.5 | 8/12 (66.7%) | 3/6 (50%) |

The multilingual candidate improved rejection of absent facts but reduced
answerable recall. It failed the predeclared promotion gate, so no deployment
profile or default-model change was made. These Vietnamese-document results do
not supersede the historical 0/12 cross-language result on English documents.

The requested Luna implementer and reviewer hit their usage limit. The user
authorized the controller to finish the harness directly. It now supports
calibration-only selection and frozen-revision confirmation replay, with tests
for per-point filtering, token windows, malformed vectors, input/output safety,
configuration locks, leakage guards and promotion-gate behavior. Independent
review remains outstanding; this is not a completed production improvement.
Future experiments must choose candidates/settings on calibration and use a
new confirmation test; do not retune using the test results above.

## E5 follow-up and reproducible confirmation

The next candidate, `intfloat/multilingual-e5-small`, used its required query /
passage prefixes and a wider threshold grid fixed before calibration. Selection
is saved in `../results/2026-09-08-e5-calibration.json`. It also failed the
no-abstention-regression gate on the separately authored confirmation set.
See [confirmation results, error analysis and replay command](confirmation/README.md).
Runtime defaults, collections and application metadata are unchanged.
