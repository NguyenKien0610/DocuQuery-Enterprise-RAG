# Reproducible evaluation

The six fictional English policy documents and 30 authored questions are a small
smoke dataset, not representative evidence of enterprise or Vietnamese quality.
Expected filenames and answer references are versioned in `questions.json`.

With API and worker running and DOCUQUERY_API_KEY exported:

```bash
python -m scripts.evaluate_docuquery --ingest --workspace eval-demo --output benchmark_results_baseline.json
python -m scripts.evaluate_docuquery --workspace eval-demo --use-cache --output benchmark_results_warmup.json
python -m scripts.evaluate_docuquery --workspace eval-demo --use-cache --concurrency 4 --output benchmark_results_cached.json
python -m scripts.evaluate_docuquery --workspace eval-demo --concurrency 4 --output benchmark_results_concurrent.json
```

The default bypasses cache reads AND writes. The two cache runs warm then measure
the same questions. Inspect the per-response `cached` fields before attributing
speedup. Use a fresh dedicated workspace for another corpus; ingestion does not
reset existing data. Requests invoke Gemini and consume its quota.

The key must have a grant for the chosen workspace. The legacy key only permits
`DOCUQUERY_WORKSPACE_ID`; an arbitrary `eval-demo` header does not grant access.
Set that workspace on the backend or configure an explicit owner grant before
ingestion. The evaluator now requires a workspace argument/environment value.

## Retrieval-only and held-out questions

```bash
python -m scripts.evaluate_docuquery --workspace eval-demo --ingest --retrieval-only --questions evaluation/questions-heldout.json --output benchmark_results_heldout.json
```

`retrieval_only: true` on `/api/v1/query` uses the same embedding, workspace and
published-revision filters, top K and score threshold as normal queries. It skips
answer-cache reads/writes and never calls Gemini. A nonempty result has status
`retrieved`, an empty answer and citations; no hits gives `insufficient_context`.
This measures evidence retrieval, not answer generation or faithfulness.

`questions-heldout.json` contains 24 authored questions: 18 answerable (12
Vietnamese, 6 English) and 6 unanswerable (3 in each language). They are disjoint
from the original 30 questions, but use the **same six fictional English source
documents**. Vietnamese queries therefore test cross-language retrieval. This is
question-held-out, not an independently collected or document-held-out dataset.
Once used for tuning, these questions are no longer an untouched test set: use
fresh questions/documents for subsequent confirmation. Do not tune the threshold
against this report and then report the same score as unseen performance.

Schema version 2 metrics:

- Recall/MRR use answerable questions only; request failures remain zero in that
  denominator. No answerable cases produces null, not a misleading zero.
- `failure_rate` counts service errors/degraded/unrecognized responses; an
  `insufficient_context` response is not an infrastructure failure.
- Abstention precision is correct abstentions / all abstentions. Abstention
  recall is correct abstentions / all unanswerable questions, including errors.
- `answerable_abstention_rate` measures answerable questions rejected. Undefined
  ratios are null. Service errors never count as correct abstentions.
- Generated-only latency includes only actual `generated` responses and is null
  for retrieval-only runs. All-request latency remains separately reported.

An unanswerable question can retrieve topically similar but insufficient chunks.
In retrieval-only mode this is a missed abstention, not proof the LLM would
hallucinate. Keep human faithfulness/relevance grades null until a separate
generation run is actually reviewed. Do not compare schema v2 `failure_rate`
directly with the historical report's non-generated-response rate.

Metrics: document-level Recall@K (deduplicated filenames), MRR, all-request
p50/p95, generated-only p50/p95, status counts, throughput, service-failure rate,
and separate abstention metrics. K must not exceed
the API's RAG_TOP_K. Compare the same corpus/configuration across runs. The
current baseline has no reranker; do not claim a reranker improvement.

Faithfulness and answer relevance deliberately remain null for human grading:
- Faithfulness: 1 only when every factual claim is supported by returned context;
  0 if any material claim is unsupported. Check citations against their text.
- Answer relevance: 1 if the response directly and correctly answers the question
  against its reference; 0 otherwise.

Record grader, model, threshold, machine, date, and disagreements alongside the
results. Do not substitute keyword overlap for faithfulness. Expand with unseen,
unanswerable, adversarial and Vietnamese questions before tuning the threshold
or making CV quality claims. Prompt instructions alone do not prevent injection.

After grading **every** row with integer 0/1 values, aggregate with:

```bash
python -m scripts.evaluate_docuquery --review-results benchmark_results_baseline.json
```

Ungraded rows are rejected rather than silently omitted from the denominator.

## Language metrics and offline summaries

New API evaluation reports add `language_metrics`, `questions_sha256` and
`run_label` without changing the schema-v2 aggregate meanings. Each language
group uses the same formulas as the overall report: only answerable questions
enter Recall/MRR, errors remain in their denominators, and service failures are
never successful abstentions. Missing/empty language labels become `unknown`.
Undefined ratios are null, not zero. No language is inferred from question text.

```bash
python -m scripts.evaluate_docuquery --workspace eval-demo --retrieval-only --questions evaluation/questions-heldout.json --run-label "local evidence demo" --output benchmark_results_demo.json
python -m scripts.evaluate_docuquery --summarize-results benchmark_results_demo.json
python -m scripts.evaluate_docuquery --summarize-results evaluation/results/2026-09-08-retrieval-heldout.json
```

The first command calls the API (no Gemini in retrieval-only mode). The last two
only read recorded per-response rows: no credentials, HTTP, model loading or
report rewriting. They print aggregate and language-level retrieval, abstention,
failure/status and latency metrics. Historical row-based reports without language
are summarized under `unknown`. The compact `2026-09-08-smoke.json` has aggregates
only and cannot be reaggregated by language; it is rejected with an explanation
rather than inventing missing rows. Existing report files remain unchanged.

`questions_sha256` hashes the exact bytes loaded before requests start, including
whitespace; it does not hash a file re-read after evaluation. `run_label` is only
an operator-supplied label, NOT independently verified model/configuration metadata.
Matching question hashes alone do not make two runs comparable: also record the
corpus, embedding model, score threshold, chunking and runtime configuration.
The offline summary reuses recorded row scores, not a new inference run. It does
not fabricate human answer-quality grades or infer per-language throughput from
overlapping concurrent request times.

## Recorded smoke run

`results/2026-09-08-smoke.json` records a real end-to-end run: 8 generated answers,
15 provider failures (429/quota), 7 abstentions, document Recall@5/MRR 0.767.
Its all-request latency includes fast failures and abstentions. This demonstrates
the evaluator detects degradation; it is not a successful production benchmark.

## Recorded retrieval-only held-out run

`results/2026-09-08-retrieval-heldout.json` was produced through the Docker API
and Celery worker, with all six corpus documents ingested into an initially empty
workspace. Configuration inspected on the running API: `all-MiniLM-L6-v2`, top K
5, threshold 0.35, chunk size 1000, overlap 150, concurrency 1, cache disabled.
No Gemini generation was requested. No threshold/model tuning preceded this run.

Observed answerable document Recall@5/MRR: **0.2222** (4/18). Vietnamese Recall:
**0/12**; English Recall: **4/6**. Service failure rate was zero, but 14/18
answerable questions were rejected. Correct unanswerable abstentions: 5/6;
abstention precision was only 5/19. The sixth unanswerable query retrieved context
that did not establish its requested fact. Generated latency and human answer
grades are null, as expected.

This is evidence of a cross-language retrieval weakness under this configuration,
not proof that changing the embedding model alone will solve it. Next experiments
should compare multilingual embeddings and threshold choices on a separate
calibration set, reindex into a new collection, and confirm on newly collected
held-out documents/questions. Do not overwrite this baseline or lower the threshold
on these 24 questions and advertise the resulting score as unseen performance.
