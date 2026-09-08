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

Metrics: document-level Recall@K (deduplicated filenames), MRR, all-request
p50/p95, generated-only p50/p95, status counts, throughput, and failed/non-generated response rate. K must not exceed
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

## Recorded smoke run

`results/2026-09-08-smoke.json` records a real end-to-end run: 8 generated answers,
15 provider failures (429/quota), 7 abstentions, document Recall@5/MRR 0.767.
Its all-request latency includes fast failures and abstentions. This demonstrates
the evaluator detects degradation; it is not a successful production benchmark.
