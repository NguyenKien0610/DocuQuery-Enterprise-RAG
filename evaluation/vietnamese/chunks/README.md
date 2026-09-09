# Production chunk retrieval evaluation

This synthetic evaluation exercises the public DocuQuery upload, task-status,
and `retrieval_only` query APIs. It is not a human-graded answer-quality
benchmark and must not be presented as a real company-policy dataset.

The calibration and confirmation splits each contain eight documents, so the
document set is larger than the default `K=5`. Their filenames and question
strings are disjoint. The confirmation corpus must be run only after the
retrieval configuration is frozen on calibration; do not tune a threshold or
model using confirmation results.

Each question has `sources` for document-level labels and `evidence_spans` for
chunk-level labels:

```json
{
  "id": "cal-01",
  "language": "vi",
  "query_type": "paraphrase",
  "query": "Mỗi năm nhân viên được hưởng bao nhiêu ngày phép có lương?",
  "sources": ["cal-leave.txt"],
  "evidence_spans": [
    {"source_file": "cal-leave.txt", "text": "Nhân viên được nghỉ phép hưởng lương 14 ngày trong mỗi năm dương lịch."}
  ]
}
```

The labels include paraphrases, Vietnamese without diacritics, Vietnamese
queries against English documents (and vice versa), and hard same-topic
unsupported questions. A span counts only when its labeled source and text
occur in a returned production chunk; filename recall is never used as a
proxy for evidence-span recall.

## Running the evaluator

Use an owner-authorized, dedicated workspace. The evaluator never calls reset
or delete and does not generate answers, read/write answer cache, or call
Gemini. `--ingest` is explicit; omit it when documents are already present in
the dedicated workspace.

```powershell
$env:DOCUQUERY_API_KEY = "<owner-key>"
$env:DOCUQUERY_WORKSPACE_ID = "<dedicated-calibration-workspace>"
python -m scripts.evaluate_chunk_retrieval `
  --corpus evaluation/vietnamese/chunks/calibration/corpus `
  --questions evaluation/vietnamese/chunks/calibration/questions.json `
  --ingest --output evaluation/results/chunk-calibration.json
```

Run confirmation in a separate owner workspace (or an explicitly preserved
workspace containing only that corpus):

```powershell
python -m scripts.evaluate_chunk_retrieval `
  --corpus evaluation/vietnamese/chunks/confirmation/corpus `
  --questions evaluation/vietnamese/chunks/confirmation/questions.json `
  --ingest --output evaluation/results/chunk-confirmation.json
```

Reports hash the exact question bytes, each corpus file, a canonical corpus
manifest, the combined input fingerprint, and non-secret evaluator
configuration. The public query response does not expose the server's actual
embedding model, threshold, or effective top-K, so those fields are explicitly
reported as unverified rather than inferred from requested settings.

Reports contain separate `document_metrics` and `evidence_metrics`. Human
faithfulness/citation grades remain `null` because retrieval-only requests do
not generate answers and this fixture has no human review. A failed/degraded
API response never receives retrieval credit, even if a response accidentally
contains context.
