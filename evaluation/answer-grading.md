# Offline answer-support review

`python -m scripts.grade_answers` prepares and consumes human reviews of
evaluator response rows. It never calls the API, an embedding model, or an LLM.
The evaluator report must contain its per-response `results`; aggregate-only
reports cannot be used because they do not contain the question, answer, and
returned evidence needed for review.

Export a review:

```text
python -m scripts.grade_answers export benchmark_results.json answer-review.json
```

Each item contains `row_id`, `question`, `answer`, `evidence`, `status`, the
deterministic `citation_valid` result, and an explicit blank `grade`. The report
fingerprint is a SHA-256 digest of the canonical JSON report. A row's existing
`id` is retained; otherwise `row-1`, `row-2`, and so on are assigned.

Fill each reviewable item's `grade` in a copy of the review file (or a separate
file) with one of these human labels. Items marked `not_applicable` stay blank;
they represent provider failures, retrieval-only responses, abstentions with no
answer, or other rows that do not contain a generated answer for factual review.

| Label | Meaning |
| --- | --- |
| `supported` | The answer's material factual claims are supported by the returned evidence. |
| `unsupported` | A material claim is not established by the returned evidence. |
| `contradictory` | The evidence materially conflicts with the answer. |
| `abstained` | The answer appropriately declines because the evidence is insufficient. |

For a separate grade file, use the same `report_fingerprint` and a complete,
unique `grades` list, for example:

```json
{
  "report_fingerprint": "<64 lowercase hex characters>",
  "grades": [
    {"row_id": "q-1", "grade": "supported"}
  ]
}
```

Bind and validate it offline. Ingestion requires the original evaluator report,
rebuilds the review items from that report, and therefore ignores edited
question/answer/evidence fields in a review copy:

```text
python -m scripts.grade_answers ingest benchmark_results.json human-grades.json graded-review.json
python -m scripts.grade_answers summary graded-review.json
```

Ingestion rejects a mismatched fingerprint, missing or duplicate row, unknown
row ID, blank grade, and any label outside the four values above. The summary
reports categorical human-label counts separately from deterministic citation
syntax counts. It does not infer factual support from citation indexes and does
not manufacture numeric quality scores for unreviewed answers.

Citation syntax is intentionally narrow: generated text must contain at least
one exact `[Source N]` reference, where `N` is a positive index no greater than
the number of returned evidence items. This is syntax/index validation only;
`[Source 1]` being valid does not show that Source 1 supports the claim.
