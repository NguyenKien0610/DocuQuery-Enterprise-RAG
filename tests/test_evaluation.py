import json
from pathlib import Path

import pytest

from scripts import benchmark_docuquery
from scripts.evaluate_docuquery import percentile, retrieval_scores, summarize_review


def test_document_metrics_deduplicate_and_respect_rank():
    assert retrieval_scores(["wrong", "right", "right"], ["right"], 2) == (1.0, 0.5)
    assert retrieval_scores(["wrong", "right"], ["right"], 1) == (0.0, 0.0)
    assert retrieval_scores([], ["right"], 5) == (0.0, 0.0)


def test_percentiles_interpolate():
    assert percentile([4, 1, 3, 2], 0.5) == 2.5


def test_fixture_references_existing_sources():
    questions = json.loads(
        Path("evaluation/questions.json").read_text(encoding="utf-8")
    )
    assert len(questions) == 30
    assert len({case["query"] for case in questions}) == 30
    for case in questions:
        assert case["reference"]
        for source in case["sources"]:
            assert (Path("evaluation/corpus") / source).is_file()


@pytest.mark.parametrize("status", ["degraded", "insufficient_context", None])
def test_latency_benchmark_rejects_non_generated_responses(monkeypatch, status):
    monkeypatch.setattr(
        benchmark_docuquery, "_request_json", lambda *args, **kwargs: {"status": status}
    )
    with pytest.raises(RuntimeError, match="did not generate"):
        benchmark_docuquery.run_query("http://unused", "q", 1, "key", "workspace")


def test_human_metrics_require_complete_grades():
    with pytest.raises(ValueError, match="Every response"):
        summarize_review([{"faithfulness": None, "answer_relevance": None}])
    assert summarize_review(
        [
            {"faithfulness": 1, "answer_relevance": 1},
            {"faithfulness": 0, "answer_relevance": 1},
        ]
    ) == {"reviewed_responses": 2, "faithfulness": 0.5, "answer_relevance": 1}
