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


def test_abstention_metrics_do_not_reward_service_errors():
    from scripts.evaluate_docuquery import abstention_scores

    rows = [
        {"sources": [], "response": {"status": "insufficient_context"}},
        {"sources": [], "response": {"status": "degraded"}},
        {"sources": ["a"], "response": {"status": "insufficient_context"}},
        {"sources": ["a"], "response": {"status": "retrieved"}},
    ]
    assert abstention_scores(rows) == {
        "unanswerable_count": 2,
        "abstention_precision": 0.5,
        "abstention_recall": 0.5,
        "answerable_abstention_rate": 0.5,
    }


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


def test_heldout_questions_are_disjoint_and_include_unanswerable_cases():
    baseline = json.loads(Path("evaluation/questions.json").read_text(encoding="utf-8"))
    heldout = json.loads(
        Path("evaluation/questions-heldout.json").read_text(encoding="utf-8")
    )
    assert not {r["query"] for r in baseline} & {r["query"] for r in heldout}
    assert len({r["id"] for r in heldout}) == len(heldout)
    assert {r["language"] for r in heldout} == {"en", "vi"}
    assert any(not r["sources"] for r in heldout)
    for row in heldout:
        assert row["reference"]
        for source in row["sources"]:
            assert (Path("evaluation/corpus") / source).is_file()


def test_evaluator_separates_answerable_metrics_and_service_failures(
    monkeypatch, tmp_path
):
    from argparse import Namespace
    from types import SimpleNamespace

    from scripts import evaluate_docuquery as evaluator

    cases = [
        {"query": "known", "sources": ["a.txt"]},
        {"query": "unknown", "sources": []},
    ]
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(cases), encoding="utf-8")

    def post(url, **kwargs):
        assert kwargs["json"]["retrieval_only"] is True
        known = kwargs["json"]["query"] == "known"
        payload = {
            "status": "retrieved" if known else "insufficient_context",
            "context": [{"source_file": "a.txt"}] if known else [],
        }
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)

    monkeypatch.setattr(evaluator.requests, "post", post)
    report = evaluator.evaluate(
        Namespace(
            questions=path,
            concurrency=1,
            k=5,
            api_key="test",
            workspace="eval",
            base_url="http://test",
            use_cache=False,
            retrieval_only=True,
        )
    )
    assert report["document_recall_at_k"] == 1
    assert report["document_mrr"] == 1
    assert report["failure_rate"] == 0
    assert report["abstention_recall"] == 1
    assert report["generated_p50_seconds"] is None


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


def test_summary_groups_languages_with_correct_denominators():
    from scripts import evaluate_docuquery as evaluator

    rows = [
        {"language": "vi", "sources": ["a"], "recall": 1, "reciprocal_rank": 0.5, "seconds": 2,
         "failed": False, "response": {"status": "retrieved"}},
        {"language": "vi", "sources": [], "recall": 0, "reciprocal_rank": 0, "seconds": 1,
         "failed": False, "response": {"status": "insufficient_context"}},
        {"language": "vi", "sources": [], "recall": 0, "reciprocal_rank": 0, "seconds": 3,
         "failed": True, "error": "request_failed"},
        {"language": "en", "sources": ["a"], "recall": 0, "reciprocal_rank": 0, "seconds": 1,
         "failed": False, "response": {"status": "insufficient_context"}},
        {"sources": [], "recall": 0, "reciprocal_rank": 0, "seconds": 1,
         "failed": False, "response": {"status": "insufficient_context"}},
    ]
    report = evaluator.summarize_results({"results": rows, "retrieval_only": True})
    assert report["document_recall_at_k"] == 0.5
    assert report["failure_rate"] == 0.2
    assert report["abstention_recall"] == pytest.approx(2 / 3)
    vi = report["language_metrics"]["vi"]
    assert vi["count"] == 3
    assert vi["answerable_count"] == 1
    assert vi["document_recall_at_k"] == 1
    assert vi["document_mrr"] == 0.5
    assert vi["abstention_recall"] == 0.5
    assert vi["failure_rate"] == pytest.approx(1 / 3)
    assert report["language_metrics"]["en"]["abstention_recall"] is None
    assert report["language_metrics"]["unknown"]["document_recall_at_k"] is None
    assert report["generated_p50_seconds"] is None


def test_offline_summary_needs_no_credentials_or_network_and_preserves_report(tmp_path, monkeypatch, capsys):
    from scripts import evaluate_docuquery as evaluator

    path = tmp_path / "historical.json"
    path.write_text(json.dumps({"schema_version": 1, "results": [
        {"sources": ["a"], "recall": 1, "reciprocal_rank": 1, "seconds": 2,
         "failed": False, "response": {"status": "generated"}},
    ]}), encoding="utf-8")
    before = path.read_bytes()
    monkeypatch.delenv("DOCUQUERY_API_KEY", raising=False)
    monkeypatch.delenv("DOCUQUERY_WORKSPACE_ID", raising=False)
    monkeypatch.setattr(evaluator.requests, "post", lambda *args, **kwargs: pytest.fail("Offline summary used HTTP"))
    monkeypatch.setattr("sys.argv", ["evaluate", "--summarize-results", str(path)])
    evaluator.main()
    summary = json.loads(capsys.readouterr().out)
    assert summary["language_metrics"]["unknown"]["document_recall_at_k"] == 1
    assert summary["generated_p50_seconds"] == 2
    assert path.read_bytes() == before


def test_evaluator_fingerprints_original_question_bytes_and_preserves_unicode(tmp_path, monkeypatch):
    import hashlib
    from argparse import Namespace
    from types import SimpleNamespace

    from scripts import evaluate_docuquery as evaluator

    raw = '[{"query":"Nghỉ phép?","sources":[],"language":"vi"}]'.encode("utf-8")
    path = tmp_path / "questions.json"
    path.write_bytes(raw)

    def post(*args, **kwargs):
        # A file modified during requests must not change this run's fingerprint.
        path.write_bytes(raw + b"\n")
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            "status": "insufficient_context", "context": [],
        })

    monkeypatch.setattr(evaluator.requests, "post", post)
    args = Namespace(questions=path, concurrency=1, k=5, api_key="secret-not-in-report", workspace="eval",
                     base_url="http://unused", use_cache=False, retrieval_only=True, run_label="demo tiếng Việt")
    first = evaluator.evaluate(args)
    second = evaluator.evaluate(args)
    assert first["questions_sha256"] == hashlib.sha256(raw).hexdigest()
    assert second["questions_sha256"] == hashlib.sha256(raw + b"\n").hexdigest()
    assert first["run_label"] == "demo tiếng Việt"
    assert first["language_metrics"]["vi"]["abstention_recall"] == 1
    assert "secret-not-in-report" not in json.dumps(first)


def test_empty_summary_has_null_ratios_and_no_languages():
    from scripts import evaluate_docuquery as evaluator

    report = evaluator.summarize_results({"results": []})
    assert report["count"] == 0
    assert report["failure_rate"] is None
    assert report["document_recall_at_k"] is None
    assert report["abstention_recall"] is None
    assert report["language_metrics"] == {}


def test_failed_response_is_never_a_correct_abstention():
    from scripts.evaluate_docuquery import abstention_scores

    metrics = abstention_scores([
        {"sources": [], "failed": True, "response": {"status": "insufficient_context"}},
    ])
    assert metrics["abstention_recall"] == 0
    assert metrics["abstention_precision"] is None


def test_summary_does_not_invent_rows_from_aggregate_only_report():
    from scripts import evaluate_docuquery as evaluator

    with pytest.raises(ValueError, match="per-response"):
        evaluator.summarize_results({"document_recall_at_k": 0.7})


def test_retrieval_only_summary_never_reports_generation_latency():
    from scripts import evaluate_docuquery as evaluator

    summary = evaluator.summarize_results({"retrieval_only": True, "results": [
        {"sources": ["a"], "recall": 0, "reciprocal_rank": 0, "seconds": 2,
         "failed": True, "response": {"status": "generated"}},
    ]})
    assert summary["generated_p50_seconds"] is None
    assert summary["generated_p95_seconds"] is None
    assert summary["failure_rate"] == 1
