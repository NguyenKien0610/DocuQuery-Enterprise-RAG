import hashlib
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_document_and_evidence_scores_are_reported_separately():
    from scripts.evaluate_chunk_retrieval import score_case

    case = {
        "id": "q1",
        "query": "Bao lau?",
        "sources": ["policy.txt"],
        "evidence_spans": [
            {"source_file": "policy.txt", "text": "Luu hoa don trong 18 thang."}
        ],
    }
    contexts = [
        {"source_file": "other.txt", "chunk_index": 0, "text": "Other policy."},
        {
            "source_file": "policy.txt",
            "chunk_index": 3,
            "text": "Luu hoa don trong 18 thang.",
        },
    ]

    result = score_case(case, contexts, k=5)

    assert result["document_recall_at_k"] == 1
    assert result["document_mrr"] == 0.5
    assert result["evidence_recall_at_k"] == 1
    assert result["evidence_mrr"] == 0.5


def test_span_matching_requires_the_labeled_source_and_text():
    from scripts.evaluate_chunk_retrieval import evidence_scores

    contexts = [
        {
            "source_file": "wrong.txt",
            "chunk_index": 0,
            "text": "Luu hoa don trong 18 thang.",
        }
    ]
    assert evidence_scores(contexts, [
        {"source_file": "policy.txt", "text": "Luu hoa don trong 18 thang."}
    ], 5) == (0.0, 0.0)


def test_failed_api_status_does_not_earn_retrieval_credit(monkeypatch):
    from scripts import evaluate_chunk_retrieval as evaluator

    case = {
        "id": "q1",
        "query": "Q",
        "sources": ["policy.txt"],
        "evidence_spans": [{"source_file": "policy.txt", "text": "fact"}],
    }
    monkeypatch.setattr(
        evaluator,
        "_request_json",
        lambda *args, **kwargs: {
            "status": "degraded",
            "context": [{"source_file": "policy.txt", "text": "fact"}],
        },
    )

    row = evaluator._query_case(
        case, base_url="http://test", api_key="key", workspace="dedicated", k=5, timeout=1
    )

    assert row["failed"] is True
    assert row["document_recall_at_k"] == 0
    assert row["evidence_recall_at_k"] == 0


def test_dataset_validation_rejects_missing_or_duplicate_evidence_labels(tmp_path):
    from scripts.evaluate_chunk_retrieval import load_dataset

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "policy.txt").write_text("Một quy định.", encoding="utf-8")
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps(
            [
                {
                    "id": "duplicate",
                    "query": "Q",
                    "sources": ["policy.txt"],
                    "evidence_spans": [
                        {"source_file": "policy.txt", "text": "Một quy định."},
                        {"source_file": "policy.txt", "text": "Một quy định."},
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate evidence"):
        load_dataset(corpus, questions)


def test_dataset_loader_rejects_evidence_text_not_present_in_source(tmp_path):
    from scripts.evaluate_chunk_retrieval import load_dataset

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "policy.txt").write_text("The retention period is 18 months.", encoding="utf-8")
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps([{
            "id": "q1",
            "query": "How long?",
            "sources": ["policy.txt"],
            "evidence_spans": [{"source_file": "policy.txt", "text": "The retention period is 19 months."}],
        }]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not present"):
        load_dataset(corpus, questions)


def test_evaluate_uses_upload_status_and_retrieval_only_without_reset_or_generation(
    tmp_path, monkeypatch
):
    from scripts import evaluate_chunk_retrieval as evaluator

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    document = corpus / "policy.txt"
    document.write_text("Luu hoa don trong 18 thang.", encoding="utf-8")
    questions = tmp_path / "questions.json"
    raw_questions = json.dumps(
        [
            {
                "id": "q1",
                "language": "vi",
                "query": "Bao lau?",
                "sources": ["policy.txt"],
                "evidence_spans": [
                    {"source_file": "policy.txt", "text": "Luu hoa don trong 18 thang."}
                ],
            }
        ]
    ).encode("utf-8")
    questions.write_bytes(raw_questions)
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        if method == "POST" and url.endswith("/documents/upload"):
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"task_id": "task-1", "document_id": "doc-1"},
            )
        if method == "GET" and url.endswith("/documents/status/task-1"):
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"status": "SUCCESS", "result": {"chunks_indexed": 1}},
            )
        if method == "POST" and url.endswith("/query"):
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {
                    "status": "retrieved",
                    "answer": "",
                    "context": [
                        {
                            "source_file": "policy.txt",
                            "chunk_index": 0,
                            "text": "Luu hoa don trong 18 thang.",
                        }
                    ],
                },
            )
        raise AssertionError(f"Unexpected {method} {url}")

    monkeypatch.setattr(evaluator.requests, "request", request)
    report = evaluator.evaluate(
        Namespace(
            corpus=corpus,
            questions=questions,
            base_url="http://test",
            api_key="secret",
            workspace="dedicated",
            k=5,
            timeout=10,
            max_wait=30,
            poll_interval=0,
            ingest=True,
            run_label="calibration",
        )
    )

    assert report["retrieval_only"] is True
    assert report["human_grades"] is None
    assert report["document_metrics"]["recall_at_k"] == 1
    assert report["evidence_metrics"]["recall_at_k"] == 1
    assert report["questions_sha256"] == hashlib.sha256(raw_questions).hexdigest()
    assert all(method != "DELETE" for method, _, _ in calls)
    query_calls = [item for item in calls if item[1].endswith("/query")]
    assert query_calls[0][2]["json"] == {
        "query": "Bao lau?",
        "retrieval_only": True,
        "use_cache": False,
    }


def test_dataset_loader_keeps_calibration_and_confirmation_document_sets_disjoint(
    tmp_path,
):
    from scripts.evaluate_chunk_retrieval import assert_disjoint_corpora

    first = tmp_path / "cal"
    second = tmp_path / "confirm"
    first.mkdir()
    second.mkdir()
    (first / "one.txt").write_text("one", encoding="utf-8")
    (second / "two.txt").write_text("two", encoding="utf-8")
    assert_disjoint_corpora(first, second) == ("one.txt", "two.txt")

    (second / "one.txt").write_text("collision", encoding="utf-8")
    with pytest.raises(ValueError, match="document-disjoint"):
        assert_disjoint_corpora(first, second)


def test_corpus_disjointness_rejects_renamed_identical_content(tmp_path):
    from scripts.evaluate_chunk_retrieval import assert_disjoint_corpora

    first = tmp_path / "cal"
    second = tmp_path / "confirm"
    first.mkdir()
    second.mkdir()
    (first / "cal.txt").write_text("same policy", encoding="utf-8")
    (second / "renamed.txt").write_text("same policy", encoding="utf-8")

    with pytest.raises(ValueError, match="document-disjoint"):
        assert_disjoint_corpora(first, second)


def test_question_disjointness_rejects_reused_ids_or_queries():
    from scripts.evaluate_chunk_retrieval import assert_disjoint_questions

    first = [{"id": "q1", "query": "Same question"}]
    with pytest.raises(ValueError, match="question-disjoint"):
        assert_disjoint_questions(first, [{"id": "q1", "query": "Different"}])
    with pytest.raises(ValueError, match="question-disjoint"):
        assert_disjoint_questions(first, [{"id": "q2", "query": "Same question"}])


def test_abstention_requires_insufficient_context_status():
    from scripts.evaluate_chunk_retrieval import summarize_results

    rows = [
        {"answerable": False, "abstained": True, "failed": False, "response": {"status": "retrieved"}},
        {"answerable": False, "abstained": True, "failed": False, "response": {"status": "insufficient_context"}},
    ]
    assert summarize_results(rows)["abstention"] == {
        "precision": 1.0,
        "recall": 0.5,
        "answerable_abstention_rate": None,
    }


def test_evaluation_rejects_corpus_mutation_during_queries(tmp_path, monkeypatch):
    from scripts import evaluate_chunk_retrieval as evaluator

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = corpus / "policy.txt"
    source.write_text("A retention period is 18 months.", encoding="utf-8")
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([{
        "id": "q1",
        "query": "How long?",
        "sources": ["policy.txt"],
        "evidence_spans": [{"source_file": "policy.txt", "text": "A retention period is 18 months."}],
    }]), encoding="utf-8")

    def mutate(*args, **kwargs):
        source.write_text("changed", encoding="utf-8")
        return {
            "id": "q1", "query": "How long?", "sources": ["policy.txt"],
            "evidence_spans": [{"source_file": "policy.txt", "text": "A retention period is 18 months."}],
            "answerable": True, "document_recall_at_k": 1, "document_mrr": 1,
            "evidence_recall_at_k": 1, "evidence_mrr": 1, "retrieved_sources": ["policy.txt"],
            "retrieved_chunk_count": 1, "abstained": False, "response": {"status": "retrieved"},
            "failed": False, "seconds": 0,
        }

    monkeypatch.setattr(evaluator, "_query_case", mutate)
    with pytest.raises(ValueError, match="changed during evaluation"):
        evaluator.evaluate(Namespace(
            corpus=corpus, questions=questions, base_url="http://test", api_key="key",
            workspace="dedicated", k=5, timeout=1, concurrency=1, ingest=False,
            poll_interval=0, max_wait=5,
        ))


def test_fixture_corpora_are_larger_than_top_k_and_have_required_query_variants():
    from scripts.evaluate_chunk_retrieval import (
        assert_disjoint_corpora,
        assert_disjoint_questions,
        load_dataset,
    )

    root = Path("evaluation/vietnamese/chunks")
    calibration = load_dataset(root / "calibration/corpus", root / "calibration/questions.json")
    confirmation = load_dataset(root / "confirmation/corpus", root / "confirmation/questions.json")
    assert len(calibration.corpus_files) > 5
    assert len(confirmation.corpus_files) > 5
    assert_disjoint_corpora(
        root / "calibration/corpus", root / "confirmation/corpus"
    )
    assert_disjoint_questions(calibration, confirmation)
    variants = {
        case["query_type"]
        for dataset in (calibration, confirmation)
        for case in dataset.questions
    }
    assert {"paraphrase", "no_diacritic", "cross_language", "hard_negative"} <= variants
