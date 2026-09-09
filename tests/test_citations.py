import json

import pytest

from scripts import grade_answers
from src.citations import validate_citations


@pytest.mark.parametrize(
    "answer, source_count",
    [
        ("The policy allows 20 days [Source 1].", 1),
        ("See [Source 1] and [Source 2].", 2),
        ("See [Source 2].", 2),
    ],
)
def test_validate_citations_accepts_canonical_in_range_references(answer, source_count):
    assert validate_citations(answer, source_count) is True


@pytest.mark.parametrize(
    "answer, source_count",
    [
        ("No citation", 1),
        ("[Source 0]", 1),
        ("[Source 2]", 1),
        ("[source 1]", 1),
        ("[Source one]", 1),
        ("[Source 1x]", 1),
        ("[Source 1", 1),
        ("[Source1]", 1),
    ],
)
def test_validate_citations_rejects_missing_out_of_range_or_malformed_references(
    answer, source_count
):
    assert validate_citations(answer, source_count) is False


def test_validate_citations_rejects_invalid_source_count():
    assert validate_citations("Answer [Source 1]", 0) is False
    assert validate_citations("Answer [Source 1]", -1) is False
    assert validate_citations("Answer [Source 1]", True) is False
    assert validate_citations("Answer [Source 1]", 1.0) is False
    assert validate_citations(None, 1) is False


def test_validate_citations_handles_very_large_indexes_without_integer_overflow():
    huge_index = "1" + "0" * 5000
    assert validate_citations(f"Answer [Source {huge_index}]", 1) is False


def _report():
    return {
        "schema_version": 2,
        "results": [
            {
                "id": "q-1",
                "query": "What is the limit?",
                "response": {
                    "status": "generated",
                    "answer": "The limit is 20 [Source 1].",
                    "context": [{"source_file": "policy.txt", "text": "20"}],
                },
            },
            {
                "query": "Can this be answered?",
                "response": {
                    "status": "insufficient_context",
                    "answer": "",
                    "context": [],
                },
            },
        ],
    }


def test_export_review_contains_question_answer_evidence_and_blank_human_grade():
    review = grade_answers.export_review(_report())

    assert review["report_fingerprint"]
    assert [item["row_id"] for item in review["items"]] == ["q-1", "row-2"]
    assert review["items"][0]["question"] == "What is the limit?"
    assert review["items"][0]["answer"] == "The limit is 20 [Source 1]."
    assert review["items"][0]["evidence"][0]["text"] == "20"
    assert review["items"][0]["grade"] is None
    assert review["items"][1]["grade"] is None
    assert review["items"][0]["citation_valid"] is True
    assert review["items"][1]["citation_valid"] is None
    assert review["items"][1]["not_applicable"] is True


def test_export_uses_model_response_context_not_top_level_evidence():
    report = {
        "results": [
            {
                "query": "Which source is authoritative?",
                "evidence": [{"text": "golden answer span"}],
                "response": {
                    "status": "generated",
                    "answer": "The model answer [Source 1].",
                    "context": [{"text": "returned model context"}],
                },
            }
        ]
    }
    review = grade_answers.export_review(report)
    assert review["items"][0]["evidence"] == [{"text": "returned model context"}]


def test_export_marks_failed_answer_rows_not_applicable():
    report = {
        "results": [
            {
                "query": "Question",
                "failed": True,
                "response": {
                    "status": "generated",
                    "answer": "transport error text",
                    "context": [],
                },
            }
        ]
    }
    assert grade_answers.export_review(report)["items"][0]["not_applicable"] is True


def test_ingest_grades_requires_matching_fingerprint_and_complete_unique_rows():
    report = _report()
    review = grade_answers.export_review(report)
    grades = {
        "report_fingerprint": review["report_fingerprint"],
        "grades": [
            {"row_id": "q-1", "grade": "supported"},
            {"row_id": "row-2"},
        ],
    }

    graded = grade_answers.ingest_grades(report, grades)
    assert graded["report_fingerprint"] == review["report_fingerprint"]
    assert graded["items"][0]["grade"] == "supported"
    assert grade_answers.summarize_grades(graded) == {
        "report_fingerprint": review["report_fingerprint"],
        "grading_method": "human categorical labels",
        "reviewed_answers": 1,
        "grade_counts": {
            "supported": 1,
            "unsupported": 0,
            "contradictory": 0,
            "abstained": 0,
        },
        "citation_syntax_counts": {"valid": 1, "invalid": 0},
        "not_applicable_count": 1,
    }

    with pytest.raises(ValueError, match="fingerprint"):
        grade_answers.ingest_grades(
            report,
            {"report_fingerprint": "wrong", "grades": grades["grades"]},
        )
    with pytest.raises(ValueError, match="missing"):
        grade_answers.ingest_grades(
            report,
            {
                "report_fingerprint": review["report_fingerprint"],
                "grades": [],
            },
        )
    with pytest.raises(ValueError, match="duplicate"):
        grade_answers.ingest_grades(
            report,
            {
                "report_fingerprint": review["report_fingerprint"],
                "grades": [
                    {"row_id": "q-1", "grade": "supported"},
                    {"row_id": "q-1", "grade": "abstained"},
                ],
            },
        )
    with pytest.raises(ValueError, match="invalid grade"):
        grade_answers.ingest_grades(
            report,
            {
                "report_fingerprint": review["report_fingerprint"],
                "grades": [
                    {"row_id": "q-1", "grade": "supported"},
                    {"row_id": "row-2", "grade": "maybe"},
                ],
            },
        )


def test_grade_answers_round_trip_is_offline_json_and_does_not_score_factuality():
    report = _report()
    review = grade_answers.export_review(report)
    graded = grade_answers.ingest_grades(
        report,
        {
            "report_fingerprint": review["report_fingerprint"],
            "grades": [
                {"row_id": "q-1", "grade": "contradictory"},
            ],
        },
    )
    encoded = json.dumps(graded, ensure_ascii=False)
    assert "score" not in encoded.lower()
    assert "automatic" not in encoded.lower()


def test_summary_recomputes_citation_syntax_from_review_item_content():
    report = _report()
    review = grade_answers.export_review(report)
    graded = grade_answers.ingest_grades(
        report,
        {
            "report_fingerprint": review["report_fingerprint"],
            "grades": [{"row_id": "q-1", "grade": "supported"}],
        },
    )
    graded["items"][0]["citation_valid"] = False
    assert grade_answers.summarize_grades(graded)["citation_syntax_counts"] == {
        "valid": 1,
        "invalid": 0,
    }


def test_export_rejects_aggregate_only_reports():
    with pytest.raises(ValueError, match="results"):
        grade_answers.export_review({"document_recall_at_k": 1.0})


def test_cli_does_not_overwrite_existing_output(tmp_path):
    report_path = tmp_path / "report.json"
    output_path = tmp_path / "review.json"
    report_path.write_text(json.dumps(_report()), encoding="utf-8")
    output_path.write_text("keep me", encoding="utf-8")

    with pytest.raises(ValueError, match="exists"):
        grade_answers.main(["export", str(report_path), str(output_path)])
    assert output_path.read_text(encoding="utf-8") == "keep me"


def test_cli_ingest_reexports_from_original_report(tmp_path):
    report = _report()
    report_path = tmp_path / "report.json"
    edited_review_path = tmp_path / "edited-review.json"
    graded_path = tmp_path / "graded.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    review = grade_answers.export_review(report)
    review["items"][0]["evidence"] = [{"text": "tampered"}]
    review["items"][0]["grade"] = "supported"
    edited_review_path.write_text(json.dumps(review), encoding="utf-8")

    grade_answers.main(
        ["ingest", str(report_path), str(edited_review_path), str(graded_path)]
    )
    graded = json.loads(graded_path.read_text(encoding="utf-8"))
    assert graded["items"][0]["evidence"] == [{"source_file": "policy.txt", "text": "20"}]
