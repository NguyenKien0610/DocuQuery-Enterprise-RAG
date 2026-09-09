"""Offline export and ingestion of human answer-support reviews.

The module deliberately has no provider or network calls.  Citation syntax is
checked by :func:`src.citations.validate_citations`; factual support remains a
human categorical judgment.
"""

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.citations import validate_citations

SCHEMA_VERSION = 1
GRADE_VALUES = ("supported", "unsupported", "contradictory", "abstained")
_FINGERPRINT = re.compile(r"[0-9a-f]{64}")


def report_fingerprint(report: Mapping[str, Any]) -> str:
    """Return the stable SHA-256 fingerprint of a JSON evaluator report."""
    if not isinstance(report, Mapping):
        raise ValueError("Evaluator report must be a JSON object")
    try:
        encoded = json.dumps(
            report, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Evaluator report must contain JSON-compatible values") from exc
    return hashlib.sha256(encoded).hexdigest()


def _report_rows(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = report.get("results")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Evaluator report must contain a non-empty results list")
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("Every evaluator result must be a JSON object")
    return rows


def _row_id(row: Mapping[str, Any], index: int) -> str:
    value = row.get("row_id", row.get("id"))
    if value is None:
        return f"row-{index}"
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Result row {index} has an invalid row id")
    return value.strip()


def _answer_and_response(row: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    response = row.get("response", {})
    if response is None:
        response = {}
    if not isinstance(response, Mapping):
        raise ValueError("Evaluator response must be a JSON object")
    answer = response.get("answer", row.get("answer", ""))
    if answer is None:
        answer = ""
    if not isinstance(answer, str):
        raise ValueError("Evaluator answer must be a string")
    return answer, response


def _evidence(row: Mapping[str, Any], response: Mapping[str, Any]) -> list[Any]:
    # Only the context returned by the model/API is evidence for this review.
    # Dataset-side evidence/golden spans are not authoritative model output.
    value = response.get("context", row.get("context", []))
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Evaluator evidence/context must be a list")
    return value


def export_review(report: Mapping[str, Any]) -> dict[str, Any]:
    """Create review items with explicit blank human grades from evaluator rows."""
    rows = _report_rows(report)
    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        row_id = _row_id(row, index)
        if row_id in seen:
            raise ValueError(f"Duplicate result row id: {row_id}")
        seen.add(row_id)
        question = row.get("question", row.get("query"))
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Result row {index} needs a non-empty question/query")
        answer, response = _answer_and_response(row)
        evidence = _evidence(row, response)
        status = response.get("status", row.get("status"))
        if status is not None and not isinstance(status, str):
            raise ValueError(f"Result row {index} has an invalid status")
        not_applicable = status in {
            "degraded",
            "insufficient_context",
            "retrieved",
            "request_failed",
        } or row.get("failed") is True or not answer.strip()
        items.append(
            {
                "row_id": row_id,
                "question": question,
                "answer": answer,
                "evidence": evidence,
                "status": status,
                "not_applicable": not_applicable,
                "citation_valid": (
                    validate_citations(answer, len(evidence))
                    if not not_applicable
                    else None
                ),
                "grade": None,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "report_fingerprint": report_fingerprint(report),
        "grading_method": "human categorical labels",
        "grade_values": list(GRADE_VALUES),
        "items": items,
    }


def _review_items(review: Mapping[str, Any]) -> tuple[str, list[Mapping[str, Any]]]:
    if not isinstance(review, Mapping):
        raise ValueError("Review must be a JSON object")
    fingerprint = review.get("report_fingerprint")
    if not isinstance(fingerprint, str) or _FINGERPRINT.fullmatch(fingerprint) is None:
        raise ValueError("Review has an invalid report fingerprint")
    items = review.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Review must contain a non-empty items list")
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("Every review item must be a JSON object")
        row_id = item.get("row_id")
        if not isinstance(row_id, str) or not row_id.strip():
            raise ValueError("Review item has an invalid row id")
        row_id = row_id.strip()
        if row_id in seen:
            raise ValueError(f"duplicate review row id: {row_id}")
        seen.add(row_id)
        not_applicable = item.get("not_applicable", False)
        if not isinstance(not_applicable, bool):
            raise ValueError(f"Review item {row_id} has an invalid not_applicable flag")
        citation_valid = item.get("citation_valid")
        if citation_valid is not None and not isinstance(citation_valid, bool):
            raise ValueError(f"Review item {row_id} has an invalid citation_valid flag")
    return fingerprint, items


def _grade_entries(grades: Any) -> tuple[str | None, list[Mapping[str, Any]]]:
    if isinstance(grades, Mapping):
        fingerprint = grades.get("report_fingerprint")
        entries = grades.get("grades", grades.get("items"))
    else:
        fingerprint, entries = None, grades
    if not isinstance(entries, list):
        raise ValueError("Grades must contain a list of grade entries")
    if any(not isinstance(entry, Mapping) for entry in entries):
        raise ValueError("Every grade entry must be a JSON object")
    return fingerprint, entries


def ingest_grades(
    report: Mapping[str, Any], grades: Mapping[str, Any] | Sequence[Any]
) -> dict[str, Any]:
    """Re-export *report* and bind complete human grades to its row IDs."""
    review = export_review(report)
    expected_fingerprint, items = _review_items(review)

    supplied_fingerprint, entries = _grade_entries(grades)
    if supplied_fingerprint != expected_fingerprint:
        raise ValueError("Grade report fingerprint does not match review fingerprint")

    expected_ids = {str(item["row_id"]) for item in items}
    reviewable_ids = {
        str(item["row_id"])
        for item in items
        if not item.get("not_applicable", False)
    }
    supplied_ids: set[str] = set()
    assigned: dict[str, str] = {}
    for entry in entries:
        row_id = entry.get("row_id")
        if not isinstance(row_id, str) or not row_id.strip():
            raise ValueError("Grade entry has an invalid row id")
        row_id = row_id.strip()
        if row_id in supplied_ids:
            raise ValueError(f"duplicate grade row id: {row_id}")
        supplied_ids.add(row_id)
        if row_id not in expected_ids:
            raise ValueError(f"invalid grade row id: {row_id}")
        grade = entry.get("grade")
        if row_id not in reviewable_ids:
            if grade is not None:
                raise ValueError(f"invalid grade for not-applicable row {row_id}")
            continue
        if not isinstance(grade, str) or grade not in GRADE_VALUES:
            raise ValueError(f"invalid grade for row {row_id}")
        assigned[row_id] = grade

    missing = reviewable_ids - supplied_ids
    if missing:
        raise ValueError(f"missing grade entries: {', '.join(sorted(missing))}")

    graded_items = []
    for item in items:
        graded_item = dict(item)
        if str(item["row_id"]) in reviewable_ids:
            graded_item["grade"] = assigned[str(item["row_id"])]
        graded_items.append(graded_item)
    return {**review, "items": graded_items}


def summarize_grades(
    review_or_graded: Mapping[str, Any],
    grades: Mapping[str, Any] | Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Summarize human labels and deterministic citation syntax counts only."""
    graded = ingest_grades(review_or_graded, grades) if grades is not None else review_or_graded
    fingerprint, items = _review_items(graded)
    counts = {grade: 0 for grade in GRADE_VALUES}
    citation_counts = {"valid": 0, "invalid": 0}
    not_applicable = 0
    reviewed = 0
    for item in items:
        if item.get("not_applicable", False):
            if item.get("grade") is not None:
                raise ValueError("Not-applicable item cannot have a human grade")
            not_applicable += 1
            continue
        grade = item.get("grade")
        if not isinstance(grade, str) or grade not in GRADE_VALUES:
            raise ValueError("Missing or invalid human grade")
        reviewed += 1
        counts[grade] += 1
        answer = item.get("answer")
        evidence = item.get("evidence")
        citation_valid = (
            validate_citations(answer, len(evidence))
            if isinstance(answer, str) and isinstance(evidence, list)
            else item.get("citation_valid")
        )
        if isinstance(citation_valid, bool):
            citation_counts["valid" if citation_valid else "invalid"] += 1
    return {
        "report_fingerprint": fingerprint,
        "grading_method": "human categorical labels",
        "reviewed_answers": reviewed,
        "grade_counts": counts,
        "citation_syntax_counts": citation_counts,
        "not_applicable_count": not_applicable,
    }


def _read_json(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON: {path}") from exc


def _write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
    except FileExistsError as exc:
        raise ValueError(f"Output already exists: {destination}") from exc


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export", help="Export blank human review items")
    export.add_argument("report")
    export.add_argument("output")

    ingest = commands.add_parser("ingest", help="Ingest grades against an original report")
    ingest.add_argument("report")
    ingest.add_argument("grades")
    ingest.add_argument("output")

    summary = commands.add_parser("summary", help="Print a human-grade summary")
    summary.add_argument("graded_review")

    args = parser.parse_args(argv)
    if args.command == "export":
        _write_json(args.output, export_review(_read_json(args.report)))
    elif args.command == "ingest":
        _write_json(
            args.output,
            ingest_grades(_read_json(args.report), _read_json(args.grades)),
        )
    else:
        print(json.dumps(summarize_grades(_read_json(args.graded_review)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
