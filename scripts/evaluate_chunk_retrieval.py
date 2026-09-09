"""Evaluate production chunk retrieval through the public DocuQuery API.

This evaluator only sends ``retrieval_only`` queries.  Ingestion is opt-in and
uses the public upload/status endpoints, so the caller must provide a dedicated
workspace and this script never resets or deletes workspace data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from scripts.benchmark_docuquery import upload_and_wait

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
VALID_QUERY_TYPES = {"paraphrase", "no_diacritic", "cross_language", "hard_negative"}


@dataclass(frozen=True)
class Dataset:
    corpus_dir: Path
    question_path: Path
    corpus_files: tuple[str, ...]
    questions: list[dict[str, Any]]
    question_bytes: bytes
    corpus_sha256: str
    corpus_file_sha256: dict[str, str]


def normalize_text(value: str) -> str:
    """Normalize labels and returned chunk text for stable substring matching."""

    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _span_source(span: dict[str, Any]) -> str:
    source = span.get("source_file", span.get("source"))
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Evidence span requires a non-empty source_file")
    return Path(source.replace("\\", "/")).name


def _span_text(span: dict[str, Any]) -> str:
    text = span.get("text", span.get("snippet"))
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Evidence span requires non-empty text")
    return text.strip()


def load_dataset(corpus_dir: str | Path, question_path: str | Path) -> Dataset:
    """Load and validate authored labels without invoking a model or API."""

    corpus_path = Path(corpus_dir)
    questions_path = Path(question_path)
    if not corpus_path.is_dir():
        raise ValueError(f"Corpus directory does not exist: {corpus_path}")
    try:
        question_bytes = questions_path.read_bytes()
        raw_questions = json.loads(question_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read questions: {questions_path}") from exc
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError("Questions must be a non-empty JSON array")

    corpus_paths = sorted(
        path for path in corpus_path.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not corpus_paths:
        raise ValueError("Corpus must contain at least one PDF, DOCX, or TXT file")
    corpus_files = tuple(path.name for path in corpus_paths)
    file_hashes = {path.name: _sha256(path.read_bytes()) for path in corpus_paths}
    manifest = [{"name": name, "sha256": file_hashes[name]} for name in corpus_files]
    corpus_sha256 = _sha256(_canonical_json(manifest))
    corpus_set = set(corpus_files)
    corpus_texts = {
        name: (corpus_path / name).read_text(encoding="utf-8")
        for name in corpus_files
        if Path(name).suffix.lower() == ".txt"
    }
    seen_ids: set[str] = set()
    questions: list[dict[str, Any]] = []
    for index, raw_case in enumerate(raw_questions):
        if not isinstance(raw_case, dict):
            raise ValueError(f"Question {index} must be an object")
        case = dict(raw_case)
        case_id = case.get("id", f"question-{index + 1}")
        query = case.get("query")
        sources = case.get("sources", [])
        spans = case.get("evidence_spans", [])
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen_ids:
            raise ValueError(f"Question IDs must be unique non-empty strings: {case_id!r}")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Question {case_id} requires a non-empty query")
        if not isinstance(sources, list) or any(not isinstance(source, str) for source in sources):
            raise ValueError(f"Question {case_id} sources must be a list of filenames")
        source_names = [Path(source.replace("\\", "/")).name for source in sources]
        if len(source_names) != len(set(source_names)):
            raise ValueError(f"Question {case_id} has duplicate source labels")
        if any(source not in corpus_set for source in source_names):
            raise ValueError(f"Question {case_id} references a missing source document")
        if not isinstance(spans, list):
            raise ValueError(f"Question {case_id} evidence_spans must be a list")
        normalized_spans: list[dict[str, str]] = []
        seen_spans: set[tuple[str, str]] = set()
        for span in spans:
            if not isinstance(span, dict):
                raise ValueError(f"Question {case_id} evidence spans must be objects")
            source = _span_source(span)
            text = _span_text(span)
            if source not in corpus_set or source not in source_names:
                raise ValueError(f"Question {case_id} evidence span references an unexpected source")
            key = (source, normalize_text(text))
            if key in seen_spans:
                raise ValueError(f"Question {case_id} has duplicate evidence span")
            source_text = corpus_texts.get(source)
            if source_text is not None and normalize_text(text) not in normalize_text(source_text):
                raise ValueError(f"Question {case_id} evidence text is not present in {source}")
            seen_spans.add(key)
            normalized_spans.append({"source_file": source, "text": text})
        if source_names and not normalized_spans:
            raise ValueError(f"Question {case_id} requires evidence spans for answerable sources")
        if not source_names and normalized_spans:
            raise ValueError(f"Question {case_id} cannot label evidence spans without sources")
        query_type = case.get("query_type")
        if query_type is not None and query_type not in VALID_QUERY_TYPES:
            raise ValueError(f"Question {case_id} has unsupported query_type: {query_type!r}")
        case["id"] = case_id
        case["query"] = query.strip()
        case["sources"] = source_names
        case["evidence_spans"] = normalized_spans
        case["answerable"] = bool(source_names)
        seen_ids.add(case_id)
        questions.append(case)
    return Dataset(
        corpus_dir=corpus_path,
        question_path=questions_path,
        corpus_files=corpus_files,
        questions=questions,
        question_bytes=question_bytes,
        corpus_sha256=corpus_sha256,
        corpus_file_sha256=file_hashes,
    )


def assert_disjoint_corpora(first: str | Path, second: str | Path) -> tuple[str, ...]:
    """Require two corpus directories to use different document filenames."""

    first_paths = [
        path for path in Path(first).iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    second_paths = [
        path for path in Path(second).iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    first_names = {path.name for path in first_paths}
    second_names = {path.name for path in second_paths}
    overlap = sorted(first_names & second_names)
    first_hashes = {_sha256(path.read_bytes()) for path in first_paths}
    second_hashes = {_sha256(path.read_bytes()) for path in second_paths}
    if overlap or first_hashes & second_hashes:
        detail = ", ".join(overlap) if overlap else "identical document content"
        raise ValueError(f"Corpora must be document-disjoint; overlap: {detail}")
    return tuple(sorted(first_names)) + tuple(sorted(second_names))


def _question_cases(value: Dataset | str | Path | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(value, Dataset):
        return value.questions
    if isinstance(value, list):
        return value
    try:
        raw = json.loads(Path(value).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read question set: {value}") from exc
    if not isinstance(raw, list):
        raise ValueError("Question set must be a JSON array")
    return [case for case in raw if isinstance(case, dict)]


def assert_disjoint_questions(
    first: Dataset | str | Path | list[dict[str, Any]],
    second: Dataset | str | Path | list[dict[str, Any]],
) -> None:
    """Reject calibration/confirmation question ID or normalized-query reuse."""

    first_cases = _question_cases(first)
    second_cases = _question_cases(second)
    first_ids = {case.get("id") for case in first_cases}
    second_ids = {case.get("id") for case in second_cases}
    first_queries = {normalize_text(str(case.get("query", ""))) for case in first_cases}
    second_queries = {normalize_text(str(case.get("query", ""))) for case in second_cases}
    if first_ids & second_ids or first_queries & second_queries:
        raise ValueError("Question sets must be question-disjoint by ID and query")


def _assert_inputs_unchanged(dataset: Dataset) -> None:
    try:
        current_questions = dataset.question_path.read_bytes()
        current_paths = sorted(
            path for path in dataset.corpus_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        current_hashes = {path.name: _sha256(path.read_bytes()) for path in current_paths}
    except OSError as exc:
        raise ValueError("Corpus or questions changed during evaluation") from exc
    if current_questions != dataset.question_bytes:
        raise ValueError("Corpus or questions changed during evaluation")
    if tuple(path.name for path in current_paths) != dataset.corpus_files or current_hashes != dataset.corpus_file_sha256:
        raise ValueError("Corpus or questions changed during evaluation")


def document_scores(contexts: list[dict[str, Any]], relevant: list[str], k: int) -> tuple[float | None, float | None]:
    if not relevant:
        return None, None
    ranked = list(dict.fromkeys(str(item.get("source_file", "")) for item in contexts[:k]))
    expected = set(relevant)
    recall = len(set(ranked) & expected) / len(expected)
    reciprocal_rank = next((1 / rank for rank, name in enumerate(ranked, 1) if name in expected), 0.0)
    return recall, reciprocal_rank


def evidence_scores(
    contexts: list[dict[str, Any]], evidence_spans: list[dict[str, str]], k: int
) -> tuple[float | None, float | None]:
    if not evidence_spans:
        return None, None
    found_ranks: list[int] = []
    for span in evidence_spans:
        source = _span_source(span)
        needle = normalize_text(_span_text(span))
        rank = next(
            (
                index
                for index, context in enumerate(contexts[:k], 1)
                if str(context.get("source_file", "")) == source
                and needle in normalize_text(str(context.get("text", "")))
            ),
            None,
        )
        if rank is not None:
            found_ranks.append(rank)
    recall = len(found_ranks) / len(evidence_spans)
    return recall, (1 / min(found_ranks) if found_ranks else 0.0)


def score_case(case: dict[str, Any], contexts: list[dict[str, Any]], k: int) -> dict[str, Any]:
    document_recall, document_mrr = document_scores(contexts, case["sources"], k)
    evidence_recall, evidence_mrr = evidence_scores(contexts, case["evidence_spans"], k)
    return {
        "document_recall_at_k": document_recall,
        "document_mrr": document_mrr,
        "evidence_recall_at_k": evidence_recall,
        "evidence_mrr": evidence_mrr,
        "retrieved_sources": list(dict.fromkeys(str(item.get("source_file", "")) for item in contexts[:k])),
        "retrieved_chunk_count": len(contexts[:k]),
        "abstained": not contexts,
    }


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row.get("answerable")]
    unanswerable = [row for row in rows if not row.get("answerable")]
    abstentions = [
        row for row in rows if row.get("response", {}).get("status") == "insufficient_context"
    ]
    valid_abstentions = [row for row in abstentions if not row.get("failed")]
    correct_abstentions = [
        row
        for row in unanswerable
        if row.get("response", {}).get("status") == "insufficient_context"
        and not row.get("failed")
    ]
    return {
        "count": len(rows),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "document_metrics": {
            "recall_at_k": _mean(
                [row["document_recall_at_k"] for row in answerable if row["document_recall_at_k"] is not None]
            ),
            "mrr": _mean([row["document_mrr"] for row in answerable if row["document_mrr"] is not None]),
        },
        "evidence_metrics": {
            "recall_at_k": _mean(
                [row["evidence_recall_at_k"] for row in answerable if row["evidence_recall_at_k"] is not None]
            ),
            "mrr": _mean([row["evidence_mrr"] for row in answerable if row["evidence_mrr"] is not None]),
        },
        "abstention": {
            "precision": (len(correct_abstentions) / len(valid_abstentions) if valid_abstentions else None),
            "recall": (len(correct_abstentions) / len(unanswerable) if unanswerable else None),
            "answerable_abstention_rate": (
                sum(
                    1
                    for row in answerable
                    if row.get("response", {}).get("status") == "insufficient_context"
                ) / len(answerable)
                if answerable
                else None
            ),
        },
        "failure_rate": (
            sum(1 for row in rows if row.get("failed")) / len(rows) if rows else None
        ),
        "status_counts": dict(Counter(
            str(row.get("response", {}).get("status", "request_failed"))
            for row in rows
        )),
    }


def _request_json(
    method: str,
    url: str,
    *,
    api_key: str,
    workspace: str,
    **kwargs: Any,
) -> dict[str, Any]:
    supplied_headers = kwargs.pop("headers", {})
    response = requests.request(
        method,
        url,
        headers={
            **supplied_headers,
            "X-API-Key": api_key,
            "X-Workspace-ID": workspace,
        },
        **kwargs,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object from {url}")
    return payload


def _query_case(
    case: dict[str, Any], *, base_url: str, api_key: str, workspace: str, k: int, timeout: int
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        payload = _request_json(
            "POST",
            f"{base_url}/api/v1/query",
            api_key=api_key,
            workspace=workspace,
            json={"query": case["query"], "retrieval_only": True, "use_cache": False},
            timeout=timeout,
        )
        contexts = payload.get("context", [])
        if not isinstance(contexts, list) or any(not isinstance(item, dict) for item in contexts):
            raise ValueError("Query response context must be a list of objects")
        status = payload.get("status")
        failed = status not in {"retrieved", "insufficient_context"}
        result = score_case(case, contexts, k)
        result["abstained"] = status == "insufficient_context"
        if failed:
            # A degraded/error response is infrastructure evidence, not retrieval
            # evidence, even when an implementation accidentally includes context.
            if case["sources"]:
                result["document_recall_at_k"] = 0.0
                result["document_mrr"] = 0.0
            if case["evidence_spans"]:
                result["evidence_recall_at_k"] = 0.0
                result["evidence_mrr"] = 0.0
        return {
            **case,
            **result,
            "response": payload,
            "failed": failed,
            "seconds": time.perf_counter() - started,
        }
    except (requests.RequestException, ValueError, KeyError, TypeError):
        return {
            **case,
            **score_case(case, [], k),
            "response": {"status": "request_failed"},
            "abstained": False,
            "failed": True,
            "error": "request_failed",
            "seconds": time.perf_counter() - started,
        }


def _config(args: argparse.Namespace, base_url: str) -> dict[str, Any]:
    concurrency = getattr(args, "concurrency", 1)
    return {
        "base_url": base_url,
        "workspace": args.workspace,
        "k": args.k,
        "concurrency": concurrency,
        "timeout": args.timeout,
        "poll_interval": args.poll_interval,
        "max_wait": args.max_wait,
        "ingest": bool(getattr(args, "ingest", False)),
        "retrieval_only": True,
        "use_cache": False,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if not args.workspace:
        raise ValueError("A dedicated workspace is required")
    concurrency = getattr(args, "concurrency", 1)
    if args.k < 1 or concurrency < 1:
        raise ValueError("K and concurrency must be positive")
    dataset = load_dataset(args.corpus, args.questions)
    base_url = str(args.base_url).rstrip("/")
    if getattr(args, "ingest", False):
        for name in dataset.corpus_files:
            upload_and_wait(
                base_url,
                dataset.corpus_dir / name,
                args.timeout,
                args.poll_interval,
                args.max_wait,
                args.api_key,
                args.workspace,
            )

    query_args = dict(
        base_url=base_url,
        api_key=args.api_key,
        workspace=args.workspace,
        k=args.k,
        timeout=args.timeout,
    )
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(lambda case: _query_case(case, **query_args), dataset.questions))
    _assert_inputs_unchanged(dataset)
    config = _config(args, base_url)
    config_sha256 = _sha256(_canonical_json(config))
    input_fingerprint = _sha256(
        _canonical_json({"corpus_sha256": dataset.corpus_sha256, "questions_sha256": _sha256(dataset.question_bytes)})
    )
    return {
        "schema_version": 1,
        "evaluator": "chunk-retrieval-v1",
        "evaluated_at": datetime.now(UTC).isoformat(),
        "workspace": args.workspace,
        "k": args.k,
        "retrieval_only": True,
        "run_label": getattr(args, "run_label", None),
        "human_grades": None,
        "server_config": {
            "requested_k": args.k,
            "effective_top_k": None,
            "score_threshold": None,
            "embedding_model": None,
            "verification": "not_returned_by_public_api",
        },
        "corpus": {
            "path": str(dataset.corpus_dir),
            "documents": list(dataset.corpus_files),
            "document_count": len(dataset.corpus_files),
            "files_sha256": dataset.corpus_file_sha256,
            "sha256": dataset.corpus_sha256,
        },
        "questions": {"path": str(dataset.question_path), "count": len(dataset.questions)},
        "questions_sha256": _sha256(dataset.question_bytes),
        "input_fingerprint_sha256": input_fingerprint,
        "config": config,
        "config_sha256": config_sha256,
        **summarize_results(rows),
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("DOCUQUERY_API_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.getenv("DOCUQUERY_API_KEY"))
    parser.add_argument("--workspace", default=os.getenv("DOCUQUERY_WORKSPACE_ID"))
    parser.add_argument("--corpus", default="evaluation/vietnamese/chunks/calibration/corpus")
    parser.add_argument("--questions", default="evaluation/vietnamese/chunks/calibration/questions.json")
    parser.add_argument("--output", default="chunk_retrieval_results.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-wait", type=int, default=600)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--run-label")
    parser.add_argument("--ingest", action="store_true", help="Upload corpus through the public API before querying")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output report")
    args = parser.parse_args()
    if not args.api_key:
        parser.error("--api-key or DOCUQUERY_API_KEY is required")
    if not args.workspace:
        parser.error("--workspace or DOCUQUERY_WORKSPACE_ID must name a dedicated workspace")
    if args.k < 1 or args.concurrency < 1:
        parser.error("--k and --concurrency must be positive")
    output = Path(args.output).resolve()
    question_path = Path(args.questions).resolve()
    if output == question_path:
        parser.error("--output must not overwrite the question labels")
    corpus_path = Path(args.corpus).resolve()
    if corpus_path == output.parent or corpus_path in output.parents:
        parser.error("--output must not be inside the corpus directory")
    if output.exists() and not args.overwrite:
        parser.error("Output already exists; pass --overwrite to replace it")
    try:
        report = evaluate(args)
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        parser.error(str(exc))
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
