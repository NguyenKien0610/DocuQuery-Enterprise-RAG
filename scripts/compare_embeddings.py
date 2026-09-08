"""Bounded document-level embedding retrieval comparison.

Each UTF-8 TXT file is embedded as one document. This is an experiment for
comparing encoders, not the production chunked retrieval pipeline.
"""

import argparse
import hashlib
import json
import statistics
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

from scripts.evaluate_docuquery import abstention_scores, retrieval_scores

BASELINE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CANDIDATE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_MODELS = (BASELINE_MODEL, CANDIDATE_MODEL)
THRESHOLD_GRID = (0.2, 0.3, 0.35, 0.4, 0.5, 0.6)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_corpus(path: str | Path) -> list[dict[str, Any]]:
    corpus_path = Path(path)
    if not corpus_path.is_dir():
        raise ValueError(f"Corpus directory does not exist: {corpus_path}")
    files = sorted(corpus_path.glob("*.txt"), key=lambda item: item.name)
    if not files:
        raise ValueError(f"Corpus directory contains no TXT files: {corpus_path}")
    documents = []
    for file in files:
        raw = file.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Corpus file is not UTF-8: {file.name}") from exc
        documents.append({"filename": file.name, "text": text, "raw": raw})
    return documents


def _query_key(query: str) -> str:
    return unicodedata.normalize("NFC", " ".join(query.split()).casefold())


def _normalise_questions(payload: Any, path: str | Path) -> list[dict[str, Any]]:
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Question file must contain a non-empty JSON list: {path}")
    questions, seen = [], set()
    for index, item in enumerate(payload, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Question {index} must be a JSON object: {path}")
        query = item.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Question {index} needs a non-empty query: {path}")
        key = _query_key(query)
        if key in seen:
            raise ValueError(f"Question queries must be unique within a file: {query}")
        seen.add(key)
        sources = item.get("sources")
        if not isinstance(sources, list) or any(
            not isinstance(source, str) or not source for source in sources
        ):
            raise ValueError(f"Question {index} needs a sources list: {path}")
        language, question_id = item.get("language", "unknown"), item.get("id", f"row-{index}")
        if not isinstance(language, str) or not language:
            raise ValueError(f"Question {index} needs a language string: {path}")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"Question {index} needs a non-empty id: {path}")
        questions.append(
            {
                "id": question_id,
                "language": language,
                "query": query,
                "sources": list(dict.fromkeys(sources)),
            }
        )
    return questions


def _load_questions(path: str | Path) -> list[dict[str, Any]]:
    question_path = Path(path)
    if not question_path.is_file():
        raise ValueError(f"Question file does not exist: {question_path}")
    try:
        payload = json.loads(question_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read question JSON: {question_path}") from exc
    return _normalise_questions(payload, question_path)


def validate_question_sets(
    calibration: list[dict[str, Any]], questions: list[dict[str, Any]]
) -> None:
    calibration_queries = {_query_key(item["query"]) for item in calibration}
    test_queries = {_query_key(item["query"]) for item in questions}
    if calibration_queries & test_queries:
        raise ValueError("Calibration and test query strings must be disjoint")


def _normalise_embeddings(values: Any) -> np.ndarray:
    embeddings = np.asarray(values, dtype=float)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    if embeddings.ndim != 2 or embeddings.shape[1] == 0:
        raise ValueError("Encoder must return a two-dimensional embedding matrix")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if not np.isfinite(embeddings).all() or not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("Encoder returned non-finite or zero vectors")
    return embeddings / norms


def _encode(encoder: Any, texts: list[str]) -> np.ndarray:
    values = encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = _normalise_embeddings(values)
    if embeddings.shape[0] != len(texts):
        raise ValueError("Encoder returned incorrect number of vectors")
    return embeddings


def document_token_lengths(
    encoder: Any, texts: list[str], names: list[str] | None = None
) -> list[int]:
    """Return untruncated token lengths and reject documents beyond the model window."""
    tokenizer, max_seq_length = getattr(encoder, "tokenizer", None), getattr(
        encoder, "max_seq_length", None
    )
    if tokenizer is None or max_seq_length is None:
        raise ValueError(
            "Encoder must expose tokenizer and max_seq_length for document-window validation"
        )
    try:
        encoded = tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)
        lengths = [len(ids) for ids in encoded["input_ids"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Encoder tokenizer did not return input_ids") from exc
    if len(lengths) != len(texts):
        raise ValueError("Encoder returned incorrect number of token sequences")
    for index, length in enumerate(lengths):
        if length > max_seq_length:
            name = names[index] if names else f"document {index + 1}"
            raise ValueError(
                f"Document exceeds model window: {name} has {length} tokens, "
                f"max_seq_length is {max_seq_length}"
            )
    return lengths


def _metadata(model_id: str, encoder: Any) -> dict[str, Any]:
    revision = getattr(encoder, "revision", None) or getattr(encoder, "_revision", None)
    config = getattr(encoder, "config", None)
    config_hash = getattr(config, "_commit_hash", None)
    if config_hash is None and callable(getattr(encoder, "_first_module", None)):
        module = encoder._first_module()
        config_hash = getattr(
            getattr(getattr(module, "auto_model", None), "config", None),
            "_commit_hash",
            None,
        )
    return {
        "model_id": model_id,
        "revision": revision or config_hash,
        "config_commit_hash": config_hash,
        "max_seq_length": int(encoder.max_seq_length),
    }


def _retrieval_rows(
    questions: list[dict[str, Any]], query_embeddings: np.ndarray,
    document_embeddings: np.ndarray, filenames: list[str], k: int, threshold: float
) -> list[dict[str, Any]]:
    similarities = query_embeddings @ document_embeddings.T
    rows = []
    for question, scores in zip(questions, similarities, strict=True):
        ranked = np.argsort(-scores, kind="stable")[:k]
        best_score = float(scores[ranked[0]]) if len(ranked) else 0.0
        selected = [index for index in ranked if float(scores[index]) >= threshold]
        retrieved = [
            {"filename": filenames[index], "score": float(scores[index])}
            for index in selected
        ]
        retrieved_filenames = [item["filename"] for item in retrieved]
        recall, reciprocal_rank = retrieval_scores(
            retrieved_filenames, question["sources"], k
        )
        candidates = [
            {"filename": filenames[index], "score": float(scores[index])} for index in ranked
        ]
        ranking_recall, ranking_mrr = retrieval_scores(
            [item["filename"] for item in candidates], question["sources"], k
        )
        error_type = None
        if not question["sources"]:
            if retrieved:
                error_type = "unsupported_context"
        elif ranking_recall < 1:
            error_type = "ranking_miss"
        elif recall < 1:
            error_type = "threshold_rejection"
        rows.append(
            {
                **question,
                "retrieved": retrieved,
                "retrieved_filenames": retrieved_filenames,
                "max_score": best_score,
                "abstained": not retrieved,
                "recall": recall,
                "reciprocal_rank": reciprocal_rank,
                "candidates": candidates,
                "ranking_recall": ranking_recall,
                "ranking_reciprocal_rank": ranking_mrr,
                "ranking_top1_hit": bool(candidates and candidates[0]["filename"] in question["sources"]),
                "error_type": error_type,
            }
        )
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row["sources"]]
    abstention = abstention_scores(
        [
            {
                "sources": row["sources"],
                "response": {
                    "status": "insufficient_context" if row["abstained"] else "retrieved"
                },
            }
            for row in rows
        ]
    )
    return {
        "count": len(rows),
        "answerable_count": len(answerable),
        "document_recall_at_k": statistics.mean(row["recall"] for row in answerable)
        if answerable
        else None,
        "document_mrr": statistics.mean(row["reciprocal_rank"] for row in answerable)
        if answerable
        else None,
        "ranking_recall_at_k": statistics.mean(row["ranking_recall"] for row in answerable)
        if answerable else None,
        "ranking_mrr": statistics.mean(row["ranking_reciprocal_rank"] for row in answerable)
        if answerable else None,
        "ranking_top1_accuracy": statistics.mean(row["ranking_top1_hit"] for row in answerable)
        if answerable else None,
        "error_counts": dict(Counter(row["error_type"] for row in rows if row["error_type"])),
        **abstention,
    }


def _split_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": rows,
        "aggregate": _aggregate(rows),
        "by_language": {
            language: _aggregate([row for row in rows if row["language"] == language])
            for language in sorted({row["language"] for row in rows})
        },
    }


def balanced_selection_score(
    document_recall: float | None, abstention_recall: float | None
) -> float:
    if document_recall is None or abstention_recall is None:
        raise ValueError("Calibration needs both answerable and unanswerable questions")
    return (document_recall + abstention_recall) / 2


def select_threshold(
    metrics_by_threshold: dict[float, dict[str, Any]],
    threshold_grid: tuple[float, ...] = THRESHOLD_GRID,
) -> tuple[float, float]:
    choices = [
        (
            balanced_selection_score(
                metrics_by_threshold[threshold]["document_recall_at_k"],
                metrics_by_threshold[threshold]["abstention_recall"],
            ),
            threshold,
        )
        for threshold in threshold_grid
    ]
    score, threshold = max(choices, key=lambda choice: (choice[0], choice[1]))
    return threshold, score


def load_encoder(model_id: str, revision: str | None = None) -> Any:
    """Load SentenceTransformer lazily, only when an experiment is run."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id, revision=revision)


def run_experiment(
    *, corpus: str | Path, calibration: str | Path, questions: str | Path | None = None,
    model_ids: list[str] | tuple[str, ...] = DEFAULT_MODELS, k: int = 5,
    threshold_grid: tuple[float, ...] = THRESHOLD_GRID,
    load_model: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    if k < 1 or not model_ids or not threshold_grid:
        raise ValueError("A positive K, at least one model, and a grid are required")
    if any(not np.isfinite(threshold) or threshold < 0 or threshold > 1 for threshold in threshold_grid):
        raise ValueError("Thresholds must be between 0 and 1")
    documents = _load_corpus(corpus)
    calibration_raw = Path(calibration).read_bytes()
    test_raw = Path(questions).read_bytes() if questions is not None else None
    calibration_rows = _normalise_questions(json.loads(calibration_raw), calibration)
    test_rows = (
        _normalise_questions(json.loads(test_raw), questions)
        if test_raw is not None and questions is not None else []
    )
    validate_question_sets(calibration_rows, test_rows)
    if not any(row["sources"] for row in calibration_rows) or not any(
        not row["sources"] for row in calibration_rows
    ):
        raise ValueError("Calibration needs answerable and unanswerable questions")
    filenames, known_sources = [d["filename"] for d in documents], {
        d["filename"] for d in documents
    }
    for row in calibration_rows + test_rows:
        if unknown := set(row["sources"]) - known_sources:
            raise ValueError(f"Question references missing corpus files: {sorted(unknown)}")
    texts, loader, models = [d["text"] for d in documents], load_model or load_encoder, []
    for model_id in model_ids:
        encoder = loader(model_id)
        passage_prefix = "passage: " if model_id == "intfloat/multilingual-e5-small" else ""
        query_prefix = "query: " if passage_prefix else ""
        passages = [passage_prefix + text for text in texts]
        calibration_queries = [query_prefix + row["query"] for row in calibration_rows]
        token_lengths = document_token_lengths(encoder, passages, filenames)
        document_token_lengths(encoder, calibration_queries)
        document_embeddings = _encode(encoder, passages)
        calibration_embeddings = _encode(encoder, calibration_queries)
        dimensions = {
            document_embeddings.shape[1], calibration_embeddings.shape[1]
        }
        if len(dimensions) != 1:
            raise ValueError(f"Encoder returned inconsistent dimensions for {model_id}")
        rows_by_threshold, metrics_by_threshold = {}, {}
        for threshold in threshold_grid:
            rows = _retrieval_rows(
                calibration_rows, calibration_embeddings, document_embeddings,
                filenames, k, threshold
            )
            rows_by_threshold[threshold], metrics_by_threshold[threshold] = rows, _aggregate(rows)
        threshold, selection_score = select_threshold(metrics_by_threshold, threshold_grid)
        calibration_report = _split_report(rows_by_threshold[threshold])
        calibration_report.update(
            {"threshold": threshold, "grid": list(threshold_grid), "selection_score": selection_score,
             "curve": [{"threshold": value, **metrics_by_threshold[value]} for value in threshold_grid]}
        )
        test_report = None
        if test_rows:
            test_queries = [query_prefix + row["query"] for row in test_rows]
            document_token_lengths(encoder, test_queries)
            test_embeddings = _encode(encoder, test_queries)
            if test_embeddings.shape[1] != document_embeddings.shape[1]:
                raise ValueError(f"Encoder returned inconsistent dimensions for {model_id}")
            test_report = _split_report(
                _retrieval_rows(test_rows, test_embeddings, document_embeddings, filenames, k, threshold)
            )
        models.append(
            {
                **_metadata(model_id, encoder), "document_token_lengths": token_lengths,
                "passage_prefix": passage_prefix, "query_prefix": query_prefix,
                "calibration": calibration_report, "test": test_report,
            }
        )
    corpus_sha256 = sha256_bytes(
        b"".join(d["filename"].encode() + b"\0" + d["raw"] + b"\0" for d in documents)
    )
    return {
        "schema_version": 1, "experiment": "embedding_retrieval_comparison",
        "document_level": True,
        "scope_note": "Full TXT documents are embedded as single vectors; this is not the production chunk pipeline.",
        "k": k, "threshold_grid": list(threshold_grid),
        "fingerprints": {
            "corpus_sha256": corpus_sha256,
            "calibration_questions_sha256": sha256_bytes(calibration_raw),
            "test_questions_sha256": sha256_bytes(test_raw) if test_raw is not None else None,
        },
        "models": models,
    }


def run_confirmation(
    *, corpus: str | Path, questions: str | Path, selection_report: str | Path,
    load_model: Callable[[str, str], Any] | None = None,
) -> dict[str, Any]:
    """Replay a baseline/candidate pair without threshold selection or config overrides."""
    selection_raw = Path(selection_report).read_bytes()
    selection = json.loads(selection_raw)
    documents = _load_corpus(corpus)
    question_raw = Path(questions).read_bytes()
    rows = _normalise_questions(json.loads(question_raw), questions)
    filenames = [doc["filename"] for doc in documents]
    known_sources = set(filenames)
    if any(set(row["sources"]) - known_sources for row in rows):
        raise ValueError("Confirmation questions reference missing corpus files")
    configs = []
    try:
        if selection["experiment"] != "embedding_retrieval_comparison" or selection["document_level"] is not True:
            raise ValueError("Expected a document-level calibration report")
        k = selection["k"]
        if type(k) is not int or k < 1 or len(selection["models"]) != 2:
            raise ValueError("Confirmation requires positive K and exactly baseline then candidate")
        for model in selection["models"]:
            if model["test"] is not None:
                raise ValueError("Selection report must be calibration-only")
            model_id, revision = model["model_id"], model["revision"]
            if not isinstance(model_id, str) or not model_id or not isinstance(revision, str) or not revision:
                raise ValueError("Model ID and pinned revision are required")
            threshold = model["calibration"]["threshold"]
            if type(threshold) not in (int, float) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError("Invalid frozen threshold")
            prefixes = (model["passage_prefix"], model["query_prefix"])
            if any(not isinstance(prefix, str) for prefix in prefixes):
                raise ValueError("Frozen input prefixes must be strings")
            calibration_rows = _normalise_questions(model["calibration"]["rows"], selection_report)
            validate_question_sets(calibration_rows, rows)
            calibration_sources = {source for row in calibration_rows for source in row["sources"]}
            if known_sources & calibration_sources:
                raise ValueError("Confirmation corpus and positive calibration documents must be disjoint")
            configs.append((model_id, revision, threshold, *prefixes))
        if configs[0][0] == configs[1][0]:
            raise ValueError("Baseline and candidate model IDs must differ")
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Malformed calibration selection report") from exc

    models = []
    for model_id, revision, threshold, passage_prefix, query_prefix in configs:
        encoder = (load_model or load_encoder)(model_id, revision)
        metadata = _metadata(model_id, encoder)
        if metadata["revision"] != revision:
            raise ValueError(f"Loaded revision differs from frozen selection: {model_id}")
        passages = [passage_prefix + doc["text"] for doc in documents]
        queries = [query_prefix + row["query"] for row in rows]
        token_lengths = document_token_lengths(encoder, passages, filenames)
        document_token_lengths(encoder, queries)
        document_vectors, query_vectors = _encode(encoder, passages), _encode(encoder, queries)
        if document_vectors.shape[1] != query_vectors.shape[1]:
            raise ValueError(f"Encoder returned inconsistent dimensions for {model_id}")
        test = _split_report(_retrieval_rows(rows, query_vectors, document_vectors, filenames, k, threshold))
        models.append({
            **metadata, "threshold": threshold, "passage_prefix": passage_prefix,
            "query_prefix": query_prefix, "document_token_lengths": token_lengths, "test": test,
        })

    baseline, candidate = [model["test"] for model in models]
    baseline_recall = baseline["by_language"].get("vi", {}).get("document_recall_at_k")
    candidate_recall = candidate["by_language"].get("vi", {}).get("document_recall_at_k")
    baseline_abstention = baseline["aggregate"]["abstention_recall"]
    candidate_abstention = candidate["aggregate"]["abstention_recall"]
    evaluable = all(value is not None for value in (
        baseline_recall, candidate_recall, baseline_abstention, candidate_abstention,
    ))
    return {
        "schema_version": 2, "experiment": "frozen_threshold_confirmation",
        "document_level": True, "k": k,
        "scope_note": "Offline whole-document retrieval; no answer generation. Replaying a seen set is not fresh evidence.",
        "fingerprints": {
            "selection_report_sha256": sha256_bytes(selection_raw),
            "test_questions_sha256": sha256_bytes(question_raw),
            "corpus_sha256": sha256_bytes(b"".join(
                doc["filename"].encode() + b"\0" + doc["raw"] + b"\0" for doc in documents
            )),
        },
        "models": models,
        "promotion_gate": {
            "rule": "Strictly improve Vietnamese recall without reducing unanswerable abstention",
            "evaluable": evaluable,
            "passed": bool(evaluable and candidate_recall > baseline_recall and candidate_abstention >= baseline_abstention),
            "baseline_vi_recall": baseline_recall, "candidate_vi_recall": candidate_recall,
            "baseline_abstention": baseline_abstention, "candidate_abstention": candidate_abstention,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--calibration")
    mode.add_argument("--selection-report", help="Replay pinned calibration config; requires --questions")
    parser.add_argument("--questions", help="Omit for calibration-only selection; no test file is read")
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--k", type=int)
    parser.add_argument("--threshold-grid", nargs="+", type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    inputs = [Path(path) for path in (args.calibration, args.questions, args.selection_report) if path]
    inputs.extend(Path(args.corpus).glob("*.txt"))
    if any(output.resolve() == path.resolve() or (output.exists() and path.exists() and output.samefile(path)) for path in inputs):
        parser.error("Output must not overwrite an input file")
    if output.exists() and not args.overwrite:
        parser.error(f"Output exists; pass --overwrite to replace it: {output}")
    try:
        if args.selection_report:
            if not args.questions or any(value is not None for value in (args.models, args.k, args.threshold_grid)):
                parser.error("Confirmation requires --questions and forbids model/K/threshold overrides")
            report = run_confirmation(
                corpus=args.corpus, questions=args.questions, selection_report=args.selection_report,
            )
        else:
            report = run_experiment(
                corpus=args.corpus, calibration=args.calibration, questions=args.questions,
                model_ids=args.models or DEFAULT_MODELS, k=args.k if args.k is not None else 5,
                threshold_grid=tuple(args.threshold_grid) if args.threshold_grid is not None else THRESHOLD_GRID,
            )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w" if args.overwrite else "x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"output": str(output), "models": [model["model_id"] for model in report["models"]]}, indent=2))


if __name__ == "__main__":
    main()
