"""Evaluate a dedicated workspace through the public API; no automatic reset."""

import argparse
import json
import os
import statistics
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import requests


def retrieval_scores(
    sources: list[str], relevant: list[str], k: int
) -> tuple[float, float]:
    ranked = list(dict.fromkeys(sources))[:k]
    expected = set(relevant)
    recall = len(set(ranked) & expected) / len(expected) if expected else 0.0
    reciprocal_rank = next(
        (1 / rank for rank, name in enumerate(ranked, 1) if name in expected), 0.0
    )
    return recall, reciprocal_rank


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def evaluate(args: argparse.Namespace) -> dict:
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    if not questions or args.concurrency < 1 or args.k < 1:
        raise ValueError("Nonempty questions and positive concurrency/K are required")
    headers = {"X-API-Key": args.api_key, "X-Workspace-ID": args.workspace}
    base = args.base_url.rstrip("/")

    def query(case):
        started = time.perf_counter()
        try:
            response = requests.post(
                f"{base}/api/v1/query",
                headers=headers,
                json={"query": case["query"], "use_cache": args.use_cache},
                timeout=120,
            )
            response.raise_for_status()
            payload = response.json()
            sources = [chunk["source_file"] for chunk in payload.get("context", [])]
            recall, mrr = retrieval_scores(sources, case["sources"], args.k)
            return {
                **case,
                "response": payload,
                "recall": recall,
                "reciprocal_rank": mrr,
                "failed": payload.get("status") != "generated",
                "seconds": time.perf_counter() - started,
                "faithfulness": None,
                "answer_relevance": None,
            }
        except (requests.RequestException, ValueError, KeyError):
            return {
                **case,
                "failed": True,
                "seconds": time.perf_counter() - started,
                "recall": 0.0,
                "reciprocal_rank": 0.0,
                "error": "request_failed",
            }

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(query, questions))
    elapsed = time.perf_counter() - started
    latencies = [row["seconds"] for row in results]
    generated_latencies = [row["seconds"] for row in results if not row["failed"]]
    return {
        "evaluated_at": datetime.now(UTC).isoformat(),
        "workspace": args.workspace,
        "k": args.k,
        "concurrency": args.concurrency,
        "use_cache": args.use_cache,
        "reranker": False,
        "document_recall_at_k": statistics.mean(row["recall"] for row in results),
        "document_mrr": statistics.mean(row["reciprocal_rank"] for row in results),
        "failure_rate": statistics.mean(row["failed"] for row in results),
        "status_counts": dict(
            Counter(
                row.get("response", {}).get("status", "request_failed")
                for row in results
            )
        ),
        "generated_p50_seconds": percentile(generated_latencies, 0.5)
        if generated_latencies
        else None,
        "generated_p95_seconds": percentile(generated_latencies, 0.95)
        if generated_latencies
        else None,
        "p50_seconds": percentile(latencies, 0.5),
        "p95_seconds": percentile(latencies, 0.95),
        "requests_per_second": len(results) / elapsed,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.getenv("DOCUQUERY_API_BASE_URL", "http://localhost:8000"),
    )
    parser.add_argument("--api-key", default=os.getenv("DOCUQUERY_API_KEY"))
    parser.add_argument("--workspace", default=f"eval-{uuid.uuid4().hex}")
    parser.add_argument("--questions", default="evaluation/questions.json")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument(
        "--ingest", action="store_true", help="Upload bundled corpus before evaluation"
    )
    parser.add_argument("--output", default="benchmark_results_quality.json")
    parser.add_argument(
        "--review-results", help="Summarize a JSON result after human grading"
    )
    args = parser.parse_args()
    if args.review_results:
        report = json.loads(Path(args.review_results).read_text(encoding="utf-8"))
        print(json.dumps(summarize_review(report["results"]), indent=2))
        return
    if not args.api_key:
        parser.error("DOCUQUERY_API_KEY or --api-key is required")
    if args.concurrency < 1 or args.k < 1:
        parser.error("--concurrency and --k must be positive")
    if args.ingest:
        # Run with python -m scripts.evaluate_docuquery from the repository root.
        from scripts.benchmark_docuquery import upload_and_wait

        for document in sorted(Path("evaluation/corpus").glob("*.txt")):
            upload_and_wait(
                args.base_url.rstrip("/"),
                document,
                120,
                2,
                600,
                args.api_key,
                args.workspace,
            )
    result = evaluate(args)
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "results"}, indent=2
        )
    )


def summarize_review(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No reviewed responses")
    metrics = {}
    for metric in ("faithfulness", "answer_relevance"):
        scores = [row.get(metric) for row in rows]
        if any(type(score) is not int or score not in (0, 1) for score in scores):
            raise ValueError(f"Every response needs a human 0/1 grade for {metric}")
        metrics[metric] = statistics.mean(
            int(score) for score in scores if score is not None
        )
    return {"reviewed_responses": len(rows), **metrics}


if __name__ == "__main__":
    main()
