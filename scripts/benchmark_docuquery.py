import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import requests


def _request_json(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    response = requests.request(method, url, **kwargs)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object from {url}, got {type(payload).__name__}.")
    return payload


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * percentile))
    return ordered[index]


def reset_workspace(base_url: str, timeout: int) -> dict[str, Any]:
    return _request_json("DELETE", f"{base_url}/api/v1/workspace/reset", timeout=timeout)


def upload_and_wait(
    base_url: str,
    document_path: Path,
    timeout: int,
    poll_interval: float,
    max_wait: int,
) -> dict[str, Any]:
    content_type = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".txt": "text/plain",
    }.get(document_path.suffix.lower(), "application/octet-stream")

    started_at = time.perf_counter()
    with document_path.open("rb") as file_obj:
        upload_payload = _request_json(
            "POST",
            f"{base_url}/api/v1/documents/upload",
            files={"file": (document_path.name, file_obj, content_type)},
            timeout=timeout,
        )

    task_id = str(upload_payload["task_id"])
    poll_count = 0

    while True:
        if time.perf_counter() - started_at > max_wait:
            raise TimeoutError(f"Task {task_id} did not finish within {max_wait} seconds.")

        status_payload = _request_json(
            "GET",
            f"{base_url}/api/v1/documents/status/{task_id}",
            timeout=timeout,
        )
        poll_count += 1
        status = str(status_payload.get("status", "")).upper()

        if status == "SUCCESS":
            elapsed = time.perf_counter() - started_at
            result = status_payload.get("result") or {}
            return {
                "task_id": task_id,
                "status": status,
                "elapsed_seconds": elapsed,
                "poll_count": poll_count,
                "chunks_indexed": result.get("chunks_indexed"),
                "source": result.get("source"),
            }

        if status == "FAILURE":
            raise RuntimeError(f"Task {task_id} failed: {status_payload.get('error')}")

        time.sleep(poll_interval)


def run_query(base_url: str, query: str, timeout: int) -> dict[str, Any]:
    started_at = time.perf_counter()
    payload = _request_json(
        "POST",
        f"{base_url}/api/v1/query",
        json={"query": query},
        timeout=timeout,
    )
    elapsed = time.perf_counter() - started_at
    return {
        "elapsed_seconds": elapsed,
        "cached": bool(payload.get("cached", False)),
        "answer_chars": len(str(payload.get("answer", ""))),
        "context_count": len(payload.get("context", []) or []),
    }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    base_url = args.base_url.rstrip("/")
    document_path = Path(args.file).resolve()
    if not document_path.exists():
        raise FileNotFoundError(document_path)

    reset_payload = reset_workspace(base_url, args.timeout) if args.reset else None
    ingestion = upload_and_wait(
        base_url=base_url,
        document_path=document_path,
        timeout=args.timeout,
        poll_interval=args.poll_interval,
        max_wait=args.max_wait,
    )

    cache_miss = run_query(base_url, args.query, args.timeout)
    cache_hits = [run_query(base_url, args.query, args.timeout) for _ in range(args.cache_hit_runs)]
    hit_latencies = [item["elapsed_seconds"] for item in cache_hits]

    return {
        "base_url": base_url,
        "document": {
            "path": str(document_path),
            "name": document_path.name,
            "size_bytes": document_path.stat().st_size,
        },
        "reset": reset_payload,
        "ingestion": ingestion,
        "query": {
            "text": args.query,
            "cache_miss": cache_miss,
            "cache_hit_runs": cache_hits,
            "cache_hit_average_seconds": statistics.mean(hit_latencies) if hit_latencies else 0.0,
            "cache_hit_p95_seconds": _percentile(hit_latencies, 0.95),
            "cache_speedup": (
                cache_miss["elapsed_seconds"] / statistics.mean(hit_latencies)
                if hit_latencies and statistics.mean(hit_latencies) > 0
                else None
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DocuQuery upload and query flows.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--file", required=True, help="Document file to upload for benchmarking.")
    parser.add_argument("--query", required=True, help="Question to benchmark.")
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-wait", type=int, default=600)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--cache-hit-runs", type=int, default=5)
    parser.add_argument("--reset", action="store_true", help="Reset workspace before benchmarking.")
    args = parser.parse_args()

    results = benchmark(args)
    output_path = Path(args.output)
    rendered_results = json.dumps(results, ensure_ascii=False, indent=2)
    output_path.write_text(rendered_results, encoding="utf-8")
    sys.stdout.buffer.write(rendered_results.encode("utf-8"))
    sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    main()
