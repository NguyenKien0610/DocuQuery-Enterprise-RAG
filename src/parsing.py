"""Bounded document parsing in a disposable subprocess (also on Windows)."""

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import docx2txt
from pypdf import PdfReader


def parse(path: str) -> list[dict]:
    maximum = int(os.getenv("MAX_PARSED_BYTES", "8388608"))
    suffix = Path(path).suffix.lower()
    sections = []
    total = 0
    texts: Iterator[tuple[str, int | None]]
    if suffix == ".pdf":
        reader = PdfReader(path)
        if len(reader.pages) > int(os.getenv("MAX_DOCUMENT_PAGES", "200")):
            raise ValueError("Document page limit exceeded")
        texts = (
            (page.extract_text() or "", index + 1)
            for index, page in enumerate(reader.pages)
        )
    elif suffix == ".docx":
        texts = iter([(docx2txt.process(path) or "", None)])
    elif suffix == ".txt":
        texts = iter([(Path(path).read_text(encoding="utf-8"), None)])
    else:
        raise ValueError("Unsupported document")
    for text, page in texts:
        total += len(text.encode("utf-8"))
        if total > maximum:
            raise ValueError("Parsed text limit exceeded")
        sections.append({"text": text, "page_number": page})
    return sections


def parse_isolated(path: str) -> list[dict]:
    completed = subprocess.run(
        [sys.executable, "-m", "src.parsing", path],
        capture_output=True,
        check=False,
        timeout=float(os.getenv("DOCUMENT_PARSE_TIMEOUT_SECONDS", "60")),
        cwd=Path(__file__).resolve().parents[1],
    )
    if completed.returncode:
        raise ValueError("Document parsing failed or exceeded its limits")
    return json.loads(completed.stdout)


if __name__ == "__main__":
    try:
        sys.stdout.buffer.write(
            json.dumps(parse(sys.argv[1]), ensure_ascii=False).encode("utf-8")
        )
    except Exception:
        sys.exit(1)
