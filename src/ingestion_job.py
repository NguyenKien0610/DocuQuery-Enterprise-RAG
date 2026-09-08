"""Disposable ingestion process; its OS lock is released if the worker kills it."""

import json
import sys

from src import state
from src.rag_engine import ingest_document


def main():
    arguments = json.loads(sys.stdin.buffer.read())
    try:
        result = ingest_document(*arguments)
    except state.WorkspaceBusyError:
        sys.exit(75)
    except state.TaskCancelledError:
        sys.exit(76)
    except Exception:
        import logging

        logging.exception("Ingestion job failed")
        sys.exit(1)
    sys.stdout.buffer.write(json.dumps(result).encode("utf-8"))


if __name__ == "__main__":
    main()
