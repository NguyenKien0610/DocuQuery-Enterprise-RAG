import os
from pathlib import Path

from celery import Celery
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

from src.rag_engine import ingest_document

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

celery_app = Celery(
    "docuquery_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
)


@celery_app.task(name="process_document_task")
def process_document_task(file_path: str) -> dict:
    return ingest_document(file_path)
