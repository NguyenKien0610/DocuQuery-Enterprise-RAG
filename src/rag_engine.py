import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import docx2txt
import redis
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "docuquery_hybrid_v1")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
CACHE_TTL_SECONDS = 3600
FALLBACK_CACHE_TTL_SECONDS = 300
TASK_WORKSPACE_TTL_SECONDS = 86400
WORKSPACE_LOCK_TIMEOUT_SECONDS = int(os.getenv("WORKSPACE_LOCK_TIMEOUT_SECONDS", "600"))
WORKSPACE_LOCK_BLOCKING_TIMEOUT_SECONDS = float(
    os.getenv("WORKSPACE_LOCK_BLOCKING_TIMEOUT_SECONDS", "1")
)

TOP_K = int(os.getenv("RAG_TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "models/gemini-3.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

qdrant_client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
    check_compatibility=False,
)
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=True,
)

@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=GEMINI_CHAT_MODEL,
        api_key=GOOGLE_API_KEY,
        retries=0,
    )

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def _task_workspace_key(task_id: str) -> str:
    return f"rag:task_workspace:{task_id}"


def register_task_workspace(task_id: str, workspace_id: str) -> None:
    redis_client.setex(
        _task_workspace_key(task_id),
        TASK_WORKSPACE_TTL_SECONDS,
        workspace_id,
    )


def remove_task_workspace(task_id: str) -> None:
    redis_client.delete(_task_workspace_key(task_id))


def task_belongs_to_workspace(task_id: str, workspace_id: str) -> bool:
    return redis_client.get(_task_workspace_key(task_id)) == workspace_id

answer_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Ban la mot tro ly AI phan tich tai lieu chuyen nghiep. "
        "Dua vao [Context] duoi day, hay tra loi [Question] cua nguoi dung. "
        "MENH LENH: Hay phan tich ky y dinh cua nguoi dung. "
        "Neu ho yeu cau tom tat ngan gon, hay tra loi suc tich bang gach dau dong. "
        "Neu ho yeu cau trinh bay chi tiet, giai thich sau hoac can ke, hay tra loi "
        "that day du, chi tiet va khong gioi han do dai. "
        "Chi su dung thong tin trong Context.\n\n"
        "[Context]\n{context}\n\n"
        "[Question]\n{question}\n\n"
        "[Answer]"
    ),
)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class WorkspaceBusyError(RuntimeError):
    pass


@contextmanager
def workspace_lock(workspace_id: str) -> Iterator[None]:
    lock = redis_client.lock(
        name=f"rag:workspace_lock:{workspace_id}",
        timeout=WORKSPACE_LOCK_TIMEOUT_SECONDS,
        blocking_timeout=WORKSPACE_LOCK_BLOCKING_TIMEOUT_SECONDS,
    )
    if not lock.acquire(blocking=True):
        raise WorkspaceBusyError(f"Workspace is busy: {workspace_id}")
    try:
        yield
    finally:
        lock.release()


class _PdfDocumentLoader:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def load(self) -> list[dict[str, Any]]:
        reader = PdfReader(self.file_path)
        return [
            {
                "text": page.extract_text() or "",
                "page_number": index + 1,
            }
            for index, page in enumerate(reader.pages)
        ]


class _DocxDocumentLoader:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def load(self) -> list[dict[str, Any]]:
        return [{"text": docx2txt.process(self.file_path) or "", "page_number": None}]


class _TextDocumentLoader:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def load(self) -> list[dict[str, Any]]:
        return [
            {
                "text": Path(self.file_path).read_text(encoding="utf-8"),
                "page_number": None,
            }
        ]


def _collection_exists() -> bool:
    collections = qdrant_client.get_collections().collections
    names = {collection.name for collection in collections}
    return COLLECTION_NAME in names


def _create_document_loader(file_path: str):
    file_extension = Path(file_path).suffix.lower()

    if file_extension == ".pdf":
        return _PdfDocumentLoader(file_path)
    if file_extension == ".docx":
        return _DocxDocumentLoader(file_path)
    if file_extension == ".txt":
        return _TextDocumentLoader(file_path)

    raise ValueError(f"Unsupported file extension: {file_extension}")


def _load_document_sections(file_path: str) -> list[dict[str, Any]]:
    loader = _create_document_loader(file_path)
    return loader.load()


def _ensure_collection(vector_size: int) -> None:
    if _collection_exists():
        return

    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )


def ensure_qdrant_collection() -> None:
    if _collection_exists():
        return

    probe_vector = get_embeddings().embed_query("qdrant_collection_init")
    _ensure_collection(vector_size=len(probe_vector))


def _workspace_filter(workspace_id: str) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="workspace_id",
                match=models.MatchValue(value=workspace_id),
            )
        ]
    )


def _delete_workspace_files(upload_root: Path, workspace_id: str) -> int:
    resolved_root = upload_root.resolve()
    workspace_dir = (resolved_root / workspace_id).resolve()
    if workspace_dir.parent != resolved_root:
        raise ValueError("Workspace upload path escapes the upload root.")
    if not workspace_dir.exists():
        return 0

    deleted_files = 0
    for path in workspace_dir.iterdir():
        if path.is_file():
            path.unlink()
            deleted_files += 1
    workspace_dir.rmdir()
    return deleted_files


def reset_workspace(workspace_id: str, upload_root: Path) -> dict[str, Any]:
    with workspace_lock(workspace_id):
        deleted_files = _delete_workspace_files(upload_root, workspace_id)
        if _collection_exists():
            qdrant_client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.FilterSelector(
                    filter=_workspace_filter(workspace_id)
                ),
                wait=True,
            )
        corpus_version = advance_corpus_version(workspace_id)

    return {
        "status": "workspace_reset",
        "workspace_id": workspace_id,
        "collection": COLLECTION_NAME,
        "corpus_version": corpus_version,
        "vector_store_cleared": True,
        "deleted_files": deleted_files,
    }


def get_corpus_version(workspace_id: str) -> int:
    stored_version = redis_client.get(f"rag:corpus_version:{workspace_id}")
    return int(stored_version) if stored_version is not None else 0


def advance_corpus_version(workspace_id: str) -> int:
    return int(redis_client.incr(f"rag:corpus_version:{workspace_id}"))


def _cache_key(query_text: str, workspace_id: str, corpus_version: int) -> str:
    normalized = query_text.strip().lower()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"rag:answer:{workspace_id}:{corpus_version}:{digest}"


def _response_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content)


def _serialize_context_chunk(result: Any) -> dict[str, Any] | None:
    payload = getattr(result, "payload", None) or {}
    text = str(payload.get("text", "")).strip()
    if not text:
        return None

    source_path = str(payload.get("source", "")).strip()
    source_file = str(payload.get("source_file", "")).strip()
    if not source_file:
        source_file = Path(source_path).name if source_path else "Unknown source"
    document_id = str(payload.get("document_id", "")).strip()
    chunk_index = int(payload.get("chunk_index", -1))
    page_number = payload.get("page_number")
    if page_number is not None:
        try:
            page_number = int(page_number)
        except (TypeError, ValueError):
            page_number = None

    return {
        "source_file": source_file,
        "document_id": document_id,
        "chunk_index": chunk_index,
        "page_number": page_number,
        "text": text,
    }


def _deserialize_cached_context(cache_value: str) -> tuple[str, list[dict[str, Any]]]:
    try:
        payload = json.loads(cache_value)
    except json.JSONDecodeError:
        return cache_value, []

    if not isinstance(payload, dict):
        return cache_value, []

    answer = str(payload.get("answer", ""))
    raw_context = payload.get("context", [])
    context_items = [dict(item) for item in raw_context if isinstance(item, dict)]
    return answer, context_items


def _fallback_answer(query_text: str, context_chunks: list[str]) -> str:
    if context_chunks:
        context_preview = "\n\n".join(context_chunks[:2])
        return (
            "Gemini is temporarily unavailable, so this response is based on the nearest "
            "indexed document chunks only.\n\n"
            f"Question: {query_text}\n\n"
            f"Relevant context:\n{context_preview}"
        )

    return (
        "Gemini is temporarily unavailable and no relevant document context was found for "
        f"the question: {query_text}."
    )


def _should_retry_llm_error(error: Exception) -> bool:
    message = str(error).lower()
    non_retryable_markers = (
        "resource_exhausted",
        "quota exceeded",
        "not_found",
        "permission_denied",
        "invalid_argument",
        "api key",
    )
    if any(marker in message for marker in non_retryable_markers):
        return False

    retryable_markers = (
        "timeout",
        "timed out",
        "deadline exceeded",
        "internal",
        "unavailable",
        "service unavailable",
        "connection reset",
        "temporarily unavailable",
        "429",
        "500",
        "502",
        "503",
        "504",
    )
    return any(marker in message for marker in retryable_markers)


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception(_should_retry_llm_error),
    reraise=True,
)
def _invoke_llm(prompt: str) -> Any:
    return get_llm().invoke(prompt)


def ingest_document(
    file_path: str,
    workspace_id: str,
    document_id: str,
    source_file: str,
) -> dict[str, Any]:
    absolute_path = str(Path(file_path).resolve())
    with workspace_lock(workspace_id):
        document_sections = _load_document_sections(absolute_path)
        chunk_records: list[dict[str, Any]] = []
        for section in document_sections:
            section_text = str(section.get("text", "")).strip()
            if not section_text:
                continue
            for chunk in text_splitter.split_text(section_text):
                chunk_records.append(
                    {
                        "text": chunk,
                        "page_number": section.get("page_number"),
                    }
                )

        if not chunk_records:
            raise ValueError(f"No extractable text found in document: {source_file}")

        chunks = [record["text"] for record in chunk_records]
        vectors = get_embeddings().embed_documents(chunks)
        if len(vectors) != len(chunks) or not vectors:
            raise ValueError("Embedding generation returned an invalid vector batch.")

        _ensure_collection(vector_size=len(vectors[0]))
        points = [
            models.PointStruct(
                id=str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"docuquery:{workspace_id}:{document_id}:{chunk_index}",
                    )
                ),
                vector=vector,
                payload={
                    "text": record["text"],
                    "workspace_id": workspace_id,
                    "document_id": document_id,
                    "source_file": source_file,
                    "chunk_index": chunk_index,
                    "page_number": record["page_number"],
                },
            )
            for chunk_index, (record, vector) in enumerate(zip(chunk_records, vectors))
        ]
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )
        corpus_version = advance_corpus_version(workspace_id)

    return {
        "status": "ingested",
        "document_id": document_id,
        "source_file": source_file,
        "chunks_indexed": len(points),
        "corpus_version": corpus_version,
    }


def ask_question(query_text: str, workspace_id: str) -> dict[str, Any]:
    corpus_version = get_corpus_version(workspace_id)
    key = _cache_key(query_text, workspace_id, corpus_version)
    cached_payload = redis_client.get(key)
    if cached_payload:
        cached_answer, cached_context = _deserialize_cached_context(cached_payload)
        return {
            "query": query_text,
            "answer": cached_answer,
            "cached": True,
            "context": cached_context,
        }

    try:
        query_vector = get_embeddings().embed_query(query_text)
    except Exception as exc:
        answer = _fallback_answer(query_text, [])
        redis_client.setex(
            key,
            FALLBACK_CACHE_TTL_SECONDS,
            json.dumps({"answer": answer, "context": []}, ensure_ascii=False),
        )
        return {
            "query": query_text,
            "answer": answer,
            "cached": False,
            "context": [],
        }

    search_response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=_workspace_filter(workspace_id),
        limit=TOP_K,
        with_payload=True,
    )
    results = getattr(search_response, "points", [])

    context_items = [
        serialized_chunk
        for result in results
        if (serialized_chunk := _serialize_context_chunk(result)) is not None
    ]
    context_chunks = [item["text"] for item in context_items]
    context = "\n\n".join(context_chunks) if context_chunks else "No relevant context found."

    prompt = answer_prompt.format(context=context, question=query_text)

    try:
        response = _invoke_llm(prompt)
        answer = _response_text(response.content).strip()
    except Exception as exc:
        answer = _fallback_answer(query_text, context_chunks)
        redis_client.setex(
            key,
            FALLBACK_CACHE_TTL_SECONDS,
            json.dumps({"answer": answer, "context": context_items}, ensure_ascii=False),
        )
        return {
            "query": query_text,
            "answer": answer,
            "cached": False,
            "context": context_items,
        }

    redis_client.setex(
        key,
        CACHE_TTL_SECONDS,
        json.dumps({"answer": answer, "context": context_items}, ensure_ascii=False),
    )
    return {
        "query": query_text,
        "answer": answer,
        "cached": False,
        "context": context_items,
    }
