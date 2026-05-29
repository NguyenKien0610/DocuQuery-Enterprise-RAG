import hashlib
import os
import uuid
from pathlib import Path
from typing import Any

import pdfplumber
import redis
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "docuquery_hybrid_v1")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
CACHE_TTL_SECONDS = 3600
FALLBACK_CACHE_TTL_SECONDS = 300

TOP_K = int(os.getenv("RAG_TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "models/gemini-2.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=True,
)

embeddings = HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)
llm = ChatGoogleGenerativeAI(
    model=GEMINI_CHAT_MODEL,
    api_key=GOOGLE_API_KEY,
    retries=0,
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

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


def _extract_pdf_text(file_path: str) -> str:
    pages: list[str] = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
    except Exception:
        reader = PdfReader(file_path)
        for page in reader.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def _ensure_collection(vector_size: int) -> None:
    collections = qdrant_client.get_collections().collections
    names = {collection.name for collection in collections}
    if COLLECTION_NAME in names:
        return

    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )


def ensure_qdrant_collection() -> None:
    collections = qdrant_client.get_collections().collections
    names = {collection.name for collection in collections}
    if COLLECTION_NAME in names:
        return

    probe_vector = embeddings.embed_query("qdrant_collection_init")
    _ensure_collection(vector_size=len(probe_vector))


def _cache_key(query_text: str) -> str:
    normalized = query_text.strip().lower()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"rag:answer:{digest}"


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


def _fallback_answer(query_text: str, context_chunks: list[str], error: Exception) -> str:
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
        f"the question: {query_text}. Error: {error}"
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
    return llm.invoke(prompt)


def ingest_document(file_path: str) -> dict[str, Any]:
    absolute_path = str(Path(file_path).resolve())
    document_text = _extract_pdf_text(absolute_path)
    if not document_text:
        raise ValueError(f"No extractable text found in PDF: {absolute_path}")

    chunks = text_splitter.split_text(document_text)
    if not chunks:
        raise ValueError(f"Unable to create chunks from PDF: {absolute_path}")

    points_indexed = 0
    for idx, chunk in enumerate(chunks):
        vector_batch = embeddings.embed_documents([chunk])
        if not vector_batch:
            raise ValueError(f"Embedding generation returned no vector for chunk {idx}.")

        vector = vector_batch[0]
        if points_indexed == 0:
            _ensure_collection(vector_size=len(vector))

        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": chunk,
                "source": absolute_path,
                "chunk_index": idx,
            },
        )
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[point])
        points_indexed += 1

    return {
        "status": "ingested",
        "source": absolute_path,
        "chunks_indexed": points_indexed,
    }


def ask_question(query_text: str) -> dict[str, Any]:
    key = _cache_key(query_text)
    cached_answer = redis_client.get(key)
    if cached_answer:
        return {"query": query_text, "answer": cached_answer, "cached": True}

    try:
        query_vector = embeddings.embed_query(query_text)
    except Exception as exc:
        answer = _fallback_answer(query_text, [], exc)
        redis_client.setex(key, FALLBACK_CACHE_TTL_SECONDS, answer)
        return {"query": query_text, "answer": answer, "cached": False}

    search_response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K,
        with_payload=True,
    )
    results = getattr(search_response, "points", [])

    context_chunks = [
        str(result.payload.get("text", "")).strip()
        for result in results
        if result.payload and result.payload.get("text")
    ]
    context = "\n\n".join(context_chunks) if context_chunks else "No relevant context found."

    prompt = answer_prompt.format(context=context, question=query_text)

    try:
        response = _invoke_llm(prompt)
        answer = _response_text(response.content).strip()
    except Exception as exc:
        answer = _fallback_answer(query_text, context_chunks, exc)
        redis_client.setex(key, FALLBACK_CACHE_TTL_SECONDS, answer)
        return {"query": query_text, "answer": answer, "cached": False}

    redis_client.setex(key, CACHE_TTL_SECONDS, answer)
    return {"query": query_text, "answer": answer, "cached": False}
