# Project: DocuQuery v2.0 - Enterprise RAG API with Async Processing & Semantic Cache

## 1. Objective
Upgrade the DocuQuery system into a highly scalable, distributed RAG (Retrieval-Augmented Generation) backend. The system handles large PDF uploads asynchronously using Celery and Redis. It uses Qdrant as a persistent Vector Database and implements Semantic Caching to reduce LLM API calls and latency.

## 2. Tech Stack
* **Framework:** FastAPI, Uvicorn
* **AI & RAG:** LangChain, Google Gemini API (langchain-google-genai for Embeddings & Chat)
* **Vector Database:** Qdrant (via Docker)
* **Async Task Queue:** Celery
* **Message Broker & Cache:** Redis (via Docker)
* **Testing:** Pytest

## 3. System Architecture & Flow
### Flow 1: Async Document Ingestion
1. Client calls `POST /api/v1/documents/upload` with a PDF file.
2. FastAPI saves the file temporarily and dispatches a Celery task `process_document_task`. API immediately returns a `task_id`.
3. Celery Worker picks up the task, extracts text, chunks it, generates Google Gemini embeddings (`GoogleGenerativeAIEmbeddings`), and upserts into Qdrant.

### Flow 2: Querying with Semantic Cache
1. Client calls `POST /api/v1/query` with a question.
2. FastAPI hashes/embeds the query and checks the Redis Semantic Cache.
3. **Cache Hit:** Returns the cached answer immediately.
4. **Cache Miss:** Retrieve context from Qdrant -> Call Gemini LLM (`ChatGoogleGenerativeAI`) -> Cache in Redis -> Return answer.

## 4. API Endpoints
* `POST /api/v1/documents/upload` (multipart) -> `{"task_id": "uuid"}`
* `GET /api/v1/documents/status/{task_id}` -> Task status
* `POST /api/v1/query` (JSON) -> `{"query": "...", "answer": "...", "cached": true/false}`

## 5. Directory Structure
/docuquery-v2
  ├── src/
  │   ├── main.py            # FastAPI application & endpoints
  │   ├── worker.py          # Celery app and background tasks (PDF processing)
  │   ├── rag_engine.py      # LangChain logic, Qdrant setup, and Semantic Cache implementation
  │   └── schemas.py         # Pydantic models
  ├── docker-compose.yml     # Orchestrates Redis and Qdrant
  ├── requirements.txt
  └── .env                   # Stores OPENAI_API_KEY