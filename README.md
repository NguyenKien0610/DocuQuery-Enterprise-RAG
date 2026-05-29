# DocuQuery Enterprise RAG

DocuQuery Enterprise RAG is a document question-answering system built around FastAPI, Celery, Redis, Qdrant, and Streamlit.

The current architecture uses hybrid retrieval:
- Local embeddings with `sentence-transformers` through `langchain-huggingface`
- Gemini chat generation through `langchain-google-genai`
- Redis semantic cache for repeated questions
- Qdrant as the persistent vector database
- Celery for asynchronous PDF ingestion

## Architecture

### Ingestion flow
1. A client uploads a PDF to `POST /api/v1/documents/upload`.
2. FastAPI stores the file in `uploads/` and sends a Celery task.
3. The Celery worker extracts text, chunks it, creates local embeddings, and upserts vectors into Qdrant.

### Query flow
1. A client sends a question to `POST /api/v1/query`.
2. The backend checks Redis for a cached answer.
3. On cache miss, it embeds the query locally, retrieves the nearest chunks from Qdrant, formats a prompt, and sends the prompt to Gemini.
4. The final answer is cached in Redis and returned to the client.

## Tech Stack

- Backend API: FastAPI, Uvicorn
- Async worker: Celery
- Broker and cache: Redis
- Vector database: Qdrant
- Retrieval and prompting: LangChain
- Local embeddings: `all-MiniLM-L6-v2`
- LLM: Gemini via `langchain-google-genai`
- Frontend: Streamlit

## Project Structure

```text
DocuQuery-Enterprise-RAG/
├── frontend/
│   └── app.py
├── src/
│   ├── main.py
│   ├── rag_engine.py
│   ├── schemas.py
│   └── worker.py
├── uploads/
├── .env.example
├── docker-compose.yml
├── docs.md
├── README.md
└── requirements.txt
```

## Requirements

- Python 3.10+
- Docker Desktop or Docker Engine
- A valid `GOOGLE_API_KEY`

## Environment Setup

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_key_here
```

Optional variables:

```env
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=docuquery_hybrid_v1
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
UPLOAD_DIR=uploads
RAG_TOP_K=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2
GEMINI_CHAT_MODEL=models/gemini-2.5-flash
```

## Installation

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Start infrastructure services:

```bash
docker compose up -d
```

## Run the Backend

Start the FastAPI server:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

Start the Celery worker in another terminal:

```bash
celery -A src.worker.celery_app worker --loglevel=info --pool=solo
```

`--pool=solo` is the safest default on Windows.

## Run the Frontend

Start Streamlit:

```bash
streamlit run frontend/app.py
```

Frontend URL:

```text
http://localhost:8501
```

Backend URL:

```text
http://localhost:8000
```

## API Endpoints

### Upload document

```http
POST /api/v1/documents/upload
Content-Type: multipart/form-data
```

Response:

```json
{
  "task_id": "uuid"
}
```

### Check ingestion status

```http
GET /api/v1/documents/status/{task_id}
```

Response:

```json
{
  "task_id": "uuid",
  "status": "SUCCESS",
  "result": {
    "status": "ingested",
    "source": "path-to-file",
    "chunks_indexed": 42
  },
  "error": null
}
```

### Query documents

```http
POST /api/v1/query
Content-Type: application/json
```

Request:

```json
{
  "query": "Summarize this document"
}
```

Response:

```json
{
  "query": "Summarize this document",
  "answer": "...",
  "cached": false
}
```

## Frontend Features

- PDF upload from the sidebar
- Chat-style interface with `st.chat_message`
- Persistent session chat history
- Visual `Cached` marker for Redis cache hits
- Basic backend connection error handling

## Notes

- Upload and indexing do not consume Gemini quota because embeddings are local.
- Detailed questions can take longer because Gemini is only used during answer generation.
- Qdrant uses a separate collection name, `docuquery_hybrid_v1`, to avoid vector dimension conflicts with older embedding strategies.

## Troubleshooting

### `FAILURE` in task status after upload

Check:
- Redis is running on port `6379`
- Qdrant is running on port `6333`
- The Celery worker is running

### `404 NOT_FOUND` from Gemini

Set `GEMINI_CHAT_MODEL` in `.env` to a model available in your Gemini project, for example:

```env
GEMINI_CHAT_MODEL=models/gemini-2.5-flash
```

### Streamlit times out on long answers

The frontend already waits up to `120` seconds. If needed, increase the timeout in [frontend/app.py](./frontend/app.py).

## License

This project is licensed under the terms of the [LICENSE](./LICENSE) file.
