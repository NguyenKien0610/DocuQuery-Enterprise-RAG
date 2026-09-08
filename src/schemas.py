from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class UploadResponse(BaseModel):
    task_id: str
    document_id: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    use_cache: bool = True
    retrieval_only: bool = False

    @field_validator("query")
    @classmethod
    def strip_non_empty_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Query must not be empty.")
        return normalized


class ContextChunk(BaseModel):
    source_file: str
    document_id: str
    chunk_index: int
    page_number: int | None = None
    text: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    cached: bool
    status: Literal["generated", "retrieved", "degraded", "insufficient_context"] = (
        "generated"
    )
    error_code: str | None = None
    context: list[ContextChunk] = Field(default_factory=list)
