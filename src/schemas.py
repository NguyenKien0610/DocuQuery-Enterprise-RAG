from typing import Any

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    task_id: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None


class QueryRequest(BaseModel):
    query: str


class ContextChunk(BaseModel):
    source_file: str
    source_path: str
    chunk_index: int
    page_number: int | None = None
    text: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    cached: bool
    context: list[ContextChunk] = Field(default_factory=list)
