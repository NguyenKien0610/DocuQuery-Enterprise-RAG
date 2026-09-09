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


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)

    @field_validator("content")
    @classmethod
    def non_empty_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("History content must not be empty")
        return value.strip()


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    use_cache: bool = True
    retrieval_only: bool = False
    history: list[ConversationMessage] = Field(default_factory=list, max_length=6)

    @field_validator("history")
    @classmethod
    def completed_turns(cls, value: list[ConversationMessage]) -> list[ConversationMessage]:
        if len(value) % 2 or any(message.role != ("user" if i % 2 == 0 else "assistant") for i, message in enumerate(value)):
            raise ValueError("History must contain up to three completed user/assistant turns")
        return value

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
