"""Shared data and API schemas."""
from dataclasses import dataclass, field
from typing import Any, Literal
from pydantic import BaseModel, Field

@dataclass(slots=True)
class Document:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class SearchResult:
    text: str
    metadata: dict[str, Any]
    score: float

class Citation(BaseModel):
    source: str
    chunk_id: str

class RAGAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"]
    insufficient_context: bool

INSUFFICIENT = RAGAnswer(
    answer="В предоставленных документах недостаточно информации для ответа на этот вопрос.",
    citations=[], confidence="low", insufficient_context=True,
)

