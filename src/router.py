"""Small structured LLM classifier used before document retrieval."""
import json
from enum import Enum
from pathlib import Path
from typing import Protocol
from pydantic import BaseModel, ValidationError
from .llm import LLM

class Route(str, Enum):
    DOCUMENT_QUESTION = "DOCUMENT_QUESTION"
    GENERAL_OR_OUT_OF_SCOPE = "GENERAL_OR_OUT_OF_SCOPE"

class RouteResult(BaseModel):
    route: Route

class Router(Protocol):
    def classify(self, question: str) -> Route: ...

class LLMRouter:
    """Classify intent with a tiny JSON response; fail closed on malformed output."""
    def __init__(self, llm: LLM, prompts_dir: Path) -> None:
        self.llm = llm
        self.prompt = (prompts_dir / "router.md").read_text(encoding="utf-8")

    def classify(self, question: str) -> Route:
        if not question.strip():
            return Route.GENERAL_OR_OUT_OF_SCOPE
        raw = self.llm.generate(self.prompt, f"QUESTION:\n{question}")
        try:
            return RouteResult.model_validate_json(raw).route
        except (ValidationError, json.JSONDecodeError):
            # Do not retrieve or expose documents when routing is uncertain.
            return Route.GENERAL_OR_OUT_OF_SCOPE

