"""Conservative rule router: explicit conversational/meta requests skip retrieval."""
from enum import Enum

class Route(str, Enum):
    DOCUMENT_QUESTION = "DOCUMENT_QUESTION"
    GENERAL_OR_OUT_OF_SCOPE = "GENERAL_OR_OUT_OF_SCOPE"

def route(question: str) -> Route:
    normalized = question.strip().lower()
    general = {"привет", "здравствуй", "кто ты", "расскажи анекдот", "hello", "hi", "who are you"}
    return Route.GENERAL_OR_OUT_OF_SCOPE if not normalized or normalized.rstrip("!?.,") in general else Route.DOCUMENT_QUESTION

