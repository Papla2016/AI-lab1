"""BM25 retrieval over the exact chunks persisted in Chroma."""
import re
from rank_bm25 import BM25Okapi
from .schemas import Document, SearchResult

def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w-]+", text.lower(), flags=re.UNICODE)

class BM25Retriever:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.index = BM25Okapi([tokenize(d.text) for d in documents]) if documents else None

    def search(self, query: str, k: int) -> list[SearchResult]:
        if not self.index: return []
        scores = self.index.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [SearchResult(self.documents[i].text, self.documents[i].metadata, float(scores[i])) for i in order if scores[i] > 0]

