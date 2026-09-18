"""Dense retriever adapter."""
from .schemas import SearchResult
from .vector_store import VectorStore

class DenseRetriever:
    def __init__(self, store: VectorStore) -> None: self.store = store
    def search(self, query: str, k: int) -> list[SearchResult]: return self.store.search(query, k)

