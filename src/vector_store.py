"""Persistent Chroma storage, including texts used by sparse retrieval."""
from pathlib import Path
from .embeddings import Embeddings
from .schemas import Document, SearchResult

class VectorStore:
    def __init__(self, path: Path, collection_name: str, embeddings: Embeddings) -> None:
        import chromadb
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        self.name, self.embeddings = collection_name, embeddings
        self.collection = self.client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    def replace(self, chunks: list[Document]) -> None:
        try:
            self.client.delete_collection(self.name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=self.name, metadata={"hnsw:space": "cosine"})
        for start in range(0, len(chunks), 64):
            batch = chunks[start:start + 64]
            self.collection.add(ids=[d.metadata["chunk_id"] for d in batch], documents=[d.text for d in batch],
                metadatas=[d.metadata for d in batch], embeddings=self.embeddings.encode_documents([d.text for d in batch]))

    def all(self) -> list[Document]:
        result = self.collection.get(include=["documents", "metadatas"])
        return [Document(text=t, metadata=m) for t, m in zip(result["documents"], result["metadatas"])]

    def search(self, query: str, k: int) -> list[SearchResult]:
        if self.collection.count() == 0:
            return []
        result = self.collection.query(query_embeddings=[self.embeddings.encode_query(query)], n_results=min(k, self.collection.count()), include=["documents", "metadatas", "distances"])
        return [SearchResult(t, m, max(0.0, 1.0 - float(d))) for t, m, d in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])]

