"""Real local Chroma lifecycle tests with tiny deterministic embeddings."""
from pathlib import Path

from src.backend import ChatService
from src.config import Settings
from src.schemas import Document
from src.vector_store import VectorStore


class TinyEmbeddings:
    def encode_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def encode_query(self, text):
        return [1.0, 0.0]


class UnusedLLM:
    def generate(self, *args):
        raise AssertionError("Indexing and retrieval must not call the LLM")


def test_existing_store_reads_new_collection_after_another_store_reindexes(tmp_path, monkeypatch):
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    first = VectorStore(tmp_path, "cross_session_read", TinyEmbeddings())
    first.replace([Document("Old text", {"source": "old.txt", "chunk_id": "old"})])
    second = VectorStore(tmp_path, "cross_session_read", TinyEmbeddings())
    assert second.search("question", 1)[0].metadata["source"] == "old.txt"

    first.replace([Document("New text", {"source": "new.txt", "chunk_id": "new"})])

    assert second.all()[0].text == "New text"
    assert second.search("question", 1)[0].metadata["source"] == "new.txt"


def test_cached_chat_service_store_observes_another_sessions_index(tmp_path, monkeypatch):
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    settings = Settings(
        _env_file=None, data_dir=tmp_path / "data", storage_dir=tmp_path / "storage",
        collection_name="cross_session_service", rerank_enabled=False,
        prompts_dir=Path(__file__).resolve().parents[1] / "prompts",
    )

    def store_factory(cfg):
        return VectorStore(cfg.storage_dir, cfg.collection_name, TinyEmbeddings())

    first = ChatService(settings, llm=UnusedLLM(), store_factory=store_factory)
    second = ChatService(settings, llm=UnusedLLM(), store_factory=store_factory)
    first.store.replace([Document("Old text", {"source": "old.txt", "chunk_id": "old"})])
    assert second.search("question", retrieval="dense")[0][0].metadata["source"] == "old.txt"
    cached_store = second.store
    settings.data_dir.mkdir()
    (settings.data_dir / "fresh.txt").write_text("Новая документация после индексации.", encoding="utf-8")

    report = first.index_documents()

    assert report.document_count == report.chunk_count == 1
    assert second.store is cached_store
    assert second.search("question", retrieval="dense")[0][0].metadata["source"] == "fresh.txt"
