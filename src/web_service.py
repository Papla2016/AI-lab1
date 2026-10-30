"""Browser-facing orchestration for the existing RAG pipeline.

No new retrieval or generation algorithm lives here: the CLI and UI share
the same document loader, Chroma store, prompts, and RAGPipeline.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .chunking import chunk_documents
from .config import Settings
from .embeddings import SentenceTransformerEmbeddings
from .llm import LLM, OpenAICompatibleLLM
from .loaders import load_documents
from .rag import RAGPipeline
from .reranker import Reranker
from .router import Route
from .schemas import RAGAnswer, SearchResult
from .vector_store import VectorStore

ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@dataclass(frozen=True)
class IndexSummary:
    files: int
    document_parts: int
    chunks: int


class DocumentRouter:
    """Explicitly run RAG instead of the optional LLM intent classifier."""

    def classify(self, question: str) -> Route:
        return Route.DOCUMENT_QUESTION


class RAGLabService:
    def __init__(
        self,
        settings: Settings,
        store_factory: Callable[[Settings], VectorStore] | None = None,
        llm_factory: Callable[[Settings], LLM] | None = None,
        reranker_factory: Callable[[str, bool], Reranker] | None = None,
    ) -> None:
        self.settings = settings
        self._store_factory = store_factory or self._default_store
        self._llm_factory = llm_factory or self._default_llm
        self._reranker_factory = reranker_factory or Reranker
        self._store: VectorStore | None = None
        self._llm: LLM | None = None
        self._rerankers: dict[bool, Reranker] = {}

    @staticmethod
    def _default_store(settings: Settings) -> VectorStore:
        embedding = SentenceTransformerEmbeddings(settings.embedding_model)
        return VectorStore(settings.storage_dir, settings.collection_name, embedding)

    @staticmethod
    def _default_llm(settings: Settings) -> LLM:
        return OpenAICompatibleLLM(settings.llm_base_url, settings.llm_api_key, settings.llm_model)

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = self._store_factory(self.settings)
        return self._store

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = self._llm_factory(self.settings)
        return self._llm

    def files(self) -> list[str]:
        folder = self.settings.data_dir
        if not folder.exists():
            return []
        return sorted(
            path.relative_to(folder).as_posix()
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
        )

    def save_document(self, filename: str, content: bytes) -> str:
        name = filename.strip()
        if (
            not name or name in {".", ".."} or "/" in name or "\\" in name
            or "\x00" in name or Path(name).suffix.lower() not in ALLOWED_EXTENSIONS
        ):
            raise ValueError("Допускаются только имена PDF, Markdown или TXT без пути.")
        if not content or len(content) > MAX_UPLOAD_BYTES:
            raise ValueError("Файл должен быть непустым и не больше 25 МиБ.")
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also blocks overwriting a pre-existing symlink.
        try:
            with (self.settings.data_dir / name).open("xb") as target:
                target.write(content)
        except FileExistsError:
            raise ValueError(f"Файл {name} уже существует: переименуйте его.") from None
        return name

    def index_documents(self) -> IndexSummary:
        documents = load_documents(self.settings.data_dir)
        chunks = chunk_documents(documents, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            raise ValueError("Нет читаемых документов. Прежний индекс не изменён.")
        self.store.replace(chunks)
        return IndexSummary(
            files=len({doc.metadata["source"] for doc in documents}),
            document_parts=len(documents),
            chunks=len(chunks),
        )

    def indexed_chunks(self) -> int:
        return self.store.collection.count()

    @staticmethod
    def _validate(question: str, mode: str, strategy: str, top_n: int, final_k: int) -> None:
        if not question.strip():
            raise ValueError("Введите вопрос.")
        if mode not in {"dense", "hybrid"}:
            raise ValueError("Неизвестный режим поиска.")
        if strategy not in {"zero_shot", "few_shot", "cot_structured"}:
            raise ValueError("Неизвестная стратегия промпта.")
        if not 1 <= final_k <= top_n <= 100:
            raise ValueError("Требуется 1 <= final_k <= top_n <= 100.")

    def _pipeline(self, top_n: int, final_k: int, rerank: bool, routing: bool) -> RAGPipeline:
        if rerank not in self._rerankers:
            self._rerankers[rerank] = self._reranker_factory(self.settings.rerank_model, rerank)
        return RAGPipeline(
            self.store, self.llm, self.settings.prompts_dir,
            top_n=top_n, final_k=final_k,
            min_score=self.settings.min_dense_score,
            rrf_k=self.settings.rrf_k,
            reranker=self._rerankers[rerank],
            router=None if routing else DocumentRouter(),
        )

    def search(
        self, question: str, mode: str, top_n: int, final_k: int, rerank: bool
    ) -> tuple[list[SearchResult], list[SearchResult]]:
        self._validate(question, mode, "zero_shot", top_n, final_k)
        # Search should work when Ollama is offline. This LLM is not called here.
        pipeline = self._pipeline(top_n, final_k, rerank, routing=False)
        return pipeline.retrieve(question, mode)

    def ask(
        self, question: str, mode: str, strategy: str,
        top_n: int, final_k: int, rerank: bool, routing: bool = True,
    ) -> tuple[RAGAnswer, dict]:
        self._validate(question, mode, strategy, top_n, final_k)
        return self._pipeline(top_n, final_k, rerank, routing).ask(
            question, mode, strategy, debug=True,
        )

    def baseline(self, question: str) -> str:
        if not question.strip():
            raise ValueError("Введите вопрос.")
        system = (self.settings.prompts_dir / "baseline.md").read_text(encoding="utf-8")
        return self.llm.generate(system, question)
