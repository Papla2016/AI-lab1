"""Reusable application services shared by Streamlit and the command line.

The LLM and document store are lazy: a normal conversation neither loads an
embedding model nor needs an index. RAG history is deliberately excluded from
retrieval and generation, so previous model answers cannot become evidence.
"""
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable

from .chunking import chunk_documents
from .config import Settings
from .embeddings import SentenceTransformerEmbeddings
from .llm import LLM, OpenAICompatibleLLM
from .loaders import load_documents
from .rag import RAGPipeline, RAGValidationError
from .reranker import Reranker
from .router import LLMRouter, Route
from .schemas import INSUFFICIENT, RAGAnswer, SearchResult
from .vector_store import VectorStore

DEFAULT_INSTRUCTIONS = "Отвечайте на русском языке, ясно и по существу."


@dataclass(slots=True)
class ChatResponse:
    text: str
    mode: str
    answer: RAGAnswer | None = None
    trace: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0


@dataclass(slots=True)
class IndexReport:
    document_count: int
    chunk_count: int
    errors: list[str] = field(default_factory=list)


def default_store(settings: Settings) -> VectorStore:
    embeddings = SentenceTransformerEmbeddings(settings.embedding_model)
    return VectorStore(settings.storage_dir, settings.collection_name, embeddings)


class ChatService:
    def __init__(self, settings: Settings, llm: LLM | None = None,
                 store_factory: Callable[[Settings], VectorStore] | None = None) -> None:
        self.settings = settings
        self._llm = llm
        self._store_factory = store_factory or default_store
        self._store: VectorStore | None = None
        self._rerankers: dict[bool, Reranker] = {}

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = OpenAICompatibleLLM(self.settings.llm_base_url, self.settings.llm_api_key, self.settings.llm_model)
        return self._llm

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = self._store_factory(self.settings)
        return self._store

    def _pipeline(self, top_k: int, rerank: bool) -> RAGPipeline:
        if rerank not in self._rerankers:
            self._rerankers[rerank] = Reranker(self.settings.rerank_model, rerank)
        return RAGPipeline(self.store, self.llm, self.settings.prompts_dir,
                           top_n=top_k, final_k=min(top_k, self.settings.final_k),
                           min_score=self.settings.min_dense_score, rrf_k=self.settings.rrf_k,
                           reranker=self._rerankers[rerank])

    @staticmethod
    def _check_query(question: str, retrieval: str, strategy: str, top_k: int) -> None:
        if not question.strip():
            raise ValueError("Введите непустой вопрос.")
        if retrieval not in {"dense", "hybrid"}:
            raise ValueError("Retrieval должен быть dense или hybrid.")
        if strategy not in {"zero_shot", "few_shot", "cot_structured"}:
            raise ValueError("Неизвестная стратегия промпта.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k должен быть положительным целым числом.")

    def respond(self, question: str, mode: str = "rag", history: list[dict[str, str]] | None = None,
                instructions: str = "", retrieval: str = "hybrid", strategy: str = "zero_shot",
                top_k: int = 12, rerank: bool = False, debug: bool = False,
                classify_rag: bool = False) -> ChatResponse:
        started = perf_counter()
        self._check_query(question, retrieval, strategy, top_k)
        if mode not in {"rag", "chat", "auto"}:
            raise ValueError("Режим должен быть rag, chat или auto.")
        requested_mode = mode
        route = Route.DOCUMENT_QUESTION if mode == "rag" else Route.GENERAL_OR_OUT_OF_SCOPE
        router_seconds = 0.0
        if mode == "auto" or (mode == "rag" and classify_rag):
            router_started = perf_counter()
            route = LLMRouter(self.llm, self.settings.prompts_dir).classify(question)
            router_seconds = perf_counter() - router_started
            if mode == "auto":
                mode = "rag" if route is Route.DOCUMENT_QUESTION else "chat"
        trace: dict = {"question": question, "router": route.value, "requested_mode": requested_mode,
                       "generation_seconds": 0.0, "router_seconds": router_seconds}
        answer = None
        if mode == "rag":
            if route is Route.GENERAL_OR_OUT_OF_SCOPE:
                # CLI/evaluation preserve their original scope check without
                # loading models or falling back to an ungrounded answer.
                answer = INSUFFICIENT.model_copy()
                trace["retrieval_mode"] = retrieval
            else:
                try:
                    answer, rag_trace = self._pipeline(top_k, rerank).ask(
                        question, retrieval, strategy, debug, instructions=instructions, route=route)
                except RAGValidationError as exc:
                    exc.trace = {**trace, **exc.trace, "elapsed_seconds": perf_counter() - started}
                    raise
                trace.update(rag_trace)
            text = answer.answer
        else:
            # Only actual user/assistant messages are accepted. UI metadata and
            # arbitrary 'system' entries are never promoted into instructions.
            messages = []
            for item in history or []:
                role, content = item.get("role"), item.get("content")
                if role in {"user", "assistant"} and isinstance(content, str):
                    messages.append({"role": role, "content": content})
            messages.append({"role": "user", "content": question})
            system = "Вы — полезный собеседник. Отвечайте на любые вопросы пользователя обычным текстом или Markdown."
            if instructions.strip():
                system += "\n\nПользовательские инструкции:\n" + instructions
            generation_started = perf_counter()
            text = self.llm.chat(messages, system=system)
            trace["generation_seconds"] = perf_counter() - generation_started
            if debug:
                trace["final_prompt"] = {"system": system, "messages": messages}
        elapsed = perf_counter() - started
        trace["elapsed_seconds"] = elapsed
        return ChatResponse(text=text, mode=mode, answer=answer, trace=trace, elapsed_seconds=elapsed)

    def search(self, question: str, retrieval: str | None = None, top_k: int | None = None,
               rerank: bool | None = None) -> tuple[list[SearchResult], list[SearchResult]]:
        retrieval = retrieval or self.settings.retrieval_mode
        top_k = self.settings.top_n if top_k is None else top_k
        self._check_query(question, retrieval, "zero_shot", top_k)
        return self._pipeline(top_k, self.settings.rerank_enabled if rerank is None else rerank).retrieve(question, retrieval)

    def save_upload(self, filename: str, data: bytes) -> Path:
        """Save one local document; reject traversal and preserve existing files."""
        name = filename.strip()
        if not name or name in {".", ".."} or "/" in name or "\\" in name or "\x00" in name:
            raise ValueError("Имя файла должно содержать только название, без пути.")
        if Path(name).suffix.lower() not in {".pdf", ".txt", ".md"}:
            raise ValueError("Поддерживаются только PDF, TXT и Markdown (.md).")
        if not data:
            raise ValueError("Нельзя загрузить пустой файл.")
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        target = self.settings.data_dir / name
        try:
            # Exclusive creation also rejects existing symlinks and avoids
            # truncating files if a session uploads the same name again.
            with target.open("xb") as output:
                output.write(data)
        except FileExistsError:
            raise ValueError("Файл с таким именем уже существует. Переименуйте загружаемый файл.") from None
        return target

    def index_documents(self) -> IndexReport:
        errors: list[str] = []
        documents = load_documents(self.settings.data_dir, errors=errors)
        count = len({doc.metadata["source"] for doc in documents})
        chunks = chunk_documents(documents, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            errors.append("В data/ нет читаемых PDF, Markdown или TXT документов. Существующий индекс сохранён.")
            return IndexReport(count, 0, errors)
        self.store.replace(chunks)
        return IndexReport(count, len(chunks), errors)
