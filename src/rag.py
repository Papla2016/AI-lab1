"""Retrieval, grounded prompt construction, generation, and citation validation."""
import json
from pathlib import Path
from pydantic import ValidationError
from .bm25_retriever import BM25Retriever
from .hybrid_retriever import reciprocal_rank_fusion
from .llm import LLM
from .reranker import Reranker
from .router import Route, route
from .schemas import INSUFFICIENT, RAGAnswer, SearchResult
from .vector_store import VectorStore

class RAGPipeline:
    def __init__(self, store: VectorStore, llm: LLM, prompts_dir: Path, top_n: int = 12, final_k: int = 5,
                 min_score: float = .2, rrf_k: int = 60, reranker: Reranker | None = None) -> None:
        self.store, self.llm, self.prompts_dir = store, llm, prompts_dir
        self.top_n, self.final_k, self.min_score, self.rrf_k = top_n, final_k, min_score, rrf_k
        self.reranker = reranker or Reranker("", False)

    def retrieve(self, question: str, mode: str) -> tuple[list[SearchResult], list[SearchResult]]:
        dense = self.store.search(question, self.top_n)
        candidates = dense if mode == "dense" else reciprocal_rank_fusion([dense, BM25Retriever(self.store.all()).search(question, self.top_n)], self.top_n, self.rrf_k)
        return candidates, self.reranker.rerank(question, candidates, self.final_k)

    def _prompt(self, question: str, results: list[SearchResult], strategy: str) -> tuple[str, str, str]:
        context = "\n\n".join(f"[source={x.metadata['source']}; chunk_id={x.metadata['chunk_id']}; page={x.metadata.get('page', 'n/a')}]\n{x.text}" for x in results)
        system = (self.prompts_dir / "system.md").read_text()
        if strategy == "few_shot": system += "\n\n" + (self.prompts_dir / "few_shot.md").read_text()
        system += "\n\n" + (self.prompts_dir / "structured.md").read_text()
        user = (self.prompts_dir / "user.md").read_text().format(question=question, context=context)
        return system, user, context

    def ask(self, question: str, mode: str, strategy: str, debug: bool = False) -> tuple[RAGAnswer, dict]:
        decision = route(question)
        trace: dict = {"question": question, "router": decision.value, "retrieval_mode": mode}
        if decision is Route.GENERAL_OR_OUT_OF_SCOPE:
            return INSUFFICIENT.model_copy(), trace
        before, after = self.retrieve(question, mode)
        trace.update(before_rerank=before, after_rerank=after)
        # Dense score gates both modes: an RRF number is not a semantic similarity.
        dense_best = self.store.search(question, 1)
        if not after or not dense_best or dense_best[0].score < self.min_score:
            return INSUFFICIENT.model_copy(), trace
        system, user, context = self._prompt(question, after, strategy)
        trace.update(context=context, final_prompt=f"SYSTEM:\n{system}\n\nUSER:\n{user}")
        raw = self.llm.generate(system, user)
        for attempt in range(2):
            try:
                answer = RAGAnswer.model_validate_json(raw)
                allowed = {(x.metadata["source"], x.metadata["chunk_id"]) for x in after}
                if any((c.source, c.chunk_id) not in allowed for c in answer.citations):
                    raise ValueError("citation is not present in context")
                return answer, trace
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                if attempt == 0:
                    raw = self.llm.generate(system, user + f"\n\nИсправьте JSON. Ошибка проверки: {exc}")
        raise ValueError("LLM twice returned invalid structured output or citations")

