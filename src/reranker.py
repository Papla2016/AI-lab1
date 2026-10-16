"""Optional local cross-encoder reranker preserving complete results metadata."""
from .schemas import SearchResult

class Reranker:
    def __init__(self, model_name: str, enabled: bool) -> None:
        self.model = None
        if enabled:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[SearchResult], k: int) -> list[SearchResult]:
        if not self.model: return results[:k]
        scores = self.model.predict([(query, item.text) for item in results])
        ranked = [SearchResult(item.text, item.metadata, float(score)) for item, score in zip(results, scores)]
        return sorted(ranked, key=lambda item: item.score, reverse=True)[:k]

