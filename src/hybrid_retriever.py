"""Reciprocal Rank Fusion for dense and sparse ranked lists."""
from .schemas import SearchResult

def reciprocal_rank_fusion(lists: list[list[SearchResult]], k: int, rrf_k: int = 60) -> list[SearchResult]:
    fused: dict[str, SearchResult] = {}
    for results in lists:
        for rank, item in enumerate(results, start=1):
            key = item.metadata["chunk_id"]
            if key not in fused: fused[key] = SearchResult(item.text, item.metadata, 0.0)
            fused[key].score += 1.0 / (rrf_k + rank)
    return sorted(fused.values(), key=lambda item: item.score, reverse=True)[:k]

