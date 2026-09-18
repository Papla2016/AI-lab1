from pathlib import Path
from src.chunking import chunk_documents
from src.hybrid_retriever import reciprocal_rank_fusion
from src.rag import RAGPipeline
from src.schemas import Document, SearchResult

class FakeStore:
    result = SearchResult("Пакет ставится командой tool install.", {"source": "guide.txt", "chunk_id": "abc"}, .9)
    def search(self, query: str, k: int): return [self.result]
    def all(self): return [Document(self.result.text, self.result.metadata)]

class FakeLLM:
    calls = 0
    def generate(self, system: str, user: str) -> str:
        self.calls += 1
        if self.calls == 1: return '{"bad": true}'
        return '{"answer":"Используйте tool install.","citations":[{"source":"guide.txt","chunk_id":"abc"}],"confidence":"high","insufficient_context":false}'

def test_chunk_ids_are_stable_and_unique():
    docs = [Document("Первый абзац.\n\nВторой длинный абзац.", {"source": "a.txt"})]
    first = chunk_documents(docs, 20, 5)
    second = chunk_documents(docs, 20, 5)
    assert [x.metadata["chunk_id"] for x in first] == [x.metadata["chunk_id"] for x in second]
    assert len({x.metadata["chunk_id"] for x in first}) == len(first)

def test_rrf_merges_same_chunk():
    item = FakeStore.result
    assert reciprocal_rank_fusion([[item], [item]], 2)[0].score == 2 / 61

def test_structured_retry_and_citation_validation():
    prompts = Path(__file__).parents[1] / "prompts"
    llm = FakeLLM()
    answer, trace = RAGPipeline(FakeStore(), llm, prompts, min_score=.2).ask("Как установить пакет?", "dense", "zero_shot")
    assert llm.calls == 2
    assert answer.citations[0].chunk_id == "abc"
    assert trace["after_rerank"][0].metadata["source"] == "guide.txt"

def test_router_skips_general_request():
    prompts = Path(__file__).parents[1] / "prompts"
    answer, trace = RAGPipeline(FakeStore(), FakeLLM(), prompts).ask("Привет!", "dense", "zero_shot")
    assert answer.insufficient_context
    assert trace["router"] == "GENERAL_OR_OUT_OF_SCOPE"

