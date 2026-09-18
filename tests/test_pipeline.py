from pathlib import Path
import pytest
from src.chunking import chunk_documents
from src.hybrid_retriever import reciprocal_rank_fusion
from src.loaders import load_documents
from src.rag import RAGPipeline
from src.router import LLMRouter, Route
from src.schemas import Document, SearchResult

PROMPTS = Path(__file__).parents[1] / "prompts"
VALID = '{"answer":"Используйте tool install.","citations":[{"source":"guide.txt","chunk_id":"abc"}],"confidence":"high","insufficient_context":false}'

class FakeStore:
    result = SearchResult("Пакет ставится командой tool install.", {"source": "guide.txt", "chunk_id": "abc"}, .9)
    calls = 0
    def search(self, query: str, k: int):
        self.calls += 1
        return [self.result]
    def all(self): return [Document(self.result.text, self.result.metadata)]

class SequenceLLM:
    def __init__(self, *responses: str): self.responses, self.prompts = list(responses), []
    def generate(self, system: str, user: str) -> str:
        self.prompts.append((system, user))
        return self.responses.pop(0)

class FixedRouter:
    def __init__(self, result: Route = Route.DOCUMENT_QUESTION): self.result = result
    def classify(self, question: str) -> Route: return self.result

def pipeline(llm, router=None):
    return RAGPipeline(FakeStore(), llm, PROMPTS, min_score=.2, router=router or FixedRouter())

def test_chunk_ids_are_stable_unique_and_size_is_bounded():
    docs = [Document("Первый абзац.\n\n" + "длинный текст " * 30, {"source": "a.txt"})]
    first, second = chunk_documents(docs, 80, 15), chunk_documents(docs, 80, 15)
    assert [x.metadata["chunk_id"] for x in first] == [x.metadata["chunk_id"] for x in second]
    assert len({x.metadata["chunk_id"] for x in first}) == len(first)
    assert all(len(x.text) <= 80 for x in first)

def test_nested_files_have_distinct_relative_sources(tmp_path):
    (tmp_path / "one").mkdir(); (tmp_path / "two").mkdir()
    (tmp_path / "one/guide.txt").write_text("one", encoding="utf-8")
    (tmp_path / "two/guide.txt").write_text("two", encoding="utf-8")
    assert {d.metadata["source"] for d in load_documents(tmp_path)} == {"one/guide.txt", "two/guide.txt"}

def test_rrf_merges_same_chunk():
    item = FakeStore.result
    assert reciprocal_rank_fusion([[item], [item]], 2)[0].score == 2 / 61

def test_zero_shot_and_cot_structured_are_different():
    zero_llm = SequenceLLM(VALID)
    pipeline(zero_llm).ask("Как установить пакет?", "dense", "zero_shot")
    cot_llm = SequenceLLM('{"relevant_chunk_ids":["abc"],"reason":"совпадает"}', VALID)
    _, trace = pipeline(cot_llm).ask("Как установить пакет?", "dense", "cot_structured")
    assert len(zero_llm.prompts) == 1 and len(cot_llm.prompts) == 2
    assert "Предварительно выбранные" in trace["final_prompt"]
    assert zero_llm.prompts[0][1] != cot_llm.prompts[-1][1]

@pytest.mark.parametrize("invalid", [
    '{"answer":"Есть ответ.","citations":[],"confidence":"high","insufficient_context":false}',
    '{"answer":"Нет ответа.","citations":[{"source":"guide.txt","chunk_id":"abc"}],"confidence":"low","insufficient_context":true}',
])
def test_invalid_citation_state_triggers_single_retry(invalid):
    llm = SequenceLLM(invalid, VALID)
    answer, _ = pipeline(llm).ask("Как установить пакет?", "dense", "zero_shot")
    assert len(llm.prompts) == 2 and answer.citations[0].chunk_id == "abc"

def test_valid_citation_passes_without_retry():
    llm = SequenceLLM(VALID)
    answer, _ = pipeline(llm).ask("Как установить пакет?", "dense", "zero_shot")
    assert len(llm.prompts) == 1 and not answer.insufficient_context

def test_out_of_scope_llm_router_skips_retrieval():
    llm = SequenceLLM('{"route":"GENERAL_OR_OUT_OF_SCOPE"}')
    store = FakeStore(); store.calls = 0
    rag = RAGPipeline(store, llm, PROMPTS, router=LLMRouter(llm, PROMPTS))
    answer, trace = rag.ask("Какова температура поверхности Венеры?", "dense", "zero_shot")
    assert answer.insufficient_context and store.calls == 0
    assert trace["router"] == "GENERAL_OR_OUT_OF_SCOPE"

