"""Run the same question matrix and write raw, reproducible results to Markdown."""
import argparse
import json
from pathlib import Path
from src.config import ROOT, Settings
from src.embeddings import SentenceTransformerEmbeddings
from src.llm import OpenAICompatibleLLM
from src.rag import RAGPipeline
from src.reranker import Reranker
from src.vector_store import VectorStore

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/results.md")
    args, cfg = parser.parse_args(), Settings()
    embed = SentenceTransformerEmbeddings(cfg.embedding_model)
    store = VectorStore(cfg.storage_dir, cfg.collection_name, embed)
    pipeline = RAGPipeline(store, OpenAICompatibleLLM(cfg.llm_base_url, cfg.llm_api_key, cfg.llm_model), cfg.prompts_dir,
        cfg.top_n, cfg.final_k, cfg.min_dense_score, cfg.rrf_k, Reranker(cfg.rerank_model, cfg.rerank_enabled))
    questions = json.loads((ROOT / "evaluation/questions.json").read_text())
    rows = ["# Результаты evaluation", "", "Результаты получены реальным запуском; файл можно пересоздать.", ""]
    for q in questions:
        for mode in ("dense", "hybrid"):
            for strategy in ("zero_shot", "few_shot", "structured"):
                answer, trace = pipeline.ask(q["question"], mode, strategy)
                retrieved = trace.get("after_rerank", [])
                rows += [f"## Q{q['id']} · {mode} · {strategy}", "", f"**Question:** {q['question']}",
                    f"**Expected in documents:** `{q['expected_in_documents']}`",
                    f"**Retrieved:** `{[(x.metadata['source'], x.metadata['chunk_id']) for x in retrieved]}`",
                    "```json", answer.model_dump_json(indent=2), "```", ""]
    args.output.write_text("\n".join(rows), encoding="utf-8")
    print(args.output)

if __name__ == "__main__": main()

