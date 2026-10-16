"""Command-line interface for indexing, retrieval, and RAG."""
import argparse
import json
import logging
from .chunking import chunk_documents
from .config import Settings
from .embeddings import SentenceTransformerEmbeddings
from .llm import OpenAICompatibleLLM
from .loaders import load_documents
from .rag import RAGPipeline
from .reranker import Reranker
from .schemas import SearchResult
from .vector_store import VectorStore

def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Лабораторный RAG-конвейер")
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("index", help="переиндексировать документы из data/")
    for name, help_text in (("search", "выполнить retrieval без генерации"), ("ask", "получить RAG-ответ"),
                            ("baseline", "спросить LLM напрямую, без RAG")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("question")
        command.add_argument("--retrieval", choices=["dense", "hybrid"])
        command.add_argument("--top-k", type=int)
        command.add_argument("--debug", action="store_true")
        if name == "ask": command.add_argument("--prompt", choices=["zero_shot", "few_shot", "cot_structured"], default="zero_shot")
    return root

def components(settings: Settings) -> tuple[VectorStore, SentenceTransformerEmbeddings]:
    embeddings = SentenceTransformerEmbeddings(settings.embedding_model)
    return VectorStore(settings.storage_dir, settings.collection_name, embeddings), embeddings

def serialise_result(item: SearchResult) -> dict:
    return {"score": round(item.score, 6), "source": item.metadata["source"], "page": item.metadata.get("page"),
            "chunk_id": item.metadata["chunk_id"], "text": item.text}

def main() -> None:
    args, settings = parser().parse_args(), Settings()
    logging.basicConfig(level=logging.DEBUG if getattr(args, "debug", False) else logging.INFO, format="%(levelname)s: %(message)s")
    if args.command == "index":
        documents = load_documents(settings.data_dir)
        if not documents: raise SystemExit("В data/ нет читаемых PDF, Markdown или TXT документов.")
        chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
        store, _ = components(settings)
        store.replace(chunks)
        print(f"Проиндексировано документов: {len(documents)}, чанков: {len(chunks)}")
        return
    llm = OpenAICompatibleLLM(settings.llm_base_url, settings.llm_api_key, settings.llm_model)
    if args.command == "baseline":
        prompt = (settings.prompts_dir / "baseline.md").read_text(encoding="utf-8")
        print(llm.generate(prompt, args.question))
        return
    store, _ = components(settings)
    mode, k = args.retrieval or settings.retrieval_mode, args.top_k or settings.top_n
    pipeline = RAGPipeline(store, llm,
        settings.prompts_dir, k, settings.final_k, settings.min_dense_score, settings.rrf_k,
        Reranker(settings.rerank_model, settings.rerank_enabled))
    if args.command == "search":
        before, _ = pipeline.retrieve(args.question, mode)
        print(json.dumps([serialise_result(x) for x in before[:k]], ensure_ascii=False, indent=2))
        return
    answer, trace = pipeline.ask(args.question, mode, args.prompt, args.debug)
    if args.debug:
        printable = dict(trace)
        for key in ("before_rerank", "after_rerank"): printable[key] = [serialise_result(x) for x in printable.get(key, [])]
        print("=== DEBUG (секреты не включены) ===")
        print(json.dumps(printable, ensure_ascii=False, indent=2))
        print("=== END DEBUG ===")
    print(answer.model_dump_json(indent=2))

if __name__ == "__main__": main()
