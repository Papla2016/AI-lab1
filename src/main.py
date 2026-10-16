"""Command-line interface for indexing, retrieval, and RAG."""
import argparse
import json
import logging
from .backend import ChatService, default_store
from .config import Settings
from .embeddings import SentenceTransformerEmbeddings
from .llm import LLMError
from .rag import serialise_result
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
    store = default_store(settings)
    return store, store.embeddings

def main() -> None:
    args, settings = parser().parse_args(), Settings()
    logging.basicConfig(level=logging.DEBUG if getattr(args, "debug", False) else logging.INFO, format="%(levelname)s: %(message)s")
    # Avoid request-level logging, which can contain Authorization headers and
    # credentials embedded in URLs, even when application diagnostics are on.
    for logger_name in ("openai", "httpx", "httpcore"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    try:
        _run(args, settings, ChatService(settings))
    except (LLMError, ValueError) as exc:
        raise SystemExit(str(exc)) from None


def _run(args: argparse.Namespace, settings: Settings, service: ChatService) -> None:
    if args.command == "index":
        report = service.index_documents()
        for error in report.errors:
            logging.warning(error)
        if not report.chunk_count:
            raise SystemExit("В data/ нет читаемых PDF, Markdown или TXT документов.")
        print(f"Проиндексировано документов: {report.document_count}, чанков: {report.chunk_count}")
        return
    if args.command == "baseline":
        prompt = (settings.prompts_dir / "baseline.md").read_text(encoding="utf-8")
        print(service.respond(args.question, mode="chat", instructions=prompt).text)
        return
    mode = args.retrieval or settings.retrieval_mode
    k = settings.top_n if args.top_k is None else args.top_k
    if args.command == "search":
        before, _ = service.search(args.question, retrieval=mode, top_k=k, rerank=settings.rerank_enabled)
        print(json.dumps([serialise_result(x) for x in before[:k]], ensure_ascii=False, indent=2))
        return
    response = service.respond(args.question, mode="rag", retrieval=mode, strategy=args.prompt,
                               top_k=k, rerank=settings.rerank_enabled, debug=args.debug, classify_rag=True)
    if args.debug:
        print("=== DEBUG (секреты не включены) ===")
        print(json.dumps(response.trace, ensure_ascii=False, indent=2))
        print("=== END DEBUG ===")
    print(response.answer.model_dump_json(indent=2))

if __name__ == "__main__": main()
