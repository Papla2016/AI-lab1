"""Run reusable RAG evaluation and preserve the original Markdown CLI."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from src.backend import ChatService
from src.config import ROOT, Settings
from .runner import RunManager, load_questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Выполнить матрицу RAG evaluation и сохранить результаты.")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/results.md")
    parser.add_argument("--questions", type=Path, default=ROOT / "evaluation/questions.json")
    parser.add_argument("--retrieval", nargs="+", choices=["dense", "hybrid"], default=["dense", "hybrid"])
    parser.add_argument("--prompt", nargs="+", choices=["zero_shot", "few_shot", "cot_structured"],
                        default=["zero_shot", "few_shot", "cot_structured"])
    parser.add_argument("--rerank", action="store_true", help="Включить reranking")
    parser.add_argument("--compare-rerank", action="store_true", help="Сравнить сценарии с reranking и без него")
    parser.add_argument("--no-baseline", action="store_true", help="Не запускать дополнительный No-RAG baseline")
    args, settings = parser.parse_args(), Settings()
    manager = RunManager()
    run_id = manager.start_evaluation(ChatService(settings), load_questions(args.questions),
                                      retrieval_modes=args.retrieval, strategies=args.prompt,
                                      rerank_options=(False, True) if args.compare_rerank else (args.rerank or settings.rerank_enabled,),
                                      top_k=settings.top_n, include_baseline=not args.no_baseline)
    previous = -1
    interrupted = False
    try:
        while True:
            snapshot = manager.snapshot(run_id)
            if snapshot["completed"] != previous:
                print(f"Выполнено {snapshot['completed']} / {snapshot['total']}", flush=True)
                previous = snapshot["completed"]
            if snapshot["status"] not in {"Running", "Cancelling"}:
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        interrupted = True
        manager.cancel(run_id)
        print("Остановка после текущего запроса. Уже полученные результаты сохранены.", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(manager.export(run_id, "markdown"), encoding="utf-8")
    print(args.output)
    result = manager.snapshot(run_id)
    if interrupted or result["status"] in {"Cancelled", "Cancelling"}:
        raise SystemExit(130)
    if result["status"] == "Error" or result["metrics"].get("technical_errors", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
