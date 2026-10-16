"""Reusable evaluation jobs with live progress, cancellation, and durable reports.

Only the pytest adapter uses a subprocess. RAG scenarios call the same ChatService
as the CLI and Streamlit; their cancellation is cooperative between requests.
"""
from __future__ import annotations

import csv
import io
import itertools
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any
from uuid import uuid4
from datetime import datetime, timezone

from src.config import ROOT, Settings
from src.schemas import RAGAnswer

RUN_LOCK = threading.Lock()
_ACTIVE_IDS: set[str] = set()
_ACTIVE_IDS_LOCK = threading.RLock()
_RESULTS = ROOT / "evaluation" / "results"
_QUESTIONS = ROOT / "evaluation" / "questions.json"
_USER_QUESTIONS = ROOT / "evaluation" / "user_questions.json"
_USER_LOCK = threading.Lock()
_TERMINAL = {"Passed", "Failed", "Error", "Cancelled", "Skipped"}


def _safe(value: Any, secrets: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        return {str(k): "[скрыто]" if re.search(r"api.?key|password|secret|token|authorization", str(k), re.I)
                else _safe(v, secrets) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v, secrets) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            if len(secret) >= 4:
                value = value.replace(secret, "[скрыто]")
        value = re.sub(r"(?i)(bearer\s+)[\w.\-]+", r"\1[скрыто]", value)
        value = re.sub(r"(?i)((?:api[_-]?key|token|password|secret)\s*[=:]\s*)[^\s,;]+", r"\1[скрыто]", value)
        return re.sub(r"(https?://)[^/@\s]+:[^/@\s]+@", r"\1[скрыто]@", value)
    if hasattr(value, "model_dump"):
        return _safe(value.model_dump(), secrets)
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_questions(path: Path | str | None = None) -> list[dict]:
    rows = json.loads(Path(path or _QUESTIONS).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Набор вопросов должен быть JSON-массивом.")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError("У каждого сценария должен быть непустой вопрос.")
        if not isinstance(row.get("expected_in_documents"), bool):
            raise ValueError("expected_in_documents должен быть логическим значением.")
    return rows


def load_user_questions(path: Path | str | None = None) -> list[dict]:
    target = Path(path or _USER_QUESTIONS)
    return load_questions(target) if target.exists() else []


def save_user_question(question: str, expected_in_documents: bool, reference_answer: str = "",
                       expected_source: str = "", question_id: Any = None,
                       path: Path | str | None = None) -> dict:
    if not question.strip():
        raise ValueError("Введите текст вопроса.")
    if not isinstance(expected_in_documents, bool):
        raise ValueError("Укажите, ожидается ли ответ в документах.")
    target = Path(path or _USER_QUESTIONS)
    if target.resolve() == _QUESTIONS.resolve():
        raise ValueError("Исходный набор вопросов защищён от изменения.")
    with _USER_LOCK:
        rows = load_user_questions(target)
        row = {"id": question_id if question_id is not None else "user-" + uuid4().hex[:12],
               "question": question.strip(), "expected_in_documents": expected_in_documents,
               "reference_answer": reference_answer.strip(), "expected_source": expected_source.strip()}
        rows = [item for item in rows if item["id"] != row["id"]] + [row]
        _atomic_json(target, rows)
    return row


def delete_user_question(question_id: Any, path: Path | str | None = None) -> None:
    target = Path(path or _USER_QUESTIONS)
    if target.resolve() == _QUESTIONS.resolve():
        raise ValueError("Исходный набор вопросов защищён от изменения.")
    with _USER_LOCK:
        _atomic_json(target, [item for item in load_user_questions(target) if item["id"] != question_id])


def _normalise(text: str) -> str:
    return " ".join(text.casefold().split())


def assess_answer(answer: Any, trace: dict, question: dict) -> dict[str, bool | None]:
    """Check contracts separately from factual correctness; references use exact text."""
    checks: dict[str, bool | None] = dict.fromkeys(
        ("json_valid", "citations_valid", "expected_context", "correct_refusal", "expected_source", "reference_exact_match"))
    try:
        structured = RAGAnswer.model_validate(answer.model_dump() if hasattr(answer, "model_dump") else answer)
    except (ValueError, TypeError):
        checks["json_valid"] = False
        return checks
    checks["json_valid"] = True
    fragments = trace.get("after_rerank", trace.get("after", [])) or []
    allowed = set()
    for fragment in fragments:
        meta = fragment.get("metadata", fragment) if isinstance(fragment, dict) else fragment.metadata
        allowed.add((meta.get("source"), meta.get("chunk_id")))
    citations = {(citation.source, citation.chunk_id) for citation in structured.citations}
    checks["citations_valid"] = (not citations if structured.insufficient_context else bool(citations)) and citations.issubset(allowed)
    checks["expected_context"] = structured.insufficient_context != question["expected_in_documents"]
    if not question["expected_in_documents"]:
        checks["correct_refusal"] = structured.insufficient_context and not citations
    expected_source = question.get("expected_source", "").strip()
    if expected_source:
        checks["expected_source"] = any(source == expected_source for source, _ in citations)
    reference = question.get("reference_answer", "").strip()
    if reference:
        checks["reference_exact_match"] = _normalise(structured.answer) == _normalise(reference)
    return checks


def calculate_metrics(scenarios: list[dict]) -> dict:
    rag = [s for s in scenarios if s.get("mode", "rag") == "rag" and s["status"] in {"Passed", "Failed", "Error"}]
    completed = [s for s in rag if s.get("execution_status") == "Completed"]
    def percent(key: str) -> float | None:
        values = [s.get("checks", {}).get(key) for s in rag if s.get("checks", {}).get(key) is not None]
        return round(100 * sum(bool(v) for v in values) / len(values), 2) if values else None
    generation_times = [s["trace"]["generation_seconds"] for s in rag if s.get("trace", {}).get("generation_seconds") is not None]
    return {"average_latency_seconds": sum(s["elapsed_seconds"] for s in rag) / len(rag) if rag else None,
            "average_generation_seconds": sum(generation_times) / len(generation_times) if generation_times else None,
            "valid_json_percent": percent("json_valid"), "valid_citations_percent": percent("citations_valid"),
            "correct_refusals_percent": percent("correct_refusal"),
            "sufficient_context_percent": 100 * sum(not s["answer"].get("insufficient_context", True) for s in completed if s.get("answer")) / len(completed) if completed else None,
            "technical_errors": sum(s.get("execution_status") == "Error" for s in scenarios)}


class RunManager:
    """Thread-safe job registry. Cache one instance in Streamlit's cache_resource."""
    def __init__(self, results_dir: Path | str = _RESULTS) -> None:
        self.results_dir = Path(results_dir)
        self._runs: dict[str, dict] = {}
        self._events: dict[str, threading.Event] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._secrets: dict[str, tuple[str, ...]] = {}
        self._lock = threading.RLock()

    def _create(self, kind: str, scenarios: list[dict], settings: dict, secrets: tuple[str, ...] = ()) -> str:
        if not RUN_LOCK.acquire(blocking=False):
            raise RuntimeError("Уже выполняется тестирование или переиндексация. Дождитесь завершения текущей операции.")
        try:
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:10]
            try:
                configured_secret = Settings().llm_api_key
            except (OSError, ValueError):
                configured_secret = ""
            env_secrets = tuple(value for name, value in os.environ.items()
                                if re.search(r"(?:API_KEY|TOKEN|SECRET|PASSWORD)$", name, re.I) and value)
            with self._lock:
                self._events[run_id] = threading.Event()
                self._secrets[run_id] = tuple(secret for secret in secrets + env_secrets + (configured_secret,) if secret and secret != "ollama")
                self._runs[run_id] = {"id": run_id, "timestamp": datetime.now(timezone.utc).isoformat(),
                                      "kind": kind, "status": "Running", "settings": settings,
                                      "scenarios": scenarios, "started_monotonic": time.monotonic(),
                                      "elapsed_seconds": 0.0, "total": len(scenarios), "completed": 0,
                                      "current": None, "counts": {}, "metrics": {}}
                with _ACTIVE_IDS_LOCK:
                    _ACTIVE_IDS.add(run_id)
                self._persist(run_id)
            return run_id
        except BaseException:
            with _ACTIVE_IDS_LOCK:
                if "run_id" in locals():
                    _ACTIVE_IDS.discard(run_id)
            RUN_LOCK.release()
            raise

    def _refresh(self, run: dict) -> None:
        run["total"] = len(run["scenarios"])
        run["completed"] = sum(row["status"] in _TERMINAL for row in run["scenarios"])
        run["counts"] = {status: sum(row["status"] == status for row in run["scenarios"])
                         for status in ("Pending", "Running", "Passed", "Failed", "Error", "Skipped", "Cancelled")}
        run["metrics"] = calculate_metrics(run["scenarios"])
        if run["status"] in {"Running", "Cancelling"}:
            run["elapsed_seconds"] = time.monotonic() - run["started_monotonic"]

    def _persist(self, run_id: str) -> None:
        run = self._runs[run_id]
        self._refresh(run)
        payload = {k: v for k, v in run.items() if k != "started_monotonic"}
        _atomic_json(self.results_dir / (run_id + ".json"), _safe(payload, self._secrets[run_id]))

    def snapshot(self, run_id: str) -> dict:
        with self._lock:
            if run_id in self._runs:
                self._refresh(self._runs[run_id])
                return json.loads(json.dumps(_safe({k: v for k, v in self._runs[run_id].items()
                                                   if k != "started_monotonic"}, self._secrets[run_id]), ensure_ascii=False))
        if not re.fullmatch(r"[\w-]+", run_id):
            raise ValueError("Некорректный ID запуска.")
        path = self.results_dir / (run_id + ".json")
        return self._recover(path, json.loads(path.read_text(encoding="utf-8")))

    def _recover(self, path: Path, run: dict) -> dict:
        with _ACTIVE_IDS_LOCK:
            if run.get("status") in {"Running", "Cancelling"} and run["id"] not in _ACTIVE_IDS:
                run.update(status="Error", current=None, error="Запуск прерван перезапуском приложения; полученные результаты сохранены.")
                for row in run.get("scenarios", []):
                    if row["status"] in {"Pending", "Running"}:
                        row.update(status="Cancelled", execution_status="Cancelled", quality_status="NotEvaluated")
                self._refresh(run)
                _atomic_json(path, run)
        return run

    get_run = snapshot

    def list_runs(self) -> list[dict]:
        runs = {}
        if self.results_dir.exists():
            for path in self.results_dir.glob("*.json"):
                try:
                    run = json.loads(path.read_text(encoding="utf-8"))
                    runs[run["id"]] = self._recover(path, run)
                except (OSError, ValueError, KeyError):
                    continue
        with self._lock:
            runs.update({run_id: self.snapshot(run_id) for run_id in self._runs})
        return sorted(runs.values(), key=lambda r: r["timestamp"], reverse=True)

    def cancel(self, run_id: str) -> None:
        with self._lock:
            if run_id not in self._events or self._runs[run_id]["status"] not in {"Running", "Cancelling"}:
                return
            self._events[run_id].set()
            run = self._runs[run_id]
            run["status"] = "Cancelling"
            for scenario in run["scenarios"]:
                if scenario["status"] == "Pending":
                    scenario.update(status="Cancelled", execution_status="Cancelled", quality_status="NotEvaluated")
            process = self._processes.get(run_id)
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            self._persist(run_id)

    def _finish(self, run_id: str, error: Exception | None = None) -> None:
        with self._lock:
            run = self._runs[run_id]
            run["elapsed_seconds"] = time.monotonic() - run["started_monotonic"]
            run["status"] = "Cancelled" if self._events[run_id].is_set() else "Error" if error else "Completed"
            if error:
                run["error"] = str(error)
            for row in run["scenarios"]:
                if row["status"] in {"Pending", "Running"}:
                    row.update(status="Cancelled" if self._events[run_id].is_set() else "Error",
                               execution_status="Cancelled" if self._events[run_id].is_set() else "Error")
            run["current"] = None
            try:
                self._persist(run_id)
            finally:
                with _ACTIVE_IDS_LOCK:
                    _ACTIVE_IDS.discard(run_id)
                RUN_LOCK.release()

    def start_evaluation(self, service: Any, questions: list[dict], retrieval_modes=("dense", "hybrid"),
                         strategies=("zero_shot", "few_shot", "cot_structured"), rerank_options=(False,),
                         top_k: int = 12, instructions: str = "", debug: bool = True,
                         include_baseline: bool = False) -> str:
        if not questions or not retrieval_modes or not strategies or not rerank_options:
            raise ValueError("Выберите вопросы, retrieval, стратегии и настройки reranking.")
        if any(mode not in {"dense", "hybrid"} for mode in retrieval_modes):
            raise ValueError("Неизвестный retrieval mode.")
        if any(strategy not in {"zero_shot", "few_shot", "cot_structured"} for strategy in strategies):
            raise ValueError("Неизвестная стратегия промпта.")
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k должен быть положительным целым числом.")
        for question in questions:
            if not question.get("question", "").strip() or not isinstance(question.get("expected_in_documents"), bool):
                raise ValueError("Вопрос должен содержать текст и expected_in_documents.")
        cfg = getattr(service, "settings", None)
        model = getattr(cfg, "llm_model", "unknown")
        scenarios = []
        for index, (question, retrieval, strategy, rerank) in enumerate(itertools.product(questions, retrieval_modes, strategies, rerank_options)):
            scenarios.append({"id": str(index + 1), "name": f"Q{question.get('id', index + 1)} · {retrieval} · {strategy} · rerank={rerank}",
                              "question": question["question"], "question_id": question.get("id"), "expected_in_documents": question["expected_in_documents"],
                              "question_data": dict(question), "retrieval": retrieval, "strategy": strategy, "rerank": bool(rerank),
                              "model": model, "parameters": {"top_k": top_k, "instructions": instructions, "debug": debug, "classify_rag": True},
                              "mode": "rag", "status": "Pending", "execution_status": "Pending",
                              "quality_status": "NotEvaluated", "elapsed_seconds": 0.0, "checks": {}, "trace": {}, "answer": None, "text": "", "error": ""})
        if include_baseline:
            outside = next((q for q in questions if not q["expected_in_documents"]), None)
            if outside:
                scenarios.insert(0, {"id": "baseline", "name": "No-RAG baseline", "mode": "chat", "question": outside["question"],
                                     "question_id": outside.get("id"), "question_data": outside, "retrieval": "none", "strategy": "baseline",
                                     "rerank": False, "model": model, "parameters": {"debug": debug}, "status": "Pending", "execution_status": "Pending",
                                     "quality_status": "NotEvaluated", "elapsed_seconds": 0.0, "checks": {}, "trace": {}, "answer": None, "text": "", "error": ""})
        options = {"retrieval_modes": list(retrieval_modes), "strategies": list(strategies), "rerank_options": list(rerank_options),
                   "top_k": top_k, "instructions": instructions, "debug": debug, "model": model,
                   "reference_criterion": "Нормализованное точное совпадение с эталоном; без эталона нужна ручная проверка фактов."}
        secret = getattr(cfg, "llm_api_key", "")
        run_id = self._create("evaluation", scenarios, options, (secret,) if secret else ())
        try:
            threading.Thread(target=self._run_evaluation, args=(run_id, service, top_k, instructions, debug), daemon=True).start()
        except BaseException:
            self._finish(run_id, RuntimeError("Не удалось запустить рабочий поток."))
            raise
        return run_id

    def _run_evaluation(self, run_id: str, service: Any, top_k: int, instructions: str, debug: bool) -> None:
        fatal = None
        try:
            for row in self._runs[run_id]["scenarios"]:
                if self._events[run_id].is_set():
                    break
                with self._lock:
                    if self._events[run_id].is_set():
                        break
                    row.update(status="Running", execution_status="Running")
                    self._runs[run_id]["current"] = dict(row)
                    self._persist(run_id)
                started = time.monotonic()
                try:
                    if row["mode"] == "chat":
                        baseline = (ROOT / "prompts" / "baseline.md").read_text(encoding="utf-8")
                        response = service.respond(row["question"], mode="chat", instructions=baseline, debug=debug)
                    else:
                        response = service.respond(row["question"], mode="rag", instructions=instructions,
                                                   retrieval=row["retrieval"], strategy=row["strategy"], top_k=top_k,
                                                   rerank=row["rerank"], debug=debug, classify_rag=True)
                    with self._lock:
                        row["trace"] = dict(response.trace)
                        if not debug:
                            row["trace"].pop("final_prompt", None)
                        row["text"] = response.text
                        row["answer"] = response.answer.model_dump() if response.answer is not None else None
                        row["execution_status"] = "Completed"
                        if row["mode"] == "rag":
                            row["checks"] = assess_answer(row["answer"], row["trace"], row["question_data"])
                            row["quality_status"] = "Passed" if all(v for v in row["checks"].values() if v is not None) else "Failed"
                            row["status"] = row["quality_status"]
                        else:
                            row.update(status="Passed", quality_status="NotEvaluated")
                except Exception as exc:
                    with self._lock:
                        row.update(status="Error", execution_status="Error", error=f"{type(exc).__name__}: {exc}",
                                   traceback=traceback.format_exc())
                        if hasattr(exc, "trace"):
                            row["trace"] = dict(exc.trace)
                            if not debug:
                                row["trace"].pop("final_prompt", None)
                        if hasattr(exc, "raw_response"):
                            row["raw_response"] = exc.raw_response
                            row["text"] = str(exc.raw_response)
                        if hasattr(exc, "json_valid") or hasattr(exc, "citations_valid"):
                            row["checks"] = {"json_valid": getattr(exc, "json_valid", None),
                                             "citations_valid": getattr(exc, "citations_valid", None)}
                finally:
                    with self._lock:
                        row["elapsed_seconds"] = time.monotonic() - started
                        self._runs[run_id]["current"] = None
                        self._persist(run_id)
        except Exception as exc:
            fatal = exc
        finally:
            self._finish(run_id, fatal)

    def start_pytest(self, paths: list[str] | None = None) -> str:
        run_id = self._create("pytest", [], {"paths": paths or ["tests/"]})
        try:
            threading.Thread(target=self._run_pytest, args=(run_id, paths or ["tests/"]), daemon=True).start()
        except BaseException:
            self._finish(run_id, RuntimeError("Не удалось запустить pytest."))
            raise
        return run_id

    def _run_pytest(self, run_id: str, paths: list[str]) -> None:
        fatal = None
        process = None
        try:
            with tempfile.TemporaryDirectory(prefix="rag-pytest-") as temporary:
                events_path = Path(temporary) / "events.jsonl"
                events_path.touch()
                with (Path(temporary) / "output.txt").open("w+") as output:
                    process = subprocess.Popen([sys.executable, "-m", "evaluation.pytest_worker", "--events", str(events_path), *paths],
                                               cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                    with self._lock:
                        self._processes[run_id] = process
                    with events_path.open(encoding="utf-8") as events:
                        while True:
                            line = events.readline()
                            if line:
                                self._pytest_event(run_id, json.loads(line))
                                continue
                            if process.poll() is not None:
                                break
                            if self._events[run_id].is_set():
                                try:
                                    os.killpg(process.pid, signal.SIGTERM)
                                    process.wait(timeout=2)
                                except subprocess.TimeoutExpired:
                                    os.killpg(process.pid, signal.SIGKILL)
                                except ProcessLookupError:
                                    pass
                            time.sleep(0.05)
                    return_code = process.wait()
                    output.seek(0)
                    with self._lock:
                        self._runs[run_id]["output"] = output.read()[-40000:]
                    if return_code not in (0, 1) and not self._events[run_id].is_set():
                        fatal = RuntimeError(f"pytest завершился с кодом {return_code}; подробности в журнале запуска.")
        except Exception as exc:
            fatal = exc
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            with self._lock:
                self._processes.pop(run_id, None)
            self._finish(run_id, fatal)

    def _pytest_event(self, run_id: str, event: dict) -> None:
        with self._lock:
            run = self._runs[run_id]
            if event["event"] == "collected":
                known = {row["name"] for row in run["scenarios"]}
                for name in event["tests"]:
                    if name not in known:
                        run["scenarios"].append({"id": name, "name": name, "mode": "pytest", "status": "Pending",
                                                 "execution_status": "Pending", "quality_status": "NotEvaluated", "elapsed_seconds": 0, "error": "", "checks": {}})
            else:
                row = next((s for s in run["scenarios"] if s["name"] == event["name"]), None)
                if row is None:
                    row = {"id": event["name"], "name": event["name"], "mode": "pytest", "checks": {}}
                    run["scenarios"].append(row)
                if event["event"] == "started":
                    row.update(status="Running", execution_status="Running", elapsed_seconds=0)
                    run["current"] = dict(row)
                elif event["event"] == "result":
                    row.update(status=event["status"], execution_status="Error" if event["status"] == "Error" else "Completed",
                               quality_status=event["status"], elapsed_seconds=event["elapsed_seconds"], error=event.get("error", ""))
            self._persist(run_id)

    def export(self, run_id: str, format: str = "json") -> str:
        return export_run(self.snapshot(run_id), format)


def export_run(run: dict, format: str = "json") -> str:
    if format == "json":
        return json.dumps(run, ensure_ascii=False, indent=2)
    if format == "csv":
        output = io.StringIO()
        fields = ["id", "name", "question", "retrieval", "strategy", "rerank", "model", "status", "execution_status",
                  "quality_status", "elapsed_seconds", "text", "error", "checks", "answer", "trace", "question_id",
                  "question_data", "parameters", "raw_response", "traceback"]
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in run["scenarios"]:
            rendered = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()}
            # Prevent spreadsheet formula execution on exported untrusted model text.
            writer.writerow({k: "'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v for k, v in rendered.items()})
        return output.getvalue()
    if format in {"markdown", "md"}:
        rows = [f"# Результаты {run['kind']}", "", f"Запуск: {run['id']} · {run['timestamp']}", "", f"Статус: {run['status']}", "",
                "Метрики проверяют структуру и поведение. Фактическая правильность требует эталона или ручной проверки.", "", "## Метрики", "", "```json", json.dumps(run.get("metrics", {}), ensure_ascii=False, indent=2), "```", ""]
        for scenario in run["scenarios"]:
            rows += [f"## {scenario['name']}", "", f"**Вопрос:** {scenario.get('question', scenario['name'])}", "",
                     f"**Статус:** {scenario['status']} · **Выполнение:** {scenario.get('execution_status')} · **Качество:** {scenario.get('quality_status')}",
                     f"**Время:** {scenario.get('elapsed_seconds', 0):.3f} с", "", "````json", json.dumps(scenario, ensure_ascii=False, indent=2), "````", ""]
        return "\n".join(rows)
    raise ValueError("Поддерживаются JSON, CSV и Markdown.")
