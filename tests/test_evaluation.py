"""Contract checks for reusable jobs; no model server or index is required."""
from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from evaluation.runner import (RUN_LOCK, RunManager, assess_answer, delete_user_question,
                               export_run, load_questions, load_user_questions, save_user_question)
from src.schemas import RAGAnswer


class MockService:
    settings = SimpleNamespace(llm_model="mock-model", llm_api_key="private-test-key")

    def __init__(self, *, forged=False, insufficient=False, error=None):
        self.forged, self.insufficient, self.error = forged, insufficient, error
        self.calls = []

    def respond(self, question, **kwargs):
        self.calls.append((question, kwargs))
        if self.error:
            raise self.error
        if kwargs["mode"] == "chat":
            return SimpleNamespace(text="Вне корпуса: ответ без retrieval", answer=None, trace={})
        answer = RAGAnswer(answer="Ответ", confidence="low" if self.insufficient else "high",
                           insufficient_context=self.insufficient,
                           citations=[] if self.insufficient else [{"source": "forged.txt" if self.forged else "guide.txt", "chunk_id": "abc"}])
        return SimpleNamespace(text=answer.answer, answer=answer,
                               trace={"after_rerank": [{"source": "guide.txt", "chunk_id": "abc", "page": 1, "score": .9, "text": "evidence"}],
                                      "before_rerank": [], "context": "evidence", "router": "DOCUMENT_QUESTION"})


def question(expected=True, **kwargs):
    return {"id": 1, "question": "Тестовый вопрос?", "expected_in_documents": expected, **kwargs}


def wait(manager, run_id, timeout=8):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        result = manager.snapshot(run_id)
        if result["status"] not in {"Running", "Cancelling"}:
            return result
        time.sleep(.01)
    manager.cancel(run_id)
    pytest.fail("Background job did not finish")


def test_default_matrix_preserves_42_cases_and_every_report(tmp_path):
    manager, service = RunManager(tmp_path), MockService()
    questions = load_questions()
    result = wait(manager, manager.start_evaluation(service, questions))
    assert len(service.calls) == 42
    assert result["total"] == result["completed"] == 42
    assert result["counts"]["Passed"] == 36
    assert result["counts"]["Failed"] == 6  # A generated out-of-corpus answer is a quality failure.
    assert result["metrics"]["technical_errors"] == 0
    assert result["metrics"]["correct_refusals_percent"] == 0
    assert result["metrics"]["valid_json_percent"] == 100
    persisted = json.loads((tmp_path / (result["id"] + ".json")).read_text())
    assert persisted["completed"] == 42
    assert persisted["timestamp"].endswith("+00:00")
    assert "private-test-key" not in json.dumps(persisted)


def test_citations_and_expected_source_checked_independently():
    service = MockService(forged=True)
    response = service.respond("?", mode="rag")
    checks = assess_answer(response.answer, response.trace, question(expected_source="guide.txt"))
    assert checks["json_valid"] is True
    assert checks["citations_valid"] is False
    assert checks["expected_context"] is True
    assert checks["expected_source"] is False
    assert checks["reference_exact_match"] is None


def test_refusal_schema_and_explicit_exact_reference():
    response = MockService(insufficient=True).respond("?", mode="rag")
    checks = assess_answer(response.answer, response.trace, question(False, reference_answer="  ОТВЕТ "))
    assert checks["correct_refusal"] is True
    assert checks["citations_valid"] is True
    assert checks["reference_exact_match"] is True
    assert assess_answer({"answer": "invalid"}, {}, question())["json_valid"] is False


def test_background_progress_cancel_and_global_interlock(tmp_path):
    entered, release = threading.Event(), threading.Event()

    class BlockingService(MockService):
        def respond(self, text, **kwargs):
            entered.set()
            release.wait(4)
            return super().respond(text, **kwargs)

    manager = RunManager(tmp_path)
    run_id = manager.start_evaluation(BlockingService(), [question()], strategies=("zero_shot",), retrieval_modes=("dense", "hybrid"))
    try:
        assert entered.wait(2)
        running = manager.snapshot(run_id)
        assert running["current"]["retrieval"] == "dense"
        assert running["counts"]["Running"] == 1
        assert not RUN_LOCK.acquire(blocking=False)
        with pytest.raises(RuntimeError, match="Уже выполняется"):
            RunManager(tmp_path).start_evaluation(MockService(), [question()])
        manager.cancel(run_id)
        cancelling = manager.snapshot(run_id)
        assert cancelling["status"] == "Cancelling"
        assert cancelling["counts"]["Cancelled"] == 1
    finally:
        release.set()
    cancelled = wait(manager, run_id)
    assert cancelled["status"] == "Cancelled"
    assert cancelled["counts"]["Passed"] == 1
    assert cancelled["completed"] == 2
    assert RUN_LOCK.acquire(blocking=False)
    RUN_LOCK.release()


def test_error_secrets_redacted_and_remaining_cases_continue(tmp_path):
    service = MockService(error=ConnectionError("Connection failed: private-test-key api_key=second-secret"))
    manager = RunManager(tmp_path)
    result = wait(manager, manager.start_evaluation(service, [question()], strategies=("zero_shot",), retrieval_modes=("dense", "hybrid")))
    assert result["counts"]["Error"] == 2
    assert result["metrics"]["technical_errors"] == 2
    assert result["metrics"]["valid_json_percent"] is None
    rendered = manager.export(result["id"])
    assert "private-test-key" not in rendered
    assert "second-secret" not in rendered
    assert "ConnectionError" in rendered


def test_baseline_is_free_text_and_excluded_from_rag_quality(tmp_path):
    manager, service = RunManager(tmp_path), MockService(insufficient=True)
    result = wait(manager, manager.start_evaluation(service, [question(False)], retrieval_modes=("dense",),
                                                   strategies=("zero_shot",), include_baseline=True))
    assert result["total"] == 2
    assert service.calls[0][1]["mode"] == "chat"
    assert "instructions" in service.calls[0][1]
    assert result["scenarios"][0]["quality_status"] == "NotEvaluated"
    assert result["metrics"]["correct_refusals_percent"] == 100


def test_custom_question_storage_and_all_export_formats(tmp_path):
    source = load_questions()
    path = tmp_path / "custom.json"
    row = save_user_question("Что такое пакет?", True, reference_answer="Пакет", expected_source="guide.txt", path=path)
    assert load_user_questions(path)[0] == row
    changed = save_user_question("Другой вопрос", False, question_id=row["id"], path=path)
    assert load_user_questions(path) == [changed]
    delete_user_question(row["id"], path=path)
    assert load_user_questions(path) == []
    assert load_questions() == source
    manager = RunManager(tmp_path / "results")
    result = wait(manager, manager.start_evaluation(MockService(), [question()], retrieval_modes=("dense",), strategies=("zero_shot",)))
    assert json.loads(manager.export(result["id"], "json"))["id"] == result["id"]
    assert "guide.txt" in manager.export(result["id"], "csv")
    assert "# Результаты" in manager.export(result["id"], "markdown")
    assert RunManager(tmp_path / "results").list_runs()[0]["id"] == result["id"]
    with pytest.raises(ValueError):
        manager.snapshot("../../escape")


def test_pytest_worker_manager_live_results_and_tracebacks(tmp_path):
    test = tmp_path / "test_job.py"
    test.write_text("import pytest\ndef test_pass(): assert True\ndef test_fail(): assert False, 'diagnostic'\n@pytest.mark.skip(reason='optional')\ndef test_skip(): pass\n")
    manager = RunManager(tmp_path / "reports")
    result = wait(manager, manager.start_pytest([str(test)]))
    assert result["status"] == "Completed"
    assert result["counts"]["Passed"] == result["counts"]["Failed"] == result["counts"]["Skipped"] == 1
    failed = next(row for row in result["scenarios"] if row["status"] == "Failed")
    assert "AssertionError" in failed["error"]
    assert "diagnostic" in failed["error"]
    assert all(row["elapsed_seconds"] >= 0 for row in result["scenarios"])


def test_pytest_cancellation_keeps_completed_tests(tmp_path):
    test = tmp_path / "test_slow.py"
    test.write_text("import time\ndef test_first(): pass\ndef test_slow(): time.sleep(15)\ndef test_last(): pass\n")
    manager = RunManager(tmp_path / "reports")
    run_id = manager.start_pytest([str(test)])
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        snapshot = manager.snapshot(run_id)
        if snapshot["counts"].get("Passed") == 1 and snapshot["counts"].get("Running") == 1:
            break
        time.sleep(.02)
    manager.cancel(run_id)
    result = wait(manager, run_id)
    assert result["status"] == "Cancelled"
    assert result["counts"]["Passed"] == 1
    assert result["counts"]["Cancelled"] == 2


def test_legacy_evaluation_cli_keeps_markdown_and_no_rag_baseline(tmp_path, monkeypatch):
    from evaluation import evaluate
    service = MockService(insufficient=True)
    manager = RunManager(tmp_path / "reports")
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(json.dumps([question(False)]))
    output = tmp_path / "results.md"
    monkeypatch.setattr(evaluate, "ChatService", lambda settings: service)
    monkeypatch.setattr(evaluate, "RunManager", lambda: manager)
    monkeypatch.setattr("sys.argv", ["evaluate", "--output", str(output), "--questions", str(questions_path),
                                     "--retrieval", "dense", "--prompt", "zero_shot"])
    evaluate.main()
    markdown = output.read_text()
    assert "No-RAG baseline" in markdown
    assert "dense" in markdown and "zero_shot" in markdown
    assert service.calls[0][1]["mode"] == "chat"
    assert len(service.calls) == 2


def test_orphaned_runs_recovered_without_losing_finished_results(tmp_path):
    manager = RunManager(tmp_path)
    result = wait(manager, manager.start_evaluation(MockService(), [question()], retrieval_modes=("dense",), strategies=("zero_shot",)))
    result["status"] = "Running"
    result["scenarios"].append({"id": "lost", "name": "Interrupted", "mode": "rag", "status": "Running", "checks": {}})
    path = tmp_path / (result["id"] + ".json")
    path.write_text(json.dumps(result))
    recovered = RunManager(tmp_path).list_runs()[0]
    assert recovered["status"] == "Error"
    assert "перезапуском" in recovered["error"]
    assert recovered["counts"]["Passed"] == 1
    assert recovered["counts"]["Cancelled"] == 1
    assert recovered["scenarios"][0]["answer"] == result["scenarios"][0]["answer"]
    assert json.loads(path.read_text())["status"] == "Error"


@pytest.mark.parametrize("debug", [False, True])
def test_failed_structured_output_keeps_diagnostics_and_distinct_metrics(tmp_path, debug):
    class InvalidCitations(ValueError):
        raw_response = '{"answer":"Ошибочный ответ","citations":[{"source":"invented.txt","chunk_id":"fake"}],"confidence":"high","insufficient_context":false}'
        trace = {"after_rerank": [{"source": "real.txt", "chunk_id": "actual", "text": "context"}],
                 "generation_seconds": .5, "final_prompt": "Grounded prompt"}
        json_valid = True
        citations_valid = False

    manager = RunManager(tmp_path)
    result = wait(manager, manager.start_evaluation(MockService(error=InvalidCitations("bad citation")), [question()],
                                                   retrieval_modes=("dense",), strategies=("zero_shot",), debug=debug))
    row = result["scenarios"][0]
    assert row["status"] == row["execution_status"] == "Error"
    assert row["raw_response"] == InvalidCitations.raw_response
    assert row["trace"]["after_rerank"][0]["source"] == "real.txt"
    assert ("final_prompt" in row["trace"]) is debug
    assert row["checks"] == {"json_valid": True, "citations_valid": False}
    assert "InvalidCitations" in row["traceback"]
    assert result["metrics"]["valid_json_percent"] == 100
    assert result["metrics"]["valid_citations_percent"] == 0
    csv = manager.export(result["id"], "csv")
    assert "question_data" in csv and "raw_response" in csv and "parameters" in csv
    assert "invented.txt" in csv and "top_k" in csv


def test_evaluation_cli_exports_then_exits_nonzero_on_technical_error(tmp_path, monkeypatch):
    from evaluation import evaluate
    manager = RunManager(tmp_path / "reports")
    output = tmp_path / "results.md"
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(json.dumps([question()]))
    monkeypatch.setattr(evaluate, "ChatService", lambda settings: MockService(error=ConnectionError("Ollama unavailable")))
    monkeypatch.setattr(evaluate, "RunManager", lambda: manager)
    monkeypatch.setattr("sys.argv", ["evaluate", "--output", str(output), "--questions", str(questions_path),
                                     "--retrieval", "dense", "--prompt", "zero_shot", "--no-baseline"])
    with pytest.raises(SystemExit) as exc:
        evaluate.main()
    assert exc.value.code == 1
    assert output.exists()
    assert "ConnectionError" in output.read_text()
