"""Dashboard tests use an in-memory manager: no nested pytest or LLM servers."""
from copy import deepcopy
import json

import pytest
from streamlit.testing.v1 import AppTest

from src.config import Settings
from web.testing import _safe, comparison_rows


class FakeManager:
    def __init__(self):
        self.runs = {}
        self.calls = []
        self.cancelled = []

    def list_runs(self):
        return list(self.runs.values())

    def snapshot(self, run_id):
        return deepcopy(self.runs[run_id])

    get_run = snapshot

    def _create(self, kind, **changes):
        run_id = f"run-{len(self.runs) + 1}"
        self.runs[run_id] = {
            "id": run_id, "timestamp": "2026-10-16T12:00:00+00:00", "kind": kind,
            "status": "Completed", "total": 0, "completed": 0, "elapsed_seconds": 0.1,
            "scenarios": [], "current": None, "metrics": {}, "counts": {}, "settings": {},
            **changes,
        }
        return run_id

    def start_pytest(self):
        self.calls.append(("pytest",))
        return self._create("pytest")

    def start_evaluation(self, service, questions, **options):
        self.calls.append(("evaluation", questions, options))
        return self._create("evaluation")

    def cancel(self, run_id):
        self.cancelled.append(run_id)
        self.runs[run_id]["status"] = "Cancelled"

    def export(self, run_id, format="json"):
        return json.dumps(self.runs[run_id])


@pytest.fixture
def dashboard(monkeypatch):
    import evaluation.runner as runner
    import web.testing as testing

    manager = FakeManager()
    user_questions = []
    monkeypatch.setattr(testing, "get_manager", lambda: manager)
    monkeypatch.setattr(testing, "_service", lambda settings: object())
    monkeypatch.setattr(runner, "load_user_questions", lambda: list(user_questions))

    def save(question, expected, reference, source):
        row = dict(id="user-1", question=question, expected_in_documents=expected,
                   reference_answer=reference, expected_source=source)
        user_questions.append(row)
        return row

    monkeypatch.setattr(runner, "save_user_question", save)
    monkeypatch.setattr(runner, "delete_user_question", lambda ident: user_questions.clear())
    app = AppTest.from_string(
        "from src.config import Settings\n"
        "from web.testing import render_testing\n"
        "render_testing(Settings(_env_file=None))\n"
    )
    return app, manager, user_questions


def button(app, label):
    return next(element for element in app.button if element.label == label)


def test_standard_matrix_starts_42_scenarios_without_real_llm(dashboard):
    app, manager, _ = dashboard
    app.run()
    assert not app.exception
    assert any("42" in caption.value for caption in app.caption)
    button(app, "Запустить RAG evaluation").click().run()
    assert not app.exception
    kind, questions, options = manager.calls[0]
    assert kind == "evaluation" and len(questions) == 7
    assert options["retrieval_modes"] == ("dense", "hybrid")
    assert options["strategies"] == ("zero_shot", "few_shot", "cot_structured")
    assert options["rerank_options"] == (False,)


def test_unit_launch_is_mocked_and_busy_run_can_cancel(dashboard):
    app, manager, _ = dashboard
    run_id = manager._create("pytest", status="Running", total=2,
                             current={"name": "tests/test_pipeline.py::test_example"})
    app.run()
    assert not app.exception
    assert button(app, "Запустить pytest").disabled
    assert button(app, "Запустить RAG evaluation").disabled
    button(app, "Остановить прогон и сохранить результаты").click().run()
    assert not app.exception
    assert manager.cancelled == [run_id]
    assert manager.runs[run_id]["status"] == "Cancelled"
    button(app, "Запустить pytest").click().run()
    assert not app.exception
    assert ("pytest",) in manager.calls


def test_custom_questions_add_and_delete_separate_dataset(dashboard):
    app, _, user_questions = dashboard
    app.run()
    next(element for element in app.text_area if element.label == "Новый тестовый вопрос").set_value("Что такое пакет?")
    next(element for element in app.text_input if element.label.startswith("Ожидаемый источник")).set_value("guide.txt")
    button(app, "Добавить вопрос").click().run()
    assert not app.exception
    assert user_questions[0]["question"] == "Что такое пакет?"
    assert user_questions[0]["expected_source"] == "guide.txt"
    assert button(app, "Запустить пользовательские тесты")
    button(app, "Удалить выбранный вопрос").click().run()
    assert not app.exception
    assert user_questions == []


def test_live_results_and_tracebacks_show_without_waiting_for_completion(dashboard):
    app, manager, _ = dashboard
    run_id = manager._create("pytest", status="Running", total=2, completed=1,
                             counts={"Failed": 1}, scenarios=[{
                                 "id": "test-failed", "name": "test_bad", "status": "Failed",
                                 "execution_status": "Failed", "elapsed_seconds": 0.125,
                                 "error": "AssertionError: ожидается другое значение",
                                 "traceback": "test_bad.py:12\nassert False",
                             }])
    app.session_state["testing_run_id"] = run_id
    app.run()
    assert not app.exception
    assert any("Failed · test_bad" in expander.label for expander in app.expander)
    assert any("AssertionError" in error.value for error in app.error)
    assert any("test_bad.py:12" in code.value for code in app.code)
    assert app.get("progress")[0].proto.text == "Выполнено 1 / 2"


def test_comparisons_keep_question_and_rerank_groups_distinct():
    rows = comparison_rows([
        {"question_id": 1, "question": "A?", "retrieval": "dense", "strategy": "zero_shot",
         "rerank": rerank, "status": "Passed", "elapsed_seconds": 1 + rerank,
         "execution_status": "Completed", "checks": {"json_valid": True, "citations_valid": rerank}}
        for rerank in (False, True)
    ])
    assert len(rows) == 2
    assert [row["Корректные цитаты, %"] for row in rows] == [0, 100]
    assert [row["Среднее время, с"] for row in rows] == [1, 2]


def test_rag_details_support_flat_chunks_and_viewing_same_saved_run(dashboard):
    app, manager, _ = dashboard
    run_id = manager._create("evaluation", total=1, completed=1, counts={"Passed": 1},
                             scenarios=[{
                                 "id": "1", "name": "Q1 · dense · zero_shot", "question": "Что такое пакет?",
                                 "question_id": 1, "retrieval": "dense", "strategy": "zero_shot",
                                 "status": "Passed", "execution_status": "Completed", "quality_status": "Passed",
                                 "elapsed_seconds": 0.2,
                                 "answer": {"answer": "Архив файлов.", "confidence": "high", "insufficient_context": False,
                                            "citations": [{"source": "guide.txt", "chunk_id": "c1"}]},
                                 "trace": {"router": "document_question", "after_rerank": [
                                     {"source": "guide.txt", "page": 1, "chunk_id": "c1", "score": 0.7, "text": "Архив файлов."}],
                                           "context": "Документ", "final_prompt": {"system": "Пример", "messages": []}},
                                 "checks": {"json_valid": True, "citations_valid": True, "expected_context": True},
                             }])
    app.session_state["testing_run_id"] = run_id
    app.run()
    assert not app.exception
    next(element for element in app.multiselect if element.key == "previous_testing_runs").set_value([run_id]).run()
    assert not app.exception
    assert sum("Passed · Q1" in expander.label for expander in app.expander) == 2
    assert any("guide.txt" in frame.value.to_string() for frame in app.dataframe)


def test_display_and_export_remove_llm_secret():
    settings = Settings(_env_file=None, llm_api_key="private-example-123")
    result = _safe({"error": "bad private-example-123", "API_KEY": "other", "nested": ["private-example-123"]}, settings)
    assert "private-example-123" not in json.dumps(result)
    assert result["API_KEY"] == "[скрыто]"
