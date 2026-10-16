"""Exercise actual Streamlit controls with a mock backend, without Ollama/models."""
import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest

from src.backend import ChatResponse, DEFAULT_INSTRUCTIONS, IndexReport
from src.llm import LLMError
from src.schemas import RAGAnswer


@pytest.fixture
def app(monkeypatch):
    calls = []

    class Service:
        def __init__(self, settings):
            self.settings = settings

        def respond(self, question, **options):
            calls.append((question, options))
            if question == "ошибка сервера":
                raise LLMError("Не удалось подключиться к Ollama. Запустите `ollama serve`.")
            rag = options["mode"] == "rag"
            answer = RAGAnswer(answer="Используйте apt-get.", confidence="high", insufficient_context=False,
                               citations=[{"source": "guide.txt", "chunk_id": "real-123"}]) if rag else None
            return ChatResponse(text=answer.answer if answer else "Вот функция сортировки.",
                                mode="rag" if rag else "chat", answer=answer, elapsed_seconds=.25,
                                trace={"router": "DOCUMENT_QUESTION" if rag else "GENERAL_OR_OUT_OF_SCOPE",
                                       "final_prompt": {"messages": [{"role": "user", "content": question}]},
                                       "after_rerank": [{"source": "guide.txt", "chunk_id": "real-123",
                                                         "page": 2, "text": "Используйте apt-get.", "score": .9}]})

        def index_documents(self):
            return IndexReport(1, 3, ["Не удалось прочитать broken.pdf."])

    monkeypatch.setattr("src.backend.ChatService", Service)
    at = AppTest.from_file(Path(__file__).parents[1] / "app.py").run(timeout=15)
    assert not at.exception
    return at, calls


def test_chat_history_settings_diagnostics_and_reset(app):
    at, calls = app
    assert [tab.label for tab in at.tabs] == ["Чат", "Документы", "Тестирование"]
    at.selectbox(key="work_mode").select("Обычный чат")
    at.text_area(key="instructions").set_value("Показывай примеры кода")
    next(box for box in at.checkbox if box.label == "Подробная диагностика").check()
    at.chat_input[0].set_value("Напиши Python-функцию сортировки").run()
    assert not at.exception
    assert calls[-1][1]["instructions"] == "Показывай примеры кода"
    assert calls[-1][1]["mode"] == "chat"
    assert calls[-1][1]["history"] == []
    at.chat_input[0].set_value("А теперь по убыванию").run()
    assert calls[-1][1]["history"] == [
        {"role": "user", "content": "Напиши Python-функцию сортировки"},
        {"role": "assistant", "content": "Вот функция сортировки."},
    ]
    assert any("Режим: Обычный чат" in text.value for text in at.caption)
    next(button for button in at.button if button.label == "Сбросить инструкции").click().run()
    assert at.text_area(key="instructions").value == DEFAULT_INSTRUCTIONS
    at.button(key="clear_history").click().run()
    assert at.session_state["messages"] == []


def test_rag_sources_index_errors_and_ollama_recovery(app):
    at, calls = app
    at.chat_input[0].set_value("Как установить пакет?").run()
    assert not at.exception
    assert calls[-1][1]["mode"] == "rag"
    assert any("guide.txt" in item.value and "real-123" in item.value for item in at.text)
    next(button for button in at.button if button.label == "Переиндексировать документы").click().run()
    assert any("чанков: 3" in item.value for item in at.info)
    assert any("broken.pdf" in item.value for item in at.error)
    at.selectbox(key="work_mode").select("Обычный чат")
    at.chat_input[0].set_value("ошибка сервера").run()
    assert not at.exception
    assert any("Ollama" in item.value for item in at.error)
    at.chat_input[0].set_value("Повторим запрос").run()
    assert not at.exception
    assert not any(item["content"] == "Не удалось подключиться к Ollama. Запустите `ollama serve`."
                   for item in calls[-1][1]["history"])
