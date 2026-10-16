"""CLI compatibility checks without embeddings, Ollama, or a persistent index."""
import json
import sys

import pytest

from src import main as cli
from src.backend import ChatResponse, IndexReport
from src.config import Settings
from src.llm import LLMError
from src.schemas import INSUFFICIENT, SearchResult


@pytest.mark.parametrize("command", ["ask", "search", "baseline"])
def test_existing_cli_question_commands_keep_their_flags(command):
    args = cli.parser().parse_args([
        command, "Произвольный вопрос", "--retrieval", "dense", "--top-k", "3", "--debug",
    ])
    assert args.command == command
    assert args.question == "Произвольный вопрос"
    assert args.retrieval == "dense" and args.top_k == 3 and args.debug


def test_ask_keeps_all_prompt_strategies():
    for strategy in ("zero_shot", "few_shot", "cot_structured"):
        assert cli.parser().parse_args(["ask", "Вопрос", "--prompt", strategy]).prompt == strategy


def test_index_command_accepts_no_question():
    assert cli.parser().parse_args(["index"]).command == "index"


@pytest.fixture
def cli_service(monkeypatch, tmp_path):
    calls = []
    settings = Settings(_env_file=None, data_dir=tmp_path / "data", storage_dir=tmp_path / "storage")
    item = SearchResult("Документированный ответ.", {"source": "guide.txt", "chunk_id": "abc", "page": 2}, .9)

    class FakeService:
        def __init__(self, cfg):
            assert cfg is settings

        def index_documents(self):
            calls.append(("index", {}))
            return IndexReport(2, 7)

        def search(self, question, **kwargs):
            calls.append(("search", {"question": question, **kwargs}))
            return [item], [item]

        def respond(self, question, **kwargs):
            calls.append(("respond", {"question": question, **kwargs}))
            if kwargs["mode"] == "chat":
                return ChatResponse("Свободный ответ, без JSON.", "chat")
            return ChatResponse(
                INSUFFICIENT.answer, "rag", answer=INSUFFICIENT.model_copy(),
                trace={"router": "DOCUMENT_QUESTION", "after_rerank": [{"source": "guide.txt", "chunk_id": "abc"}]},
            )

    monkeypatch.setattr(cli, "Settings", lambda: settings)
    monkeypatch.setattr(cli, "ChatService", FakeService)
    return calls


def invoke(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["rag", *args])
    cli.main()


def test_cli_index_uses_shared_indexer(cli_service, monkeypatch, capsys):
    invoke(monkeypatch, "index")
    assert cli_service == [("index", {})]
    assert "документов: 2, чанков: 7" in capsys.readouterr().out


def test_cli_search_serializes_real_source_metadata(cli_service, monkeypatch, capsys):
    invoke(monkeypatch, "search", "Как установить?", "--retrieval", "dense", "--top-k", "3")
    results = json.loads(capsys.readouterr().out)
    assert results[0]["source"] == "guide.txt"
    assert results[0]["chunk_id"] == "abc" and results[0]["page"] == 2
    assert cli_service[0] == ("search", {
        "question": "Как установить?", "retrieval": "dense", "top_k": 3, "rerank": False,
    })


def test_cli_ask_outputs_valid_json_and_uses_selected_strategy(cli_service, monkeypatch, capsys):
    invoke(monkeypatch, "ask", "Вопрос по документам", "--prompt", "few_shot")
    answer = json.loads(capsys.readouterr().out)
    assert answer["insufficient_context"] is True
    assert answer["citations"] == []
    method, parameters = cli_service[0]
    assert method == "respond" and parameters["mode"] == "rag"
    assert parameters["strategy"] == "few_shot"
    assert parameters["classify_rag"] is True


def test_cli_ask_debug_trace_remains_json_serializable(cli_service, monkeypatch, capsys):
    invoke(monkeypatch, "ask", "Вопрос", "--debug")
    output = capsys.readouterr().out
    assert "=== DEBUG" in output and "guide.txt" in output
    assert "=== END DEBUG ===" in output


def test_cli_baseline_outputs_plain_chat_without_loading_index(cli_service, monkeypatch, capsys):
    invoke(monkeypatch, "baseline", "Напиши функцию сортировки")
    assert capsys.readouterr().out.strip() == "Свободный ответ, без JSON."
    assert len(cli_service) == 1
    assert cli_service[0][0] == "respond" and cli_service[0][1]["mode"] == "chat"
    assert cli_service[0][1]["instructions"]


def test_cli_reports_missing_ollama_without_traceback(monkeypatch):
    class FailingService:
        def __init__(self, settings):
            pass

        def respond(self, *args, **kwargs):
            raise LLMError("Не удалось подключиться к Ollama. Запустите `ollama serve`.")

    monkeypatch.setattr(cli, "ChatService", FailingService)
    with pytest.raises(SystemExit, match="Ollama"):
        invoke(monkeypatch, "baseline", "Привет")
