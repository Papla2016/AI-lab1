"""Русский Streamlit-интерфейс общего RAG backend."""
from dataclasses import asdict
import json

import streamlit as st

from src.backend import ChatService, DEFAULT_INSTRUCTIONS
from src.config import Settings
from evaluation.runner import RUN_LOCK
from web.testing import render_testing


MODES = {"RAG — по документам": "rag", "Обычный чат": "chat", "Автоматический режим": "auto"}
MODE_NAMES = {"rag": "RAG", "chat": "Обычный чат"}
CONFIDENCE = {"low": "низкая", "medium": "средняя", "high": "высокая"}


def reset_instructions() -> None:
    st.session_state["instructions"] = DEFAULT_INSTRUCTIONS


def get_service(settings: Settings) -> ChatService:
    # Keep clients local to a browser session. Do not put configuration in diagnostics.
    if st.session_state.get("service_settings") != settings:
        st.session_state["chat_service"] = ChatService(settings)
        st.session_state["service_settings"] = settings
    return st.session_state["chat_service"]


def render_response(message: dict) -> None:
    st.markdown(message["content"])
    if message.get("error"):
        return
    if message.get("mode"):
        st.caption(f"Режим: {MODE_NAMES.get(message['mode'], message['mode'])} · "
                   f"Время: {message.get('elapsed_seconds', 0):.2f} с")
    answer = message.get("answer")
    if answer:
        st.caption(f"Уверенность: {CONFIDENCE[answer['confidence']]}")
        if answer["insufficient_context"]:
            st.info("В документах недостаточно информации для ответа.")
        if answer["citations"]:
            with st.expander("Источники и цитаты", expanded=True):
                fragments = message.get("trace", {}).get("after_rerank", [])
                for citation in answer["citations"]:
                    matching = next((item for item in fragments if item["source"] == citation["source"]
                                     and item["chunk_id"] == citation["chunk_id"]), None)
                    page = f" · страница {matching['page']}" if matching and matching.get("page") else ""
                    st.text(f"{citation['source']}{page} · {citation['chunk_id']}")
                    if matching:
                        st.markdown(matching["text"])
    if message.get("debug"):
        with st.expander("Подробная диагностика"):
            trace = message.get("trace", {})
            st.write("Router:", trace.get("router", "не вызывался"))
            for field, label in (("before_rerank", "Фрагменты до reranking"),
                                 ("after_rerank", "Фрагменты после reranking")):
                st.markdown(f"**{label}**")
                st.json(trace.get(field, []))
            st.markdown("**Контекст, переданный модели**")
            st.code(trace.get("context", "Контекст документов не передавался."), language=None)
            st.markdown("**Итоговый промпт**")
            prompt = trace.get("final_prompt", "Промпт не формировался.")
            st.code(json.dumps(prompt, ensure_ascii=False, indent=2) if isinstance(prompt, dict) else prompt,
                    language=None)


def render_chat(service: ChatService, options: dict) -> None:
    st.subheader("Диалог")
    st.caption("Задайте свой вопрос. Например: «Напиши Python-функцию сортировки» в обычном чате.")
    if st.button("Очистить историю", key="clear_history"):
        st.session_state["messages"] = []
    messages = st.session_state.setdefault("messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            render_response(message)
    question = st.chat_input("Введите сообщение", key="question")
    if question:
        if not RUN_LOCK.acquire(blocking=False):
            st.warning("Сейчас выполняется тестовый прогон или индексация. Дождитесь завершения или остановите прогон.")
            return
        history = [{"role": m["role"], "content": m["content"]} for m in messages if not m.get("error")]
        messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        try:
            with st.chat_message("assistant"):
                with st.spinner("Модель готовит ответ…"):
                    result = service.respond(question, history=history, **options)
                message = {"role": "assistant", "content": result.text, "mode": result.mode,
                           "answer": result.answer.model_dump() if result.answer else None,
                           "trace": result.trace, "elapsed_seconds": result.elapsed_seconds,
                           "debug": options["debug"]}
                messages.append(message)
                render_response(message)
        except Exception as exc:
            # Known service errors have safe, actionable messages. Never expose API exceptions.
            from src.llm import LLMError
            text = str(exc) if isinstance(exc, LLMError) else (
                "Не удалось обработать запрос. Проверьте документы, локальные модели и настройки RAG.")
            st.error(text)
            messages.append({"role": "assistant", "content": text, "error": True})
        finally:
            RUN_LOCK.release()


def render_documents(service: ChatService, settings: Settings) -> None:
    st.subheader("Документы")
    st.caption("Загрузите PDF, TXT или Markdown, затем переиндексируйте корпус. Существующие файлы не перезаписываются.")
    uploads = st.file_uploader("Выберите документы", type=["pdf", "txt", "md"],
                               accept_multiple_files=True, key="documents")
    if st.button("Сохранить файлы", disabled=not uploads):
        if not RUN_LOCK.acquire(blocking=False):
            st.warning("Дождитесь завершения текущего запроса или тестового прогона.")
        else:
            try:
                for upload in uploads:
                    try:
                        path = service.save_upload(upload.name, upload.getvalue())
                        st.success(f"Сохранён файл: {path.name}")
                    except ValueError as exc:
                        st.error(str(exc))
                    except Exception:
                        st.error("Не удалось сохранить файл. Проверьте права на каталог data/.")
            finally:
                RUN_LOCK.release()
    if st.button("Переиндексировать документы", type="primary"):
        if not RUN_LOCK.acquire(blocking=False):
            st.warning("Дождитесь завершения текущего запроса или тестового прогона.")
        else:
            try:
                with st.spinner("Чтение документов и построение индекса…"):
                    report = service.index_documents()
                st.session_state["index_report"] = asdict(report)
            except Exception:
                st.error("Индексация не выполнена. Проверьте наличие читаемых документов и доступность embedding-модели.")
            finally:
                RUN_LOCK.release()
    report = st.session_state.get("index_report")
    if report:
        st.info(f"Документов: {report['document_count']} · чанков: {report['chunk_count']}")
        for error in report["errors"]:
            st.error(error)
    files = sorted(path.relative_to(settings.data_dir).as_posix()
                   for path in settings.data_dir.rglob("*")
                   if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"})
    with st.expander(f"Файлы в корпусе: {len(files)}", expanded=not files):
        if files:
            for name in files:
                st.text(name)
        else:
            st.info("Документы ещё не загружены.")


def main() -> None:
    st.set_page_config(page_title="RAG · Локальный чат", page_icon="💬", layout="wide")
    st.title("RAG · Локальный чат")
    st.caption("Диалог с Ollama, поиск по своим документам и проверка результатов")
    try:
        settings = Settings()
    except Exception:
        st.error("Не удалось прочитать настройки. Проверьте значения в .env.")
        return
    with st.sidebar:
        st.header("Настройки")
        label = st.selectbox("Режим работы", list(MODES), key="work_mode")
        model = st.text_input("Модель Ollama", value=settings.llm_model,
                              help="Например, qwen2.5:3b или qwen2.5:7b. Сначала загрузите её командой ollama pull.")
        retrieval = st.selectbox("Retrieval", ["dense", "hybrid"],
                                 index=0 if settings.retrieval_mode == "dense" else 1)
        strategy = st.selectbox("Стратегия промпта", ["zero_shot", "few_shot", "cot_structured"])
        top_k = st.number_input("top_k — кандидатов для поиска", min_value=1, max_value=100,
                                value=min(settings.top_n, 100), step=1)
        rerank = st.checkbox("Reranking", value=settings.rerank_enabled)
        debug = st.checkbox("Подробная диагностика")
        st.divider()
        st.subheader("Пользовательские инструкции")
        st.session_state.setdefault("instructions", DEFAULT_INSTRUCTIONS)
        st.text_area("Дополнительные пожелания к ответу", key="instructions", height=150,
                     help="В RAG пожелания не отменяют grounding, формат JSON и проверку цитат.")
        st.button("Сбросить инструкции", on_click=reset_instructions)
        st.caption("RAG отвечает только по документам. Обычный чат использует историю диалога. Автоматический режим выбирает router.")
    settings = settings.model_copy(update={"llm_model": model.strip() or settings.llm_model})
    service = get_service(settings)
    chat, documents, testing = st.tabs(["Чат", "Документы", "Тестирование"])
    with chat:
        render_chat(service, {"mode": MODES[label], "retrieval": retrieval, "strategy": strategy,
                              "top_k": int(top_k), "rerank": rerank, "debug": debug,
                              "instructions": st.session_state["instructions"]})
    with documents:
        render_documents(service, settings)
    with testing:
        render_testing(settings)


if __name__ == "__main__":
    main()
