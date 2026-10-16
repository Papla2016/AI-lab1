"""Russian testing dashboard backed by the shared, asynchronous run manager."""
from __future__ import annotations

from collections import defaultdict
import json
import re
from typing import Any

import streamlit as st

from src.config import ROOT, Settings

ACTIVE_STATUSES = ("Pending", "Running", "Cancelling")


@st.cache_resource
def get_manager():
    # No embedding model, vector database, or LLM is loaded when opening this tab.
    from evaluation.runner import RunManager

    return RunManager(ROOT / "evaluation" / "results")


def _service(settings: Settings):
    service = st.session_state.get("chat_service")
    if service is None:
        from src.backend import ChatService

        service = ChatService(settings)
        st.session_state["chat_service"] = service
    return service


def _safe(value: Any, settings: Settings) -> Any:
    """Defence in depth for displayed errors and exports from external services."""
    if isinstance(value, dict):
        return {
            key: ("[скрыто]" if any(term in str(key).lower() for term in
                                   ("api_key", "password", "secret", "token"))
                  else _safe(item, settings))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_safe(item, settings) for item in value]
    if isinstance(value, str):
        secret = settings.llm_api_key
        if secret and secret != "ollama":
            value = value.replace(secret, "[скрыто]")
        value = re.sub(r"(?i)(bearer\s+)[\w.\-]+", r"\1[скрыто]", value)
        value = re.sub(r"(?i)((?:api[_-]?key|token|password|secret)\s*[=:]\s*)[^\s,;]+", r"\1[скрыто]", value)
        return re.sub(r"(https?://)[^/@\s]+:[^/@\s]+@", r"\1[скрыто]@", value)
    return value


def comparison_rows(scenarios: list[dict]) -> list[dict]:
    """Group metrics by question and actual configuration, never call them accuracy."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for scenario in scenarios:
        if scenario.get("status") in ("Pending", "Running", "Cancelled"):
            continue
        key = (scenario.get("question_id", scenario.get("question", "")),
               scenario.get("question", ""), scenario.get("retrieval", "—"),
               scenario.get("strategy", "—"), bool(scenario.get("rerank", False)))
        groups[key].append(scenario)
    rows = []
    for (_, question, retrieval, strategy, rerank), items in groups.items():
        def percentage(check: str):
            values = [item.get("checks", {}).get(check) for item in items
                      if item.get("checks", {}).get(check) is not None]
            return round(100 * sum(bool(value) for value in values) / len(values), 1) if values else None

        rows.append({
            "Вопрос": question,
            "Retrieval": retrieval,
            "Стратегия": strategy,
            "Reranking": "вкл." if rerank else "выкл.",
            "Сценариев": len(items),
            "Среднее время, с": round(sum(item.get("elapsed_seconds", 0) for item in items) / len(items), 3),
            "Валидный JSON, %": percentage("json_valid"),
            "Корректные цитаты, %": percentage("citations_valid"),
            "Корректные отказы, %": percentage("correct_refusal"),
            "Технических ошибок": sum(item.get("execution_status") == "Error" for item in items),
        })
    return rows


def _launch_evaluation(manager, settings: Settings, questions: list[dict], prefix: str, busy: bool):
    retrieval_modes = st.multiselect("Retrieval", ["dense", "hybrid"], default=["dense", "hybrid"],
                                    key=f"{prefix}_retrieval")
    strategies = st.multiselect("Стратегии промптов", ["zero_shot", "few_shot", "cot_structured"],
                               default=["zero_shot", "few_shot", "cot_structured"], key=f"{prefix}_strategies")
    rerank_options = st.multiselect("Сравнение reranking", [False, True], default=[False],
                                   format_func=lambda value: "Включён" if value else "Выключен",
                                   key=f"{prefix}_rerank")
    top_k = st.number_input("Число фрагментов top_k", min_value=1, max_value=100,
                            value=min(settings.top_n, 100), key=f"{prefix}_top_k")
    instructions = st.text_area("Дополнительные инструкции для прогона", key=f"{prefix}_instructions")
    debug = st.checkbox("Сохранять подробную диагностику и итоговый промпт", value=True,
                        key=f"{prefix}_debug")
    baseline = st.checkbox("Добавить сравнение с ответом без RAG", key=f"{prefix}_baseline")
    total = len(questions) * len(retrieval_modes) * len(strategies) * len(rerank_options)
    baseline_available = any(not question["expected_in_documents"] for question in questions)
    st.caption(f"Сценариев RAG: {total}. Полный стандартный набор: 7 × 2 × 3 = 42."
               + (" Дополнительно будет выполнен baseline без RAG." if baseline and baseline_available else ""))
    if baseline and not baseline_available:
        st.caption("Для baseline выберите хотя бы один вопрос вне документального корпуса.")
    if st.button("Запустить RAG evaluation" if prefix == "standard" else "Запустить пользовательские тесты",
                 key=f"{prefix}_start", disabled=busy or total == 0, type="primary"):
        try:
            run_id = manager.start_evaluation(
                _service(settings), questions,
                retrieval_modes=tuple(retrieval_modes), strategies=tuple(strategies),
                rerank_options=tuple(rerank_options), top_k=int(top_k),
                instructions=instructions, debug=debug, include_baseline=baseline,
            )
            st.session_state["testing_run_id"] = run_id
            st.session_state["testing_last_status"] = "Running"
            st.rerun()
        except Exception as exc:
            st.error(_safe(f"Не удалось запустить проверку: {exc}", settings))


def _question_editor(settings: Settings):
    from evaluation.runner import delete_user_question, load_user_questions, save_user_question

    try:
        questions = load_user_questions()
    except (OSError, ValueError) as exc:
        st.error(_safe(f"Не удалось прочитать пользовательские вопросы: {exc}", settings))
        questions = []
    st.caption("Пользовательские вопросы сохраняются отдельно от evaluation/questions.json.")
    with st.form("user_question_form", clear_on_submit=True):
        question = st.text_area("Новый тестовый вопрос")
        expected = st.checkbox("Ответ ожидается в документах", value=True)
        reference = st.text_area("Эталонный ответ (необязательно)")
        source = st.text_input("Ожидаемый источник (необязательно)")
        if st.form_submit_button("Добавить вопрос"):
            if not question.strip():
                st.warning("Введите текст вопроса.")
            else:
                try:
                    save_user_question(question.strip(), expected, reference.strip(), source.strip())
                    st.success("Вопрос сохранён.")
                    st.rerun()
                except (OSError, ValueError) as exc:
                    st.error(_safe(f"Не удалось сохранить вопрос: {exc}", settings))
    if questions:
        by_id = {str(question["id"]): question for question in questions}
        selected = st.selectbox("Вопрос для удаления", list(by_id),
                                format_func=lambda ident: by_id[ident]["question"], key="delete_user_question")
        if st.button("Удалить выбранный вопрос", key="delete_user_question_button"):
            try:
                delete_user_question(by_id[selected]["id"])
                st.rerun()
            except (OSError, ValueError) as exc:
                st.error(_safe(f"Не удалось удалить вопрос: {exc}", settings))
    return questions


def _chunks(items: list[dict]):
    rows = []
    for item in items:
        metadata = item.get("metadata", item)
        rows.append({"Документ": metadata.get("source", "—"), "Страница": metadata.get("page", "—"),
                     "Chunk ID": metadata.get("chunk_id", "—"), "Score": item.get("score"),
                     "Текст": item.get("text", "")})
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.caption("Фрагментов нет.")


def _scenario_card(scenario: dict, settings: Settings, key: str):
    scenario = _safe(scenario, settings)
    label = f"{scenario.get('status', 'Pending')} · {scenario.get('name', scenario.get('question', scenario.get('id', '')))}"
    with st.expander(label):
        elapsed = scenario.get("elapsed_seconds", 0)
        st.caption(f"Время: {elapsed:.3f} с")
        if scenario.get("question"):
            st.markdown("**Входные данные**")
            st.write(scenario["question"])
            st.json({key: scenario.get(key) for key in
                     ("question_id", "question_data", "retrieval", "strategy", "rerank", "model", "parameters")
                     if key in scenario})
        answer = scenario.get("answer") or {}
        trace = scenario.get("trace") or {}
        if answer:
            st.markdown("**Генерация**")
            st.markdown(answer.get("answer", ""))
            st.json({key: answer.get(key) for key in ("confidence", "insufficient_context", "citations")})
        elif scenario.get("text"):
            st.markdown(scenario["text"])
        if trace:
            st.markdown("**Retrieval**")
            before, after = trace.get("before_rerank", []), trace.get("after_rerank", [])
            st.caption(f"До reranking: {len(before)}; после: {len(after)}.")
            st.write("До reranking")
            _chunks(before)
            st.write("После reranking")
            _chunks(after)
            st.write("Результат router:", trace.get("router", "—"))
            if "context" in trace:
                st.text_area("Контекст модели", str(trace["context"]), disabled=True,
                             key=f"{key}_context_{scenario['id']}")
            if "final_prompt" in trace:
                prompt = trace["final_prompt"]
                st.code(prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False, indent=2), language=None)
        st.markdown("**Результат проверки**")
        st.write("Технический статус:", scenario.get("execution_status", scenario.get("status", "—")))
        st.write("Оценка по заданным критериям:", scenario.get("quality_status", "—"))
        checks = scenario.get("checks", {})
        labels = {"json_valid": "Структура RAGAnswer", "citations_valid": "Цитаты из настоящего контекста",
                  "expected_context": "Ожидаемое наличие ответа в корпусе", "correct_refusal": "Корректный отказ вне корпуса",
                  "expected_source": "Ожидаемый источник", "reference_exact_match": "Точное совпадение с эталоном"}
        if checks:
            st.dataframe([{"Проверка": labels.get(key, key),
                           "Результат": "—" if value is None else ("Passed" if value else "Failed")}
                          for key, value in checks.items()], hide_index=True, width="stretch")
        if scenario.get("error"):
            st.error(str(scenario["error"]))
            if "\n" in str(scenario["error"]) and not scenario.get("traceback"):
                st.code(str(scenario["error"]), language="text")
        if scenario.get("traceback"):
            st.code(str(scenario["traceback"]), language="text")


def _metrics(run: dict):
    values = run.get("metrics", {})
    if not values or run.get("kind") == "pytest":
        return
    definitions = [
        ("Среднее время генерации", "average_generation_seconds", " с"),
        ("Среднее время сценария", "average_latency_seconds", " с"),
        ("Валидный JSON", "valid_json_percent", "%"),
        ("Корректные цитаты", "valid_citations_percent", "%"),
        ("Корректные отказы вне корпуса", "correct_refusals_percent", "%"),
        ("Достаточный контекст", "sufficient_context_percent", "%"),
        ("Технические ошибки", "technical_errors", ""),
    ]
    columns = st.columns(3)
    for index, (label, key, suffix) in enumerate(definitions):
        value = values.get(key)
        shown = "—" if value is None else f"{value:.2f}{suffix}"
        columns[index % 3].metric(label, shown)
    st.caption("Эти показатели проверяют формат, цитирование и поведение отказов. "
               "Фактическая правильность ответа требует эталона или ручной оценки; "
               "сравнение с эталоном здесь означает точное совпадение текста.")


def _comparison(scenarios: list[dict], key: str):
    rows = comparison_rows(scenarios)
    if not rows:
        return
    with st.expander("Сравнение конфигураций", expanded=False):
        st.dataframe(rows, hide_index=True, width="stretch")
        questions = list(dict.fromkeys(row["Вопрос"] for row in rows))
        selected = st.selectbox("Вопрос для диаграмм", questions, key=f"{key}_comparison_question")
        selected_rows = [row for row in rows if row["Вопрос"] == selected]
        for field in ("Среднее время, с", "Валидный JSON, %", "Корректные цитаты, %"):
            points = [{"Конфигурация": f"{row['Retrieval']} / {row['Стратегия']} / rerank {row['Reranking']}",
                       field: row[field]} for row in selected_rows if row[field] is not None]
            if points:
                st.bar_chart(points, x="Конфигурация", y=field)


def _exports(manager, run: dict, settings: Settings, key: str):
    columns = st.columns(3)
    for column, (file_format, suffix, mime) in zip(columns, [
        ("json", "json", "application/json"), ("csv", "csv", "text/csv"),
        ("markdown", "md", "text/markdown"),
    ]):
        try:
            data = _safe(manager.export(run["id"], format=file_format), settings)
            column.download_button(f"Скачать {suffix.upper()}", data=data,
                                   file_name=f"{run['id']}.{suffix}", mime=mime,
                                   key=f"{key}_download_{suffix}")
        except (OSError, ValueError) as exc:
            column.error(_safe(f"Ошибка экспорта: {exc}", settings))


def _run_contents(manager, run: dict, settings: Settings, key: str):
    run = _safe(run, settings)
    total, completed = run.get("total", 0), run.get("completed", 0)
    st.progress(min(completed / total, 1.0) if total else 0.0,
                text=f"Выполнено {completed} / {total}")
    elapsed = run.get("elapsed_seconds", 0)
    st.caption(f"Статус: {run.get('status', 'Pending')} · Общее время: {elapsed:.1f} с · ID: {run['id']}")
    counts = run.get("counts", {})
    st.write(f"Успешных: {counts.get('Passed', 0)} · Неуспешных: {counts.get('Failed', 0)} · "
             f"Ошибок: {counts.get('Error', 0)} · Пропущено: {counts.get('Skipped', 0)} · "
             f"Отменено: {counts.get('Cancelled', 0)}")
    current = run.get("current")
    if current:
        st.info(f"Сейчас: {current.get('question', current.get('name', ''))} · "
                f"{current.get('retrieval', '—')} · {current.get('strategy', '—')} · "
                f"reranking: {'вкл.' if current.get('rerank') else 'выкл.'}")
    if run.get("error"):
        st.error(str(run["error"]))
    with st.expander("Параметры запуска"):
        st.json(run.get("settings", {}))
    _metrics(run)
    for scenario in run.get("scenarios", []):
        if scenario.get("status") not in ("Pending", "Running"):
            _scenario_card(scenario, settings, key)
    if run.get("kind") != "pytest":
        _comparison(run.get("scenarios", []), key)
    _exports(manager, run, settings, key)


@st.fragment(run_every="1s")
def _live_run(settings: Settings):
    run_id = st.session_state.get("testing_run_id")
    if not run_id:
        return
    manager = get_manager()
    try:
        run = manager.snapshot(run_id)
    except (OSError, ValueError, KeyError) as exc:
        st.error(_safe(f"Не удалось прочитать результаты: {exc}", settings))
        return
    st.subheader("Текущий запуск")
    if run.get("status") in ("Pending", "Running"):
        if st.button("Остановить прогон и сохранить результаты", key="cancel_testing_run"):
            manager.cancel(run_id)
            st.info("Запрошена остановка. Уже полученные результаты сохранены; текущий запрос завершится или прервётся по таймауту.")
            run = manager.snapshot(run_id)
    elif run.get("status") == "Cancelling":
        st.info("Остановка запрошена. Ожидается завершение текущего запроса или его таймаут; частичные результаты доступны ниже.")
    _run_contents(manager, run, settings, f"live_{run_id}")
    previous = st.session_state.get("testing_last_status")
    st.session_state["testing_last_status"] = run.get("status")
    if previous in ACTIVE_STATUSES and run.get("status") not in ACTIVE_STATUSES:
        st.rerun(scope="app")


def render_testing(settings: Settings):
    """Render controls and a polling fragment without blocking Streamlit's thread."""
    from evaluation.runner import load_questions

    st.subheader("Тестирование")
    st.caption("Результаты появляются после каждого сценария без обновления страницы. "
               "Одновременно выполняется один прогон; результаты сохраняются в evaluation/results/.")
    manager = get_manager()
    try:
        active_runs = [run for run in manager.list_runs() if run.get("status") in ACTIVE_STATUSES]
        busy = bool(active_runs)
        if active_runs and st.session_state.get("testing_run_id") not in {run["id"] for run in active_runs}:
            # Reopening the browser still exposes the active job and its Stop button.
            st.session_state["testing_run_id"] = active_runs[0]["id"]
            st.session_state["testing_last_status"] = "Running"
    except (OSError, ValueError) as exc:
        st.error(_safe(f"Не удалось прочитать список запусков: {exc}", settings))
        busy = False
    if busy:
        st.info("Выполняется прогон. Следующий запуск доступен после его завершения или остановки.")
    with st.expander("Unit-тесты", expanded=True):
        st.caption("Запуск существующего набора tests/ через pytest. Ollama для mock-тестов не требуется.")
        if st.button("Запустить pytest", disabled=busy, key="start_pytest"):
            try:
                st.session_state["testing_run_id"] = manager.start_pytest()
                st.session_state["testing_last_status"] = "Running"
                st.rerun()
            except Exception as exc:
                st.error(_safe(f"Не удалось запустить pytest: {exc}", settings))
    with st.expander("RAG Evaluation", expanded=True):
        try:
            questions = load_questions()
            by_id = {str(question["id"]): question for question in questions}
            selected = st.multiselect("Вопросы стандартного набора", list(by_id), default=list(by_id),
                                      format_func=lambda ident: f"{ident}. {by_id[ident]['question']}",
                                      key="standard_questions")
            _launch_evaluation(manager, settings, [by_id[ident] for ident in selected], "standard", busy)
        except (OSError, ValueError) as exc:
            st.error(_safe(f"Не удалось загрузить сценарии: {exc}", settings))
    with st.expander("Пользовательские тесты"):
        questions = _question_editor(settings)
        if questions:
            by_id = {str(question["id"]): question for question in questions}
            selected = st.multiselect("Пользовательские вопросы для запуска", list(by_id), default=list(by_id),
                                      format_func=lambda ident: by_id[ident]["question"], key="custom_questions")
            _launch_evaluation(manager, settings, [by_id[ident] for ident in selected], "custom", busy)
    _live_run(settings)
    with st.expander("Предыдущие запуски и экспорт"):
        runs = manager.list_runs()
        if not runs:
            st.caption("Сохранённых запусков пока нет.")
            return
        by_id = {run["id"]: run for run in runs}
        selected = st.multiselect("Запуски для просмотра и сравнения", list(by_id),
                                  format_func=lambda ident: f"{by_id[ident].get('timestamp', '')} · "
                                  f"{by_id[ident].get('kind', '')} · {by_id[ident].get('status', '')} · {ident}",
                                  key="previous_testing_runs")
        scenarios = []
        for run_id in selected:
            run = manager.get_run(run_id)
            st.markdown(f"**Запуск {run_id}**")
            _run_contents(manager, run, settings, f"history_{run_id}")
            if run.get("kind") != "pytest":
                scenarios.extend(run.get("scenarios", []))
        if len(selected) > 1:
            _comparison(scenarios, "cross_runs")
