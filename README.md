# Лабораторная работа № 1: воспроизводимый RAG

Проект реализует реальный локальный конвейер: PDF/Markdown/TXT → рекурсивный chunking → SentenceTransformers → персистентный Chroma → dense или BM25+dense/RRF retrieval → необязательный cross-encoder → защищённый prompt → OpenAI-compatible LLM → проверенный Pydantic JSON с цитатами. Код не привязан к ALT Linux.

## Быстрый старт (Python 3.11+)

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
```

По умолчанию используется лёгкая multilingual E5-модель. При первом запуске она загружается из Hugging Face. Для полностью офлайн-запуска заранее поместите модель в кэш или задайте `EMBEDDING_MODEL` как путь к локальной модели.

Настройте в `.env` любой OpenAI-compatible сервер. Пример для локального Ollama:

```bash
ollama pull qwen2.5:7b
ollama serve
# значения из .env.example уже указывают на http://localhost:11434/v1
```

Ключи никогда не коммитятся: `.env` исключён из Git. Файл `.env` необязателен — безопасные локальные defaults применяются из `src/config.py`.

## Данные и индексация

Самостоятельно положите документы `.pdf`, `.md`, `.txt` в `data/` (включая подпапки), затем:

```bash
python -m src.main index
```

PDF читается постранично; ошибки отдельного файла логируются, а остальные файлы продолжают обрабатываться. Повторная индексация атомарно с точки зрения коллекции заменяет прежний корпус, поэтому удалённые документы не остаются в выдаче. Индекс хранится в `storage/`.

Размер chunk по умолчанию — 1000 символов, overlap — 150 (`CHUNK_SIZE`, `CHUNK_OVERLAP`). Это практичный баланс: слишком малые фрагменты разрывают определения и логические связи; слишком большие смешивают темы, снижают точность retrieval и расходуют context window. Рекурсивное разбиение предпочитает абзацы, затем строки, предложения и слова. Стабильный `chunk_id` образуется из источника, страницы, позиции и текста.

## Retrieval отдельно от LLM

```bash
python -m src.main search "Как обновить пакеты?" --retrieval dense --top-k 5
python -m src.main search "Как обновить пакеты?" --retrieval hybrid --top-k 5
```

JSON выдачи содержит score, source, page, chunk_id и полный текст. Hybrid — не две независимые выдачи: rankings dense и BM25 объединяются Reciprocal Rank Fusion, устойчивым к несовместимым шкалам scores.

## Вопрос к RAG

```bash
python -m src.main ask "Чем отличается Sisyphus от стабильной ветки?" --retrieval hybrid --prompt few_shot --debug
python -m src.main ask "Как работает apt?" --retrieval dense --prompt zero_shot
```

`--debug` печатает вопрос, router, режим, кандидатов до/после rerank, scores, context и итоговый prompt, но не конфигурацию и не секреты. `--prompt` принимает `zero_shot`, `few_shot`, `structured` (последний — zero-shot с обязательной финальной схемой). Reranking включается `RERANK_ENABLED=true`; тогда top-N кандидатов оценивает локальный cross-encoder и оставляет `FINAL_K` с неизменёнными metadata.

Пример формы ответа (конкретные значения появятся только из вашего корпуса):

```json
{
  "answer": "Команда описана в руководстве.",
  "citations": [{"source": "guide.md", "chunk_id": "8f12ab34cd56ef78"}],
  "confidence": "high",
  "insufficient_context": false
}
```

Citation допускается только для реально переданного chunk. Неверная схема или выдуманная citation вызывает один исправляющий retry, затем явную ошибку. Пустой/слабый retrieval (`MIN_DENSE_SCORE`) возвращает детерминированный `insufficient_context=true`, не обращаясь к LLM. Router также пропускает retrieval для явно разговорных запросов; прочие вопросы направляет к документам консервативно, чтобы не отбрасывать предметные формулировки.

## Evaluation и anti-hallucination

После индексации и запуска LLM выполните:

```bash
python -m evaluation.evaluate --output evaluation/results.md
pytest -q
```

Скрипт прогоняет 7 вопросов через dense/hybrid и zero-shot/few-shot/structured, сохраняя sources, chunk IDs и ответы. Последний вопрос намеренно вне предполагаемого корпуса. Неграундированная LLM могла бы правдоподобно угадать ответ, но данный pipeline должен отсечь слабый retrieval либо вернуть недостаточность по prompt. Это результат эксперимента, а не заранее заявленная метрика: перенесите фактические наблюдения в `report.md`. Инструкции внутри документов изолированы delimiters CONTEXT и system prompt требует считать их данными, что защищает от простой prompt injection (но не является абсолютной sandbox-защитой).

## Диагностика

* `python -m src.main --help` — все команды.
* Пустая `data/` даёт понятную ошибку и не стирает существующий индекс.
* Пустой индекс даёт пустой search/недостаточный контекст.
* Ошибка соединения при `ask` означает, что OpenAI-compatible сервер не запущен или `.env` неверен.
* Настройки моделей, порогов, top-N и final-k документированы в `.env.example`.

