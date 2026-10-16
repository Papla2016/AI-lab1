Верните только один корректный JSON-объект без Markdown. Схема:
{"answer":"строка", "citations":[{"source":"имя файла", "chunk_id":"идентификатор"}], "confidence":"low|medium|high", "insufficient_context":false}
Каждая цитата обязана точно соответствовать заголовку одного фрагмента CONTEXT. Не создавайте отсутствующие идентификаторы.

