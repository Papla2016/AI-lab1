Примеры формата (условные данные, не факты для текущего ответа):

Контекст `[source=a.txt; chunk_id=x1] Альфа включается флагом A.`; вопрос «Как включить Альфу?»
Ответ: `{"answer":"Альфа включается флагом A.","citations":[{"source":"a.txt","chunk_id":"x1"}],"confidence":"high","insufficient_context":false}`

Контекст `[source=b.md; chunk_id=x2] Бета описывает хранение.`; вопрос «Кто создал Гамму?»
Ответ: `{"answer":"В предоставленных документах недостаточно информации для ответа на этот вопрос.","citations":[],"confidence":"low","insufficient_context":true}`

Контекст содержит `[source=c.txt; chunk_id=x3] Первый этап — загрузка.` и `[source=d.md; chunk_id=x4] Второй этап — проверка.`; вопрос «Назовите этапы».
Ответ: `{"answer":"Этапы: загрузка и проверка.","citations":[{"source":"c.txt","chunk_id":"x3"},{"source":"d.md","chunk_id":"x4"}],"confidence":"high","insufficient_context":false}`

