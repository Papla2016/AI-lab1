"""Safe, actionable error descriptions without exposing API keys or HTTP bodies.

Only stable exception class names and HTTP status codes are displayed.
Never print provider exception messages verbatim: they may contain secrets.
"""

HINTS = {
    400: "Некорректный запрос к модели. Проверьте поддержку response_format, temperature и идентификатор модели.",
    401: "Ошибка авторизации. Проверьте LLM_API_KEY и провайдера API.",
    403: "Нет прав на модель или API. Проверьте ограничения проекта и доступ к модели.",
    404: "Модель или API endpoint не найдены. Проверьте LLM_MODEL и LLM_BASE_URL.",
    408: "Истекло время ожидания ответа API.",
    422: "API отклонил параметры запроса или формат ответа.",
    429: "Превышена квота или лимит запросов. Проверьте лимиты в панели провайдера.",
    500: "Ошибка сервера поставщика LLM.",
    502: "Шлюз поставщика LLM недоступен.",
    503: "Сервис поставщика LLM временно недоступен.",
    504: "Сервис поставщика LLM не ответил вовремя.",
}

CLASS_HINTS = {
    "APIConnectionError": "Не удалось установить соединение с LLM API. Проверьте сеть, DNS, прокси и LLM_BASE_URL.",
    "APITimeoutError": "LLM API не ответил вовремя.",
    "ConnectError": "Нет соединения с API или локальным сервисом.",
    "ReadTimeout": "Истекло время ожидания ответа.",
    "ModuleNotFoundError": "Отсутствует зависимость Python. Проверьте окружение и requirements.txt.",
}


def describe_error(exc: Exception) -> str:
    """Return a bounded message; API error response bodies are never included."""
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        hint = HINTS.get(status, f"Ошибка HTTP {status} при обращении к поставщику LLM.")
        return f"{name} (HTTP {status}): {hint}"
    if name in CLASS_HINTS:
        return f"{name}: {CLASS_HINTS[name]}"
    if isinstance(exc, (ValueError, FileNotFoundError)):
        # Internal validation messages come from this application, not provider HTTP bodies.
        message = str(exc).strip()[:250]
        return f"{name}: {message}" if message else name
    return f"{name}: ошибка при выполнении операции. Подробности смотрите в терминале сервера."
