"""One OpenAI-compatible transport for structured RAG and conversational text."""
from typing import Protocol


class LLMError(RuntimeError):
    """A user-facing error that never includes credentials or request internals."""


class LLM(Protocol):
    def generate(self, system: str, user: str) -> str: ...
    def chat(self, messages: list[dict[str, str]], system: str = "") -> str: ...


class OpenAICompatibleLLM:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        from openai import OpenAI
        try:
            self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=120, max_retries=0)
        except (TypeError, ValueError):
            raise LLMError("Проверьте настройки LLM_BASE_URL и LLM_API_KEY в .env.") from None
        self.model = model

    def _complete(self, messages: list[dict[str, str]], *, structured: bool) -> str:
        from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError

        options = {"model": self.model, "temperature": 0, "messages": messages}
        if structured:
            options["response_format"] = {"type": "json_object"}
        try:
            response = self.client.chat.completions.create(**options)
            return response.choices[0].message.content or ("{}" if structured else "")
        except APITimeoutError:
            raise LLMError("Ollama не ответила вовремя. Проверьте сервер и попробуйте модель qwen2.5:3b для CPU.") from None
        except APIConnectionError:
            raise LLMError("Не удалось подключиться к Ollama. Запустите `ollama serve` и проверьте LLM_BASE_URL в .env.") from None
        except AuthenticationError:
            raise LLMError("Сервер LLM отклонил авторизацию. Проверьте LLM_API_KEY в .env.") from None
        except APIStatusError as exc:
            if exc.status_code == 404:
                message = "Модель или endpoint не найдены. Проверьте LLM_MODEL и выполните `ollama pull` для выбранной модели."
            else:
                message = "Сервер LLM не смог обработать запрос. Проверьте состояние Ollama и доступность модели."
            raise LLMError(message) from None

    def generate(self, system: str, user: str) -> str:
        return self._complete([{"role": "system", "content": system}, {"role": "user", "content": user}], structured=True)

    def chat(self, messages: list[dict[str, str]], system: str = "") -> str:
        payload = ([{"role": "system", "content": system}] if system else []) + [dict(item) for item in messages]
        return self._complete(payload, structured=False)

