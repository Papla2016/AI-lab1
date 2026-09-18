"""LLM abstraction and OpenAI-compatible implementation."""
from typing import Protocol

class LLM(Protocol):
    def generate(self, system: str, user: str) -> str: ...

class OpenAICompatibleLLM:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        from openai import OpenAI
        self.client, self.model = OpenAI(base_url=base_url, api_key=api_key), model

    def generate(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(model=self.model, temperature=0,
            response_format={"type": "json_object"}, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return response.choices[0].message.content or "{}"

