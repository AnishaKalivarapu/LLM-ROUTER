from __future__ import annotations

import httpx

from .base import Completion, ProviderError, post_json


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str, base_url: str, client: httpx.AsyncClient, timeout: float = 30.0):
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._client = client
        self._timeout = timeout

    async def complete(self, model, prompt, *, system, max_tokens, temperature) -> Completion:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
        }
        data = await post_json(
            self._client,
            f"{self._base}/chat/completions",
            headers={"Authorization": f"Bearer {self._key}"},
            payload=payload,
            provider=self.name,
            timeout=self._timeout,
        )
        try:
            text = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage") or {}
            return Completion(
                text=text,
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError("openai returned a malformed response", retryable=False) from exc
