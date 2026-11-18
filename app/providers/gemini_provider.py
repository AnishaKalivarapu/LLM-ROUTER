from __future__ import annotations

import httpx

from .base import Completion, ProviderError, post_json


class GeminiProvider:
    """Native Gemini generateContent API (not the OpenAI-compat shim)."""

    name = "gemini"

    def __init__(self, api_key: str, base_url: str, client: httpx.AsyncClient, timeout: float = 30.0):
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._client = client
        self._timeout = timeout

    async def complete(self, model, prompt, *, system, max_tokens, temperature) -> Completion:
        payload: dict = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        data = await post_json(
            self._client,
            f"{self._base}/models/{model}:generateContent",
            headers={"x-goog-api-key": self._key},  # header, not query string -> key never lands in URLs/logs
            payload=payload,
            provider=self.name,
            timeout=self._timeout,
        )

        if data.get("promptFeedback", {}).get("blockReason"):
            raise ProviderError(f"gemini blocked the prompt: {data['promptFeedback']['blockReason']}", retryable=False)
        try:
            candidate = data["candidates"][0]
            parts = candidate.get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
            usage = data.get("usageMetadata") or {}
            # Gemini 2.5 "thinking" tokens are billed as output tokens.
            out = int(usage.get("candidatesTokenCount", 0)) + int(usage.get("thoughtsTokenCount", 0))
            return Completion(text=text, input_tokens=int(usage.get("promptTokenCount", 0)), output_tokens=out)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError("gemini returned a malformed response", retryable=False) from exc
