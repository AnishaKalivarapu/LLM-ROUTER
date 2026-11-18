import httpx
import pytest

from app.providers.base import ProviderError
from app.providers.gemini_provider import GeminiProvider
from app.providers.openai_provider import OpenAIProvider


def client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def call(provider, **kw):
    return await provider.complete("m", "hello", system=kw.get("system"), max_tokens=10, temperature=0)


async def test_openai_request_and_parse():
    seen = {}

    def handler(req: httpx.Request):
        seen["url"] = str(req.url)
        seen["auth"] = req.headers["authorization"]
        seen["body"] = req.read().decode()
        return httpx.Response(200, json={"choices": [{"message": {"content": "hey"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}})

    p = OpenAIProvider("sk-x", "https://api.openai.com/v1", client(handler))
    out = await call(p, system="be brief")
    assert (out.text, out.input_tokens, out.output_tokens) == ("hey", 3, 2)
    assert seen["url"].endswith("/chat/completions") and seen["auth"] == "Bearer sk-x"
    assert '"role":"system"' in seen["body"].replace(" ", "")


async def test_gemini_request_and_parse_includes_thinking_tokens():
    seen = {}

    def handler(req: httpx.Request):
        seen["url"] = str(req.url)
        seen["key"] = req.headers["x-goog-api-key"]
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "he"}, {"text": "llo"}]}}],
            "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2, "thoughtsTokenCount": 6},
        })

    p = GeminiProvider("g-x", "https://generativelanguage.googleapis.com/v1beta", client(handler))
    out = await call(p)
    assert out.text == "hello" and out.input_tokens == 4 and out.output_tokens == 8
    assert seen["url"].endswith("/models/m:generateContent") and "g-x" not in seen["url"]
    assert seen["key"] == "g-x"


@pytest.mark.parametrize("status,retryable", [(429, True), (500, True), (503, True), (400, False), (401, False), (404, False)])
async def test_http_status_mapping(status, retryable):
    p = OpenAIProvider("k", "http://x", client(lambda r: httpx.Response(status, text="err")))
    with pytest.raises(ProviderError) as e:
        await call(p)
    assert e.value.retryable is retryable


async def test_timeout_is_retryable():
    def handler(req):
        raise httpx.ReadTimeout("slow")
    p = OpenAIProvider("k", "http://x", client(handler))
    with pytest.raises(ProviderError) as e:
        await call(p)
    assert e.value.retryable


async def test_gemini_blocked_prompt_not_retryable():
    p = GeminiProvider("k", "http://x", client(lambda r: httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})))
    with pytest.raises(ProviderError) as e:
        await call(p)
    assert not e.value.retryable


async def test_malformed_openai_body():
    p = OpenAIProvider("k", "http://x", client(lambda r: httpx.Response(200, json={"nope": 1})))
    with pytest.raises(ProviderError):
        await call(p)
