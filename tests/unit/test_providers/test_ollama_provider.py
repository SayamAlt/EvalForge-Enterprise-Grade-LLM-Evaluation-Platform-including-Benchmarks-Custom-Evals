import pytest, httpx
from unittest.mock import AsyncMock, patch
from evalforge.providers.ollama_provider import OllamaProvider
from evalforge.providers.base import GenerationRequest, GenerationResponse
from app.core.exceptions import RateLimitError, ProviderTimeoutError, ProviderError

@pytest.fixture
def provider():
    with patch("evalforge.providers.ollama_provider.settings") as mock_settings:
        mock_settings.ollama_base_url = "http://localhost:11434"
        p = OllamaProvider("llama3.1")
    return p

def mock_ollama_response(content="hello", input_tokens=10, output_tokens=5):
    data = {
        "message": {"content": content},
        "prompt_eval_count": input_tokens,
        "eval_count": output_tokens,
        "done": True,
        "done_reason": "stop",
    }
    req = httpx.Request("POST", "http://localhost:11434/api/chat")
    return httpx.Response(200, json=data, request=req)

async def test_provider_name(provider):
    assert provider.provider_name == "ollama"

async def test_generate_success(provider):
    provider._client.post = AsyncMock(return_value=mock_ollama_response())
    result = await provider.generate(GenerationRequest(prompt="hello"))
    assert isinstance(result, GenerationResponse)
    assert result.text == "hello"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.latency_ms >= 0
    assert result.finish_reason == "stop"

async def test_generate_rate_limit_raises(provider):
    req = httpx.Request("POST", "http://localhost:11434/api/chat")
    resp = httpx.Response(429, request=req)
    provider._client.post = AsyncMock(
        side_effect=httpx.HTTPStatusError("429 Too Many Requests", request=req, response=resp)
    )
    with pytest.raises(RateLimitError):
        await provider.generate(GenerationRequest(prompt="test"))

async def test_generate_timeout_raises(provider):
    provider._client.post = AsyncMock(
        side_effect=httpx.TimeoutException("request timed out")
    )
    with pytest.raises(ProviderTimeoutError):
        await provider.generate(GenerationRequest(prompt="test"))

async def test_generate_http_error_non_429_raises_provider_error(provider):
    req = httpx.Request("POST", "http://localhost:11434/api/chat")
    resp = httpx.Response(500, request=req)
    provider._client.post = AsyncMock(
        side_effect=httpx.HTTPStatusError("500 Internal Server Error", request=req, response=resp)
    )
    with pytest.raises(ProviderError):
        await provider.generate(GenerationRequest(prompt="test"))

async def test_count_tokens_returns_int(provider):
    result = provider.count_tokens("hello world this is a test")
    assert isinstance(result, int)
    assert result > 0

async def test_system_prompt_prepended(provider):
    provider._client.post = AsyncMock(return_value=mock_ollama_response())
    await provider.generate(GenerationRequest(prompt="hello", system="Be concise"))
    call_kwargs = provider._client.post.call_args.kwargs
    messages = call_kwargs["json"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Be concise"
    assert messages[1]["role"] == "user"

async def test_estimate_cost_always_zero(provider):
    resp = GenerationResponse(
        model="llama3.1",
        text="",
        input_tokens=10_000,
        output_tokens=5_000,
        latency_ms=50,
    )
    assert provider.estimate_cost(resp) == 0.0

async def test_generate_done_false_gives_length_finish_reason(provider):
    data = {
        "message": {"content": "truncated"},
        "prompt_eval_count": 10,
        "eval_count": 100,
        "done": False,
    }
    req = httpx.Request("POST", "http://localhost:11434/api/chat")
    mock_resp = httpx.Response(200, json=data, request=req)
    provider._client.post = AsyncMock(return_value=mock_resp)
    result = await provider.generate(GenerationRequest(prompt="test"))
    assert result.finish_reason == "length"