import pytest, httpx
from unittest.mock import AsyncMock, MagicMock, patch
from groq import RateLimitError as GroqRateLimit, APITimeoutError
from evalforge.providers.groq_provider import GroqProvider, GROQ_PRICING
from evalforge.providers.base import GenerationRequest, GenerationResponse
from app.core.exceptions import RateLimitError, ProviderTimeoutError, ProviderError

@pytest.fixture
def provider():
    with patch("evalforge.providers.groq_provider.settings") as mock_settings:
        mock_settings.groq_api_key = "fake-key"
        p = GroqProvider("llama-3.3-70b")
    return p

def mock_groq_response(model="llama-3.3-70b", content="hello", input_tokens=10, output_tokens=5):
    r = MagicMock()
    r.model = model
    r.choices = [MagicMock()]
    r.choices[0].message.content = content
    r.choices[0].finish_reason = "stop"
    r.usage.prompt_tokens = input_tokens
    r.usage.completion_tokens = output_tokens
    r.model_dump.return_value = {}
    return r

def groq_rate_limit_error():
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(429, request=req)
    return GroqRateLimit("rate limited", response=resp, body=None)

def groq_timeout_error():
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return APITimeoutError(request=req)

async def test_provider_name(provider):
    assert provider.provider_name == "groq"

async def test_generate_success(provider):
    provider._client.chat.completions.create = AsyncMock(return_value=mock_groq_response())
    result = await provider.generate(GenerationRequest(prompt="hello"))
    assert isinstance(result, GenerationResponse)
    assert result.text == "hello"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.latency_ms >= 0

async def test_generate_rate_limit_raises(provider):
    provider._client.chat.completions.create = AsyncMock(side_effect=groq_rate_limit_error())
    with pytest.raises(RateLimitError):
        await provider.generate(GenerationRequest(prompt="test"))

async def test_generate_timeout_raises(provider):
    provider._client.chat.completions.create = AsyncMock(side_effect=groq_timeout_error())
    with pytest.raises(ProviderTimeoutError):
        await provider.generate(GenerationRequest(prompt="test"))

async def test_count_tokens_returns_int(provider):
    result = provider.count_tokens("hello world this is a test")
    assert isinstance(result, int)
    assert result > 0

async def test_system_prompt_prepended(provider):
    provider._client.chat.completions.create = AsyncMock(return_value=mock_groq_response())
    await provider.generate(GenerationRequest(prompt="hello", system="Be brief"))
    call_kwargs = provider._client.chat.completions.create.call_args.kwargs
    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Be brief"

async def test_estimate_cost_known_model(provider):
    resp = GenerationResponse(
        model="llama-3.3-70b-versatile",
        text="",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        latency_ms=100,
    )
    inp, out = GROQ_PRICING["llama-3.3-70b"]
    assert provider.estimate_cost(resp) == pytest.approx(inp + out)

async def test_estimate_cost_unknown_model_returns_zero(provider):
    resp = GenerationResponse(
        model="nonexistent-model",
        text="",
        input_tokens=100,
        output_tokens=50,
        latency_ms=10,
    )
    assert provider.estimate_cost(resp) == 0.0

async def test_generate_generic_exception_raises_provider_error(provider):
    provider._client.chat.completions.create = AsyncMock(side_effect=Exception("unexpected"))
    with pytest.raises(ProviderError):
        await provider.generate(GenerationRequest(prompt="test"))