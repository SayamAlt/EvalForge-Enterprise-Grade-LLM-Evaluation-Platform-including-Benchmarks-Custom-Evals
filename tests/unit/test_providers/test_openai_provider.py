import pytest, httpx
from unittest.mock import AsyncMock, MagicMock, patch
from openai import RateLimitError as OAIRateLimit, APITimeoutError
from evalforge.providers.openai_provider import OpenAIProvider, MODEL_PRICING
from evalforge.providers.base import GenerationRequest, GenerationResponse
from app.core.exceptions import RateLimitError, ProviderTimeoutError, ProviderError

@pytest.fixture
def provider():
    with patch("evalforge.providers.openai_provider.settings") as mock_settings:
        mock_settings.openai_api_key = "fake-key"
        p = OpenAIProvider("gpt-4o")
    return p

def mock_openai_response(model="gpt-4o", content="hello", input_tokens=10, output_tokens=5):
    r = MagicMock()
    r.model = model
    r.choices = [MagicMock()]
    r.choices[0].message.content = content
    r.choices[0].finish_reason = "stop"
    r.usage.prompt_tokens = input_tokens
    r.usage.completion_tokens = output_tokens
    r.model_dump.return_value = {}
    return r

def openai_rate_limit_error():
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(429, request=req)
    return OAIRateLimit("rate limited", response=resp, body=None)

def openai_timeout_error():
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return APITimeoutError(request=req)

async def test_provider_name(provider):
    assert provider.provider_name == "openai"

async def test_generate_success(provider):
    provider._client.chat.completions.create = AsyncMock(return_value=mock_openai_response())
    result = await provider.generate(GenerationRequest(prompt="hello"))
    assert isinstance(result, GenerationResponse)
    assert result.text == "hello"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.latency_ms >= 0
    assert result.finish_reason == "stop"

async def test_generate_rate_limit_raises(provider):
    provider._client.chat.completions.create = AsyncMock(side_effect=openai_rate_limit_error())
    
    with pytest.raises(RateLimitError):
        await provider.generate(GenerationRequest(prompt="test"))

async def test_generate_timeout_raises(provider):
    provider._client.chat.completions.create = AsyncMock(side_effect=openai_timeout_error())
    
    with pytest.raises(ProviderTimeoutError):
        await provider.generate(GenerationRequest(prompt="test"))

async def test_count_tokens_returns_int(provider):
    result = provider.count_tokens("hello world this is a test")
    assert isinstance(result, int)
    assert result > 0

async def test_system_prompt_prepended(provider):
    provider._client.chat.completions.create = AsyncMock(return_value=mock_openai_response())
    await provider.generate(GenerationRequest(prompt="hello", system="You are helpful"))
    call_kwargs = provider._client.chat.completions.create.call_args.kwargs
    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are helpful"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "hello"

async def test_estimate_cost_known_model(provider):
    resp = GenerationResponse(
        model="gpt-4o",
        text="",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        latency_ms=100,
    )
    inp, out = MODEL_PRICING["gpt-4o"]
    assert provider.estimate_cost(resp) == pytest.approx(inp + out)

async def test_estimate_cost_unknown_model_returns_zero(provider):
    resp = GenerationResponse(
        model="nonexistent-model-xyz",
        text="",
        input_tokens=100,
        output_tokens=50,
        latency_ms=10,
    )
    assert provider.estimate_cost(resp) == 0.0

async def test_generate_with_chat_history(provider):
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "How are you?"},
    ]
    provider._client.chat.completions.create = AsyncMock(return_value=mock_openai_response())
    result = await provider.generate(GenerationRequest(prompt=history))
    assert isinstance(result, GenerationResponse)
    call_kwargs = provider._client.chat.completions.create.call_args.kwargs
    assert call_kwargs["messages"] == history

async def test_generate_generic_exception_raises(provider):
    provider._client.chat.completions.create = AsyncMock(
        side_effect=Exception("unexpected error")
    )
    
    with pytest.raises(Exception):
        await provider.generate(GenerationRequest(prompt="test"))