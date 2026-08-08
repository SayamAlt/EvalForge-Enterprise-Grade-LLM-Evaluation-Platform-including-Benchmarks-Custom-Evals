import pytest, httpx
from unittest.mock import AsyncMock, MagicMock, patch
from anthropic import RateLimitError as ANTRateLimit, APITimeoutError
from evalforge.providers.anthropic_provider import AnthropicProvider, ANTHROPIC_PRICING
from evalforge.providers.base import GenerationRequest, GenerationResponse
from app.core.exceptions import RateLimitError, ProviderTimeoutError, ProviderError

@pytest.fixture
def provider():
    with patch("evalforge.providers.anthropic_provider.settings") as mock_settings:
        mock_settings.anthropic_api_key = "fake-key"
        p = AnthropicProvider("claude-sonnet-4-6")
    return p

def mock_ant_response(model="claude-sonnet-4-6", content="hello", input_tokens=10, output_tokens=5):
    r = MagicMock()
    r.model = model
    r.content = [MagicMock(text=content)]
    r.usage.input_tokens = input_tokens
    r.usage.output_tokens = output_tokens
    r.stop_reason = "end_turn"
    r.model_dump.return_value = {}
    return r

def ant_rate_limit_error():
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(429, request=req)
    return ANTRateLimit("rate limited", response=resp, body=None)

def ant_timeout_error():
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return APITimeoutError(request=req)

async def test_provider_name(provider):
    assert provider.provider_name == "anthropic"

async def test_generate_success(provider):
    provider._client.messages.create = AsyncMock(return_value=mock_ant_response())
    result = await provider.generate(GenerationRequest(prompt="hello"))
    assert isinstance(result, GenerationResponse)
    assert result.text == "hello"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.latency_ms >= 0
    assert result.finish_reason == "end_turn"

async def test_generate_rate_limit_raises(provider):
    provider._client.messages.create = AsyncMock(side_effect=ant_rate_limit_error())
    with pytest.raises(RateLimitError):
        await provider.generate(GenerationRequest(prompt="test"))

async def test_generate_timeout_raises(provider):
    provider._client.messages.create = AsyncMock(side_effect=ant_timeout_error())
    with pytest.raises(ProviderTimeoutError):
        await provider.generate(GenerationRequest(prompt="test"))

async def test_count_tokens_returns_int(provider):
    result = provider.count_tokens("hello world this is a test")
    assert isinstance(result, int)
    assert result > 0

async def test_system_prompt_passed_as_param_not_in_messages(provider):
    """Anthropic rejects 'system' role in messages[]; system must be a separate param."""
    provider._client.messages.create = AsyncMock(return_value=mock_ant_response())
    await provider.generate(GenerationRequest(prompt="hello", system="Be concise"))
    call_kwargs = provider._client.messages.create.call_args.kwargs
    messages = call_kwargs["messages"]
    roles = [m["role"] for m in messages]
    assert "system" not in roles
    assert call_kwargs["system"] == "Be concise"

async def test_estimate_cost_known_model(provider):
    resp = GenerationResponse(
        model="claude-sonnet-4-6",
        text="",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        latency_ms=100,
    )
    inp, out = ANTHROPIC_PRICING["claude-sonnet-4-6"]
    assert provider.estimate_cost(resp) == pytest.approx(inp + out)

async def test_estimate_cost_unknown_model_returns_zero(provider):
    resp = GenerationResponse(
        model="nonexistent-claude-model",
        text="",
        input_tokens=100,
        output_tokens=50,
        latency_ms=10,
    )
    assert provider.estimate_cost(resp) == 0.0

async def test_generate_generic_exception_raises_provider_error(provider):
    provider._client.messages.create = AsyncMock(side_effect=Exception("unexpected"))
    with pytest.raises(ProviderError):
        await provider.generate(GenerationRequest(prompt="test"))