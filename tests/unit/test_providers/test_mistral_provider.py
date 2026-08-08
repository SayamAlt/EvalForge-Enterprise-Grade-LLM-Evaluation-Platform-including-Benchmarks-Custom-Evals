import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from evalforge.providers.mistral_provider import MistralProvider, MISTRAL_PRICING
from evalforge.providers.base import GenerationRequest, GenerationResponse
from app.core.exceptions import RateLimitError, ProviderTimeoutError, ProviderError

@pytest.fixture
def provider():
    with patch("evalforge.providers.mistral_provider.Mistral") as MockMistral, \
         patch("evalforge.providers.mistral_provider.settings") as mock_settings:
        mock_settings.mistral_api_key = "fake-key"
        p = MistralProvider("mistral-large")
    return p

def mock_mistral_response(model="mistral-large", content="hello", input_tokens=10, output_tokens=5):
    r = MagicMock()
    r.model = model
    r.choices = [MagicMock()]
    r.choices[0].message.content = content
    r.choices[0].finish_reason = "stop"
    r.usage.prompt_tokens = input_tokens
    r.usage.completion_tokens = output_tokens
    return r

async def test_provider_name(provider):
    assert provider.provider_name == "mistral"

async def test_generate_success(provider):
    provider._client.chat.complete_async = AsyncMock(
        return_value=mock_mistral_response()
    )
    result = await provider.generate(GenerationRequest(prompt="hello"))
    assert isinstance(result, GenerationResponse)
    assert result.text == "hello"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.latency_ms >= 0

async def test_generate_rate_limit_raises(provider):
    provider._client.chat.complete_async = AsyncMock(
        side_effect=Exception("429 rate limit exceeded")
    )
    with pytest.raises(RateLimitError):
        await provider.generate(GenerationRequest(prompt="test"))

async def test_generate_timeout_raises(provider):
    provider._client.chat.complete_async = AsyncMock(
        side_effect=Exception("timeout: deadline exceeded")
    )
    with pytest.raises(ProviderTimeoutError):
        await provider.generate(GenerationRequest(prompt="test"))

async def test_generate_generic_exception_raises_provider_error(provider):
    provider._client.chat.complete_async = AsyncMock(
        side_effect=Exception("unexpected API error")
    )
    with pytest.raises(ProviderError):
        await provider.generate(GenerationRequest(prompt="test"))

async def test_count_tokens_returns_int(provider):
    result = provider.count_tokens("hello world this is a test")
    assert isinstance(result, int)
    assert result > 0

async def test_system_prompt_prepended(provider):
    provider._client.chat.complete_async = AsyncMock(
        return_value=mock_mistral_response()
    )
    await provider.generate(GenerationRequest(prompt="hello", system="Be concise"))
    call_kwargs = provider._client.chat.complete_async.call_args.kwargs
    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Be concise"

async def test_estimate_cost_known_model(provider):
    resp = GenerationResponse(
        model="mistral-large",
        text="",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        latency_ms=100,
    )
    inp, out = MISTRAL_PRICING["mistral-large"]
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