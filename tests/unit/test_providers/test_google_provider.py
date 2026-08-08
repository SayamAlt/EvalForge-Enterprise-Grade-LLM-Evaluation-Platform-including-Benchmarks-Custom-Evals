import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from evalforge.providers.google_provider import GoogleProvider, GOOGLE_PRICING
from evalforge.providers.base import GenerationRequest, GenerationResponse
from app.core.exceptions import RateLimitError, ProviderTimeoutError, ProviderError

@pytest.fixture
def provider():
    with patch("evalforge.providers.google_provider.genai") as mock_genai, \
         patch("evalforge.providers.google_provider.settings") as mock_settings:
        mock_settings.google_api_key = "fake-key"
        p = GoogleProvider("gemini-2.5-flash")
    return p

def mock_google_response(content="hello", input_tokens=10, output_tokens=5):
    r = MagicMock()
    r.text = content
    r.candidates = [MagicMock(finish_reason="STOP")]
    r.usage_metadata.prompt_token_count = input_tokens
    r.usage_metadata.candidates_token_count = output_tokens
    return r

async def test_provider_name(provider):
    assert provider.provider_name == "google"

async def test_generate_success(provider):
    provider._client.aio.models.generate_content = AsyncMock(
        return_value=mock_google_response()
    )
    result = await provider.generate(GenerationRequest(prompt="hello"))
    assert isinstance(result, GenerationResponse)
    assert result.text == "hello"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.latency_ms >= 0

async def test_generate_rate_limit_raises(provider):
    provider._client.aio.models.generate_content = AsyncMock(
        side_effect=Exception("429 RESOURCE_EXHAUSTED: quota exceeded")
    )
    with pytest.raises(RateLimitError):
        await provider.generate(GenerationRequest(prompt="test"))

async def test_generate_timeout_raises(provider):
    provider._client.aio.models.generate_content = AsyncMock(
        side_effect=Exception("DEADLINE_EXCEEDED: request timed out")
    )
    with pytest.raises(ProviderTimeoutError):
        await provider.generate(GenerationRequest(prompt="test"))

async def test_generate_generic_exception_raises_provider_error(provider):
    provider._client.aio.models.generate_content = AsyncMock(
        side_effect=Exception("some unexpected API error")
    )
    with pytest.raises(ProviderError):
        await provider.generate(GenerationRequest(prompt="test"))

async def test_count_tokens_returns_int(provider):
    result = provider.count_tokens("hello world this is a test")
    assert isinstance(result, int)
    assert result > 0

async def test_generate_no_candidates_returns_stop(provider):
    r = MagicMock()
    r.text = ""
    r.candidates = []
    r.usage_metadata.prompt_token_count = 5
    r.usage_metadata.candidates_token_count = 0
    provider._client.aio.models.generate_content = AsyncMock(return_value=r)
    result = await provider.generate(GenerationRequest(prompt="test"))
    assert result.finish_reason == "stop"

async def test_estimate_cost_known_model(provider):
    resp = GenerationResponse(
        model="gemini-2.5-flash",
        text="",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        latency_ms=100,
    )
    inp, out = GOOGLE_PRICING["gemini-2.5-flash"]
    assert provider.estimate_cost(resp) == pytest.approx(inp + out)

async def test_estimate_cost_unknown_model_returns_zero(provider):
    resp = GenerationResponse(
        model="nonexistent-gemini-model",
        text="",
        input_tokens=100,
        output_tokens=50,
        latency_ms=10,
    )
    assert provider.estimate_cost(resp) == 0.0