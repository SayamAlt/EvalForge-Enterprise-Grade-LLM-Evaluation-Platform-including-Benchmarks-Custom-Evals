import time
from anthropic import AsyncAnthropic, RateLimitError as ANTRateLimit, APITimeoutError
from app.core.config import settings
from evalforge.providers.base import BaseProvider, GenerationRequest, GenerationResponse
from app.core.exceptions import RateLimitError, ProviderError, ProviderTimeoutError

ANTHROPIC_PRICING = {
    "claude-fable-5": (10.00, 50.00),   # (input, output) per million tokens
    "claude-mythos-5": (10.00, 50.00),  
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),   
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-opus-4-5": (5.00, 25.00),
    "claude-opus-4-1": (15.00, 75.00),  
    "claude-opus-4": (15.00, 75.00),   
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-sonnet-4": (3.00, 15.00),   
    "claude-haiku-3-5": (0.80, 4.00),   
    "claude-3-7-sonnet": (3.00, 15.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-opus": (15.00, 75.00),
    "claude-3-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25)
}

class AnthropicProvider(BaseProvider):
    
    def __init__(self, model: str, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        
    @property
    def provider_name(self) -> str:
        return "anthropic"
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        start_time = time.perf_counter()
        messages = []

        if isinstance(request.prompt, str):
            messages.append({"role": "user", "content": request.prompt})
        if isinstance(request.prompt, list):
            messages.extend(request.prompt)

        async def _call(include_temperature: bool) -> object:
            kwargs = dict(
                model=self.model,
                max_tokens=request.max_tokens,
                messages=messages,
                system=request.system or "",
            )
            if include_temperature:
                kwargs["temperature"] = request.temperature
            return await self._client.messages.create(**kwargs)

        try:
            try:
                response = await _call(include_temperature=True)
            except Exception as e:
                err = str(e)
                if "temperature" in err and ("deprecated" in err or "not supported" in err):
                    response = await _call(include_temperature=False)
                else:
                    raise
            latency_ms = (time.perf_counter() - start_time) * 1000
            text_blocks = [b for b in response.content if hasattr(b, "text")]
            return GenerationResponse(
                model=response.model,
                text=text_blocks[0].text if text_blocks else "",
                latency_ms=latency_ms,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                finish_reason=response.stop_reason or "stop",
                raw=response.model_dump(),
            )
        except ANTRateLimit as e:
            raise RateLimitError(str(e))
        except APITimeoutError as e:
            raise ProviderTimeoutError(str(e))
        except Exception as e:
            raise ProviderError(str(e))
        
    def count_tokens(self, text: str) -> int:
        return len(text) // 4
    
    def estimate_cost(self, response: GenerationResponse) -> float:
        for prefix, (input_price, output_price) in ANTHROPIC_PRICING.items():
            if response.model.startswith(prefix):
                return (
                    response.input_tokens / 1_000_000 * input_price +
                    response.output_tokens / 1_000_000 * output_price
                )
        return 0.0