import time
from groq import AsyncGroq, RateLimitError as GroqRateLimit, APITimeoutError
from app.core.config import settings
from evalforge.providers.base import BaseProvider, GenerationRequest, GenerationResponse
from app.core.exceptions import RateLimitError, ProviderError, ProviderTimeoutError

GROQ_PRICING = {
    # (input, output) per million tokens
    "llama-3.3-70b": (0.59, 0.79),
    "llama-3.1-70b": (0.59, 0.79),
    "llama-3.1-8b": (0.05, 0.08),
    "llama3-70b": (0.59, 0.79),
    "llama3-8b": (0.05, 0.08),
    "gemma2-9b": (0.20, 0.20),
    "gemma-7b": (0.07, 0.07),
    "mixtral-8x7b": (0.24, 0.24)
}

class GroqProvider(BaseProvider):

    def __init__(self, model: str, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self._client = AsyncGroq(api_key=settings.groq_api_key)

    @property
    def provider_name(self) -> str:
        return "groq"

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        start = time.perf_counter()
        messages = []

        if request.system:
            messages.append({"role": "system", "content": request.system})
        if isinstance(request.prompt, str):
            messages.append({"role": "user", "content": request.prompt})
        if isinstance(request.prompt, list):
            messages.extend(request.prompt)

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop=request.stop or None,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            return GenerationResponse(
                model=response.model,
                text=response.choices[0].message.content or "",
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                latency_ms=latency_ms,
                finish_reason=response.choices[0].finish_reason or "stop",
                raw=response.model_dump(),
            )
        except GroqRateLimit as e:
            raise RateLimitError(str(e))
        except APITimeoutError as e:
            raise ProviderTimeoutError(str(e))
        except Exception as e:
            raise ProviderError(str(e))

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def estimate_cost(self, response: GenerationResponse) -> float:
        for prefix, (input_price, output_price) in GROQ_PRICING.items():
            if response.model.startswith(prefix):
                return (
                    response.input_tokens / 1_000_000 * input_price +
                    response.output_tokens / 1_000_000 * output_price
                )
        return 0.0