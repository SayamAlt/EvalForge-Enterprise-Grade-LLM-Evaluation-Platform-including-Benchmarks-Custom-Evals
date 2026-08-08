import time
from openai import AsyncOpenAI, RateLimitError as DSRateLimit, APITimeoutError
from app.core.config import settings
from evalforge.providers.base import BaseProvider, GenerationRequest, GenerationResponse
from app.core.exceptions import RateLimitError, ProviderError, ProviderTimeoutError

DEEPSEEK_PRICING = {
    # (input, output) per million tokens
    "deepseek-reasoner": (0.55, 2.19),
    "deepseek-chat": (0.27, 1.10)
}

class DeepSeekProvider(BaseProvider):

    def __init__(self, model: str, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self._client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url="https://api.deepseek.com",
        )

    @property
    def provider_name(self) -> str:
        return "deepseek"

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
        except DSRateLimit as e:
            raise RateLimitError(str(e))
        except APITimeoutError as e:
            raise ProviderTimeoutError(str(e))
        except Exception as e:
            raise ProviderError(str(e))

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def estimate_cost(self, response: GenerationResponse) -> float:
        for prefix, (input_price, output_price) in DEEPSEEK_PRICING.items():
            if response.model.startswith(prefix):
                return (
                    response.input_tokens / 1_000_000 * input_price +
                    response.output_tokens / 1_000_000 * output_price
                )
        return 0.0