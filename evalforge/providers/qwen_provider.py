import time
from openai import AsyncOpenAI, RateLimitError as QwenRateLimit, APITimeoutError
from app.core.config import settings
from evalforge.providers.base import BaseProvider, GenerationRequest, GenerationResponse
from app.core.exceptions import RateLimitError, ProviderError, ProviderTimeoutError

QWEN_PRICING = {
    # (input, output) per million tokens
    "qwen3-235b": (0.50, 2.00),
    "qwen3-30b": (0.10, 0.40),
    "qwen3-7b": (0.05, 0.20),
    "qwen3": (0.05, 0.20),
    "qwen2.5-72b": (0.40, 1.20),
    "qwen2.5-32b": (0.20, 0.60),
    "qwen2.5-7b": (0.05, 0.15),
    "qwen2.5": (0.05, 0.15)
}

class QwenProvider(BaseProvider):

    def __init__(self, model: str, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self._client = AsyncOpenAI(
            api_key=settings.dashscope_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    @property
    def provider_name(self) -> str:
        return "qwen"

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
        except QwenRateLimit as e:
            raise RateLimitError(str(e))
        except APITimeoutError as e:
            raise ProviderTimeoutError(str(e))
        except Exception as e:
            raise ProviderError(str(e))

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def estimate_cost(self, response: GenerationResponse) -> float:
        for prefix, (input_price, output_price) in QWEN_PRICING.items():
            if response.model.startswith(prefix):
                return (
                    response.input_tokens / 1_000_000 * input_price +
                    response.output_tokens / 1_000_000 * output_price
                )
        return 0.0