import time
from mistralai.client import Mistral
from app.core.config import settings
from evalforge.providers.base import BaseProvider, GenerationRequest, GenerationResponse
from app.core.exceptions import RateLimitError, ProviderError, ProviderTimeoutError

MISTRAL_PRICING = {
    # (input, output) per million tokens
    "mistral-large": (2.00, 6.00),
    "mistral-medium": (0.40, 1.20),
    "mistral-small": (0.10, 0.30),
    "mistral-nemo": (0.15, 0.15),
    "codestral": (0.20, 0.60),
    "open-mistral-7b": (0.25, 0.25),
    "open-mixtral-8x7b": (0.70, 0.70),
    "open-mixtral-8x22b": (2.00, 6.00)
}

class MistralProvider(BaseProvider):

    def __init__(self, model: str, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self._client = Mistral(api_key=settings.mistral_api_key)

    @property
    def provider_name(self) -> str:
        return "mistral"

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
            response = await self._client.chat.complete_async(
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
                finish_reason=str(response.choices[0].finish_reason or "stop"),
                raw={},
            )
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                raise RateLimitError(err)
            if "timeout" in err.lower() or "deadline" in err.lower():
                raise ProviderTimeoutError(err)
            raise ProviderError(err)

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def estimate_cost(self, response: GenerationResponse) -> float:
        for prefix, (input_price, output_price) in MISTRAL_PRICING.items():
            if response.model.startswith(prefix):
                return (
                    response.input_tokens / 1_000_000 * input_price +
                    response.output_tokens / 1_000_000 * output_price
                )
        return 0.0