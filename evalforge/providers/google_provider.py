import time
from google import genai
from google.genai import types
from app.core.config import settings
from evalforge.providers.base import BaseProvider, GenerationRequest, GenerationResponse
from app.core.exceptions import RateLimitError, ProviderError, ProviderTimeoutError

GOOGLE_PRICING = {
    # (input, output) per million tokens
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30)
}

class GoogleProvider(BaseProvider):

    def __init__(self, model: str, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self._client = genai.Client(api_key=settings.google_api_key)

    @property
    def provider_name(self) -> str:
        return "google"

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        start = time.perf_counter()

        if isinstance(request.prompt, str):
            contents = request.prompt
        else:
            contents = [
                types.Content(
                    role="model" if msg["role"] == "assistant" else msg["role"],
                    parts=[types.Part(text=msg["content"])],
                )
                for msg in request.prompt
                if msg["role"] != "system"
            ]

        config = types.GenerateContentConfig(
            max_output_tokens=request.max_tokens,
            temperature=request.temperature,
            system_instruction=request.system or None,
            stop_sequences=request.stop or None,
        )

        try:
            response = await self._client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            finish = str(response.candidates[0].finish_reason).lower() if response.candidates else "stop"
            return GenerationResponse(
                model=self.model,
                text=response.text or "",
                input_tokens=response.usage_metadata.prompt_token_count or 0,
                output_tokens=response.usage_metadata.candidates_token_count or 0,
                latency_ms=latency_ms,
                finish_reason=finish,
                raw={},
            )
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                raise RateLimitError(err)
            if "DEADLINE_EXCEEDED" in err or "timeout" in err.lower():
                raise ProviderTimeoutError(err)
            raise ProviderError(err)

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def estimate_cost(self, response: GenerationResponse) -> float:
        for prefix, (input_price, output_price) in GOOGLE_PRICING.items():
            if response.model.startswith(prefix):
                return (
                    response.input_tokens / 1_000_000 * input_price +
                    response.output_tokens / 1_000_000 * output_price
                )
        return 0.0