import time
import httpx
from app.core.config import settings
from evalforge.providers.base import BaseProvider, GenerationRequest, GenerationResponse
from app.core.exceptions import RateLimitError, ProviderError, ProviderTimeoutError

class OllamaProvider(BaseProvider):

    def __init__(self, model: str, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=120.0)

    @property
    def provider_name(self) -> str:
        return "ollama"

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
            response = await self._client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_predict": request.max_tokens,
                        "temperature": request.temperature,
                        "stop": request.stop or [],
                    },
                },
            )
            response.raise_for_status()
            data = response.json()

            latency_ms = (time.perf_counter() - start) * 1000
            return GenerationResponse(
                model=self.model,
                text=data["message"]["content"],
                input_tokens=data.get("prompt_eval_count", 0),
                output_tokens=data.get("eval_count", 0),
                latency_ms=latency_ms,
                finish_reason="stop" if data.get("done") else "length",
                raw=data,
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError(str(e))
            raise ProviderError(str(e))
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(str(e))
        except Exception as e:
            raise ProviderError(str(e))

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def estimate_cost(self, response: GenerationResponse) -> float:
        return 0.0  # local inference; no API cost