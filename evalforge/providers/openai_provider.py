import time, tiktoken
from openai import AsyncOpenAI, RateLimitError as OAIRateLimit, APITimeoutError
from app.core.config import settings
from app.core.exceptions import ProviderError, RateLimitError, ProviderTimeoutError
from evalforge.providers.base import BaseProvider, GenerationRequest, GenerationResponse

MODEL_PRICING = {
    "gpt-5.6-sol": (5.00, 30.00),      # (input, output) per million tokens
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.5-pro": (30.00, 180.00),
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.4-pro": (30.00, 180.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.3-codex": (1.75, 14.00),
    "gpt-5.2": (1.75, 14.00),
    "gpt-5.1": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5": (1.25, 10.00),
    "o1-pro": (150.00, 600.00),
    "o1-mini": (1.10, 4.40),
    "o1": (15.00, 60.00),
    "o3-pro": (20.00, 80.00),
    "o3-mini": (1.10, 4.40),
    "o3": (2.00, 8.00),
    "o4-mini": (1.10, 4.40),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50)
}

class OpenAIProvider(BaseProvider):
    
    def __init__(self, model: str, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        
    @property
    def provider_name(self) -> str:
        return "openai"
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        start = time.perf_counter()
        messages = []
        
        if request.system:
            messages.append({"role": "system", "content": request.system})
        if isinstance(request.prompt, str):
            messages.append({"role": "user", "content": request.prompt})
        if isinstance(request.prompt, list):
            messages.extend(request.prompt)
          
        is_reasoning = self.model.startswith(("o1", "o3", "o4"))
        call_kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "stop": request.stop or None,
            **request.extra,
        }
        if is_reasoning:
            call_kwargs["max_completion_tokens"] = request.max_tokens
        else:
            call_kwargs["max_tokens"] = request.max_tokens
            call_kwargs["temperature"] = request.temperature

        try:
            response = await self._client.chat.completions.create(**call_kwargs)
            latency_ms = (time.perf_counter() - start) * 1000
            return GenerationResponse(
                model=response.model,
                text=response.choices[0].message.content or "",
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                latency_ms=latency_ms,
                finish_reason=response.choices[0].finish_reason or "stop",
                raw=response.model_dump()
            )
        except OAIRateLimit as e:
            raise RateLimitError(str(e))
        except APITimeoutError as e:
            raise ProviderTimeoutError(str(e))
        except ProviderError as e:
            raise ProviderError(str(e)) 
        except Exception as e:
            raise Exception(str(e))
        
    def count_tokens(self, text: str) -> int:
        try:
            enc = tiktoken.encoding_for_model(self.model)
            return len(enc.encode(text))
        except Exception as e:
            return len(text) // 4
        
    def estimate_cost(self, response: GenerationResponse) -> float:
        for prefix, (input_price, output_price) in MODEL_PRICING.items():
            if response.model.startswith(prefix):
                return (
                    response.input_tokens / 1_000_000 * input_price +
                    response.output_tokens / 1_000_000 * output_price
                )
                
        return 0.0