"""
Base class for all LLM provider integrations.

CONCEPT: The Provider Abstraction Layer
────────────────────────────────────────
EvalForge evaluates multiple models: GPT-4o, Claude 3.5, Gemini 1.5,
Llama 3, Mistral, etc. Each has a different SDK, different request/response
format, different error types, different pricing.

The Provider Abstraction solves this: every concrete provider implements
the same `generate()` interface. The evaluation engine calls `generate()`
and never needs to know if it's talking to OpenAI or Ollama.

This is the Strategy Pattern:
  evaluator.run(dataset, provider=OpenAIProvider("gpt-4o"))
  evaluator.run(dataset, provider=AnthropicProvider("claude-sonnet-4-6"))
  # Same evaluator, different strategies, comparable results

Key design decisions:
  - `GenerationRequest` is provider-agnostic (prompt, max_tokens, temperature)
  - `GenerationResponse` always returns tokens + latency for cost tracking
  - `count_tokens()` estimates without an API call (for cost estimation)
  - `estimate_cost()` is optional but enables cost comparison across providers
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

@dataclass
class GenerationRequest:
    """
    A single generation request — provider-agnostic.

    Attributes:
        prompt:       Input text, OR a list of message dicts for chat models.
                      Chat format: [{"role": "user", "content": "..."}, ...]
        max_tokens:   Max tokens to generate in the response.
        temperature:  Sampling temperature. Use 0.0 for deterministic eval.
        system:       System prompt (supported by most models).
        stop:         Sequences that halt generation early.
        extra:        Provider-specific overrides (passed through as kwargs).
    """
    prompt: str | list[dict[str, str]]
    max_tokens: int = 2048
    temperature: float = 0.0
    system: str | None = None
    stop: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

@dataclass
class GenerationResponse:
    """
    The structured result from one LLM call.

    Attributes:
        text:           The generated output text.
        input_tokens:   Prompt tokens consumed (used for cost calculation).
        output_tokens:  Generated tokens (used for cost calculation).
        latency_ms:     Wall-clock time of the API call in milliseconds.
        model:          Actual model ID used (may differ from alias).
        finish_reason:  Why generation stopped: "stop" | "length" | "error"
        raw:            Full provider response dict (useful for debugging).
    """
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    finish_reason: str = "stop"
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def latency_seconds(self) -> float:
        return self.latency_ms / 1000.0


class BaseProvider(ABC):
    """
    Abstract base class for all LLM provider integrations.

    Subclasses implement:
      - generate()      → async API call, returns GenerationResponse
      - count_tokens()  → local estimate, no API call
      - provider_name   → string identifier ("openai", "anthropic", ...)

    Optionally override:
      - estimate_cost() → USD cost for a completed response

    Usage:
        provider = OpenAIProvider(model="gpt-4o")
        response = await provider.generate(
            GenerationRequest(prompt="Explain photosynthesis in one sentence.")
        )
        print(response.text)
        print(f"Cost: ${provider.estimate_cost(response):.6f}")
        print(f"Latency: {response.latency_ms:.0f}ms")
    """

    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        self._extra_kwargs = kwargs

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Lowercase identifier: "openai" | "anthropic" | "google" | "ollama" | "huggingface"."""

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Send a generation request and return a structured response.

        Implementations must:
          - Measure wall-clock latency and include in GenerationResponse
          - Map provider errors to EvalForge exceptions (ProviderError, RateLimitError)
          - Handle retries at the provider level (or let the evaluator handle it)
        """

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Estimate the token count for `text` without an API call.

        Used for:
          - Pre-flight cost estimation
          - Ensuring prompts fit within context windows
          - Batch cost summaries in EvalReport

        Implementations can use tiktoken (OpenAI), tokenizer (HuggingFace),
        or a rough character-based heuristic (~4 chars per token).
        """

    def estimate_cost(self, response: GenerationResponse) -> float:
        """
        Estimate cost in USD for a completed generation.

        Default: 0.0 (unknown pricing).
        Override in concrete providers using current pricing tables.

        Example (GPT-4o as of 2024):
            input_cost  = response.input_tokens  / 1_000_000 * 5.00
            output_cost = response.output_tokens / 1_000_000 * 15.00
            return input_cost + output_cost
        """
        return 0.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider={self.provider_name!r}, model={self.model!r})"