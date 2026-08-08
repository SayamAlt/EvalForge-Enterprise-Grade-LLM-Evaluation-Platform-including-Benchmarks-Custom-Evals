from evalforge.providers.openai_provider import OpenAIProvider
from evalforge.providers.anthropic_provider import AnthropicProvider
from evalforge.providers.ollama_provider import OllamaProvider
from evalforge.providers.deepseek_provider import DeepSeekProvider
from evalforge.providers.google_provider import GoogleProvider
from evalforge.providers.groq_provider import GroqProvider
from evalforge.providers.qwen_provider import QwenProvider
from evalforge.providers.writer_provider import WriterProvider
from evalforge.providers.minimax_provider import MinimaxProvider
from evalforge.providers.mistral_provider import MistralProvider
from evalforge.providers.moonshot_provider import MoonshotProvider
from evalforge.providers.base import BaseProvider
from app.core.exceptions import ConfigurationError

MODEL_REGISTRY: dict[str, type[BaseProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
    "deepseek": DeepSeekProvider,
    "google": GoogleProvider,
    "groq": GroqProvider,
    "qwen": QwenProvider,
    "writer": WriterProvider,
    "minimax": MinimaxProvider,
    "mistral": MistralProvider,
    "moonshot": MoonshotProvider
}

def get_provider(provider_name: str, model: str, **kwargs) -> BaseProvider:
    """ 
    Instantiates an LLM provider by name.
    Raises a configuration error if provider name is unknown.
    """
    provider = MODEL_REGISTRY.get(provider_name.lower())

    if provider is None:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ConfigurationError(
            f"Unknown LLM provider name: {provider_name}. "
            f"Available provider names: {available}"
        )
    
    return provider(model=model, **kwargs)

def list_providers() -> list[dict]:
    """ Returns all registered providers with their respective model names. """
    return [{"provider": provider.lower()} for provider in list(MODEL_REGISTRY.keys())]