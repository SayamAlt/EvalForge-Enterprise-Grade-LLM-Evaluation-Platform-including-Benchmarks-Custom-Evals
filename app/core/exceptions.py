"""
Custom exception hierarchy for EvalForge.

CONCEPT: Typed Exceptions
──────────────────────────
Using typed exceptions instead of generic `raise Exception("...")`:
  - The global error handler in main.py maps each type to an HTTP status
  - Callers can catch specific error types (except ProviderError)
  - Log messages are consistent - the handler logs exc.detail

Tree:
    EvalForgeError              (base - maps to 500)
    ├── NotFoundError           (404)
    ├── ValidationError         (422)
    ├── ConfigurationError      (400)
    ├── DatasetError            (500)
    ├── EvaluationError         (500)
    └── ProviderError           (502)
        ├── RateLimitError      (429)
        └── ProviderTimeoutError(504)
"""
from http import HTTPStatus

class EvalForgeError(Exception):
    """ Base exception - all EvalForge errors inherit from this. """

    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    detail: str = "An unexpected error occurred."

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or self.__class__.detail
        super().__init__(self.detail)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(detail={self.detail!r})"

class NotFoundError(EvalForgeError):
    status_code = HTTPStatus.NOT_FOUND
    detail = "Resource not found."

class ValidationError(EvalForgeError):
    status_code = HTTPStatus.UNPROCESSABLE_ENTITY
    detail = "Validation error."

class ConfigurationError(EvalForgeError):
    status_code = HTTPStatus.BAD_REQUEST
    detail = "Configuration error."

class DatasetError(EvalForgeError):
    """ Raised for dataset loading, parsing, or schema errors. """
    detail = "Dataset error."

class EvaluationError(EvalForgeError):
    """ Raised when an evaluation pipeline fails unrecoverably. """
    detail = "Evaluation pipeline failed."

class ProviderError(EvalForgeError):
    """ Raised when an LLM provider API call fails. """
    status_code = HTTPStatus.BAD_GATEWAY
    detail = "LLM provider error."

class RateLimitError(ProviderError):
    """ Raised on HTTP 429 from any provider. """
    status_code = HTTPStatus.TOO_MANY_REQUESTS
    detail = "LLM provider rate limit exceeded. Retry after backoff."

class ProviderTimeoutError(ProviderError):
    """ Raised when a provider call exceeds eval_timeout_seconds. """
    status_code = HTTPStatus.GATEWAY_TIMEOUT
    detail = "LLM provider call timed out."