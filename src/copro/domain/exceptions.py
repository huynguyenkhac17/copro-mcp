"""Domain-specific exceptions.

Application/MCP layer catches these exceptions, not the bare Exceptions
from the provider/infra layers.
"""


class LocalAIGatewayError(Exception):
    """Base class for all domain errors in the system."""


class ConfigurationError(LocalAIGatewayError):
    """Missing/invalid config (provider not found, budget invalid, etc)."""


class NoUsableContextError(LocalAIGatewayError):
    """file_paths were given but none yielded any content, and the request carries no
    other context: sending the task alone would only get "I don't see any file".
    `warnings` are the file warnings (file_access_denied:..., file_denied_sensitive:...)
    so the failed result still carries them."""

    def __init__(self, message: str = "", *, warnings: list[str] | None = None) -> None:
        super().__init__(message)
        self.warnings = list(warnings or [])


class ProviderUnavailableError(LocalAIGatewayError):
    """Cannot connect to the provider (e.g. Ollama is not running).

    `retry_after` is how long until the provider can take a request again, when it is
    known (a 429 with Retry-After/retryDelay, or a key drained per the rate-limit
    headers); 0 means another key is free right now. None = no hint, back off blindly."""

    def __init__(self, message: str = "", *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class RequestTooLargeError(LocalAIGatewayError):
    """The prompt exceeds what this provider accepts per request (e.g. Groq free tier:
    8K tokens/minute per key -> HTTP 413). Not retryable: the same prompt will be refused
    again, so the fallback chain should move on immediately."""


class ProviderTimeoutError(LocalAIGatewayError):
    """Provider did not respond within the allowed time."""


class InvalidResponseError(LocalAIGatewayError):
    """Provider returned data in an unexpected format."""
