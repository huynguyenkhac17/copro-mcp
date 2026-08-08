"""Domain-specific exceptions.

Application/MCP layer catches these exceptions, not the bare Exceptions
from the provider/infra layers.
"""


class LocalAIGatewayError(Exception):
    """Base class for all domain errors in the system."""


class ConfigurationError(LocalAIGatewayError):
    """Missing/invalid config (provider not found, budget invalid, etc)."""


class ProviderUnavailableError(LocalAIGatewayError):
    """Cannot connect to the provider (e.g. Ollama is not running)."""


class ProviderTimeoutError(LocalAIGatewayError):
    """Provider did not respond within the allowed time."""


class InvalidResponseError(LocalAIGatewayError):
    """Provider returned data in an unexpected format."""
