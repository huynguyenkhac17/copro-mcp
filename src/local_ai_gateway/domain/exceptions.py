"""Domain-specific exceptions.

Application/MCP layer bat cac exception nay, khong bat Exception tran trui
o tang provider/infra.
"""


class LocalAIGatewayError(Exception):
    """Base class cho moi loi domain cua he thong."""


class ConfigurationError(LocalAIGatewayError):
    """Config sai/thieu (provider khong ton tai, budget khong hop le, ...)."""


class ProviderUnavailableError(LocalAIGatewayError):
    """Khong ket noi duoc toi provider (vi du Ollama chua chay)."""


class ProviderTimeoutError(LocalAIGatewayError):
    """Provider khong tra loi kip trong thoi gian cho phep."""


class InvalidResponseError(LocalAIGatewayError):
    """Provider tra ve du lieu khong dung dinh dang mong doi."""
