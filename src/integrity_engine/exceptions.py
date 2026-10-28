"""Domain exceptions with actionable error messages."""


class IntegrityEngineError(Exception):
    """Base exception for expected engine failures."""


class OptionalDependencyError(IntegrityEngineError):
    """Raised when an optional feature lacks its runtime dependency."""


class ConfigurationError(IntegrityEngineError):
    """Raised when configuration or CLI overrides are invalid."""
