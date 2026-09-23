"""Exception hierarchy shared by all agents and providers."""


class DocExtractorError(Exception):
    """Base class for all pipeline errors."""


class ConfigError(DocExtractorError):
    """Invalid configuration or lexicon file."""


class IngestError(DocExtractorError):
    """The input document could not be read."""


class StageValidationError(DocExtractorError):
    """Input or output validation of a stage failed."""

    def __init__(self, stage: str, message: str, *, phase: str = "output") -> None:
        self.stage = stage
        self.phase = phase
        super().__init__(f"[{stage}] {phase} validation failed: {message}")


class StageFailedError(DocExtractorError):
    """A stage exhausted its retries."""

    def __init__(self, stage: str, attempts: int, cause: BaseException) -> None:
        self.stage = stage
        self.attempts = attempts
        self.cause = cause
        super().__init__(f"[{stage}] failed after {attempts} attempt(s): {cause}")


class LLMTransientError(DocExtractorError):
    """Retryable provider error (timeouts, rate limits, 5xx)."""


class LLMPermanentError(DocExtractorError):
    """Non-retryable provider error (bad request, invalid JSON after retry)."""
