"""structlog configuration. Call `configure_logging` once per process."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_CONFIGURED = False


class _LazyStderrLogger:
    """Writes each rendered line to *the current* `sys.stderr`.

    structlog's `PrintLogger` captures the stream once, which breaks when a test
    harness (pytest capsys) swaps and closes `sys.stderr` between tests.
    """

    def msg(self, message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    log = debug = info = warning = warn = error = critical = exception = fatal = msg


class _LazyStderrLoggerFactory:
    def __call__(self, *args: Any) -> _LazyStderrLogger:
        return _LazyStderrLogger()


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    global _CONFIGURED
    renderer: Any
    if fmt == "json":
        renderer = structlog.processors.JSONRenderer(sort_keys=True)
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        logger_factory=_LazyStderrLoggerFactory(),
        cache_logger_on_first_use=False,
    )
    _CONFIGURED = True


def get_logger(**initial: Any) -> structlog.stdlib.BoundLogger:
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger().bind(**initial)


def bind_context(**kwargs: Any) -> None:
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()
