"""
Centralized logging configuration for portfolio-lab.

Load once at application entry points (CLI scripts, cron jobs) via::

    from src.utils.log_config import configure_logging
    configure_logging()

This replaces scattered ``logging.basicConfig()`` calls.  Module-level
loggers should use the standard pattern::

    import logging
    logger = logging.getLogger(__name__)

Individual modules should **never** call ``logging.basicConfig()`` —
the root logger is configured here.

Environment variables
---------------------
LOG_LEVEL : str
    Override log level.  Defaults to ``"INFO"``.
JSON_LOGS : str
    Set to ``"1"`` to enable JSON-structured output via python-json-logger.
CRON_RUN_ID : str
    Optional correlation ID injected into every log record for
    cron pipeline tracing.
"""

import logging
import logging.config
import os
import sys
import uuid

__all__ = ["configure_logging"]


class CorrelationIDFilter(logging.Filter):
    """Inject CRON_RUN_ID (or a generated UUID) into every log record.

    This enables tracing all log messages from a single cron pipeline run.
    """

    def __init__(self, cron_run_id: str | None = None):
        super().__init__()
        self._cron_run_id = cron_run_id or os.environ.get(
            "CRON_RUN_ID", ""
        )

    def filter(self, record: logging.LogRecord) -> bool:
        record.cron_run_id = self._cron_run_id  # type: ignore[attr-defined]
        return True


def configure_logging(level: str | None = None) -> None:
    """Configure the root logger for the application.

    Parameters
    ----------
    level : str, optional
        Override log level.  Defaults to the ``LOG_LEVEL`` environment
        variable, falling back to ``"INFO"``.
    """
    effective_level = level or os.environ.get("LOG_LEVEL", "INFO").upper()

    # Determine formatter
    use_json = os.environ.get("JSON_LOGS", "").strip() in ("1", "true", "yes")

    if use_json:
        try:
            from pythonjsonlogger.json import JsonFormatter

            formatter_config = {
                "()": "pythonjsonlogger.json.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s %(cron_run_id)s",
                "rename_fields": {
                    "asctime": "timestamp",
                    "levelname": "level",
                    "name": "logger",
                },
                "static_fields": {
                    "service": "portfolio-lab",
                },
            }
        except ImportError:
            use_json = False

    if not use_json:
        formatter_config = {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }

    # Use dictConfig for a single, authoritative configuration.
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": formatter_config,
            "brief": {
                "format": "%(levelname)s | %(message)s",
            },
        },
        "filters": {
            "correlation_id": {
                "()": f"{__name__}.CorrelationIDFilter",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": sys.stderr,
                "formatter": "standard",
                "level": effective_level,
                "filters": ["correlation_id"],
            },
        },
        "root": {
            "level": effective_level,
            "handlers": ["console"],
        },
    })
