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
"""

import logging
import logging.config
import os
import sys

__all__ = ["configure_logging"]


def configure_logging(level: str | None = None) -> None:
    """Configure the root logger for the application.

    Parameters
    ----------
    level : str, optional
        Override log level.  Defaults to the ``LOG_LEVEL`` environment
        variable, falling back to ``"INFO"``.
    """
    effective_level = level or os.environ.get("LOG_LEVEL", "INFO").upper()

    # Use dictConfig for a single, authoritative configuration.
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "brief": {
                "format": "%(levelname)s | %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": sys.stderr,
                "formatter": "standard",
                "level": effective_level,
            },
        },
        "root": {
            "level": effective_level,
            "handlers": ["console"],
        },
    })
