"""
Rate limiting and retry utilities for external API calls.

Wraps Yahoo Finance, FRED, and other external API calls with
exponential backoff retry and rate limiting.

Environment variables
---------------------
API_RETRY_MAX : int
    Maximum number of retries (default: 3).
API_RETRY_BACKOFF : float
    Base backoff in seconds (default: 1.0).
API_RATE_LIMIT_RPM : int
    Max requests per minute (default: 60).
"""

import os
import time
import threading
from functools import wraps
from typing import TypeVar, Callable, Any

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

import logging

logger = logging.getLogger(__name__)

__all__ = [
    "retry_on_api_error",
    "rate_limited",
    "RateLimiter",
]

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

_RETRY_MAX = int(os.environ.get("API_RETRY_MAX", "3"))
_RETRY_BACKOFF = float(os.environ.get("API_RETRY_BACKOFF", "1.0"))


def retry_on_api_error(
    max_retries: int = _RETRY_MAX,
    backoff: float = _RETRY_BACKOFF,
    retryable_exceptions: tuple = (ConnectionError, TimeoutError, OSError),
):
    """Decorator that retries API calls with exponential backoff.

    Combines with the existing pybreaker circuit breaker for full resilience:
    - tenacity handles transient errors (connection, timeout)
    - pybreaker handles sustained failures (circuit open)

    Usage::

        @retry_on_api_error()
        def fetch_data(symbol):
            return yf.Ticker(symbol).history(period="1d")
    """
    return retry(
        retry=retry_if_exception_type(retryable_exceptions),
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=backoff, min=backoff, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Simple token-bucket rate limiter for API calls.

    Thread-safe.  Usage::

        limiter = RateLimiter(max_rpm=60)
        limiter.wait()
        response = requests.get(url)
    """

    def __init__(self, max_rpm: int | None = None):
        self._max_rpm = max_rpm or int(os.environ.get("API_RATE_LIMIT_RPM", "60"))
        self._interval = 60.0 / self._max_rpm  # seconds between tokens
        self._last_request = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Block until a request slot is available."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request
            if elapsed < self._interval:
                sleep_time = self._interval - elapsed
                logger.debug("Rate limiter: sleeping %.2fs", sleep_time)
                time.sleep(sleep_time)
            self._last_request = time.monotonic()


# Module-level rate limiter instances
_yahoo_limiter = RateLimiter(max_rpm=60)   # Yahoo Finance: ~60 req/min
_fred_limiter = RateLimiter(max_rpm=30)    # FRED API: ~30 req/min


def rate_limited(api: str = "yahoo"):
    """Decorator that rate-limits API calls.

    Usage::

        @rate_limited("yahoo")
        def fetch_yahoo(symbol):
            return yf.Ticker(symbol).history()
    """
    limiter_map = {
        "yahoo": _yahoo_limiter,
        "fred": _fred_limiter,
    }
    limiter = limiter_map.get(api)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            if limiter:
                limiter.wait()
            return func(*args, **kwargs)
        return wrapper
    return decorator
