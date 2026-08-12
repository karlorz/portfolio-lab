"""Tests for rate limiting and retry utilities."""

import time

import pytest

from src.utils.rate_limiter import (
    RateLimiter,
    retry_on_api_error,
    rate_limited,
)


class TestRateLimiter:
    """Test RateLimiter token-bucket implementation."""

    def test_no_wait_when_rate_not_exceeded(self):
        """First request should not wait."""
        limiter = RateLimiter(max_rpm=60)
        start = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1  # Should be near-instant

    def test_wait_between_requests(self):
        """Second request immediately after first should wait."""
        limiter = RateLimiter(max_rpm=60)  # 1 request per second
        limiter.wait()
        start = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - start
        # Should wait ~1s between requests at 60 RPM
        assert elapsed >= 0.8  # Allow some tolerance

    def test_custom_rpm(self):
        """Custom RPM should adjust interval."""
        limiter = RateLimiter(max_rpm=120)  # 0.5s interval
        assert limiter._interval == pytest.approx(0.5, abs=0.01)

    def test_env_var_override(self, monkeypatch):
        """API_RATE_LIMIT_RPM env var should override default."""
        monkeypatch.setenv("API_RATE_LIMIT_RPM", "30")
        limiter = RateLimiter()
        assert limiter._max_rpm == 30
        assert limiter._interval == pytest.approx(2.0, abs=0.01)


class TestRetryDecorator:
    """Test retry_on_api_error decorator."""

    def test_no_retry_on_success(self):
        """Successful call should not retry."""
        call_count = 0

        @retry_on_api_error(max_retries=3)
        def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = succeed()
        assert result == "ok"
        assert call_count == 1

    def test_retry_on_connection_error(self):
        """Should retry on ConnectionError and eventually succeed."""
        call_count = 0

        @retry_on_api_error(max_retries=3, backoff=0.01)
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("timeout")
            return "ok"

        result = flaky()
        assert result == "ok"
        assert call_count == 3

    def test_reraise_after_max_retries(self):
        """Should reraise after exhausting retries."""
        @retry_on_api_error(max_retries=2, backoff=0.01)
        def always_fails():
            raise ConnectionError("down")

        with pytest.raises(ConnectionError, match="down"):
            always_fails()

    def test_no_retry_on_non_retryable(self):
        """Should not retry on non-retryable exceptions (e.g., ValueError)."""
        call_count = 0

        @retry_on_api_error(max_retries=3, backoff=0.01)
        def bad_input():
            nonlocal call_count
            call_count += 1
            raise ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            bad_input()
        assert call_count == 1  # No retry


class TestRateLimitedDecorator:
    """Test rate_limited decorator."""

    def test_yahoo_rate_limit(self):
        """@rate_limited("yahoo") should call the function."""
        @rate_limited("yahoo")
        def fetch(symbol):
            return f"data:{symbol}"

        result = fetch("SPY")
        assert result == "data:SPY"

    def test_fred_rate_limit(self):
        """@rate_limited("fred") should call the function."""
        @rate_limited("fred")
        def fetch_fred(series_id):
            return f"fred:{series_id}"

        result = fetch_fred("GDP")
        assert result == "fred:GDP"

    def test_unknown_api_no_rate_limit(self):
        """Unknown API name should not rate-limit (pass-through)."""
        @rate_limited("unknown")
        def fetch(symbol):
            return f"data:{symbol}"

        result = fetch("SPY")
        assert result == "data:SPY"
