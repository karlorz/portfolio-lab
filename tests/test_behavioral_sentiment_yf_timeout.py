"""G6: every yfinance ``Ticker.history(...)`` call is explicitly time-bounded.

Regression: unbounded yfinance calls could stall scheduled jobs past their
deadline. The shipped fetchers must (a) pass an explicit ``timeout=`` kwarg on
every call and (b) degrade to their documented fallback when the network
stalls/fails (simulated here with a fake ticker whose ``history()`` raises),
instead of hanging or propagating.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.data.behavioral_sentiment_fetcher import BehavioralSentimentFetcher

TIMEOUT_KWARG = "timeout"


class FakeTicker:
    """Records history() kwargs; returns a canned frame or raises ``fail``."""

    def __init__(self, *, fail: Exception | None = None, closes: list[float] | None = None):
        self.calls: list[dict] = []
        self._fail = fail
        self._closes = closes if closes is not None else [100.0]

    def history(self, **kwargs):
        self.calls.append(kwargs)
        if self._fail is not None:
            raise self._fail
        frame = pd.DataFrame({"Close": self._closes})
        frame.index = pd.DatetimeIndex([datetime(2026, 8, 7)] * len(self._closes), name="Date")
        return frame


@pytest.fixture
def fetcher(tmp_path, monkeypatch):
    """Fetcher on a throwaway cache DB with yfinance replaced by FakeTicker."""

    def _install(fake: FakeTicker) -> BehavioralSentimentFetcher:
        monkeypatch.setattr(
            "src.data.behavioral_sentiment_fetcher.yf.Ticker",
            lambda _symbol: fake,
        )
        return BehavioralSentimentFetcher(cache_db=tmp_path / "bs_test.db")

    return _install


def test_fetch_yf_passes_explicit_timeout(fetcher):
    fake = FakeTicker()
    f = fetcher(fake)

    value = f._fetch_yf("^VIX", default=16.0)

    assert value == 100.0
    assert len(fake.calls) == 1
    assert fake.calls[0]["period"] == "1d"
    assert TIMEOUT_KWARG in fake.calls[0], "history() must pass an explicit timeout="
    assert fake.calls[0][TIMEOUT_KWARG] > 0


def test_fetch_yf_stall_degrades_to_default(fetcher, monkeypatch):
    """A stalled network (TimeoutError) must not hang: fallback is returned."""
    # Keep tenacity backoff sleeps instant so the failure path runs fast.
    monkeypatch.setattr("tenacity.nap.sleep", lambda _seconds: None)
    fake = FakeTicker(fail=TimeoutError("stalled"))
    f = fetcher(fake)

    value = f._fetch_yf("^VIX", default=16.0)

    assert value == 16.0
    assert f._yf_cache["^VIX"][0] == 16.0


def test_fetch_yf_other_failures_degrade_to_default(fetcher):
    fake = FakeTicker(fail=RuntimeError("boom"))
    f = fetcher(fake)

    assert f._fetch_yf("^SKEW", default=0.0) == 0.0


def test_fetch_put_call_ratio_passes_timeout(fetcher):
    fake = FakeTicker(closes=[0.6, 0.7])
    f = fetcher(fake)

    value = f._fetch_put_call_ratio()

    assert value == pytest.approx(0.65)
    assert fake.calls[0]["period"] == "5d"
    assert TIMEOUT_KWARG in fake.calls[0], "history() must pass an explicit timeout="


def test_fetch_put_call_ratio_stall_degrades_to_065(fetcher):
    """P/C fallback is the documented 0.65 historical average."""
    fake = FakeTicker(fail=TimeoutError("stalled"))
    f = fetcher(fake)

    assert f._fetch_put_call_ratio() == 0.65
    assert f._yf_cache["^CPCE"][0] == 0.65


def test_fetch_yf_timeout_is_not_retried(fetcher, monkeypatch):
    """A stalled endpoint must not be retried: one attempt, fallback value.

    G6 follow-up (2026-08-11): 13 consecutive hourly data-job timeouts traced
    to yfinance stalls. Each stalled history() burns its 10s timeout; retrying
    a stalling endpoint just burns the budget again (2 attempts + backoff ×
    4 tickers ≈ 84-120s of the 300s job budget). Timeouts therefore fail fast
    to the documented fallback; only fast transient errors retry.
    """
    monkeypatch.setattr("tenacity.nap.sleep", lambda _seconds: None)
    fake = FakeTicker(fail=TimeoutError("stalled"))
    f = fetcher(fake)

    value = f._fetch_yf("^VIX", default=16.0)

    assert value == 16.0
    assert len(fake.calls) == 1, "timeout must not trigger a retry (budget burn)"


def test_fetch_yf_transient_error_degrades_single_attempt(fetcher, monkeypatch):
    """Fast connection errors degrade to the fallback in one attempt.

    The former ``retry_on_api_error`` decorator could never fire here: every
    exception it would retry (OSError family) is caught inside ``_fetch_yf``,
    which degrades to the documented default. The dead decorator was removed;
    this test locks the real shipped behavior — bounded single attempt, no
    hidden retry multiplication in the hourly job budget.
    """
    monkeypatch.setattr("tenacity.nap.sleep", lambda _seconds: None)
    fake = FakeTicker(fail=ConnectionError("connection refused"))
    f = fetcher(fake)

    value = f._fetch_yf("^VIX", default=16.0)

    assert value == 16.0
    assert len(fake.calls) == 1, "single bounded attempt, fallback on failure"
