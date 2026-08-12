#!/usr/bin/env python3
"""
Direct unit tests for ``src/signals/sample_counts.py`` ``get_live_sample_count``
(C1 provenance helper, 12L) — test file owed by the TEST-GAP coverage gap
(module has zero direct test references; indirect coverage via test_generator
only).

Covers the live-path contract with a monkeypatched ``price_cache.get_prices``:
list → length, missing ticker → 0, empty/non-list → 0, raised read errors
(OSError/TypeError/ValueError) → 0.
"""
from src.signals.sample_counts import get_live_sample_count


def test_live_sample_count_list(monkeypatch):
    """List history → its length."""
    monkeypatch.setattr(
        "src.data.price_cache.get_prices", lambda: {"SPY": [1, 2, 3, 4]}
    )
    assert get_live_sample_count("SPY") == 4


def test_live_sample_count_missing_ticker(monkeypatch):
    """Unknown ticker → 0 (missing key defaults to empty list)."""
    monkeypatch.setattr(
        "src.data.price_cache.get_prices", lambda: {"SPY": [1, 2]}
    )
    assert get_live_sample_count("GLD") == 0


def test_live_sample_count_empty_list(monkeypatch):
    """Empty history → 0."""
    monkeypatch.setattr("src.data.price_cache.get_prices", lambda: {"SPY": []})
    assert get_live_sample_count("SPY") == 0


def test_live_sample_count_non_list(monkeypatch):
    """Non-list payload for the ticker → 0 (defensive)."""
    monkeypatch.setattr(
        "src.data.price_cache.get_prices", lambda: {"SPY": "not-a-list"}
    )
    assert get_live_sample_count("SPY") == 0


def test_live_sample_count_read_error_returns_zero(monkeypatch):
    """Prices read failure (OSError) → 0, never propagates."""
    def boom():
        raise OSError("prices.json missing")

    monkeypatch.setattr("src.data.price_cache.get_prices", boom)
    assert get_live_sample_count("SPY") == 0
