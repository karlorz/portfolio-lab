"""Unit tests for src.strategy.ensemble_voter_collect CollectMixin."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.signals.regime_spec import Regime, SignalReading
from src.signals.signal_source import SignalSource
from src.strategy.ensemble_voter_collect import CollectMixin


class DummyCollector(CollectMixin):
    """Test harness incorporating CollectMixin."""

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.db_path = data_path / "ensemble_signals.db"
        self.signal_aggregator = None
        self.current_readings = None

        # Thresholds matching EnsembleVoter
        self.CRISIS_VOL_THRESHOLD = 0.40
        self.CRISIS_DRAWDOWN_THRESHOLD = -0.15
        self.HIGH_VOL_VOL_THRESHOLD = 0.25
        self.HIGH_VOL_DRAWDOWN_THRESHOLD = -0.08
        self.HIGH_VOL_MOM_THRESHOLD = -0.05
        self.RECOVERY_DRAWDOWN_THRESHOLD = -0.08
        self.RECOVERY_MOM_THRESHOLD = 0.02
        self.LOW_VOL_VOL_THRESHOLD = 0.12
        self.LOW_VOL_MOM_THRESHOLD = 0.01


class TestInitDb:
    """Test _init_db table creation."""

    def test_creates_tables(self, tmp_path: Path):
        collector = DummyCollector(tmp_path)
        collector._init_db()
        assert collector.db_path.is_file()

        with sqlite3.connect(collector.db_path) as conn:
            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
        assert "ensemble_votes" in tables
        assert "source_readings" in tables


class TestDetectRegime:
    """Test detect_regime heuristics."""

    def test_empty_df_returns_normal_default(self, tmp_path: Path):
        collector = DummyCollector(tmp_path)
        assert collector.detect_regime(pd.DataFrame()) == (Regime.NORMAL, 0.5)

    def test_none_df_loads_or_returns_normal(self, tmp_path: Path, monkeypatch):
        collector = DummyCollector(tmp_path)
        monkeypatch.setattr(collector, "_load_price_data", lambda: None)
        assert collector.detect_regime(None) == (Regime.NORMAL, 0.5)

    def test_short_df_returns_normal_default(self, tmp_path: Path):
        collector = DummyCollector(tmp_path)
        short_df = pd.DataFrame({"SPY": [100.0] * 10})
        assert collector.detect_regime(short_df) == (Regime.NORMAL, 0.5)

    def test_crisis_regime_on_high_drawdown(self, tmp_path: Path):
        collector = DummyCollector(tmp_path)
        # 30 days of falling prices creating > 15% drawdown
        prices = [100.0 * (0.99 ** i) for i in range(30)]
        df = pd.DataFrame({"SPY": prices})
        regime, conf = collector.detect_regime(df)
        assert regime == Regime.CRISIS
        assert 0.5 <= conf <= 0.9

    def test_low_vol_regime_on_low_vol_positive_mom(self, tmp_path: Path):
        collector = DummyCollector(tmp_path)
        # Steady slow upward march (very low vol, positive momentum)
        prices = [100.0 + i * 0.1 for i in range(30)]
        df = pd.DataFrame({"SPY": prices})
        regime, conf = collector.detect_regime(df)
        assert regime == Regime.LOW_VOL
        assert conf >= 0.5


class TestSignalAggregatorDelegation:
    """Test _ensure_signal_aggregator and collection methods."""

    def test_ensure_signal_aggregator_creates_default(self, tmp_path: Path):
        collector = DummyCollector(tmp_path)
        agg = collector._ensure_signal_aggregator()
        assert agg is not None
        assert collector.signal_aggregator is agg

    def test_collect_signals_delegates_and_stores_current_readings(self, tmp_path: Path):
        collector = DummyCollector(tmp_path)
        mock_agg = MagicMock()
        mock_readings = {
            SignalSource.MULTI_SPEED_MOM: SignalReading(
                source=SignalSource.MULTI_SPEED_MOM,
                timestamp="2026-08-17T12:00:00Z",
                value=0.5,
                confidence=0.8,
                weight=0.2,
                regime_fit="normal",
            )
        }
        mock_agg.collect.return_value = mock_readings
        collector.signal_aggregator = mock_agg

        readings = collector.collect_signals(date="2026-08-17", regime=Regime.NORMAL)
        assert readings == mock_readings
        assert collector.current_readings == mock_readings
        mock_agg.collect.assert_called_once_with(date="2026-08-17", regime=Regime.NORMAL)


class TestStaticZeroBaselineSources:
    """Test _static_zero_baseline_sources and _pin_zero_baseline_weights."""

    def test_static_zero_baseline_sources_returns_set(self):
        zeros = DummyCollector._static_zero_baseline_sources("CRISIS")
        assert isinstance(zeros, set)

    def test_pin_zero_baseline_weights_preserves_non_zeros(self):
        weights = {"s1": 0.5, "s2": 0.5}
        pinned = DummyCollector._pin_zero_baseline_weights(weights, "NORMAL")
        assert sum(pinned.values()) == pytest.approx(1.0)
