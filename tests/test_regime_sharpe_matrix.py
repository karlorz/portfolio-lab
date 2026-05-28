"""Tests for src/monitor/regime_sharpe_matrix.py — data-driven regime gating.

Tests the computation of per-signal, per-regime Sharpe ratios with
stationary bootstrap significance testing, and wiring into RegimeGate.
"""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """Fixed RNG for reproducible tests."""
    return np.random.RandomState(42)


@pytest.fixture
def sample_returns(rng):
    """750 days of synthetic signal returns across 3 regimes (250 each)."""
    n_per = 250
    n = n_per * 3
    dates = pd.bdate_range("2020-01-01", periods=n)
    regimes = ["NORMAL"] * n_per + ["HIGH_VOL"] * n_per + ["CRISIS"] * n_per
    # Signal returns: clear directional drift per regime
    returns = np.concatenate([
        rng.normal(0.002, 0.01, n_per),   # NORMAL: strong positive drift
        rng.normal(0.0, 0.02, n_per),      # HIGH_VOL: zero drift, high vol
        rng.normal(-0.003, 0.025, n_per),  # CRISIS: strong negative drift
    ])
    return pd.DataFrame({
        "date": dates,
        "signal": "alternative_data",
        "regime": regimes,
        "daily_return": returns,
    }).set_index("date")


@pytest.fixture
def multi_signal_returns(rng):
    """500 days of returns for 3 signals across 3 regimes."""
    n = 500
    dates = pd.bdate_range("2018-01-01", periods=n)
    regimes = ["NORMAL"] * 280 + ["HIGH_VOL"] * 120 + ["CRISIS"] * 100

    signals = {}
    for sig_name, drift, vol in [
        ("alternative_data", 0.001, 0.01),
        ("international_momentum", 0.0005, 0.015),
        ("cross_asset_rv", -0.0005, 0.012),
    ]:
        returns = np.concatenate([
            rng.normal(drift, vol, 280),
            rng.normal(drift * 0.5, vol * 1.5, 120),
            rng.normal(-abs(drift), vol * 2.5, 100),
        ])
        signals[sig_name] = returns

    rows = []
    for sig_name, returns in signals.items():
        for i, (d, r, reg) in enumerate(zip(dates, returns, regimes)):
            rows.append({
                "date": d,
                "signal": sig_name,
                "regime": reg,
                "daily_return": r,
            })
    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------------------
# Tests: compute_sharpe
# ---------------------------------------------------------------------------

class TestComputeSharpe:
    """Test the basic Sharpe ratio computation."""

    def test_positive_sharpe_for_positive_returns(self):
        from src.monitor.regime_sharpe_matrix import compute_sharpe
        returns = pd.Series(np.random.RandomState(0).normal(0.001, 0.01, 100))
        sharpe = compute_sharpe(returns)
        assert sharpe > 0

    def test_negative_sharpe_for_negative_returns(self):
        from src.monitor.regime_sharpe_matrix import compute_sharpe
        returns = pd.Series(np.random.RandomState(0).normal(-0.002, 0.01, 100))
        sharpe = compute_sharpe(returns)
        assert sharpe < 0

    def test_zero_sharpe_for_zero_mean(self):
        from src.monitor.regime_sharpe_matrix import compute_sharpe
        # Use large sample to ensure CLT convergence
        returns = pd.Series(np.random.RandomState(12345).normal(0.0, 0.01, 5000))
        sharpe = compute_sharpe(returns)
        assert abs(sharpe) < 0.15  # should be near zero with 5000 samples

    def test_annualization_factor(self):
        from src.monitor.regime_sharpe_matrix import compute_sharpe
        # Daily mean=0.001, std=0.01 → annualized Sharpe ≈ 0.001/0.01 * sqrt(252) ≈ 1.59
        returns = pd.Series(np.random.RandomState(0).normal(0.001, 0.01, 500))
        sharpe = compute_sharpe(returns)
        assert 1.0 < sharpe < 2.0

    def test_insufficient_observations_returns_nan(self):
        from src.monitor.regime_sharpe_matrix import compute_sharpe
        returns = pd.Series(np.random.RandomState(0).normal(0.001, 0.01, 10))
        sharpe = compute_sharpe(returns, min_obs=30)
        assert np.isnan(sharpe)

    def test_custom_risk_free_rate(self):
        from src.monitor.regime_sharpe_matrix import compute_sharpe
        returns = pd.Series(np.random.RandomState(0).normal(0.001, 0.01, 100))
        sharpe_no_rf = compute_sharpe(returns, risk_free_rate=0.0)
        sharpe_high_rf = compute_sharpe(returns, risk_free_rate=0.10)
        assert sharpe_no_rf > sharpe_high_rf

    def test_empty_series_returns_nan(self):
        from src.monitor.regime_sharpe_matrix import compute_sharpe
        sharpe = compute_sharpe(pd.Series(dtype=float))
        assert np.isnan(sharpe)


# ---------------------------------------------------------------------------
# Tests: compute_hit_rate
# ---------------------------------------------------------------------------

class TestComputeHitRate:
    """Test directional hit rate computation."""

    def test_perfect_hit_rate(self):
        from src.monitor.regime_sharpe_matrix import compute_hit_rate
        signals = pd.Series([1.0, 1.0, 1.0, 1.0])
        returns = pd.Series([0.01, 0.02, 0.01, 0.03])
        assert compute_hit_rate(signals, returns) == 1.0

    def test_zero_hit_rate(self):
        from src.monitor.regime_sharpe_matrix import compute_hit_rate
        signals = pd.Series([1.0, 1.0, 1.0, 1.0])
        returns = pd.Series([-0.01, -0.02, -0.01, -0.03])
        assert compute_hit_rate(signals, returns) == 0.0

    def test_mixed_hit_rate(self):
        from src.monitor.regime_sharpe_matrix import compute_hit_rate
        signals = pd.Series([1.0, -1.0, 1.0, -1.0])
        returns = pd.Series([0.01, -0.02, -0.01, 0.03])
        # signs match: +/+ → yes, -/- → yes, +/- → no, -/+ → no
        assert compute_hit_rate(signals, returns) == 0.5

    def test_zero_returns_not_counted(self):
        from src.monitor.regime_sharpe_matrix import compute_hit_rate
        signals = pd.Series([1.0, 1.0, 0.0, 1.0])
        returns = pd.Series([0.01, 0.0, 0.01, 0.02])
        # Only first and last have nonzero returns → both match
        hr = compute_hit_rate(signals, returns)
        assert hr == 1.0


# ---------------------------------------------------------------------------
# Tests: bootstrap_sharpe_ci
# ---------------------------------------------------------------------------

class TestBootstrapSharpeCI:
    """Test stationary bootstrap for Sharpe confidence intervals."""

    def test_positive_sharpe_high_p_positive(self, rng):
        from src.monitor.regime_sharpe_matrix import bootstrap_sharpe_ci
        returns = pd.Series(rng.normal(0.002, 0.01, 200))
        result = bootstrap_sharpe_ci(returns, n_bootstrap=1000, seed=42)
        assert result["p_positive"] > 0.80

    def test_negative_sharpe_low_p_positive(self, rng):
        from src.monitor.regime_sharpe_matrix import bootstrap_sharpe_ci
        returns = pd.Series(rng.normal(-0.003, 0.01, 200))
        result = bootstrap_sharpe_ci(returns, n_bootstrap=1000, seed=42)
        assert result["p_positive"] < 0.20

    def test_ci_contains_true_sharpe(self, rng):
        from src.monitor.regime_sharpe_matrix import bootstrap_sharpe_ci
        returns = pd.Series(rng.normal(0.001, 0.01, 500))
        result = bootstrap_sharpe_ci(returns, n_bootstrap=2000, seed=42)
        true_sharpe = 0.001 / 0.01 * np.sqrt(252)
        assert result["ci_95_low"] < true_sharpe < result["ci_95_high"]

    def test_insufficient_data_returns_nan(self, rng):
        from src.monitor.regime_sharpe_matrix import bootstrap_sharpe_ci
        returns = pd.Series(rng.normal(0.001, 0.01, 5))
        result = bootstrap_sharpe_ci(returns, n_bootstrap=100, min_obs=30, seed=42)
        assert np.isnan(result["p_positive"])
        assert np.isnan(result["ci_95_low"])

    def test_deterministic_with_seed(self, rng):
        from src.monitor.regime_sharpe_matrix import bootstrap_sharpe_ci
        returns = pd.Series(rng.normal(0.001, 0.01, 100))
        r1 = bootstrap_sharpe_ci(returns, n_bootstrap=500, seed=123)
        r2 = bootstrap_sharpe_ci(returns, n_bootstrap=500, seed=123)
        assert r1["p_positive"] == r2["p_positive"]
        assert r1["ci_95_low"] == r2["ci_95_low"]


# ---------------------------------------------------------------------------
# Tests: compute_regime_sharpe_matrix
# ---------------------------------------------------------------------------

class TestComputeRegimeSharpeMatrix:
    """Test the full matrix computation pipeline."""

    def test_returns_dict_of_dicts(self, sample_returns):
        from src.monitor.regime_sharpe_matrix import compute_regime_sharpe_matrix
        matrix = compute_regime_sharpe_matrix(sample_returns, n_bootstrap=100, seed=42)
        assert isinstance(matrix, dict)
        assert "alternative_data" in matrix
        assert "NORMAL" in matrix["alternative_data"]

    def test_entries_have_required_fields(self, sample_returns):
        from src.monitor.regime_sharpe_matrix import compute_regime_sharpe_matrix
        matrix = compute_regime_sharpe_matrix(sample_returns, n_bootstrap=100, seed=42)
        entry = matrix["alternative_data"]["NORMAL"]
        assert hasattr(entry, "sharpe")
        assert hasattr(entry, "hit_rate")
        assert hasattr(entry, "n_obs")
        assert hasattr(entry, "p_positive")
        assert hasattr(entry, "ci_95_low")
        assert hasattr(entry, "ci_95_high")

    def test_normal_regime_positive_sharpe(self, sample_returns):
        from src.monitor.regime_sharpe_matrix import compute_regime_sharpe_matrix
        matrix = compute_regime_sharpe_matrix(sample_returns, n_bootstrap=100, seed=42)
        entry = matrix["alternative_data"]["NORMAL"]
        assert entry.sharpe > 0

    def test_crisis_regime_negative_sharpe(self, sample_returns):
        from src.monitor.regime_sharpe_matrix import compute_regime_sharpe_matrix
        matrix = compute_regime_sharpe_matrix(sample_returns, n_bootstrap=100, seed=42)
        entry = matrix["alternative_data"]["CRISIS"]
        assert entry.sharpe < 0

    def test_multi_signal_matrix(self, multi_signal_returns):
        from src.monitor.regime_sharpe_matrix import compute_regime_sharpe_matrix
        matrix = compute_regime_sharpe_matrix(multi_signal_returns, n_bootstrap=100, seed=42)
        assert len(matrix) == 3
        for sig in ["alternative_data", "international_momentum", "cross_asset_rv"]:
            assert sig in matrix
            for regime in ["NORMAL", "HIGH_VOL", "CRISIS"]:
                assert regime in matrix[sig]

    def test_min_obs_gating(self, rng):
        from src.monitor.regime_sharpe_matrix import compute_regime_sharpe_matrix
        # Create data where one regime has only 10 observations
        dates = pd.bdate_range("2020-01-01", periods=110)
        regimes = ["NORMAL"] * 100 + ["RECOVERY"] * 10
        returns = rng.normal(0.001, 0.01, 110)
        df = pd.DataFrame({
            "signal": "test_signal",
            "regime": regimes,
            "daily_return": returns,
        }, index=dates)
        matrix = compute_regime_sharpe_matrix(df, n_bootstrap=100, min_obs=30, seed=42)
        # RECOVERY should have n_obs=10, below min_obs=30
        entry = matrix["test_signal"]["RECOVERY"]
        assert entry.n_obs == 10
        assert np.isnan(entry.sharpe) or entry.p_positive == 0.0

    def test_empty_dataframe_returns_empty(self):
        from src.monitor.regime_sharpe_matrix import compute_regime_sharpe_matrix
        df = pd.DataFrame(columns=["signal", "regime", "daily_return"])
        matrix = compute_regime_sharpe_matrix(df, n_bootstrap=100, seed=42)
        assert matrix == {}


# ---------------------------------------------------------------------------
# Tests: derive_gate_rules
# ---------------------------------------------------------------------------

class TestDeriveGateRules:
    """Test converting Sharpe matrix to RegimeGate-compatible rules."""

    def test_gates_off_negative_sharpe(self):
        from src.monitor.regime_sharpe_matrix import derive_gate_rules, RegimeSharpeEntry
        matrix = {
            "signal_a": {
                "NORMAL": RegimeSharpeEntry("signal_a", "NORMAL", 0.8, 0.55, 0.05, 200, 0.98, 0.3, 1.3),
                "CRISIS": RegimeSharpeEntry("signal_a", "CRISIS", -0.5, 0.42, -0.03, 100, 0.10, -1.2, 0.2),
            }
        }
        rules = derive_gate_rules(matrix, p_threshold=0.90)
        assert "signal_a" in rules
        assert "CRISIS" in rules["signal_a"]
        assert "NORMAL" not in rules["signal_a"]

    def test_keeps_signal_on_when_significant(self):
        from src.monitor.regime_sharpe_matrix import derive_gate_rules, RegimeSharpeEntry
        matrix = {
            "signal_b": {
                "NORMAL": RegimeSharpeEntry("signal_b", "NORMAL", 1.0, 0.58, 0.06, 300, 0.99, 0.5, 1.5),
                "LOW_VOL": RegimeSharpeEntry("signal_b", "LOW_VOL", 0.3, 0.52, 0.02, 200, 0.85, -0.1, 0.7),
            }
        }
        rules = derive_gate_rules(matrix, p_threshold=0.90)
        # LOW_VOL has p_positive=0.85 < 0.90 → gate OFF
        assert "LOW_VOL" in rules.get("signal_b", set())
        # NORMAL has p_positive=0.99 → not in rules
        assert "NORMAL" not in rules.get("signal_b", set())

    def test_skips_insufficient_data(self):
        from src.monitor.regime_sharpe_matrix import derive_gate_rules, RegimeSharpeEntry
        matrix = {
            "signal_c": {
                "RECOVERY": RegimeSharpeEntry("signal_c", "RECOVERY", np.nan, np.nan, np.nan, 10, np.nan, np.nan, np.nan),
            }
        }
        rules = derive_gate_rules(matrix, p_threshold=0.90)
        # Insufficient data → no rules generated for this signal/regime
        assert "signal_c" not in rules or "RECOVERY" not in rules.get("signal_c", set())

    def test_returns_dict_of_sets(self):
        from src.monitor.regime_sharpe_matrix import derive_gate_rules, RegimeSharpeEntry
        matrix = {
            "sig": {
                "R1": RegimeSharpeEntry("sig", "R1", -1.0, 0.3, -0.1, 100, 0.05, -2.0, 0.0),
            }
        }
        rules = derive_gate_rules(matrix)
        assert isinstance(rules, dict)
        assert isinstance(rules["sig"], set)


# ---------------------------------------------------------------------------
# Tests: derive_regime_weight_multipliers
# ---------------------------------------------------------------------------

class TestDeriveRegimeWeightMultipliers:
    """Test converting Sharpe matrix to soft weight multipliers."""

    def test_baseline_multiplier_for_normal(self):
        from src.monitor.regime_sharpe_matrix import derive_regime_weight_multipliers, RegimeSharpeEntry
        matrix = {
            "signal_a": {
                "NORMAL": RegimeSharpeEntry("signal_a", "NORMAL", 0.5, 0.55, 0.03, 300, 0.95, 0.1, 0.9),
            }
        }
        multipliers = derive_regime_weight_multipliers(matrix)
        # Moderate Sharpe → multiplier near 1.0
        assert 0.8 <= multipliers["signal_a"]["NORMAL"] <= 1.2

    def test_high_sharpe_gets_boost(self):
        from src.monitor.regime_sharpe_matrix import derive_regime_weight_multipliers, RegimeSharpeEntry
        matrix = {
            "signal_a": {
                "NORMAL": RegimeSharpeEntry("signal_a", "NORMAL", 1.5, 0.62, 0.08, 300, 0.99, 0.8, 2.2),
            }
        }
        multipliers = derive_regime_weight_multipliers(matrix)
        assert multipliers["signal_a"]["NORMAL"] > 1.0

    def test_negative_sharpe_gets_penalty(self):
        from src.monitor.regime_sharpe_matrix import derive_regime_weight_multipliers, RegimeSharpeEntry
        matrix = {
            "signal_a": {
                "CRISIS": RegimeSharpeEntry("signal_a", "CRISIS", -0.5, 0.40, -0.04, 100, 0.08, -1.2, 0.2),
            }
        }
        multipliers = derive_regime_weight_multipliers(matrix)
        assert multipliers["signal_a"]["CRISIS"] < 1.0

    def test_multiplier_capped(self):
        from src.monitor.regime_sharpe_matrix import derive_regime_weight_multipliers, RegimeSharpeEntry
        matrix = {
            "signal_a": {
                "NORMAL": RegimeSharpeEntry("signal_a", "NORMAL", 5.0, 0.80, 0.15, 300, 0.999, 3.0, 7.0),
            }
        }
        multipliers = derive_regime_weight_multipliers(matrix)
        assert multipliers["signal_a"]["NORMAL"] <= 1.5

    def test_multiplier_floored(self):
        from src.monitor.regime_sharpe_matrix import derive_regime_weight_multipliers, RegimeSharpeEntry
        matrix = {
            "signal_a": {
                "CRISIS": RegimeSharpeEntry("signal_a", "CRISIS", -3.0, 0.20, -0.15, 100, 0.01, -5.0, -1.0),
            }
        }
        multipliers = derive_regime_weight_multipliers(matrix)
        assert multipliers["signal_a"]["CRISIS"] >= 0.3


# ---------------------------------------------------------------------------
# Tests: format_for_gate_update
# ---------------------------------------------------------------------------

class TestFormatForGateUpdate:
    """Test output format matches RegimeGate.update_from_performance()."""

    def test_output_format_matches_gate_api(self):
        from src.monitor.regime_sharpe_matrix import compute_regime_sharpe_matrix, format_for_gate_update
        rng = np.random.RandomState(42)
        dates = pd.bdate_range("2020-01-01", periods=200)
        df = pd.DataFrame({
            "signal": ["sig_a"] * 100 + ["sig_b"] * 100,
            "regime": ["NORMAL"] * 50 + ["CRISIS"] * 50 + ["NORMAL"] * 50 + ["CRISIS"] * 50,
            "daily_return": rng.normal(0.001, 0.01, 200),
        }, index=dates)
        matrix = compute_regime_sharpe_matrix(df, n_bootstrap=100, seed=42)
        gate_input = format_for_gate_update(matrix)
        # Should be {regime: {signal: sharpe_ratio}}
        assert isinstance(gate_input, dict)
        for regime, signals in gate_input.items():
            assert isinstance(signals, dict)
            for signal, sharpe in signals.items():
                assert isinstance(sharpe, float)


# ---------------------------------------------------------------------------
# Tests: extract_signal_regime_data
# ---------------------------------------------------------------------------

class TestExtractSignalRegimeData:
    """Test extraction of signal-regime data from SQLite for matrix computation."""

    def test_returns_dataframe_with_required_columns(self, tmp_path):
        from src.monitor.regime_sharpe_matrix import extract_signal_regime_data
        import sqlite3

        # Create a minimal ensemble_signals.db
        db_path = tmp_path / "ensemble_signals.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE ensemble_votes (
                timestamp TEXT PRIMARY KEY,
                regime TEXT,
                regime_confidence REAL,
                num_sources INTEGER,
                consensus REAL,
                agreement_ratio REAL,
                equity_bias REAL,
                duration_bias REAL,
                gold_bias REAL,
                action TEXT,
                confidence REAL,
                reasoning TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE source_readings (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                source TEXT,
                value REAL,
                confidence REAL,
                weight REAL,
                regime_fit TEXT,
                explanation TEXT
            )
        """)
        conn.commit()
        conn.close()

        # Create price data
        dates = pd.bdate_range("2024-01-01", periods=10)
        prices = pd.DataFrame({"SPY": 450.0 + np.arange(10) * 0.5}, index=dates)

        df = extract_signal_regime_data(db_path, prices)
        # Empty DB → empty DataFrame with correct columns
        assert isinstance(df, pd.DataFrame)
        for col in ["signal", "regime", "daily_return"]:
            assert col in df.columns

    def test_extracts_signal_readings_with_regime(self, tmp_path):
        from src.monitor.regime_sharpe_matrix import extract_signal_regime_data
        import sqlite3

        db_path = tmp_path / "ensemble_signals.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE ensemble_votes (
                timestamp TEXT PRIMARY KEY,
                regime TEXT, regime_confidence REAL,
                num_sources INTEGER, consensus REAL,
                agreement_ratio REAL, equity_bias REAL,
                duration_bias REAL, gold_bias REAL,
                action TEXT, confidence REAL, reasoning TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE source_readings (
                id INTEGER PRIMARY KEY,
                timestamp TEXT, source TEXT, value REAL,
                confidence REAL, weight REAL,
                regime_fit TEXT, explanation TEXT
            )
        """)

        # Insert 60 days of data
        dates = pd.bdate_range("2024-01-01", periods=60)
        for i, d in enumerate(dates):
            ts = d.strftime("%Y-%m-%d")
            regime = "NORMAL" if i < 30 else "HIGH_VOL"
            conn.execute(
                "INSERT INTO ensemble_votes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (ts, regime, 0.8, 6, 0.5, 0.7, 0.1, -0.05, 0.02, "BUY", 0.6, ""),
            )
            conn.execute(
                "INSERT INTO source_readings VALUES (?,?,?,?,?,?,?,?)",
                (i, ts, "alternative_data", 0.3 + i * 0.01, 0.7, 0.3, regime, ""),
            )
        conn.commit()
        conn.close()

        # Price data
        prices = pd.DataFrame(
            {"SPY": 450.0 + np.arange(60) * 0.3},
            index=dates,
        )

        df = extract_signal_regime_data(db_path, prices)
        assert len(df) > 0
        assert "alternative_data" in df["signal"].values
        assert "NORMAL" in df["regime"].values
        assert "HIGH_VOL" in df["regime"].values

    def test_empty_database_returns_empty_dataframe(self, tmp_path):
        from src.monitor.regime_sharpe_matrix import extract_signal_regime_data
        import sqlite3

        db_path = tmp_path / "ensemble_signals.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE ensemble_votes (timestamp TEXT, regime TEXT)")
        conn.execute("CREATE TABLE source_readings (timestamp TEXT, source TEXT, value REAL)")
        conn.commit()
        conn.close()

        prices = pd.DataFrame({"SPY": [450.0]})
        df = extract_signal_regime_data(db_path, prices)
        assert df.empty

    def test_missing_database_returns_empty_dataframe(self, tmp_path):
        from src.monitor.regime_sharpe_matrix import extract_signal_regime_data
        db_path = tmp_path / "nonexistent.db"
        prices = pd.DataFrame({"SPY": [450.0]})
        df = extract_signal_regime_data(db_path, prices)
        assert df.empty


# ---------------------------------------------------------------------------
# Tests: update_gate_from_history
# ---------------------------------------------------------------------------

class TestUpdateGateFromHistory:
    """Test wiring: extract → compute → update RegimeGate."""

    def test_updates_gate_rules_from_data(self, tmp_path, rng):
        from src.monitor.regime_sharpe_matrix import update_gate_from_history
        from src.signals.regime_gate import RegimeGate
        import sqlite3

        # Create DB with 100 days of data
        db_path = tmp_path / "ensemble_signals.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE ensemble_votes (
                timestamp TEXT PRIMARY KEY,
                regime TEXT, regime_confidence REAL,
                num_sources INTEGER, consensus REAL,
                agreement_ratio REAL, equity_bias REAL,
                duration_bias REAL, gold_bias REAL,
                action TEXT, confidence REAL, reasoning TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE source_readings (
                id INTEGER PRIMARY KEY,
                timestamp TEXT, source TEXT, value REAL,
                confidence REAL, weight REAL,
                regime_fit TEXT, explanation TEXT
            )
        """)

        dates = pd.bdate_range("2024-01-01", periods=100)
        for i, d in enumerate(dates):
            ts = d.strftime("%Y-%m-%d")
            regime = "NORMAL" if i < 50 else "CRISIS"
            conn.execute(
                "INSERT INTO ensemble_votes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (ts, regime, 0.8, 6, 0.5, 0.7, 0.1, -0.05, 0.02, "BUY", 0.6, ""),
            )
            conn.execute(
                "INSERT INTO source_readings VALUES (?,?,?,?,?,?,?,?)",
                (i, ts, "test_signal", 0.5, 0.7, 0.3, regime, ""),
            )
        conn.commit()
        conn.close()

        prices = pd.DataFrame(
            {"SPY": 450.0 + np.arange(100) * 0.5},
            index=dates,
        )

        gate = RegimeGate()
        matrix = update_gate_from_history(
            gate, db_path, prices, n_bootstrap=100, seed=42,
        )

        # Matrix should have data
        assert len(matrix) > 0
        assert "test_signal" in matrix

    def test_empty_db_returns_empty_matrix(self, tmp_path):
        from src.monitor.regime_sharpe_matrix import update_gate_from_history
        from src.signals.regime_gate import RegimeGate
        import sqlite3

        db_path = tmp_path / "ensemble_signals.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE ensemble_votes (timestamp TEXT, regime TEXT)")
        conn.execute("CREATE TABLE source_readings (timestamp TEXT, source TEXT, value REAL)")
        conn.commit()
        conn.close()

        prices = pd.DataFrame({"SPY": [450.0]})
        gate = RegimeGate()
        matrix = update_gate_from_history(gate, db_path, prices)
        assert matrix == {}


# ---------------------------------------------------------------------------
# Tests: load_persisted_gate_rules
# ---------------------------------------------------------------------------

class TestLoadPersistedGateRules:
    """Test loading data-driven gate rules from persisted JSON."""

    def test_loads_valid_rules(self, tmp_path):
        from src.monitor.regime_sharpe_matrix import load_persisted_gate_rules
        import json

        persist_path = tmp_path / "regime_gate_persisted.json"
        data = {
            "gate_rules": {
                "signal_a": ["CRISIS", "HIGH_VOL"],
                "signal_b": ["LOW_VOL"],
            },
            "weight_multipliers": {},
            "computed_at": datetime.now().isoformat(),
            "n_observations": 500,
        }
        persist_path.write_text(json.dumps(data))

        rules = load_persisted_gate_rules(persist_path)
        assert rules is not None
        assert "signal_a" in rules
        assert rules["signal_a"] == {"CRISIS", "HIGH_VOL"}
        assert rules["signal_b"] == {"LOW_VOL"}

    def test_returns_none_when_missing(self, tmp_path):
        from src.monitor.regime_sharpe_matrix import load_persisted_gate_rules
        rules = load_persisted_gate_rules(tmp_path / "nonexistent.json")
        assert rules is None

    def test_returns_none_when_stale(self, tmp_path):
        from src.monitor.regime_sharpe_matrix import load_persisted_gate_rules
        import json
        from datetime import datetime, timedelta

        persist_path = tmp_path / "regime_gate_persisted.json"
        data = {
            "gate_rules": {"sig": ["CRISIS"]},
            "computed_at": (datetime.now() - timedelta(hours=48)).isoformat(),
        }
        persist_path.write_text(json.dumps(data))

        rules = load_persisted_gate_rules(persist_path, max_age_hours=24)
        assert rules is None

    def test_loads_recent_data(self, tmp_path):
        from src.monitor.regime_sharpe_matrix import load_persisted_gate_rules
        import json

        persist_path = tmp_path / "regime_gate_persisted.json"
        data = {
            "gate_rules": {"sig": ["CRISIS"]},
            "computed_at": datetime.now().isoformat(),
        }
        persist_path.write_text(json.dumps(data))

        rules = load_persisted_gate_rules(persist_path, max_age_hours=24)
        assert rules is not None
        assert "sig" in rules

    def test_returns_none_when_empty_rules(self, tmp_path):
        from src.monitor.regime_sharpe_matrix import load_persisted_gate_rules
        import json

        persist_path = tmp_path / "regime_gate_persisted.json"
        data = {"gate_rules": {}, "computed_at": datetime.now().isoformat()}
        persist_path.write_text(json.dumps(data))

        rules = load_persisted_gate_rules(persist_path)
        assert rules is None

    def test_returns_none_when_no_gate_rules_key(self, tmp_path):
        from src.monitor.regime_sharpe_matrix import load_persisted_gate_rules
        import json

        persist_path = tmp_path / "regime_gate_persisted.json"
        data = {"other_key": "value"}
        persist_path.write_text(json.dumps(data))

        rules = load_persisted_gate_rules(persist_path)
        assert rules is None
