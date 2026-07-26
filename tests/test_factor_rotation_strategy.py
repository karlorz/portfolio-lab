#!/usr/bin/env python3
"""
Tests for strategy/factor_rotation.py — FactorScore, FactorMomentumEngine,
FactorRotationBacktest.
"""
import json
import sqlite3
import numpy as np
import pandas as pd

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.strategy.factor_rotation import (
    FactorScore,
    FactorMomentumEngine,
    FactorRotationBacktest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_factor_score(**overrides):
    defaults = dict(
        symbol="MTUM",
        factor_name="Momentum",
        price=100.0,
        return_12m=0.15,
        return_6m=0.10,
        return_3m=0.05,
        volatility=0.18,
        sharpe_12m=0.83,
        momentum_score=0.12,
        rank=1,
    )
    defaults.update(overrides)
    return FactorScore(**defaults)


def _make_engine(tmp_path, **kwargs):
    defaults = dict(
        db_path=tmp_path / "market.db",
        lookback_months=12,
        top_n=2,
        min_momentum=0.0,
        vol_lookback=20,
    )
    defaults.update(kwargs)
    return FactorMomentumEngine(**defaults)


def _seed_prices(db_path, symbols=None, days=300, start_price=100.0):
    """Insert fake price data into the SQLite DB for testing."""
    symbols = symbols or ["MTUM", "VTV", "QUAL", "USMV", "SPY", "QQQ"]
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS prices (symbol TEXT, date TEXT, close REAL, volume REAL)")
    dates = pd.bdate_range(end=datetime.now(), periods=days)

    for sym in symbols:
        price = start_price
        for i, dt in enumerate(dates):
            ret = np.random.normal(0.0004, 0.015)
            price *= (1 + ret)
            conn.execute(
                "INSERT INTO prices VALUES (?, ?, ?, ?)",
                (sym, dt.strftime("%Y-%m-%d"), round(price, 2), 1000000),
            )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# FactorScore tests
# ---------------------------------------------------------------------------

class TestFactorScore:
    def test_creation(self):
        score = _make_factor_score()
        assert score.symbol == "MTUM"
        assert score.return_12m == 0.15
        assert score.rank == 1

    def test_default_ml_fields(self):
        score = _make_factor_score()
        assert score.value_momentum_synergy == 0.0
        assert score.momentum_acceleration == 0.0
        assert score.vol_adjusted_momentum == 0.0
        assert score.regime_momentum == 0.0
        assert score.factor_divergence == 0.0

    def test_custom_ml_fields(self):
        score = _make_factor_score(
            value_momentum_synergy=0.05,
            momentum_acceleration=0.02,
            vol_adjusted_momentum=0.8,
            regime_momentum=0.1,
            factor_divergence=0.3,
            composite_ml_score=0.15,
        )
        assert score.value_momentum_synergy == 0.05
        assert score.composite_ml_score == 0.15


# ---------------------------------------------------------------------------
# FactorMomentumEngine tests
# ---------------------------------------------------------------------------

class TestFactorMomentumEngine:
    def test_init_defaults(self, tmp_path):
        engine = _make_engine(tmp_path)
        assert engine.lookback_months == 12
        assert engine.top_n == 2
        assert engine.min_momentum == 0.0

    def test_init_custom(self, tmp_path):
        engine = _make_engine(tmp_path, top_n=3, min_momentum=0.05)
        assert engine.top_n == 3
        assert engine.min_momentum == 0.05

    def test_factors_defined(self, tmp_path):
        engine = _make_engine(tmp_path)
        assert "MTUM" in engine.FACTORS
        assert "VTV" in engine.FACTORS
        assert "SPY" in engine.FACTORS
        assert "QQQ" in engine.FACTORS

    def test_universe_is_factor_keys(self, tmp_path):
        engine = _make_engine(tmp_path)
        assert set(engine.universe) == set(engine.FACTORS.keys())

    def test_max_per_category(self, tmp_path):
        engine = _make_engine(tmp_path)
        assert engine.max_per_category == 1


class TestFetchPriceData:
    def test_no_db_returns_empty(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine._fetch_price_data("MTUM")
        assert result == []

    def test_returns_data_from_db(self, tmp_path):
        np.random.seed(42)
        _seed_prices(tmp_path / "market.db")
        engine = _make_engine(tmp_path)
        result = engine._fetch_price_data("MTUM", days=300)
        assert len(result) > 0
        assert "close" in result[0]
        assert "date" in result[0]

    def test_limited_days(self, tmp_path):
        np.random.seed(42)
        _seed_prices(tmp_path / "market.db")
        engine = _make_engine(tmp_path)
        result = engine._fetch_price_data("MTUM", days=50)
        assert len(result) <= 50


class TestCalculateFactorScore:
    def test_insufficient_data_returns_none(self, tmp_path):
        """If DB has <252 days of data, _calculate_factor_score returns None."""
        np.random.seed(42)
        db = tmp_path / "market.db"
        _seed_prices(db, days=100)
        engine = _make_engine(tmp_path)
        score = engine._calculate_factor_score("MTUM")
        assert score is None

    def test_returns_factor_score_with_enough_data(self, tmp_path):
        np.random.seed(42)
        _seed_prices(tmp_path / "market.db", days=300)
        engine = _make_engine(tmp_path)
        score = engine._calculate_factor_score("MTUM")
        assert score is not None
        assert isinstance(score, FactorScore)
        assert score.symbol == "MTUM"
        assert score.price > 0
        assert score.volatility > 0

    def test_momentum_score_fields(self, tmp_path):
        np.random.seed(42)
        _seed_prices(tmp_path / "market.db", days=300)
        engine = _make_engine(tmp_path)
        score = engine._calculate_factor_score("SPY")
        assert score is not None
        # Returns should be floats
        assert isinstance(score.return_12m, float)
        assert isinstance(score.return_6m, float)
        assert isinstance(score.return_3m, float)
        assert isinstance(score.sharpe_12m, float)
        assert isinstance(score.momentum_score, float)

    def test_unknown_symbol_returns_none(self, tmp_path):
        np.random.seed(42)
        _seed_prices(tmp_path / "market.db", days=300, symbols=["SPY"])
        engine = _make_engine(tmp_path)
        score = engine._calculate_factor_score("UNKNOWN_TICKER")
        assert score is None


class TestEvaluate:
    def test_no_data_returns_error(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine.evaluate()
        assert "error" in result
        assert result["selected_factors"] == []

    def test_evaluate_with_data(self, tmp_path):
        np.random.seed(42)
        _seed_prices(tmp_path / "market.db")
        engine = _make_engine(tmp_path)
        result = engine.evaluate()
        assert "timestamp" in result
        assert "selected_factors" in result
        assert "allocation" in result
        assert "current_scores" in result
        assert "recommendation" in result
        assert "signal_strength" in result

    def test_evaluate_selects_top_n(self, tmp_path):
        np.random.seed(42)
        _seed_prices(tmp_path / "market.db")
        engine = _make_engine(tmp_path, top_n=3)
        result = engine.evaluate()
        assert len(result["selected_factors"]) <= 3

    def test_allocation_sums_to_one(self, tmp_path):
        np.random.seed(42)
        _seed_prices(tmp_path / "market.db")
        engine = _make_engine(tmp_path)
        result = engine.evaluate()
        alloc = result["allocation"]
        if alloc:
            total = sum(alloc.values())
            assert abs(total - 1.0) < 0.01

    def test_diversity_constraint(self, tmp_path):
        """Each category should appear at most max_per_category times."""
        np.random.seed(42)
        _seed_prices(tmp_path / "market.db")
        engine = _make_engine(tmp_path, top_n=4)
        result = engine.evaluate()
        # Count categories in selected factors
        categories = [engine.FACTORS[s]["category"] for s in result["selected_factors"]]
        for cat in set(categories):
            assert categories.count(cat) <= engine.max_per_category

    def test_signal_strength_bounded(self, tmp_path):
        np.random.seed(42)
        _seed_prices(tmp_path / "market.db")
        engine = _make_engine(tmp_path)
        result = engine.evaluate()
        assert 0.0 <= result["signal_strength"] <= 1.0

    def test_as_of_decision_is_unchanged_by_future_rows(self, tmp_path):
        np.random.seed(42)
        db = tmp_path / "market.db"
        _seed_prices(db, days=320)
        engine = _make_engine(tmp_path)
        as_of = "2026-06-30"

        before = engine.evaluate(as_of=as_of)

        conn = sqlite3.connect(db)
        for symbol in engine.universe:
            conn.execute(
                "INSERT INTO prices VALUES (?, ?, ?, ?)",
                (symbol, "2099-01-04", 999999.0, 1000000),
            )
        conn.commit()
        conn.close()

        after = engine.evaluate(as_of=as_of)
        assert after["as_of"] == as_of
        assert after["selected_factors"] == before["selected_factors"]
        assert after["allocation"] == before["allocation"]
        assert after["current_scores"] == before["current_scores"]

    def test_min_momentum_filter(self, tmp_path):
        """Factors with return_12m < min_momentum should be excluded."""
        np.random.seed(42)
        _seed_prices(tmp_path / "market.db", start_price=100.0)
        engine = _make_engine(tmp_path, min_momentum=0.50)  # Very high threshold
        result = engine.evaluate()
        # With random data, unlikely to have >50% 12m return
        # Either empty selection or only high-return factors
        for sym in result["selected_factors"]:
            score = result["current_scores"].get(sym)
            if score:
                assert score["return_12m"] >= 0.50


class TestGenerateAllocation:
    def test_empty_selection_returns_spy(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine._generate_allocation([])
        assert result == {"SPY": 1.0}

    def test_inverse_volatility_weighting(self, tmp_path):
        engine = _make_engine(tmp_path)
        s1 = _make_factor_score(symbol="A", volatility=0.10)
        s2 = _make_factor_score(symbol="B", volatility=0.20)
        alloc = engine._generate_allocation([("A", s1), ("B", s2)])
        assert abs(sum(alloc.values()) - 1.0) < 0.01
        # Lower vol factor should get higher weight
        assert alloc["A"] > alloc["B"]

    def test_equal_volatility_equal_weight(self, tmp_path):
        engine = _make_engine(tmp_path)
        s1 = _make_factor_score(symbol="A", volatility=0.15)
        s2 = _make_factor_score(symbol="B", volatility=0.15)
        alloc = engine._generate_allocation([("A", s1), ("B", s2)])
        assert abs(alloc["A"] - 0.5) < 0.01
        assert abs(alloc["B"] - 0.5) < 0.01

    def test_zero_vol_floor(self, tmp_path):
        """Zero volatility should not cause division by zero."""
        engine = _make_engine(tmp_path)
        s1 = _make_factor_score(symbol="A", volatility=0.0)
        alloc = engine._generate_allocation([("A", s1)])
        assert "A" in alloc
        assert alloc["A"] == 1.0


class TestCalculateMlFeatures:
    def test_returns_dict(self, tmp_path):
        engine = _make_engine(tmp_path)
        score = _make_factor_score(symbol="VTV")
        features = engine._calculate_ml_features("VTV", {"VTV": score, "SPY": _make_factor_score(symbol="SPY")})
        assert isinstance(features, dict)
        assert "value_momentum_synergy" in features
        assert "momentum_acceleration" in features
        assert "vol_adjusted_momentum" in features
        assert "regime_momentum" in features
        assert "factor_divergence" in features
        assert "composite_ml_score" in features

    def test_unknown_symbol_returns_empty(self, tmp_path):
        engine = _make_engine(tmp_path)
        features = engine._calculate_ml_features("UNKNOWN", {})
        assert features == {}

    def test_vix_regime_context(self, tmp_path):
        engine = _make_engine(tmp_path)
        score = _make_factor_score(symbol="MTUM")
        features_low = engine._calculate_ml_features("MTUM", {"MTUM": score}, vix_level=12.0)
        features_high = engine._calculate_ml_features("MTUM", {"MTUM": score}, vix_level=35.0)
        # High VIX should dampen regime_momentum
        assert features_high["regime_momentum"] <= features_low["regime_momentum"]

    def test_composite_is_ridge_shrunk(self, tmp_path):
        engine = _make_engine(tmp_path)
        score = _make_factor_score(symbol="MTUM")
        features = engine._calculate_ml_features("MTUM", {"MTUM": score})
        # Ridge shrinkage: composite_ml_score = raw * 0.9
        raw = (
            score.momentum_score * 0.3 +
            features.get("value_momentum_synergy", 0) * 0.2 +
            features["momentum_acceleration"] * 0.15 +
            features["vol_adjusted_momentum"] * 0.1 +
            features["regime_momentum"] * 0.15 +
            features["factor_divergence"] * 0.1
        )
        expected = raw * 0.9
        assert abs(features["composite_ml_score"] - expected) < 0.001


class TestEvaluateTsfm:
    def test_no_data_returns_error(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine.evaluate_tsfm()
        assert "error" in result or "tsfm" in result

    def test_tsfm_with_data(self, tmp_path):
        np.random.seed(42)
        _seed_prices(tmp_path / "market.db")
        engine = _make_engine(tmp_path)
        result = engine.evaluate_tsfm(vix_level=20.0)
        assert "tsfm" in result
        assert "recommendation_tsfm" in result
        tsfm = result["tsfm"]
        assert tsfm["enabled"] is True
        assert tsfm["vix_context"] == 20.0

    def test_tsfm_vix_context(self, tmp_path):
        np.random.seed(42)
        _seed_prices(tmp_path / "market.db")
        engine = _make_engine(tmp_path)
        result = engine.evaluate_tsfm(vix_level=35.0)
        assert result["tsfm"]["vix_context"] == 35.0


class TestEvaluateMlEnhanced:
    def test_no_data_returns_error(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine.evaluate_ml_enhanced()
        assert "error" in result

    def test_ml_enhanced_with_data(self, tmp_path):
        np.random.seed(42)
        _seed_prices(tmp_path / "market.db")
        engine = _make_engine(tmp_path)
        result = engine.evaluate_ml_enhanced(vix_level=20.0)
        assert "ml_enhanced" in result
        assert result["ml_enhanced"] is True
        assert "ml_scores" in result
        assert "selected_factors_ml" in result
        assert "ml_recommendation" in result

    def test_ml_scores_have_features(self, tmp_path):
        np.random.seed(42)
        _seed_prices(tmp_path / "market.db")
        engine = _make_engine(tmp_path)
        result = engine.evaluate_ml_enhanced()
        ml_scores = result["ml_scores"]
        for sym, features in ml_scores.items():
            assert "composite_ml_score" in features
            assert "value_momentum_synergy" in features
            assert "momentum_acceleration" in features
            assert "vol_adjusted_momentum" in features
            assert "regime_momentum" in features
            assert "factor_divergence" in features


class TestSignalStrength:
    def test_empty_selection_returns_zero(self, tmp_path):
        engine = _make_engine(tmp_path)
        assert engine._calculate_signal_strength([]) == 0.0

    def test_bounded_0_to_1(self, tmp_path):
        engine = _make_engine(tmp_path)
        scores = [
            ("A", _make_factor_score(momentum_score=0.5, volatility=0.15)),
            ("B", _make_factor_score(momentum_score=0.3, volatility=0.20)),
        ]
        strength = engine._calculate_signal_strength(scores)
        assert 0.0 <= strength <= 1.0

    def test_strong_signal(self, tmp_path):
        engine = _make_engine(tmp_path)
        scores = [
            ("A", _make_factor_score(momentum_score=0.8, volatility=0.10)),
            ("B", _make_factor_score(momentum_score=0.6, volatility=0.12)),
        ]
        strength = engine._calculate_signal_strength(scores)
        assert strength > 0.5

    def test_weak_signal(self, tmp_path):
        engine = _make_engine(tmp_path)
        scores = [
            ("A", _make_factor_score(momentum_score=-0.3, volatility=0.40)),
            ("B", _make_factor_score(momentum_score=-0.2, volatility=0.35)),
        ]
        strength = engine._calculate_signal_strength(scores)
        assert strength < 0.7


class TestGenerateRecommendation:
    def test_empty_selection(self, tmp_path):
        engine = _make_engine(tmp_path)
        rec = engine._generate_recommendation([], {})
        assert "Hold SPY" in rec or "No factor" in rec

    def test_concentrated_category(self, tmp_path):
        engine = _make_engine(tmp_path)
        scores = [
            ("MTUM", _make_factor_score(symbol="MTUM")),
            ("VTV", _make_factor_score(symbol="VTV")),
        ]
        # Force same category by modifying FACTORS
        orig_mtum_cat = engine.FACTORS["MTUM"]["category"]
        engine.FACTORS["MTUM"]["category"] = "value"
        rec = engine._generate_recommendation(scores, {})
        assert "concentrated" in rec or "Rotate" in rec
        engine.FACTORS["MTUM"]["category"] = orig_mtum_cat

    def test_strong_momentum(self, tmp_path):
        engine = _make_engine(tmp_path)
        scores = [
            ("MTUM", _make_factor_score(return_12m=0.30)),
            ("USMV", _make_factor_score(symbol="USMV", return_12m=0.25)),
        ]
        rec = engine._generate_recommendation(scores, {})
        assert "strong" in rec


class TestGenerateTsfmRecommendation:
    def test_empty_selection(self, tmp_path):
        engine = _make_engine(tmp_path)
        rec = engine._generate_tsfm_recommendation([], {}, 20.0)
        assert "No factors" in rec or "Risk-off" in rec

    def test_with_selection(self, tmp_path):
        engine = _make_engine(tmp_path)
        score = _make_factor_score(tsfm_allocation_scalar=1.5)
        rec = engine._generate_tsfm_recommendation(
            [("MTUM", score)], {"MTUM": score}, vix_level=18.0
        )
        assert "TSFM" in rec


class TestGenerateMlRecommendation:
    def test_empty_selection(self, tmp_path):
        engine = _make_engine(tmp_path)
        rec = engine._generate_ml_recommendation([], {})
        assert "No factors" in rec or "Hold SPY" in rec

    def test_with_selection(self, tmp_path):
        engine = _make_engine(tmp_path)
        score = _make_factor_score(composite_ml_score=0.4)
        rec = engine._generate_ml_recommendation([("MTUM", score)], {"MTUM": score})
        assert "ML-Enhanced" in rec

    def test_signal_strength_labels(self, tmp_path):
        engine = _make_engine(tmp_path)
        strong = _make_factor_score(composite_ml_score=0.5)
        rec = engine._generate_ml_recommendation([("MTUM", strong)], {"MTUM": strong})
        assert "strong" in rec

        weak = _make_factor_score(composite_ml_score=0.05)
        rec = engine._generate_ml_recommendation([("VTV", weak)], {"VTV": weak})
        assert "weak" in rec


# ---------------------------------------------------------------------------
# FactorRotationBacktest tests
# ---------------------------------------------------------------------------

class TestFactorRotationBacktest:
    def test_no_db_returns_error(self, tmp_path):
        engine = _make_engine(tmp_path)
        bt = FactorRotationBacktest(engine)
        result = bt.run_backtest("2020-01-01", "2025-01-01")
        assert "error" in result

    def test_insufficient_data_returns_error(self, tmp_path):
        np.random.seed(42)
        db = tmp_path / "market.db"
        _seed_prices(db, days=100)
        engine = _make_engine(tmp_path)
        bt = FactorRotationBacktest(engine)
        result = bt.run_backtest("2020-01-01", "2025-01-01")
        assert "error" in result

    def test_backtest_with_data(self, tmp_path):
        np.random.seed(42)
        db = tmp_path / "market.db"
        _seed_prices(
            db,
            symbols=list(FactorMomentumEngine.FACTORS),
            days=400,
        )
        engine = _make_engine(tmp_path)
        bt = FactorRotationBacktest(engine)
        result = bt.run_backtest("2024-01-01", "2026-01-01")
        assert result["strategy"] == "factor_momentum_rotation"
        assert result["status"] == "completed"
        assert "cagr" in result
        assert "sharpe_ratio" in result
        assert "max_drawdown" in result
        assert "trade_count" in result
        evidence = result["profitability_evidence"]
        assert evidence["point_in_time"] is True
        assert evidence["data"]["mode"] == "real"
        assert evidence["promotion_eligible"] is True
        assert evidence["costs"]["total_dollars"] > 0

    def test_backtest_no_spy_returns_error(self, tmp_path):
        np.random.seed(42)
        db = tmp_path / "market.db"
        _seed_prices(db, symbols=["MTUM", "VTV"], days=400)
        engine = _make_engine(tmp_path)
        bt = FactorRotationBacktest(engine)
        result = bt.run_backtest("2024-01-01", "2026-01-01")
        # Without SPY data for benchmark, should error
        assert "error" in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_negative_momentum(self, tmp_path):
        """When all factors have negative momentum, some should still be selected
        if min_momentum=0 (default)."""
        np.random.seed(42)
        db = tmp_path / "market.db"
        # Create declining prices
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE IF NOT EXISTS prices (symbol TEXT, date TEXT, close REAL, volume REAL)")
        dates = pd.bdate_range(end=datetime.now(), periods=300)
        for sym in ["MTUM", "SPY", "VTV", "QUAL", "USMV", "QQQ"]:
            price = 200.0
            for dt in dates:
                price *= 0.999  # Declining
                conn.execute(
                    "INSERT INTO prices VALUES (?, ?, ?, ?)",
                    (sym, dt.strftime("%Y-%m-%d"), round(price, 2), 1000000),
                )
        conn.commit()
        conn.close()

        engine = _make_engine(tmp_path)
        result = engine.evaluate()
        # Should still produce a result (possibly with recommendation to hold SPY)
        assert "selected_factors" in result

    def test_single_factor_available(self, tmp_path):
        np.random.seed(42)
        db = tmp_path / "market.db"
        _seed_prices(db, symbols=["MTUM", "SPY"], days=300)
        engine = _make_engine(tmp_path)
        # Override universe to just 2 symbols
        engine.universe = ["MTUM", "SPY"]
        result = engine.evaluate()
        assert "selected_factors" in result
        assert len(result["selected_factors"]) <= 2

    def test_zero_volatility_score(self, tmp_path):
        """FactorScore with zero volatility should not crash allocation."""
        engine = _make_engine(tmp_path)
        score = _make_factor_score(volatility=0.0)
        alloc = engine._generate_allocation([("A", score)])
        assert "A" in alloc

    def test_factor_categories_covered(self, tmp_path):
        engine = _make_engine(tmp_path)
        categories = set(info["category"] for info in engine.FACTORS.values())
        assert "value" in categories
        assert "momentum" in categories
        assert "quality" in categories
        assert "low_vol" in categories
        assert "small" in categories
        assert "core" in categories


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
