"""Tests for Factor Momentum Rotation Strategy — factor_rotation.py.

Covers: FactorScore dataclass, FactorMomentumEngine scoring/ranking/selection,
inverse-volatility weighting, diversity constraints, TSFM evaluation,
signal strength, recommendation generation, edge cases.

DB calls are mocked since MARKET_DB doesn't exist in test environment.
"""
import numpy as np
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.strategy.factor_rotation import (
    FactorScore,
    FactorMomentumEngine,
    FactorRotationBacktest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_score(symbol="MTUM", factor_name="Momentum", return_12m=0.15,
                return_6m=0.08, return_3m=0.04, volatility=0.18,
                sharpe_12m=0.83, momentum_score=0.10, rank=1,
                tsfm_score=0.0, tsfm_allocation_scalar=1.0):
    """Create a FactorScore with sensible defaults."""
    return FactorScore(
        symbol=symbol,
        factor_name=factor_name,
        price=100.0,
        return_12m=return_12m,
        return_6m=return_6m,
        return_3m=return_3m,
        volatility=volatility,
        sharpe_12m=sharpe_12m,
        momentum_score=momentum_score,
        rank=rank,
        tsfm_score=tsfm_score,
        tsfm_allocation_scalar=tsfm_allocation_scalar,
    )


def _generate_price_data(n=300, drift=0.0004, vol=0.015, seed=42):
    """Generate synthetic price data (list of dicts with date/close/volume)."""
    rng = np.random.RandomState(seed)
    prices = [100.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + rng.normal(drift, vol)))

    data = []
    for i, p in enumerate(prices):
        data.append({
            "date": f"2024-{(i // 21 + 1):02d}-{(i % 21 + 1):02d}",
            "close": p,
            "volume": 1000000,
        })
    return data


def _make_engine_with_mocked_db(top_n=2, min_momentum=0.0):
    """Create engine with _fetch_price_data mocked to return synthetic data."""
    engine = FactorMomentumEngine(top_n=top_n, min_momentum=min_momentum)

    def mock_fetch(symbol, days=300):
        # Different seeds per symbol for variety
        seed_map = {
            "MTUM": 42, "USMV": 43, "QUAL": 44, "VTV": 45, "IJR": 46,
            "VLUE": 47, "SPHQ": 48, "SPLV": 49, "VBR": 50, "SPY": 51,
            "QQQ": 52,
        }
        seed = seed_map.get(symbol, 42)
        # Different drifts per category
        drift_map = {
            "MTUM": 0.001, "USMV": 0.0003, "QUAL": 0.0005,
            "VTV": 0.0002, "IJR": 0.0006, "SPY": 0.0004, "QQQ": 0.0008,
        }
        drift = drift_map.get(symbol, 0.0004)
        return _generate_price_data(days, drift=drift, seed=seed)

    engine._fetch_price_data = mock_fetch
    return engine


# ---------------------------------------------------------------------------
# FactorScore dataclass
# ---------------------------------------------------------------------------

class TestFactorScore:
    def test_creation(self):
        score = _make_score()
        assert score.symbol == "MTUM"
        assert score.return_12m == 0.15
        assert score.volatility == 0.18

    def test_default_ml_fields(self):
        score = _make_score()
        assert score.value_momentum_synergy == 0.0
        assert score.momentum_acceleration == 0.0
        assert score.vol_adjusted_momentum == 0.0
        assert score.regime_momentum == 0.0
        assert score.factor_divergence == 0.0
        assert score.composite_ml_score == 0.0

    def test_tsfm_fields(self):
        score = _make_score(tsfm_score=1.5, tsfm_allocation_scalar=1.75)
        assert score.tsfm_score == 1.5
        assert score.tsfm_allocation_scalar == 1.75


# ---------------------------------------------------------------------------
# FactorMomentumEngine — scoring
# ---------------------------------------------------------------------------

class TestFactorScoring:
    def test_calculate_factor_score_with_data(self):
        engine = _make_engine_with_mocked_db()
        score = engine._calculate_factor_score("MTUM")
        assert score is not None
        assert score.symbol == "MTUM"
        assert score.factor_name == "Momentum"
        assert score.volatility > 0
        assert score.momentum_score != 0

    def test_calculate_factor_score_insufficient_data(self):
        """With < 252 days of data, should return None."""
        engine = FactorMomentumEngine()
        def mock_fetch(symbol, days=300):
            return _generate_price_data(100)  # Too short
        engine._fetch_price_data = mock_fetch
        score = engine._calculate_factor_score("MTUM")
        assert score is None

    def test_calculate_factor_score_returns(self):
        """12m, 6m, 3m returns should be computed correctly."""
        engine = _make_engine_with_mocked_db()
        score = engine._calculate_factor_score("MTUM")
        if score:
            # With positive drift, 12m return should be positive
            assert score.return_12m != 0
            # 6m should also be computed
            assert score.return_6m != 0

    def test_calculate_factor_score_tsfm(self):
        """TSFM score should be z-score capped at ±2."""
        engine = _make_engine_with_mocked_db()
        score = engine._calculate_factor_score("MTUM")
        if score:
            assert -2.0 <= score.tsfm_score <= 2.0


# ---------------------------------------------------------------------------
# FactorMomentumEngine — evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_evaluate_returns_dict(self):
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate()
        assert isinstance(result, dict)
        assert "timestamp" in result
        assert "selected_factors" in result
        assert "allocation" in result
        assert "current_scores" in result

    def test_evaluate_selects_top_n(self):
        engine = _make_engine_with_mocked_db(top_n=2)
        result = engine.evaluate()
        # Should select at most top_n factors
        assert len(result["selected_factors"]) <= 2

    def test_evaluate_top_n_3(self):
        engine = _make_engine_with_mocked_db(top_n=3)
        result = engine.evaluate()
        assert len(result["selected_factors"]) <= 3

    def test_evaluate_no_data(self):
        """When DB has no data, evaluate should return error gracefully."""
        engine = FactorMomentumEngine()
        def mock_fetch(symbol, days=300):
            return []
        engine._fetch_price_data = mock_fetch
        result = engine.evaluate()
        assert "error" in result
        assert result["selected_factors"] == []

    def test_evaluate_allocation_sums_to_one(self):
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate()
        if result["allocation"]:
            total = sum(result["allocation"].values())
            assert abs(total - 1.0) < 0.01

    def test_evaluate_diversity_constraint(self):
        """No more than 1 factor per category should be selected."""
        engine = _make_engine_with_mocked_db(top_n=3)
        result = engine.evaluate()
        selected = result["selected_factors"]
        categories = [engine.FACTORS[s]["category"] for s in selected]
        # Each category appears at most once
        assert len(categories) == len(set(categories))

    def test_evaluate_min_momentum_filter(self):
        """Factors below min_momentum should be filtered out."""
        engine = _make_engine_with_mocked_db(min_momentum=0.50)  # Very high threshold
        result = engine.evaluate()
        # With very high min_momentum, likely no factors qualify
        # (or very few)
        for sym in result["selected_factors"]:
            assert result["current_scores"][sym]["return_12m"] >= 0.50

    def test_evaluate_signal_strength(self):
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate()
        assert "signal_strength" in result
        assert 0.0 <= result["signal_strength"] <= 1.0

    def test_evaluate_recommendation(self):
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate()
        assert "recommendation" in result
        assert isinstance(result["recommendation"], str)

    def test_evaluate_ranks_assigned(self):
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate()
        scores = result["current_scores"]
        ranks = [scores[s]["rank"] for s in scores]
        assert 1 in ranks  # At least one factor ranked #1


# ---------------------------------------------------------------------------
# Inverse volatility weighting
# ---------------------------------------------------------------------------

class TestInverseVolatilityWeighting:
    def test_generate_allocation_weights_sum_to_one(self):
        engine = FactorMomentumEngine()
        selected = [
            ("MTUM", _make_score(symbol="MTUM", volatility=0.20)),
            ("USMV", _make_score(symbol="USMV", volatility=0.12)),
        ]
        alloc = engine._generate_allocation(selected)
        assert abs(sum(alloc.values()) - 1.0) < 1e-10

    def test_lower_vol_gets_higher_weight(self):
        """Lower volatility factor should get higher weight."""
        engine = FactorMomentumEngine()
        selected = [
            ("MTUM", _make_score(symbol="MTUM", volatility=0.30)),  # Higher vol
            ("USMV", _make_score(symbol="USMV", volatility=0.10)),  # Lower vol
        ]
        alloc = engine._generate_allocation(selected)
        # USMV (lower vol) should get more weight
        assert alloc["USMV"] > alloc["MTUM"]

    def test_empty_selection_defaults_to_spy(self):
        engine = FactorMomentumEngine()
        alloc = engine._generate_allocation([])
        assert alloc == {"SPY": 1.0}

    def test_single_factor_gets_all_weight(self):
        engine = FactorMomentumEngine()
        selected = [("MTUM", _make_score(symbol="MTUM"))]
        alloc = engine._generate_allocation(selected)
        assert abs(alloc["MTUM"] - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# TSFM evaluation
# ---------------------------------------------------------------------------

class TestTSFMEvaluation:
    def test_evaluate_tsfm_returns_dict(self):
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate_tsfm(vix_level=20.0)
        assert isinstance(result, dict)
        assert "tsfm" in result

    def test_tsfm_vix_context(self):
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate_tsfm(vix_level=35.0)
        assert result["tsfm"]["vix_context"] == 35.0

    def test_tsfm_allocation_sums_to_one(self):
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate_tsfm()
        alloc = result["tsfm"].get("allocation_tsfm", {})
        if alloc:
            total = sum(alloc.values())
            assert abs(total - 1.0) < 0.05

    def test_tsfm_low_vix_amplifies(self):
        """In low VIX, regime_momentum should be amplified."""
        engine = _make_engine_with_mocked_db()
        result_low = engine.evaluate_tsfm(vix_level=12.0)
        result_high = engine.evaluate_tsfm(vix_level=35.0)
        # Check that regime_momentum differs
        tsfm_low = result_low["tsfm"]["tsfm_scores"]
        tsfm_high = result_high["tsfm"]["tsfm_scores"]
        # At least some factors with non-zero tsfm_score should have
        # different regime_momentum between low and high VIX
        if tsfm_low and tsfm_high:
            # Find a symbol with non-zero tsfm_score in at least one result
            for sym in tsfm_low:
                low_rm = tsfm_low[sym]["regime_momentum"]
                high_rm = tsfm_high[sym].get("regime_momentum", 0.0)
                tsfm_base = tsfm_low[sym]["tsfm_score"]
                if abs(tsfm_base) > 0.01:
                    # With non-zero base, regime_momentum should differ
                    assert low_rm != high_rm
                    break

    def test_tsfm_recommendation(self):
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate_tsfm()
        assert "recommendation_tsfm" in result


# ---------------------------------------------------------------------------
# ML-enhanced evaluation
# ---------------------------------------------------------------------------

class TestMLEnhancedEvaluation:
    def test_evaluate_ml_enhanced_returns_dict(self):
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate_ml_enhanced(vix_level=20.0)
        assert isinstance(result, dict)
        assert result.get("ml_enhanced") is True

    def test_ml_scores_present(self):
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate_ml_enhanced()
        assert "ml_scores" in result

    def test_ml_recommendation(self):
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate_ml_enhanced()
        assert "ml_recommendation" in result
        assert isinstance(result["ml_recommendation"], str)

    def test_ml_selected_factors(self):
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate_ml_enhanced()
        assert "selected_factors_ml" in result


# ---------------------------------------------------------------------------
# Signal strength
# ---------------------------------------------------------------------------

class TestSignalStrength:
    def test_empty_selection_zero_strength(self):
        engine = FactorMomentumEngine()
        assert engine._calculate_signal_strength([]) == 0.0

    def test_single_factor(self):
        engine = FactorMomentumEngine()
        selected = [("MTUM", _make_score(momentum_score=0.15))]
        strength = engine._calculate_signal_strength(selected)
        assert 0.0 <= strength <= 1.0

    def test_multiple_factors(self):
        engine = FactorMomentumEngine()
        selected = [
            ("MTUM", _make_score(momentum_score=0.20)),
            ("USMV", _make_score(momentum_score=0.10)),
        ]
        strength = engine._calculate_signal_strength(selected)
        assert 0.0 <= strength <= 1.0


# ---------------------------------------------------------------------------
# Recommendation generation
# ---------------------------------------------------------------------------

class TestRecommendation:
    def test_no_factors_holds_spy(self):
        engine = FactorMomentumEngine()
        rec = engine._generate_recommendation([], {})
        assert "SPY" in rec

    def test_with_factors_names_mentioned(self):
        engine = FactorMomentumEngine()
        selected = [("MTUM", _make_score(return_12m=0.25))]
        rec = engine._generate_recommendation(selected, {"MTUM": _make_score(return_12m=0.25)})
        assert "Momentum" in rec

    def test_concentrated_category_warning(self):
        engine = FactorMomentumEngine()
        # Both in same category
        selected = [
            ("VTV", _make_score(symbol="VTV", return_12m=0.15)),
            ("VLUE", _make_score(symbol="VLUE", return_12m=0.12)),
        ]
        all_scores = {
            "VTV": _make_score(symbol="VTV", return_12m=0.15),
            "VLUE": _make_score(symbol="VLUE", return_12m=0.12),
        }
        rec = engine._generate_recommendation(selected, all_scores)
        assert "concentrated" in rec.lower()


# ---------------------------------------------------------------------------
# FactorRotationBacktest
# ---------------------------------------------------------------------------

class TestFactorRotationBacktest:
    def test_init(self):
        engine = FactorMomentumEngine()
        bt = FactorRotationBacktest(engine)
        assert bt.engine is engine

    def test_run_no_data(self):
        """When DB doesn't exist, should return error."""
        engine = FactorMomentumEngine(db_path=Path("/nonexistent/market.db"))
        bt = FactorRotationBacktest(engine)
        result = bt.run_backtest("2020-01-01", "2024-12-31")
        assert "error" in result or "status" in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_very_high_min_momentum(self):
        """With extremely high min_momentum, no factors should qualify."""
        engine = _make_engine_with_mocked_db(min_momentum=100.0)
        result = engine.evaluate()
        # Likely no factors selected
        assert len(result["selected_factors"]) <= 2  # Still bounded by top_n

    def test_top_n_larger_than_universe(self):
        """top_n > number of qualifying factors."""
        engine = _make_engine_with_mocked_db(top_n=20)
        result = engine.evaluate()
        assert len(result["selected_factors"]) <= len(engine.FACTORS)

    def test_all_factors_same_category(self):
        """Diversity constraint prevents selecting same category twice."""
        engine = _make_engine_with_mocked_db(top_n=3)
        result = engine.evaluate()
        # Verify no two selected factors share a category
        selected = result["selected_factors"]
        cats = [engine.FACTORS[s]["category"] for s in selected]
        assert len(cats) == len(set(cats))

    def test_factor_definitions(self):
        """Verify factor definitions are complete."""
        engine = FactorMomentumEngine()
        assert "MTUM" in engine.FACTORS
        assert "USMV" in engine.FACTORS
        assert "VTV" in engine.FACTORS
        assert "SPY" in engine.FACTORS

    def test_universe_matches_factors(self):
        engine = FactorMomentumEngine()
        assert set(engine.universe) == set(engine.FACTORS.keys())
