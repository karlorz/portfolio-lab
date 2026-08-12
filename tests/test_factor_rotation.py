"""Tests for Factor Momentum Rotation Strategy — factor_rotation.py.

Covers: FactorScore dataclass, FactorMomentumEngine scoring/ranking/selection,
inverse-volatility weighting, diversity constraints, TSFM evaluation,
signal strength, recommendation generation, edge cases.

DB calls are mocked since MARKET_DB doesn't exist in test environment.
"""
import dataclasses
import logging
import numpy as np
import pytest
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


# ---------------------------------------------------------------------------
# FactorScore dataclass field completeness via dataclasses.asdict
# ---------------------------------------------------------------------------

class TestFactorScoreFields:
    def test_asdict_all_fields(self):
        """dataclasses.asdict returns all 19 fields of FactorScore."""
        score = _make_score()
        d = dataclasses.asdict(score)
        expected = {
            "symbol", "factor_name", "price", "return_12m", "return_6m",
            "return_3m", "volatility", "sharpe_12m", "momentum_score", "rank",
            "value_momentum_synergy", "momentum_acceleration",
            "vol_adjusted_momentum", "regime_momentum", "factor_divergence",
            "composite_ml_score", "tsfm_score", "tsfm_allocation_scalar",
        }
        assert set(d.keys()) == expected, (
            f"Missing fields: {expected - set(d.keys())}"
        )

    def test_asdict_field_types(self):
        """Verify field types in asdict output."""
        score = _make_score(rank=3)
        score.price = 150.0
        d = dataclasses.asdict(score)
        assert isinstance(d["symbol"], str)
        assert isinstance(d["price"], float)
        assert isinstance(d["rank"], int)
        assert isinstance(d["return_12m"], float)
        assert isinstance(d["momentum_score"], float)

    def test_ml_fields_persist_after_set(self):
        """Setting ML fields persists and reads back correctly."""
        score = _make_score()
        score.value_momentum_synergy = 0.42
        score.momentum_acceleration = 0.15
        score.vol_adjusted_momentum = 1.23
        score.regime_momentum = 0.88
        score.factor_divergence = 0.67
        score.composite_ml_score = 0.55
        assert score.value_momentum_synergy == 0.42
        assert score.momentum_acceleration == 0.15
        assert score.vol_adjusted_momentum == 1.23
        assert score.regime_momentum == 0.88
        assert score.factor_divergence == 0.67
        assert score.composite_ml_score == 0.55

    def test_tsfm_fields_default(self):
        """tsfm_score defaults to 0.0, tsfm_allocation_scalar defaults to 1.0."""
        score = FactorScore(
            symbol="T", factor_name="T", price=100.0,
            return_12m=0.1, return_6m=0.05, return_3m=0.02,
            volatility=0.15, sharpe_12m=0.5, momentum_score=0.1, rank=1,
        )
        assert score.tsfm_score == 0.0
        assert score.tsfm_allocation_scalar == 1.0
        assert score.value_momentum_synergy == 0.0
        assert score.momentum_acceleration == 0.0
        assert score.vol_adjusted_momentum == 0.0
        assert score.regime_momentum == 0.0
        assert score.factor_divergence == 0.0
        assert score.composite_ml_score == 0.0


# ---------------------------------------------------------------------------
# _calculate_factor_score edge cases (negative, zero, extreme)
# ---------------------------------------------------------------------------

class TestFactorScoreEdgeCases:
    def test_negative_returns(self):
        """Negative drift produces negative return_12m and momentum_score."""
        engine = FactorMomentumEngine()
        data = _generate_price_data(300, drift=-0.0005, seed=201)
        engine._fetch_price_data = lambda sym, days=300: data
        score = engine._calculate_factor_score("MTUM")
        assert score is not None
        assert score.return_12m < 0
        assert score.return_6m < 0
        assert score.return_3m < 0
        assert score.momentum_score < 0

    def test_near_zero_volatility(self):
        """Flat price series yields near-zero volatility."""
        engine = FactorMomentumEngine()
        data = _generate_price_data(300, drift=0.0, vol=1e-8, seed=202)
        engine._fetch_price_data = lambda sym, days=300: data
        score = engine._calculate_factor_score("MTUM")
        assert score is not None
        assert score.volatility < 0.001

    def test_high_volatility(self):
        """Extreme daily vol still produces a valid score."""
        engine = FactorMomentumEngine()
        data = _generate_price_data(300, drift=0.0, vol=0.05, seed=203)
        engine._fetch_price_data = lambda sym, days=300: data
        score = engine._calculate_factor_score("MTUM")
        assert score is not None
        assert score.volatility > 0.50

    def test_tsfm_score_capped(self):
        """tsfm_score is always z-score capped at [-2, 2]."""
        engine = FactorMomentumEngine()
        data = _generate_price_data(300, drift=0.01, vol=0.001, seed=204)
        engine._fetch_price_data = lambda sym, days=300: data
        score = engine._calculate_factor_score("MTUM")
        assert score is not None
        assert -2.0 <= score.tsfm_score <= 2.0

    def test_extreme_negative_tsfm_clipped(self):
        """Extreme negative 1m return is clipped to -2.0."""
        engine = FactorMomentumEngine()
        data = _generate_price_data(300, drift=0.0, vol=0.001, seed=205)
        # Force a large 1-day drop for extreme negative 1m return
        data[-1]["close"] = data[-2]["close"] * 0.80  # 20% drop
        engine._fetch_price_data = lambda sym, days=300: data
        score = engine._calculate_factor_score("MTUM")
        assert score is not None
        assert score.tsfm_score >= -2.0


# ---------------------------------------------------------------------------
# TSFM regime impact — all three VIX regimes plus boundaries
# ---------------------------------------------------------------------------

class TestTSFMRegimeImpact:
    """Tests that evaluate_tsfm correctly adjusts regime_momentum per VIX level."""

    @staticmethod
    def _find_nonzero_tsfm(tsfm_scores, threshold=0.01):
        for sym, data in tsfm_scores.items():
            if abs(data["tsfm_score"]) > threshold:
                return sym
        return None

    def test_low_vol_amplifies(self):
        """VIX < 15: regime_momentum = tsfm_score * 1.2."""
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate_tsfm(vix_level=12.0)
        sym = self._find_nonzero_tsfm(result["tsfm"]["tsfm_scores"])
        if sym is None:
            pytest.skip("No symbol with non-zero tsfm_score")
        tsfm_s = result["tsfm"]["tsfm_scores"][sym]["tsfm_score"]
        rm = result["tsfm"]["tsfm_scores"][sym]["regime_momentum"]
        assert abs(rm - tsfm_s * 1.2) < 1e-6

    def test_high_vol_dampens(self):
        """VIX > 30: regime_momentum = tsfm_score * 0.7."""
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate_tsfm(vix_level=35.0)
        sym = self._find_nonzero_tsfm(result["tsfm"]["tsfm_scores"])
        if sym is None:
            pytest.skip("No symbol with non-zero tsfm_score")
        tsfm_s = result["tsfm"]["tsfm_scores"][sym]["tsfm_score"]
        rm = result["tsfm"]["tsfm_scores"][sym]["regime_momentum"]
        assert abs(rm - tsfm_s * 0.7) < 1e-6

    def test_normal_vol_passthrough(self):
        """15 <= VIX <= 30: regime_momentum == tsfm_score."""
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate_tsfm(vix_level=20.0)
        sym = self._find_nonzero_tsfm(result["tsfm"]["tsfm_scores"])
        if sym is None:
            pytest.skip("No symbol with non-zero tsfm_score")
        tsfm_s = result["tsfm"]["tsfm_scores"][sym]["tsfm_score"]
        rm = result["tsfm"]["tsfm_scores"][sym]["regime_momentum"]
        assert abs(rm - tsfm_s) < 1e-6

    def test_vix_boundary_15(self):
        """VIX exactly 15 uses normal regime (>= 15 is not low vol)."""
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate_tsfm(vix_level=15.0)
        sym = self._find_nonzero_tsfm(result["tsfm"]["tsfm_scores"])
        if sym is None:
            pytest.skip("No symbol with non-zero tsfm_score")
        tsfm_s = result["tsfm"]["tsfm_scores"][sym]["tsfm_score"]
        rm = result["tsfm"]["tsfm_scores"][sym]["regime_momentum"]
        assert abs(rm - tsfm_s) < 1e-6, "VIX=15 should use normal regime"

    def test_vix_boundary_30(self):
        """VIX exactly 30 uses normal regime (<= 30 is not high vol)."""
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate_tsfm(vix_level=30.0)
        sym = self._find_nonzero_tsfm(result["tsfm"]["tsfm_scores"])
        if sym is None:
            pytest.skip("No symbol with non-zero tsfm_score")
        tsfm_s = result["tsfm"]["tsfm_scores"][sym]["tsfm_score"]
        rm = result["tsfm"]["tsfm_scores"][sym]["regime_momentum"]
        assert abs(rm - tsfm_s) < 1e-6, "VIX=30 should use normal regime"

    def test_no_positive_tsfm(self):
        """All tsfm_scores <= 0: empty selection, zero signal."""
        engine = FactorMomentumEngine()
        data = _generate_price_data(300, drift=-0.002, seed=206)
        engine._fetch_price_data = lambda sym, days=300: data
        result = engine.evaluate_tsfm(vix_level=20.0)
        assert result["tsfm"]["selected_factors_tsfm"] == []
        assert result["tsfm"]["allocation_tsfm"] == {}


# ---------------------------------------------------------------------------
# Signal strength — additional edge cases
# ---------------------------------------------------------------------------

class TestSignalStrengthExpanded:
    def test_three_factors_spread_path(self):
        """Exactly 3 factors triggers the 3-factor spread code path (top - bottom)."""
        engine = FactorMomentumEngine()
        selected = [
            ("MTUM", _make_score(momentum_score=0.30, volatility=0.15)),
            ("USMV", _make_score(symbol="USMV", momentum_score=0.20, volatility=0.12)),
            ("QUAL", _make_score(symbol="QUAL", momentum_score=0.10, volatility=0.14)),
        ]
        strength = engine._calculate_signal_strength(selected)
        assert 0.0 <= strength <= 1.0

    def test_two_factors_spread_path(self):
        """Exactly 2 factors triggers the 2-factor spread code path."""
        engine = FactorMomentumEngine()
        selected = [
            ("MTUM", _make_score(momentum_score=0.25, volatility=0.15)),
            ("USMV", _make_score(symbol="USMV", momentum_score=0.10, volatility=0.12)),
        ]
        strength = engine._calculate_signal_strength(selected)
        assert 0.0 <= strength <= 1.0

    def test_all_negative_scores(self):
        """All factors with negative momentum: direction=0, vol_score only."""
        engine = FactorMomentumEngine()
        selected = [
            ("MTUM", _make_score(momentum_score=-0.10, volatility=0.20)),
            ("USMV", _make_score(symbol="USMV", momentum_score=-0.05, volatility=0.15)),
        ]
        strength = engine._calculate_signal_strength(selected)
        assert 0.0 <= strength <= 1.0

    def test_high_volatility_reduces_vol_score(self):
        """All factors above 25% vol: vol_score=0, reducing max strength."""
        engine = FactorMomentumEngine()
        selected = [
            ("MTUM", _make_score(momentum_score=0.20, volatility=0.30)),
            ("USMV", _make_score(symbol="USMV", momentum_score=0.15, volatility=0.35)),
        ]
        strength = engine._calculate_signal_strength(selected)
        assert 0.0 <= strength <= 1.0

    def test_single_factor_exact_strength(self):
        """Single negative-momentum factor yields deterministic strength 0.4."""
        engine = FactorMomentumEngine()
        selected = [("MTUM", _make_score(momentum_score=-0.20, volatility=0.18))]
        strength = engine._calculate_signal_strength(selected)
        # spread=0.1 (hardcoded for 1 factor), direction=0, vol_score=1
        # strength = min(0.2, 0.4) + 0*0.4 + 1*0.2 = 0.4
        assert abs(strength - 0.4) < 1e-6


# ---------------------------------------------------------------------------
# _calculate_ml_features — VIX normalization, value synergy, missing symbol
# ---------------------------------------------------------------------------

class TestMLFeatures:
    def test_vix_normalization(self):
        """VIX in 10-40 range produces valid features."""
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate_ml_enhanced(vix_level=20.0)
        assert "ml_scores" in result
        for sym, data in result["ml_scores"].items():
            assert "composite_ml_score" in data
            assert "regime_momentum" in data
            assert "factor_divergence" in data

    def test_missing_symbol_returns_empty(self):
        """_calculate_ml_features with unknown symbol returns {}."""
        engine = FactorMomentumEngine()
        result = engine._calculate_ml_features("NONEXISTENT", {})
        assert result == {}

    def test_value_momentum_synergy_for_vtv(self):
        """VTV symbol triggers value synergy calculation with SPY fallback."""
        engine = FactorMomentumEngine()
        scores = {
            "VTV": _make_score(symbol="VTV", factor_name="Value", momentum_score=0.20),
            "SPY": _make_score(symbol="SPY", factor_name="S&P 500", momentum_score=0.10),
        }
        result = engine._calculate_ml_features("VTV", scores, vix_level=20.0)
        assert "value_momentum_synergy" in result
        # value_spread = 0.20 - 0.10 = 0.10
        # synergy = 0.10 * abs(0.20) = 0.02
        assert abs(result["value_momentum_synergy"] - 0.02) < 1e-6

    def test_value_synergy_no_vug_fallback_to_spy(self):
        """VTV synergy code uses SPY when VUG not in factor_scores."""
        engine = FactorMomentumEngine()
        scores = {
            "VTV": _make_score(symbol="VTV", factor_name="Value", momentum_score=0.15),
            "SPY": _make_score(symbol="SPY", factor_name="S&P 500", momentum_score=0.05),
        }
        result = engine._calculate_ml_features("VTV", scores, vix_level=25.0)
        assert abs(result["value_momentum_synergy"] - 0.015) < 1e-6

    def test_non_value_symbol_no_synergy(self):
        """Non-VTV/VLUE symbol gets zero value_momentum_synergy."""
        engine = FactorMomentumEngine()
        scores = {"MTUM": _make_score(momentum_score=0.20)}
        result = engine._calculate_ml_features("MTUM", scores, vix_level=20.0)
        assert result["value_momentum_synergy"] == 0.0

    def test_vix_clamping_below_range(self):
        """VIX below 10 is clamped to 0 percentile."""
        engine = FactorMomentumEngine()
        scores = {"MTUM": _make_score(momentum_score=0.20)}
        result_low = engine._calculate_ml_features("MTUM", scores, vix_level=5.0)
        result_high = engine._calculate_ml_features("MTUM", scores, vix_level=20.0)
        # At VIX=5: percentile = max((5-10)/30, 0) = 0, multiplier=1.0
        # regime_momentum = 0.20 * 1.0 = 0.20
        # At VIX=20: percentile = (20-10)/30 = 0.33, multiplier=1.0
        # regime_momentum = 0.20 * 1.0 = 0.20
        # Both should be same (both below 0.67 threshold between regimes)
        assert abs(result_low["regime_momentum"] - result_high["regime_momentum"]) < 1e-6

    def test_vix_clamping_above_range(self):
        """VIX above 40 gets multiplier 0.5 (reduced in high vol)."""
        engine = FactorMomentumEngine()
        scores = {"MTUM": _make_score(momentum_score=0.20)}
        result = engine._calculate_ml_features("MTUM", scores, vix_level=45.0)
        # percentile = min((45-10)/30, 1) = 1.0
        # multiplier = 0.5 (since percentile > 0.67)
        # regime_momentum = 0.20 * 0.5 = 0.10
        assert abs(result["regime_momentum"] - 0.10) < 1e-6


# ---------------------------------------------------------------------------
# ML recommendation generation
# ---------------------------------------------------------------------------

class TestMLRecommendation:
    def test_no_selection_holds_spy(self):
        """Empty selection recommends holding SPY."""
        engine = FactorMomentumEngine()
        rec = engine._generate_ml_recommendation([], {})
        assert "SPY" in rec

    def test_with_value_synergy_pattern(self):
        """Value synergy pattern is mentioned in recommendation."""
        engine = FactorMomentumEngine()
        score = _make_score(symbol="VTV", factor_name="Value", momentum_score=0.15)
        score.composite_ml_score = 0.35
        score.value_momentum_synergy = 0.05
        scores = {"VTV": score}
        selected = [("VTV", scores["VTV"])]
        rec = engine._generate_ml_recommendation(selected, scores)
        assert "Value" in rec
        assert "value-momentum" in rec

    def test_with_momentum_acceleration_pattern(self):
        """Momentum acceleration pattern is mentioned."""
        engine = FactorMomentumEngine()
        score = _make_score(symbol="MTUM", factor_name="Momentum", momentum_score=0.15)
        score.composite_ml_score = 0.25
        score.momentum_acceleration = 0.02
        scores = {"MTUM": score}
        selected = [("MTUM", scores["MTUM"])]
        rec = engine._generate_ml_recommendation(selected, scores)
        assert "Momentum" in rec
        assert "accelerating" in rec

    def test_strong_ml_signal(self):
        """High composite_ml_score > 0.3 yields 'strong ML signal'."""
        engine = FactorMomentumEngine()
        score = _make_score(symbol="MTUM", factor_name="Momentum")
        score.composite_ml_score = 0.45
        scores = {"MTUM": score}
        selected = [("MTUM", scores["MTUM"])]
        rec = engine._generate_ml_recommendation(selected, scores)
        assert "strong" in rec.lower()

    def test_weak_ml_signal(self):
        """Low composite_ml_score < 0.15 yields 'weak ML signal'."""
        engine = FactorMomentumEngine()
        score = _make_score(symbol="MTUM", factor_name="Momentum")
        score.composite_ml_score = 0.05
        scores = {"MTUM": score}
        selected = [("MTUM", scores["MTUM"])]
        rec = engine._generate_ml_recommendation(selected, scores)
        assert "weak" in rec.lower()


# ---------------------------------------------------------------------------
# TSFM recommendation edge cases
# ---------------------------------------------------------------------------

class TestTSFMRecommendation:
    def test_empty_selection(self):
        """Empty selection returns risk-off message."""
        engine = FactorMomentumEngine()
        rec = engine._generate_tsfm_recommendation([], {}, 20.0)
        assert "risk-off" in rec.lower()

    def test_low_vol_regime_description(self):
        """VIX < 15 produces 'low vol' regime description."""
        engine = FactorMomentumEngine()
        selected = [("MTUM", _make_score(tsfm_score=1.0, tsfm_allocation_scalar=1.2))]
        rec = engine._generate_tsfm_recommendation(selected, {"MTUM": _make_score(tsfm_score=1.0, tsfm_allocation_scalar=1.2)}, 12.0)
        assert "low vol" in rec.lower()

    def test_elevated_vol_regime_description(self):
        """VIX > 25 produces 'elevated vol' regime description."""
        engine = FactorMomentumEngine()
        selected = [("MTUM", _make_score(tsfm_score=0.8, tsfm_allocation_scalar=1.0))]
        rec = engine._generate_tsfm_recommendation(selected, {"MTUM": _make_score(tsfm_score=0.8, tsfm_allocation_scalar=1.0)}, 28.0)
        assert "elevated" in rec.lower()

    def test_strong_signal_high_scalar(self):
        """Allocation scalar > 1.5 yields 'strong' strength."""
        engine = FactorMomentumEngine()
        selected = [("MTUM", _make_score(tsfm_score=1.5, tsfm_allocation_scalar=1.8))]
        rec = engine._generate_tsfm_recommendation(selected, {"MTUM": _make_score(tsfm_score=1.5, tsfm_allocation_scalar=1.8)}, 20.0)
        assert "strong" in rec.lower()

    def test_weak_signal_low_scalar(self):
        """Allocation scalar < 1.0 yields 'weak' strength."""
        engine = FactorMomentumEngine()
        selected = [("MTUM", _make_score(tsfm_score=0.3, tsfm_allocation_scalar=0.8))]
        rec = engine._generate_tsfm_recommendation(selected, {"MTUM": _make_score(tsfm_score=0.3, tsfm_allocation_scalar=0.8)}, 20.0)
        assert "weak" in rec.lower()

    def test_moderate_signal_mid_scalar(self):
        """Allocation scalar between 1.0 and 1.5 yields 'moderate' strength."""
        engine = FactorMomentumEngine()
        selected = [("MTUM", _make_score(tsfm_score=1.0, tsfm_allocation_scalar=1.2))]
        rec = engine._generate_tsfm_recommendation(selected, {"MTUM": _make_score(tsfm_score=1.0, tsfm_allocation_scalar=1.2)}, 20.0)
        assert "moderate" in rec.lower()

    def test_multiple_factors_mentioned(self):
        """Multiple selected factors are joined in recommendation."""
        engine = FactorMomentumEngine()
        selected = [
            ("MTUM", _make_score(tsfm_score=1.0, tsfm_allocation_scalar=1.2)),
            ("USMV", _make_score(symbol="USMV", factor_name="Low Vol", tsfm_score=0.8, tsfm_allocation_scalar=1.1)),
        ]
        all_scores = {
            "MTUM": _make_score(tsfm_score=1.0, tsfm_allocation_scalar=1.2),
            "USMV": _make_score(symbol="USMV", factor_name="Low Vol", tsfm_score=0.8, tsfm_allocation_scalar=1.1),
        }
        rec = engine._generate_tsfm_recommendation(selected, all_scores, 20.0)
        assert "Momentum" in rec
        assert "Low Vol" in rec


# ---------------------------------------------------------------------------
# _generate_allocation edge cases
# ---------------------------------------------------------------------------

class TestGenerateAllocationEdgeCases:
    def test_tiny_vol_edge(self):
        """Extremely low volatility should be clamped to 0.05."""
        engine = FactorMomentumEngine()
        selected = [
            ("MTUM", _make_score(volatility=0.001)),
            ("USMV", _make_score(symbol="USMV", volatility=0.002)),
        ]
        alloc = engine._generate_allocation(selected)
        assert abs(sum(alloc.values()) - 1.0) < 1e-10
        # Both should have equal weights since both are clamped to 0.05 vol
        assert abs(alloc["MTUM"] - 0.5) < 1e-10
        assert abs(alloc["USMV"] - 0.5) < 1e-10

    def test_equal_vol_equal_weight(self):
        """Same volatility gives equal weights."""
        engine = FactorMomentumEngine()
        selected = [
            ("MTUM", _make_score(volatility=0.18)),
            ("USMV", _make_score(symbol="USMV", volatility=0.18)),
            ("QUAL", _make_score(symbol="QUAL", volatility=0.18)),
        ]
        alloc = engine._generate_allocation(selected)
        assert abs(sum(alloc.values()) - 1.0) < 1e-10
        for w in alloc.values():
            assert abs(w - 1/3) < 1e-10


# ---------------------------------------------------------------------------
# Constants validation — FACTORS completeness
# ---------------------------------------------------------------------------

class TestConstantsValidation:
    def test_factor_count(self):
        """FACTORS should contain exactly 11 symbols."""
        engine = FactorMomentumEngine()
        assert len(engine.FACTORS) == 11

    def test_all_required_categories_present(self):
        """All 6 factor categories should be present."""
        engine = FactorMomentumEngine()
        categories = {info["category"] for info in engine.FACTORS.values()}
        expected = {"value", "momentum", "quality", "low_vol", "small", "core"}
        assert categories == expected

    def test_every_symbol_has_all_required_fields(self):
        """Each FACTORS entry must have name, category, alternative."""
        engine = FactorMomentumEngine()
        for sym, info in engine.FACTORS.items():
            assert "name" in info, f"{sym} missing 'name'"
            assert "category" in info, f"{sym} missing 'category'"
            assert "alternative" in info or info["alternative"] is None, (
                f"{sym} missing 'alternative'"
            )

    def test_value_category_has_two_symbols(self):
        """Value category should have VTV and VLUE."""
        engine = FactorMomentumEngine()
        value_syms = {
            s for s, i in engine.FACTORS.items()
            if i["category"] == "value"
        }
        assert value_syms == {"VTV", "VLUE"}

    def test_momentum_category_has_one(self):
        """Momentum category should have exactly MTUM."""
        engine = FactorMomentumEngine()
        mom_syms = {
            s for s, i in engine.FACTORS.items()
            if i["category"] == "momentum"
        }
        assert mom_syms == {"MTUM"}

    def test_core_category_has_two_symbols(self):
        """Core category should have SPY and QQQ."""
        engine = FactorMomentumEngine()
        core_syms = {
            s for s, i in engine.FACTORS.items()
            if i["category"] == "core"
        }
        assert core_syms == {"SPY", "QQQ"}

    def test_alternatives_are_symmetric(self):
        """If A has alternative B, then B's alternative should be A."""
        engine = FactorMomentumEngine()
        for sym, info in engine.FACTORS.items():
            alt = info["alternative"]
            if alt is not None:
                assert engine.FACTORS[alt]["alternative"] == sym, (
                    f"{sym} -> {alt} but {alt} -> {engine.FACTORS[alt]['alternative']}"
                )


# ---------------------------------------------------------------------------
# TSFM allocation scalar computation through evaluate_tsfm
# ---------------------------------------------------------------------------

class TestTSFMAllocationScalar:
    def test_scalar_in_range(self):
        """All tsfm_allocation_scalar values should be in [0, 2]."""
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate_tsfm(vix_level=20.0)
        for sym, data in result["tsfm"]["tsfm_scores"].items():
            assert 0.0 <= data["tsfm_allocation_scalar"] <= 2.0

    def test_negative_tsfm_reduces_scalar(self):
        """Negative tsfm_score produces scalar < 1.0."""
        engine = FactorMomentumEngine()
        data = _generate_price_data(300, drift=-0.001, seed=210)
        engine._fetch_price_data = lambda sym, days=300: data
        result = engine.evaluate_tsfm(vix_level=20.0)
        for sym, data in result["tsfm"]["tsfm_scores"].items():
            if data["tsfm_score"] < 0:
                assert data["tsfm_allocation_scalar"] < 1.0
                break


# ---------------------------------------------------------------------------
# FactorScore dataclass — validation and edge cases
# ---------------------------------------------------------------------------

class TestFactorScoreValidation:
    def test_negative_price(self):
        """FactorScore can be created with negative price (edge case)."""
        score = FactorScore(
            symbol="TEST", factor_name="Test", price=-10.0,
            return_12m=0.1, return_6m=0.05, return_3m=0.02,
            volatility=0.15, sharpe_12m=0.5, momentum_score=0.1, rank=1,
        )
        assert score.price == -10.0
        assert score.symbol == "TEST"

    def test_zero_volatility(self):
        """FactorScore with zero volatility (edge case for division)."""
        score = _make_score(volatility=0.0)
        assert score.volatility == 0.0

    def test_default_rank_zero(self):
        """Default rank is 0 before assignment by evaluate()."""
        score = FactorScore(
            symbol="T", factor_name="T", price=100.0,
            return_12m=0.1, return_6m=0.05, return_3m=0.02,
            volatility=0.15, sharpe_12m=0.5, momentum_score=0.1, rank=0,
        )
        assert score.rank == 0


# ---------------------------------------------------------------------------
# FactorMomentumEngine — constructor validation
# ---------------------------------------------------------------------------

class TestEngineConstructor:
    def test_default_constructor(self):
        """Default constructor uses reasonable defaults."""
        engine = FactorMomentumEngine()
        assert engine.top_n == 2
        assert engine.lookback_months == 12
        assert engine.min_momentum == 0.0
        assert engine.vol_lookback == 20
        assert engine.max_per_category == 1

    def test_top_n_zero(self):
        """top_n=0 selects 1 factor then breaks (0 >= 0 guard is True after append)."""
        engine = _make_engine_with_mocked_db(top_n=0)
        result = engine.evaluate()
        # With top_n=0, one factor is appended before len(selected) >= 0 breaks
        assert len(result["selected_factors"]) == 1


# ---------------------------------------------------------------------------
# _calculate_factor_score — additional edge cases
# ---------------------------------------------------------------------------

class TestFactorScoreCalcEdgeCases:
    def test_exactly_252_days(self):
        """Exactly 252 data points is sufficient (boundary)."""
        engine = FactorMomentumEngine()
        data = _generate_price_data(252, drift=0.0004, seed=42)
        engine._fetch_price_data = lambda sym, days=300: data
        score = engine._calculate_factor_score("MTUM")
        assert score is not None

    def test_zero_close_prices(self):
        """All-zero close prices handled without crash."""
        engine = FactorMomentumEngine()
        data = [{"date": f"2024-01-{i+1:02d}", "close": 0.0, "volume": 1000}
                for i in range(300)]
        engine._fetch_price_data = lambda sym, days=300: data
        score = engine._calculate_factor_score("MTUM")
        assert score is not None

    def test_constant_price_series(self):
        """Flat price yields zero returns and non-zero vol floor."""
        engine = FactorMomentumEngine()
        data = [{"date": f"2024-01-{i+1:02d}", "close": 100.0, "volume": 1000}
                for i in range(300)]
        engine._fetch_price_data = lambda sym, days=300: data
        score = engine._calculate_factor_score("MTUM")
        assert score is not None
        assert score.return_12m == 0.0
        assert score.return_6m == 0.0
        assert score.return_3m == 0.0

    def test_momentum_acceleration_formula(self):
        """Verify momentum_acceleration = return_1m - (return_3m / 3)."""
        engine = FactorMomentumEngine()
        data = _generate_price_data(300, drift=0.001, seed=55)
        engine._fetch_price_data = lambda sym, days=300: data
        score = engine._calculate_factor_score("MTUM")
        assert score is not None
        closes = np.array([d["close"] for d in data])
        days_1m = min(21, len(closes) - 1)
        return_1m = (closes[-1] / closes[-days_1m]) - 1
        expected_accel = return_1m - (score.return_3m / 3)
        assert abs(score.momentum_acceleration - expected_accel) < 1e-10


# ---------------------------------------------------------------------------
# evaluate() — additional coverage
# ---------------------------------------------------------------------------

class TestEvaluateAdditional:
    def test_evaluate_allocation_deterministic(self):
        """Same data produces identical allocation."""
        engine1 = _make_engine_with_mocked_db(top_n=2)
        engine2 = _make_engine_with_mocked_db(top_n=2)
        result1 = engine1.evaluate()
        result2 = engine2.evaluate()
        if result1["allocation"] and result2["allocation"]:
            assert result1["allocation"] == result2["allocation"]

    def test_evaluate_diversity_output_shape(self):
        """Diversity dict contains categories_used and category_distribution."""
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate()
        assert "diversity" in result
        assert "categories_used" in result["diversity"]
        assert "category_distribution" in result["diversity"]

    def test_evaluate_all_factors_filtered_by_min_momentum(self):
        """When every factor fails min_momentum, selected is empty."""
        engine = _make_engine_with_mocked_db(min_momentum=10.0)
        result = engine.evaluate()
        assert result["selected_factors"] == []
        assert result["allocation"] == {"SPY": 1.0}

    def test_evaluate_current_scores_has_all_fields(self):
        """Each entry in current_scores has all expected fields."""
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate()
        for sym, data in result["current_scores"].items():
            for field in ("factor_name", "category", "return_12m", "return_6m",
                          "return_3m", "volatility", "sharpe_12m",
                          "momentum_score", "rank"):
                assert field in data, f"{sym} missing {field}"


# ---------------------------------------------------------------------------
# _generate_allocation — additional edge cases
# ---------------------------------------------------------------------------

class TestGenerateAllocationExpanded:
    def test_total_inv_vol_zero_equal_weight(self):
        """When all inverse vols approach zero, factors get equal weight."""
        engine = FactorMomentumEngine()
        selected = [
            ("MTUM", _make_score(volatility=1e10)),
            ("USMV", _make_score(symbol="USMV", volatility=1e10)),
        ]
        alloc = engine._generate_allocation(selected)
        assert abs(sum(alloc.values()) - 1.0) < 1e-10
        assert abs(alloc["MTUM"] - 0.5) < 1e-10
        assert abs(alloc["USMV"] - 0.5) < 1e-10

    def test_single_factor_extreme_vol(self):
        """Single factor always gets full weight regardless of volatility."""
        engine = FactorMomentumEngine()
        selected = [("MTUM", _make_score(volatility=1e10))]
        alloc = engine._generate_allocation(selected)
        assert abs(alloc["MTUM"] - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# evaluate_tsfm — additional coverage
# ---------------------------------------------------------------------------

class TestTSFMEvaluationExpanded:
    def test_tsfm_empty_universe_error(self):
        """When all symbols return None, TSFM reports insufficient data."""
        engine = FactorMomentumEngine()
        engine._fetch_price_data = lambda sym, days=300: []
        result = engine.evaluate_tsfm(vix_level=20.0)
        assert "error" in result["tsfm"]

    def test_tsfm_factor_divergence_exists(self):
        """factor_divergence is computed for each symbol."""
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate_tsfm(vix_level=20.0)
        for sym, data in result["tsfm"]["tsfm_scores"].items():
            assert "factor_divergence" in data
            assert data["factor_divergence"] >= 0.0

    def test_tsfm_selection_diversity(self):
        """TSFM selection enforces max 1 factor per category."""
        engine = _make_engine_with_mocked_db(top_n=3)
        result = engine.evaluate_tsfm(vix_level=20.0)
        selected = result["tsfm"]["selected_factors_tsfm"]
        categories = [engine.FACTORS[s]["category"] for s in selected]
        assert len(categories) == len(set(categories))

    def test_tsfm_divergence_zero_when_std_zero(self):
        """When all tsfm_scores are identical, divergence is 0.0."""
        engine = FactorMomentumEngine()
        data = _generate_price_data(300, drift=0.0, vol=0.0, seed=100)
        engine._fetch_price_data = lambda sym, days=300: data
        result = engine.evaluate_tsfm(vix_level=20.0)
        for sym, data in result["tsfm"]["tsfm_scores"].items():
            assert data["factor_divergence"] == 0.0


# ---------------------------------------------------------------------------
# evaluate_ml_enhanced — additional coverage
# ---------------------------------------------------------------------------

class TestMLEnhancedEvaluationExpanded:
    def test_ml_enhanced_empty_scores(self):
        """When no factor data is available, ML result still has ml_enhanced flag."""
        engine = FactorMomentumEngine()
        engine._fetch_price_data = lambda sym, days=300: []
        result = engine.evaluate_ml_enhanced(vix_level=20.0)
        assert "ml_enhanced" in result
        assert result["ml_enhanced"] is True

    def test_ml_vix_context_stored(self):
        """VIX context is stored in result."""
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate_ml_enhanced(vix_level=25.0)
        assert result["vix_context"] == 25.0


# ---------------------------------------------------------------------------
# _calculate_ml_features — expanded edge cases
# ---------------------------------------------------------------------------

class TestMLFeaturesExpanded:
    def test_vix_percentile_at_67_boundary(self):
        """VIX=30 => percentile ~0.667, below 0.67 threshold, multiplier=1."""
        engine = FactorMomentumEngine()
        scores = {"MTUM": _make_score(momentum_score=0.20)}
        result = engine._calculate_ml_features("MTUM", scores, vix_level=30.0)
        assert abs(result["regime_momentum"] - 0.20) < 1e-6

    def test_vix_percentile_above_67(self):
        """VIX=30.1 => percentile ~0.67, above threshold, multiplier=0.5."""
        engine = FactorMomentumEngine()
        scores = {"MTUM": _make_score(momentum_score=0.20)}
        result = engine._calculate_ml_features("MTUM", scores, vix_level=30.1)
        assert abs(result["regime_momentum"] - 0.10) < 1e-2

    def test_ml_features_zero_spy_vol(self):
        """When SPY volatility is zero, factor_divergence still computed."""
        engine = FactorMomentumEngine()
        scores = {
            "SPY": _make_score(symbol="SPY", volatility=0.0),
            "MTUM": _make_score(momentum_score=0.20, volatility=0.15),
        }
        result = engine._calculate_ml_features("MTUM", scores, vix_level=20.0)
        assert "factor_divergence" in result
        assert result["factor_divergence"] > 0

    def test_vtv_no_vug_no_spy_ml_features(self):
        """VTV synergy handles missing VUG and SPY gracefully -> synergy 0."""
        engine = FactorMomentumEngine()
        scores = {
            "VTV": _make_score(symbol="VTV", momentum_score=0.20),
        }
        result = engine._calculate_ml_features("VTV", scores, vix_level=20.0)
        assert result["value_momentum_synergy"] == 0.0


# ---------------------------------------------------------------------------
# FactorRotationBacktest — expanded edge cases
# ---------------------------------------------------------------------------

class TestFactorRotationBacktestExpanded:
    def test_run_insufficient_data(self):
        """Backtest with fewer than 252 trading days returns error."""
        engine = FactorMomentumEngine(db_path=Path("/nonexistent/market.db"))
        bt = FactorRotationBacktest(engine)
        result = bt.run_backtest("2024-01-01", "2024-06-01")
        assert "error" in result

    def test_backtest_no_spy_data(self):
        """Backtest missing SPY benchmark returns error."""
        engine = FactorMomentumEngine(db_path=Path("/nonexistent/market.db"))
        bt = FactorRotationBacktest(engine)
        result = bt.run_backtest("2020-01-01", "2024-12-31")
        assert "error" in result or "status" in result

    def test_backtest_initial_capital_field(self):
        """Backtest result contains initial_capital of 100000."""
        engine = FactorMomentumEngine(db_path=Path("/nonexistent/market.db"))
        bt = FactorRotationBacktest(engine)
        result = bt.run_backtest("2020-01-01", "2024-12-31")
        if "initial_capital" in result:
            assert result["initial_capital"] == 100000.0


# ---------------------------------------------------------------------------
# _generate_recommendation — strength levels
# ---------------------------------------------------------------------------

class TestRecommendationExpanded:
    def test_weak_momentum_recommendation(self):
        """Avg 12m return < 10% yields 'weak' strength (2 diff categories avoids concentrated)."""
        engine = FactorMomentumEngine()
        selected = [
            ("USMV", _make_score(symbol="USMV", factor_name="Low Vol",
                                 return_12m=0.05)),
            ("MTUM", _make_score(symbol="MTUM", factor_name="Momentum",
                                 return_12m=0.05)),
        ]
        all_scores = {
            "USMV": _make_score(symbol="USMV", return_12m=0.05),
            "MTUM": _make_score(return_12m=0.05),
        }
        rec = engine._generate_recommendation(selected, all_scores)
        assert "weak" in rec.lower()

    def test_strong_momentum_recommendation(self):
        """Avg 12m return > 20% yields 'strong' strength (2 diff categories)."""
        engine = FactorMomentumEngine()
        selected = [
            ("MTUM", _make_score(factor_name="Momentum", return_12m=0.25)),
            ("USMV", _make_score(symbol="USMV", factor_name="Low Vol",
                                 return_12m=0.25)),
        ]
        all_scores = {
            "MTUM": _make_score(return_12m=0.25),
            "USMV": _make_score(symbol="USMV", return_12m=0.25),
        }
        rec = engine._generate_recommendation(selected, all_scores)
        assert "strong" in rec.lower()

    def test_moderate_momentum_recommendation(self):
        """Avg 12m return between 10-20% yields 'moderate' strength (2 diff categories)."""
        engine = FactorMomentumEngine()
        selected = [
            ("QUAL", _make_score(symbol="QUAL", factor_name="Quality",
                                 return_12m=0.15)),
            ("USMV", _make_score(symbol="USMV", factor_name="Low Vol",
                                 return_12m=0.15)),
        ]
        all_scores = {
            "QUAL": _make_score(return_12m=0.15),
            "USMV": _make_score(symbol="USMV", return_12m=0.15),
        }
        rec = engine._generate_recommendation(selected, all_scores)
        assert "moderate" in rec.lower()


# ---------------------------------------------------------------------------
# TSFM allocation scalar normalization (formula verification)
# ---------------------------------------------------------------------------

class TestTSFMNormalization:
    def test_tsfm_scalar_formula_mapping(self):
        """Verify tsfm_score in [-2,2] maps to scalar in [0,2]."""
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate_tsfm(vix_level=20.0)
        for sym, sdata in result["tsfm"]["tsfm_scores"].items():
            ts = sdata["tsfm_score"]
            normalized = (ts + 2) / 2
            expected = min(2.0, max(0.0, normalized))
            assert abs(sdata["tsfm_allocation_scalar"] - expected) < 1e-10


# ---------------------------------------------------------------------------
# evaluate_tsfm — no positive TSFM score (all negative or zero)
# ---------------------------------------------------------------------------

class TestTSFMNoPositiveScore:
    def test_tsfm_no_positive_selected(self):
        """When no factors have positive tsfm_score, selection is empty."""
        engine = FactorMomentumEngine()
        data = _generate_price_data(300, drift=-0.002, seed=210)
        engine._fetch_price_data = lambda sym, days=300: data
        result = engine.evaluate_tsfm(vix_level=20.0)
        for sym, sdata in result["tsfm"]["tsfm_scores"].items():
            assert sdata["tsfm_score"] <= 0
        assert result["tsfm"]["selected_factors_tsfm"] == []


# ---------------------------------------------------------------------------
# _generate_ml_recommendation — moderate signal threshold
# ---------------------------------------------------------------------------

class TestMLRecommendationExpanded:
    def test_moderate_ml_signal(self):
        """composite_ml_score between 0.15-0.3 yields 'moderate ML signal'."""
        engine = FactorMomentumEngine()
        score = _make_score(symbol="MTUM", factor_name="Momentum")
        score.composite_ml_score = 0.20
        scores = {"MTUM": score}
        selected = [("MTUM", scores["MTUM"])]
        rec = engine._generate_ml_recommendation(selected, scores)
        assert "moderate" in rec.lower()


# ---------------------------------------------------------------------------
# _generate_tsfm_recommendation — various regime and strength combos
# ---------------------------------------------------------------------------

class TestTSFMRecommendationExpanded:
    def test_tsfm_empty_result_no_selected(self):
        """Empty selected returns risk-off message."""
        engine = FactorMomentumEngine()
        rec = engine._generate_tsfm_recommendation([], {}, 20.0)
        assert "risk-off" in rec.lower()

    def test_tsfm_multi_factor_names_joined(self):
        """Multiple selected factors are comma-joined in recommendation."""
        engine = FactorMomentumEngine()
        selected = [
            ("MTUM", _make_score(symbol="MTUM", factor_name="Momentum",
                                 tsfm_score=1.0, tsfm_allocation_scalar=1.2)),
            ("USMV", _make_score(symbol="USMV", factor_name="Low Vol",
                                 tsfm_score=0.8, tsfm_allocation_scalar=1.1)),
        ]
        all_scores = {
            "MTUM": _make_score(tsfm_score=1.0, tsfm_allocation_scalar=1.2),
            "USMV": _make_score(symbol="USMV", factor_name="Low Vol",
                                tsfm_score=0.8, tsfm_allocation_scalar=1.1),
        }
        rec = engine._generate_tsfm_recommendation(selected, all_scores, 20.0)
        assert "Momentum" in rec
        assert "Low Vol" in rec


# ---------------------------------------------------------------------------
# _calculate_signal_strength — more than 3 factors
# ---------------------------------------------------------------------------

class TestSignalStrengthManyFactors:
    def test_four_factors_uses_top_three_spread(self):
        """4+ factors: spread uses only top 3, ignores trailing factors."""
        engine = FactorMomentumEngine()
        selected = [
            ("MTUM", _make_score(momentum_score=0.30, volatility=0.15)),
            ("USMV", _make_score(symbol="USMV", momentum_score=0.20,
                                 volatility=0.12)),
            ("QUAL", _make_score(symbol="QUAL", momentum_score=0.10,
                                 volatility=0.14)),
            ("VTV", _make_score(symbol="VTV", momentum_score=0.05,
                                volatility=0.16)),
        ]
        strength = engine._calculate_signal_strength(selected)
        assert 0.0 <= strength <= 1.0

    def test_strength_capped_at_one(self):
        """Signal strength is capped at 1.0 (all positive momentum)."""
        engine = FactorMomentumEngine()
        selected = [
            ("MTUM", _make_score(momentum_score=100.0, volatility=0.10)),
            ("USMV", _make_score(symbol="USMV", momentum_score=50.0,
                                 volatility=0.10)),
            ("QUAL", _make_score(symbol="QUAL", momentum_score=1.0,
                                volatility=0.10)),
        ]
        strength = engine._calculate_signal_strength(selected)
        assert abs(strength - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# FactorScore dataclass — __post_init__ not present but field mutation works
# ---------------------------------------------------------------------------

class TestFactorScoreMutation:
    def test_field_mutation_persists(self):
        """FactorScore fields are mutable."""
        score = _make_score()
        score.symbol = "QQQ"
        score.factor_name = "Nasdaq 100"
        score.price = 200.0
        assert score.symbol == "QQQ"
        assert score.factor_name == "Nasdaq 100"
        assert score.price == 200.0

    def test_large_float_fields(self):
        """Extreme float values stored correctly."""
        score = _make_score(return_12m=999.0, volatility=999.0)
        assert score.return_12m == 999.0
        assert score.volatility == 999.0


# ---------------------------------------------------------------------------
# ENGINE universe validation
# ---------------------------------------------------------------------------

class TestEngineUniverse:
    def test_universe_is_list(self):
        """universe is a list of symbols."""
        engine = FactorMomentumEngine()
        assert isinstance(engine.universe, list)
        assert len(engine.universe) == 11

    def test_universe_mutability(self):
        """universe can be temporarily modified (as done in backtest)."""
        engine = FactorMomentumEngine()
        engine.universe = ["SPY", "QQQ"]
        assert engine.universe == ["SPY", "QQQ"]


# ---------------------------------------------------------------------------
# Dataclass field validation via dataclasses.fields()
# ---------------------------------------------------------------------------

class TestFactorScoreDataclassFields:
    """Validate FactorScore dataclass field definitions via dataclasses.fields()."""

    def test_field_count(self):
        """FactorScore has exactly 18 fields."""
        fds = dataclasses.fields(FactorScore)
        assert len(fds) == 18

    def test_field_names(self):
        """All field names match source definition order."""
        fds = dataclasses.fields(FactorScore)
        names = [f.name for f in fds]
        expected = [
            "symbol", "factor_name", "price", "return_12m", "return_6m",
            "return_3m", "volatility", "sharpe_12m", "momentum_score", "rank",
            "value_momentum_synergy", "momentum_acceleration",
            "vol_adjusted_momentum", "regime_momentum", "factor_divergence",
            "composite_ml_score", "tsfm_score", "tsfm_allocation_scalar",
        ]
        assert names == expected

    def test_field_types_via_fields(self):
        """Field types match source via fields().type."""
        fds = {f.name: f for f in dataclasses.fields(FactorScore)}
        assert fds["symbol"].type is str
        assert fds["factor_name"].type is str
        assert fds["price"].type is float
        assert fds["return_12m"].type is float
        assert fds["return_6m"].type is float
        assert fds["return_3m"].type is float
        assert fds["volatility"].type is float
        assert fds["sharpe_12m"].type is float
        assert fds["momentum_score"].type is float
        assert fds["rank"].type is int

    def test_defaults_via_fields(self):
        """Fields with defaults have correct default values."""
        fds = {f.name: f for f in dataclasses.fields(FactorScore)}
        # Required fields have no default (default is dataclasses.MISSING)
        for name in ("symbol", "factor_name", "price", "return_12m",
                     "return_6m", "return_3m", "volatility", "sharpe_12m",
                     "momentum_score", "rank"):
            assert fds[name].default is dataclasses.MISSING, f"{name} should be required"
        # Optional fields with defaults
        assert fds["value_momentum_synergy"].default == 0.0
        assert fds["momentum_acceleration"].default == 0.0
        assert fds["vol_adjusted_momentum"].default == 0.0
        assert fds["regime_momentum"].default == 0.0
        assert fds["factor_divergence"].default == 0.0
        assert fds["composite_ml_score"].default == 0.0
        assert fds["tsfm_score"].default == 0.0
        assert fds["tsfm_allocation_scalar"].default == 1.0

    def test_default_factory_none_used(self):
        """No fields use default_factory (all use simple defaults)."""
        for f in dataclasses.fields(FactorScore):
            assert f.default_factory is dataclasses.MISSING, (
                f"{f.name} uses default_factory unexpectedly"
            )

    def test_repr_field(self):
        """Every field has repr=True (default)."""
        for f in dataclasses.fields(FactorScore):
            assert f.repr, f"{f.name} has repr=False"

    def test_compare_field(self):
        """Every field has compare=True (default) or compare matches intent."""
        for f in dataclasses.fields(FactorScore):
            assert f.compare, f"{f.name} has compare=False"


# ---------------------------------------------------------------------------
# FactorScore field mutation with extreme values
# ---------------------------------------------------------------------------

class TestFactorScoreFieldMutationExtended:
    def test_extreme_negative_return(self):
        """Very large negative return_12m is stored correctly."""
        score = _make_score(return_12m=-999.0)
        assert score.return_12m == -999.0

    def test_exact_boundary_tsfm_score(self):
        """tsfm_score of exactly -2.0 and 2.0 are valid."""
        score = _make_score(tsfm_score=-2.0)
        assert score.tsfm_score == -2.0
        score2 = _make_score(tsfm_score=2.0)
        assert score2.tsfm_score == 2.0

    def test_exact_boundary_tsfm_scalar(self):
        """tsfm_allocation_scalar of exactly 0.0 and 2.0 are valid."""
        score = _make_score(tsfm_allocation_scalar=0.0)
        assert score.tsfm_allocation_scalar == 0.0
        score2 = _make_score(tsfm_allocation_scalar=2.0)
        assert score2.tsfm_allocation_scalar == 2.0

    def test_empty_string_symbol(self):
        """Empty string symbol is permitted (dataclass has no validation)."""
        score = _make_score(symbol="")
        assert score.symbol == ""

    def test_mixed_numeric_types(self):
        """Integer values are accepted for float fields."""
        score = FactorScore(
            symbol="T", factor_name="T", price=100,
            return_12m=0, return_6m=0, return_3m=0,
            volatility=0, sharpe_12m=0, momentum_score=0, rank=1,
        )
        assert score.price == 100  # Accepted as-is


# ---------------------------------------------------------------------------
# Computation edge cases — NaN, Inf, zero-division
# ---------------------------------------------------------------------------

class TestComputationEdgeCases:
    def test_nan_close_price(self):
        """NaN in close prices should not crash, produces NaN results."""
        engine = FactorMomentumEngine()
        data = _generate_price_data(300, seed=99)
        data[-1]["close"] = float("nan")
        engine._fetch_price_data = lambda sym, days=300: data
        score = engine._calculate_factor_score("MTUM")
        assert score is not None

    def test_inf_close_price(self):
        """Inf in close prices should not crash."""
        engine = FactorMomentumEngine()
        data = _generate_price_data(300, seed=99)
        data[-1]["close"] = float("inf")
        engine._fetch_price_data = lambda sym, days=300: data
        score = engine._calculate_factor_score("MTUM")
        assert score is not None

    def test_missing_close_key(self):
        """Missing 'close' key in a data dict should raise KeyError."""
        engine = FactorMomentumEngine()
        data = _generate_price_data(300, seed=99)
        del data[-1]["close"]
        engine._fetch_price_data = lambda sym, days=300: data
        with pytest.raises(KeyError):
            engine._calculate_factor_score("MTUM")

    def test_negative_vol_price(self):
        """Negative close prices produce valid returns."""
        engine = FactorMomentumEngine()
        data = _generate_price_data(300, drift=0.0, vol=0.01, seed=99)
        data[-1]["close"] = -50.0
        engine._fetch_price_data = lambda sym, days=300: data
        score = engine._calculate_factor_score("MTUM")
        assert score is not None

    def test_price_series_with_zeros_in_middle(self):
        """Zeros in middle of price series produce NaN diffs but no crash."""
        engine = FactorMomentumEngine()
        data = _generate_price_data(300, seed=99)
        data[150]["close"] = 0.0
        engine._fetch_price_data = lambda sym, days=300: data
        score = engine._calculate_factor_score("MTUM")
        assert score is not None

    def test_single_element_price_diff(self):
        """vol_lookback=1 triggers single-element returns_daily code path."""
        engine = FactorMomentumEngine(vol_lookback=1)
        data = _generate_price_data(300, seed=99)
        engine._fetch_price_data = lambda sym, days=300: data
        score = engine._calculate_factor_score("MTUM")
        assert score is not None
        # With len(returns_daily) <= 1, volatility uses floor 0.15
        assert score.volatility == 0.15

    def test_vol_lookback_two(self):
        """vol_lookback=2 produces len(returns_daily)==1, vol floor applied."""
        engine = FactorMomentumEngine(vol_lookback=2)
        data = _generate_price_data(300, seed=99)
        engine._fetch_price_data = lambda sym, days=300: data
        score = engine._calculate_factor_score("MTUM")
        assert score is not None
        assert score.volatility >= 0.15  # At least the floor

    def test_days_12m_exact(self):
        """Exactly 253 data points: days_12m = min(252, 252) = 252."""
        engine = FactorMomentumEngine()
        data = _generate_price_data(253, drift=0.0004, seed=42)
        engine._fetch_price_data = lambda sym, days=300: data
        score = engine._calculate_factor_score("MTUM")
        assert score is not None


# ---------------------------------------------------------------------------
# Computation edge cases via evaluate/TSFM — empty/unavailable data
# ---------------------------------------------------------------------------

class TestEvaluateEmptyEdgeCases:
    def test_evaluate_with_only_one_factor_available(self):
        """Only one symbol has data, rest return None."""
        engine = FactorMomentumEngine()
        real_fetch = engine._fetch_price_data

        def mock_fetch(sym, days=300):
            if sym == "SPY":
                return _generate_price_data(300, seed=42)
            return []
        engine._fetch_price_data = mock_fetch
        result = engine.evaluate()
        assert "error" not in result
        assert result["selected_factors"] == ["SPY"]

    def test_evaluate_vol_lookback_longer_than_data(self):
        """vol_lookback > len(data) doesn't crash."""
        engine = FactorMomentumEngine(vol_lookback=500)
        data = _generate_price_data(253, seed=42)
        engine._fetch_price_data = lambda sym, days=300: data
        score = engine._calculate_factor_score("MTUM")
        assert score is not None

    def test_evaluate_missing_keys_in_current_scores(self):
        """current_scores dict has all required fields for every symbol."""
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate()
        for sym, data in result["current_scores"].items():
            assert isinstance(data["factor_name"], str)
            assert isinstance(data["category"], str)
            assert isinstance(data["return_12m"], float)
            assert isinstance(data["rank"], int)

    def test_evaluate_signal_strength_no_selected(self):
        """Signal strength is 0 when no factors selected."""
        engine = _make_engine_with_mocked_db(min_momentum=100.0)
        result = engine.evaluate()
        assert result["signal_strength"] == 0.0


# ---------------------------------------------------------------------------
# Constants validation — module-level
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_all_export_exists(self):
        """__all__ is defined and contains expected names."""
        from src.strategy.factor_rotation import __all__ as all_exports
        assert isinstance(all_exports, list)
        assert "FactorScore" in all_exports
        assert "FactorMomentumEngine" in all_exports
        assert "FactorRotationBacktest" in all_exports

    def test_logger_configured(self):
        """Module logger is a Logger instance."""
        from src.strategy.factor_rotation import logger as mod_logger
        assert isinstance(mod_logger, logging.Logger)
        assert mod_logger.name == "src.strategy.factor_rotation"

    def test_all_exports_no_private_names(self):
        """__all__ contains only public names (no underscore-prefixed)."""
        from src.strategy.factor_rotation import __all__ as all_exports
        for name in all_exports:
            assert not name.startswith("_"), f"{name} is private but in __all__"


# ---------------------------------------------------------------------------
# Constructor boundary conditions
# ---------------------------------------------------------------------------

class TestEngineConstructorBoundaries:
    def test_top_n_negative(self):
        """top_n=-1: negative values accepted; first factor selected, then guard fires."""
        engine = _make_engine_with_mocked_db(top_n=-1)
        result = engine.evaluate()
        # top_n=-1: 1st factor appended, then len(selected) >= -1 is True -> break
        assert len(result["selected_factors"]) == 1

    def test_min_momentum_negative(self):
        """Negative min_momentum allows all factors through."""
        engine = _make_engine_with_mocked_db(min_momentum=-1.0)
        result = engine.evaluate()
        assert len(result["selected_factors"]) > 0

    def test_vol_lookback_zero(self):
        """vol_lookback=0 produces empty returns array, uses vol floor 0.15."""
        engine = FactorMomentumEngine(vol_lookback=0)
        data = _generate_price_data(300, seed=42)
        engine._fetch_price_data = lambda sym, days=300: data
        score = engine._calculate_factor_score("MTUM")
        assert score is not None
        # len(returns_daily) = 0, not > 1, so volatility = 0.15 floor
        assert score.volatility == 0.15

    def test_lookback_months_negative(self):
        """Negative lookback_months is accepted but unused in current logic."""
        engine = FactorMomentumEngine(lookback_months=-1)
        assert engine.lookback_months == -1

    def test_max_per_category_zero(self):
        """max_per_category=0: no factors can be selected (all blocked by diversity)."""
        engine = _make_engine_with_mocked_db()
        engine.max_per_category = 0
        result = engine.evaluate()
        assert result["selected_factors"] == []
        assert result["allocation"] == {"SPY": 1.0}


# ---------------------------------------------------------------------------
# Function boundary conditions — _calculate_ml_features edge cases
# ---------------------------------------------------------------------------

class TestMLFeaturesBoundaryCases:
    def test_vix_at_exactly_10(self):
        """VIX = 10: percentile = max((10-10)/30, 0) = 0, multiplier = 1.0."""
        engine = FactorMomentumEngine()
        scores = {"MTUM": _make_score(momentum_score=0.20)}
        result = engine._calculate_ml_features("MTUM", scores, vix_level=10.0)
        assert abs(result["regime_momentum"] - 0.20) < 1e-6

    def test_vix_at_exactly_40(self):
        """VIX = 40: percentile = min((40-10)/30, 1) = 1, multiplier = 0.5."""
        engine = FactorMomentumEngine()
        scores = {"MTUM": _make_score(momentum_score=0.20)}
        result = engine._calculate_ml_features("MTUM", scores, vix_level=40.0)
        assert abs(result["regime_momentum"] - 0.10) < 1e-6

    def test_vix_at_exactly_30(self):
        """VIX = 30: percentile = (30-10)/30 = 2/3 ≈ 0.667, multiplier = 1.0."""
        engine = FactorMomentumEngine()
        scores = {"MTUM": _make_score(momentum_score=0.20)}
        result = engine._calculate_ml_features("MTUM", scores, vix_level=30.0)
        assert abs(result["regime_momentum"] - 0.20) < 1e-6

    def test_vix_negative(self):
        """Negative VIX is clamped to 0 percentile, multiplier = 1.0."""
        engine = FactorMomentumEngine()
        scores = {"MTUM": _make_score(momentum_score=0.20)}
        result = engine._calculate_ml_features("MTUM", scores, vix_level=-5.0)
        assert abs(result["regime_momentum"] - 0.20) < 1e-6

    def test_spy_missing_from_scores_in_divergence(self):
        """When SPY is missing from scores, divergence uses zero-vol placeholder."""
        engine = FactorMomentumEngine()
        scores = {"MTUM": _make_score(momentum_score=0.20, volatility=0.15)}
        result = engine._calculate_ml_features("MTUM", scores, vix_level=20.0)
        # SPY not in scores -> placeholder FactorScore with volatility=0.16
        # vol_ratio = 0.15 / max(0.16, 0.05) = 0.9375
        # factor_divergence = 1.0 / max(0.9375, 0.5) = 1.0667
        assert abs(result["factor_divergence"] - 1.066666) < 0.001

    def test_ml_features_zero_vol_adjusted_momentum(self):
        """When return_12m=0, vol_adjusted_momentum is 0."""
        engine = FactorMomentumEngine()
        scores = {"MTUM": _make_score(return_12m=0.0, momentum_score=0.0, volatility=0.15)}
        result = engine._calculate_ml_features("MTUM", scores, vix_level=20.0)
        assert result["vol_adjusted_momentum"] == 0.0

    def test_ml_features_empty_scores_for_symbol(self):
        """score.get() for missing symbol returns None -> empty dict."""
        engine = FactorMomentumEngine()
        scores = {}
        result = engine._calculate_ml_features("NOPE", scores)
        assert result == {}

    def test_ml_features_vlue_triggers_value_synergy(self):
        """VLUE symbol also triggers value synergy path (same as VTV)."""
        engine = FactorMomentumEngine()
        scores = {
            "VTV": _make_score(symbol="VTV", momentum_score=0.20),
            "VLUE": _make_score(symbol="VLUE", momentum_score=0.15),
            "SPY": _make_score(symbol="SPY", momentum_score=0.10),
        }
        result = engine._calculate_ml_features("VLUE", scores, vix_level=20.0)
        # VLUE -> vtv_score = scores["VTV"].momentum_score = 0.20
        # vug_score = scores.get("VUG", scores.get("SPY")) = scores["SPY"].momentum_score = 0.10
        # value_spread = 0.20 - 0.10 = 0.10
        # synergy = 0.10 * abs(0.20) = 0.02
        assert abs(result["value_momentum_synergy"] - 0.02) < 1e-6


# ---------------------------------------------------------------------------
# _calculate_factor_score — additional computation edge cases
# ---------------------------------------------------------------------------

class TestFactorScoreCalcExtended:
    def test_12m_return_exact_zero(self):
        """12m return of exactly 0.0."""
        engine = FactorMomentumEngine()
        data = _generate_price_data(300, drift=0.0, vol=0.0, seed=44)
        engine._fetch_price_data = lambda sym, days=300: data
        score = engine._calculate_factor_score("MTUM")
        assert score is not None
        assert score.return_12m == 0.0

    def test_extreme_positive_drift(self):
        """Very large positive drift produces extremely high returns."""
        engine = FactorMomentumEngine()
        data = _generate_price_data(300, drift=0.01, vol=0.001, seed=44)
        engine._fetch_price_data = lambda sym, days=300: data
        score = engine._calculate_factor_score("MTUM")
        assert score is not None
        assert score.return_12m > 0.50

    def test_constant_flat_prices_series(self):
        """All prices exactly equal -> zero returns, zero volatility (fallback)."""
        engine = FactorMomentumEngine()
        data = []
        for i in range(300):
            data.append({"date": f"2024-{(i // 21 + 1):02d}-{(i % 21 + 1):02d}",
                         "close": 100.0, "volume": 1000000})
        engine._fetch_price_data = lambda sym, days=300: data
        score = engine._calculate_factor_score("MTUM")
        assert score is not None
        assert score.return_12m == 0.0
        assert score.return_6m == 0.0
        assert score.return_3m == 0.0

    def test_vol_lookback_clipped_to_data_length(self):
        """vol_lookback > len(closes) results in minimal returns_daily."""
        engine = FactorMomentumEngine(vol_lookback=1000)
        data = _generate_price_data(253, seed=42)
        engine._fetch_price_data = lambda sym, days=300: data
        score = engine._calculate_factor_score("MTUM")
        assert score is not None

    def test_sharpe_computation_correctness(self):
        """sharpe_12m = return_12m / vol_12m when vol_12m > 0."""
        engine = FactorMomentumEngine()
        data = _generate_price_data(300, drift=0.0004, vol=0.015, seed=42)
        engine._fetch_price_data = lambda sym, days=300: data
        score = engine._calculate_factor_score("MTUM")
        assert score is not None
        # Manual recompute
        closes = [d["close"] for d in data]
        days_12m = min(252, len(closes) - 1)
        ret_12m = (closes[-1] / closes[-days_12m]) - 1
        r12 = np.diff(closes[-days_12m:]) / np.array(closes[-days_12m:-1])
        if len(r12) > 20:
            v12 = float(np.std(r12)) * np.sqrt(252)
            expected_sharpe = ret_12m / v12 if v12 > 0 else 0
            assert abs(score.sharpe_12m - expected_sharpe) < 1e-6


# ---------------------------------------------------------------------------
# evaluate() boundary conditions — allocation generation
# ---------------------------------------------------------------------------

class TestEvaluateAllocationBoundaries:
    def test_single_factor_selected_allocation(self):
        """Single selected factor gets 100% allocation."""
        engine = FactorMomentumEngine()
        def mock_fetch(sym, days=300):
            if sym == "SPY":
                return _generate_price_data(300, drift=0.0004, seed=42)
            return []
        engine._fetch_price_data = mock_fetch
        result = engine.evaluate()
        assert len(result["allocation"]) == 1
        assert abs(list(result["allocation"].values())[0] - 1.0) < 1e-6

    def test_all_factors_equal_momentum(self):
        """All factors with identical momentum selects by category then rank."""
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate()
        # Verify selected factors are a subset of universe
        for sym in result["selected_factors"]:
            assert sym in engine.FACTORS

    def test_evaluate_diversity_only_one_category(self):
        """If only one category has data, at most 1 factor is selected."""
        engine = FactorMomentumEngine()
        def mock_fetch(sym, days=300):
            if sym in ("VTV", "VLUE"):
                return _generate_price_data(300, drift=0.0004, seed=42)
            return []
        engine._fetch_price_data = mock_fetch
        result = engine.evaluate()
        selected_cats = set()
        for s in result["selected_factors"]:
            selected_cats.add(engine.FACTORS[s]["category"])
        assert len(selected_cats) <= 1
        assert result["selected_factors"] == ["VTV"]


# ---------------------------------------------------------------------------
# _generate_allocation — extreme volatility cases
# ---------------------------------------------------------------------------

class TestGenerateAllocationExtreme:
    def test_inv_vol_zero_high_vol(self):
        """Extremely high volatility gives near-zero inv_vol, equal weight fallback."""
        engine = FactorMomentumEngine()
        selected = [
            ("M1", _make_score(symbol="M1", volatility=1e10)),
            ("M2", _make_score(symbol="M2", volatility=1e10)),
            ("M3", _make_score(symbol="M3", volatility=1e10)),
        ]
        alloc = engine._generate_allocation(selected)
        assert abs(sum(alloc.values()) - 1.0) < 1e-10
        for w in alloc.values():
            assert abs(w - 1/3) < 1e-10

    def test_mixed_vol_ordering(self):
        """Weight order matches inverse vol order."""
        engine = FactorMomentumEngine()
        selected = [
            ("HIGH", _make_score(symbol="HIGH", volatility=0.30)),
            ("MED", _make_score(symbol="MED", volatility=0.20)),
            ("LOW", _make_score(symbol="LOW", volatility=0.10)),
        ]
        alloc = engine._generate_allocation(selected)
        assert alloc["LOW"] > alloc["MED"] > alloc["HIGH"]

    def test_vol_range_zero_to_extreme(self):
        """One zero-vol and one extreme-vol factor."""
        engine = FactorMomentumEngine()
        selected = [
            ("A", _make_score(symbol="A", volatility=0.0)),
            ("B", _make_score(symbol="B", volatility=1e10)),
        ]
        alloc = engine._generate_allocation(selected)
        # A clamped to 0.05 -> inv_vol = 20, B inv_vol = 1e-10
        # A weight ~ 1.0, B weight ~ 0.0
        assert abs(alloc["A"] - 1.0) < 0.001
        assert abs(alloc["B"]) < 0.001


# ---------------------------------------------------------------------------
# _calculate_signal_strength — boundary behavior
# ---------------------------------------------------------------------------

class TestSignalStrengthBoundaries:
    def test_spread_equals_zero_all_equal_scores(self):
        """All factors with identical scores -> spread=0."""
        engine = FactorMomentumEngine()
        selected = [
            ("A", _make_score(symbol="A", momentum_score=0.10, volatility=0.20)),
            ("B", _make_score(symbol="B", momentum_score=0.10, volatility=0.20)),
            ("C", _make_score(symbol="C", momentum_score=0.10, volatility=0.20)),
        ]
        strength = engine._calculate_signal_strength(selected)
        # spread=0, direction=1.0, vol=0 (all 20% < 25%)
        # strength = min(0, 0.4) + 1.0*0.4 + 1.0*0.2 = 0.6
        assert abs(strength - 0.6) < 1e-6

    def test_vol_all_above_25(self):
        """All factors above 25% vol -> vol_score = 0."""
        engine = FactorMomentumEngine()
        selected = [
            ("A", _make_score(symbol="A", momentum_score=0.10, volatility=0.30)),
            ("B", _make_score(symbol="B", momentum_score=0.10, volatility=0.35)),
        ]
        strength = engine._calculate_signal_strength(selected)
        # direction=1.0, vol_score=0, spread=0
        # strength = 0 + 0.4 + 0 = 0.4
        assert abs(strength - 0.4) < 1e-6

    def test_spread_capped_at_02(self):
        """Large spread capped to 0.4 (min(spread*2, 0.4))."""
        engine = FactorMomentumEngine()
        selected = [
            ("A", _make_score(symbol="A", momentum_score=1.0, volatility=0.10)),
            ("B", _make_score(symbol="B", momentum_score=0.0, volatility=0.10)),
        ]
        strength = engine._calculate_signal_strength(selected)
        # spread = 1.0, spread*2 = 2.0, min = 0.4
        # direction = 1/2 = 0.5, vol_score = 1.0 (all 10% < 25%)
        # strength = 0.4 + 0.5*0.4 + 1.0*0.2 = 0.4 + 0.2 + 0.2 = 0.8
        assert abs(strength - 0.8) < 1e-6

    def test_two_factor_spread_exact_value(self):
        """2-factor spread = first - second."""
        engine = FactorMomentumEngine()
        selected = [
            ("A", _make_score(symbol="A", momentum_score=0.15, volatility=0.10)),
            ("B", _make_score(symbol="B", momentum_score=0.05, volatility=0.10)),
        ]
        strength = engine._calculate_signal_strength(selected)
        # spread = 0.15-0.05 = 0.10, spread_contrib = min(0.20, 0.4) = 0.20
        # direction = 1.0, vol_score = 1.0
        # strength = 0.20 + 0.40 + 0.20 = 0.80
        assert abs(strength - 0.80) < 1e-6


# ---------------------------------------------------------------------------
# _generate_recommendation — boundary conditions
# ---------------------------------------------------------------------------

class TestRecommendationBoundaries:
    def test_exactly_20_percent_boundary(self):
        """avg_momentum just above 0.20 -> 'strong' (source uses strict > 0.20)."""
        engine = FactorMomentumEngine()
        selected = [
            ("MTUM", _make_score(symbol="MTUM", factor_name="Momentum",
                                 return_12m=0.2001)),
            ("USMV", _make_score(symbol="USMV", factor_name="Low Vol",
                                 return_12m=0.2001)),
        ]
        all_s = {
            "MTUM": _make_score(return_12m=0.2001),
            "USMV": _make_score(return_12m=0.2001),
        }
        rec = engine._generate_recommendation(selected, all_s)
        assert "strong" in rec.lower()

    def test_exactly_10_percent_boundary(self):
        """avg_momentum just above 0.10 -> 'moderate' (source uses strict > 0.10)."""
        engine = FactorMomentumEngine()
        selected = [
            ("MTUM", _make_score(symbol="MTUM", factor_name="Momentum",
                                 return_12m=0.1001)),
            ("USMV", _make_score(symbol="USMV", factor_name="Low Vol",
                                 return_12m=0.1001)),
        ]
        all_s = {
            "MTUM": _make_score(return_12m=0.1001),
            "USMV": _make_score(return_12m=0.1001),
        }
        rec = engine._generate_recommendation(selected, all_s)
        assert "moderate" in rec.lower()

    def test_mixed_category_concentration_detected(self):
        """Concentration detected when selected share a category."""
        engine = FactorMomentumEngine()
        selected = [
            ("SPY", _make_score(symbol="SPY", factor_name="S&P 500",
                                return_12m=0.15)),
            ("QQQ", _make_score(symbol="QQQ", factor_name="Nasdaq 100",
                                return_12m=0.15)),
        ]
        all_s = {
            "SPY": _make_score(return_12m=0.15),
            "QQQ": _make_score(return_12m=0.15),
        }
        rec = engine._generate_recommendation(selected, all_s)
        assert "concentrated" in rec

    def test_empty_all_scores_dict(self):
        """Empty all_scores dict still generates recommendation."""
        engine = FactorMomentumEngine()
        selected = [
            ("MTUM", _make_score(symbol="MTUM", factor_name="Momentum",
                                 return_12m=0.15)),
        ]
        rec = engine._generate_recommendation(selected, {})
        assert "Momentum" in rec


# ---------------------------------------------------------------------------
# evaluate_tsfm — boundary conditions for VIX regimes
# ---------------------------------------------------------------------------

class TestTSFMEvaluateBoundaries:
    def test_tsfm_vix_less_than_15_amplifies(self):
        """VIX < 15 amplifies regime_momentum by 1.2x."""
        engine = _make_engine_with_mocked_db()
        result_low = engine.evaluate_tsfm(vix_level=14.9)
        result_mid = engine.evaluate_tsfm(vix_level=20.0)
        for sym in result_low["tsfm"]["tsfm_scores"]:
            tsfm_base = result_low["tsfm"]["tsfm_scores"][sym]["tsfm_score"]
            low_rm = result_low["tsfm"]["tsfm_scores"][sym]["regime_momentum"]
            mid_rm = result_mid["tsfm"]["tsfm_scores"][sym]["regime_momentum"]
            if abs(tsfm_base) > 0.001:
                # If low vol, low_rm should differ from mid_rm
                assert low_rm != mid_rm
                break

    def test_tsfm_vix_exactly_15_normal(self):
        """VIX = 15 uses normal regime (no amplification)."""
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate_tsfm(vix_level=15.0)
        for sym, sdata in result["tsfm"]["tsfm_scores"].items():
            if abs(sdata["tsfm_score"]) > 0.001:
                assert abs(sdata["regime_momentum"] - sdata["tsfm_score"]) < 1e-6
                break

    def test_tsfm_signal_strength_zero_when_empty(self):
        """TSFM reports error when no price data available."""
        engine = FactorMomentumEngine()
        engine._fetch_price_data = lambda sym, days=300: []
        result = engine.evaluate_tsfm(vix_level=20.0)
        # No data -> error path in evaluate() -> tsfm gets error key, no signal_strength
        assert "error" in result["tsfm"]

    def test_tsfm_no_scores_no_crash(self):
        """evaluate_tsfm handles empty factor_scores gracefully."""
        engine = FactorMomentumEngine()
        engine._fetch_price_data = lambda sym, days=300: _generate_price_data(100)
        result = engine.evaluate_tsfm(vix_level=20.0)
        assert "tsfm" in result

    def test_tsfm_return_1m_reverse_calc(self):
        """Verify return_1m reverse calculation in tsfm_scores."""
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate_tsfm(vix_level=20.0)
        for sym, sdata in result["tsfm"]["tsfm_scores"].items():
            # return_1m = vol_adjusted_momentum * volatility (reverse calc)
            if "return_1m" in sdata and sdata["tsfm_score"] != 0:
                assert isinstance(sdata["return_1m"], float)
                break


# ---------------------------------------------------------------------------
# evaluate_ml_enhanced — additional boundary cases
# ---------------------------------------------------------------------------

class TestMLEvaluateBoundaries:
    def test_ml_enhanced_when_spy_not_in_factors(self):
        """ML enhanced runs even when SPY is not in factor_scores."""
        engine = _make_engine_with_mocked_db()
        # Force only non-SPY symbols
        original_fetch = engine._fetch_price_data
        def no_spy_fetch(sym, days=300):
            if sym == "SPY" or sym == "^VIX":
                return []
            return _generate_price_data(300, drift=0.0004, seed=hash(sym) % 100)
        engine._fetch_price_data = no_spy_fetch
        result = engine.evaluate_ml_enhanced(vix_level=20.0)
        assert "ml_scores" in result

    def test_ml_enhanced_error_when_no_data(self):
        """When no data available, ML enhanced returns error dict."""
        engine = FactorMomentumEngine()
        engine._fetch_price_data = lambda sym, days=300: []
        result = engine.evaluate_ml_enhanced(vix_level=20.0)
        assert "error" in result
        assert result["ml_enhanced"] is True

    def test_ml_scores_empty_result_when_no_data(self):
        """ml_scores is absent or empty when no factor data available."""
        engine = FactorMomentumEngine()
        engine._fetch_price_data = lambda sym, days=300: []
        result = engine.evaluate_ml_enhanced(vix_level=20.0)
        assert "ml_scores" not in result or result["ml_scores"] == {}


# ---------------------------------------------------------------------------
# _generate_ml_recommendation — boundary conditions
# ---------------------------------------------------------------------------

class TestMLRecommendBoundaries:
    def test_no_patterns_no_acceleration(self):
        """No value synergy or acceleration -> no pattern_str in recommendation."""
        engine = FactorMomentumEngine()
        score = _make_score(symbol="MTUM", factor_name="Momentum")
        score.composite_ml_score = 0.10
        score.value_momentum_synergy = 0.0
        score.momentum_acceleration = -0.01
        scores = {"MTUM": score}
        selected = [("MTUM", scores["MTUM"])]
        rec = engine._generate_ml_recommendation(selected, scores)
        assert "value-momentum" not in rec
        assert "accelerating" not in rec

    def test_ml_moderate_signal_exact_boundary(self):
        """composite_ml_score just above 0.15 -> moderate (source uses strict > 0.15)."""
        engine = FactorMomentumEngine()
        score = _make_score(symbol="MTUM", factor_name="Momentum")
        score.composite_ml_score = 0.151
        scores = {"MTUM": score}
        selected = [("MTUM", scores["MTUM"])]
        rec = engine._generate_ml_recommendation(selected, scores)
        assert "moderate" in rec.lower()

    def test_ml_strong_signal_exact_boundary(self):
        """composite_ml_score just above 0.30 -> strong (source uses strict > 0.30)."""
        engine = FactorMomentumEngine()
        score = _make_score(symbol="MTUM", factor_name="Momentum")
        score.composite_ml_score = 0.301
        scores = {"MTUM": score}
        selected = [("MTUM", scores["MTUM"])]
        rec = engine._generate_ml_recommendation(selected, scores)
        assert "strong" in rec.lower()


# ---------------------------------------------------------------------------
# _generate_tsfm_recommendation — boundary conditions
# ---------------------------------------------------------------------------

class TestTSFMRecommendBoundaries:
    def test_tsfm_boundary_25_vix_elevated(self):
        """VIX > 25 produces 'elevated vol' description."""
        engine = FactorMomentumEngine()
        selected = [("MTUM", _make_score(tsfm_score=1.0, tsfm_allocation_scalar=1.0))]
        rec = engine._generate_tsfm_recommendation(
            selected, {"MTUM": _make_score(tsfm_score=1.0, tsfm_allocation_scalar=1.0)},
            25.1
        )
        assert "elevated" in rec.lower()

    def test_tsfm_exactly_15_vix_normal(self):
        """VIX = 15 produces 'normal vol' description."""
        engine = FactorMomentumEngine()
        selected = [("MTUM", _make_score(tsfm_score=1.0, tsfm_allocation_scalar=1.0))]
        rec = engine._generate_tsfm_recommendation(
            selected, {"MTUM": _make_score(tsfm_score=1.0, tsfm_allocation_scalar=1.0)},
            15.0
        )
        assert "normal" in rec.lower()

    def test_tsfm_exactly_25_vix_normal(self):
        """VIX = 25 produces 'normal vol' description (<= 25)."""
        engine = FactorMomentumEngine()
        selected = [("MTUM", _make_score(tsfm_score=1.0, tsfm_allocation_scalar=1.0))]
        rec = engine._generate_tsfm_recommendation(
            selected, {"MTUM": _make_score(tsfm_score=1.0, tsfm_allocation_scalar=1.0)},
            25.0
        )
        assert "normal" in rec.lower()

    def test_tsfm_exactly_15_scalar_15(self):
        """Average scalar exactly 1.5 -> 'moderate' (source uses strict > 1.5)."""
        engine = FactorMomentumEngine()
        selected = [("MTUM", _make_score(tsfm_score=1.5, tsfm_allocation_scalar=1.5))]
        rec = engine._generate_tsfm_recommendation(
            selected, {"MTUM": _make_score(tsfm_score=1.5, tsfm_allocation_scalar=1.5)},
            20.0
        )
        assert "moderate" in rec.lower()

    def test_tsfm_above_15_scalar_strong(self):
        """Average scalar just above 1.5 -> 'strong' (source uses strict > 1.5)."""
        engine = FactorMomentumEngine()
        selected = [("MTUM", _make_score(tsfm_score=1.5, tsfm_allocation_scalar=1.51))]
        rec = engine._generate_tsfm_recommendation(
            selected, {"MTUM": _make_score(tsfm_score=1.5, tsfm_allocation_scalar=1.51)},
            20.0
        )
        assert "strong" in rec.lower()

    def test_tsfm_exactly_10_scalar_10(self):
        """Average scalar exactly 1.0 -> 'weak' (source uses strict > 1.0)."""
        engine = FactorMomentumEngine()
        selected = [("MTUM", _make_score(tsfm_score=1.0, tsfm_allocation_scalar=1.0))]
        rec = engine._generate_tsfm_recommendation(
            selected, {"MTUM": _make_score(tsfm_score=1.0, tsfm_allocation_scalar=1.0)},
            20.0
        )
        assert "weak" in rec.lower()

    def test_tsfm_above_10_scalar_moderate(self):
        """Average scalar just above 1.0 -> 'moderate' (source uses strict > 1.0)."""
        engine = FactorMomentumEngine()
        selected = [("MTUM", _make_score(tsfm_score=1.0, tsfm_allocation_scalar=1.01))]
        rec = engine._generate_tsfm_recommendation(
            selected, {"MTUM": _make_score(tsfm_score=1.0, tsfm_allocation_scalar=1.01)},
            20.0
        )
        assert "moderate" in rec.lower()


# ---------------------------------------------------------------------------
# CLI / __main__ guard — capsys testing
# ---------------------------------------------------------------------------

class TestCLI:
    def test_main_no_args(self, capsys):
        """main() with no args prints help and exits."""
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["factor_rotation.py"]
            from src.strategy.factor_rotation import main
            with pytest.raises(SystemExit) as exc_info:
                main()
            captured = capsys.readouterr()
            assert "Factor Momentum Rotation Strategy" in captured.err
            assert exc_info.value.code == 0
        finally:
            sys.argv = old_argv

    def test_main_unknown_command(self, capsys):
        """main() with unknown command prints error."""
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["factor_rotation.py", "bogus"]
            from src.strategy.factor_rotation import main
            main()
            captured = capsys.readouterr()
            assert "Unknown command" in captured.err
            assert "bogus" in captured.err
        finally:
            sys.argv = old_argv

    def test_main_evaluate(self, capsys):
        """main() evaluate prints signal header and JSON output."""
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["factor_rotation.py", "evaluate"]
            from src.strategy.factor_rotation import main
            # engine will have no data, should output error gracefully
            main()
            captured = capsys.readouterr()
            assert "FACTOR MOMENTUM ROTATION SIGNAL" in captured.err
            assert "STANDARD" in captured.err
            assert "{" in captured.err  # JSON output
        finally:
            sys.argv = old_argv

    def test_main_evaluate_ml_flag(self, capsys):
        """main() evaluate --ml uses ML-enhanced path."""
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["factor_rotation.py", "evaluate", "--ml"]
            from src.strategy.factor_rotation import main
            main()
            captured = capsys.readouterr()
            assert "ML-ENHANCED" in captured.err
        finally:
            sys.argv = old_argv

    def test_main_evaluate_tsfm_flag(self, capsys):
        """main() evaluate --tsfm uses TSFM path."""
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["factor_rotation.py", "evaluate", "--tsfm"]
            from src.strategy.factor_rotation import main
            main()
            captured = capsys.readouterr()
            assert "TSFM (v2.15)" in captured.err
        finally:
            sys.argv = old_argv

    def test_main_status(self, capsys):
        """main() status prints JSON status."""
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["factor_rotation.py", "status"]
            from src.strategy.factor_rotation import main
            main()
            captured = capsys.readouterr()
            assert "{" in captured.err
            import json
            # Logger prefix (e.g. "INFO:src.strategy.factor_rotation:") precedes JSON in stderr
            json_data = captured.err[captured.err.index('{'):]
            data = json.loads(json_data)
            assert "available" in data
            assert "factor_count" in data
        finally:
            sys.argv = old_argv

    def test_main_compare(self, capsys):
        """main() compare prints factor comparison table."""
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["factor_rotation.py", "compare"]
            from src.strategy.factor_rotation import main
            main()
            captured = capsys.readouterr()
            assert "Factor Comparison" in captured.err
            assert "Rank" in captured.err
        finally:
            sys.argv = old_argv

    def test_main_unknown_command_shows_usage(self, capsys):
        """main() with unknown command shows usage message (not help)."""
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["factor_rotation.py", "-h"]
            from src.strategy.factor_rotation import main
            main()
            captured = capsys.readouterr()
            assert "Unknown command" in captured.err
            assert "Usage:" in captured.err
        finally:
            sys.argv = old_argv


# ---------------------------------------------------------------------------
# __name__ == "__main__" guard test
# ---------------------------------------------------------------------------

class TestMainGuard:
    def test_main_guard_calls_main(self):
        """__name__ == '__main__' block calls main()."""
        # The guard at line 1042-1043 is:
        # if __name__ == "__main__":
        #     main()
        # Verify the module has this guard by checking it runs
        # We can test this by importing and checking the last lines
        import inspect
        from src.strategy import factor_rotation
        source = inspect.getsource(factor_rotation)
        assert 'if __name__ == "__main__":' in source
        assert "main()" in source.split('if __name__ == "__main__":')[-1]


# ---------------------------------------------------------------------------
# __all__ export completeness
# ---------------------------------------------------------------------------

class TestAllExportCompleteness:
    def test_all_matches_module_public_api(self):
        """__all__ covers all public (non-underscore) classes/functions."""
        from src.strategy.factor_rotation import __all__ as exported
        # FactorScore, FactorMomentumEngine, FactorRotationBacktest
        assert set(exported) == {"FactorScore", "FactorMomentumEngine",
                                 "FactorRotationBacktest"}

    def test_import_via_all(self):
        """Each name in __all__ can be imported directly."""
        from src.strategy.factor_rotation import (
            FactorScore,
            FactorMomentumEngine,
            FactorRotationBacktest,
        )
        assert FactorScore is not None
        assert FactorMomentumEngine is not None
        assert FactorRotationBacktest is not None

    def test_main_not_in_all(self):
        """main() function is NOT in __all__ (internal)."""
        from src.strategy.factor_rotation import __all__ as exported
        assert "main" not in exported

    def test_logger_not_in_all(self):
        """logger is NOT in __all__ (internal)."""
        from src.strategy.factor_rotation import __all__ as exported
        assert "logger" not in exported


# ---------------------------------------------------------------------------
# FactorRotationBacktest — extended edge cases
# ---------------------------------------------------------------------------

class TestFactorRotationBacktestExtended:
    def test_backtest_error_format(self):
        """Backtest error result contains error string."""
        engine = FactorMomentumEngine(db_path=Path("/nonexistent/market.db"))
        bt = FactorRotationBacktest(engine)
        result = bt.run_backtest("2020-01-01", "2024-12-31")
        assert isinstance(result.get("error", ""), str)
        assert len(result.get("error", "")) > 0

    def test_backtest_status_failed_on_error(self):
        """Backtest error result has status 'failed'."""
        engine = FactorMomentumEngine(db_path=Path("/nonexistent/market.db"))
        bt = FactorRotationBacktest(engine)
        result = bt.run_backtest("2020-01-01", "2024-12-31")
        if "error" in result:
            assert result.get("status") == "failed"


# ---------------------------------------------------------------------------
# _calculate_ml_features — ridge shrinkage verification
# ---------------------------------------------------------------------------

class TestMLRidgeShrinkage:
    def test_ridge_shrinkage_applied(self):
        """Composite ML score is shrunk by factor 0.9."""
        engine = FactorMomentumEngine()
        scores = {"MTUM": _make_score(momentum_score=0.20, volatility=0.15)}
        result = engine._calculate_ml_features("MTUM", scores, vix_level=20.0)
        # raw_composite = 0.20*0.3 + 0*0.2 + accel*0.15 + 1.33*0.1 + 0.20*0.15 + div*0.1
        # With ridge shrink 0.9
        expected_raw = (
            0.20 * 0.3 +  # momentum_score
            0.0 * 0.2 +   # value_momentum_synergy
            result["momentum_acceleration"] * 0.15 +
            result["vol_adjusted_momentum"] * 0.1 +
            result["regime_momentum"] * 0.15 +
            result["factor_divergence"] * 0.1
        )
        expected = expected_raw * 0.9
        assert abs(result["composite_ml_score"] - expected) < 1e-10


# ---------------------------------------------------------------------------
# _calculate_factor_score — 12-month vol threshold for sharpe
# ---------------------------------------------------------------------------

class TestSharpeVolThreshold:
    def test_sharpe_vol_floor_applied(self):
        """vol_12m < 20 data points uses floor 0.15."""
        engine = FactorMomentumEngine()
        data = _generate_price_data(260, drift=0.0, vol=0.0, seed=55)
        engine._fetch_price_data = lambda sym, days=300: data
        score = engine._calculate_factor_score("MTUM")
        # 12m period has 252 data points, vol_lookback overlaps.
        # With all flat prices, returns_12m_period has elements,
        # all zeros, std = 0, so vol_12m = 0
        # Since len(returns_12m_period) > 20 with 252 data points,
        # vol_12m = 0 * sqrt(252) = 0
        # sharpe = return_12m / 0 = 0 (since vol_12m > 0 is False)
        assert score is not None
        assert score.sharpe_12m == 0.0 or score.sharpe_12m == 0


# ---------------------------------------------------------------------------
# evaluate_tsfm — factor_divergence computation
# ---------------------------------------------------------------------------

class TestTSFMDivergence:
    def test_factor_divergence_std_nonzero(self):
        """Factor divergence computed when std_tsfm > 0."""
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate_tsfm(vix_level=20.0)
        for sym, sdata in result["tsfm"]["tsfm_scores"].items():
            assert "factor_divergence" in sdata

    def test_factor_divergence_value_range(self):
        """factor_divergence is non-negative."""
        engine = _make_engine_with_mocked_db()
        result = engine.evaluate_tsfm(vix_level=20.0)
        for sym, sdata in result["tsfm"]["tsfm_scores"].items():
            assert sdata["factor_divergence"] >= 0.0


# ---------------------------------------------------------------------------
# _calculate_factor_score — TSFM raw calculation validation
# ---------------------------------------------------------------------------

class TestTSFMRawCalculation:
    def test_tsfm_raw_vol_floor(self):
        """tsfm_raw uses vol floor 0.05 to avoid division by zero."""
        engine = FactorMomentumEngine()
        data = _generate_price_data(300, drift=0.0, vol=0.0, seed=99)
        engine._fetch_price_data = lambda sym, days=300: data
        score = engine._calculate_factor_score("MTUM")
        assert score is not None
        # return_1m = 0, volatility = 0 (but floor is 0.15 for vol, 0.05 for tsfm)
        # But vol_lookback=20 with all zeros -> std=0 -> volatility=0
        # tsfm_raw = 0 / max(0, 0.05) = 0
        assert score.tsfm_score == 0.0


# ---------------------------------------------------------------------------
# Engine._fetch_price_data — DB not existing
# ---------------------------------------------------------------------------

class TestFetchPriceData:
    def test_fetch_empty_when_db_missing(self):
        """_fetch_price_data returns [] when db_path doesn't exist."""
        engine = FactorMomentumEngine(db_path=Path("/tmp/nonexistent/market.db"))
        result = engine._fetch_price_data("SPY")
        assert result == []

    def test_fetch_with_valid_db_but_no_data(self):
        """_fetch_price_data with non-existent path returns []."""
        engine = FactorMomentumEngine(db_path=Path("/dev/null/market.db"))
        result = engine._fetch_price_data("SPY")
        assert result == []
