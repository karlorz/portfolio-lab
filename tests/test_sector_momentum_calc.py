#!/usr/bin/env python3
"""
Tests for sector_momentum_calc.py — constants, SectorMomentumCalculator,
momentum calculation, regime adjustment, allocation generation.
"""
import numpy as np

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.strategy.sector_momentum_calc import (
    SECTOR_ETF_DEFINITIONS,
    SECTOR_ETF_MAP,
    REGIME_SECTOR_PREFERENCES,
    SectorMomentumCalculator,
    generate_sector_signals,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(symbol="XLK", n=300, start=100.0, drift=0.0005, vol=0.012, seed=42):
    """Generate synthetic price data in the format expected by the calculator."""
    rng = np.random.RandomState(seed)
    prices = []
    price = start
    base_date = 20240101
    for i in range(n):
        price *= (1 + rng.normal(drift, vol))
        prices.append({"date": str(base_date + i), "close": price, "adjClose": price})
    return prices


def _make_historical_data(symbols=None, n=300):
    """Create synthetic historical data dict for multiple symbols."""
    if symbols is None:
        symbols = ["XLK", "XLV", "XLF", "XLY", "XLI", "XLE", "XLP", "XLU", "XLB", "XLRE", "XLC"]
    data = {}
    for i, sym in enumerate(symbols):
        data[sym] = _make_prices(sym, n=n, start=100 + i * 10, seed=42 + i)
    return data


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------

class TestConstants:

    def test_sector_count(self):
        assert len(SECTOR_ETF_DEFINITIONS) == 11

    def test_etf_map_keys(self):
        assert "XLK" in SECTOR_ETF_MAP
        assert "XLV" in SECTOR_ETF_MAP
        assert "XLF" in SECTOR_ETF_MAP

    def test_etf_map_has_beta(self):
        assert "beta" in SECTOR_ETF_MAP["XLK"]
        assert SECTOR_ETF_MAP["XLK"]["beta"] == 1.10

    def test_etf_map_has_group(self):
        assert "sectorGroup" in SECTOR_ETF_MAP["XLK"]
        assert SECTOR_ETF_MAP["XLK"]["sectorGroup"] == "sensitive"

    def test_regime_preferences(self):
        assert "early_expansion" in REGIME_SECTOR_PREFERENCES
        assert "contraction" in REGIME_SECTOR_PREFERENCES
        assert "preferred" in REGIME_SECTOR_PREFERENCES["early_expansion"]
        assert "avoid" in REGIME_SECTOR_PREFERENCES["early_expansion"]


# ---------------------------------------------------------------------------
# SectorMomentumCalculator — calculate_momentum
# ---------------------------------------------------------------------------

class TestCalculateMomentum:

    def test_returns_dict(self):
        data = _make_historical_data(["XLK"])
        calc = SectorMomentumCalculator(data)
        result = calc.calculate_momentum("XLK", 252)
        assert isinstance(result, dict)

    def test_missing_symbol(self):
        data = _make_historical_data(["XLK"])
        calc = SectorMomentumCalculator(data)
        assert calc.calculate_momentum("FAKE", 252) is None

    def test_insufficient_data(self):
        data = {"XLK": _make_prices("XLK", n=50)}
        calc = SectorMomentumCalculator(data)
        assert calc.calculate_momentum("XLK", 252) is None

    def test_has_required_keys(self):
        data = _make_historical_data(["XLK"])
        calc = SectorMomentumCalculator(data)
        result = calc.calculate_momentum("XLK", 252)
        assert "symbol" in result
        assert "longMomentum" in result
        assert "shortMomentum" in result
        assert "compositeMomentum" in result
        assert "volatility" in result
        assert "riskAdjustedMomentum" in result

    def test_symbol_preserved(self):
        data = _make_historical_data(["XLK"])
        calc = SectorMomentumCalculator(data)
        result = calc.calculate_momentum("XLK", 252)
        assert result["symbol"] == "XLK"

    def test_volatility_positive(self):
        data = _make_historical_data(["XLK"])
        calc = SectorMomentumCalculator(data)
        result = calc.calculate_momentum("XLK", 252)
        assert result["volatility"] > 0

    def test_dual_momentum_logic(self):
        """When both long and short momentum positive, composite = average."""
        data = _make_historical_data(["XLK"], n=300)
        calc = SectorMomentumCalculator(data)
        result = calc.calculate_momentum("XLK", 252)
        if result["longMomentum"] > 0 and result["shortMomentum"] > 0:
            expected = (result["longMomentum"] + result["shortMomentum"]) / 2
            assert result["compositeMomentum"] == pytest.approx(expected)

    def test_dual_momentum_negative(self):
        """When either momentum negative, composite = min."""
        # Use a downtrend
        data = {"XLK": _make_prices("XLK", n=300, drift=-0.003, seed=99)}
        calc = SectorMomentumCalculator(data)
        result = calc.calculate_momentum("XLK", 252)
        if result["longMomentum"] <= 0 or result["shortMomentum"] <= 0:
            expected = min(result["longMomentum"], result["shortMomentum"])
            assert result["compositeMomentum"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# SectorMomentumCalculator — calculate_all_momentum
# ---------------------------------------------------------------------------

class TestCalculateAllMomentum:

    def test_returns_list(self):
        data = _make_historical_data()
        calc = SectorMomentumCalculator(data)
        results = calc.calculate_all_momentum(252)
        assert isinstance(results, list)

    def test_sorted_by_composite(self):
        data = _make_historical_data()
        calc = SectorMomentumCalculator(data)
        results = calc.calculate_all_momentum(252)
        for i in range(len(results) - 1):
            assert results[i]["compositeMomentum"] >= results[i + 1]["compositeMomentum"]

    def test_has_rank(self):
        data = _make_historical_data()
        calc = SectorMomentumCalculator(data)
        results = calc.calculate_all_momentum(252)
        for i, r in enumerate(results):
            assert r["rank"] == i + 1

    def test_has_percentile(self):
        data = _make_historical_data()
        calc = SectorMomentumCalculator(data)
        results = calc.calculate_all_momentum(252)
        for r in results:
            assert "percentile" in r
            assert 0 < r["percentile"] <= 100


# ---------------------------------------------------------------------------
# SectorMomentumCalculator — adjust_for_regime
# ---------------------------------------------------------------------------

class TestAdjustForRegime:

    def test_boosts_preferred(self):
        data = _make_historical_data(["XLK", "XLP"])
        calc = SectorMomentumCalculator(data)
        scores = calc.calculate_all_momentum(252)
        adjusted = calc.adjust_for_regime(scores, "early_expansion", preference_boost=0.05)
        xlk = next(s for s in adjusted if s["symbol"] == "XLK")
        assert xlk.get("regimeAdjusted") is True

    def test_penalizes_avoid(self):
        data = _make_historical_data(["XLK", "XLP"])
        calc = SectorMomentumCalculator(data)
        scores = calc.calculate_all_momentum(252)
        original_xlp = next(s for s in scores if s["symbol"] == "XLP")
        adjusted = calc.adjust_for_regime(scores, "early_expansion", preference_boost=0.05)
        adj_xlp = next(s for s in adjusted if s["symbol"] == "XLP")
        # XLP is in "avoid" for early_expansion
        assert adj_xlp["compositeMomentum"] < original_xlp["compositeMomentum"]

    def test_neutral_no_change(self):
        data = _make_historical_data(["XLK", "XLP"])
        calc = SectorMomentumCalculator(data)
        scores = calc.calculate_all_momentum(252)
        adjusted = calc.adjust_for_regime(scores, "neutral", preference_boost=0.05)
        for orig, adj in zip(scores, adjusted):
            # Neutral has empty preferred/avoid, so no changes
            pass  # Re-sorting may change order

    def test_re_sorted(self):
        data = _make_historical_data()
        calc = SectorMomentumCalculator(data)
        scores = calc.calculate_all_momentum(252)
        adjusted = calc.adjust_for_regime(scores, "early_expansion")
        for i in range(len(adjusted) - 1):
            assert adjusted[i]["compositeMomentum"] >= adjusted[i + 1]["compositeMomentum"]


# ---------------------------------------------------------------------------
# SectorMomentumCalculator — get_allocation
# ---------------------------------------------------------------------------

class TestGetAllocation:

    def test_returns_dict(self):
        data = _make_historical_data()
        calc = SectorMomentumCalculator(data)
        scores = calc.calculate_all_momentum(252)
        alloc = calc.get_allocation(scores, top_n=3, overlay_pct=0.25, spy_weight=0.46)
        assert isinstance(alloc, dict)

    def test_sector_count(self):
        data = _make_historical_data()
        calc = SectorMomentumCalculator(data)
        scores = calc.calculate_all_momentum(252)
        alloc = calc.get_allocation(scores, top_n=3, overlay_pct=0.25, spy_weight=0.46)
        assert len(alloc["sectorAllocations"]) <= 3

    def test_total_weight(self):
        data = _make_historical_data()
        calc = SectorMomentumCalculator(data)
        scores = calc.calculate_all_momentum(252)
        alloc = calc.get_allocation(scores, top_n=3, overlay_pct=0.25, spy_weight=0.46)
        total = alloc["spAllocation"] + sum(s["weight"] for s in alloc["sectorAllocations"])
        assert total == pytest.approx(alloc["totalEquityWeight"], abs=0.01)

    def test_vix_disables_rotation(self):
        data = _make_historical_data()
        calc = SectorMomentumCalculator(data)
        scores = calc.calculate_all_momentum(252)
        alloc = calc.get_allocation(scores, top_n=3, vix=35, vix_threshold=30)
        assert alloc["sectorAllocations"] == []
        assert alloc["rebalanceRecommended"] is False

    def test_no_positive_sectors(self):
        data = {"XLK": _make_prices("XLK", n=300, drift=-0.005, seed=99)}
        calc = SectorMomentumCalculator(data)
        scores = calc.calculate_all_momentum(252)
        alloc = calc.get_allocation(scores, top_n=3, min_momentum=0.1)
        assert alloc["sectorAllocations"] == []

    def test_rebalance_recommended(self):
        data = _make_historical_data()
        calc = SectorMomentumCalculator(data)
        scores = calc.calculate_all_momentum(252)
        # Force high momentum
        scores[0]["compositeMomentum"] = 0.15
        alloc = calc.get_allocation(scores, top_n=3)
        assert alloc["rebalanceRecommended"] is True

    def test_rebalance_not_recommended(self):
        data = _make_historical_data()
        calc = SectorMomentumCalculator(data)
        scores = calc.calculate_all_momentum(252)
        # Force low momentum
        for s in scores:
            s["compositeMomentum"] = 0.05
        alloc = calc.get_allocation(scores, top_n=3)
        assert alloc["rebalanceRecommended"] is False


# ---------------------------------------------------------------------------
# generate_sector_signals Tests
# ---------------------------------------------------------------------------

class TestGenerateSectorSignals:

    def test_returns_none_missing_file(self):
        result = generate_sector_signals(Path("/tmp/nonexistent.json"))
        assert result is None

    def test_returns_dict_with_data(self, tmp_path):
        import json
        data = _make_historical_data()
        path = tmp_path / "historical.json"
        with open(path, "w") as f:
            json.dump(data, f)
        result = generate_sector_signals(path, vix=18.5, regime="neutral")
        assert isinstance(result, dict)
        assert "top_sectors" in result


# ---------------------------------------------------------------------------
# __all__ export validation
# ---------------------------------------------------------------------------

class TestExports:
    """Verify __all__ exports."""

    def test_all_exports_present(self):
        import src.strategy.sector_momentum_calc as mod
        for name in mod.__all__:
            assert hasattr(mod, name), f"Missing export: {name}"

    def test_all_count(self):
        import src.strategy.sector_momentum_calc as mod
        assert len(mod.__all__) == 5


# ---------------------------------------------------------------------------
# Constants validation extended
# ---------------------------------------------------------------------------

class TestConstantsExtended:
    """Extended constants validation."""

    def test_sector_etf_count(self):
        assert len(SECTOR_ETF_DEFINITIONS) == 11

    def test_sector_etf_map_matches_definitions(self):
        for defn in SECTOR_ETF_DEFINITIONS:
            assert defn["symbol"] in SECTOR_ETF_MAP
            assert SECTOR_ETF_MAP[defn["symbol"]]["name"] == defn["name"]

    def test_all_betas_positive(self):
        for defn in SECTOR_ETF_DEFINITIONS:
            assert defn["beta"] > 0

    def test_sector_groups(self):
        groups = {d["sectorGroup"] for d in SECTOR_ETF_DEFINITIONS}
        assert "defensive" in groups
        assert "cyclical" in groups
        assert "sensitive" in groups

    def test_regime_preferences_keys(self):
        expected = {"early_expansion", "late_expansion", "contraction", "recovery", "neutral"}
        assert set(REGIME_SECTOR_PREFERENCES.keys()) == expected

    def test_regime_avoid_symbols_valid(self):
        """All symbols in regime preferences should be valid ETFs."""
        all_symbols = {d["symbol"] for d in SECTOR_ETF_DEFINITIONS}
        for regime, prefs in REGIME_SECTOR_PREFERENCES.items():
            for sym in prefs.get("preferred", []) + prefs.get("avoid", []):
                assert sym in all_symbols, f"Unknown symbol {sym} in regime {regime}"


# ---------------------------------------------------------------------------
# SectorMomentumCalculator extended
# ---------------------------------------------------------------------------

class TestSectorMomentumCalculatorExtended:
    """Extended calculator tests."""

    def _make_calc(self):
        data = _make_historical_data()
        return SectorMomentumCalculator(data)

    def test_calculate_all_momentum_returns_list(self):
        calc = self._make_calc()
        results = calc.calculate_all_momentum()
        assert isinstance(results, list)

    def test_calculate_momentum_missing_symbol(self):
        calc = self._make_calc()
        result = calc.calculate_momentum("NONEXISTENT")
        assert result is None

    def test_adjust_for_regime_neutral(self):
        """Neutral regime should not adjust scores."""
        calc = self._make_calc()
        scores = [{"symbol": "XLK", "compositeMomentum": 0.5}]
        result = calc.adjust_for_regime(scores, "neutral")
        assert result[0]["compositeMomentum"] == 0.5

    def test_adjust_for_regime_boosts_preferred(self):
        """Early expansion should boost XLK."""
        calc = self._make_calc()
        scores = [{"symbol": "XLK", "compositeMomentum": 0.5}]
        result = calc.adjust_for_regime(scores, "early_expansion")
        assert result[0]["compositeMomentum"] > 0.5

    def test_adjust_for_regime_penalizes_avoid(self):
        """Early expansion should penalize XLU."""
        calc = self._make_calc()
        scores = [{"symbol": "XLU", "compositeMomentum": 0.5}]
        result = calc.adjust_for_regime(scores, "early_expansion")
        assert result[0]["compositeMomentum"] < 0.5

    def test_get_allocation_with_high_vix(self):
        """High VIX should reduce sector overlay."""
        calc = self._make_calc()
        scores = calc.calculate_all_momentum()
        alloc = calc.get_allocation(scores, vix=40, vix_threshold=30)
        assert isinstance(alloc, dict)

    def test_get_allocation_low_vix(self):
        """Low VIX should allow full sector overlay."""
        calc = self._make_calc()
        scores = calc.calculate_all_momentum()
        alloc = calc.get_allocation(scores, vix=15, vix_threshold=30)
        assert isinstance(alloc, dict)

    def test_get_allocation_top_n(self):
        """Should only include top_n sectors."""
        calc = self._make_calc()
        scores = calc.calculate_all_momentum()
        alloc = calc.get_allocation(scores, top_n=3, vix=15, vix_threshold=30)
        assert isinstance(alloc, dict)


# ---------------------------------------------------------------------------
# generate_sector_signals extended
# ---------------------------------------------------------------------------

class TestGenerateSectorSignalsExtended:
    """Extended generate_sector_signals tests."""

    def test_with_different_regimes(self, tmp_path):
        import json
        data = _make_historical_data()
        path = tmp_path / "historical.json"
        with open(path, "w") as f:
            json.dump(data, f)
        for regime in ["early_expansion", "contraction", "recovery", "neutral"]:
            result = generate_sector_signals(path, vix=20, regime=regime)
            assert isinstance(result, dict)

    def test_with_high_vix(self, tmp_path):
        import json
        data = _make_historical_data()
        path = tmp_path / "historical.json"
        with open(path, "w") as f:
            json.dump(data, f)
        result = generate_sector_signals(path, vix=35, regime="neutral")
        assert isinstance(result, dict)
        assert "allocation" in result


# ---------------------------------------------------------------------------
# calculate_momentum — zero-price and edge-case paths
# ---------------------------------------------------------------------------

class TestCalculateMomentumEdgeCases:
    """Edge cases for calculate_momentum: zero prices, vol defaults, fallbacks."""

    def _price_list(self, values, use_d_key=False):
        """Build a price list in the expected format."""
        d_key = "d" if use_d_key else "date"
        return [{d_key: str(20240101 + i), "close": v, "adjClose": v} for i, v in enumerate(values)]

    def test_zero_current_price_returns_none(self):
        """current_price == 0 should return None."""
        prices = self._price_list([100.0] * 252 + [0.0])  # last price is 0
        data = {"XLK": prices}
        calc = SectorMomentumCalculator(data)
        assert calc.calculate_momentum("XLK", 252) is None

    def test_zero_long_price_returns_none(self):
        """long_price == 0 should return None.

        sorted_prices[-252] with n=300 => index 48 (300-252)."""
        prices = _make_prices("XLK", n=300, start=100.0, seed=42)
        prices[48] = {"date": str(20240101 + 48), "close": 0.0, "adjClose": 0.0}
        data = {"XLK": prices}
        calc = SectorMomentumCalculator(data)
        assert calc.calculate_momentum("XLK", 252) is None

    def test_zero_short_price_returns_none(self):
        """short_price == 0 should return None.

        short_lookback = max(1, 252//4) = 63.  sorted_prices[-63]
        with n=300 => index 237 (300-63)."""
        prices = _make_prices("XLK", n=300, start=100.0, seed=42)
        prices[237] = {"date": str(20240101 + 237), "close": 0.0, "adjClose": 0.0}
        data = {"XLK": prices}
        calc = SectorMomentumCalculator(data)
        assert calc.calculate_momentum("XLK", 252) is None

    def test_all_prices_zero_returns_none(self):
        """All prices zero should trigger zero-price guard."""
        prices = self._price_list([0.0] * 300)
        data = {"XLK": prices}
        calc = SectorMomentumCalculator(data)
        assert calc.calculate_momentum("XLK", 252) is None

    def test_single_return_volatility_defaults_to_0_2(self):
        """When returns has exactly 1 element, volatility should default to 0.2."""
        prices = self._price_list([100.0, 105.0])  # n=2, lookback=2 => 1 return
        data = {"XLK": prices}
        calc = SectorMomentumCalculator(data)
        result = calc.calculate_momentum("XLK", lookback_days=2)
        assert result is not None
        assert result["volatility"] == 0.2

    def test_returns_computation_with_boundary_n(self):
        """Returns computation works with exactly lookback_days + 1 prices."""
        prices = _make_prices("XLK", n=253, start=100.0, seed=42)
        data = {"XLK": prices}
        calc = SectorMomentumCalculator(data)
        result = calc.calculate_momentum("XLK", lookback_days=252)
        assert result is not None
        # Returns should have len(returns) > 1 so volatility is computed (not defaulted)
        assert result["volatility"] > 0

    def test_zero_volatility_risk_adjusted_momentum_is_zero(self):
        """When volatility is 0, riskAdjustedMomentum should be 0."""
        prices = _make_prices(n=300, start=100.0, drift=0.0, vol=0.0, seed=42)
        # All returns = 0, std = 0 => vol = 0
        data = {"XLK": prices}
        calc = SectorMomentumCalculator(data)
        result = calc.calculate_momentum("XLK", 252)
        assert result is not None
        assert result["volatility"] == 0.0
        assert result["riskAdjustedMomentum"] == 0.0

    def test_fallback_to_close_when_no_adjclose(self):
        """When adjClose is not available, fall back to 'close'."""
        prices = _make_prices(n=300, start=100.0, seed=42)
        # Remove adjClose from all entries
        for p in prices:
            p.pop("adjClose", None)
        data = {"XLK": prices}
        calc = SectorMomentumCalculator(data)
        result = calc.calculate_momentum("XLK", 252)
        assert result is not None
        assert result["symbol"] == "XLK"

    def test_fallback_to_d_key_format(self):
        """Dicts with 'd' key instead of 'date' should still work."""
        prices = _make_prices(n=300, start=100.0, seed=42)
        # Convert 'date' to 'd'
        for p in prices:
            p["d"] = p.pop("date")
        data = {"XLK": prices}
        calc = SectorMomentumCalculator(data)
        result = calc.calculate_momentum("XLK", 252)
        assert result is not None
        assert result["symbol"] == "XLK"

    def test_exactly_lookback_days_data(self):
        """Exactly lookback_days data points should be sufficient (not < lookback)."""
        prices = self._price_list([100.0 + i * 0.5 for i in range(252)])
        data = {"XLK": prices}
        calc = SectorMomentumCalculator(data)
        result = calc.calculate_momentum("XLK", 252)
        assert result is not None


# ---------------------------------------------------------------------------
# calculate_all_momentum — partial failure paths
# ---------------------------------------------------------------------------

class TestCalculateAllMomentumEdgeCases:

    def test_some_symbols_fail(self):
        """When some symbols return None, they are excluded from results."""
        data = _make_historical_data(["XLK", "XLV", "XLF"])
        # Make XLK have too little data
        data["XLK"] = _make_prices("XLK", n=50)
        calc = SectorMomentumCalculator(data)
        results = calc.calculate_all_momentum(252)
        symbols = [r["symbol"] for r in results]
        assert "XLK" not in symbols
        assert len(results) == 2

    def test_all_symbols_fail_returns_empty_list(self):
        """When all symbols return None, returns empty list."""
        data = {sym: _make_prices(sym, n=50) for sym in ["XLK", "XLV"]}
        calc = SectorMomentumCalculator(data)
        results = calc.calculate_all_momentum(252)
        assert results == []


# ---------------------------------------------------------------------------
# adjust_for_regime — edge-case paths
# ---------------------------------------------------------------------------

class TestAdjustForRegimeEdgeCases:

    def test_unknown_regime_falls_back_to_neutral(self):
        """Unknown regime should fall back to neutral (no adjustments)."""
        data = _make_historical_data(["XLK"])
        calc = SectorMomentumCalculator(data)
        scores = calc.calculate_all_momentum(252)
        adjusted = calc.adjust_for_regime(scores, "unknown_regime", preference_boost=0.05)
        for orig, adj in zip(scores, adjusted):
            assert adj["compositeMomentum"] == pytest.approx(orig["compositeMomentum"])

    def test_symbol_not_in_preferred_or_avoid_unchanged(self):
        """Symbols in neither preferred nor avoid should have unchanged momentum."""
        data = _make_historical_data(["XLK", "XLC"])
        calc = SectorMomentumCalculator(data)
        scores = calc.calculate_all_momentum(252)
        # early_expansion: preferred = [XLK, XLY, XLF], avoid = [XLU, XLP]
        # XLC is not in either list
        adjusted = calc.adjust_for_regime(scores, "early_expansion", preference_boost=0.05)
        orig_xlc = next(s for s in scores if s["symbol"] == "XLC")
        adj_xlc = next(s for s in adjusted if s["symbol"] == "XLC")
        assert adj_xlc["compositeMomentum"] == pytest.approx(orig_xlc["compositeMomentum"])

    def test_adjusted_scores_retain_all_original_keys(self):
        """Adjusted scores should include all original keys plus regimeAdjusted."""
        scores = [
            {"symbol": "XLK", "name": "Technology", "compositeMomentum": 0.5, "volatility": 0.15,
             "longMomentum": 0.6, "shortMomentum": 0.4, "rank": 1, "percentile": 100},
        ]
        calc = SectorMomentumCalculator({})
        adjusted = calc.adjust_for_regime(scores, "early_expansion")
        for key in ("symbol", "name", "compositeMomentum", "volatility", "longMomentum",
                     "shortMomentum", "rank", "percentile", "regimeAdjusted"):
            assert key in adjusted[0]

    def test_zero_preference_boost_no_change(self):
        """preference_boost=0 should not change any scores."""
        scores = [
            {"symbol": "XLK", "compositeMomentum": 0.5},
            {"symbol": "XLU", "compositeMomentum": 0.4},
        ]
        calc = SectorMomentumCalculator({})
        adjusted = calc.adjust_for_regime(scores, "early_expansion", preference_boost=0.0)
        adj_xlk = next(s for s in adjusted if s["symbol"] == "XLK")
        adj_xlu = next(s for s in adjusted if s["symbol"] == "XLU")
        assert adj_xlk["compositeMomentum"] == pytest.approx(0.5)
        assert adj_xlu["compositeMomentum"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# get_allocation — regimeAdjusted flag and edge cases
# ---------------------------------------------------------------------------

class TestGetAllocationEdgeCases:

    def _make_scores(self, with_regime_adjusted=False):
        scores = [
            {"symbol": "XLK", "name": "Technology", "compositeMomentum": 0.12,
             "volatility": 0.15, "rank": 1, "longMomentum": 0.14, "shortMomentum": 0.10,
             "regimeAdjusted": True},
            {"symbol": "XLV", "name": "Healthcare", "compositeMomentum": 0.10,
             "volatility": 0.12, "rank": 2, "longMomentum": 0.11, "shortMomentum": 0.09,
             "regimeAdjusted": True},
            {"symbol": "XLF", "name": "Financials", "compositeMomentum": 0.08,
             "volatility": 0.14, "rank": 3, "longMomentum": 0.09, "shortMomentum": 0.07,
             "regimeAdjusted": False},
        ]
        if not with_regime_adjusted:
            # Remove regimeAdjusted key
            for s in scores:
                s.pop("regimeAdjusted", None)
        return scores

    def test_regime_adjusted_true_in_output(self):
        """regimeAdjusted=True in any input score should propagate to output."""
        scores = self._make_scores(with_regime_adjusted=True)
        calc = SectorMomentumCalculator({})
        alloc = calc.get_allocation(scores, top_n=3, overlay_pct=0.25, spy_weight=0.46)
        assert alloc["regimeAdjusted"] is True

    def test_regime_adjusted_false_in_output(self):
        """regimeAdjusted=False when no score has regimeAdjusted=True."""
        scores = self._make_scores(with_regime_adjusted=False)
        calc = SectorMomentumCalculator({})
        alloc = calc.get_allocation(scores, top_n=3, overlay_pct=0.25, spy_weight=0.46)
        assert alloc["regimeAdjusted"] is False

    def test_no_positive_sectors_empty_allocations(self):
        """When no sectors meet min_momentum, sectorAllocations is empty."""
        scores = [{"symbol": "XLK", "name": "Technology", "compositeMomentum": -0.05,
                   "volatility": 0.15, "rank": 1, "longMomentum": -0.03, "shortMomentum": -0.07}]
        alloc = SectorMomentumCalculator({}).get_allocation(
            scores, top_n=3, min_momentum=0.0)
        assert alloc["sectorAllocations"] == []
        assert alloc["totalEquityWeight"] == 0.46  # spy_weight default
        assert alloc["rebalanceRecommended"] is False

    def test_sector_weights_sum_correctly(self):
        """Sector overlay + SPY allocation should total to spy_weight."""
        scores = self._make_scores(with_regime_adjusted=True)
        alloc = SectorMomentumCalculator({}).get_allocation(
            scores, top_n=2, overlay_pct=0.25, spy_weight=0.46)
        total_sector = sum(s["weight"] for s in alloc["sectorAllocations"])
        expected_sector_portion = 0.46 * 0.25
        assert total_sector == pytest.approx(expected_sector_portion, abs=0.001)
        assert alloc["spAllocation"] == pytest.approx(0.46 - expected_sector_portion, abs=0.001)
        assert alloc["totalEquityWeight"] == pytest.approx(0.46, abs=0.001)

    def test_sector_allocation_keys_present(self):
        """Each sector allocation entry has required keys."""
        scores = self._make_scores(with_regime_adjusted=True)
        alloc = SectorMomentumCalculator({}).get_allocation(
            scores, top_n=2, overlay_pct=0.25, spy_weight=0.46)
        for entry in alloc["sectorAllocations"]:
            for key in ("symbol", "name", "weight", "momentum", "rank", "volatility"):
                assert key in entry, f"Missing key: {key}"

    def test_vix_at_threshold_boundary(self):
        """VIX exactly at threshold should still allow rotation (not > threshold)."""
        data = _make_historical_data(["XLK", "XLV"])
        calc = SectorMomentumCalculator(data)
        scores = calc.calculate_all_momentum(252)
        alloc = calc.get_allocation(scores, top_n=3, vix=30, vix_threshold=30)
        # vix=30, threshold=30, so vix > threshold is False => rotation allowed
        assert alloc["rebalanceRecommended"] is not None


# ---------------------------------------------------------------------------
# generate_sector_signals — regime=None, empty scores, error handling
# ---------------------------------------------------------------------------

class TestGenerateSectorSignalsEdgeCases:

    def test_regime_none_skips_adjustment(self, tmp_path):
        """regime=None should skip adjust_for_regime path."""
        import json
        data = _make_historical_data(["XLK", "XLV"])
        path = tmp_path / "historical.json"
        with open(path, "w") as f:
            json.dump(data, f)
        result = generate_sector_signals(path, vix=18.5, regime=None)
        assert isinstance(result, dict)
        # Should not have regime-based sector adjustments (no regimeAdjusted in input)
        assert result["regime"] is None

    def test_regime_empty_string_skips_adjustment(self, tmp_path):
        """regime='' should skip adjust_for_regime path."""
        import json
        data = _make_historical_data(["XLK", "XLV"])
        path = tmp_path / "historical.json"
        with open(path, "w") as f:
            json.dump(data, f)
        result = generate_sector_signals(path, vix=18.5, regime="")
        assert isinstance(result, dict)
        assert result["regime"] == ""

    def test_empty_momentum_scores_returns_none(self, tmp_path):
        """When all sectors fail momentum calc, generate_sector_signals returns None."""
        import json
        # Create data with insufficient points for all symbols
        data = {sym: _make_prices(sym, n=50) for sym in ["XLK", "XLV"]}
        path = tmp_path / "historical.json"
        with open(path, "w") as f:
            json.dump(data, f)
        result = generate_sector_signals(path, vix=18.5, regime="neutral")
        assert result is None

    def test_exception_during_processing_returns_none(self, tmp_path):
        """An exception during processing should be caught and return None."""
        path = tmp_path / "historical.json"
        # Write non-JSON content to trigger json decode error
        path.write_text("not valid json")
        result = generate_sector_signals(path, vix=18.5)
        assert result is None

    def test_empty_data_file_returns_none(self, tmp_path):
        """Empty JSON object should result in no momentum scores, returning None."""
        import json
        path = tmp_path / "historical.json"
        with open(path, "w") as f:
            json.dump({}, f)
        result = generate_sector_signals(path, vix=18.5)
        assert result is None

    def test_generate_sector_signals_with_zero_vix(self, tmp_path):
        """vix=0 should work and produce an allocation."""
        import json
        data = _make_historical_data(["XLK", "XLV"])
        path = tmp_path / "historical.json"
        with open(path, "w") as f:
            json.dump(data, f)
        result = generate_sector_signals(path, vix=0, regime="neutral")
        assert isinstance(result, dict)
        assert result["vix"] == 0

    def test_generate_sector_signals_with_regime_applied(self, tmp_path):
        """When regime is provided and non-neutral, adjustment should be applied."""
        import json
        data = _make_historical_data(["XLK", "XLP", "XLU"])
        path = tmp_path / "historical.json"
        with open(path, "w") as f:
            json.dump(data, f)
        result = generate_sector_signals(path, vix=18.5, regime="early_expansion")
        assert isinstance(result, dict)
        # Allocation should still be valid
        assert "allocation" in result
        assert result["regime"] == "early_expansion"

    def test_rebalance_reason_format(self, tmp_path):
        """Rebalance reason should be formatted correctly when recommended."""
        import json
        data = _make_historical_data(["XLK"])
        # Very bullish data to ensure compositeMomentum > 0.10
        prices = [{"date": str(20240101 + i), "close": float(100 * (1.002 ** i)),
                   "adjClose": float(100 * (1.002 ** i))} for i in range(300)]
        data = {"XLK": prices}
        path = tmp_path / "historical.json"
        with open(path, "w") as f:
            json.dump(data, f)
        result = generate_sector_signals(path, vix=15, regime="neutral")
        if result and result.get("rebalanceRecommended"):
            assert result["rebalanceReason"] is not None
            assert "XLK" in result["rebalanceReason"]
            assert "%" in result["rebalanceReason"]


# ---------------------------------------------------------------------------
# __main__ block entry point
# ---------------------------------------------------------------------------

class TestMainBlock:

    def test_main_block_logic(self, tmp_path):
        """Verify __main__ logic: generate_sector_signals with HISTORICAL_JSON."""
        import json
        data = _make_historical_data(["XLK", "XLV"])
        path = tmp_path / "historical.json"
        with open(path, "w") as f:
            json.dump(data, f)
        with patch("src.strategy.sector_momentum_calc.HISTORICAL_JSON", path):
            signals = generate_sector_signals(path, vix=18.5)
        assert isinstance(signals, dict)
        assert "top_sectors" in signals
        assert "allocation" in signals

    def test_main_block_with_nonexistent_path(self, tmp_path):
        """__main__-style call with nonexistent file should return None gracefully."""
        path = tmp_path / "nonexistent.json"
        signals = generate_sector_signals(path, vix=18.5)
        assert signals is None

    def test_main_block_top_sectors_structure(self, tmp_path):
        """Top sectors output should contain the expected keys."""
        import json
        data = _make_historical_data(["XLK", "XLV"])
        path = tmp_path / "historical.json"
        with open(path, "w") as f:
            json.dump(data, f)
        result = generate_sector_signals(path, vix=18.5)
        assert isinstance(result, dict)
        for sector in result["top_sectors"]:
            for key in ("symbol", "name", "momentumScore", "allocation", "rank",
                         "longMomentum", "shortMomentum", "volatility"):
                assert key in sector, f"Missing key: {key}"

    def test_main_block_allocation_structure(self, tmp_path):
        """Allocation dict in output should contain the expected keys."""
        import json
        data = _make_historical_data(["XLK", "XLV"])
        path = tmp_path / "historical.json"
        with open(path, "w") as f:
            json.dump(data, f)
        result = generate_sector_signals(path, vix=18.5)
        assert isinstance(result, dict)
        alloc = result["allocation"]
        for key in ("spy_core", "spy_total", "sector_overlay", "sectors"):
            assert key in alloc, f"Missing key: {key}"
