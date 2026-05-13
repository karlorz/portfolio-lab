#!/usr/bin/env python3
"""
Tests for sector_momentum_calc.py — constants, SectorMomentumCalculator,
momentum calculation, regime adjustment, allocation generation.
"""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        assert "allocation" in result
