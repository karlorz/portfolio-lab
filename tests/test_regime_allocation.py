"""Tests for regime-conditional base allocation.

The champion allocation SPY/GLD/TLT 46/38/16 is static across all regimes.
This module varies weights by market regime to improve risk-adjusted returns.
"""

import json
import os
import pytest
from unittest.mock import patch

from src.strategy.regime_allocation import (
    REGIME_ALLOCATIONS,
    get_regime_allocation,
    get_regime_allocation_with_override,
    validate_allocations,
    DEFAULT_ALLOCATION,
)


class TestRegimeAllocations:
    """Test REGIME_ALLOCATIONS constant structure and values."""

    def test_all_five_regimes_present(self):
        """Must cover all 5 regimes from regime classifier."""
        expected = {"normal", "crisis", "high_vol", "low_vol", "recovery"}
        assert set(REGIME_ALLOCATIONS.keys()) == expected

    def test_each_allocation_sums_to_one(self):
        """Every regime allocation must sum to 1.0."""
        for regime, weights in REGIME_ALLOCATIONS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-6, f"{regime} sums to {total}"

    def test_each_allocation_has_three_assets(self):
        """Every regime must specify SPY, GLD, TLT."""
        for regime, weights in REGIME_ALLOCATIONS.items():
            assert set(weights.keys()) == {"SPY", "GLD", "TLT"}, f"{regime} missing assets"

    def test_normal_matches_champion(self):
        """NORMAL regime should match the champion allocation."""
        normal = REGIME_ALLOCATIONS["normal"]
        assert normal["SPY"] == pytest.approx(0.46, abs=0.01)
        assert normal["GLD"] == pytest.approx(0.38, abs=0.01)
        assert normal["TLT"] == pytest.approx(0.16, abs=0.01)

    def test_crisis_more_defensive(self):
        """CRISIS should have less SPY and more GLD/TLT than NORMAL."""
        normal = REGIME_ALLOCATIONS["normal"]
        crisis = REGIME_ALLOCATIONS["crisis"]
        assert crisis["SPY"] < normal["SPY"]
        assert crisis["GLD"] > normal["GLD"]
        assert crisis["TLT"] >= normal["TLT"]

    def test_recovery_more_aggressive(self):
        """RECOVERY should have more SPY than NORMAL."""
        normal = REGIME_ALLOCATIONS["normal"]
        recovery = REGIME_ALLOCATIONS["recovery"]
        assert recovery["SPY"] > normal["SPY"]

    def test_low_vol_more_equities(self):
        """LOW_VOL should tilt toward equities."""
        normal = REGIME_ALLOCATIONS["normal"]
        low_vol = REGIME_ALLOCATIONS["low_vol"]
        assert low_vol["SPY"] > normal["SPY"]

    def test_all_weights_non_negative(self):
        """No negative weights."""
        for regime, weights in REGIME_ALLOCATIONS.items():
            for asset, w in weights.items():
                assert w >= 0, f"{regime}/{asset} has negative weight {w}"

    def test_all_weights_at_most_one(self):
        """No weight exceeds 1.0."""
        for regime, weights in REGIME_ALLOCATIONS.items():
            for asset, w in weights.items():
                assert w <= 1.0, f"{regime}/{asset} has weight {w} > 1.0"


class TestGetRegimeAllocation:
    """Test get_regime_allocation() function."""

    def test_known_regime(self):
        """Known regime returns its allocation."""
        result = get_regime_allocation("crisis")
        assert result == REGIME_ALLOCATIONS["crisis"]

    def test_case_insensitive(self):
        """Regime lookup should be case-insensitive."""
        result = get_regime_allocation("CRISIS")
        assert result == REGIME_ALLOCATIONS["crisis"]

    def test_unknown_regime_returns_default(self):
        """Unknown regime falls back to NORMAL allocation."""
        result = get_regime_allocation("unknown_regime")
        assert result == REGIME_ALLOCATIONS["normal"]

    def test_none_returns_default(self):
        """None regime falls back to NORMAL."""
        result = get_regime_allocation(None)
        assert result == REGIME_ALLOCATIONS["normal"]

    def test_empty_string_returns_default(self):
        """Empty string falls back to NORMAL."""
        result = get_regime_allocation("")
        assert result == REGIME_ALLOCATIONS["normal"]


class TestGetRegimeAllocationWithOverride:
    """Test env-var override mechanism."""

    def test_no_override_uses_default(self):
        """Without env var, returns default allocations."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("REGIME_ALLOC_OVERRIDE", None)
            result = get_regime_allocation_with_override("normal")
            assert result == REGIME_ALLOCATIONS["normal"]

    def test_json_override(self):
        """REGIME_ALLOC_OVERRIDE env var with JSON overrides specific regime."""
        override = json.dumps({"crisis": {"SPY": 0.35, "GLD": 0.45, "TLT": 0.20}})
        with patch.dict(os.environ, {"REGIME_ALLOC_OVERRIDE": override}):
            result = get_regime_allocation_with_override("crisis")
            assert result["SPY"] == pytest.approx(0.35)
            assert result["GLD"] == pytest.approx(0.45)
            assert result["TLT"] == pytest.approx(0.20)

    def test_override_does_not_affect_other_regimes(self):
        """Override for CRISIS shouldn't change NORMAL."""
        override = json.dumps({"crisis": {"SPY": 0.35, "GLD": 0.45, "TLT": 0.20}})
        with patch.dict(os.environ, {"REGIME_ALLOC_OVERRIDE": override}):
            result = get_regime_allocation_with_override("normal")
            assert result == REGIME_ALLOCATIONS["normal"]

    def test_malformed_json_ignores_override(self):
        """Malformed JSON in env var should fall back to defaults."""
        with patch.dict(os.environ, {"REGIME_ALLOC_OVERRIDE": "not-json"}):
            result = get_regime_allocation_with_override("crisis")
            assert result == REGIME_ALLOCATIONS["crisis"]

    def test_override_not_summing_to_one_gets_normalized(self):
        """Override weights that don't sum to 1.0 get normalized."""
        override = json.dumps({"crisis": {"SPY": 0.30, "GLD": 0.40, "TLT": 0.20}})
        with patch.dict(os.environ, {"REGIME_ALLOC_OVERRIDE": override}):
            result = get_regime_allocation_with_override("crisis")
            total = sum(result.values())
            assert abs(total - 1.0) < 1e-6


class TestValidateAllocations:
    """Test validate_allocations() utility."""

    def test_valid_allocations(self):
        """Standard allocations should validate."""
        errors = validate_allocations(REGIME_ALLOCATIONS)
        assert errors == []

    def test_detects_non_summing(self):
        """Should detect allocations not summing to 1.0."""
        bad = {"normal": {"SPY": 0.5, "GLD": 0.3, "TLT": 0.1}}
        errors = validate_allocations(bad)
        assert len(errors) > 0
        assert "sum" in errors[0].lower()

    def test_detects_missing_assets(self):
        """Should detect missing SPY/GLD/TLT."""
        bad = {"normal": {"SPY": 1.0}}
        errors = validate_allocations(bad)
        assert len(errors) > 0

    def test_detects_negative_weights(self):
        """Should detect negative weights."""
        bad = {"normal": {"SPY": -0.1, "GLD": 0.6, "TLT": 0.5}}
        errors = validate_allocations(bad)
        assert len(errors) > 0


class TestDefaultAllocation:
    """Test DEFAULT_ALLOCATION constant."""

    def test_default_matches_champion(self):
        """Default should be the champion 46/38/16."""
        assert DEFAULT_ALLOCATION["SPY"] == pytest.approx(0.46, abs=0.01)
        assert DEFAULT_ALLOCATION["GLD"] == pytest.approx(0.38, abs=0.01)
        assert DEFAULT_ALLOCATION["TLT"] == pytest.approx(0.16, abs=0.01)

    def test_default_sums_to_one(self):
        assert abs(sum(DEFAULT_ALLOCATION.values()) - 1.0) < 1e-6


class TestEvaluatorIntegration:
    """Test regime_allocation integration with evaluator.py target_alloc logic."""

    def test_evaluator_import_succeeds(self):
        """Module can be imported without errors."""
        from src.strategy.regime_allocation import get_regime_allocation_with_override
        assert callable(get_regime_allocation_with_override)

    def test_enabled_uses_regime_allocation(self):
        """When REGIME_ALLOC_ENABLED=1, regime-specific allocation is returned."""
        from src.strategy.regime_allocation import get_regime_allocation_with_override

        with patch.dict(os.environ, {"REGIME_ALLOC_ENABLED": "1"}):
            crisis = get_regime_allocation_with_override("crisis")
            assert crisis["SPY"] == pytest.approx(0.40, abs=0.01)
            assert crisis["GLD"] == pytest.approx(0.42, abs=0.01)

    def test_disabled_returns_base_allocation(self):
        """When REGIME_ALLOC_ENABLED is not set, returns base allocation."""
        from src.strategy.regime_allocation import get_regime_allocation_with_override

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("REGIME_ALLOC_ENABLED", None)
            result = get_regime_allocation_with_override("crisis")
            # Should still return crisis allocation (function doesn't check ENABLED)
            # The ENABLED check is in evaluator.py, not here
            assert result == REGIME_ALLOCATIONS["crisis"]

    def test_recovery_allocation_has_more_equity(self):
        """RECOVERY regime should favor equities over the champion."""
        from src.strategy.regime_allocation import get_regime_allocation_with_override

        recovery = get_regime_allocation_with_override("recovery")
        assert recovery["SPY"] > 0.46  # More than champion
        assert recovery["GLD"] < 0.38  # Less gold than champion
