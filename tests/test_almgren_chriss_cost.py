"""Tests for src/strategy/almgren_chriss_cost.py — Almgren-Chriss Cost Model."""

import json
import tempfile
from pathlib import Path
import pytest
from src.strategy.almgren_chriss_cost import (
    TurnoverCostEstimate,
    CostParameters,
    AlmgrenChrissCostModel,
    get_default_cost_aversion,
    compute_cost_penalty,
    DEFAULT_SPREAD_COST,
    DEFAULT_IMPACT_COST,
    DEFAULT_COST_CALIBRATION,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def model():
    """Model with no TCA calibration, pointed at a temp dir to avoid real data."""
    with tempfile.TemporaryDirectory() as d:
        yield AlmgrenChrissCostModel(data_dir=Path(d), use_tca_calibration=False)


@pytest.fixture
def model_with_tca():
    """Model pointed at a temp dir with TCA feedback state."""
    with tempfile.TemporaryDirectory() as d:
        data_dir = Path(d)
        tca_state = {
            "status": "ok",
            "overall_quality": 68.0,
            "symbols": {
                "SPY": {"cost_calibration": 1.2, "avg_quality": 55.0, "quality_bucket": "fair"},
                "GLD": {"cost_calibration": 0.9, "avg_quality": 78.0, "quality_bucket": "good"},
                "TLT": {"cost_calibration": 1.5, "avg_quality": 42.0, "quality_bucket": "poor"},
            },
        }
        (data_dir / "tca_feedback_state.json").write_text(json.dumps(tca_state))
        yield AlmgrenChrissCostModel(data_dir=data_dir, use_tca_calibration=True)


# ── TurnoverCostEstimate ──────────────────────────────────────────────────

class TestTurnoverCostEstimate:
    def test_dataclass_fields(self):
        est = TurnoverCostEstimate(
            total_cost_bps=10.0, spread_cost_bps=3.0, impact_cost_bps=7.0,
            symbol_costs={"SPY": {"delta": 0.04, "total_bps": 10.0}},
            calibration_source="default", active_turnover_pct=4.0,
        )
        assert est.total_cost_bps == 10.0
        assert est.spread_cost_bps == 3.0
        assert est.impact_cost_bps == 7.0

    def test_to_dict(self):
        est = TurnoverCostEstimate(
            total_cost_bps=5.0, spread_cost_bps=2.0, impact_cost_bps=3.0,
            symbol_costs={"SPY": {"delta": 0.02, "total_bps": 5.0}},
            calibration_source="default", active_turnover_pct=2.0,
        )
        d = est.to_dict()
        assert d["total_cost_bps"] == 5.0
        assert d["calibration_source"] == "default"
        assert "SPY" in d["symbol_costs"]


# ── CostParameters ────────────────────────────────────────────────────────

class TestCostParameters:
    def test_dataclass_and_to_dict(self):
        cp = CostParameters(
            spread={"SPY": 0.5}, impact={"SPY": 0.3},
            calibration_source="default",
        )
        d = cp.to_dict()
        assert d["spread"]["SPY"] == 0.5
        assert d["impact"]["SPY"] == 0.3
        assert d["calibration_source"] == "default"


# ── AlmgrenChrissCostModel ────────────────────────────────────────────────

class TestInit:
    def test_default_parameters(self, model):
        assert model.use_tca_calibration is False
        assert model.default_cost_aversion == 0.01

    def test_custom_cost_aversion(self):
        model = AlmgrenChrissCostModel(
            use_tca_calibration=False, default_cost_aversion=0.05,
        )
        assert model.default_cost_aversion == 0.05

    def test_tca_calibration_enabled(self):
        model = AlmgrenChrissCostModel(use_tca_calibration=True)
        assert model.use_tca_calibration is True


class TestLoadTcaFeedback:
    def test_no_file_returns_none(self, model):
        with tempfile.TemporaryDirectory() as d:
            model.data_dir = Path(d)
            result = model._load_tca_feedback()
        assert result is None

    def test_no_data_status_returns_none(self, model):
        with tempfile.TemporaryDirectory() as d:
            data_dir = Path(d)
            (data_dir / "tca_feedback_state.json").write_text(
                json.dumps({"status": "no_data", "symbols": {}})
            )
            model.data_dir = data_dir
            model._tca_feedback = None
            result = model._load_tca_feedback()
        assert result is None

    def test_valid_feedback_returns_data(self, model):
        with tempfile.TemporaryDirectory() as d:
            data_dir = Path(d)
            tca = {"status": "ok", "overall_quality": 75.0, "symbols": {"SPY": {"cost_calibration": 1.1}}}
            (data_dir / "tca_feedback_state.json").write_text(json.dumps(tca))
            model.data_dir = data_dir
            model._tca_feedback = None
            result = model._load_tca_feedback()
        assert result is not None
        assert result["overall_quality"] == 75.0

    def test_invalid_json_returns_none(self, model):
        with tempfile.TemporaryDirectory() as d:
            data_dir = Path(d)
            (data_dir / "tca_feedback_state.json").write_text("not json")
            model.data_dir = data_dir
            model._tca_feedback = None
            result = model._load_tca_feedback()
        assert result is None

    def test_cached_feedback(self, model):
        with tempfile.TemporaryDirectory() as d:
            data_dir = Path(d)
            tca = {"status": "ok", "symbols": {"SPY": {"cost_calibration": 1.5}}}
            (data_dir / "tca_feedback_state.json").write_text(json.dumps(tca))
            model.data_dir = data_dir
            result1 = model._load_tca_feedback()
            result2 = model._load_tca_feedback()  # cached
        assert result1 is result2  # Same object (cached)


class TestSymbolCalibration:
    def test_no_feedback_returns_default(self, model):
        assert model._get_symbol_calibration("SPY") == 1.0

    def test_known_symbol_returns_calibration(self, model_with_tca):
        assert model_with_tca._get_symbol_calibration("SPY") == 1.2
        assert model_with_tca._get_symbol_calibration("GLD") == 0.9

    def test_unknown_symbol_returns_default(self, model_with_tca):
        assert model_with_tca._get_symbol_calibration("QQQ") == 1.0


class TestGetCostParams:
    def test_default_params_no_calibration(self, model):
        params = model.get_cost_params(["SPY", "GLD", "TLT"])
        assert params.spread["SPY"] == DEFAULT_SPREAD_COST["SPY"]
        assert params.impact["SPY"] == DEFAULT_IMPACT_COST["SPY"]
        assert params.calibration_source == "default"

    def test_params_with_tca_calibration(self, model_with_tca):
        params = model_with_tca.get_cost_params(["SPY", "GLD", "TLT"])
        # SPY spread = 0.5 * 1.2 = 0.6
        assert params.spread["SPY"] == pytest.approx(0.6)
        # GLD spread = 1.0 * 0.9 = 0.9
        assert params.spread["GLD"] == pytest.approx(0.9)
        # TLT spread = 1.2 * 1.5 = 1.8
        assert params.spread["TLT"] == pytest.approx(1.8)
        assert params.calibration_source == "tca_feedback"

    def test_unknown_asset_uses_defaults(self, model):
        params = model.get_cost_params(["UNKNOWN"])
        assert params.spread["UNKNOWN"] == 2.0  # default for unknown
        assert params.impact["UNKNOWN"] == 1.0

    def test_disabled_calibration_uses_defaults(self, model):
        model.use_tca_calibration = False
        params = model.get_cost_params(["SPY"])
        assert params.spread["SPY"] == DEFAULT_SPREAD_COST["SPY"]
        assert params.calibration_source == "default"

    def test_cost_aversion_passed_through(self, model):
        model.default_cost_aversion = 0.05
        params = model.get_cost_params(["SPY"])
        assert params.cost_aversion_default == 0.05


class TestGetTcaCalibrationSummary:
    def test_no_feedback(self, model):
        summary = model.get_tca_calibration_summary()
        assert summary["source"] == "default"
        assert summary["factors"] == {}

    def test_with_feedback(self, model_with_tca):
        summary = model_with_tca.get_tca_calibration_summary()
        assert summary["source"] == "tca_feedback"
        assert "SPY" in summary["factors"]
        assert summary["factors"]["SPY"]["cost_calibration"] == 1.2
        assert summary["overall_quality"] == 68.0


class TestEstimateTurnoverCost:
    def test_simple_rebalance(self, model):
        cur = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        tgt = {"SPY": 0.50, "GLD": 0.35, "TLT": 0.15}
        est = model.estimate_turnover_cost(cur, tgt)
        assert est.total_cost_bps > 0
        assert est.spread_cost_bps > 0
        assert "SPY" in est.symbol_costs

    def test_spy_cost_calculation(self, model):
        """Verify exact cost formula: spread * |delta| * 100 + impact * delta² * 10000."""
        # SPY: 0.46 → 0.50, delta = 0.04
        cur = {"SPY": 0.46}
        tgt = {"SPY": 0.50}
        est = model.estimate_turnover_cost(cur, tgt)
        # spread = 0.5 * 0.04 * 100 = 2.0 bps
        # impact = 0.3 * 0.04² * 10000 = 0.3 * 0.0016 * 10000 = 4.8 bps
        # total = 6.8 bps
        assert est.spread_cost_bps == pytest.approx(2.0)
        assert est.impact_cost_bps == pytest.approx(4.8)
        assert est.total_cost_bps == pytest.approx(6.8)

    def test_no_change_zero_cost(self, model):
        cur = {"SPY": 0.50, "TLT": 0.50}
        tgt = {"SPY": 0.50, "TLT": 0.50}
        est = model.estimate_turnover_cost(cur, tgt)
        assert est.total_cost_bps == pytest.approx(0.0)
        assert est.active_turnover_pct == pytest.approx(0.0)

    def test_new_asset_has_cost(self, model):
        cur = {"SPY": 1.0}
        tgt = {"SPY": 0.95, "GLD": 0.05}
        est = model.estimate_turnover_cost(cur, tgt)
        assert est.total_cost_bps > 0
        assert "GLD" in est.symbol_costs

    def test_fully_exited_asset_has_cost(self, model):
        cur = {"SPY": 0.90, "TLT": 0.10}
        tgt = {"SPY": 1.0}
        est = model.estimate_turnover_cost(cur, tgt)
        assert "TLT" in est.symbol_costs
        assert est.total_cost_bps > 0

    def test_symbol_costs_structure(self, model):
        cur = {"SPY": 0.50}
        tgt = {"SPY": 0.55}
        est = model.estimate_turnover_cost(cur, tgt)
        sc = est.symbol_costs["SPY"]
        assert "current" in sc
        assert "target" in sc
        assert "delta" in sc
        assert "turnover_pct" in sc
        assert "spread_bps" in sc
        assert "impact_bps" in sc
        assert "total_bps" in sc

    def test_active_turnover_half_of_total(self, model):
        """Active turnover = total_turnover_pct / 2 (single-sided)."""
        cur = {"SPY": 0.40, "GLD": 0.60}
        tgt = {"SPY": 0.60, "GLD": 0.40}
        est = model.estimate_turnover_cost(cur, tgt)
        # delta SPY = 0.20, delta GLD = -0.20
        # total_turnover_pct = 20 + 20 = 40
        # active_turnover = 40 / 2 = 20
        assert est.active_turnover_pct == pytest.approx(20.0)

    def test_btc_higher_cost_than_spy(self, model):
        cur = {"SPY": 0.50, "BTC": 0.50}
        tgt = {"SPY": 0.55, "BTC": 0.45}
        est = model.estimate_turnover_cost(cur, tgt)
        spy_cost = est.symbol_costs["SPY"]["total_bps"]
        btc_cost = est.symbol_costs["BTC"]["total_bps"]
        # Same delta of 0.05, BTC should cost much more
        assert btc_cost > spy_cost * 3


class TestEdgeCases:
    def test_empty_weights(self, model):
        est = model.estimate_turnover_cost({}, {})
        assert est.total_cost_bps == 0.0

    def test_mismatched_symbols(self, model):
        cur = {"SPY": 0.50}
        tgt = {"GLD": 0.50}
        est = model.estimate_turnover_cost(cur, tgt)
        assert "SPY" in est.symbol_costs
        assert "GLD" in est.symbol_costs

    def test_very_large_turnover(self, model):
        cur = {"SPY": 1.0}
        tgt = {"GLD": 1.0}
        est = model.estimate_turnover_cost(cur, tgt)
        assert est.total_cost_bps > 0


# ── Convenience Functions ─────────────────────────────────────────────────

class TestConvenienceFunctions:
    def test_get_default_cost_aversion(self):
        assert get_default_cost_aversion() == 0.01

    def test_compute_cost_penalty_basic(self):
        cur = {"SPY": 0.50, "GLD": 0.50}
        tgt = {"SPY": 0.55, "GLD": 0.45}
        spread = {"SPY": 0.5, "GLD": 1.0}
        impact = {"SPY": 0.3, "GLD": 0.6}
        penalty = compute_cost_penalty(tgt, cur, spread, impact, cost_aversion=0.01)
        assert penalty > 0
        assert isinstance(penalty, float)

    def test_compute_cost_penalty_no_change(self):
        cur = {"SPY": 0.50}
        tgt = {"SPY": 0.50}
        penalty = compute_cost_penalty(
            tgt, cur, {"SPY": 0.5}, {"SPY": 0.3}, cost_aversion=0.01
        )
        assert penalty == pytest.approx(0.0)

    def test_compute_cost_penalty_negligible_delta(self):
        cur = {"SPY": 0.50}
        tgt = {"SPY": 0.5000000001}
        penalty = compute_cost_penalty(
            tgt, cur, {"SPY": 0.5}, {"SPY": 0.3}, cost_aversion=0.01
        )
        assert penalty == pytest.approx(0.0)

    def test_compute_cost_penalty_unknown_symbol_defaults(self):
        cur = {"NEW": 0.50}
        tgt = {"NEW": 0.60}
        penalty = compute_cost_penalty(
            tgt, cur, {}, {}, cost_aversion=0.01
        )
        assert penalty > 0

    def test_compute_cost_penalty_scales_with_aversion(self):
        cur = {"SPY": 0.50}
        tgt = {"SPY": 0.60}
        spread = {"SPY": 0.5}
        impact = {"SPY": 0.3}
        p1 = compute_cost_penalty(tgt, cur, spread, impact, cost_aversion=0.01)
        p2 = compute_cost_penalty(tgt, cur, spread, impact, cost_aversion=0.05)
        assert p2 == pytest.approx(p1 * 5.0)


# ── Constants ─────────────────────────────────────────────────────────────

class TestDefaultCosts:
    def test_spy_costs(self):
        assert DEFAULT_SPREAD_COST["SPY"] == 0.5
        assert DEFAULT_IMPACT_COST["SPY"] == 0.3

    def test_crypto_higher_costs(self):
        assert DEFAULT_SPREAD_COST["BTC"] > DEFAULT_SPREAD_COST["SPY"]
        assert DEFAULT_IMPACT_COST["ETH"] > DEFAULT_IMPACT_COST["SPY"]

    def test_default_calibration(self):
        assert DEFAULT_COST_CALIBRATION == 1.0
