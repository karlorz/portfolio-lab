"""Tests for commodity curve overlay v3.20."""
import pytest
import numpy as np
from datetime import datetime


class TestCurveRegime:
    """Dataclass and regime enum tests."""

    def test_curve_regime_enum_values(self):
        from src.signals.commodity_curve import CurveRegime
        assert CurveRegime.CONTANGO.value == -1
        assert CurveRegime.FLAT.value == 0
        assert CurveRegime.BACKWARDATION.value == 1

    def test_commodity_curve_signal_defaults(self):
        from src.signals.commodity_curve import CommodityCurveSignal, CurveRegime
        signal = CommodityCurveSignal(
            ticker="DBC",
            front_month_price=25.0,
            deferred_month_price=24.0,
            regime=CurveRegime.BACKWARDATION,
            spread_pct=4.0,
            timestamp=datetime(2026, 5, 14)
        )
        assert signal.ticker == "DBC"
        assert signal.front_month_price == 25.0
        assert signal.deferred_month_price == 24.0
        assert signal.regime == CurveRegime.BACKWARDATION
        assert signal.spread_pct == 4.0

    def test_commodity_curve_signal_contango(self):
        from src.signals.commodity_curve import CommodityCurveSignal, CurveRegime
        signal = CommodityCurveSignal(
            ticker="DBC",
            front_month_price=24.0,
            deferred_month_price=25.0,
            regime=CurveRegime.CONTANGO,
            spread_pct=-4.0,
            timestamp=datetime(2026, 5, 14)
        )
        assert signal.regime == CurveRegime.CONTANGO
        assert signal.spread_pct < 0

    def test_commodity_curve_signal_flat(self):
        from src.signals.commodity_curve import CommodityCurveSignal, CurveRegime
        signal = CommodityCurveSignal(
            ticker="USO",
            front_month_price=25.0,
            deferred_month_price=25.1,
            regime=CurveRegime.FLAT,
            spread_pct=-0.4,
            timestamp=datetime(2026, 5, 14)
        )
        assert signal.regime == CurveRegime.FLAT


class TestSpreadCalculation:
    """Front-month vs deferred-month spread calculation."""

    def test_compute_spread_backwardation(self):
        from src.signals.commodity_curve import compute_curve_spread, CurveRegime
        regime, spread = compute_curve_spread(
            front_price=25.0, deferred_price=24.0
        )
        assert regime == CurveRegime.BACKWARDATION
        assert spread > 0
        assert abs(spread - 4.0) < 0.01

    def test_compute_spread_contango(self):
        from src.signals.commodity_curve import compute_curve_spread, CurveRegime
        regime, spread = compute_curve_spread(
            front_price=24.0, deferred_price=25.0
        )
        assert regime == CurveRegime.CONTANGO
        assert spread < 0

    def test_compute_spread_flat(self):
        from src.signals.commodity_curve import compute_curve_spread, CurveRegime
        regime, spread = compute_curve_spread(
            front_price=25.0, deferred_price=24.9
        )
        assert regime == CurveRegime.FLAT
        assert -1.0 < spread < 0.5

    def test_compute_spread_zero_prices(self):
        from src.signals.commodity_curve import compute_curve_spread, CurveRegime
        regime, spread = compute_curve_spread(
            front_price=0.0, deferred_price=0.0
        )
        assert regime == CurveRegime.FLAT
        assert spread == 0.0

    def test_compute_spread_negative_prices(self):
        from src.signals.commodity_curve import compute_curve_spread, CurveRegime
        regime, spread = compute_curve_spread(
            front_price=-1.5, deferred_price=-1.0
        )
        assert regime == CurveRegime.CONTANGO
        assert spread < -1.0

    def test_compute_spread_threshold_boundary_contango(self):
        from src.signals.commodity_curve import compute_curve_spread, CurveRegime
        regime, spread = compute_curve_spread(
            front_price=100.0, deferred_price=101.0
        )
        assert spread == pytest.approx(-1.0)
        assert regime == CurveRegime.FLAT

    def test_compute_spread_threshold_boundary_backwardation(self):
        from src.signals.commodity_curve import compute_curve_spread, CurveRegime
        regime, spread = compute_curve_spread(
            front_price=100.0, deferred_price=99.5
        )
        assert spread == pytest.approx(0.5)
        assert regime == CurveRegime.FLAT

    def test_compute_spread_strong_backwardation(self):
        from src.signals.commodity_curve import compute_curve_spread, CurveRegime
        regime, spread = compute_curve_spread(
            front_price=100.0, deferred_price=90.0
        )
        assert regime == CurveRegime.BACKWARDATION
        assert spread > 5.0


class TestCurveFetcher:
    """Fetch and compute curve regimes from price data."""

    def test_fetch_curve_for_dbc_mock(self, tmp_path):
        from src.signals.commodity_curve import fetch_curve_signal, CurveRegime
        import json

        mock_prices = {
            "DBC": [
                {"d": "2026-05-14", "p": 25.0},
                {"d": "2026-04-23", "p": 24.0},
            ]
        }
        p = tmp_path / "prices.json"
        p.write_text(json.dumps(mock_prices))

        signal = fetch_curve_signal("DBC", prices_path=str(p))
        assert signal.ticker == "DBC"
        assert isinstance(signal.regime, CurveRegime)
        assert signal.front_month_price > 0

    def test_fetch_curve_unknown_ticker(self):
        from src.signals.commodity_curve import fetch_curve_signal
        with pytest.raises(ValueError, match="No price data"):
            fetch_curve_signal("UNKNOWN_TICKER_XYZ", prices_path="/nonexistent")

    def test_fetch_curve_insufficient_data(self, tmp_path):
        import json
        prices = {"DBC": [{"d": "2026-05-14", "p": 25.0}]}
        p = tmp_path / "prices.json"
        p.write_text(json.dumps(prices))

        from src.signals.commodity_curve import fetch_curve_signal
        with pytest.raises(ValueError, match="Insufficient"):
            fetch_curve_signal("DBC", prices_path=str(p))


class TestBulkFetch:
    """Bulk curve regime fetch for multiple commodity ETFs."""

    def test_fetch_all_commodity_curves(self, tmp_path):
        from src.signals.commodity_curve import fetch_all_curves, COMMODITY_ETFS
        import json

        mock_prices = {
            ticker: [
                {"d": "2026-05-14", "p": 25.0},
                {"d": "2026-04-23", "p": 24.0},
            ]
            for ticker in COMMODITY_ETFS
        }
        p = tmp_path / "prices.json"
        p.write_text(json.dumps(mock_prices))

        results = fetch_all_curves(prices_path=str(p))
        assert len(results) == len(COMMODITY_ETFS)
        assert all(s.ticker in COMMODITY_ETFS for s in results.values())

    def test_fetch_all_respects_ticker_filter(self, tmp_path):
        from src.signals.commodity_curve import fetch_all_curves
        import json

        mock_prices = {
            "DBC": [
                {"d": "2026-05-14", "p": 25.0},
                {"d": "2026-04-23", "p": 24.0},
            ],
            "USO": [
                {"d": "2026-05-14", "p": 70.0},
                {"d": "2026-04-23", "p": 72.0},
            ],
        }
        p = tmp_path / "prices.json"
        p.write_text(json.dumps(mock_prices))

        results = fetch_all_curves(prices_path=str(p), tickers=["DBC"])
        assert len(results) == 1
        assert "DBC" in results
        assert "USO" not in results

    def test_get_curve_summary(self, tmp_path):
        from src.signals.commodity_curve import fetch_all_curves, get_curve_summary
        import json

        mock_prices = {
            "DBC": [
                {"d": "2026-05-14", "p": 25.0},
                {"d": "2026-04-23", "p": 24.0},  # backwardation
            ],
            "USO": [
                {"d": "2026-05-14", "p": 70.0},
                {"d": "2026-04-23", "p": 72.0},  # contango
            ],
        }
        p = tmp_path / "prices.json"
        p.write_text(json.dumps(mock_prices))

        results = fetch_all_curves(prices_path=str(p), tickers=["DBC", "USO"])
        summary = get_curve_summary(results)
        assert summary["total"] == 2
        assert summary["backwardation"] >= 1
        assert summary["contango"] >= 1


class TestAllocationGate:
    """Curve-gated commodity allocation."""

    def test_allocation_allowed_backwardation(self):
        from src.signals.commodity_curve import (
            CurveRegime, CommodityCurveSignal, get_commodity_allocation
        )
        signal = CommodityCurveSignal(
            ticker="DBC", front_month_price=25.0, deferred_month_price=24.0,
            regime=CurveRegime.BACKWARDATION, spread_pct=4.0,
            timestamp=datetime(2026, 5, 14)
        )
        alloc = get_commodity_allocation(signal, base_weight=5.0)
        assert alloc == pytest.approx(5.0)

    def test_allocation_zero_contango(self):
        from src.signals.commodity_curve import (
            CurveRegime, CommodityCurveSignal, get_commodity_allocation
        )
        signal = CommodityCurveSignal(
            ticker="DBC", front_month_price=24.0, deferred_month_price=25.0,
            regime=CurveRegime.CONTANGO, spread_pct=-4.0,
            timestamp=datetime(2026, 5, 14)
        )
        alloc = get_commodity_allocation(signal, base_weight=5.0)
        assert alloc == 0.0

    def test_allocation_flat_reduced(self):
        from src.signals.commodity_curve import (
            CurveRegime, CommodityCurveSignal, get_commodity_allocation
        )
        signal = CommodityCurveSignal(
            ticker="DBC", front_month_price=25.0, deferred_month_price=24.9,
            regime=CurveRegime.FLAT, spread_pct=0.4,
            timestamp=datetime(2026, 5, 14)
        )
        alloc = get_commodity_allocation(signal, base_weight=5.0)
        assert 0.0 < alloc < 5.0

    def test_allocation_none_signal_returns_zero(self):
        from src.signals.commodity_curve import get_commodity_allocation
        assert get_commodity_allocation(None, base_weight=5.0) == 0.0

    def test_compute_portfolio_commodity_weight_no_signals(self):
        from src.signals.commodity_curve import compute_commodity_allocation
        result = compute_commodity_allocation({}, max_weight=5.0)
        assert result["dbc_weight"] == 0.0
        assert result["allocation_allowed"] is False

    def test_compute_allocation_backwardation(self):
        from src.signals.commodity_curve import (
            CurveRegime, CommodityCurveSignal, compute_commodity_allocation
        )
        signal = CommodityCurveSignal(
            ticker="DBC", front_month_price=25.0, deferred_month_price=24.0,
            regime=CurveRegime.BACKWARDATION, spread_pct=4.0,
            timestamp=datetime(2026, 5, 14)
        )
        result = compute_commodity_allocation({"DBC": signal}, max_weight=5.0)
        assert result["dbc_weight"] == pytest.approx(5.0)
        assert result["allocation_allowed"] is True

    def test_compute_allocation_includes_signal_details(self):
        from src.signals.commodity_curve import (
            CurveRegime, CommodityCurveSignal, compute_commodity_allocation
        )
        signal = CommodityCurveSignal(
            ticker="DBC", front_month_price=25.0, deferred_month_price=24.0,
            regime=CurveRegime.BACKWARDATION, spread_pct=4.0,
            timestamp=datetime(2026, 5, 14)
        )
        result = compute_commodity_allocation({"DBC": signal})
        assert "signals" in result
        assert result["signals"]["DBC"]["regime"] == "BACKWARDATION"
        assert result["signals"]["DBC"]["spread_pct"] == 4.0


class TestCLI:
    """CLI interface tests."""

    def test_cli_fetch_command(self, monkeypatch, capsys, tmp_path):
        import sys, json

        mock_prices = {
            "DBC": [
                {"d": "2026-05-14", "p": 25.0},
                {"d": "2026-04-23", "p": 24.0},
            ]
        }
        p = tmp_path / "prices.json"
        p.write_text(json.dumps(mock_prices))

        test_args = ["commodity_curve", "fetch", "--ticker", "DBC", "--prices", str(p)]
        monkeypatch.setattr(sys, 'argv', test_args)

        from src.signals.commodity_curve import main
        try:
            main()
        except SystemExit:
            pass

        captured = capsys.readouterr()
        assert "DBC" in captured.out

    def test_cli_status_command(self, monkeypatch, capsys, tmp_path):
        import sys, json

        mock_prices = {
            "DBC": [
                {"d": "2026-05-14", "p": 25.0},
                {"d": "2026-04-23", "p": 24.0},
            ],
            "USO": [
                {"d": "2026-05-14", "p": 70.0},
                {"d": "2026-04-23", "p": 72.0},
            ],
        }
        p = tmp_path / "prices.json"
        p.write_text(json.dumps(mock_prices))

        test_args = ["commodity_curve", "status", "--prices", str(p)]
        monkeypatch.setattr(sys, 'argv', test_args)

        from src.signals.commodity_curve import main
        try:
            main()
        except SystemExit:
            pass

        captured = capsys.readouterr()
        assert "Commodity Curve Status" in captured.out

    def test_cli_regime_command(self, monkeypatch, capsys, tmp_path):
        import sys, json

        mock_prices = {
            "DBC": [
                {"d": "2026-05-14", "p": 25.0},
                {"d": "2026-04-23", "p": 24.0},
            ]
        }
        p = tmp_path / "prices.json"
        p.write_text(json.dumps(mock_prices))

        test_args = ["commodity_curve", "regime", "--ticker", "DBC", "--prices", str(p)]
        monkeypatch.setattr(sys, 'argv', test_args)

        from src.signals.commodity_curve import main
        try:
            main()
        except SystemExit:
            pass

        captured = capsys.readouterr()
        assert "DBC" in captured.out


class TestIntegration:
    """Integration tests using real project prices.json."""

    def test_fetch_dbc_from_real_data(self):
        from src.signals.commodity_curve import fetch_curve_signal, PRICES_PATH
        if not PRICES_PATH.exists():
            pytest.skip("prices.json not available")

        import json
        with open(PRICES_PATH) as f:
            data = json.load(f)
        if "DBC" not in data:
            pytest.skip("DBC not in prices.json")

        signal = fetch_curve_signal("DBC")
        assert signal.ticker == "DBC"
        assert signal.front_month_price > 0
        assert signal.deferred_month_price > 0
        assert isinstance(signal.spread_pct, float)

    def test_fetch_all_from_real_data(self):
        from src.signals.commodity_curve import fetch_all_curves, get_curve_summary, PRICES_PATH
        if not PRICES_PATH.exists():
            pytest.skip("prices.json not available")

        import json
        with open(PRICES_PATH) as f:
            data = json.load(f)
        available = [t for t in ["DBC", "GSG", "USO"] if t in data]
        if not available:
            pytest.skip("No commodity ETFs in prices.json")

        results = fetch_all_curves(tickers=available)
        assert len(results) > 0, f"Should find at least one commodity ETF from {available}"

        summary = get_curve_summary(results)
        assert summary["total"] == len(results)
        assert summary["backwardation"] + summary["contango"] + summary["flat"] == len(results)

    def test_compute_allocation_from_real_data(self):
        from src.signals.commodity_curve import (
            fetch_all_curves, compute_commodity_allocation, PRICES_PATH
        )
        if not PRICES_PATH.exists():
            pytest.skip("prices.json not available")

        import json
        with open(PRICES_PATH) as f:
            data = json.load(f)
        available = [t for t in ["DBC", "GSG", "USO"] if t in data]
        if not available:
            pytest.skip("No commodity ETFs in prices.json")

        results = fetch_all_curves(tickers=available)
        alloc = compute_commodity_allocation(results)

        assert "dbc_weight" in alloc
        assert "allocation_allowed" in alloc
        assert isinstance(alloc["dbc_weight"], float)
        assert 0.0 <= alloc["dbc_weight"] <= 5.0
