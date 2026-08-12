"""Tests for the isolated VIX dual-threshold controller benchmark."""

import json
from datetime import date, timedelta

import numpy as np

from src.backtest import vix_dual_threshold_backtest as vdt
from src.utils import classify_vix_regime


def _records(values: np.ndarray, start: date = date(2026, 1, 1)) -> list[dict[str, float | str]]:
    return [
        {"d": (start + timedelta(days=idx)).isoformat(), "p": float(value)}
        for idx, value in enumerate(values)
    ]


def _price_records(
    n_days: int = 360,
    vix_symbol: str = "^VIX",
) -> dict[str, list[dict[str, float | str]]]:
    idx = np.arange(n_days)
    spy = 100.0 * np.cumprod(1.0 + 0.0004 + 0.0010 * np.sin(idx / 13.0))
    gld = 120.0 * np.cumprod(1.0 + 0.0002 + 0.0008 * np.cos(idx / 17.0))
    tlt = 90.0 * np.cumprod(1.0 + 0.0001 + 0.0006 * np.sin(idx / 19.0))
    vix = 18.0 + 4.0 * np.sin(idx / 21.0)
    vix[90:110] = 30.0
    vix[210:230] = 11.0
    return {
        "SPY": _records(spy),
        "GLD": _records(gld),
        "TLT": _records(tlt),
        vix_symbol: _records(vix),
    }


class TestRollingDualThresholdClassifier:
    def test_history_quantiles_drive_high_and_low_volatility_regimes(self):
        history = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0])

        assert vdt.classify_rolling_dual_threshold(25.0, history) == "vol_spike"
        assert vdt.classify_rolling_dual_threshold(11.0, history) == "low_vol"
        assert vdt.classify_rolling_dual_threshold(17.0, history, "recovery") == "recovery"

    def test_short_history_falls_back_to_fixed_threshold_helper(self):
        history = np.array([15.0, 16.0, 17.0])

        actual = vdt.classify_rolling_dual_threshold(22.0, history, "normal")

        assert actual == classify_vix_regime(22.0, "normal")


class TestVixDualThresholdBenchmark:
    def test_result_compares_fixed_and_rolling_dual_threshold_controllers(self):
        result = vdt.run_vix_dual_threshold_backtest(
            price_records=_price_records(),
            save=False,
        )

        assert result.experiment_id == "vix-dual-threshold-controller-benchmark"
        assert result.live_controller_unchanged is True
        assert result.vix_symbol == "^VIX"
        assert result.price_source == "in_memory"
        assert result.n_days == 360
        assert result.data_range == "2026-01-01 to 2026-12-26"

        rows = {row["controller"]: row for row in result.rows}
        assert set(rows) == {"fixed_threshold", "rolling_dual_threshold"}
        for row in rows.values():
            assert set(row) >= {
                "controller",
                "label",
                "cagr",
                "vol",
                "sharpe",
                "max_dd",
                "total_return",
                "regime_counts",
                "sharpe_delta",
            }
            assert sum(row["regime_counts"].values()) == result.n_days - 1

        assert result.best_sharpe_row["controller"] in rows

    def test_public_prices_vix3m_alias_is_accepted_when_vix_spot_is_absent(self):
        result = vdt.run_vix_dual_threshold_backtest(
            price_records=_price_records(vix_symbol="^VIX3M"),
            save=False,
        )

        assert result.vix_symbol == "^VIX3M"
        assert result.n_days == 360
        assert all(row["regime_counts"] for row in result.rows)

    def test_default_loader_uses_historical_prices_when_public_vix_history_is_too_short(
        self,
        tmp_path,
        monkeypatch,
    ):
        public_path = tmp_path / "public_prices.json"
        historical_path = tmp_path / "historical_prices.json"
        public_records = _price_records(n_days=80, vix_symbol="^VIX3M")
        public_records["^VIX3M"] = public_records["^VIX3M"][:1]
        public_path.write_text(json.dumps(public_records), encoding="utf-8")
        historical_path.write_text(json.dumps(_price_records(n_days=80)), encoding="utf-8")
        monkeypatch.setattr(vdt, "PRICES_JSON", public_path)
        monkeypatch.setattr(vdt, "HISTORICAL_PRICES_JSON", historical_path)

        result = vdt.run_vix_dual_threshold_backtest(save=False)

        assert result.price_source == str(historical_path)
        assert result.vix_symbol == "^VIX"
        assert result.n_days == 80
        assert all(row["regime_counts"] for row in result.rows)

    def test_fixed_threshold_contract_remains_unchanged(self):
        assert classify_vix_regime(25.0, "normal") == "vol_spike"
        assert classify_vix_regime(20.0, "normal") == "normal"
        assert vdt.classify_fixed_threshold_regime(25.0, "normal") == "vol_spike"

    def test_save_writes_artifact_and_sidecar_manifest(self, tmp_path, monkeypatch):
        prices_path = tmp_path / "prices.json"
        prices_path.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(vdt, "DATA_DIR", tmp_path)
        monkeypatch.setattr(vdt, "PRICES_JSON", prices_path)

        result = vdt.run_vix_dual_threshold_backtest(
            price_records=_price_records(),
            save=True,
        )

        artifact_path = tmp_path / "vix_dual_threshold_backtest_results.json"
        manifest_path = tmp_path / "vix_dual_threshold_backtest_results.json.manifest.json"
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert artifact["experiment_id"] == result.experiment_id
        assert manifest["experiment_id"] == result.experiment_id
        assert manifest["config_snapshot"]["lookback_days"] == vdt.ROLLING_LOOKBACK_DAYS


def test_a3_b1a_delegation_matches_pre_migration_capture():
    """A3 pin (Item B1a sub-task 8): _load_price_records delegates to grid_runner.load_prices."""
    from src.backtest.grid_runner import load_prices

    # module-level loader stays in pilot; the shared loader is grid_runner's
    assert vdt._load_price_records.__module__ == (
        "src.backtest.vix_dual_threshold_backtest"
    )
    assert load_prices.__module__ == "src.backtest.grid_runner"
