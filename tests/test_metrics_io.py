#!/usr/bin/env python3
"""
Direct unit tests for ``src/utils/metrics_io.py`` (Items A2+A3 layer-leak
fix, 2026-08-12) — test file owed by the TEST-GAP coverage gap (module has
zero direct test references; indirect coverage via test_generator only).

Covers the public surface: ``compute_metrics_from_returns`` (known-returns
fixture vs independent numpy recomputation, rel=1e-9), ``save_results_json``
(tmp_path writes + JSON shape, validator, data_snapshot, chmod 0644),
``_PUBLIC_JSON_MODE``, and the off-path ``_maybe_record_backtest_experiment``
no-ops. The public-output projection branch (real public tree needed) stays
covered by test_generator / delivery-contract suites.
"""
import json
import os

import numpy as np
import pytest

from src.utils.metrics_io import (
    _PUBLIC_JSON_MODE,
    _maybe_record_backtest_experiment,
    compute_metrics_from_returns,
    save_results_json,
)


def _expected_metrics(returns, risk_free_rate, trading_days_per_year=252):
    """Independent recomputation mirroring the documented formulas."""
    arr = np.array(returns, dtype=float)
    n = len(arr)
    if n == 0:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
        }
    total_return = float(np.prod(1 + arr) - 1)
    years = n / trading_days_per_year
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    daily_vol = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    annualized_vol = daily_vol * np.sqrt(trading_days_per_year)
    sharpe = (cagr - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0.0
    cumulative = np.cumprod(1 + arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_dd = float(np.min(drawdown))
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0
    return {
        "total_return": round(total_return, 6),
        "cagr": round(cagr, 6),
        "volatility": round(annualized_vol, 6),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 6),
        "calmar": round(calmar, 4),
    }


def test_compute_metrics_known_fixture():
    """Core metrics match an independent recomputation (rel=1e-9)."""
    returns = [0.01, -0.005, 0.02, 0.0, 0.015]
    expected = _expected_metrics(returns, 0.0)
    result = compute_metrics_from_returns(returns, risk_free_rate=0.0)
    for key, value in expected.items():
        assert result[key] == pytest.approx(value, rel=1e-9), key


def test_compute_metrics_risk_free_override():
    """Higher risk-free rate lowers the Sharpe (explicit override)."""
    returns = [0.01, 0.005, -0.002, 0.008]
    low = compute_metrics_from_returns(returns, risk_free_rate=0.0)
    high = compute_metrics_from_returns(returns, risk_free_rate=0.05)
    assert high["sharpe"] < low["sharpe"]
    assert high["sharpe"] == pytest.approx(
        _expected_metrics(returns, 0.05)["sharpe"], rel=1e-9
    )


def test_compute_metrics_empty_returns():
    """Empty series → all-zero metrics."""
    assert compute_metrics_from_returns([], risk_free_rate=0.0) == {
        "total_return": 0.0,
        "cagr": 0.0,
        "volatility": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "calmar": 0.0,
    }


def test_compute_metrics_single_return_zero_vol():
    """Single observation → zero volatility, zero sharpe/calmar."""
    result = compute_metrics_from_returns([0.01], risk_free_rate=0.0)
    assert result["volatility"] == 0.0
    assert result["sharpe"] == 0.0
    assert result["calmar"] == 0.0
    assert result["total_return"] == 0.01


def test_compute_metrics_constant_returns_zero_vol():
    """Zero-variance series → sharpe 0, no division blowup."""
    # 0.0 is exactly representable → np.std(ddof=1) is exactly 0.0.
    result = compute_metrics_from_returns([0.0] * 10, risk_free_rate=0.0)
    assert result["volatility"] == 0.0
    assert result["sharpe"] == 0.0
    assert result["max_drawdown"] == 0.0
    assert result["total_return"] == 0.0


def test_save_results_json_output_path(tmp_path):
    """Explicit output_path → JSON round-trip + world-readable mode."""
    out = tmp_path / "out.json"
    data = {"total_return": 0.05, "weights": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}}
    save_results_json(data, output_path=str(out))
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8")) == data
    assert os.stat(out).st_mode & 0o777 == 0o644


def test_save_results_json_default_dir(tmp_path):
    """default_dir → auto-named backtest_results.json under it."""
    default_dir = tmp_path / "nested" / "dir"
    save_results_json({"a": 1}, default_dir=default_dir)
    path = default_dir / "backtest_results.json"
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}


def test_save_results_json_no_path_noop():
    """Neither output_path nor default_dir → silent no-op."""
    assert save_results_json({"a": 1}) is None


def test_save_results_json_validator(tmp_path):
    """Validator transforms data before serialization."""
    out = tmp_path / "validated.json"

    def validator(data):
        data["validated"] = True
        return data

    save_results_json({"a": 1}, output_path=str(out), validator=validator)
    assert json.loads(out.read_text(encoding="utf-8")) == {"a": 1, "validated": True}


def test_save_results_json_validator_failure_falls_back(tmp_path):
    """Raising validator → original data written unchanged."""
    out = tmp_path / "fallback.json"

    def bad_validator(data):
        raise RuntimeError("boom")

    save_results_json({"a": 1}, output_path=str(out), validator=bad_validator)
    assert json.loads(out.read_text(encoding="utf-8")) == {"a": 1}


def test_save_results_json_data_snapshot(tmp_path):
    """data_snapshot embeds under _data_snapshot without touching payload."""
    out = tmp_path / "snapshot.json"
    save_results_json(
        {"a": 1},
        output_path=str(out),
        data_snapshot={"prices_until": "2026-07-06"},
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["a"] == 1
    assert payload["_data_snapshot"] == {"prices_until": "2026-07-06"}


def test_public_json_mode_is_world_readable():
    """Public JSON mode pin — dashboard/Caddy-served files must be 0644."""
    assert _PUBLIC_JSON_MODE == 0o644


def test_maybe_record_backtest_experiment_off_path_noop(tmp_path):
    """No manifest / no experiment_id → no registry side effects."""
    path = tmp_path / "out.json"
    assert _maybe_record_backtest_experiment({}, path, None) is None
    assert _maybe_record_backtest_experiment({}, path, {"name": "x"}) is None
