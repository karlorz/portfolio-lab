"""Isolated VIX dual-threshold controller benchmark.

Compares the live fixed-threshold VIX regime helper against a rolling
dual-threshold controller without changing dashboard or evaluator authority.

Usage:
    python -m src.backtest.vix_dual_threshold_backtest
    python -m src.backtest.vix_dual_threshold_backtest --save
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Mapping, Sequence

import numpy as np

from src.backtest.metrics import compute_metrics, save_results_json
from src.paths import (
    DATA_DIR,
    PRICES_JSON,
    VIX_CRISIS_THRESHOLD,
    VIX_LOW_VOL_THRESHOLD,
    VIX_VOL_SPIKE_THRESHOLD,
)
from src.strategy.regime_allocation import DEFAULT_ALLOCATION, REGIME_ALLOCATIONS
from src.utils import classify_vix_regime

logger = logging.getLogger(__name__)

__all__ = [
    "VixControllerBacktestRow",
    "VixDualThresholdBacktestResult",
    "classify_fixed_threshold_regime",
    "classify_rolling_dual_threshold",
    "run_vix_dual_threshold_backtest",
]

INITIAL_CAPITAL = 100000.0
RISK_SYMBOLS = ("SPY", "GLD", "TLT")
VIX_SYMBOL_CANDIDATES = ("^VIX", "VIX", "^VIX3M")
HISTORICAL_PRICES_JSON = DATA_DIR / "prices.json"
EXPERIMENT_ID = "vix-dual-threshold-controller-benchmark"
ROLLING_LOOKBACK_DAYS = 252
ROLLING_MIN_OBSERVATIONS = 63
ROLLING_LOW_QUANTILE = 0.25
ROLLING_HIGH_QUANTILE = 0.75
MIN_BENCHMARK_OBSERVATIONS = ROLLING_MIN_OBSERVATIONS + 1


@dataclass
class VixControllerBacktestRow:
    """Single VIX controller benchmark result row."""

    controller: str
    label: str
    cagr: float
    vol: float
    sharpe: float
    max_dd: float
    total_return: float
    sortino: float
    calmar: float
    regime_counts: dict[str, int]
    sharpe_delta: float


@dataclass
class VixDualThresholdBacktestResult:
    """Complete fixed-vs-rolling VIX controller benchmark result."""

    experiment_id: str
    timestamp: str
    data_range: str
    n_days: int
    vix_symbol: str
    price_source: str
    live_controller_unchanged: bool
    fixed_thresholds: dict[str, float]
    rolling_config: dict[str, float | int]
    rows: list[dict]
    best_sharpe_row: dict | None
    recommendation: str


def classify_fixed_threshold_regime(vix_level: float | None, trend_regime: str = "normal") -> str:
    """Classify using the existing live fixed-threshold helper."""

    return classify_vix_regime(vix_level, trend_regime)


def classify_rolling_dual_threshold(
    vix_level: float | None,
    vix_history: Sequence[float],
    trend_regime: str = "normal",
    *,
    lookback_days: int = ROLLING_LOOKBACK_DAYS,
    min_observations: int = ROLLING_MIN_OBSERVATIONS,
    low_quantile: float = ROLLING_LOW_QUANTILE,
    high_quantile: float = ROLLING_HIGH_QUANTILE,
) -> str:
    """Classify VIX regime with rolling low/high quantile thresholds.

    Short histories deliberately fall back to the current fixed-threshold helper
    so the benchmark has an explicit warm-up behavior and does not invent early
    dynamic thresholds from insufficient data.
    """

    if vix_level is None:
        return trend_regime

    history = np.asarray(vix_history, dtype=float)
    history = history[np.isfinite(history)]
    if len(history) < min_observations:
        return classify_fixed_threshold_regime(vix_level, trend_regime)

    recent = history[-lookback_days:]
    low_threshold = float(np.quantile(recent, low_quantile))
    high_threshold = float(np.quantile(recent, high_quantile))

    if vix_level > high_threshold:
        return "vol_spike"
    if vix_level < low_threshold and trend_regime != "crisis":
        return "low_vol"
    return trend_regime


def run_vix_dual_threshold_backtest(
    *,
    price_records: Mapping[str, Sequence[Mapping[str, float | str]]] | None = None,
    save: bool = False,
) -> VixDualThresholdBacktestResult:
    """Run the isolated fixed-vs-rolling VIX controller benchmark."""

    if price_records is None:
        records, price_source = _load_price_records()
    else:
        records = price_records
        price_source = "in_memory"
    dates, series, vix_symbol = _align_price_records(records)
    rows = _run_controller_comparison(series)
    fixed_sharpe = rows[0].sharpe if rows else 0.0
    result_rows = [
        asdict(row) | {"sharpe_delta": round(row.sharpe - fixed_sharpe, 4)}
        for row in rows
    ]
    best_row = max(result_rows, key=lambda row: row["sharpe"], default=None)

    result = VixDualThresholdBacktestResult(
        experiment_id=EXPERIMENT_ID,
        timestamp=datetime.now(UTC).isoformat(),
        data_range=f"{dates[0]} to {dates[-1]}" if dates else "",
        n_days=len(dates),
        vix_symbol=vix_symbol,
        price_source=price_source,
        live_controller_unchanged=True,
        fixed_thresholds={
            "crisis": float(VIX_CRISIS_THRESHOLD),
            "vol_spike": float(VIX_VOL_SPIKE_THRESHOLD),
            "low_vol": float(VIX_LOW_VOL_THRESHOLD),
        },
        rolling_config={
            "lookback_days": ROLLING_LOOKBACK_DAYS,
            "min_observations": ROLLING_MIN_OBSERVATIONS,
            "low_quantile": ROLLING_LOW_QUANTILE,
            "high_quantile": ROLLING_HIGH_QUANTILE,
        },
        rows=result_rows,
        best_sharpe_row=best_row,
        recommendation=_build_recommendation(result_rows),
    )

    if save:
        output_path = DATA_DIR / "vix_dual_threshold_backtest_results.json"
        result_payload = asdict(result)
        save_results_json(
            result_payload,
            output_path=str(output_path),
            experiment_manifest={
                "experiment_id": EXPERIMENT_ID,
                "manifest_mode": "sidecar",
                "module": __name__,
                "command": "python -m src.backtest.vix_dual_threshold_backtest --save",
                "config_snapshot": {
                    **result_payload["rolling_config"],
                    "vix_symbol": vix_symbol,
                    "price_source": price_source,
                    "scope": "benchmark_only_no_live_authority_change",
                },
                "input_paths": [
                    price_source if price_source != "in_memory" else str(PRICES_JSON),
                ],
            },
        )
        logger.info("Saved VIX dual-threshold benchmark to %s", output_path)

    return result


def _load_price_records() -> tuple[dict[str, list[dict]], str]:
    candidates = [PRICES_JSON]
    if HISTORICAL_PRICES_JSON != PRICES_JSON:
        candidates.append(HISTORICAL_PRICES_JSON)

    fallback: tuple[dict[str, list[dict]], str] | None = None
    for path in candidates:
        if not path.exists():
            continue
        with open(path) as f:
            records = json.load(f)
        dates, _, _ = _align_price_records(records)
        if fallback is None:
            fallback = (records, str(path))
        if len(dates) >= MIN_BENCHMARK_OBSERVATIONS:
            return records, str(path)

    if fallback is not None:
        return fallback
    return {}, str(PRICES_JSON)


def _align_price_records(
    price_records: Mapping[str, Sequence[Mapping[str, float | str]]],
) -> tuple[list[str], dict[str, np.ndarray], str]:
    vix_symbol = _select_vix_symbol(price_records)
    required_symbols = (*RISK_SYMBOLS, vix_symbol)
    indexed: dict[str, dict[str, float]] = {}
    for symbol in required_symbols:
        symbol_records = price_records.get(symbol) or []
        indexed[symbol] = {
            str(record["d"]): float(record["p"])
            for record in symbol_records
            if "d" in record and "p" in record
        }

    common_dates = set(indexed[RISK_SYMBOLS[0]])
    for symbol in required_symbols[1:]:
        common_dates &= set(indexed[symbol])
    dates = sorted(common_dates)

    series = {
        symbol: np.array([indexed[symbol][date] for date in dates], dtype=float)
        for symbol in RISK_SYMBOLS
    }
    series["VIX"] = np.array([indexed[vix_symbol][date] for date in dates], dtype=float)
    return dates, series, vix_symbol


def _select_vix_symbol(
    price_records: Mapping[str, Sequence[Mapping[str, float | str]]],
) -> str:
    for symbol in VIX_SYMBOL_CANDIDATES:
        if price_records.get(symbol):
            return symbol
    return VIX_SYMBOL_CANDIDATES[0]


def _run_controller_comparison(series: Mapping[str, np.ndarray]) -> list[VixControllerBacktestRow]:
    controllers = [
        (
            "fixed_threshold",
            "Current fixed-threshold VIX controller",
            lambda idx: classify_fixed_threshold_regime(float(series["VIX"][idx]), "normal"),
        ),
        (
            "rolling_dual_threshold",
            "Rolling dual-threshold VIX controller",
            lambda idx: classify_rolling_dual_threshold(
                float(series["VIX"][idx]),
                series["VIX"][:idx],
                "normal",
            ),
        ),
    ]

    rows = []
    for controller, label, classifier in controllers:
        result = _backtest_controller(series, classifier)
        rows.append(
            VixControllerBacktestRow(
                controller=controller,
                label=label,
                cagr=_as_float(result.get("cagr", 0.0)),
                vol=_as_float(result.get("vol", 0.0)),
                sharpe=_as_float(result.get("sharpe", 0.0)),
                max_dd=_as_float(result.get("max_dd", 0.0)),
                total_return=_as_float(result.get("total_return", 0.0)),
                sortino=_as_float(result.get("sortino", 0.0)),
                calmar=_as_float(result.get("calmar", 0.0)),
                regime_counts=result.get("regime_counts", {}),
                sharpe_delta=0.0,
            )
        )

    return rows


def _as_float(value: object) -> float:
    return float(value) if value is not None else 0.0


def _backtest_controller(
    series: Mapping[str, np.ndarray],
    classify_for_index,
) -> dict[str, float | dict[str, int]]:
    spy = series.get("SPY", np.array([]))
    gld = series.get("GLD", np.array([]))
    tlt = series.get("TLT", np.array([]))
    n_days = min(len(spy), len(gld), len(tlt), len(series.get("VIX", np.array([]))))
    if n_days < 2:
        return _empty_result()

    spy_ret = spy[1:n_days] / spy[: n_days - 1] - 1
    gld_ret = gld[1:n_days] / gld[: n_days - 1] - 1
    tlt_ret = tlt[1:n_days] / tlt[: n_days - 1] - 1

    equity = INITIAL_CAPITAL
    equity_curve = [equity]
    regime_counts: dict[str, int] = {}

    for offset in range(len(spy_ret)):
        idx = offset + 1
        raw_regime = str(classify_for_index(idx))
        regime_counts[raw_regime] = regime_counts.get(raw_regime, 0) + 1
        alloc = _allocation_for_regime(raw_regime)
        port_ret = (
            alloc["SPY"] * spy_ret[offset]
            + alloc["GLD"] * gld_ret[offset]
            + alloc["TLT"] * tlt_ret[offset]
        )
        equity *= 1.0 + float(port_ret)
        equity_curve.append(equity)

    metrics = compute_metrics(equity_curve, INITIAL_CAPITAL)
    return {
        "cagr": metrics.cagr,
        "vol": metrics.volatility,
        "sharpe": metrics.sharpe_ratio,
        "max_dd": metrics.max_drawdown,
        "total_return": metrics.total_return,
        "sortino": metrics.sortino_ratio,
        "calmar": metrics.calmar_ratio,
        "regime_counts": regime_counts,
    }


def _allocation_for_regime(raw_regime: str) -> dict[str, float]:
    allocation_regime = "high_vol" if raw_regime == "vol_spike" else raw_regime
    allocation = REGIME_ALLOCATIONS.get(allocation_regime, DEFAULT_ALLOCATION)
    return _normalize_allocation(allocation)


def _normalize_allocation(allocation: Mapping[str, float]) -> dict[str, float]:
    weights = {
        "SPY": float(allocation.get("SPY", DEFAULT_ALLOCATION["SPY"])),
        "GLD": float(allocation.get("GLD", DEFAULT_ALLOCATION["GLD"])),
        "TLT": float(allocation.get("TLT", DEFAULT_ALLOCATION["TLT"])),
    }
    total = sum(weights.values())
    if total <= 0:
        return dict(DEFAULT_ALLOCATION)
    return {symbol: weight / total for symbol, weight in weights.items()}


def _empty_result() -> dict[str, float | dict[str, int]]:
    return {
        "cagr": 0.0,
        "vol": 0.0,
        "sharpe": 0.0,
        "max_dd": 0.0,
        "total_return": 0.0,
        "sortino": 0.0,
        "calmar": 0.0,
        "regime_counts": {},
    }


def _build_recommendation(rows: Sequence[dict]) -> str:
    if not rows:
        return "No benchmark rows produced; check price and VIX data coverage."
    fixed = next((row for row in rows if row["controller"] == "fixed_threshold"), None)
    rolling = next((row for row in rows if row["controller"] == "rolling_dual_threshold"), None)
    if not fixed or not rolling:
        return "Benchmark incomplete; expected fixed and rolling controller rows."
    delta = float(rolling["sharpe"]) - float(fixed["sharpe"])
    if delta > 0:
        return (
            "Rolling dual-threshold controller outperformed in this isolated "
            "benchmark; keep it shadow-only until a separate promotion decision."
        )
    return (
        "Fixed-threshold controller remains competitive in this isolated "
        "benchmark; no live authority change is recommended."
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="VIX Dual-Threshold Controller Benchmark")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    args = parser.parse_args()
    result = run_vix_dual_threshold_backtest(save=args.save)
    logger.info("Best controller: %s", result.best_sharpe_row)
