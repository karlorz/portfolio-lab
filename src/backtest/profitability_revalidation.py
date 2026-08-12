"""Deterministic, fail-closed strategy profitability revalidation.

The protocol in this module is deliberately frozen. It compares only the
approved candidates, admits only complete real point-in-time data, and leaves
all results advisory. Generated result JSON belongs in the ignored runtime
results directory and is never a live allocation input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from contextlib import closing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd

from src.backtest.metrics import (
    build_profitability_evidence,
    compute_one_way_turnover,
)
from src.paths import (
    BACKTEST_RESULTS_DIR,
    BASE_ALLOCATION,
    MARKET_DB,
    PRICES_JSON,
    sqlite_connect,
)
from src.signals.tsmom_overlay import (
    LOOKBACK_DAYS,
    REBALANCE_FREQ,
    SKIP_DAYS,
    TSMOMBacktester,
    VOL_WINDOW,
)
from src.strategy.factor_rotation import FactorMomentumEngine

RESULT_SCHEMA_VERSION = "strategy-profitability-revalidation/v1"
CANDIDATE_IDS = (
    "champion_46_38_16",
    "tsmom",
    "factor_rotation",
    "combined_overlay",
    "unified_overlay",
)
CORE_ASSETS = ("SPY", "GLD", "TLT")
COMBINED_ASSETS = ("SPY", "GLD", "TLT", "IEF", "SHY", "BTC", "ETH", "VIX")


@dataclass(frozen=True)
class FrozenProtocol:
    """Pre-registered dates, costs, and parameters for the approved comparison."""

    snapshot_sha256: str = (
        "8ce67ffe7df3d182f139076cecb17c94949130b71752c3a9df0d005f420a9cd1"
    )
    raw_start: str = "2005-01-03"
    raw_end: str = "2026-07-24"
    raw_observations: int = 5423
    evaluation_start: str = "2006-02-02"
    evaluation_end: str = "2026-07-24"
    evaluation_observations: int = 5150
    development_end: str = "2020-05-28"
    holdout_start: str = "2020-05-29"
    development_observations: int = 3604
    holdout_observations: int = 1546
    development_fraction: float = 0.70
    transaction_cost_bps: float = 10.0
    stress_transaction_cost_bps: float = 20.0
    champion_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(BASE_ALLOCATION)
    )
    crisis_windows: Mapping[str, tuple[str, str]] = field(
        default_factory=lambda: {
            "global_financial_crisis": ("2007-10-09", "2009-03-09"),
            "covid_crash": ("2020-02-19", "2020-03-23"),
            "inflation_rate_shock": ("2022-01-03", "2022-12-30"),
        }
    )


FROZEN_PROTOCOL = FrozenProtocol()


def inventory_candidates(
    price_symbols: Iterable[str],
    database_symbols: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Return deterministic eligibility decisions without running a backtest."""
    prices = set(price_symbols)
    database = set(database_symbols)
    normalized_database = database | ({"VIX"} if "^VIX" in database else set())

    core_missing = sorted(set(CORE_ASSETS) - prices)
    factor_assets = tuple(FactorMomentumEngine.FACTORS)
    factor_missing = sorted(set(factor_assets) - database)
    combined_missing = sorted(set(COMBINED_ASSETS) - normalized_database)

    return {
        "champion_46_38_16": {
            "eligible": not core_missing,
            "data_mode": "real",
            "point_in_time": True,
            "missing_assets": core_missing,
            "reason": (
                "complete real SPY/GLD/TLT history"
                if not core_missing
                else "missing required real core assets"
            ),
        },
        "tsmom": {
            "eligible": not core_missing,
            "data_mode": "real",
            "point_in_time": True,
            "missing_assets": core_missing,
            "reason": (
                "complete real SPY/GLD/TLT history"
                if not core_missing
                else "missing required real TSMOM assets"
            ),
        },
        "factor_rotation": {
            "eligible": not factor_missing,
            "data_mode": "real",
            "point_in_time": True,
            "missing_assets": factor_missing,
            "reason": (
                "complete real factor universe"
                if not factor_missing
                else "missing required real factor assets"
            ),
        },
        "combined_overlay": {
            "eligible": False,
            "data_mode": "proxy",
            "point_in_time": False,
            "missing_assets": combined_missing,
            "reason": (
                "implementation uses yield-spread proxy signals"
                + (
                    f"; missing required real assets: {', '.join(combined_missing)}"
                    if combined_missing
                    else ""
                )
            ),
        },
        "unified_overlay": {
            "eligible": False,
            "data_mode": "proxy",
            "point_in_time": False,
            "missing_assets": [],
            "reason": (
                "implementation derives crypto returns and optional VIX inputs "
                "from proxies"
            ),
        },
    }


def freeze_price_index(
    payload: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    expected_dates: int | None = None,
) -> dict[str, Any]:
    """Normalize complete core prices and reject any date mismatch or bad value."""
    normalized: dict[str, dict[str, float]] = {}
    for symbol in CORE_ASSETS:
        rows = payload.get(symbol)
        if not rows:
            raise ValueError(f"missing required core price history: {symbol}")
        by_date: dict[str, float] = {}
        for row in rows:
            date = str(row.get("d", ""))
            raw_price = row.get("p")
            if not date or date in by_date:
                raise ValueError(f"{symbol} contains missing or duplicate dates")
            price = float(raw_price)
            if not math.isfinite(price) or price <= 0:
                raise ValueError(f"{symbol} contains invalid prices")
            by_date[date] = price
        normalized[symbol] = by_date

    reference_dates = sorted(normalized[CORE_ASSETS[0]])
    if expected_dates is not None and len(reference_dates) != expected_dates:
        raise ValueError(
            f"expected {expected_dates} core dates, got {len(reference_dates)}"
        )
    if any(
        sorted(normalized[symbol]) != reference_dates for symbol in CORE_ASSETS[1:]
    ):
        raise ValueError("core assets must have identical complete dates")
    if any(
        reference_dates[index] >= reference_dates[index + 1]
        for index in range(len(reference_dates) - 1)
    ):
        raise ValueError("core price dates must be strictly increasing")

    return {
        "dates": reference_dates,
        "prices": {
            symbol: [normalized[symbol][date] for date in reference_dates]
            for symbol in CORE_ASSETS
        },
    }


def build_champion_trace(
    aligned: Mapping[str, Any],
    *,
    start_index: int,
    weights: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Build the champion trace with drifting weights and scheduled rebalancing."""
    dates = list(aligned["dates"])
    allocation = dict(weights or FROZEN_PROTOCOL.champion_weights)
    if start_index < 1 or start_index >= len(dates):
        raise ValueError("start_index must leave at least one evaluation observation")
    if set(allocation) != set(CORE_ASSETS) or not math.isclose(
        sum(allocation.values()), 1.0, abs_tol=1e-12
    ):
        raise ValueError("champion weights must cover SPY/GLD/TLT and sum to one")

    target_weights = {**allocation, "CASH": 0.0}
    return _build_strategy_trace(
        aligned,
        start_index=start_index,
        initial_weights=target_weights,
        target_weights_for_rebalance=lambda _index: target_weights,
    )


def _drift_weights(
    current_weights: Mapping[str, float],
    asset_returns: Mapping[str, float],
    gross_return: float,
) -> dict[str, float]:
    portfolio_growth = 1.0 + gross_return
    if portfolio_growth <= 0.0:
        raise ValueError("portfolio value must remain positive")
    return {
        symbol: (
            weight * (1.0 + asset_returns.get(symbol, 0.0)) / portfolio_growth
        )
        for symbol, weight in current_weights.items()
    }


def _build_strategy_trace(
    aligned: Mapping[str, Any],
    *,
    start_index: int,
    initial_weights: Mapping[str, float],
    target_weights_for_rebalance: Callable[[int], Mapping[str, float]],
) -> list[dict[str, Any]]:
    dates = list(aligned["dates"])
    prices = aligned["prices"]
    current_weights = dict(initial_weights)
    trace: list[dict[str, Any]] = []
    for evaluation_index, index in enumerate(range(start_index, len(dates))):
        asset_returns = {
            symbol: prices[symbol][index] / prices[symbol][index - 1] - 1.0
            for symbol in CORE_ASSETS
        }
        gross_return = sum(
            current_weights.get(symbol, 0.0) * asset_returns[symbol]
            for symbol in CORE_ASSETS
        )
        current_weights = _drift_weights(
            current_weights,
            asset_returns,
            gross_return,
        )
        scheduled_rebalance = evaluation_index % REBALANCE_FREQ == 0
        turnover = 0.0
        if scheduled_rebalance:
            target_weights = dict(target_weights_for_rebalance(index))
            turnover = compute_one_way_turnover(
                current_weights,
                target_weights,
            )
            current_weights = target_weights
        trace.append(
            {
                "date": dates[index],
                "gross_return": float(gross_return),
                "turnover": float(turnover),
                "scheduled_rebalance": scheduled_rebalance,
            }
        )
    return trace


def build_holdout_windows(
    dates: Sequence[str],
    *,
    count: int = 3,
) -> list[list[str]]:
    """Split dates into contiguous near-equal windows, assigning remainder first."""
    if count <= 0 or len(dates) < count:
        raise ValueError("window count must be positive and no greater than dates")
    size, remainder = divmod(len(dates), count)
    windows: list[list[str]] = []
    offset = 0
    for index in range(count):
        width = size + (1 if index < remainder else 0)
        windows.append(list(dates[offset : offset + width]))
        offset += width
    return windows


def apply_verdict(
    *,
    eligible: bool,
    point_in_time: bool,
    holdout_sharpe_delta: float,
    holdout_excess_cagr: float,
    holdout_max_drawdown: float,
    champion_holdout_max_drawdown: float,
    stress_holdout_sharpe_delta: float,
    stress_holdout_excess_cagr: float,
    broad_advantage: bool,
) -> dict[str, Any]:
    """Apply the approved advisory thresholds exactly."""
    checks = {
        "eligible": bool(eligible),
        "point_in_time": bool(point_in_time),
        "holdout_sharpe_delta": bool(holdout_sharpe_delta >= 0.05),
        "holdout_excess_cagr": bool(holdout_excess_cagr > 0.0),
        "holdout_max_drawdown": bool(
            holdout_max_drawdown >= champion_holdout_max_drawdown - 3.0
        ),
        "stress_holdout_sharpe_delta": bool(
            stress_holdout_sharpe_delta > 0.0
        ),
        "stress_holdout_excess_cagr": bool(stress_holdout_excess_cagr > 0.0),
        "broad_advantage": bool(broad_advantage),
    }
    if all(checks.values()):
        verdict = "promote-for-human-review"
    elif not eligible or not point_in_time:
        verdict = "reject"
    elif holdout_sharpe_delta > 0.0 or holdout_excess_cagr > 0.0:
        verdict = "hold"
    else:
        verdict = "reject"
    return {"verdict": verdict, "checks": checks}


def write_results(payload: Mapping[str, Any], output_path: Path) -> None:
    """Write stable JSON to an explicit ignored/runtime destination."""
    def normalize_scalar(value: Any) -> Any:
        if hasattr(value, "item"):
            return value.item()
        raise TypeError(f"cannot serialize {type(value).__name__}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=normalize_scalar,
        )
        + "\n",
        encoding="utf-8",
    )


def _database_symbols(path: Path) -> set[str]:
    """Return the set of distinct symbols present in the prices table.

    B1b document-only (Item B1b sub-task 5): a symbol-set query
    (``SELECT DISTINCT symbol FROM prices``), NOT a price loader — no
    delegation to ``grid_runner.load_prices_market_db`` is possible.
    """
    if not path.exists():
        return set()
    with closing(sqlite_connect(path)) as connection:
        return {
            str(row[0])
            for row in connection.execute("SELECT DISTINCT symbol FROM prices")
        }


def _trace_subset(
    trace: Sequence[Mapping[str, Any]],
    start: str,
    end: str,
) -> list[dict[str, Any]]:
    subset = [dict(row) for row in trace if start <= str(row["date"]) <= end]
    if not subset:
        raise ValueError(f"no trace observations in requested window {start}..{end}")
    return subset


def _build_evidence(
    trace: Sequence[Mapping[str, Any]],
    *,
    assets: Sequence[str],
    provenance: Mapping[str, Any],
    transaction_cost_bps: float,
) -> dict[str, Any]:
    return build_profitability_evidence(
        dates=[str(row["date"]) for row in trace],
        gross_returns=[float(row["gross_return"]) for row in trace],
        turnovers=[float(row["turnover"]) for row in trace],
        assets=assets,
        data_mode="real",
        provenance=provenance,
        transaction_cost_bps=transaction_cost_bps,
        point_in_time=True,
        require_real_data=True,
    )


def _evidence_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in evidence.items() if key != "trace"}


def _candidate_evidence(
    trace: Sequence[Mapping[str, Any]],
    *,
    assets: Sequence[str],
    provenance: Mapping[str, Any],
    protocol: FrozenProtocol,
) -> dict[str, Any]:
    holdout_trace = _trace_subset(
        trace, protocol.holdout_start, protocol.evaluation_end
    )
    development_trace = _trace_subset(
        trace, protocol.evaluation_start, protocol.development_end
    )
    windows = build_holdout_windows(
        [str(row["date"]) for row in holdout_trace],
        count=3,
    )

    def build_summary(
        rows: Sequence[Mapping[str, Any]], cost_bps: float
    ) -> dict[str, Any]:
        return _evidence_summary(
            _build_evidence(
                rows,
                assets=assets,
                provenance=provenance,
                transaction_cost_bps=cost_bps,
            )
        )

    standard_full = _build_evidence(
        trace,
        assets=assets,
        provenance=provenance,
        transaction_cost_bps=protocol.transaction_cost_bps,
    )
    stress_full = _build_evidence(
        trace,
        assets=assets,
        provenance=provenance,
        transaction_cost_bps=protocol.stress_transaction_cost_bps,
    )
    return {
        "standard_cost": {
            "full": standard_full,
            "development": build_summary(
                development_trace, protocol.transaction_cost_bps
            ),
            "holdout": build_summary(holdout_trace, protocol.transaction_cost_bps),
            "crises": {
                name: build_summary(
                    _trace_subset(trace, start, end),
                    protocol.transaction_cost_bps,
                )
                for name, (start, end) in protocol.crisis_windows.items()
            },
            "holdout_windows": [
                build_summary(
                    _trace_subset(trace, window[0], window[-1]),
                    protocol.transaction_cost_bps,
                )
                for window in windows
            ],
        },
        "stress_cost": {
            "full": _evidence_summary(stress_full),
            "development": build_summary(
                development_trace, protocol.stress_transaction_cost_bps
            ),
            "holdout": build_summary(
                holdout_trace, protocol.stress_transaction_cost_bps
            ),
        },
    }


def _net_metrics(evidence: Mapping[str, Any]) -> Mapping[str, float]:
    return evidence["metrics"]["net"]


def _comparison(
    candidate: Mapping[str, Any],
    champion: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_holdout = _net_metrics(candidate["standard_cost"]["holdout"])
    champion_holdout = _net_metrics(champion["standard_cost"]["holdout"])
    candidate_stress = _net_metrics(candidate["stress_cost"]["holdout"])
    champion_stress = _net_metrics(champion["stress_cost"]["holdout"])

    window_edges = []
    for candidate_window, champion_window in zip(
        candidate["standard_cost"]["holdout_windows"],
        champion["standard_cost"]["holdout_windows"],
    ):
        candidate_return = _net_metrics(candidate_window)["total_return"]
        champion_return = _net_metrics(champion_window)["total_return"]
        window_edges.append(candidate_return - champion_return)
    positive_windows = sum(edge > 0.0 for edge in window_edges)

    return {
        "holdout_sharpe_delta": (
            candidate_holdout["sharpe_ratio"]
            - champion_holdout["sharpe_ratio"]
        ),
        "holdout_excess_cagr": (
            candidate_holdout["cagr"] - champion_holdout["cagr"]
        ),
        "holdout_max_drawdown": candidate_holdout["max_drawdown"],
        "champion_holdout_max_drawdown": champion_holdout["max_drawdown"],
        "stress_holdout_sharpe_delta": (
            candidate_stress["sharpe_ratio"]
            - champion_stress["sharpe_ratio"]
        ),
        "stress_holdout_excess_cagr": (
            candidate_stress["cagr"] - champion_stress["cagr"]
        ),
        "holdout_window_total_return_edges": window_edges,
        "positive_holdout_windows": positive_windows,
        "broad_advantage": positive_windows >= 2,
    }


def _run_tsmom(aligned: Mapping[str, Any]) -> list[dict[str, Any]]:
    backtester = TSMOMBacktester(
        tickers=list(CORE_ASSETS),
        base_allocation={**FROZEN_PROTOCOL.champion_weights, "CASH": 0.0},
        transaction_cost=FROZEN_PROTOCOL.transaction_cost_bps / 10000.0,
    )
    dates = list(aligned["dates"])
    prices = aligned["prices"]
    prices_frame = pd.DataFrame(
        {symbol: prices[symbol] for symbol in CORE_ASSETS},
        index=pd.to_datetime(dates),
    )
    current_weights = {
        **FROZEN_PROTOCOL.champion_weights,
        "CASH": 0.0,
    }
    start_index = LOOKBACK_DAYS + SKIP_DAYS

    def target_weights_for_rebalance(index: int) -> Mapping[str, float]:
        signals = backtester._compute_signals_at_date(prices_frame, index)
        if not signals:
            raise ValueError(f"TSMOM produced no signals on {dates[index]}")
        return backtester._weights_from_signals(signals)

    return _build_strategy_trace(
        aligned,
        start_index=start_index,
        initial_weights=current_weights,
        target_weights_for_rebalance=target_weights_for_rebalance,
    )


def run_revalidation(
    *,
    prices_path: Path = PRICES_JSON,
    market_db_path: Path = MARKET_DB,
    enforce_frozen_snapshot: bool = True,
) -> dict[str, Any]:
    """Run the frozen real-data comparison and return a deterministic artifact."""
    snapshot_bytes = prices_path.read_bytes()
    snapshot_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
    if (
        enforce_frozen_snapshot
        and snapshot_sha256 != FROZEN_PROTOCOL.snapshot_sha256
    ):
        raise ValueError(
            "price snapshot does not match the pre-registered SHA-256; "
            "freeze a new attended protocol instead of silently changing data"
        )
    payload = json.loads(snapshot_bytes)
    if not isinstance(payload, dict):
        raise ValueError("prices JSON must be a symbol mapping")
    aligned = freeze_price_index(
        payload,
        expected_dates=(
            FROZEN_PROTOCOL.raw_observations if enforce_frozen_snapshot else None
        ),
    )
    price_symbols = set(payload)
    database_symbols = _database_symbols(market_db_path)
    inventory = inventory_candidates(price_symbols, database_symbols)
    if not inventory["champion_46_38_16"]["eligible"]:
        raise ValueError("champion is ineligible; decision comparison cannot run")
    if not inventory["tsmom"]["eligible"]:
        raise ValueError("TSMOM is ineligible; decision comparison cannot run")

    champion_trace = build_champion_trace(
        aligned,
        start_index=LOOKBACK_DAYS + SKIP_DAYS,
    )
    tsmom_trace = _run_tsmom(aligned)
    champion_dates = [str(row["date"]) for row in champion_trace]
    tsmom_dates = [str(row["date"]) for row in tsmom_trace]
    if champion_dates != tsmom_dates:
        raise ValueError("eligible candidate traces do not have identical dates")
    if enforce_frozen_snapshot and (
        champion_dates[0] != FROZEN_PROTOCOL.evaluation_start
        or champion_dates[-1] != FROZEN_PROTOCOL.evaluation_end
        or len(champion_dates) != FROZEN_PROTOCOL.evaluation_observations
    ):
        raise ValueError("candidate trace does not match the frozen interval")

    provenance = {
        "source": "prices JSON",
        "path": str(prices_path),
        "sha256": snapshot_sha256,
        "snapshot_end": aligned["dates"][-1],
    }
    champion_evidence = _candidate_evidence(
        champion_trace,
        assets=CORE_ASSETS,
        provenance=provenance,
        protocol=FROZEN_PROTOCOL,
    )
    tsmom_evidence = _candidate_evidence(
        tsmom_trace,
        assets=CORE_ASSETS,
        provenance=provenance,
        protocol=FROZEN_PROTOCOL,
    )
    tsmom_comparison = _comparison(tsmom_evidence, champion_evidence)
    tsmom_verdict = apply_verdict(
        eligible=True,
        point_in_time=True,
        **{
            key: tsmom_comparison[key]
            for key in (
                "holdout_sharpe_delta",
                "holdout_excess_cagr",
                "holdout_max_drawdown",
                "champion_holdout_max_drawdown",
                "stress_holdout_sharpe_delta",
                "stress_holdout_excess_cagr",
                "broad_advantage",
            )
        },
    )

    candidates: dict[str, Any] = {
        "champion_46_38_16": {
            "eligibility": inventory["champion_46_38_16"],
            "parameters": {
                "weights": dict(FROZEN_PROTOCOL.champion_weights),
                "rebalance_frequency_days": REBALANCE_FREQ,
                "scheduled_rebalance_count": sum(
                    bool(row["scheduled_rebalance"]) for row in champion_trace
                ),
                "missing_observations": 0,
            },
            "evidence": champion_evidence,
            "comparison_to_champion": {
                "holdout_sharpe_delta": 0.0,
                "holdout_excess_cagr": 0.0,
            },
            "verdict": {
                "verdict": "hold",
                "checks": {},
                "reason": "reference champion remains the approved baseline",
            },
        },
        "tsmom": {
            "eligibility": inventory["tsmom"],
            "parameters": {
                "lookback_days": LOOKBACK_DAYS,
                "skip_days": SKIP_DAYS,
                "vol_window": VOL_WINDOW,
                "rebalance_frequency_days": REBALANCE_FREQ,
                "scheduled_rebalance_count": sum(
                    bool(row["scheduled_rebalance"]) for row in tsmom_trace
                ),
                "missing_observations": 0,
            },
            "evidence": tsmom_evidence,
            "comparison_to_champion": tsmom_comparison,
            "verdict": tsmom_verdict,
        },
    }
    for candidate_id in CANDIDATE_IDS[2:]:
        candidate_inventory = inventory[candidate_id]
        candidates[candidate_id] = {
            "eligibility": candidate_inventory,
            "evidence": None,
            "comparison_to_champion": None,
            "verdict": {
                "verdict": "reject",
                "checks": {
                    "eligible": False,
                    "point_in_time": candidate_inventory["point_in_time"],
                },
                "reason": candidate_inventory["reason"],
            },
        }

    ranking = sorted(
        CANDIDATE_IDS,
        key=lambda candidate_id: (
            candidates[candidate_id]["eligibility"]["eligible"],
            (
                _net_metrics(
                    candidates[candidate_id]["evidence"]["standard_cost"]["holdout"]
                )["sharpe_ratio"]
                if candidates[candidate_id]["evidence"]
                else float("-inf")
            ),
        ),
        reverse=True,
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "advisory_only": True,
        "live_authority_unchanged": (
            "signals.json.target_allocations -> src.broker.order_router"
        ),
        "protocol": {
            **asdict(FROZEN_PROTOCOL),
            "candidate_ids": list(CANDIDATE_IDS),
            "factor_assets": list(FactorMomentumEngine.FACTORS),
            "tsmom_warmup_observations": LOOKBACK_DAYS + SKIP_DAYS,
            "holdout_window_rule": (
                "three contiguous equal-observation windows; positive in at "
                "least two"
            ),
        },
        "data_inventory": {
            "price_symbols": sorted(price_symbols),
            "database_symbols": sorted(database_symbols),
            "snapshot_sha256": snapshot_sha256,
            "candidate_eligibility": inventory,
        },
        "candidates": candidates,
        "ranking": ranking,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen portfolio profitability revalidation"
    )
    parser.add_argument("--prices", type=Path, default=PRICES_JSON)
    parser.add_argument("--market-db", type=Path, default=MARKET_DB)
    parser.add_argument(
        "--output",
        type=Path,
        default=BACKTEST_RESULTS_DIR / "profitability_revalidation.json",
    )
    args = parser.parse_args()
    results = run_revalidation(
        prices_path=args.prices,
        market_db_path=args.market_db,
    )
    write_results(results, args.output)


if __name__ == "__main__":
    main()
