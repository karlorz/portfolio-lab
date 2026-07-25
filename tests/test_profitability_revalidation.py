import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.backtest.profitability_revalidation import (
    CANDIDATE_IDS,
    FROZEN_PROTOCOL,
    apply_verdict,
    build_champion_trace,
    build_holdout_windows,
    freeze_price_index,
    inventory_candidates,
    run_revalidation,
    write_results,
)


def _rows(dates, prices):
    return [{"d": date, "p": price} for date, price in zip(dates, prices)]


def _eligible_inputs():
    return {
        "eligible": True,
        "point_in_time": True,
        "holdout_sharpe_delta": 0.06,
        "holdout_excess_cagr": 0.5,
        "holdout_max_drawdown": -12.0,
        "champion_holdout_max_drawdown": -10.0,
        "stress_holdout_sharpe_delta": 0.01,
        "stress_holdout_excess_cagr": 0.1,
        "broad_advantage": True,
    }


def test_frozen_protocol_records_approved_mechanics():
    assert FROZEN_PROTOCOL.evaluation_start == "2006-02-02"
    assert FROZEN_PROTOCOL.evaluation_end == "2026-07-24"
    assert FROZEN_PROTOCOL.holdout_start == "2020-05-29"
    assert FROZEN_PROTOCOL.transaction_cost_bps == 10.0
    assert FROZEN_PROTOCOL.stress_transaction_cost_bps == 20.0
    assert FROZEN_PROTOCOL.champion_weights == {
        "SPY": 0.46,
        "GLD": 0.38,
        "TLT": 0.16,
    }
    assert CANDIDATE_IDS == (
        "champion_46_38_16",
        "tsmom",
        "factor_rotation",
        "combined_overlay",
        "unified_overlay",
    )


def test_inventory_fails_closed_for_missing_or_proxy_candidates():
    price_symbols = {"SPY", "GLD", "TLT", "IEF", "SHY", "^VIX"}
    db_symbols = price_symbols | {"VLUE", "MTUM", "QUAL", "USMV", "VBR", "QQQ"}

    inventory = inventory_candidates(price_symbols, db_symbols)

    assert inventory["champion_46_38_16"]["eligible"] is True
    assert inventory["tsmom"]["eligible"] is True
    assert inventory["factor_rotation"]["eligible"] is False
    assert inventory["factor_rotation"]["missing_assets"] == [
        "IJR",
        "SPHQ",
        "SPLV",
        "VTV",
    ]
    assert inventory["combined_overlay"]["eligible"] is False
    assert inventory["combined_overlay"]["missing_assets"] == ["BTC", "ETH"]
    assert inventory["unified_overlay"]["eligible"] is False
    assert inventory["unified_overlay"]["data_mode"] == "proxy"


def test_freeze_price_index_requires_complete_strictly_aligned_core_history():
    payload = {
        "SPY": _rows(["2024-01-02", "2024-01-03"], [100.0, 101.0]),
        "GLD": _rows(["2024-01-02", "2024-01-03"], [50.0, 51.0]),
        "TLT": _rows(["2024-01-02"], [90.0]),
    }

    with pytest.raises(ValueError, match="identical complete dates"):
        freeze_price_index(payload, expected_dates=2)


def test_freeze_price_index_normalizes_price_payload():
    dates = ["2024-01-02", "2024-01-03"]
    payload = {
        symbol: _rows(dates, values)
        for symbol, values in {
            "SPY": [100.0, 101.0],
            "GLD": [50.0, 51.0],
            "TLT": [90.0, 89.0],
        }.items()
    }

    aligned = freeze_price_index(payload, expected_dates=2)

    assert aligned["dates"] == dates
    assert aligned["prices"]["GLD"] == [50.0, 51.0]


def test_champion_trace_uses_prior_close_and_scheduled_rebalancing():
    aligned = {
        "dates": ["2024-01-02", "2024-01-03", "2024-01-04"],
        "prices": {
            "SPY": [100.0, 110.0, 110.0],
            "GLD": [100.0, 100.0, 110.0],
            "TLT": [100.0, 100.0, 100.0],
        },
    }

    trace = build_champion_trace(aligned, start_index=1)

    assert [row["date"] for row in trace] == ["2024-01-03", "2024-01-04"]
    assert trace[0]["gross_return"] == pytest.approx(0.046)
    assert trace[1]["gross_return"] == pytest.approx(0.038)
    assert trace[0]["scheduled_rebalance"] is True
    assert trace[0]["turnover"] > 0.0
    assert trace[1]["scheduled_rebalance"] is False
    assert trace[1]["turnover"] == 0.0


def test_holdout_windows_are_contiguous_and_cover_every_observation():
    dates = [f"2024-01-{day:02d}" for day in range(1, 11)]

    windows = build_holdout_windows(dates, count=3)

    flattened = [date for window in windows for date in window]
    assert flattened == dates
    assert [len(window) for window in windows] == [4, 3, 3]


def test_verdict_promotes_only_when_every_approved_condition_passes():
    verdict = apply_verdict(**_eligible_inputs())

    assert verdict["verdict"] == "promote-for-human-review"
    assert all(verdict["checks"].values())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("holdout_sharpe_delta", 0.0499),
        ("holdout_excess_cagr", 0.0),
        ("holdout_max_drawdown", -13.01),
        ("stress_holdout_sharpe_delta", 0.0),
        ("stress_holdout_excess_cagr", 0.0),
        ("broad_advantage", False),
    ],
)
def test_verdict_holds_positive_but_insufficient_real_candidate(field, value):
    inputs = _eligible_inputs()
    inputs[field] = value

    verdict = apply_verdict(**inputs)

    assert verdict["verdict"] == "hold"
    assert verdict["checks"][field] is False


def test_verdict_rejects_ineligible_candidate():
    inputs = _eligible_inputs()
    inputs["eligible"] = False

    verdict = apply_verdict(**inputs)

    assert verdict["verdict"] == "reject"
    assert verdict["checks"]["eligible"] is False


def test_verdict_rejects_non_point_in_time_candidate():
    inputs = _eligible_inputs()
    inputs["point_in_time"] = False

    verdict = apply_verdict(**inputs)

    assert verdict["verdict"] == "reject"
    assert verdict["checks"]["point_in_time"] is False


def test_verdict_rejects_candidate_without_any_net_holdout_advantage():
    inputs = _eligible_inputs()
    inputs["holdout_sharpe_delta"] = -0.01
    inputs["holdout_excess_cagr"] = -0.2

    verdict = apply_verdict(**inputs)

    assert verdict["verdict"] == "reject"


def test_write_results_is_deterministic_and_creates_parent(tmp_path):
    output = tmp_path / "nested" / "results.json"
    payload = {"z": 1, "a": [2, 3]}

    write_results(payload, output)
    first = output.read_text()
    write_results(payload, output)

    assert output.read_text() == first
    assert json.loads(first) == payload
    assert first.index('"a"') < first.index('"z"')


def test_run_revalidation_rejects_unregistered_price_snapshot(tmp_path):
    prices = tmp_path / "prices.json"
    prices.write_text("{}")

    with pytest.raises(ValueError, match="pre-registered SHA-256"):
        run_revalidation(
            prices_path=prices,
            market_db_path=tmp_path / "market.db",
        )


def test_turnover_is_stable_across_python_hash_seeds():
    script = """
from src.backtest.metrics import compute_one_way_turnover
previous = {f"A{i}": 0.0 for i in range(1000)}
current = {f"A{i}": (1.0 if i == 0 else 1e-16) for i in range(1000)}
print(repr(compute_one_way_turnover(previous, current)))
"""
    outputs = []
    for seed in ("1", "2", "3", "4"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", script],
                cwd=Path(__file__).parents[1],
                env=environment,
                text=True,
            ).strip()
        )

    assert len(set(outputs)) == 1
