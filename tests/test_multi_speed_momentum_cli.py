"""CLI output visibility tests for ``python -m src.signals.multi_speed_momentum``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path


def _write_prices(public_data_dir: Path, days: int = 430) -> None:
    public_data_dir.mkdir(parents=True, exist_ok=True)
    start = date(2024, 1, 1)
    current = start
    business_dates: list[date] = []
    while len(business_dates) < days:
        if current.weekday() < 5:
            business_dates.append(current)
        current += timedelta(days=1)

    prices = {}
    for symbol, start_price, drift in [
        ("SPY", 500.0, 0.0007),
        ("GLD", 190.0, 0.0003),
        ("TLT", 95.0, -0.0001),
    ]:
        value = start_price
        rows = []
        for idx, day in enumerate(business_dates):
            value *= 1 + drift + ((idx % 11) - 5) * 0.0001
            rows.append({"d": day.isoformat(), "p": round(value, 4)})
        prices[symbol] = rows

    (public_data_dir / "prices.json").write_text(json.dumps(prices), encoding="utf-8")


def _run_module(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    public_data_dir = tmp_path / "public-data"
    _write_prices(public_data_dir)
    env = {
        **os.environ,
        "PORTFOLIO_LAB_ENABLE_ML": "0",
        "PUBLIC_DATA_DIR": str(public_data_dir),
    }
    return subprocess.run(
        [sys.executable, "-m", "src.signals.multi_speed_momentum", *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=Path(__file__).resolve().parents[1],
    )


def test_status_command_emits_visible_output(tmp_path):
    result = _run_module(tmp_path, "status")
    combined = result.stdout + result.stderr

    assert result.returncode == 0
    assert "Multi-Speed Momentum Ensemble v2.56 - Status" in combined
    assert "FAST TIER" in combined
    assert "Data source" in combined


def test_live_command_emits_visible_recommendation(tmp_path):
    result = _run_module(tmp_path, "live", "--portfolio", "46/38/16")
    combined = result.stdout + result.stderr

    assert result.returncode == 0
    assert '"target_allocation"' in combined
    assert '"overall_confidence"' in combined
    assert '"SPY"' in combined
