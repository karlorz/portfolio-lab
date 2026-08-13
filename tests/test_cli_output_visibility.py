"""CLI output visibility smoke guard (CLI-VISIBILITY-SWEEP s8).

Asserts every operator-facing CLI entrypoint prints to stdout instead of
running silent (exit 0 with zero output — the defect class fixed by adding
``configure_logging()`` to each ``__main__`` block). Runs each module as a
subprocess with safe args from the item's sub-task table.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

CLI_CASES = [
    ("src.monitor.cvar_metrics", []),
    ("src.signals.tsmom_integration", ["--portfolio"]),
    ("src.strategy.hedge_selector", ["status"]),
    ("src.signals.integrator", ["portfolio", "--portfolio", "46/38/16"]),
    ("src.dashboard.generation_store", ["--list"]),
    ("src.monitor.prod_ideas", []),
    ("src.data.fred_readiness", []),
]


@pytest.mark.parametrize("module,args", CLI_CASES, ids=[case[0] for case in CLI_CASES])
def test_cli_prints_output(module: str, args: list[str]) -> None:
    env = dict(os.environ)
    env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    result = subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True,
        text=True,
        timeout=90,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert result.returncode == 0, f"{module} exited {result.returncode}: {result.stderr[-500:]}"
    # configure_logging() routes to stderr (standard logging default); the
    # defect class is a fully silent CLI (no handlers configured), so the
    # gate asserts combined stream output, not a specific stream.
    assert (result.stdout + result.stderr).strip(), f"{module} produced no output"
