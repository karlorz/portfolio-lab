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
from pathlib import Path

import pytest

CLI_CASES = [
    ("src.monitor.cvar_metrics", []),
    ("src.signals.tsmom_integration", ["--portfolio"]),
    ("src.strategy.hedge_selector", ["status"]),
    ("src.signals.integrator", ["portfolio", "--portfolio", "46/38/16"]),
    ("src.dashboard.generation_store", ["--list"]),
    ("src.monitor.prod_ideas", []),
    ("src.data.fred_readiness", []),
    # CLI-VISIBILITY-SWEEP-2 additions (2026-08-14): 13 more entrypoints.
    ("src.monitor.rebalance_health", []),
    ("src.monitor.garch_cvar", []),
    ("src.rebalancing.smart_rebalancer", []),
    ("src.analytics.calculator", []),
    ("src.strategy.regime_sentiment", []),
    ("src.strategy.vol_parity_allocator", []),
    ("src.strategy.graduation_checklist", ["report"]),
    ("src.strategy.ensemble_voter", ["vote"]),
    ("src.signals.multi_strategy_adapters", []),
    ("src.data.vix_futures", []),
    ("src.monitor.fred_readiness", []),
    ("src.chat.portfolio_query", ["what is my equity exposure"]),
    ("src.data.alternative_data", ["composite", "--ticker", "AAPL"]),
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


# RESOLVE-CLI-SMOKE (Item 2, 2026-08-16): subprocess smoke for the operator
# resolution CLI (scripts/resolve_incident.py). Every run passes all three
# --log-path/--summary-path/--kill-switch-path at tmp fixtures — the CLI
# defaults (resolve_incident.py:46-50) are the real DEFAULT_*_PATH and a
# bare run would touch live stores. Seeds are created through
# IncidentManager.record_alert (src/monitor/incident_manager.py) — no
# hand-rolled event schema.
RESOLVE_CLI = os.path.join("scripts", "resolve_incident.py")


def _run_resolve_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    return subprocess.run(
        [sys.executable, RESOLVE_CLI, *args],
        capture_output=True,
        text=True,
        timeout=90,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def _seed_firing_incident(tmp_path) -> tuple[Path, Path, Path, str]:
    from src.monitor.incident_manager import IncidentManager

    log = tmp_path / "incidents.jsonl"
    summary = tmp_path / "incidents.json"
    kill = tmp_path / "kill_switch.json"
    incident = IncidentManager(
        log_path=log, summary_path=summary, kill_switch_path=kill
    ).record_alert(
        channel="signal_staleness", level="warn", message="smoke firing event"
    )
    assert incident is not None
    return log, summary, kill, incident.incident_id


def test_resolve_incident_unknown_id_exits_1(tmp_path) -> None:
    """RESOLVE-CLI-SMOKE: unknown id on an empty tmp store → exit 1."""
    log = tmp_path / "incidents.jsonl"
    summary = tmp_path / "incidents.json"
    kill = tmp_path / "kill_switch.json"
    res = _run_resolve_cli(
        "--incident-id",
        "00000000-0000-0000-0000-000000000000",
        "--message",
        "smoke",
        "--log-path",
        str(log),
        "--summary-path",
        str(summary),
        "--kill-switch-path",
        str(kill),
    )
    assert res.returncode == 1
    assert "unknown incident id" in res.stderr


def test_resolve_incident_dry_run_leaves_tmp_store_unchanged(tmp_path) -> None:
    """RESOLVE-CLI-SMOKE: dry-run on a seeded firing event exits 0 and does
    not mutate the tmp store (C16 idempotency contract — log, summary and
    kill switch byte-unchanged)."""
    log, summary, kill, incident_id = _seed_firing_incident(tmp_path)
    # Kill switch may not exist for a first warn-level alert (escalation-
    # gated write) — snapshot whatever is present and require both
    # byte-identity and presence/absence to be unchanged.
    before = {p: (p.read_bytes() if p.exists() else None) for p in (log, summary, kill)}
    res = _run_resolve_cli(
        "--incident-id",
        incident_id,
        "--message",
        "smoke resolve",
        "--dry-run",
        "--log-path",
        str(log),
        "--summary-path",
        str(summary),
        "--kill-switch-path",
        str(kill),
    )
    assert res.returncode == 0
    assert "[dry-run]" in res.stdout
    for path, blob in before.items():
        if blob is None:
            assert not path.exists(), f"dry-run created {path.name}"
        else:
            assert path.read_bytes() == blob, f"dry-run mutated {path.name}"


def test_resolve_incident_already_resolved_exits_0(tmp_path) -> None:
    """RESOLVE-CLI-SMOKE: already-resolved id is a no-op (exit 0)."""
    from src.monitor.incident_manager import IncidentManager

    log, summary, kill, incident_id = _seed_firing_incident(tmp_path)
    IncidentManager(
        log_path=log, summary_path=summary, kill_switch_path=kill
    ).resolve_operator(incident_id, "smoke resolve")
    res = _run_resolve_cli(
        "--incident-id",
        incident_id,
        "--message",
        "smoke resolve again",
        "--log-path",
        str(log),
        "--summary-path",
        str(summary),
        "--kill-switch-path",
        str(kill),
    )
    assert res.returncode == 0
    assert "already resolved" in res.stdout


# DEPLOY-REMOTE-SMOKE (Item 20, 2026-08-16): subprocess smoke for the remote
# deploy CLI (scripts/deploy-remote.sh). Stays strictly inside the --dry-run
# contract — ssh_exec prints "[dry-run] ssh ..." and returns 0 without
# connecting (deploy-remote.sh:210-213), and main() returns before the remote
# lifecycle pipe (deploy-remote.sh:352-355). Happy-path cases prepend a fake
# `ssh` to PATH that records any real invocation: a dry-run must never reach
# it. --allow-dirty keeps the dirty-tree guard (deploy-remote.sh:170-177)
# hermetic regardless of working-tree state.
DEPLOY_REMOTE = os.path.join("scripts", "deploy-remote.sh")


def _fake_ssh_bin(tmp_path: Path) -> tuple[Path, Path]:
    """Fake `ssh` that logs any real invocation, replacing the real binary on
    PATH (also satisfies the script's own require_cmd ssh probe)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "ssh-called.log"
    ssh = bin_dir / "ssh"
    ssh.write_text(
        "#!/usr/bin/env bash\n"
        'echo "REAL SSH INVOKED: $*" >> "$DEPLOY_REMOTE_SSH_LOG"\n'
        "exit 99\n",
        encoding="utf-8",
    )
    ssh.chmod(0o755)
    return bin_dir, log


def _run_deploy_remote(
    *args: str, fake_ssh: tuple[Path, Path] | None = None
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    if fake_ssh is not None:
        bin_dir, ssh_log = fake_ssh
        env["DEPLOY_REMOTE_SSH_LOG"] = str(ssh_log)
        env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    return subprocess.run(
        ["bash", DEPLOY_REMOTE, *args],
        capture_output=True,
        text=True,
        timeout=60,  # acceptance (a): dry-run must exit in <60s
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def test_deploy_remote_dry_run_preview_never_connects(tmp_path) -> None:
    """DEPLOY-REMOTE-SMOKE: preview dry-run exits 0, prints only [dry-run]
    markers, and never invokes ssh (fake-ssh canary untouched)."""
    res = _run_deploy_remote(
        "--host",
        "dry-fake",
        "--mode",
        "preview",
        "--dry-run",
        "--allow-dirty",
        fake_ssh=_fake_ssh_bin(tmp_path),
    )
    assert res.returncode == 0
    assert "[dry-run] ssh dry-fake " in res.stderr
    combined = res.stdout + res.stderr
    assert "[dry-run] rsync project ->" in combined or "[dry-run] tar stream" in combined
    assert "Deploy completed" not in combined  # client lifecycle tail never runs
    for line in combined.splitlines():
        if "ssh" in line:
            assert "[dry-run]" in line, f"non-dry-run ssh line: {line}"
    assert not (tmp_path / "ssh-called.log").exists(), "dry-run invoked real ssh"


def test_deploy_remote_dry_run_production_variant(tmp_path) -> None:
    """DEPLOY-REMOTE-SMOKE: production dry-run variant exits 0 with markers."""
    res = _run_deploy_remote(
        "--host",
        "dry-fake",
        "--mode",
        "production",
        "--dry-run",
        "--allow-dirty",
        fake_ssh=_fake_ssh_bin(tmp_path),
    )
    assert res.returncode == 0
    assert "Deploy mode: production" in res.stdout
    assert "[dry-run] ssh dry-fake " in res.stderr
    assert not (tmp_path / "ssh-called.log").exists(), "dry-run invoked real ssh"


def test_deploy_remote_missing_host_dies() -> None:
    """DEPLOY-REMOTE-SMOKE: missing --host → usage + die (exit 1)."""
    res = _run_deploy_remote("--mode", "preview", "--dry-run")
    assert res.returncode == 1
    assert "--host is required" in res.stderr


def test_deploy_remote_unknown_option_dies() -> None:
    """DEPLOY-REMOTE-SMOKE: unknown option → die (exit 1)."""
    res = _run_deploy_remote("--host", "dry-fake", "--bogus-option")
    assert res.returncode == 1
    assert "Unknown option" in res.stderr
