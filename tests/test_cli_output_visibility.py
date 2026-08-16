"""CLI output visibility smoke guard (CLI-VISIBILITY-SWEEP s8).

Asserts every operator-facing CLI entrypoint prints to stdout instead of
running silent (exit 0 with zero output — the defect class fixed by adding
``configure_logging()`` to each ``__main__`` block). Runs each module as a
subprocess with safe args from the item's sub-task table.
"""

from __future__ import annotations

import json
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


# RELEASE-VERIFY-SMOKE (Item Q4, 2026-08-17): subprocess smoke for the
# static-release verifier CLI (scripts/verify_lab_release.py). The happy
# path builds a hermetic tmp release tree + manifest via build_lab_release
# (same helpers as test_lab_release_artifact), so the verifier never touches
# the real dist or public data.
VERIFY_RELEASE = os.path.join("scripts", "verify_lab_release.py")


def _run_verify_release(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    return subprocess.run(
        [sys.executable, VERIFY_RELEASE, *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def _write_release_tree(root: Path) -> None:
    (root / "assets").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "index.html").write_text("<main>Portfolio Lab</main>", encoding="utf-8")
    (root / "assets/app.js").write_text("console.log('ok')\n", encoding="utf-8")
    (root / "data/signals.json").write_text('{"mutable":true}\n', encoding="utf-8")


def _write_valid_manifest(root: Path) -> None:
    import importlib.util

    sys.path.insert(0, str(Path("scripts").resolve()))
    builder_path = Path("scripts/build_lab_release.py").resolve()
    spec = importlib.util.spec_from_file_location("build_lab_release_q4", builder_path)
    assert spec is not None and spec.loader is not None
    builder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = builder
    spec.loader.exec_module(builder)
    manifest = builder.build_manifest(
        root,
        source_git_sha="a" * 40,
        build_command="bun run build",
        bun_version_value="1.2.3",
        lockfile_path="bun.lock",
        lockfile_sha256="b" * 64,
        build_time_utc="2026-07-31T00:00:00Z",
    )
    builder.write_manifest(root, manifest)


def test_verify_lab_release_valid_tree_exits_0(tmp_path) -> None:
    """RELEASE-VERIFY-SMOKE: valid release tree → exit 0 with the ok marker."""
    _write_release_tree(tmp_path)
    _write_valid_manifest(tmp_path)
    res = _run_verify_release("--release-dir", str(tmp_path))
    assert res.returncode == 0, res.stderr
    assert "release verification ok" in res.stdout


def test_verify_lab_release_missing_manifest_exits_1(tmp_path) -> None:
    """RELEASE-VERIFY-SMOKE: missing _release.json → exit 1 + "missing manifest"."""
    _write_release_tree(tmp_path)
    res = _run_verify_release("--release-dir", str(tmp_path))
    assert res.returncode == 1
    assert "missing manifest" in res.stderr


def test_verify_lab_release_help_and_missing_args() -> None:
    """RELEASE-VERIFY-SMOKE: --help exits 0 with usage; missing release-dir
    arg → argparse usage error (exit 2)."""
    help_res = _run_verify_release("--help")
    assert help_res.returncode == 0
    assert "--release-dir" in help_res.stdout
    bad_res = _run_verify_release()
    assert bad_res.returncode == 2
    assert "usage" in (bad_res.stdout + bad_res.stderr)


# CRON-OVERLAP-SMOKE (Item Q5, 2026-08-17): subprocess smoke for the
# multi-backend cron overlap detector (scripts/detect_cron_overlap.py).
# The detector shells out to `crontab -l` and `hermes cron list`, so the
# tests install fake shims for both on PATH (same pattern as the
# deploy-remote fake ssh canary above) and feed --cron-status a hermetic
# tmp fixture. No live crontab/hermes state is ever read.
DETECT_OVERLAP = os.path.join("scripts", "detect_cron_overlap.py")


def _fake_cron_bins(tmp_path: Path, crontab_text: str, hermes_text: str) -> Path:
    """Install `crontab` and `hermes` shims that echo fixed fixture text."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, text in (("crontab", crontab_text), ("hermes", hermes_text)):
        shim = bin_dir / name
        shim.write_text(
            "#!/usr/bin/env bash\n"
            f'cat <<\'EOF\'\n{text}\nEOF\n',
            encoding="utf-8",
        )
        shim.chmod(0o755)
    return bin_dir


def _run_detect_overlap(
    cron_status: Path, bins: Path
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    env["PATH"] = f"{bins}{os.pathsep}{env['PATH']}"
    return subprocess.run(
        [sys.executable, DETECT_OVERLAP, "--cron-status", str(cron_status)],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def _write_cron_status(tmp_path: Path, *, names: list[str]) -> Path:
    path = tmp_path / "cron_status.json"
    path.write_text(
        json.dumps(
            {
                "backend": "tasker",
                "jobs": [
                    {
                        "name": name,
                        "enabled": True,
                        "manual_only": False,
                        "backend": "tasker",
                    }
                    for name in names
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_detect_cron_overlap_clean_exits_0(tmp_path) -> None:
    """CRON-OVERLAP-SMOKE: tasker-only fixture, empty crontab/hermes → exit
    0 with the no-overlap marker."""
    status = _write_cron_status(tmp_path, names=["portfolio-lab-data"])
    bins = _fake_cron_bins(tmp_path, crontab_text="", hermes_text="")
    res = _run_detect_overlap(status, bins)
    assert res.returncode == 0, res.stderr
    assert "No multi-backend overlap" in res.stdout


def test_detect_cron_overlap_overlapping_exits_1(tmp_path) -> None:
    """CRON-OVERLAP-SMOKE: job owned by crontab and tasker → exit 1 with the
    overlapping marker."""
    status = _write_cron_status(tmp_path, names=["portfolio-lab-data"])
    crontab_text = (
        "5 * * * * CRON_BACKEND=crontab make -C /root/projects/portfolio-lab data\n"
    )
    bins = _fake_cron_bins(tmp_path, crontab_text=crontab_text, hermes_text="")
    res = _run_detect_overlap(status, bins)
    assert res.returncode == 1
    assert "Overlapping cron jobs" in (res.stdout + res.stderr)


# DATA-QUALITY-SMOKE (Item Q6, 2026-08-17): subprocess smoke for the offline
# public price quality auditor (scripts/check_public_data_quality.py). All
# cases pass an explicit --prices tmp fixture and --json-report tmp path so
# live PUBLIC_DATA_DIR state is never read.
DATA_QUALITY = os.path.join("scripts", "check_public_data_quality.py")


def _run_data_quality(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.run(
        [sys.executable, DATA_QUALITY, *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def _clean_prices(tmp_path: Path) -> Path:
    path = tmp_path / "prices.json"
    path.write_text(
        json.dumps(
            {
                "SPY": [
                    {"d": "2026-08-13", "p": 100.0},
                    {"d": "2026-08-14", "p": 101.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_check_public_data_quality_clean_payload_exits_0(tmp_path) -> None:
    """DATA-QUALITY-SMOKE: clean synthetic payload → exit 0 with the pass
    marker."""
    prices = _clean_prices(tmp_path)
    res = _run_data_quality("--prices", str(prices))
    assert res.returncode == 0, res.stderr
    assert "price data quality check passed" in res.stdout


def test_check_public_data_quality_corrupted_payload_exits_1(tmp_path) -> None:
    """DATA-QUALITY-SMOKE: duplicate-date payload → exit 1 with stderr
    ERROR: markers."""
    prices = _clean_prices(tmp_path)
    prices.write_text(
        json.dumps(
            {
                "SPY": [
                    {"d": "2026-08-14", "p": 100.0},
                    {"d": "2026-08-14", "p": 101.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    res = _run_data_quality("--prices", str(prices))
    assert res.returncode == 1
    assert "ERROR:" in res.stderr


def test_check_public_data_quality_json_report_written(tmp_path) -> None:
    """DATA-QUALITY-SMOKE: --json-report writes the full audit report."""
    prices = _clean_prices(tmp_path)
    report = tmp_path / "report.json"
    res = _run_data_quality("--prices", str(prices), "--json-report", str(report))
    assert res.returncode == 0, res.stderr
    assert report.exists()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "public-price-data-quality-cli/v1"
    assert payload["status"] in ("ok", "warn", "fail")
    assert payload["symbols_checked"] == 1
