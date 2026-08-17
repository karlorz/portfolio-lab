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


# DATA-CONSISTENCY-SMOKE (Item Q7, 2026-08-17): subprocess smoke for the
# deploy-time public data consistency auditor
# (scripts/check_public_data_consistency.py). Every case audits a hermetic
# tmp mock app tree via --app-dir + --allow-repo-public-data +
# --skip-dist-data-match, so live PUBLIC_DATA_DIR / dist / market.db state
# is never read (fixture mirrors the coherent set builder in
# tests/test_generated_public_data_consistency_smoke.py).
DATA_CONSISTENCY = os.path.join("scripts", "check_public_data_consistency.py")


def _run_data_consistency(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # conftest exports PUBLIC_DATA_DIR for the pytest session; the ops
    # resolver prefers it over --app-dir (src/paths.py:183-185), so drop it
    # (and the live-tree override) to keep --app-dir authoritative + hermetic.
    env.pop("PUBLIC_DATA_DIR", None)
    env.pop("PORTFOLIO_LAB_LIVE_PUBLIC_DATA_DIR", None)
    return subprocess.run(
        [sys.executable, DATA_CONSISTENCY, *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def _write_consistent_app_fixture(tmp_path: Path) -> None:
    """Coherent mock app tree: source_manifest/index/health with matching
    identity and timestamp ordering (same shape as
    test_generated_public_data_consistency_smoke fixtures)."""
    import hashlib

    public_data = tmp_path / "public" / "data"
    public_data.mkdir(parents=True)
    source_generated_at = "2026-06-12T09:05:25.028Z"
    index_generated_at = "2026-06-12T09:06:00+00:00"
    source_manifest = {
        "schema_version": "market-data-source-manifest/v1",
        "generated_at": source_generated_at,
        "artifacts": [
            {"artifact": "prices.json", "provider": "Yahoo Finance", "status": "success"}
        ],
    }
    (public_data / "source_manifest.json").write_text(
        json.dumps(source_manifest, sort_keys=True), encoding="utf-8"
    )
    source_sha256 = hashlib.sha256(
        (public_data / "source_manifest.json").read_bytes()
    ).hexdigest()
    (public_data / "index.json").write_text(
        json.dumps(
            {
                "schema_version": "public-data-index/v1",
                "generated_at": index_generated_at,
                "source_manifest": {
                    "path": "source_manifest.json",
                    "schema_version": "market-data-source-manifest/v1",
                    "generated_at": source_generated_at,
                    "sha256": source_sha256,
                },
                "entries": [
                    {
                        "filename": "source_manifest.json",
                        "path": "source_manifest.json",
                        "status": "present",
                        "generated_at": source_generated_at,
                        "sha256": source_sha256,
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (public_data / "health.json").write_text(
        json.dumps(
            {"status": "ok", "generated_at": index_generated_at},
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_check_public_data_consistency_clean_app_exits_0(tmp_path) -> None:
    """DATA-CONSISTENCY-SMOKE: coherent mock app tree → exit 0 with the pass
    marker (provenance warnings on stderr are non-blocking)."""
    _write_consistent_app_fixture(tmp_path)
    res = _run_data_consistency(
        "--app-dir",
        str(tmp_path),
        "--allow-repo-public-data",
        "--skip-dist-data-match",
    )
    assert res.returncode == 0, res.stderr
    assert "public data consistency check passed" in res.stdout
    assert "WARN:" in res.stderr  # missing generator_git_sha is advisory only


def test_check_public_data_consistency_missing_required_exits_1(tmp_path) -> None:
    """DATA-CONSISTENCY-SMOKE: missing required data file (source_manifest.json)
    → exit 1 with ERROR: markers on stderr."""
    _write_consistent_app_fixture(tmp_path)
    (tmp_path / "public" / "data" / "source_manifest.json").unlink()
    res = _run_data_consistency(
        "--app-dir",
        str(tmp_path),
        "--allow-repo-public-data",
        "--skip-dist-data-match",
    )
    assert res.returncode == 1
    assert "ERROR:" in res.stderr
    assert "source_manifest.json is missing" in res.stderr


# CRON-VERIFY-SMOKE (Item Q8, 2026-08-17): subprocess smoke for the cron
# contract verifier (scripts/cron_verify.py). The --crontab case audits the
# checked-in repo crontab (same target set as `make verify-cron-sync`); the
# --status-file cases use hermetic tmp fixtures (script resolves its own
# PROJECT_ROOT on sys.path), so no live cron_status state is read.
CRON_VERIFY = os.path.join("scripts", "cron_verify.py")


def _run_cron_verify(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    return subprocess.run(
        [sys.executable, CRON_VERIFY, *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def _write_tasker_status_fixture(tmp_path: Path, *, drop: str | None = None) -> Path:
    """Hermetic tasker cron_status fixture; ``drop`` removes one registry job
    to simulate a diverged/corrupted status contract."""
    from src.tasker.registry import load_task_registry

    names = [task.id for task in load_task_registry().tasks]
    if drop is not None:
        names = [name for name in names if name != drop]
    path = tmp_path / "cron_status.json"
    path.write_text(
        json.dumps({"backend": "tasker", "jobs": [{"name": name} for name in names]}),
        encoding="utf-8",
    )
    return path


def test_cron_verify_repo_crontab_exits_0() -> None:
    """CRON-VERIFY-SMOKE: checked-in repo crontab → exit 0 with the OK marker
    (full CRON_TARGETS coverage — same contract as `make verify-cron-sync`)."""
    res = _run_cron_verify("--crontab", "crontab")
    assert res.returncode == 0, res.stderr
    assert "OK:" in res.stdout
    assert "crontab targets present" in res.stdout


def test_cron_verify_valid_status_fixture_exits_0(tmp_path) -> None:
    """CRON-VERIFY-SMOKE: hermetic tasker status fixture matching the full
    tasker registry → exit 0 with the OK marker."""
    status = _write_tasker_status_fixture(tmp_path)
    res = _run_cron_verify("--status-file", str(status))
    assert res.returncode == 0, res.stderr
    assert "OK:" in res.stdout
    assert "tasker targets present" in res.stdout


def test_cron_verify_missing_and_divergent_status_exits_1(tmp_path) -> None:
    """CRON-VERIFY-SMOKE: absent status file → exit 1 with MISSING:; a status
    fixture missing one registry job → exit 1 with FAIL:."""
    res_missing = _run_cron_verify("--status-file", str(tmp_path / "nope.json"))
    assert res_missing.returncode == 1
    assert "MISSING:" in res_missing.stdout

    from src.tasker.registry import load_task_registry

    first_job = sorted(task.id for task in load_task_registry().tasks)[0]
    status = _write_tasker_status_fixture(tmp_path, drop=first_job)
    res_fail = _run_cron_verify("--status-file", str(status))
    assert res_fail.returncode == 1
    assert "FAIL:" in res_fail.stdout
    assert first_job in res_fail.stdout


# ARTIFACT-RETENTION-SMOKE (Item Q9, 2026-08-17): subprocess smoke for the
# Labs artifact retention reporter (scripts/report_artifact_retention.py).
# Every case passes explicit hermetic tmp --data-dir/--public-data-dir so no
# live data or public tree is scanned; the reporter is report-only
# (--execute-move is explicitly refused with exit 2).
RETENTION_REPORT = os.path.join("scripts", "report_artifact_retention.py")


def _run_retention_report(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.run(
        [sys.executable, RETENTION_REPORT, *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def _empty_retention_dirs(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    public_data_dir = tmp_path / "public-data"
    data_dir.mkdir()
    public_data_dir.mkdir()
    return data_dir, public_data_dir


def test_report_artifact_retention_empty_dirs_json_counts(tmp_path) -> None:
    """ARTIFACT-RETENTION-SMOKE: empty tmp dirs → exit 0 + JSON with counts."""
    data_dir, public_data_dir = _empty_retention_dirs(tmp_path)
    res = _run_retention_report("--data-dir", str(data_dir), "--public-data-dir", str(public_data_dir))
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert "counts" in payload
    assert payload["dry_run"] is True


def test_report_artifact_retention_archive_plan_json(tmp_path) -> None:
    """ARTIFACT-RETENTION-SMOKE: --archive-plan → exit 0 + JSON archive plan."""
    data_dir, public_data_dir = _empty_retention_dirs(tmp_path)
    res = _run_retention_report(
        "--data-dir", str(data_dir), "--public-data-dir", str(public_data_dir), "--archive-plan"
    )
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert "plan" in res.stdout
    assert payload["dry_run"] is True
    assert payload["move_enabled"] is False


def test_report_artifact_retention_refuses_execute_move(tmp_path) -> None:
    """ARTIFACT-RETENTION-SMOKE: --execute-move is not implemented → exit 2
    with the refusal marker on stderr."""
    data_dir, public_data_dir = _empty_retention_dirs(tmp_path)
    res = _run_retention_report(
        "--data-dir", str(data_dir), "--public-data-dir", str(public_data_dir), "--execute-move"
    )
    assert res.returncode == 2
    assert "Move execution is not implemented" in res.stderr


# TEST-SEGMENTATION-SMOKE (Item Q10, 2026-08-17): subprocess smoke for the
# test segmentation validator (scripts/validate_test_segmentation.py). The
# --save-results case runs inside a hermetic mock lab tree via
# PORTFOLIO_LAB_PROJECT_DIR (mock tests/ + a trivial Makefile test-fast
# target), so live test suites are never invoked and results JSON lands in
# the mock tree.
TEST_SEGMENTATION = os.path.join("scripts", "validate_test_segmentation.py")


def _run_test_segmentation(
    *args: str, project_dir: Path | None = None
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    if project_dir is not None:
        env["PORTFOLIO_LAB_PROJECT_DIR"] = str(project_dir)
    return subprocess.run(
        [sys.executable, TEST_SEGMENTATION, *args],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def _write_mock_lab_tree(tmp_path: Path) -> None:
    """Small mock lab: one test file plus a trivially-green test-fast target
    (the script runs `make test-fast` inside PORTFOLIO_LAB_PROJECT_DIR)."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_demo.py").write_text(
        "def test_demo():\n    assert True\n", encoding="utf-8"
    )
    (tmp_path / "Makefile").write_text(
        "test-fast:\n\t@echo mock-test-fast-ok\n", encoding="utf-8"
    )


def test_validate_test_segmentation_help_exits_0() -> None:
    """TEST-SEGMENTATION-SMOKE: --help → exit 0 with the description marker."""
    res = _run_test_segmentation("--help")
    assert res.returncode == 0
    assert "Validate test segmentation" in res.stdout


def test_validate_test_segmentation_save_results_mock_tree(tmp_path) -> None:
    """TEST-SEGMENTATION-SMOKE: --save-results in a mock lab tree → exit 0
    with the header on stdout and results JSON written under the mock tree."""
    _write_mock_lab_tree(tmp_path)
    res = _run_test_segmentation("--save-results", project_dir=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "Test Segmentation Validation" in res.stdout
    results_file = tmp_path / "data" / "test_segmentation_results.json"
    assert results_file.exists()
    payload = json.loads(results_file.read_text(encoding="utf-8"))
    assert "analysis" in payload
    assert "test_fast" in payload
    assert payload["analysis"]["total_test_files"] == 1


# PUBLIC-INDEX-REFRESH-SMOKE (Item Q11, 2026-08-17): subprocess smoke for the
# public/index.json rebuild CLI (scripts/refresh_public_data_index.py). Every
# case passes an explicit --public-dir tmp fixture (never a bare invocation),
# so the repo public/data/index.json is never touched. Failure mode uses a
# file positioned where the public dir should be: dir creation then fails
# (FileExistsError) and the rebuild returns {"ok": false, ...}.
PUBLIC_INDEX_REFRESH = os.path.join("scripts", "refresh_public_data_index.py")


def _run_public_index_refresh(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.run(
        [sys.executable, PUBLIC_INDEX_REFRESH, *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def _write_index_source_fixture(tmp_path: Path) -> Path:
    """Hermetic public-data dir with the partial-write core inputs."""
    public_dir = tmp_path / "public-data"
    public_dir.mkdir()
    (public_dir / "source_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "market-data-source-manifest/v1",
                "generated_at": "2026-06-12T09:05:25.028Z",
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )
    (public_dir / "prices.json").write_text(
        json.dumps({"SPY": [{"d": "2026-08-14", "p": 100.0}]}), encoding="utf-8"
    )
    return public_dir


def test_refresh_public_data_index_help_exits_0() -> None:
    """PUBLIC-INDEX-REFRESH-SMOKE: --help → exit 0 with the description marker."""
    res = _run_public_index_refresh("--help")
    assert res.returncode == 0
    assert "Rebuild public/data/index.json" in res.stdout


def test_refresh_public_data_index_rebuild_and_failure_json(tmp_path) -> None:
    """PUBLIC-INDEX-REFRESH-SMOKE: source_manifest+prices fixture → exit 0
    with {"ok": true} and content_patch_source carrying --reason; a file in
    the public-dir slot (unwriteable) → exit 1 with {"ok": false}."""
    public_dir = _write_index_source_fixture(tmp_path)
    res = _run_public_index_refresh(
        "--public-dir", str(public_dir), "--reason", "smoke-q11"
    )
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert payload["ok"] is True
    assert payload["content_patch_source"] == "index_refresh:smoke-q11"
    assert (public_dir / "index.json").exists()

    blocker = tmp_path / "not-a-dir"
    blocker.write_text("i am a file", encoding="utf-8")
    res_fail = _run_public_index_refresh("--public-dir", str(blocker))
    assert res_fail.returncode == 1
    fail_payload = json.loads(res_fail.stdout)
    assert fail_payload["ok"] is False


# ROUTING-CONTRACT-SMOKE (Item Q12, 2026-08-17): subprocess smoke for the
# SkillWiki/Hermes routing invariant auditor
# (scripts/audit_routing_contract.py). All fixture-backed cases pass explicit
# --hermes-home/--skillwiki-env tmp mocks (bare invocation would inspect the
# live host /root/.hermes and /root/.skillwiki), mirroring the fixture shape
# in tests/test_audit_routing_contract.py.
AUDIT_ROUTING = os.path.join("scripts", "audit_routing_contract.py")


def _run_audit_routing(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    return subprocess.run(
        [sys.executable, AUDIT_ROUTING, *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def _write_routing_fixture(tmp_path: Path, *, divergent: bool = False) -> tuple[Path, Path]:
    """Hermetic SkillWiki dotenv + Hermes home mock; ``divergent`` breaks the
    global WIKI_PATH invariant so exactly one violation is reported."""
    skillwiki_env = tmp_path / ".skillwiki" / ".env"
    skillwiki_env.parent.mkdir(parents=True)
    with_global = "/root/other" if divergent else "/root/wiki"
    skillwiki_env.write_text(
        "\n".join(
            [
                f"WIKI_PATH={with_global}",
                "WIKI_LANG=en",
                "WIKI_DEFAULT=portfolio",
                "WIKI_PORTFOLIO_PATH=/root/wiki",
                "WIKI_FINANCE_PATH=/root/wiki-fin",
                "",
            ]
        ),
        encoding="utf-8",
    )

    hermes_home = tmp_path / ".hermes"
    (hermes_home / "cron").mkdir(parents=True)
    (hermes_home / ".env").write_text("WIKI_PATH=/root/wiki\n", encoding="utf-8")
    (hermes_home / "cron" / "jobs.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "finance-1",
                        "name": "finance-digest",
                        "enabled": True,
                        "state": "scheduled",
                        "profile": "finance",
                        "workdir": "/root/wiki-fin",
                        "script": "finance-digest-wrapper.sh",
                    },
                    {
                        "id": "portfolio-default-paused",
                        "name": "portfolio-lab-dashboard",
                        "enabled": False,
                        "state": "paused",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    finance_home = hermes_home / "profiles" / "finance"
    (finance_home / "scripts").mkdir(parents=True)
    (finance_home / ".env").write_text("WIKI_PATH=/root/wiki-fin\n", encoding="utf-8")
    for name in ("finance-digest-wrapper.sh", "finance-news-collector.py"):
        script = finance_home / "scripts" / name
        script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        script.chmod(script.stat().st_mode | 0o111)

    coder_home = hermes_home / "profiles" / "coder" / "cron"
    coder_home.mkdir(parents=True)
    coder_home.joinpath("jobs.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "coder-paused",
                        "name": "portfolio-lab-dashboard",
                        "enabled": False,
                        "state": "paused",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return skillwiki_env, hermes_home


def test_audit_routing_contract_help_exits_0() -> None:
    """ROUTING-CONTRACT-SMOKE: --help → exit 0 with the description marker."""
    res = _run_audit_routing("--help")
    assert res.returncode == 0
    assert "Audit SkillWiki and Hermes routing invariants" in res.stdout


def test_audit_routing_contract_valid_fixture_exits_0(tmp_path) -> None:
    """ROUTING-CONTRACT-SMOKE: coherent mock hermes/skillwiki → exit 0 with
    the contract-holds marker."""
    skillwiki_env, hermes_home = _write_routing_fixture(tmp_path)
    res = _run_audit_routing(
        "--hermes-home", str(hermes_home), "--skillwiki-env", str(skillwiki_env)
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "OK: routing contract holds" in res.stdout


def test_audit_routing_contract_divergent_fixture_exits_1(tmp_path) -> None:
    """ROUTING-CONTRACT-SMOKE: broken global WIKI_PATH → exit 1 with the
    violation marker."""
    skillwiki_env, hermes_home = _write_routing_fixture(tmp_path, divergent=True)
    res = _run_audit_routing(
        "--hermes-home", str(hermes_home), "--skillwiki-env", str(skillwiki_env)
    )
    assert res.returncode == 1
    assert "routing contract violation" in res.stdout
    assert "ERROR:" in res.stdout


# CRON-UPDATE-SMOKE (Item Q13, 2026-08-17): subprocess smoke for the cron
# status writer (scripts/cron_update.py). The script resolves PROJECT_ROOT
# from __file__ (no env override), so the update case runs a byte-identical
# copy under a mock root — data/cron_status.json in the repo is never
# touched. Status vocabulary is mapped at the write boundary (ok → success).
CRON_UPDATE = os.path.join("scripts", "cron_update.py")


def _run_cron_update(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.run(
        [sys.executable, script, *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def _copy_cron_update(tmp_path: Path) -> Path:
    """Byte-identical copy placed so __file__.parents[1] is the mock root."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    import shutil

    shutil.copyfile(CRON_UPDATE, scripts_dir / "cron_update.py")
    return scripts_dir / "cron_update.py"


def test_cron_update_insufficient_args_exits_1() -> None:
    """CRON-UPDATE-SMOKE: fewer than 4 positional args → exit 1 with the
    Usage marker on stderr (no status write happens)."""
    res = _run_cron_update(CRON_UPDATE, "only-one-arg")
    assert res.returncode == 1
    assert "Usage:" in res.stderr


def test_cron_update_records_job_in_mock_root(tmp_path) -> None:
    """CRON-UPDATE-SMOKE: valid args under a copied mock script → exit 0 and
    the job row (ok → success) lands in the mock data/cron_status.json."""
    copy = _copy_cron_update(tmp_path)
    res = _run_cron_update(str(copy), "portfolio-lab-smoke", "ok", "5", "tasker")
    assert res.returncode == 0, res.stderr
    status_file = tmp_path / "data" / "cron_status.json"
    assert status_file.exists()
    data = json.loads(status_file.read_text(encoding="utf-8"))
    job = data["jobs"][0]
    assert job["name"] == "portfolio-lab-smoke"
    assert job["status"] == "success"
    assert job["duration_seconds"] == 5.0
    assert job["backend"] == "tasker"


# PRUNE-LOGS-SMOKE (Item Q14, 2026-08-17): subprocess smoke for the run-log
# pruner (scripts/prune_logs.py). The script derives PROJECT_ROOT from
# __file__ and binds DATA_DIR / the tasker store from it (no env override),
# and its cron-status mirror writes the live data/cron_status.json; a bare
# invocation would touch live state. Every case therefore runs a byte-identical
# copy under a mock root whose minimal src package points DATA_DIR +
# TaskerStore at the mock tree (Q13 copy pattern) — nothing outside tmp_path
# is read or written, and run-log pruning stays strictly in --dry-run.
PRUNE_LOGS = os.path.join("scripts", "prune_logs.py")


def _run_prune_logs(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.run(
        [sys.executable, script, *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def _copy_prune_logs(tmp_path: Path) -> Path:
    """Byte-identical copy under a mock root plus a minimal mock src package:
    src/paths.py (DATA_DIR inside the mock tree) and src/tasker/store.py
    (TaskerStore returning an empty prune summary). The script's best-effort
    cron-status mirror points at the missing mock scripts/cron_update.py and
    silently no-ops (prune_logs.py:200-205)."""
    import shutil

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    shutil.copyfile(PRUNE_LOGS, scripts_dir / "prune_logs.py")
    src_dir = tmp_path / "src"
    (src_dir / "tasker").mkdir(parents=True)
    (src_dir / "__init__.py").write_text("", encoding="utf-8")
    (src_dir / "tasker" / "__init__.py").write_text("", encoding="utf-8")
    (src_dir / "paths.py").write_text(
        "from pathlib import Path\n"
        "DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'data'\n",
        encoding="utf-8",
    )
    (src_dir / "tasker" / "store.py").write_text(
        "class TaskerStore:\n"
        "    def __init__(self, db_path=None, log_dir=None):\n"
        "        pass\n"
        "    def prune_runs(self, keep_per_task=20, dry_run=False):\n"
        "        return {'deleted_files': 0, 'deleted_rows': 0, 'kept_files': 0,\n"
        "                'bytes_freed': 0, 'errors': [], 'plan': []}\n",
        encoding="utf-8",
    )
    return scripts_dir / "prune_logs.py"


def test_prune_logs_help_exits_0(tmp_path) -> None:
    """PRUNE-LOGS-SMOKE: --help → exit 0 with the description marker."""
    copy = _copy_prune_logs(tmp_path)
    res = _run_prune_logs(str(copy), "--help")
    assert res.returncode == 0
    assert "Prune per-run tasker logs" in res.stdout


def test_prune_logs_dry_run_mock_tree(tmp_path) -> None:
    """PRUNE-LOGS-SMOKE: --dry-run in the mock tree → exit 0 with the DRY RUN
    header and the per-task retention marker (nothing outside tmp_path read
    or written)."""
    copy = _copy_prune_logs(tmp_path)
    res = _run_prune_logs(str(copy), "--dry-run")
    assert res.returncode == 0, res.stderr
    assert "Prune Logs (DRY RUN):" in res.stdout
    assert "tasker_logs: keep_per_task=20" in res.stdout


# MIRROR-REPO-PUBLIC-DATA-SMOKE (Item Q15, 2026-08-17): subprocess smoke for
# the repo public/data mirror CLI (scripts/mirror_repo_public_data.py). All
# cases pass explicit hermetic tmp --source/--dest fixtures (never a bare
# invocation) so live PUBLIC_DATA_DIR and repo public/data are never read or
# written; the --dry-run case also skips the health-lag restamp branch
# (mirror_repo_public_data.py:292), so no live monitor stores are touched.
MIRROR_REPO_PUBLIC = os.path.join("scripts", "mirror_repo_public_data.py")


def _run_mirror_repo_public(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.run(
        [sys.executable, MIRROR_REPO_PUBLIC, *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def _write_mirror_source_fixture(tmp_path: Path, *, sha: str) -> Path:
    """Hermetic source tree: two governed artifacts with distinct payloads."""
    source = tmp_path / "source"
    source.mkdir(parents=True)
    (source / "signals.json").write_text(
        json.dumps({"generator_git_sha": sha, "probe": "q15"}), encoding="utf-8"
    )
    (source / "health.json").write_text(
        json.dumps({"generator_git_sha": sha, "status": "ok"}), encoding="utf-8"
    )
    return source


def test_mirror_repo_public_data_help_exits_0() -> None:
    """MIRROR-REPO-SMOKE: --help → exit 0 with the description marker."""
    res = _run_mirror_repo_public("--help")
    assert res.returncode == 0
    assert "Mirror live operator public/data" in res.stdout


def test_mirror_repo_public_data_dry_run_writes_nothing(tmp_path) -> None:
    """MIRROR-REPO-SMOKE: explicit tmp source/dest with --dry-run → exit 0
    with JSON dry_run true + the copied list, and dest stays empty."""
    source = _write_mirror_source_fixture(tmp_path, sha="a" * 40)
    dest = tmp_path / "dest"
    dest.mkdir()
    res = _run_mirror_repo_public(
        "--source", str(source), "--dest", str(dest), "--dry-run"
    )
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert payload["dry_run"] is True
    assert "signals.json" in payload["copied"]
    assert payload["copied_count"] >= 1
    assert list(dest.iterdir()) == [], "dry-run wrote into dest"


def test_mirror_repo_public_data_lag_only_exits_1(tmp_path) -> None:
    """MIRROR-REPO-SMOKE: differing source/dest generator_git_sha with
    --lag-only → exit 1 and stdout JSON carrying the signals.json lag row."""
    source = _write_mirror_source_fixture(tmp_path, sha="a" * 40)
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "signals.json").write_text(
        json.dumps({"generator_git_sha": "b" * 40, "probe": "q15"}), encoding="utf-8"
    )
    res = _run_mirror_repo_public("--source", str(source), "--dest", str(dest), "--lag-only")
    assert res.returncode == 1
    payload = json.loads(res.stdout)
    assert "lagging" in payload
    lag_row = [row for row in payload["lagging"] if row["path"] == "signals.json"]
    assert lag_row, payload
    assert lag_row[0]["lagging"] is True
    assert lag_row[0]["source_sha"] != lag_row[0]["dest_sha"]


# BUILD-LAB-RELEASE-SMOKE (Item Q16, 2026-08-17): subprocess smoke for the
# static release builder (scripts/build_lab_release.py). The happy path runs
# the full pipeline (clean-check → detached git worktree → mock build/install
# commands → manifest → self-verify) against a hermetic mock git repo with a
# single commit + bun.lock, so no live dist/ or release directories are ever
# touched. Missing --release-dir exercises argparse's required-arg exit 2.
BUILD_RELEASE = os.path.join("scripts", "build_lab_release.py")


def _run_build_release(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.run(
        [sys.executable, BUILD_RELEASE, *args],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def _write_mock_release_repo(tmp_path: Path) -> Path:
    """Hermetic git repo: one clean commit + a bun.lock (the builder requires
    both — full_git_sha on HEAD and select_lockfile fail closed otherwise)."""
    repo = tmp_path / "mock-repo"
    repo.mkdir()
    (repo / "bun.lock").write_text("mock-lockfile\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "smoke@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Q16 smoke"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "add", "bun.lock"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "mock release source"],
        check=True,
    )
    return repo


def test_build_lab_release_help_exits_0() -> None:
    """BUILD-LAB-RELEASE-SMOKE: --help → exit 0 with the description marker."""
    res = _run_build_release("--help")
    assert res.returncode == 0
    assert "Build a verified Portfolio Lab static release" in res.stdout


def test_build_lab_release_missing_release_dir_exits_2() -> None:
    """BUILD-LAB-RELEASE-SMOKE: --release-dir is required → argparse exit 2
    with the required-arg marker on stderr."""
    res = _run_build_release("--repo-dir", "irrelevant")
    assert res.returncode == 2
    assert "required" in res.stderr


def test_build_lab_release_mock_repo_writes_manifest(tmp_path) -> None:
    """BUILD-LAB-RELEASE-SMOKE: mock git repo + mock build/install commands
    → exit 0, release-manifest.json written in the release dir with the mock
    dist artifact under assets (live dist/ untouched)."""
    repo = _write_mock_release_repo(tmp_path)
    release_dir = tmp_path / "release"
    res = _run_build_release(
        "--repo-dir",
        str(repo),
        "--release-dir",
        str(release_dir),
        "--build-command",
        "mkdir -p dist && printf '<main>mock</main>\\n' > dist/index.html",
        "--install-command",
        "true",
    )
    assert res.returncode == 0, res.stdout + res.stderr
    manifest_file = release_dir / "_release.json"
    assert manifest_file.exists()
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "portfolio-lab-static-release/v1"
    asset_paths = [asset["path"] for asset in payload["assets"]]
    assert "index.html" in asset_paths
    assert (release_dir / "index.html").read_text(encoding="utf-8") == "<main>mock</main>\n"


# MARK-TO-MARKET-SMOKE (Item Q17, 2026-08-17): subprocess smoke for the
# portfolio mark-to-market CLI (scripts/mark_to_market.py). The script loads
# and saves via DATA_DIR (repo data/) with no env override, so every data-
# touching case runs a byte-identical copy under a mock root whose minimal
# src/paths.py points DATA_DIR at the mock tree (Q13 copy pattern) — repo
# data/portfolio_paper.json and data/portfolio_live.json are never touched.
MARK_TO_MARKET = os.path.join("scripts", "mark_to_market.py")


def _run_mark_to_market(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.run(
        [sys.executable, script, *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def _copy_mark_to_market(tmp_path: Path) -> Path:
    """Byte-identical copy under a mock root plus a mock src/paths.py: DATA_DIR
    resolves inside the mock tree and resolve_runtime_public_data_dir returns a
    never-used constant (all cases pass explicit --prices)."""
    import shutil

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    shutil.copyfile(MARK_TO_MARKET, scripts_dir / "mark_to_market.py")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "paths.py").write_text(
        "from pathlib import Path\n"
        "DATA_DIR = Path(__file__).resolve().parent.parent / 'data'\n"
        "def resolve_runtime_public_data_dir(*args, **kwargs):\n"
        "    return Path('/tmp/q17-nonexistent-public-data')\n",
        encoding="utf-8",
    )
    return scripts_dir / "mark_to_market.py"


def _write_mtm_prices(tmp_path: Path) -> Path:
    path = tmp_path / "prices.json"
    path.write_text(
        json.dumps(
            {
                "SPY": [{"d": "2026-08-14", "p": 450.5}],
                "GLD": [{"d": "2026-08-14", "p": 210.0}],
                "TLT": [{"d": "2026-08-14", "p": 95.0}],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_mtm_paper_portfolio(tmp_path: Path) -> Path:
    """Mock portfolio in the mock tree's data dir (the copy's DATA_DIR)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    portfolio = data_dir / "portfolio_paper.json"
    portfolio.write_text(
        json.dumps(
            {
                "mode": "paper",
                "cash": 10000.0,
                "positions": {
                    "SPY": {"shares": 10, "avg_price": 440.0, "value": 4400.0}
                },
                "history": [],
            }
        ),
        encoding="utf-8",
    )
    return portfolio


def test_mark_to_market_help_exits_0(tmp_path) -> None:
    """MARK-TO-MARKET-SMOKE: --help → exit 0 with the description marker."""
    copy = _copy_mark_to_market(tmp_path)
    res = _run_mark_to_market(str(copy), "--help")
    assert res.returncode == 0
    assert "Mark portfolio to market" in res.stdout


def test_mark_to_market_missing_portfolio_exits_1(tmp_path) -> None:
    """MARK-TO-MARKET-SMOKE: explicit prices but no mock data/portfolio
    → exit 1 with the missing-portfolio marker (nothing written)."""
    copy = _copy_mark_to_market(tmp_path)
    prices = _write_mtm_prices(tmp_path)
    res = _run_mark_to_market(str(copy), "--prices", str(prices))
    assert res.returncode == 1
    assert "ERROR: No portfolio_paper.json found" in (res.stdout + res.stderr)
    assert not (tmp_path / "data" / "portfolio_paper.json").exists()


def test_mark_to_market_mock_fixture_updates_portfolio(tmp_path) -> None:
    """MARK-TO-MARKET-SMOKE: mock portfolio + prices in the mock tree → exit 0
    with the after-value marker and the updated portfolio JSON written."""
    copy = _copy_mark_to_market(tmp_path)
    prices = _write_mtm_prices(tmp_path)
    _ = _write_mtm_paper_portfolio(tmp_path)
    res = _run_mark_to_market(str(copy), "--prices", str(prices))
    assert res.returncode == 0, res.stderr
    assert "Portfolio value (after):" in res.stdout
    updated = tmp_path / "data" / "portfolio_paper.json"
    assert updated.exists()
    payload = json.loads(updated.read_text(encoding="utf-8"))
    assert payload["positions"]["SPY"]["current_price"] == 450.5
    assert payload["positions"]["SPY"]["value"] == 4505.0
    assert len(payload["history"]) == 1


# CAPTURE-DAILY-PNL-SMOKE (Item Q18, 2026-08-17): subprocess smoke for the
# daily P&L capture CLI (scripts/capture_daily_pnl.py). The script reads and
# writes daily_pnl.jsonl / daily_pnl_latest.json under DATA_DIR with no env
# override, so every case runs a byte-identical copy under a mock root whose
# minimal src package points DATA_DIR at the mock tree AND stubs the real
# src.strategy.evaluator module (the copy imports PAPER_CONFIG at module load
# — Q13/Q17 copy pattern + evaluator stub keeps the subprocess hermetic).
# Repo data/daily_pnl.jsonl and daily_pnl_latest.json are never touched.
CAPTURE_DAILY_PNL = os.path.join("scripts", "capture_daily_pnl.py")


def _run_capture_daily_pnl(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.run(
        [sys.executable, script, *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def _copy_capture_daily_pnl(tmp_path: Path) -> Path:
    """Byte-identical copy under a mock root plus a minimal mock src package:
    paths.DATA_DIR inside the mock tree, strategy.evaluator.PAPER_CONFIG stub,
    and utils.log_config.configure_logging stub (avoids importing the real
    heavy evaluator/log-config modules in the subprocess)."""
    import shutil

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    shutil.copyfile(CAPTURE_DAILY_PNL, scripts_dir / "capture_daily_pnl.py")
    (tmp_path / "src" / "strategy").mkdir(parents=True)
    (tmp_path / "src" / "utils").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "strategy" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "utils" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "utils" / "log_config.py").write_text(
        "import logging\n"
        "def configure_logging(level=None):\n"
        "    logging.basicConfig(level=logging.INFO)\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "paths.py").write_text(
        "from pathlib import Path\n"
        "DATA_DIR = Path(__file__).resolve().parent.parent / 'data'\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "strategy" / "evaluator.py").write_text(
        "PAPER_CONFIG = {'initial_capital': 100000}\n",
        encoding="utf-8",
    )
    return scripts_dir / "capture_daily_pnl.py"


def test_capture_daily_pnl_help_exits_0(tmp_path) -> None:
    """CAPTURE-DAILY-PNL-SMOKE: --help → exit 0 with the description marker."""
    copy = _copy_capture_daily_pnl(tmp_path)
    res = _run_capture_daily_pnl(str(copy), "--help")
    assert res.returncode == 0
    assert "Capture daily P&L snapshot" in res.stdout


def test_capture_daily_pnl_missing_portfolio_exits_1(tmp_path) -> None:
    """CAPTURE-DAILY-PNL-SMOKE: no mock data/portfolio_*.json → exit 1 with
    the missing-portfolio marker (no snapshot written)."""
    copy = _copy_capture_daily_pnl(tmp_path)
    res = _run_capture_daily_pnl(str(copy))
    assert res.returncode == 1
    assert "No portfolio_paper.json found" in (res.stdout + res.stderr)
    assert not (tmp_path / "data" / "daily_pnl.jsonl").exists()


def test_capture_daily_pnl_backfill_dry_run(tmp_path) -> None:
    """CAPTURE-DAILY-PNL-SMOKE: --backfill-returns --dry-run on a mock
    daily_pnl.jsonl → exit 0 with the backfill summary marker, and the mock
    fixture stays byte-unchanged (dry-run never writes)."""
    copy = _copy_capture_daily_pnl(tmp_path)
    daily_pnl = tmp_path / "data" / "daily_pnl.jsonl"
    daily_pnl.parent.mkdir(parents=True)
    rows = [
        {"date": "2026-08-13", "total_value": 100000.0, "daily_return": 0.0},
        {"date": "2026-08-14", "total_value": 101000.0, "daily_return": 0.0},
    ]
    daily_pnl.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    before = daily_pnl.read_bytes()
    res = _run_capture_daily_pnl(str(copy), "--backfill-returns", "--dry-run")
    assert res.returncode == 0, res.stderr
    assert "backfill_daily_returns" in (res.stdout + res.stderr)
    assert daily_pnl.read_bytes() == before, "dry-run rewrote daily_pnl.jsonl"


# REBUILD-PRICES-COMPACT-SMOKE (Item Q19, 2026-08-17): subprocess smoke for
# the prices_compact rebuild CLI (scripts/rebuild_prices_compact.py). The
# script loads prices + writes compact targets via DATA_DIR / PUBLIC_DATA_DIR
# with no env override, so every case runs a byte-identical copy under a mock
# root whose minimal src/paths.py points both dirs inside the mock tree (Q13
# copy pattern). Repo data/prices_compact.json and public/data/prices_compact
# .json are never touched.
REBUILD_PRICES_COMPACT = os.path.join("scripts", "rebuild_prices_compact.py")


def _run_rebuild_prices_compact(
    script: str, *args: str
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    env["PRICES_COMPACT_N_BARS"] = "2"
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.run(
        [sys.executable, script, *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def _copy_rebuild_prices_compact(tmp_path: Path) -> Path:
    """Byte-identical copy under a mock root plus a minimal mock src/paths.py:
    DATA_DIR and PUBLIC_DATA_DIR both resolve inside the mock tree, so the copy
    never touches repo data/ or public/data/."""
    import shutil

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    shutil.copyfile(REBUILD_PRICES_COMPACT, scripts_dir / "rebuild_prices_compact.py")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "paths.py").write_text(
        "from pathlib import Path\n"
        "DATA_DIR = Path(__file__).resolve().parent.parent / 'data'\n"
        "PUBLIC_DATA_DIR = Path(__file__).resolve().parent.parent / 'public' / 'data'\n",
        encoding="utf-8",
    )
    return scripts_dir / "rebuild_prices_compact.py"


def test_rebuild_prices_compact_missing_prices_exits_1(tmp_path) -> None:
    """REBUILD-PRICES-COMPACT-SMOKE: no prices.json in any candidate path
    → exit 1 with the missing-source marker (nothing written)."""
    copy = _copy_rebuild_prices_compact(tmp_path)
    res = _run_rebuild_prices_compact(str(copy))
    assert res.returncode == 1
    assert "No prices.json found" in res.stderr
    assert not (tmp_path / "data" / "prices_compact.json").exists()


def test_rebuild_prices_compact_non_dict_payload_exits_1(tmp_path) -> None:
    """REBUILD-PRICES-COMPACT-SMOKE: prices.json holds a non-object payload
    → exit 1 with the must-be-object marker (nothing written)."""
    copy = _copy_rebuild_prices_compact(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "prices.json").write_text('["not", "a", "dict"]', encoding="utf-8")
    res = _run_rebuild_prices_compact(str(copy))
    assert res.returncode == 1
    assert "prices.json must be object" in res.stderr
    assert not (data_dir / "prices_compact.json").exists()


def test_rebuild_prices_compact_mock_fixture_exits_0(tmp_path) -> None:
    """REBUILD-PRICES-COMPACT-SMOKE: mock prices fixture in the mock tree
    → exit 0 with JSON {"ok": true, n_symbols, n_bars, written} and compact
    files written, honoring last-N truncation (PRICES_COMPACT_N_BARS=2)."""
    copy = _copy_rebuild_prices_compact(tmp_path)
    public_dir = tmp_path / "public" / "data"
    public_dir.mkdir(parents=True)
    (public_dir / "prices.json").write_text(
        json.dumps(
            {
                "SPY": [
                    {"d": "2026-08-12", "p": 440.0},
                    {"d": "2026-08-13", "p": 445.0},
                    {"d": "2026-08-14", "p": 450.5},
                ],
                "GLD": {"bars": [{"d": "2026-08-14", "p": 210.0}]},
                "meta": {"schema": "x"},
                "_private": [1, 2, 3],
            }
        ),
        encoding="utf-8",
    )
    res = _run_rebuild_prices_compact(str(copy))
    assert res.returncode == 0, res.stdout + res.stderr
    payload = json.loads(res.stdout)
    assert payload["ok"] is True
    assert payload["n_bars"] == 2
    assert payload["n_symbols"] == 2
    assert payload["written"]
    assert (public_dir / "prices_compact.json").exists()
    assert (tmp_path / "data" / "prices_compact.json").exists()
    compact = json.loads((public_dir / "prices_compact.json").read_text(encoding="utf-8"))
    assert compact["meta"]["schema"] == "prices/compact-v1"
    assert compact["meta"]["n_bars"] == 2
    assert len(compact["symbols"]["SPY"]) == 2  # last-N truncated
    assert len(compact["symbols"]["GLD"]) == 1


# RECOVERY-CLI-SMOKE (Item Q20, 2026-08-17): subprocess smoke for the
# recovery CLI (scripts/portfolio_lab_recovery.py). The four cases are
# side-effect-free (help/argparse validation/verify fail-closed before any
# system access), so the real script path is hermetic — no mock root copy
# needed and live system trees are never touched.
RECOVERY_CLI = os.path.join("scripts", "portfolio_lab_recovery.py")


def _run_recovery_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.run(
        [sys.executable, RECOVERY_CLI, *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def test_recovery_cli_help_exits_0() -> None:
    """RECOVERY-CLI-SMOKE: --help → exit 0 with the description marker."""
    res = _run_recovery_cli("--help")
    assert res.returncode == 0
    assert "Portfolio Lab recovery:" in res.stdout


def test_recovery_cli_missing_subcommand_exits_2() -> None:
    """RECOVERY-CLI-SMOKE: bare invocation → argparse exit 2 with the
    required-command marker on stderr."""
    res = _run_recovery_cli()
    assert res.returncode == 2
    assert "required: command" in res.stderr


def test_recovery_cli_verify_relative_archive_exits_1() -> None:
    """RECOVERY-CLI-SMOKE: verify with a relative --archive → exit 1 with
    the absolute-path marker on stderr (nothing inspected)."""
    res = _run_recovery_cli("verify", "--archive", "backups/x.tar")
    assert res.returncode == 1
    assert "verify --archive must be an absolute path" in res.stderr


def test_recovery_cli_verify_missing_archive_exits_1(tmp_path) -> None:
    """RECOVERY-CLI-SMOKE: verify with a nonexistent absolute --archive
    → exit 1 with stdout JSON {"ok": false, ...} error "archive file
    missing" (fails closed before any extraction)."""
    missing = tmp_path / "nonexistent.portfolio-lab-recovery.tar"
    res = _run_recovery_cli("verify", "--archive", str(missing))
    assert res.returncode == 1
    payload = json.loads(res.stdout)
    assert payload["ok"] is False
    assert "archive file missing" in payload["error"]


# VIX-TERM-STRUCTURE-SMOKE (Item Q21, 2026-08-17): subprocess smoke for the
# VIX term-structure updater CLI (scripts/update_vix_term_structure.py). The
# script has no CLI args and writes via DATA_DIR / PUBLIC_DATA_DIR (no env
# override), so every case runs a byte-identical copy under a mock root whose
# minimal src package points both dirs inside the mock tree and stubs
# src.data.vix_futures.VIXTermStructure (Q13/Q18 copy pattern; the mock src
# shadows repo src, so the stub is required). Repo data/vix_term_structure
# .json and public/data/vix_term_structure.json are never touched.
UPDATE_VIX_TS = os.path.join("scripts", "update_vix_term_structure.py")


def _run_update_vix_term_structure(script: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def _copy_update_vix_ts(tmp_path: Path) -> Path:
    """Byte-identical copy under a mock root plus a minimal mock src package:
    paths.DATA_DIR / PUBLIC_DATA_DIR inside the mock tree and a
    VIXTermStructure stub (the copy imports pandas and calls
    src.data.vix_futures.VIXTermStructure at runtime)."""
    import shutil

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    shutil.copyfile(UPDATE_VIX_TS, scripts_dir / "update_vix_term_structure.py")
    (tmp_path / "src" / "data").mkdir(parents=True)
    (tmp_path / "src" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "data" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "paths.py").write_text(
        "from pathlib import Path\n"
        "DATA_DIR = Path(__file__).resolve().parent.parent / 'data'\n"
        "PUBLIC_DATA_DIR = Path(__file__).resolve().parent.parent / 'public' / 'data'\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "data" / "vix_futures.py").write_text(
        "class VIXTermStructure:\n"
        "    def __init__(self, **kwargs):\n"
        "        self._data = kwargs\n"
        "    @classmethod\n"
        "    def from_dict(cls, raw):\n"
        "        return cls(**raw)\n"
        "    def to_dict(self):\n"
        "        return dict(self._data)\n",
        encoding="utf-8",
    )
    return scripts_dir / "update_vix_term_structure.py"


def _write_vix_market_db(data_dir: Path) -> None:
    """Hermetic market.db with ^VIX + ^VIX3M rows on the same dates."""
    import sqlite3

    data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(data_dir / "market.db"))
    conn.execute("CREATE TABLE prices (date TEXT, close REAL, symbol TEXT)")
    conn.executemany(
        "INSERT INTO prices (date, close, symbol) VALUES (?, ?, ?)",
        [
            ("2026-08-12", 15.0, "^VIX"),
            ("2026-08-13", 15.5, "^VIX"),
            ("2026-08-14", 16.0, "^VIX"),
            ("2026-08-12", 17.0, "^VIX3M"),
            ("2026-08-13", 17.2, "^VIX3M"),
            ("2026-08-14", 17.5, "^VIX3M"),
        ],
    )
    conn.commit()
    conn.close()


def test_update_vix_term_structure_missing_db_exits_1(tmp_path) -> None:
    """VIX-TERM-STRUCTURE-SMOKE: mock root without market.db → exit 1 with
    the missing-db and failed markers on stderr (nothing written)."""
    copy = _copy_update_vix_ts(tmp_path)
    res = _run_update_vix_term_structure(str(copy))
    assert res.returncode == 1
    assert "market.db not found" in res.stderr
    assert "Failed to update vix_term_structure.json" in res.stderr
    assert not (tmp_path / "data" / "vix_term_structure.json").exists()


def test_update_vix_term_structure_mock_db_writes_json(tmp_path) -> None:
    """VIX-TERM-STRUCTURE-SMOKE: hermetic market.db with ^VIX/^VIX3M rows
    → exit 0 with the success marker and vix_term_structure.json written in
    both mock data and public/data dirs."""
    copy = _copy_update_vix_ts(tmp_path)
    data_dir = tmp_path / "data"
    _write_vix_market_db(data_dir)
    res = _run_update_vix_term_structure(str(copy))
    assert res.returncode == 0, res.stderr
    assert "Successfully updated vix_term_structure.json" in res.stderr
    payload = json.loads(
        (data_dir / "vix_term_structure.json").read_text(encoding="utf-8")
    )
    assert payload["_meta"]["schema"] == "vix_term_structure/v1"
    assert payload["_meta"]["n_dates"] == 3
    assert payload["_meta"]["spot_source"] == "^VIX"
    assert payload["_meta"]["spot_is_proxy"] is False
    entry = payload["2026-08-14"]
    assert entry["vix_spot"] == 16.0
    assert entry["source"] == "market.db"
    assert (tmp_path / "public" / "data" / "vix_term_structure.json").exists()


# BENCHMARK-CRITICAL-PATHS-SMOKE (Item Q22, 2026-08-17): subprocess smoke for
# the critical-path benchmark CLI (scripts/benchmark_critical_paths.py). The
# run case needs the real src modules (the suite imports src.backtest /
# strategy / signals / dashboard internals), so the real script runs with all
# writable paths (--output / --baseline / --runtime-dir) pointed at tmp_path —
# no writes to live data/perf. Negative --runs raises ValueError in main
# (traceback on stderr → exit 1).
BENCHMARK_CP = os.path.join("scripts", "benchmark_critical_paths.py")


def _run_benchmark_cp(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.run(
        [sys.executable, BENCHMARK_CP, *args],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def test_benchmark_critical_paths_help_exits_0() -> None:
    """BENCHMARK-CRITICAL-PATHS-SMOKE: --help → exit 0 with the description
    marker."""
    res = _run_benchmark_cp("--help")
    assert res.returncode == 0
    assert "Benchmark portfolio-lab critical paths" in res.stdout


def test_benchmark_critical_paths_negative_runs_errors(tmp_path) -> None:
    """BENCHMARK-CRITICAL-PATHS-SMOKE: negative --runs → nonzero exit with
    the runs-guard marker on stderr. The script raises ValueError in main
    (exit 1 via traceback, not argparse exit 2), so the smoke pins the actual
    behavior; nothing is written."""
    output = tmp_path / "critical_paths_latest.json"
    res = _run_benchmark_cp("--runs", "-1", "--output", str(output))
    assert res.returncode == 1
    assert "--runs must be >= 1" in res.stderr
    assert not output.exists()


def test_benchmark_critical_paths_minimal_run_writes_json(tmp_path) -> None:
    """BENCHMARK-CRITICAL-PATHS-SMOKE: --runs 1 --warmup 0 with all writable
    paths in tmp_path → exit 0, "status": "ok" in stdout, and valid results
    JSON written to --output with the five benchmark cases (live data/perf
    untouched)."""
    output = tmp_path / "perf" / "critical_paths_latest.json"
    baseline = tmp_path / "perf" / "critical_paths_baseline.json"
    runtime = tmp_path / "runtime"
    res = _run_benchmark_cp(
        "--output",
        str(output),
        "--baseline",
        str(baseline),
        "--runtime-dir",
        str(runtime),
        "--runs",
        "1",
        "--warmup",
        "0",
    )
    assert res.returncode == 0, res.stderr
    assert '"status": "ok"' in res.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["runs"] == 1
    names = {case["name"] for case in payload["benchmarks"]}
    assert names == {
        "price_loading",
        "ensemble_compute_vote",
        "signal_correlation_matrix",
        "combined_regime_backtest_fixture",
        "dashboard_health_generation",
    }


# ARCHIVE-IC-PRE-CONTRACT-SMOKE (Item Q23, 2026-08-17): subprocess smoke for
# the IC pre-contract archive CLI (scripts/archive_ic_pre_contract_rows.py).
# The script loads/saves state via DATA_DIR with no env override, so every
# case runs a byte-identical copy under a mock root whose src package points
# DATA_DIR into the mock tree AND carries a byte-identical copy of the real
# src/monitor/ic_decay_monitor.py (its only src import is src.paths; numpy
# comes from the venv). Repo data/ic_monitor_state.json and
# data/ic_rebaseline_archives are never touched. NOTE: the script prints the
# returned archive path ("archived: <path>"), not a count.
ARCHIVE_IC = os.path.join("scripts", "archive_ic_pre_contract_rows.py")
IC_DECAY_MONITOR = os.path.join("src", "monitor", "ic_decay_monitor.py")


def _run_archive_ic(script: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def _copy_archive_ic(tmp_path: Path) -> Path:
    """Byte-identical copies of the script and the real ICMonitor module
    under a mock root whose src.paths points DATA_DIR into the mock tree."""
    import shutil

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    shutil.copyfile(ARCHIVE_IC, scripts_dir / "archive_ic_pre_contract_rows.py")
    (tmp_path / "src" / "monitor").mkdir(parents=True)
    (tmp_path / "src" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "monitor" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "paths.py").write_text(
        "from pathlib import Path\n"
        "DATA_DIR = Path(__file__).resolve().parent.parent / 'data'\n",
        encoding="utf-8",
    )
    shutil.copyfile(
        IC_DECAY_MONITOR, tmp_path / "src" / "monitor" / "ic_decay_monitor.py"
    )
    return scripts_dir / "archive_ic_pre_contract_rows.py"


def test_archive_ic_pre_contract_rows_empty_state(tmp_path) -> None:
    """ARCHIVE-IC-PRE-CONTRACT-SMOKE: mock root with no state file → exit 0,
    stdout "archived: <archive path>" with an empty snapshot written and the
    state file created as {} (nothing in the repo is touched)."""
    copy = _copy_archive_ic(tmp_path)
    res = _run_archive_ic(str(copy))
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip().startswith("archived: ")
    archive_path = Path(res.stdout.strip().split(" ", 1)[1])
    assert archive_path.is_file()
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    assert archive["schema"] == "ic-rebaseline-archive/v1"
    assert archive["observations"] == {}
    assert archive["staged"] == []
    saved = tmp_path / "data" / "ic_monitor_state.json"
    assert saved.exists()
    assert json.loads(saved.read_text(encoding="utf-8")) == {}


def test_archive_ic_pre_contract_rows_archives_misaligned(tmp_path) -> None:
    """ARCHIVE-IC-PRE-CONTRACT-SMOKE: populated state with a None-metadata
    ensemble_equity row → exit 0, that row archived to the snapshot and
    dropped from the re-saved state (aligned row kept)."""
    copy = _copy_archive_ic(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "ic_monitor_state.json").write_text(
        json.dumps(
            {
                "ensemble_equity": [[0.5, 0.01], [0.4, 0.02]],
                "__state_schema_version__": "ic-monitor-state/v2",
                "__observation_metadata__": {
                    "ensemble_equity": [
                        None,
                        {"prediction_field": "ensemble_voting.equity_bias"},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    res = _run_archive_ic(str(copy))
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip().startswith("archived: ")
    archives = list((data_dir / "ic_rebaseline_archives").glob("ic_pre_contract_archive_*.json"))
    assert len(archives) == 1
    archive = json.loads(archives[0].read_text(encoding="utf-8"))
    assert archive["archive_kind"] == "pre-contract-rows"
    assert archive["observations"]["ensemble_equity"] == [[0.5, 0.01]]
    assert archive["observation_metadata"]["ensemble_equity"] == [None]
    saved = json.loads((data_dir / "ic_monitor_state.json").read_text(encoding="utf-8"))
    assert saved["ensemble_equity"] == [[0.4, 0.02]]
    assert saved["__observation_metadata__"]["ensemble_equity"] == [
        {"prediction_field": "ensemble_voting.equity_bias"}
    ]


# FABER-SMA-GATE-SMOKE (Item Q24, 2026-08-17): subprocess smoke for the
# Faber 10-month SMA gate PoC CLI (scripts/faber_sma_gate.py). The script
# reads prices from a fixed path relative to __file__ (parents[1] /
# public/data/prices.json) with no env override and no src imports, so a
# byte-identical copy under a mock root points it at the mock tree (Q13 copy
# pattern). Live repo public/data/prices.json is never read.
FABER_SMA_GATE = os.path.join("scripts", "faber_sma_gate.py")


def _run_faber_sma_gate(script: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def _copy_faber_sma_gate(tmp_path: Path) -> Path:
    """Byte-identical copy under a mock root; DATA_FILE resolves to the mock
    tree's public/data/prices.json (no src package needed — stdlib + numpy/
    pandas only)."""
    import shutil

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    shutil.copyfile(FABER_SMA_GATE, scripts_dir / "faber_sma_gate.py")
    return scripts_dir / "faber_sma_gate.py"


def _write_faber_prices(public_dir: Path, n_days: int = 260) -> None:
    """Deterministic synthetic business-day prices for SPY/GLD/TLT (> SMA
    window so the 200-day gate is meaningful)."""
    import datetime
    import math

    public_dir.mkdir(parents=True, exist_ok=True)
    day = datetime.date(2025, 1, 2)
    dates: list[str] = []
    while len(dates) < n_days:
        if day.weekday() < 5:
            dates.append(day.isoformat())
        day += datetime.timedelta(days=1)
    payload = {}
    for i, sym in enumerate(["SPY", "GLD", "TLT"]):
        base = 100.0 + i * 20.0
        payload[sym] = [
            {
                "d": date,
                "p": round(
                    base * (1 + 0.0006 * idx + 0.01 * math.sin(idx / 19.0)), 4
                ),
            }
            for idx, date in enumerate(dates)
        ]
    (public_dir / "prices.json").write_text(json.dumps(payload), encoding="utf-8")


def test_faber_sma_gate_missing_prices_exits_1(tmp_path) -> None:
    """FABER-SMA-GATE-SMOKE: mock root without prices.json → exit 1 with
    FileNotFoundError on stderr (nothing else runs)."""
    copy = _copy_faber_sma_gate(tmp_path)
    res = _run_faber_sma_gate(str(copy))
    assert res.returncode == 1
    assert "FileNotFoundError" in res.stderr
    assert "prices.json" in res.stderr


def test_faber_sma_gate_fixture_run_exits_0(tmp_path) -> None:
    """FABER-SMA-GATE-SMOKE: hermetic mock prices fixture → exit 0 with the
    gate header and Sharpe-delta markers in stdout."""
    copy = _copy_faber_sma_gate(tmp_path)
    _write_faber_prices(tmp_path / "public" / "data")
    res = _run_faber_sma_gate(str(copy))
    assert res.returncode == 0, res.stderr
    assert "FABER 10-MONTH SMA GATE" in res.stdout
    assert "Sharpe delta" in res.stdout


# DONCHIAN-BREAKOUT-SMOKE (Item Q25, 2026-08-17): subprocess smoke for the
# Donchian channel breakout ensemble PoC CLI (scripts/donchian_breakout.py).
# The script reads prices from parents[1] / public/data/prices.json with no
# env override and no src imports (pandas + numpy only), so a byte-identical
# copy under a mock root points it at the mock tree (Q13/Q24 copy pattern).
# Live repo public/data/prices.json is never read.
DONCHIAN_BREAKOUT = os.path.join("scripts", "donchian_breakout.py")


def _run_donchian_breakout(script: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def _copy_donchian_breakout(tmp_path: Path) -> Path:
    """Byte-identical copy under a mock root; DATA_FILE resolves to the mock
    tree's public/data/prices.json."""
    import shutil

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    shutil.copyfile(DONCHIAN_BREAKOUT, scripts_dir / "donchian_breakout.py")
    return scripts_dir / "donchian_breakout.py"


def _write_donchian_prices(public_dir: Path, n_days: int = 260) -> None:
    """Deterministic synthetic business-day prices for SPY (>= 252d channel
    and 200d SMA window)."""
    import datetime
    import math

    public_dir.mkdir(parents=True, exist_ok=True)
    day = datetime.date(2025, 1, 2)
    dates: list[str] = []
    while len(dates) < n_days:
        if day.weekday() < 5:
            dates.append(day.isoformat())
        day += datetime.timedelta(days=1)
    payload = {
        "SPY": [
            {
                "d": date,
                "p": round(100.0 * (1 + 0.0008 * idx + 0.02 * math.sin(idx / 15.0)), 4),
            }
            for idx, date in enumerate(dates)
        ]
    }
    (public_dir / "prices.json").write_text(json.dumps(payload), encoding="utf-8")


def test_donchian_breakout_missing_prices_exits_1(tmp_path) -> None:
    """DONCHIAN-BREAKOUT-SMOKE: mock root without prices.json -> exit 1 with
    FileNotFoundError on stderr."""
    copy = _copy_donchian_breakout(tmp_path)
    res = _run_donchian_breakout(str(copy))
    assert res.returncode == 1
    assert "FileNotFoundError" in res.stderr
    assert "prices.json" in res.stderr


def test_donchian_breakout_fixture_run_exits_0(tmp_path) -> None:
    """DONCHIAN-BREAKOUT-SMOKE: hermetic mock SPY prices fixture -> exit 0 with
    breakout header and ensemble markers in stdout."""
    copy = _copy_donchian_breakout(tmp_path)
    _write_donchian_prices(tmp_path / "public" / "data")
    res = _run_donchian_breakout(str(copy))
    assert res.returncode == 0, res.stderr
    assert "DONCHIAN CHANNEL BREAKOUT ENSEMBLE" in res.stdout
    assert "Ensemble (20/55/252d)" in res.stdout


# VOL-ADJUSTED-MOMENTUM-SMOKE (Item Q26, 2026-08-17): subprocess smoke for the
# vol-adjusted momentum PoC CLI (scripts/vol_adjusted_momentum.py). The script
# reads prices from parents[1] / public/data/prices.json with no env override
# and no src imports (pandas + numpy only), so a byte-identical copy under a
# mock root points it at the mock tree (Q13/Q24/Q25 copy pattern). Live repo
# public/data/prices.json is never read.
VOL_ADJUSTED_MOMENTUM = os.path.join("scripts", "vol_adjusted_momentum.py")


def _run_vol_adjusted_momentum(script: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def _copy_vol_adjusted_momentum(tmp_path: Path) -> Path:
    """Byte-identical copy under a mock root; DATA_FILE resolves to the mock
    tree's public/data/prices.json."""
    import shutil

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    shutil.copyfile(
        VOL_ADJUSTED_MOMENTUM, scripts_dir / "vol_adjusted_momentum.py"
    )
    return scripts_dir / "vol_adjusted_momentum.py"


def _write_vol_momentum_prices(public_dir: Path, n_days: int = 280) -> None:
    """Deterministic synthetic business-day prices for SPY, GLD, TLT
    (>= 252d momentum window + 21d skip)."""
    import datetime
    import math

    public_dir.mkdir(parents=True, exist_ok=True)
    day = datetime.date(2025, 1, 2)
    dates: list[str] = []
    while len(dates) < n_days:
        if day.weekday() < 5:
            dates.append(day.isoformat())
        day += datetime.timedelta(days=1)
    payload = {}
    for i, sym in enumerate(["SPY", "GLD", "TLT"]):
        base = 100.0 + i * 25.0
        payload[sym] = [
            {
                "d": date,
                "p": round(
                    base * (1 + 0.0007 * idx + 0.015 * math.sin(idx / 17.0)), 4
                ),
            }
            for idx, date in enumerate(dates)
        ]
    (public_dir / "prices.json").write_text(json.dumps(payload), encoding="utf-8")


def test_vol_adjusted_momentum_missing_prices_exits_1(tmp_path) -> None:
    """VOL-ADJUSTED-MOMENTUM-SMOKE: mock root without prices.json -> exit 1 with
    FileNotFoundError on stderr."""
    copy = _copy_vol_adjusted_momentum(tmp_path)
    res = _run_vol_adjusted_momentum(str(copy))
    assert res.returncode == 1
    assert "FileNotFoundError" in res.stderr
    assert "prices.json" in res.stderr


def test_vol_adjusted_momentum_fixture_run_exits_0(tmp_path) -> None:
    """VOL-ADJUSTED-MOMENTUM-SMOKE: hermetic mock SPY/GLD/TLT prices fixture
    -> exit 0 with vol-adjusted momentum header and Sharpe delta markers."""
    copy = _copy_vol_adjusted_momentum(tmp_path)
    _write_vol_momentum_prices(tmp_path / "public" / "data")
    res = _run_vol_adjusted_momentum(str(copy))
    assert res.returncode == 0, res.stderr
    assert "VOL-ADJUSTED MOMENTUM" in res.stdout
    assert "Sharpe delta" in res.stdout


# GOLD-ALLOCATION-SWEEP-SMOKE (Item Q27, 2026-08-17): subprocess smoke for the
# gold allocation sensitivity sweep CLI (scripts/gold_allocation_sweep.py). The
# script imports from src.paths (PRICES_PATH = PRICES_JSON, OUTPUT_PATH =
# DATA_DIR / ...), src.backtest.metrics, and src.utils.log_config, resolving
# PROJECT_ROOT via __file__.parents[1]. A byte-identical copy under a mock root
# whose mock src package points DATA_DIR/PRICES_JSON into tmp and exposes the
# required backtest metric helpers isolates the run completely: live
# data/gold_allocation_sweep_2026.json and public/data/prices.json are never
# touched.
GOLD_ALLOCATION_SWEEP = os.path.join("scripts", "gold_allocation_sweep.py")


def _run_gold_allocation_sweep(
    script: str, *args: str
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.run(
        [sys.executable, script, *args],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def _copy_gold_allocation_sweep(tmp_path: Path) -> Path:
    """Byte-identical copy under a mock root plus a mock src package:
    src/paths.py (DATA_DIR + PRICES_JSON in tmp_path),
    src/utils/log_config.py (configure_logging stub),
    and src/backtest/metrics.py (delegating compute_metrics, compute_crisis_returns,
    save_results_json, BacktestMetrics dataclass, TRADING_DAYS_PER_YEAR)."""
    import shutil

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    shutil.copyfile(
        GOLD_ALLOCATION_SWEEP, scripts_dir / "gold_allocation_sweep.py"
    )

    src_dir = tmp_path / "src"
    (src_dir / "backtest").mkdir(parents=True)
    (src_dir / "utils").mkdir(parents=True)
    (src_dir / "__init__.py").write_text("", encoding="utf-8")
    (src_dir / "backtest" / "__init__.py").write_text("", encoding="utf-8")
    (src_dir / "utils" / "__init__.py").write_text("", encoding="utf-8")

    (src_dir / "paths.py").write_text(
        "from pathlib import Path\n"
        "DATA_DIR = Path(__file__).resolve().parent.parent / 'data'\n"
        "DATA_DIR.mkdir(parents=True, exist_ok=True)\n"
        "PRICES_JSON = Path(__file__).resolve().parent.parent / 'public' / 'data' / 'prices.json'\n",
        encoding="utf-8",
    )
    (src_dir / "utils" / "log_config.py").write_text(
        "import logging\n"
        "def configure_logging(*args, **kwargs):\n"
        "    logging.basicConfig(level=logging.INFO)\n",
        encoding="utf-8",
    )
    (src_dir / "backtest" / "metrics.py").write_text(
        "from dataclasses import dataclass\n"
        "import json\n"
        "TRADING_DAYS_PER_YEAR = 252\n"
        "@dataclass\n"
        "class BacktestMetrics:\n"
        "    cagr: float = 0.08\n"
        "    sharpe_ratio: float = 0.79\n"
        "    volatility: float = 0.10\n"
        "    max_drawdown: float = 0.12\n"
        "def compute_metrics(equity_curve, initial_capital=1.0):\n"
        "    return BacktestMetrics()\n"
        "def compute_crisis_returns(prices, trading_days, crisis_years, equity_curve=None):\n"
        "    return {'2008': -5.0, '2020': 2.0, '2022': -10.0}\n"
        "def save_results_json(data, output_path):\n"
        "    with open(output_path, 'w') as f:\n"
        "        json.dump(data, f, indent=2)\n",
        encoding="utf-8",
    )
    return scripts_dir / "gold_allocation_sweep.py"


def _write_gold_sweep_prices(public_dir: Path, n_days: int = 260) -> None:
    """Deterministic synthetic business-day prices for SPY, GLD, TLT, IEF."""
    import datetime
    import math

    public_dir.mkdir(parents=True, exist_ok=True)
    day = datetime.date(2025, 1, 2)
    dates: list[str] = []
    while len(dates) < n_days:
        if day.weekday() < 5:
            dates.append(day.isoformat())
        day += datetime.timedelta(days=1)
    payload = {}
    for i, sym in enumerate(["SPY", "GLD", "TLT", "IEF"]):
        base = 100.0 + i * 20.0
        payload[sym] = [
            {
                "d": date,
                "p": round(
                    base * (1 + 0.0006 * idx + 0.01 * math.sin(idx / 19.0)), 4
                ),
            }
            for idx, date in enumerate(dates)
        ]
    (public_dir / "prices.json").write_text(json.dumps(payload), encoding="utf-8")


def test_gold_allocation_sweep_missing_prices_exits_1(tmp_path) -> None:
    """GOLD-ALLOCATION-SWEEP-SMOKE: mock root without prices.json -> exit 1
    with FileNotFoundError on stderr."""
    copy = _copy_gold_allocation_sweep(tmp_path)
    res = _run_gold_allocation_sweep(str(copy))
    assert res.returncode == 1
    assert "FileNotFoundError" in res.stderr
    assert "prices.json" in res.stderr


def test_gold_allocation_sweep_fixture_run_exits_0(tmp_path) -> None:
    """GOLD-ALLOCATION-SWEEP-SMOKE: hermetic mock SPY/GLD/TLT/IEF prices fixture
    -> exit 0 with summary headers and results JSON written to the mock output."""
    copy = _copy_gold_allocation_sweep(tmp_path)
    _write_gold_sweep_prices(tmp_path / "public" / "data")
    res = _run_gold_allocation_sweep(str(copy))
    assert res.returncode == 0, res.stderr
    assert "GOLD ALLOCATION SENSITIVITY SWEEP" in (res.stdout + res.stderr)
    assert "TOP 20 CONFIGURATIONS" in res.stdout
    out_file = tmp_path / "data" / "gold_allocation_sweep_2026.json"
    assert out_file.exists()
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert "metadata" in payload
    assert "all_results" in payload
    assert payload["metadata"]["total_configs"] > 0


# OPTIMIZE-PORTFOLIO-SMOKE (Item Q28, 2026-08-17): subprocess smoke for the
# PyPortfolioOpt optimizer CLI (scripts/optimize_portfolio.py). The script
# loads PRICES_JSON and writes DATA_DIR / optimized_weights.json via src.paths.
# A byte-identical copy under a mock root whose mock src package points
# DATA_DIR/PRICES_JSON into tmp and stubs save_optimizer_labs_output keeps the
# run completely hermetic: repo data/optimized_weights.json and public/data/prices.json
# are never touched.
OPTIMIZE_PORTFOLIO = os.path.join("scripts", "optimize_portfolio.py")


def _run_optimize_portfolio(
    script: str, *args: str
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.run(
        [sys.executable, script, *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def _copy_optimize_portfolio(tmp_path: Path) -> Path:
    """Byte-identical copy under a mock root plus a mock src package:
    src/paths.py (DATA_DIR + PRICES_JSON in tmp_path) and
    src/research/optimizer_labs_contract.py (save_optimizer_labs_output stub)."""
    import shutil

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    shutil.copyfile(
        OPTIMIZE_PORTFOLIO, scripts_dir / "optimize_portfolio.py"
    )

    src_dir = tmp_path / "src"
    (src_dir / "research").mkdir(parents=True)
    (src_dir / "__init__.py").write_text("", encoding="utf-8")
    (src_dir / "research" / "__init__.py").write_text("", encoding="utf-8")

    (src_dir / "paths.py").write_text(
        "from pathlib import Path\n"
        "DATA_DIR = Path(__file__).resolve().parent.parent / 'data'\n"
        "DATA_DIR.mkdir(parents=True, exist_ok=True)\n"
        "PRICES_JSON = Path(__file__).resolve().parent.parent / 'public' / 'data' / 'prices.json'\n",
        encoding="utf-8",
    )
    (src_dir / "research" / "optimizer_labs_contract.py").write_text(
        "import json\n"
        "def save_optimizer_labs_output(results, output_path, symbols, target_vol=0.10):\n"
        "    with open(output_path, 'w') as f:\n"
        "        json.dump({'results': results, 'symbols': symbols, 'target_vol': target_vol}, f, indent=2)\n",
        encoding="utf-8",
    )
    return scripts_dir / "optimize_portfolio.py"


def _write_optimizer_prices(public_dir: Path, n_days: int = 120) -> None:
    """Deterministic synthetic business-day prices for SPY, GLD, TLT."""
    import datetime
    import math

    public_dir.mkdir(parents=True, exist_ok=True)
    day = datetime.date(2025, 1, 2)
    dates: list[str] = []
    while len(dates) < n_days:
        if day.weekday() < 5:
            dates.append(day.isoformat())
        day += datetime.timedelta(days=1)
    payload = {}
    for i, sym in enumerate(["SPY", "GLD", "TLT"]):
        base = 100.0 + i * 20.0
        payload[sym] = [
            {
                "d": date,
                "p": round(
                    base * (1 + 0.0005 * idx + 0.01 * math.sin(idx / 11.0 + i)), 4
                ),
            }
            for idx, date in enumerate(dates)
        ]
    (public_dir / "prices.json").write_text(json.dumps(payload), encoding="utf-8")


def test_optimize_portfolio_help_exits_0(tmp_path) -> None:
    """OPTIMIZE-PORTFOLIO-SMOKE: --help -> exit 0 with description marker."""
    copy = _copy_optimize_portfolio(tmp_path)
    res = _run_optimize_portfolio(str(copy), "--help")
    assert res.returncode == 0
    assert "Portfolio optimization via PyPortfolioOpt" in res.stdout


def test_optimize_portfolio_missing_prices_exits_0_with_error_marker(tmp_path) -> None:
    """OPTIMIZE-PORTFOLIO-SMOKE: mock root without prices.json -> exit 0
    with 'Prices file not found' logged (fail-closed, no exception raised)."""
    copy = _copy_optimize_portfolio(tmp_path)
    res = _run_optimize_portfolio(str(copy))
    assert res.returncode == 0
    assert "Prices file not found" in (res.stdout + res.stderr)


def test_optimize_portfolio_fixture_run_with_save_exits_0(tmp_path) -> None:
    """OPTIMIZE-PORTFOLIO-SMOKE: hermetic mock SPY/GLD/TLT prices fixture with --save
    -> exit 0 with optimization headers and mock data/optimized_weights.json written."""
    copy = _copy_optimize_portfolio(tmp_path)
    _write_optimizer_prices(tmp_path / "public" / "data")
    res = _run_optimize_portfolio(str(copy), "--save")
    assert res.returncode == 0, res.stderr
    assert "Portfolio Optimization Results" in res.stdout
    assert "MAX_SHARPE" in res.stdout
    out_file = tmp_path / "data" / "optimized_weights.json"
    assert out_file.exists()
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert "results" in payload
    assert "symbols" in payload
    assert "max_sharpe" in payload["results"]




