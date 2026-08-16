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
