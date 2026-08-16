import importlib.util
import hashlib
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _load_consistency_checker():
    script_path = PROJECT_ROOT / "scripts" / "check_public_data_consistency.py"
    spec = importlib.util.spec_from_file_location("check_public_data_consistency", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_public_data_artifacts(
    app_dir: Path,
    *,
    source_generated_at: str = "2026-06-12T09:05:25.028Z",
    index_generated_at: str = "2026-06-12T09:06:00+00:00",
) -> None:
    source_manifest = {
        "schema_version": "market-data-source-manifest/v1",
        "generated_at": source_generated_at,
        "artifacts": [],
    }
    source_manifest_bytes = json.dumps(source_manifest, sort_keys=True).encode()
    source_manifest_hash = hashlib.sha256(source_manifest_bytes).hexdigest()
    artifacts = {
        "source_manifest.json": source_manifest,
        "index.json": {
            "schema_version": "public-data-index/v1",
            "generated_at": index_generated_at,
            "source_manifest": {
                "path": "source_manifest.json",
                "schema_version": "market-data-source-manifest/v1",
                "generated_at": source_generated_at,
                "sha256": source_manifest_hash,
            },
            "entries": [],
        },
        "health.json": {
            "status": "ok",
            "generated_at": index_generated_at,
        },
    }
    for root_name in ("public/data", "dist/data"):
        for filename, payload in artifacts.items():
            _write_json(app_dir / root_name / filename, payload)


def test_lab_deploy_script_is_host_native_lxc_caddy_default():
    source = _read("scripts/deploy-lab-app.sh")
    lower = source.lower()

    assert "docker compose" not in lower
    assert "cloudflare" not in lower
    assert "--setup-dns" not in source
    assert "--dns-only" not in source
    assert "systemctl" in source
    assert "caddy validate" in source
    assert "portfolio-lab-update" in source
    assert "/etc/caddy/Caddyfile" in source


def test_lab_env_keeps_dns_out_of_scope_and_uses_host_native_paths():
    source = _read("config/lab-app.env")

    assert "LAB_DNS" not in source
    assert "CLOUDFLARE" not in source
    assert "DASHBOARD_HTTP_PORT" not in source
    assert "DASHBOARD_HTTPS_PORT" not in source
    assert "PORTFOLIO_LAB_APP_DIR" in source
    assert "PORTFOLIO_LAB_WEB_ROOT" in source
    assert "TASKER_SERVICE_NAME" in source
    assert "CADDY_CONFIG" in source


def test_repo_caddyfile_defaults_to_host_native_paths_with_docker_overrides_available():
    source = _read("Caddyfile")

    assert "{$PORTFOLIO_LAB_TASKER_UPSTREAM:127.0.0.1:8000}" in source
    assert "{$PORTFOLIO_LAB_PUBLIC_ROOT:/var/www/portfolio-lab}" in source
    assert "{$PORTFOLIO_LAB_WEB_ROOT:/var/www/portfolio-lab}" in source
    assert "pipeline:8000" not in source
    assert "/srv/app" not in source


def test_lab_deploy_generated_caddy_block_preserves_cache_policy():
    source = _read("scripts/deploy-lab-app.sh")

    assert 'handle /assets/*' in source
    assert 'Cache-Control "public, max-age=31536000, immutable"' in source
    assert 'handle /data/*' in source
    assert 'Cache-Control "no-cache"' in source


def test_lab_deploy_refreshes_dashboard_data_before_build():
    source = _read("scripts/deploy-lab-app.sh")
    main_body = source.split("main() {", 1)[1]

    assert "--skip-data" in source
    assert "bun run fetch-data" in source
    assert main_body.index("refresh_dashboard_data") < main_body.index("build_frontend")
    assert main_body.index("refresh_dashboard_data") < main_body.index("publish_dist")


def test_lab_deploy_builds_verifies_and_publishes_static_release():
    source = _read("scripts/deploy-lab-app.sh")
    main_body = source.split("main() {", 1)[1]
    main_body = main_body.split("}", 1)[0]

    assert "--release-dir" in source
    assert "scripts/build_lab_release.py" in source
    assert "scripts/verify_lab_release.py" in source
    assert "Building verified static release" in source
    assert "Publishing verified static app" in source
    assert main_body.index("build_frontend") < main_body.index("verify_static_release")
    assert main_body.index("verify_static_release") < main_body.index("publish_dist")


def test_lab_deploy_preserves_mutable_data_when_publishing_static_release():
    source = _read("scripts/deploy-lab-app.sh")

    assert "--exclude='/data/'" in source
    assert "--exclude='/data/**'" in source
    assert "--exclude='./data'" in source


def test_lab_deploy_checks_public_data_consistency_after_live_mirror_before_publish():
    source = _read("scripts/deploy-lab-app.sh")
    # Isolate the main() call sequence (after the last function definition),
    # so .index() resolves call sites rather than earlier function defs.
    main_body = source.split("main() {", 1)[1]
    main_body = main_body.split("}", 1)[0]

    assert "check_public_data_consistency" in source
    assert "--skip-dist-data-match" in source
    # Immutable release identity excludes /data/**; deploy validates the live
    # public data tree in place and verifies static release bytes separately.
    assert main_body.index("mirror_repo_public_data_from_live") < main_body.index("build_frontend")
    assert main_body.index("build_frontend") < main_body.index("check_public_data_consistency")
    assert main_body.index("check_public_data_consistency") < main_body.index("verify_static_release")
    assert main_body.index("verify_static_release") < main_body.index("publish_dist")
    assert '--public-dir "${PUBLIC_ROOT}/data"' in source
    assert "--allow-repo-public-data" not in source


def test_lab_deploy_warns_that_skip_data_uses_existing_artifact_consistency_gate():
    source = _read("scripts/deploy-lab-app.sh")

    assert "--skip-data set; validating existing public/data artifacts" in source


def test_public_data_consistency_checker_accepts_matching_public_and_dist_data(tmp_path: Path):
    checker = _load_consistency_checker()
    _write_public_data_artifacts(tmp_path)

    result = checker.check_public_data_consistency(tmp_path, env={}, allow_repo_public_data=True)

    assert result.ok is True, result.errors


def test_public_data_consistency_checker_rejects_stale_public_index(tmp_path: Path):
    checker = _load_consistency_checker()
    _write_public_data_artifacts(
        tmp_path,
        source_generated_at="2026-06-12T09:05:25.028Z",
        index_generated_at="2026-06-12T03:12:34.220521+00:00",
    )

    result = checker.check_public_data_consistency(tmp_path, env={}, allow_repo_public_data=True)

    assert result.ok is False
    assert any("public/data/index.json is older than source_manifest.json" in error for error in result.errors)


def test_public_data_consistency_checker_rejects_dist_data_copy_mismatch(tmp_path: Path):
    checker = _load_consistency_checker()
    _write_public_data_artifacts(tmp_path)
    _write_json(tmp_path / "dist/data/health.json", {"status": "critical"})

    result = checker.check_public_data_consistency(tmp_path, env={}, allow_repo_public_data=True)

    assert result.ok is False
    assert any("dist/data/health.json does not match public/data/health.json" in error for error in result.errors)


def test_lab_deploy_runs_fred_readiness_precheck_before_refreshing_data():
    source = _read("scripts/deploy-lab-app.sh")
    main_body = source.split("main() {", 1)[1]

    assert "check_fred_readiness" in source
    assert "PORTFOLIO_LAB_MODE=lab" in source
    assert "python_runtime.sh -m src.data.fred_readiness" in source
    assert main_body.index("check_fred_readiness") < main_body.index("refresh_dashboard_data")


def test_lab_deploy_docs_and_makefile_do_not_present_dns_or_docker_as_default():
    docs = _read("scripts/LAB_APP_DEPLOY.md")
    makefile = _read("Makefile")
    combined = f"{docs}\n{makefile}".lower()

    assert "cloudflare" not in combined
    assert "setup-dns" not in combined
    assert "docker compose" not in combined
    assert "dns setup" not in combined
    assert "caddy" in combined
    assert "portfolio-lab-update" in combined


def test_makefile_exposes_offline_data_quality_target():
    makefile = _read("Makefile")

    assert "make data-quality" in makefile
    assert ".PHONY: data-quality" in makefile
    assert "scripts/check_public_data_quality.py" in makefile
    assert "$(PYTHON_RUNTIME) scripts/check_public_data_quality.py --app-dir $(PROJECT_DIR)" in makefile


# ── Task 1B/3B: deploy unit semantics — safe systemd kill/timeout + drain ─

def _tasker_unit_block(source: str) -> str:
    marker = 'cat > "$unit_path" <<EOF'
    assert marker in source
    return source.split(marker, 1)[1].split("EOF", 1)[0]


def test_tasker_unit_keeps_safe_kill_semantics():
    """The deployed systemd unit keeps control-group containment and bounded
    shutdown: never KillMode=process/none, SIGTERM first, TimeoutStopSec bounded."""
    source = _read("scripts/deploy-lab-app.sh")
    unit_block = _tasker_unit_block(source)

    assert "KillMode=process" not in source
    assert "KillMode=none" not in source
    assert "Type=simple" in unit_block
    assert "KillSignal=SIGTERM" in unit_block
    assert "TimeoutStopSec=30" in unit_block
    assert "Restart=always" in unit_block
    # The service must be restarted by deploy (never --skip-service for this
    # release: the drain + truth semantics are part of the deployed unit).
    assert "--skip-service" in source  # flag exists for operator escape hatch
    assert "systemctl restart" in source


def test_service_main_installs_signal_handlers_and_drains():
    """Service main installs SIGTERM/SIGINT and drains before exit."""
    source = _read("src/tasker/service.py")

    assert "signal.signal(signal.SIGTERM" in source
    assert "signal.signal(signal.SIGINT" in source
    assert "service.drain(" in source
    assert "termination_cause=\"service_restart\"" in source
    assert "write_status_mirrors" in source


def test_runner_and_store_expose_named_termination_cause():
    """Planned causes are persisted and never counted as failures."""
    store_source = _read("src/tasker/store.py")
    runner_source = _read("src/tasker/runner.py")

    assert "termination_cause" in store_source
    assert "PLANNED_TERMINATION_CAUSES" in store_source
    assert "drain_active_runs" in runner_source
    assert "termination_cause=\"service_restart\"" in runner_source or "termination_cause=termination_cause" in runner_source


def test_health_check_exposes_publication_and_probe_exit_modes():
    """Health producer default exit reflects completion; probe keeps severity."""
    source = _read("src/monitor/health_check.py")

    assert "--exit-mode" in source
    assert '"publication"' in source
    assert '"probe"' in source
    assert "PORTFOLIO_LAB_HEALTH_EXIT_MODE" in source


def test_generation_publication_helpers_exist():
    """Committed mutable-data generation seams are present."""
    source = _read("src/monitor/health_check.py")
    kill_surfaces = _read("src/monitor/health_kill_surfaces.py")

    assert "write_health_generation" in source
    assert "commit_public_index" in source
    assert "generation_id" in source
    # HEALTH-CHECK-SPLIT: implementations live in health_kill_surfaces.py;
    # health_check.py re-exports the callables, so the run-id stamp seam is
    # asserted at its owning module.
    assert "producer_run_id" in kill_surfaces


# ── Task 2: --candidate-no-scheduler (default service stays scheduler-enabled) ─

def test_deploy_candidate_no_scheduler_flag_keeps_default_service_scheduler_enabled():
    """The flag exists; without it the deployed unit stays scheduler-enabled.

    The scheduler-disable strings may only appear inside the candidate branch;
    the default unit template must not contain them."""
    source = _read("scripts/deploy-lab-app.sh")

    assert "--candidate-no-scheduler" in source
    assert '--candidate-no-scheduler) CANDIDATE_NO_SCHEDULER="1"; shift ;;' in source
    assert 'CANDIDATE_NO_SCHEDULER="0"' in source
    assert 'local scheduler_args=""' in source
    assert 'local scheduler_env_line=""' in source
    # candidate branch assignments (both --no-scheduler and env disable)
    assert 'scheduler_args="--no-scheduler"' in source
    assert 'scheduler_env_line="Environment=TASKER_DISABLE_SCHEDULER=1"' in source
    # ExecStart appends scheduler args only when non-empty; env line injected
    assert '${scheduler_args:+ ${scheduler_args}}' in source
    assert '${scheduler_env_line}' in source
    # default (unflagged) unit content must stay scheduler-enabled
    unit_block = _tasker_unit_block(source)
    assert "--no-scheduler" not in unit_block
    assert "TASKER_DISABLE_SCHEDULER=1" not in unit_block
    assert "src.tasker.service" in unit_block


def test_deploy_unit_template_default_vs_candidate(tmp_path: Path):
    """Bash-evaluate the unit heredoc template with both variable branches."""
    source = _read("scripts/deploy-lab-app.sh")
    block = _tasker_unit_block(source)

    for label, args, envline, wants_scheduler in (
        ("default", "", "", False),
        ("candidate", "--no-scheduler", "Environment=TASKER_DISABLE_SCHEDULER=1", True),
    ):
        script = (
            f'scheduler_args="{args}"\n'
            f'scheduler_env_line="{envline}"\n'
            "cat <<EOF\n"
            f"{block}\n"
            "EOF\n"
        )
        res = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
        assert res.returncode == 0, res.stderr
        if wants_scheduler:
            assert "--no-scheduler" in res.stdout, label
            assert "TASKER_DISABLE_SCHEDULER=1" in res.stdout, label
            assert res.stdout.index("src.tasker.service") < res.stdout.index("--no-scheduler"), label
            # EnvironmentFile must precede the scheduler-disable env line so
            # .env.local cannot override it
            assert (
                res.stdout.index("EnvironmentFile=-") < res.stdout.index("Environment=TASKER_DISABLE_SCHEDULER=1")
            ), label
        else:
            assert "--no-scheduler" not in res.stdout, label
            assert "TASKER_DISABLE_SCHEDULER=1" not in res.stdout, label
        assert "KillSignal=SIGTERM" in res.stdout, label
        assert "TimeoutStopSec=30" in res.stdout, label


def test_deploy_candidate_no_scheduler_flag_parses_in_dry_run(tmp_path):
    """The new flag parses and the script still reaches the dry-run exit.

    The candidate path is fail-closed against the production --app-dir, so
    the candidate variant must point --app-dir at a non-production path.
    """
    script = str(PROJECT_ROOT / "scripts" / "deploy-lab-app.sh")
    for extra_flag in (["--candidate-no-scheduler"], []):
        app_dir = str(tmp_path) if extra_flag else str(PROJECT_ROOT)
        res = subprocess.run(
            [
                "bash",
                script,
                "--app-dir",
                app_dir,
                "--service-name",
                "portfolio-lab-tasker-recovery-dev",
                "--web-root",
                "/srv/pl-candidate-www",
                "--public-root",
                "/srv/pl-candidate-www",
                *extra_flag,
                "--dry-run",
                "--skip-git",
                "--skip-deps",
                "--skip-data",
                "--skip-build",
                "--skip-service",
                "--skip-caddy",
                "--skip-update-command",
                "--skip-mirror",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert res.returncode == 0, res.stderr
        assert "[dry-run]" in res.stdout


def test_deploy_candidate_fails_closed_for_authoritative_use(tmp_path):
    """--candidate-no-scheduler is for private recovery/candidate APIs only:
    it must fail closed without --skip-caddy or with production identity."""
    script = str(PROJECT_ROOT / "scripts" / "deploy-lab-app.sh")
    bad_combos = [
        # no --skip-caddy
        {
            "--service-name": "portfolio-lab-tasker-recovery-dev",
            "--web-root": "/srv/pl-candidate-www",
            "--public-root": "/srv/pl-candidate-www",
        },
        # production service name
        {
            "--skip-caddy": "",
            "--web-root": "/srv/pl-candidate-www",
            "--public-root": "/srv/pl-candidate-www",
        },
        # production web root
        {
            "--skip-caddy": "",
            "--service-name": "portfolio-lab-tasker-recovery-dev",
            "--public-root": "/srv/pl-candidate-www",
        },
        # production public root
        {
            "--skip-caddy": "",
            "--service-name": "portfolio-lab-tasker-recovery-dev",
            "--web-root": "/srv/pl-candidate-www",
        },
        # production app dir
        {
            "--skip-caddy": "",
            "--service-name": "portfolio-lab-tasker-recovery-dev",
            "--web-root": "/srv/pl-candidate-www",
            "--public-root": "/srv/pl-candidate-www",
            "--app-dir": "/root/projects/portfolio-lab",
        },
    ]
    for combo in bad_combos:
        args = ["bash", script, "--app-dir", str(PROJECT_ROOT), "--candidate-no-scheduler", "--dry-run"]
        for key, value in combo.items():
            args.append(key)
            if value:
                args.append(value)
        res = subprocess.run(args, capture_output=True, text=True, timeout=120)
        assert res.returncode != 0, combo
    # the isolated candidate combination is accepted (non-production app-dir)
    good = [
        "bash",
        script,
        "--app-dir",
        str(tmp_path),
        "--candidate-no-scheduler",
        "--service-name",
        "portfolio-lab-tasker-recovery-dev",
        "--web-root",
        "/srv/pl-candidate-www",
        "--public-root",
        "/srv/pl-candidate-www",
        "--dry-run",
        "--skip-git",
        "--skip-deps",
        "--skip-data",
        "--skip-build",
        "--skip-service",
        "--skip-caddy",
        "--skip-update-command",
        "--skip-mirror",
    ]
    res = subprocess.run(good, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
