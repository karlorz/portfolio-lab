import importlib.util
import hashlib
import json
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


def test_lab_deploy_checks_public_data_consistency_after_live_mirror_before_publish():
    source = _read("scripts/deploy-lab-app.sh")
    # Isolate the main() call sequence (after the last function definition),
    # so .index() resolves call sites rather than earlier function defs.
    main_body = source.split("main() {", 1)[1]
    main_body = main_body.split("}", 1)[0]

    assert "check_public_data_consistency" in source
    # Live WWW data must be mirrored into checkout public/data BEFORE build
    # (so dist/data captures the fresh mirror) and before the consistency
    # check (so index entries resolve). Files the operator tree has but the
    # repo mirror lacked (e.g. attribution_YYYY-MM-DD.json) otherwise
    # false-fail the pre-publish gate.
    assert main_body.index("mirror_repo_public_data_from_live") < main_body.index("build_frontend")
    assert main_body.index("build_frontend") < main_body.index("check_public_data_consistency")
    assert main_body.index("check_public_data_consistency") < main_body.index("publish_dist")


def test_lab_deploy_warns_that_skip_data_uses_existing_artifact_consistency_gate():
    source = _read("scripts/deploy-lab-app.sh")

    assert "--skip-data set; validating existing public/data and dist/data artifacts" in source


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
    assert "python_runtime.sh -m src.monitor.fred_readiness" in source
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
