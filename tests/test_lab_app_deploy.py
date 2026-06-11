from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


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
