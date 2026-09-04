"""Hermetic tests for scripts/portfolio_lab_recovery.py (recovery CLI).

Fakes only in tests: a fake ``systemctl`` injected through PLR_SYSTEMCTL;
real git/tar over temp repositories and web trees; stdlib sqlite3 used by
the tool itself (no sqlite3 executable dependency). No live hosts, no root,
no Docker. Subprocess calls use argv arrays only (never shell=True).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import subprocess
import sys
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECOVERY_SCRIPT = PROJECT_ROOT / "scripts" / "portfolio_lab_recovery.py"
SCHEMA = "portfolio-lab-recovery/v2"
ARCHIVE_SUFFIX = ".portfolio-lab-recovery.tar"
TASKER = "portfolio-lab-tasker"
DEV_SERVICE = "portfolio-lab-tasker-recovery-dev"
_NO_GENERATOR = object()
CADDY_BLOCK = (
    "# BEGIN portfolio-lab managed\n"
    "lab.karldigi.dev {\n\tencode gzip\n}\n"
    "# END portfolio-lab managed\n"
)
UNIT_TEXT = "[Unit]\nDescription=tasker\nExecStart=/opt/run\n"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, content: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def make_fake(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def set_systemctl_mode(hermetic: SimpleNamespace, mode: str) -> None:
    """Rewrite the fake systemctl: active/inactive/stop-fail/start-fail."""
    body = f'#!/bin/sh\nprintf \'%s\\n\' "$*" >> "{hermetic.systemctl_log}"\n'
    if mode == "active":
        body += 'case "$1" in\n  is-active) printf "active\\n"; exit 0 ;;\nesac\nexit 0\n'
    elif mode == "inactive":
        body += 'case "$1" in\n  is-active) printf "inactive\\n"; exit 3 ;;\nesac\nexit 0\n'
    elif mode == "stop-fail":
        body += 'case "$1" in\n  is-active) printf "active\\n"; exit 0 ;;\n  stop) exit 1 ;;\nesac\nexit 0\n'
    elif mode == "start-fail":
        body += 'case "$1" in\n  is-active) printf "active\\n"; exit 0 ;;\n  start) exit 1 ;;\nesac\nexit 0\n'
    elif mode == "failed":
        body += 'case "$1" in\n  is-active) printf "failed\\n"; exit 3 ;;\nesac\nexit 0\n'
    else:
        raise AssertionError(mode)
    make_fake(hermetic.bin / "systemctl", body)


def make_repo(root: Path) -> Path:
    """Fresh git repo with the canonical data/config files committed.

    Mirrors the host repo: ``data/``, ``logs`` and ``research-implement.md``
    are git-ignored, so runtime files under data/ never trip the dirty check,
    while tracked data files still report worktree modifications. The runtime
    sqlite db is a real SQLite database (verify uses stdlib sqlite3)."""
    subprocess.run(["git", "init", "-b", "main", str(root)], check=True, capture_output=True)
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    _write(root / "README.md", "# test repo\n")
    _write(
        root / "config/lab-app.env",
        "TASKER_SERVICE_NAME=portfolio-lab-tasker\nTASKER_HOST=127.0.0.1\nTASKER_PORT=8000\n",
    )
    _write(root / "data/ensemble_weights.json", '{"normal": {"spy": 0.46}}\n')
    _write(root / "data/vix_term_structure.json", '{"_meta": {"schema": "vix_term_structure/v1"}}\n')
    commit_all(root, "base")
    _write(root / ".gitignore", "/data/\nlogs\nresearch-implement.md\n")
    conn = sqlite3.connect(str(root / "data/market.db"))
    conn.execute("CREATE TABLE t (x TEXT)")
    conn.execute("INSERT INTO t VALUES ('hello')")
    conn.commit()
    conn.close()
    commit_all(root, "gitignore")
    return root


def commit_all(repo: Path, message: str = "base") -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "--allow-empty", "-m", message)
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def make_web_root(
    web: Path,
    source_sha: str,
    generator_sha: str | None = None,
    release_sha: str | None = None,
) -> Path:
    _write(web / "index.html", "<html>recovered</html>\n")
    _write(web / "assets/app.js", "// app\n")
    _write(
        web / "_release.json",
        json.dumps(
            {
                "schema_version": "portfolio-lab-static-release/v1",
                "source_git_sha": release_sha or source_sha,
            }
        )
        + "\n",
    )
    index: dict[str, object] = {
        "schema_version": "public-data-index/v1",
        "generated_at": "2026-08-14T00:00:00Z",
    }
    if generator_sha:
        index["generator_git_sha"] = generator_sha
    _write(web / "data/index.json", json.dumps(index) + "\n")
    _write(web / "data/prices.json", '{"prices": []}\n')
    return web


@pytest.fixture
def hermetic(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    systemctl_log = tmp_path / "systemctl.log"
    units = tmp_path / "units"
    units.mkdir()
    caddy = tmp_path / "Caddyfile"
    _write(caddy, CADDY_BLOCK)
    set_systemctl_mode(SimpleNamespace(tmp=tmp_path, bin=bin_dir, systemctl_log=systemctl_log), "active")
    status_server, status_thread, status_url = tasker_status_server(
        {"service": "tasker", "backend": "tasker", "tasks": []}
    )
    env = {
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        "PLR_GIT": "git",
        "PLR_TAR": "tar",
        # Test escape hatch: pytest's tmp_path lives under /tmp/pytest-of-*,
        # which the production guard refuses. The hatch only permits exactly
        # those trees (script check_not_forbidden); every other /tmp
        # destination stays forbidden (guard-expectation tests unchanged).
        "PLR_ALLOW_TMP_DEST": "1",
        "PLR_SYSTEMCTL": str(bin_dir / "systemctl"),
        "PLR_SYSTEMD_UNIT_DIR": str(units),
        "PLR_WIKI_DIR": str(tmp_path / "vault"),
        "PLR_CADDY_CONFIG": str(caddy),
        "PLR_TASKER_STATUS_URL": status_url,
    }
    fixture = SimpleNamespace(
        tmp=tmp_path,
        bin=bin_dir,
        env=env,
        systemctl_log=systemctl_log,
        units=units,
        caddy=caddy,
        tasker_status_url=status_url,
    )
    try:
        yield fixture
    finally:
        status_server.shutdown()
        status_thread.join(timeout=5)


def run_recovery(
    args: list[str],
    hermetic: SimpleNamespace,
    extra_env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    env = {**os.environ, **hermetic.env}
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(RECOVERY_SCRIPT), *args],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
        timeout=120,
    )


def report_of(res: subprocess.CompletedProcess) -> dict:
    assert res.returncode == 0, f"exit={res.returncode} stderr={res.stderr}"
    return json.loads(res.stdout)


def tasker_status_server(payload: object, status: int = 200) -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    """Start a hermetic localhost endpoint for the source Tasker status API."""

    body = json.dumps(payload).encode("utf-8") if not isinstance(payload, bytes) else payload

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 -- HTTP handler API spelling
            if self.path != "/api/tasker/status":
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    return server, thread, f"http://{host}:{port}/api/tasker/status"


def standard_create(
    hermetic: SimpleNamespace,
    repo: Path,
    web: Path,
    archive_name: str = "backup" + ARCHIVE_SUFFIX,
    extra: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
) -> tuple[Path, subprocess.CompletedProcess]:
    archive = hermetic.tmp / "backups" / archive_name
    archive.parent.mkdir(parents=True, exist_ok=True)
    _write(hermetic.units / f"{TASKER}.service", UNIT_TEXT)
    _write(repo / "data/tasker_status.json", json.dumps({"state": "idle", "version": 1}) + "\n")
    args = [
        "create",
        "--app-dir",
        str(repo),
        "--web-root",
        str(web),
        "--tasker-service",
        TASKER,
        "--archive",
        str(archive),
        "--storage-encryption-attested",
        *(extra or []),
    ]
    return archive, run_recovery(args, hermetic, extra_env=extra_env)


def test_create_requires_live_tasker_api_capture_before_service_stop(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    server, thread, status_url = tasker_status_server(
        {"service": "tasker", "backend": "tasker", "tasks": [{"id": "health"}]}
    )
    try:
        archive, res = standard_create(
            hermetic,
            repo,
            web,
            extra_env={"PLR_TASKER_STATUS_URL": status_url},
        )
        assert res.returncode == 0, res.stderr
        captured = json.loads(extract_tar_member(archive, "metadata/tasker-status.json"))
        assert captured["tasks"] == [{"id": "health"}]
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_create_live_tasker_api_failure_happens_before_service_stop(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    missing_url = "http://127.0.0.1:1/api/tasker/status"
    archive = hermetic.tmp / "backups" / ("live-status" + ARCHIVE_SUFFIX)
    archive.parent.mkdir(parents=True, exist_ok=True)
    _write(hermetic.units / f"{TASKER}.service", UNIT_TEXT)
    _write(repo / "data/tasker_status.json", json.dumps({"state": "idle"}) + "\n")
    res = run_recovery(
        [
            "create",
            "--app-dir",
            str(repo),
            "--web-root",
            str(web),
            "--tasker-service",
            TASKER,
            "--archive",
            str(archive),
            "--storage-encryption-attested",
        ],
        hermetic,
        extra_env={"PLR_TASKER_STATUS_URL": missing_url},
    )
    assert res.returncode != 0
    assert "tasker api" in res.stderr.lower()
    assert not hermetic.systemctl_log.exists()
    assert not archive.exists()


def standard_restore(
    hermetic: SimpleNamespace,
    archive: Path,
    mode: str,
    extra: list[str] | None = None,
    app_name: str = "app",
    web_name: str = "www",
    extra_env: dict[str, str] | None = None,
) -> tuple[Path, Path, subprocess.CompletedProcess]:
    app_dir = hermetic.tmp / app_name
    web_root = hermetic.tmp / web_name
    args = [
        "restore",
        "--archive",
        str(archive),
        "--app-dir",
        str(app_dir),
        "--web-root",
        str(web_root),
        "--target-mode",
        mode,
        *(extra or []),
    ]
    return app_dir, web_root, run_recovery(args, hermetic, extra_env=extra_env)


def _create_and_restore(
    hermetic: SimpleNamespace,
    tmp_path: Path,
    release_sha: str | None = None,
    generator_sha: str | None | object = None,
) -> tuple[Path, Path, Path, str]:
    """create + prod restore (with --allow-production-paths) + .env.local.

    Default generator_git_sha is a reachable short sha of the archived source
    so activation reaches the gate under test; pass ``_NO_GENERATOR`` to omit
    the key entirely."""
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    if generator_sha is None:
        generator_sha = source_sha[:12]
    elif generator_sha is _NO_GENERATOR:
        generator_sha = None
    web = make_web_root(tmp_path / "web", source_sha, generator_sha=generator_sha, release_sha=release_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    set_systemctl_mode(hermetic, "inactive")
    app_dir, web_root, res = standard_restore(
        hermetic, archive, "prod", extra=["--allow-production-paths"]
    )
    assert res.returncode == 0, res.stderr
    _write(app_dir / ".env.local", "PORTFOLIO_LAB_MODE=lab\n")
    return archive, app_dir, web_root, source_sha


def run_activate(hermetic: SimpleNamespace, app_dir: Path, web_root: Path) -> subprocess.CompletedProcess:
    args = [
        "activate-prod",
        "--app-dir",
        str(app_dir),
        "--web-root",
        str(web_root),
        "--tasker-service",
        TASKER,
        "--confirm-authoritative-activation",
        "--former-authority-confirmed-stopped",
        "former-host.example",
    ]
    return run_recovery(args, hermetic)


def assert_no_activation_systemctl(hermetic: SimpleNamespace) -> None:
    """Activation must not have installed/started anything (create's own
    is-active/stop/start calls may be present in the log)."""
    log = hermetic.systemctl_log.read_text(encoding="utf-8") if hermetic.systemctl_log.exists() else ""
    assert "daemon-reload" not in log
    assert "enable portfolio-lab-tasker" not in log


def read_tar_members(archive: Path) -> list[str]:
    res = subprocess.run(["tar", "-tf", str(archive)], capture_output=True, text=True, check=True)
    return res.stdout.splitlines()


def extract_tar(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xf", str(archive), "-C", str(dest)], check=True)


def extract_tar_member(archive: Path, member: str) -> bytes:
    res = subprocess.run(["tar", "-xOf", str(archive), member], capture_output=True, check=True)
    return res.stdout


def write_sidecar(archive: Path) -> None:
    Path(str(archive) + ".sha256").write_text(
        f"{_sha256_bytes(archive.read_bytes())}  {archive.name}\n", encoding="utf-8"
    )


def _recompute_manifest(work: Path) -> None:
    manifest_path = work / "recovery-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = []
    for p in sorted(work.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        rel = p.relative_to(work).as_posix()
        if rel == "recovery-manifest.json":
            continue
        entries.append(
            {"path": rel, "sha256": _sha256_bytes(p.read_bytes()), "bytes": p.stat().st_size, "mode": p.stat().st_mode & 0o7777}
        )
    manifest["members"] = entries
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def repackage(archive: Path, work: Path, mutate, recompute: bool = False) -> Path:
    """Extract archive into work, apply mutate(work), re-tar, fresh sidecar.

    With recompute=True the embedded manifest member digests are rebuilt from
    the mutated tree (mirroring the tool's own create-time computation)."""
    work.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xf", str(archive), "-C", str(work)], check=True)
    mutate(work)
    if recompute:
        _recompute_manifest(work)
    members = sorted(
        p.relative_to(work).as_posix() for p in work.rglob("*") if p.is_file() and not p.is_symlink()
    )
    rebuilt = work.parent / f"{work.name}.tar"
    # COPYFILE_DISABLE keeps macOS tar from adding AppleDouble "._*" members
    env = {**os.environ, "COPYFILE_DISABLE": "1"}
    subprocess.run(
        ["tar", "-cf", str(rebuilt), "-C", str(work), *members],
        check=True,
        env=env,
    )
    write_sidecar(rebuilt)
    return rebuilt


def _load_recovery_module():
    """Import the recovery script in-process (module level is constants and
    functions only, so this is side-effect free)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("portfolio_lab_recovery", RECOVERY_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── create: CLI guards ──────────────────────────────────────────────────────


def test_create_requires_storage_encryption_attestation(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    archive = hermetic.tmp / "backups" / ("a" + ARCHIVE_SUFFIX)
    archive.parent.mkdir(parents=True, exist_ok=True)
    res = run_recovery(
        [
            "create",
            "--app-dir",
            str(repo),
            "--web-root",
            str(web),
            "--tasker-service",
            TASKER,
            "--archive",
            str(archive),
        ],
        hermetic,
    )
    assert res.returncode != 0
    assert "--storage-encryption-attested" in res.stderr
    assert not archive.exists()
    assert not hermetic.systemctl_log.exists()


def test_create_requires_portfolio_lab_recovery_tar_suffix(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    archive = hermetic.tmp / "backups" / "backup.tar"
    archive.parent.mkdir(parents=True, exist_ok=True)
    res = run_recovery(
        [
            "create",
            "--app-dir",
            str(repo),
            "--web-root",
            str(web),
            "--tasker-service",
            TASKER,
            "--archive",
            str(archive),
            "--storage-encryption-attested",
        ],
        hermetic,
    )
    assert res.returncode != 0
    assert ARCHIVE_SUFFIX in res.stderr
    assert not archive.exists()


def test_create_rejects_relative_or_forbidden_archive_paths(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    forbidden = [
        ("relative", "relative/backup" + ARCHIVE_SUFFIX),
        ("repo", str(repo / ("backup" + ARCHIVE_SUFFIX))),
        ("web", str(web / ("backup" + ARCHIVE_SUFFIX))),
        ("vault", str(hermetic.tmp / "vault" / ("backup" + ARCHIVE_SUFFIX))),
        ("tmp", "/tmp/portfolio-lab-recovery-test/backup" + ARCHIVE_SUFFIX),
    ]
    for label, dest in forbidden:
        res = run_recovery(
            [
                "create",
                "--app-dir",
                str(repo),
                "--web-root",
                str(web),
                "--tasker-service",
                TASKER,
                "--archive",
                dest,
                "--storage-encryption-attested",
            ],
            hermetic,
        )
        assert res.returncode != 0, label
        assert "forbidden" in res.stderr.lower() or "absolute" in res.stderr.lower(), (label, res.stderr)


def test_create_rejects_missing_destination_parent(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    archive = hermetic.tmp / "no-such-dir" / ("backup" + ARCHIVE_SUFFIX)
    res = run_recovery(
        [
            "create",
            "--app-dir",
            str(repo),
            "--web-root",
            str(web),
            "--tasker-service",
            TASKER,
            "--archive",
            str(archive),
            "--storage-encryption-attested",
        ],
        hermetic,
    )
    assert res.returncode != 0
    assert "parent" in res.stderr.lower()


def test_create_rejects_existing_archive(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    archive = hermetic.tmp / "backups" / ("backup" + ARCHIVE_SUFFIX)
    archive.parent.mkdir(parents=True, exist_ok=True)
    _write(archive, "existing")
    res = run_recovery(
        [
            "create",
            "--app-dir",
            str(repo),
            "--web-root",
            str(web),
            "--tasker-service",
            TASKER,
            "--archive",
            str(archive),
            "--storage-encryption-attested",
        ],
        hermetic,
    )
    assert res.returncode != 0
    assert "already exists" in res.stderr.lower()


# ── create: dirty-source policy (rejected BEFORE tasker stop) ───────────────


def test_create_rejects_dirty_source_beyond_allowed_data_files(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    _write(repo / "README.md", "# modified\n")
    _, res = standard_create(hermetic, repo, web)
    assert res.returncode != 0
    assert "dirty" in res.stderr.lower()
    assert not hermetic.systemctl_log.exists()

    _write(repo / "README.md", "# test repo\n")
    _write(repo / "untracked.txt", "x")
    _, res = standard_create(hermetic, repo, web)
    assert res.returncode != 0
    assert "dirty" in res.stderr.lower()
    assert not hermetic.systemctl_log.exists()

    _write(repo / "data/ensemble_weights.json", '{"normal": {"spy": 0.99}}\n')
    _git(repo, "add", "-f", "data/ensemble_weights.json")
    _, res = standard_create(hermetic, repo, web)
    assert res.returncode != 0
    assert "dirty" in res.stderr.lower()
    assert not hermetic.systemctl_log.exists()


def test_create_allows_ordinary_modifications_of_the_two_data_files(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    _write(repo / "data/ensemble_weights.json", '{"normal": {"spy": 0.42, "gld": 0.38}}\n')
    _write(repo / "data/vix_term_structure.json", '{"_meta": {"schema": "vix_term_structure/v1", "dirty": true}}\n')
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    extracted = hermetic.tmp / "extracted"
    extract_tar(archive, extracted)
    assert "0.42" in (extracted / "runtime/data/ensemble_weights.json").read_text(encoding="utf-8")
    assert '"dirty": true' in (extracted / "runtime/data/vix_term_structure.json").read_text(encoding="utf-8")


def test_create_rejects_secret_keys_in_config(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    _write(
        repo / "config/lab-app.env",
        "TASKER_SERVICE_NAME=portfolio-lab-tasker\nOPENAI_API_KEY=sk-secret\n",
    )
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    _, res = standard_create(hermetic, repo, web)
    assert res.returncode != 0
    assert "secret" in res.stderr.lower()
    assert not hermetic.systemctl_log.exists()
    assert not list((hermetic.tmp / "backups").glob("*.tar"))


def test_create_rejects_secret_values_in_config(hermetic, tmp_path: Path):
    """Credential-bearing values are rejected even when the key name is benign."""
    cases = {
        "url-userinfo": "DATABASE_URL=postgres://appuser:hunter2@db.internal:5432/app\n",
        "query-token": "GRAFANA_URL=https://grafana.example/d/abc?token=xyz123\n",
        "value-prefix": "CLOUD_KEY=AKIAIOSFODNN7EXAMPLE\n",
        "bearer": "OPS_URL=https://ops.example/api?key=abc&format=json\n",
    }
    for label, config_line in cases.items():
        repo = make_repo(tmp_path / f"repo-{label}")
        _write(repo / "config/lab-app.env", "TASKER_SERVICE_NAME=portfolio-lab-tasker\n" + config_line)
        commit_all(repo)
        web = make_web_root(tmp_path / f"web-{label}", "x" * 40)
        _, res = standard_create(hermetic, repo, web)
        assert res.returncode != 0, label
        assert "secret" in res.stderr.lower(), label
        assert not hermetic.systemctl_log.exists(), label


def test_create_accepts_benign_config_values(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    _write(
        repo / "config/lab-app.env",
        "TASKER_SERVICE_NAME=portfolio-lab-tasker\n"
        "PORTFOLIO_LAB_SITE_ADDRESS=https://lab.karldigi.dev\n"
        "TASKER_HOST=127.0.0.1\nTASKER_PORT=8000\n",
    )
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    _, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr


def test_create_rejects_case_variant_secret_paths_and_unseparated_secret_keys(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    _write(repo / "config/lab-app.env", "TASKER_SERVICE_NAME=portfolio-lab-tasker\nACCESS_KEY=secret\n")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    _, res = standard_create(hermetic, repo, web)
    assert res.returncode != 0
    assert "secret" in res.stderr.lower()
    assert not hermetic.systemctl_log.exists()

    _write(repo / "config/lab-app.env", "TASKER_SERVICE_NAME=portfolio-lab-tasker\n")
    commit_all(repo)
    _write(repo / "data" / "SECRETS" / "leak.txt", "TOP SECRET\n")
    _write(web / "CREDENTIALS" / "leak.txt", "TOP SECRET\n")
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    members = set(read_tar_members(archive))
    assert not any("secrets" in member.lower() or "credentials" in member.lower() for member in members)


def test_create_fails_closed_on_failed_source_state(hermetic, tmp_path: Path):
    def make_failed(path: Path) -> None:
        make_fake(
            path,
            f'#!/bin/sh\nprintf \'%s\\n\' "$*" >> "{hermetic.systemctl_log}"\n'
            'case "$1" in\n  is-active) printf "failed\\n"; exit 3 ;;\nesac\nexit 0\n',
        )

    make_failed(hermetic.bin / "systemctl")
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode != 0
    assert "state" in res.stderr.lower()
    assert not archive.exists()
    log = hermetic.systemctl_log.read_text(encoding="utf-8").splitlines()
    assert log == ["is-active portfolio-lab-tasker"]
    assert not any("stop" in line or "start" in line for line in log)


def test_create_fails_closed_on_systemctl_error_state(hermetic, tmp_path: Path):
    make_fake(
        hermetic.bin / "systemctl",
        f'#!/bin/sh\nprintf \'%s\\n\' "$*" >> "{hermetic.systemctl_log}"\n'
        'case "$1" in\n  is-active) printf "unknown-state\\n"; exit 1 ;;\nesac\nexit 0\n',
    )
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode != 0
    assert "state" in res.stderr.lower() or "failed" in res.stderr.lower()
    assert not archive.exists()
    assert not any("stop" in line or "start" in line for line in hermetic.systemctl_log.read_text(encoding="utf-8").splitlines())


def test_verify_rejects_malformed_source_sha(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr

    def mutate(work: Path) -> None:
        manifest_path = work / "recovery-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source"]["sha"] = "DEADBEEF"  # uppercase + wrong length
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")

    rebuilt = repackage(archive, hermetic.tmp / "retar", mutate)
    res = run_recovery(["verify", "--archive", str(rebuilt)], hermetic)
    assert res.returncode != 0
    report = json.loads(res.stdout)
    assert report["checks"]["bundle_revision_matches"] is False
    assert "source" in report["error"].lower()


def test_verify_rejects_source_sha_absent_from_bundle(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr

    def mutate(work: Path) -> None:
        manifest_path = work / "recovery-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source"]["sha"] = "f" * 40
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")

    rebuilt = repackage(archive, hermetic.tmp / "retar", mutate)
    res = run_recovery(["verify", "--archive", str(rebuilt)], hermetic)
    assert res.returncode != 0
    report = json.loads(res.stdout)
    assert report["checks"]["bundle_revision_matches"] is False


# ── create: archive contents ────────────────────────────────────────────────


def test_create_archive_layout_and_sidecar(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha, generator_sha=source_sha[:12])
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    report = report_of(res)
    assert report["ok"] is True
    assert report["schema_version"] == SCHEMA
    assert report["initially_active"] is True
    assert report["service_stopped"] is True
    assert report["service_started"] is True
    assert report["storage_encryption_attested"] is True
    assert report["source_sha"] == source_sha

    expected = {
        "recovery-manifest.json",
        "source/repository.bundle",
        "source/revision.json",
        "runtime/data/ensemble_weights.json",
        "runtime/data/vix_term_structure.json",
        "runtime/data/market.db",
        "runtime/data/tasker_status.json",
        "static/web/index.html",
        "static/web/assets/app.js",
        "static/web/_release.json",
        "static/web/data/index.json",
        "static/web/data/prices.json",
        "config/lab-app.env",
        "metadata/tasker-unit.txt",
        "metadata/tasker-status.json",
        "metadata/caddy-portfolio-lab-block.txt",
        "metadata/created.json",
        "tools/portfolio_lab_recovery.py",
    }
    assert set(read_tar_members(archive)) == expected
    assert read_tar_members(archive) == sorted(read_tar_members(archive))
    # the only sidecar is <archive>.sha256
    siblings = {p.name for p in archive.parent.iterdir()}
    assert siblings == {archive.name, archive.name + ".sha256"}
    sidecar_text = (archive.parent / (archive.name + ".sha256")).read_text(encoding="utf-8")
    assert sidecar_text.split()[0] == _sha256_bytes(archive.read_bytes())

    embedded = json.loads(extract_tar_member(archive, "recovery-manifest.json"))
    assert embedded["schema_version"] == SCHEMA
    assert embedded["source"]["sha"] == source_sha
    assert embedded["data_index_generator_sha"] == source_sha[:12]
    assert embedded["storage_encryption_attested"] is True
    assert {m["path"] for m in embedded["members"]} == expected - {"recovery-manifest.json"}
    assert all(isinstance(m["mode"], int) for m in embedded["members"])
    assert all(isinstance(m["sha256"], str) and len(m["sha256"]) == 64 for m in embedded["members"])

    revision = json.loads(extract_tar_member(archive, "source/revision.json"))
    assert revision["schema_version"] == SCHEMA
    assert revision["source_sha"] == source_sha
    assert revision["app_dir"] == str(repo)
    assert revision["web_root"] == str(web)
    created = json.loads(extract_tar_member(archive, "metadata/created.json"))
    assert created["schema_version"] == SCHEMA
    assert created["archive"] == archive.name
    assert extract_tar_member(archive, "tools/portfolio_lab_recovery.py") == RECOVERY_SCRIPT.read_bytes()


def test_create_includes_sqlite_wal_and_shm(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    conn = sqlite3.connect(str(repo / "data/market.db"))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("INSERT INTO t VALUES ('wal-row')")
    conn.commit()
    try:
        archive, res = standard_create(hermetic, repo, web)
    finally:
        conn.close()
    assert res.returncode == 0, res.stderr
    members = read_tar_members(archive)
    assert "runtime/data/market.db" in members
    assert "runtime/data/market.db-wal" in members
    assert "runtime/data/market.db-shm" in members
    res = run_recovery(["verify", "--archive", str(archive)], hermetic)
    assert res.returncode == 0, res.stderr
    assert json.loads(res.stdout)["checks"]["sqlite_ok"] is True


def test_create_excludes_secrets_identity_agents_and_env(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    for secret_path in (
        repo / ".env",
        repo / ".env.local",
        repo / "data/.env",
        repo / ".ssh/id_rsa",
        repo / ".hermes/creds.json",
        repo / ".claude/settings.json",
        repo / ".grok/session.json",
        repo / "data/secrets.json",
        repo / "data/tokens.json",
        repo / "data/credentials.json",
        repo / "data/id_ed25519",
        repo / "data/agent_keys.pem",
        repo / "data/rclone.conf",
        repo / "data/broker_credentials.json",
        repo / "data/broker_session.json",
        repo / "data/alpaca_session.json",
        repo / "data/machine-id",
        web / ".env.local",
        web / ".ssh/id_rsa",
        web / "data/secrets.json",
        web / "data/tokens.json",
        web / ".hermes/creds.json",
        web / "certs/cert.pem",
    ):
        _write(secret_path, "TOP SECRET\n")
    # root-level secret/agent dirs are tracked so the tree stays clean; the
    # data/ copies are git-ignored runtime files and never trip the dirty check
    commit_all(repo)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    members = set(read_tar_members(archive))
    for m in members:
        parts = Path(m).parts
        assert not any(
            p in {".env", ".env.local", ".ssh", ".hermes", ".claude", ".grok", "secrets", "credentials", "tokens", "certs"}
            for p in parts
        ), m
    names = {Path(m).name for m in members}
    assert not any(
        "secret" in n or "token" in n or "credential" in n or n.endswith((".pem", ".key", ".crt")) for n in names
    )
    assert "id_rsa" not in names
    assert "id_ed25519" not in names
    assert "rclone.conf" not in names
    assert "broker_session.json" not in names
    assert "alpaca_session.json" not in names
    assert "machine-id" not in names


def test_create_fails_closed_on_symlinked_directory_in_runtime_data(hermetic, tmp_path: Path):
    """A symlinked directory inside the archived data tree must abort
    creation before the source service is stopped and before any archive is
    written. Collection must never follow links out of the archived tree
    (the external target holds a benignly named file with a token-like
    payload that the name-based exclusion boundary would not stop), and a
    refusal must never leave a silently incomplete recovery point."""
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    external = tmp_path / "external-notes"
    _write(external / "notes.txt", "sk-test-super-secret-value-1234\n")
    (repo / "data" / "linked").symlink_to(external, target_is_directory=True)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode != 0
    assert "symlink" in res.stderr.lower()
    assert "runtime/data" in res.stderr
    assert not hermetic.systemctl_log.exists()
    assert not archive.exists()


def test_create_fails_closed_on_symlinked_directory_in_web_tree(hermetic, tmp_path: Path):
    """Same fail-closed guarantee for the static web source tree, whose
    member collection is a distinct call path (static/web prefix)."""
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    external = tmp_path / "external-notes"
    _write(external / "notes.txt", "sk-test-super-secret-value-1234\n")
    (web / "assets" / "linked").symlink_to(external, target_is_directory=True)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode != 0
    assert "symlink" in res.stderr.lower()
    assert "static/web" in res.stderr
    assert not hermetic.systemctl_log.exists()
    assert not archive.exists()


def test_create_archives_regular_nested_directories(hermetic, tmp_path: Path):
    """Regular nested directories (no symlinks) keep being archived after the
    symlink fail-closed change."""
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    _write(repo / "data/nested/deep/payload.json", '{"x": 1}\n')
    _write(web / "assets/css/main.css", "body {}\n")
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    members = set(read_tar_members(archive))
    assert "runtime/data/nested/deep/payload.json" in members
    assert "static/web/assets/css/main.css" in members


def test_create_rejects_symlinked_web_root_before_service_stop(hermetic, tmp_path: Path):
    """A --web-root that is itself a symlink must be rejected before resolve()
    (which would silently follow it) and before any systemctl call or
    archive write."""
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    real_web = make_web_root(tmp_path / "real-web", "x" * 40)
    web_link = tmp_path / "web-link"
    web_link.symlink_to(real_web, target_is_directory=True)
    archive = hermetic.tmp / "backups" / ("web-link" + ARCHIVE_SUFFIX)
    archive.parent.mkdir(parents=True, exist_ok=True)
    _write(hermetic.units / f"{TASKER}.service", UNIT_TEXT)
    _write(repo / "data/tasker_status.json", json.dumps({"state": "idle", "version": 1}) + "\n")
    res = run_recovery(
        [
            "create",
            "--app-dir",
            str(repo),
            "--web-root",
            str(web_link),
            "--tasker-service",
            TASKER,
            "--archive",
            str(archive),
            "--storage-encryption-attested",
        ],
        hermetic,
    )
    assert res.returncode != 0
    assert "symlink" in res.stderr.lower()
    assert "web root" in res.stderr.lower()
    assert not hermetic.systemctl_log.exists()
    assert not archive.exists()


def test_create_rejects_symlinked_data_dir_before_service_stop(hermetic, tmp_path: Path):
    """The app data directory itself being a symlink must be rejected before
    resolve() (which would silently follow it) and before any systemctl call
    or archive write."""
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    external_data = tmp_path / "external-data"
    # Mirror the tracked/generated data files so a pre-fix create would
    # succeed through the followed link (the tracked files pass the
    # allowed-dirty policy; tasker_status.json is the required mirror).
    _write(external_data / "ensemble_weights.json", '{"normal": {"spy": 0.46}}\n')
    _write(external_data / "vix_term_structure.json", '{"_meta": {"schema": "vix_term_structure/v1"}}\n')
    _write(external_data / "tasker_status.json", json.dumps({"state": "idle", "version": 1}) + "\n")
    os.rename(repo / "data", tmp_path / "real-data")
    (repo / "data").symlink_to(external_data, target_is_directory=True)
    archive = hermetic.tmp / "backups" / ("data-link" + ARCHIVE_SUFFIX)
    archive.parent.mkdir(parents=True, exist_ok=True)
    _write(hermetic.units / f"{TASKER}.service", UNIT_TEXT)
    res = run_recovery(
        [
            "create",
            "--app-dir",
            str(repo),
            "--web-root",
            str(web),
            "--tasker-service",
            TASKER,
            "--archive",
            str(archive),
            "--storage-encryption-attested",
        ],
        hermetic,
    )
    assert res.returncode != 0
    assert "symlink" in res.stderr.lower()
    assert "data directory" in res.stderr.lower()
    assert not hermetic.systemctl_log.exists()
    assert not archive.exists()


def test_create_fails_closed_on_unreadable_directory_in_data_tree(hermetic, tmp_path: Path):
    """An unreadable subdirectory under an archived tree must fail closed
    before the source service is stopped instead of being silently omitted
    from the recovery point. chmod-based; not reproducible as root."""
    if getattr(os, "geteuid", lambda: 0)() == 0:
        pytest.skip("permission-denied traversal is not reproducible as root")
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    locked = repo / "data" / "locked"
    locked.mkdir()
    _write(locked / "notes.txt", "sk-test-super-secret-value-1234\n")
    locked.chmod(0o000)
    try:
        archive, res = standard_create(hermetic, repo, web)
    finally:
        locked.chmod(0o755)
    assert res.returncode != 0
    assert "cannot traverse" in res.stderr.lower()
    assert not hermetic.systemctl_log.exists()
    assert not archive.exists()


def test_collection_fails_closed_on_directory_traversal_error(monkeypatch, capsys, tmp_path: Path):
    """os.walk traversal errors (unreadable/deleted/I/O-error directories)
    must surface through the controlled die() path instead of silently
    omitting directories from the member list. Portable via os.scandir
    injection (chmod behavior is platform-dependent)."""
    mod = _load_recovery_module()
    tree = tmp_path / "data"
    (tree / "ok").mkdir(parents=True)
    _write(tree / "ok" / "a.txt", "x")
    (tree / "bad").mkdir()
    real_scandir = os.scandir

    def guarded_scandir(path):
        if os.fspath(path).endswith("bad"):
            raise PermissionError(13, "Permission denied", os.fspath(path))
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", guarded_scandir)
    with pytest.raises(SystemExit):
        list(mod._walk_tree_files(tree, "runtime/data"))
    err = capsys.readouterr().err
    assert "cannot traverse" in err
    assert "runtime/data" in err


def test_create_optional_runtime_logs_research_implement_only(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    _write(repo / "logs/research-implement.md", "# implemented\n")
    _write(repo / "logs/tasker.log", "generic log line\n")
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    members = set(read_tar_members(archive))
    assert "runtime/logs/research-implement.md" in members
    assert not any(m.startswith("runtime/logs/") for m in members if m != "runtime/logs/research-implement.md")

    repo2 = make_repo(tmp_path / "repo2")
    commit_all(repo2)
    archive2, res2 = standard_create(hermetic, repo2, web, archive_name="backup2" + ARCHIVE_SUFFIX)
    assert res2.returncode == 0, res2.stderr
    assert not any(m.startswith("runtime/logs/") for m in read_tar_members(archive2))


def test_create_captures_only_managed_caddy_block(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    _write(
        hermetic.caddy,
        "other.site {\n\trespond \"other\"\n}\n\n" + CADDY_BLOCK,
    )
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    block = extract_tar_member(archive, "metadata/caddy-portfolio-lab-block.txt").decode()
    assert block == CADDY_BLOCK.rstrip("\n")
    assert "other.site" not in block


# ── create: metadata fail-closed + service drain semantics ──────────────────


def test_create_fails_closed_when_tasker_unit_missing(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    archive = hermetic.tmp / "backups" / ("a" + ARCHIVE_SUFFIX)
    archive.parent.mkdir(parents=True, exist_ok=True)
    _write(repo / "data/tasker_status.json", json.dumps({"state": "idle"}) + "\n")
    res = run_recovery(
        [
            "create",
            "--app-dir",
            str(repo),
            "--web-root",
            str(web),
            "--tasker-service",
            TASKER,
            "--archive",
            str(archive),
            "--storage-encryption-attested",
        ],
        hermetic,
    )
    assert res.returncode != 0
    assert "tasker unit" in res.stderr.lower()
    assert not hermetic.systemctl_log.exists()
    assert not archive.exists()


def test_create_rejects_inline_secret_in_tasker_unit_before_service_stop(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    _write(hermetic.units / f"{TASKER}.service", UNIT_TEXT + "Environment=ACCESS_KEY=secret\n")
    _write(repo / "data/tasker_status.json", json.dumps({"state": "idle"}) + "\n")
    archive = hermetic.tmp / "backups" / ("unit-secret" + ARCHIVE_SUFFIX)
    archive.parent.mkdir(parents=True, exist_ok=True)

    res = run_recovery(
        [
            "create",
            "--app-dir",
            str(repo),
            "--web-root",
            str(web),
            "--tasker-service",
            TASKER,
            "--archive",
            str(archive),
            "--storage-encryption-attested",
        ],
        hermetic,
    )

    assert res.returncode != 0
    assert "secret" in res.stderr.lower()
    assert not hermetic.systemctl_log.exists()
    assert not archive.exists()


def test_create_fails_closed_when_tasker_status_missing(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    _write(hermetic.units / f"{TASKER}.service", UNIT_TEXT)
    archive = hermetic.tmp / "backups" / ("a" + ARCHIVE_SUFFIX)
    archive.parent.mkdir(parents=True, exist_ok=True)
    res = run_recovery(
        [
            "create",
            "--app-dir",
            str(repo),
            "--web-root",
            str(web),
            "--tasker-service",
            TASKER,
            "--archive",
            str(archive),
            "--storage-encryption-attested",
        ],
        hermetic,
    )
    assert res.returncode != 0
    assert "tasker status" in res.stderr.lower()
    assert not hermetic.systemctl_log.exists()


def test_create_fails_closed_when_caddy_block_missing(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    _write(hermetic.units / f"{TASKER}.service", UNIT_TEXT)
    _write(repo / "data/tasker_status.json", json.dumps({"state": "idle"}) + "\n")
    _write(hermetic.caddy, "other.site {\n\trespond \"other\"\n}\n")
    archive = hermetic.tmp / "backups" / ("a" + ARCHIVE_SUFFIX)
    archive.parent.mkdir(parents=True, exist_ok=True)
    res = run_recovery(
        [
            "create",
            "--app-dir",
            str(repo),
            "--web-root",
            str(web),
            "--tasker-service",
            TASKER,
            "--archive",
            str(archive),
            "--storage-encryption-attested",
        ],
        hermetic,
    )
    assert res.returncode != 0
    assert "caddy" in res.stderr.lower()
    assert not hermetic.systemctl_log.exists()


def test_create_stops_then_restarts_when_initially_active(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    report = report_of(res)
    assert report["initially_active"] is True
    assert report["service_stopped"] is True
    assert report["service_started"] is True
    log = hermetic.systemctl_log.read_text(encoding="utf-8").splitlines()
    assert log == ["is-active portfolio-lab-tasker", "stop portfolio-lab-tasker", "start portfolio-lab-tasker"]


def test_create_does_not_stop_or_start_when_service_inactive(hermetic, tmp_path: Path):
    set_systemctl_mode(hermetic, "inactive")
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    report = report_of(res)
    assert report["initially_active"] is False
    assert report["service_stopped"] is False
    assert report["service_started"] is False
    log = hermetic.systemctl_log.read_text(encoding="utf-8").splitlines()
    assert log == ["is-active portfolio-lab-tasker"]


def test_create_stop_failure_aborts_but_attempts_restart(hermetic, tmp_path: Path):
    set_systemctl_mode(hermetic, "stop-fail")
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode != 0
    assert "stop" in res.stderr.lower()
    log = hermetic.systemctl_log.read_text(encoding="utf-8").splitlines()
    assert log.index("stop portfolio-lab-tasker") < log.index("start portfolio-lab-tasker")
    assert not archive.exists()


def test_create_verify_failure_keeps_archive_and_restarts(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    _write(repo / "data/market.db", b"garbage not a sqlite database" * 100)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode != 0
    assert "sqlite" in res.stderr.lower()
    assert archive.exists()
    log = hermetic.systemctl_log.read_text(encoding="utf-8").splitlines()
    assert log.index("stop portfolio-lab-tasker") < log.index("start portfolio-lab-tasker")


def test_create_restart_failure_reported(hermetic, tmp_path: Path):
    # Backward-compatible create output: the archive was produced and verified
    # before the restart attempt, so a failed systemd restart must not swallow
    # the report — stdout keeps the JSON create report (ok false,
    # service_started false, archive/report fields) and stderr reports the
    # systemctl start failure.
    set_systemctl_mode(hermetic, "start-fail")
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode != 0
    report = json.loads(res.stdout)
    assert report["ok"] is False
    assert report["service_started"] is False
    assert report["command"] == "create"
    assert report["service_controller"] == "systemd"
    assert report["initially_active"] is True
    assert report["service_stopped"] is True
    assert Path(report["archive"]) == archive
    assert "start" in res.stderr.lower()
    assert archive.exists()


def craft_archive(archive: Path, entries) -> None:
    """Write an adversarial uncompressed tar with stdlib tarfile.

    entries: list of (name, type, data_or_None, linkname_or_None)."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(archive), "w") as tf:
        for name, typ, data, linkname in entries:
            info = tarfile.TarInfo(name)
            info.type = typ
            info.mode = 0o644
            if data is not None:
                info.size = len(data)
            if linkname:
                info.linkname = linkname
            tf.addfile(info, io.BytesIO(data) if data is not None else None)


def craft_member_archive(hermetic: SimpleNamespace, name: str, entries) -> Path:
    """Craft a hostile archive with a matching sidecar; returns the path."""
    archive = hermetic.tmp / "backups" / name
    craft_archive(archive, entries)
    write_sidecar(archive)
    return archive


def test_verify_and_create_work_from_non_repo_cwd(hermetic, tmp_path: Path):
    """git bundle verify needs a repo context; the tool must not depend on
    the caller's cwd being a git repository."""
    neutral = tmp_path / "neutral"
    neutral.mkdir()
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive = hermetic.tmp / "backups" / ("b" + ARCHIVE_SUFFIX)
    archive.parent.mkdir(parents=True, exist_ok=True)
    _write(hermetic.units / f"{TASKER}.service", UNIT_TEXT)
    _write(repo / "data/tasker_status.json", json.dumps({"state": "idle"}) + "\n")
    res = run_recovery(
        [
            "create",
            "--app-dir",
            str(repo),
            "--web-root",
            str(web),
            "--tasker-service",
            TASKER,
            "--archive",
            str(archive),
            "--storage-encryption-attested",
        ],
        hermetic,
        cwd=neutral,
    )
    assert res.returncode == 0, res.stderr
    res = run_recovery(["verify", "--archive", str(archive)], hermetic, cwd=neutral)
    assert res.returncode == 0, res.stderr
    assert json.loads(res.stdout)["checks"]["bundle_ok"] is True


# ── verify ──────────────────────────────────────────────────────────────────


def test_verify_ok_archive(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha, generator_sha=source_sha[:12])
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    res = run_recovery(["verify", "--archive", str(archive)], hermetic)
    assert res.returncode == 0, res.stderr
    report = json.loads(res.stdout)
    checks = report["checks"]
    assert report["ok"] is True
    assert report["schema_version"] == SCHEMA
    assert report["source_sha"] == source_sha
    assert report["source_tasker_service"] == TASKER
    assert report["archive_sha256"] == _sha256_bytes(archive.read_bytes())
    assert report["manifest_sha256"] == _sha256_bytes(extract_tar_member(archive, "recovery-manifest.json"))
    assert checks["sidecar_ok"] is True
    assert checks["archive_type"] == "tar"
    assert checks["schema_ok"] is True
    assert checks["path_safe"] is True
    assert checks["members_match"] is True
    assert checks["allowed_members"] is True
    assert checks["required_members"] is True
    assert checks["digests_match"] is True
    assert checks["modes_match"] is True
    assert checks["bundle_ok"] is True
    assert checks["bundle_revision_matches"] is True
    assert checks["revision_consistent"] is True
    assert checks["sqlite_ok"] is True
    assert checks["static_provenance_coherent"] is True
    assert checks["data_index_generator_sha"] == source_sha[:12]
    assert checks["data_index_generator_reachable"] is True


def test_verify_rejects_corrupted_archive_bytes(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    data = bytearray(archive.read_bytes())
    data[len(data) // 2] ^= 0xFF
    archive.write_bytes(bytes(data))
    res = run_recovery(["verify", "--archive", str(archive)], hermetic)
    assert res.returncode != 0
    report = json.loads(res.stdout)
    assert report["checks"]["sidecar_ok"] is False


def test_verify_rejects_invalid_sidecar_digest(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    _write(Path(str(archive) + ".sha256"), "not a digest\n")
    res = run_recovery(["verify", "--archive", str(archive)], hermetic)
    assert res.returncode != 0
    assert "digest" in res.stderr.lower()


def test_verify_rejects_missing_sidecar(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    Path(str(archive) + ".sha256").unlink()
    res = run_recovery(["verify", "--archive", str(archive)], hermetic)
    assert res.returncode != 0
    assert "sidecar" in res.stderr.lower()


def test_verify_rejects_sidecar_filename_token_mismatch(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    # correct digest, wrong filename token -> strict grammar must reject
    _write(Path(str(archive) + ".sha256"), f"{_sha256_bytes(archive.read_bytes())}  other.tar\n")
    res = run_recovery(["verify", "--archive", str(archive)], hermetic)
    assert res.returncode != 0
    assert "sidecar" in res.stderr.lower()


def test_verify_rejects_sidecar_extra_content(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    sidecar = Path(str(archive) + ".sha256")
    _write(
        sidecar,
        f"{_sha256_bytes(archive.read_bytes())}  {archive.name}\ntrailing garbage\n",
    )
    res = run_recovery(["verify", "--archive", str(archive)], hermetic)
    assert res.returncode != 0
    assert "sidecar" in res.stderr.lower()


def test_create_archive_and_sidecar_are_0600(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    assert archive.stat().st_mode & 0o777 == 0o600
    assert (archive.parent / (archive.name + ".sha256")).stat().st_mode & 0o777 == 0o600


# ── verify: adversarial archives (stdlib tarfile, never extract*) ──────────


def test_verify_rejects_suffix_masked_hardlink_member(hermetic, tmp_path: Path):
    victim = tmp_path / "victim.txt"
    _write(victim, "original\n")
    archive = craft_member_archive(
        hermetic,
        "hardlink" + ARCHIVE_SUFFIX,
        [("static/web/index.html", tarfile.LNKTYPE, None, str(victim))],
    )
    res = run_recovery(["verify", "--archive", str(archive)], hermetic)
    assert res.returncode != 0
    report = json.loads(res.stdout)
    assert report["checks"]["path_safe"] is False
    # nothing was extracted or linked through: victim unchanged, no escape file
    assert (tmp_path / "static").exists() is False
    assert victim.read_text(encoding="utf-8") == "original\n"


def test_verify_rejects_suffix_masked_intermediate_symlink_escape(hermetic, tmp_path: Path):
    escape = tmp_path / "escaped.txt"
    archive = craft_member_archive(
        hermetic,
        "symlink-escape" + ARCHIVE_SUFFIX,
        [
            ("static/web/data", tarfile.SYMTYPE, None, str(escape)),
            ("static/web/data/index.json", tarfile.REGTYPE, b'{"x":1}\n', None),
        ],
    )
    res = run_recovery(["verify", "--archive", str(archive)], hermetic)
    assert res.returncode != 0
    report = json.loads(res.stdout)
    assert report["checks"]["path_safe"] is False
    assert not escape.exists()
    assert not (tmp_path / "static").exists()


def test_verify_rejects_absolute_member(hermetic, tmp_path: Path):
    archive = craft_member_archive(
        hermetic,
        "absolute" + ARCHIVE_SUFFIX,
        [("/etc/portfolio-lab-evil", tarfile.REGTYPE, b"x\n", None)],
    )
    res = run_recovery(["verify", "--archive", str(archive)], hermetic)
    assert res.returncode != 0
    assert json.loads(res.stdout)["checks"]["path_safe"] is False


def test_verify_rejects_traversal_member(hermetic, tmp_path: Path):
    archive = craft_member_archive(
        hermetic,
        "traversal" + ARCHIVE_SUFFIX,
        [("../portfolio-lab-escape.txt", tarfile.REGTYPE, b"x\n", None)],
    )
    res = run_recovery(["verify", "--archive", str(archive)], hermetic)
    assert res.returncode != 0
    assert json.loads(res.stdout)["checks"]["path_safe"] is False
    assert not (hermetic.tmp.parent / "portfolio-lab-escape.txt").exists()


def test_verify_rejects_backslash_member(hermetic, tmp_path: Path):
    archive = craft_member_archive(
        hermetic,
        "backslash" + ARCHIVE_SUFFIX,
        [("static\\web\\index.html", tarfile.REGTYPE, b"x\n", None)],
    )
    res = run_recovery(["verify", "--archive", str(archive)], hermetic)
    assert res.returncode != 0
    assert json.loads(res.stdout)["checks"]["path_safe"] is False


def test_verify_rejects_duplicate_members(hermetic, tmp_path: Path):
    archive = craft_member_archive(
        hermetic,
        "duplicate" + ARCHIVE_SUFFIX,
        [
            ("config/lab-app.env", tarfile.REGTYPE, b"A=1\n", None),
            ("config/lab-app.env", tarfile.REGTYPE, b"A=2\n", None),
        ],
    )
    res = run_recovery(["verify", "--archive", str(archive)], hermetic)
    assert res.returncode != 0
    report = json.loads(res.stdout)
    assert report["checks"]["path_safe"] is False
    assert "duplicate" in report["error"].lower()


def test_verify_rejects_directory_member(hermetic, tmp_path: Path):
    archive = craft_member_archive(
        hermetic,
        "dir" + ARCHIVE_SUFFIX,
        [("static", tarfile.DIRTYPE, None, None)],
    )
    res = run_recovery(["verify", "--archive", str(archive)], hermetic)
    assert res.returncode != 0
    assert json.loads(res.stdout)["checks"]["path_safe"] is False


def test_verify_rejects_topology_collision(hermetic, tmp_path: Path):
    archive = craft_member_archive(
        hermetic,
        "topology" + ARCHIVE_SUFFIX,
        [
            ("config", tarfile.REGTYPE, b"x\n", None),
            ("config/lab-app.env", tarfile.REGTYPE, b"A=1\n", None),
        ],
    )
    res = run_recovery(["verify", "--archive", str(archive)], hermetic)
    assert res.returncode != 0
    report = json.loads(res.stdout)
    assert report["checks"]["path_safe"] is False
    assert "topology" in report["error"].lower()


def test_verify_rejects_appledouble_member(hermetic, tmp_path: Path):
    archive = craft_member_archive(
        hermetic,
        "appledouble" + ARCHIVE_SUFFIX,
        [("static/web/._index.html", tarfile.REGTYPE, b"x\n", None)],
    )
    res = run_recovery(["verify", "--archive", str(archive)], hermetic)
    assert res.returncode != 0
    report = json.loads(res.stdout)
    assert report["checks"]["path_safe"] is False
    assert "appledouble" in report["error"].lower()


def test_restore_never_writes_outside_targets_for_hostile_archive(hermetic, tmp_path: Path):
    """Restore of a hostile archive must fail at verification and write
    nothing outside the requested targets (no extraction, no staging)."""
    escape = tmp_path / "escaped.txt"
    archive = craft_member_archive(
        hermetic,
        "hostile" + ARCHIVE_SUFFIX,
        [
            ("static/web/data", tarfile.SYMTYPE, None, str(escape)),
            ("static/web/data/index.json", tarfile.REGTYPE, b'{"x":1}\n', None),
            ("../escape.txt", tarfile.REGTYPE, b"x\n", None),
        ],
    )
    app_dir = hermetic.tmp / "app"
    web_root = hermetic.tmp / "www"
    res = run_recovery(
        [
            "restore",
            "--archive",
            str(archive),
            "--app-dir",
            str(app_dir),
            "--web-root",
            str(web_root),
            "--target-mode",
            "dev",
        ],
        hermetic,
    )
    assert res.returncode != 0
    assert not app_dir.exists()
    assert not web_root.exists()
    assert not escape.exists()
    assert not (tmp_path / "static").exists()
    assert not (hermetic.tmp.parent / "escape.txt").exists()


def test_verify_rejects_tampered_member_with_refreshed_sidecar(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr

    def mutate(work: Path) -> None:
        _write(work / "runtime/data/ensemble_weights.json", '{"normal": {"spy": 0.99}}\n')

    rebuilt = repackage(archive, hermetic.tmp / "retar", mutate)
    res = run_recovery(["verify", "--archive", str(rebuilt)], hermetic)
    assert res.returncode != 0
    report = json.loads(res.stdout)
    assert report["checks"]["sidecar_ok"] is True
    assert report["checks"]["digests_match"] is False


def test_verify_rejects_mode_mismatch(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr

    def mutate(work: Path) -> None:
        (work / "runtime/data/ensemble_weights.json").chmod(0o755)

    rebuilt = repackage(archive, hermetic.tmp / "retar", mutate)
    res = run_recovery(["verify", "--archive", str(rebuilt)], hermetic)
    assert res.returncode != 0
    report = json.loads(res.stdout)
    assert report["checks"]["modes_match"] is False


def test_verify_rejects_unsafe_recorded_mode(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr

    def mutate(work: Path) -> None:
        (work / "runtime/data/ensemble_weights.json").chmod(0o4755)

    rebuilt = repackage(archive, hermetic.tmp / "retar", mutate, recompute=True)
    res = run_recovery(["verify", "--archive", str(rebuilt)], hermetic)
    assert res.returncode != 0
    assert "mode" in res.stderr.lower()


def test_verify_rejects_unsafe_member_paths(hermetic, tmp_path: Path):
    staging = tmp_path / "staging"
    _write(staging / "ok.txt", "x\n")
    archive = hermetic.tmp / "backups" / ("unsafe" + ARCHIVE_SUFFIX)
    archive.parent.mkdir(parents=True, exist_ok=True)
    # Build the archive with Python's tarfile so the member name is stored
    # verbatim: GNU tar strips a leading ``../`` (``tar: Removing leading
    # '../' from member names``), which would silently turn this into a safe
    # archive and defeat the check on Linux hosts.
    payload = b"x\n"
    info = tarfile.TarInfo("../evil.txt")
    info.size = len(payload)
    with tarfile.open(archive, "w") as tf:
        tf.addfile(info, io.BytesIO(payload))
    write_sidecar(archive)
    res = run_recovery(["verify", "--archive", str(archive)], hermetic)
    assert res.returncode != 0
    report = json.loads(res.stdout)
    assert report["checks"]["path_safe"] is False


def test_verify_rejects_symlink_member(hermetic, tmp_path: Path):
    staging = tmp_path / "staging"
    _write(staging / "target.txt", "x\n")
    (staging / "link.txt").symlink_to("target.txt")
    archive = hermetic.tmp / "backups" / ("symlink" + ARCHIVE_SUFFIX)
    archive.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-cf", str(archive), "-C", str(staging), "link.txt"], check=True)
    write_sidecar(archive)
    res = run_recovery(["verify", "--archive", str(archive)], hermetic)
    assert res.returncode != 0
    report = json.loads(res.stdout)
    assert report["checks"]["path_safe"] is False


def test_verify_rejects_special_member_type(hermetic, tmp_path: Path):
    import os

    staging = tmp_path / "staging"
    staging.mkdir()
    fifo = staging / "fifo"
    os.mkfifo(fifo)
    archive = hermetic.tmp / "backups" / ("fifo" + ARCHIVE_SUFFIX)
    archive.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-cf", str(archive), "-C", str(staging), "fifo"], check=True)
    write_sidecar(archive)
    res = run_recovery(["verify", "--archive", str(archive)], hermetic)
    assert res.returncode != 0
    report = json.loads(res.stdout)
    assert report["checks"]["path_safe"] is False


def test_verify_rejects_schema_mismatch(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr

    def mutate(work: Path) -> None:
        manifest_path = work / "recovery-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["schema_version"] = "portfolio-lab-recovery/v1"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")

    rebuilt = repackage(archive, hermetic.tmp / "retar", mutate)
    res = run_recovery(["verify", "--archive", str(rebuilt)], hermetic)
    assert res.returncode != 0
    report = json.loads(res.stdout)
    assert report["checks"]["schema_ok"] is False


def test_verify_rejects_disallowed_member(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr

    def mutate(work: Path) -> None:
        _write(work / "logs/tasker.log", "generic log\n")

    rebuilt = repackage(archive, hermetic.tmp / "retar", mutate, recompute=True)
    res = run_recovery(["verify", "--archive", str(rebuilt)], hermetic)
    assert res.returncode != 0
    report = json.loads(res.stdout)
    assert report["checks"]["allowed_members"] is False


def test_verify_rejects_missing_required_members(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr

    def mutate(work: Path) -> None:
        (work / "tools/portfolio_lab_recovery.py").unlink()

    rebuilt = repackage(archive, hermetic.tmp / "retar", mutate, recompute=True)
    res = run_recovery(["verify", "--archive", str(rebuilt)], hermetic)
    assert res.returncode != 0
    report = json.loads(res.stdout)
    assert report["checks"]["required_members"] is False


def test_verify_sqlite_integrity_failure(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr

    def mutate(work: Path) -> None:
        _write(work / "runtime/data/market.db", b"garbage not a sqlite database" * 100)

    rebuilt = repackage(archive, hermetic.tmp / "retar", mutate, recompute=True)
    res = run_recovery(["verify", "--archive", str(rebuilt)], hermetic)
    assert res.returncode != 0
    report = json.loads(res.stdout)
    assert report["checks"]["sqlite_ok"] is False


def test_verify_generator_reachable_in_bundle_when_different_from_source(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    commit_all(repo, "first")
    _write(repo / "src/new.py", "x = 1\n")
    source_sha = commit_all(repo, "second")
    older_short = _git(repo, "rev-parse", "--short=12", "HEAD~1").stdout.strip()
    web = make_web_root(tmp_path / "web", source_sha, generator_sha=older_short)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    res = run_recovery(["verify", "--archive", str(archive)], hermetic)
    assert res.returncode == 0, res.stderr
    checks = json.loads(res.stdout)["checks"]
    assert checks["data_index_generator_sha"] == older_short
    assert checks["data_index_generator_reachable"] is True


def test_verify_reachability_clone_uses_no_checkout(hermetic, tmp_path: Path):
    """The reachability clone must skip the worktree checkout.

    Captures the shipped verify_archive's real git argv through the PLR_GIT
    wrapper seam: the reachability probe only needs the bundle object graph
    for rev-list --all, so `git clone` must be invoked with --no-checkout
    while still resolving the generator sha to exactly one reachable commit.
    """
    repo = make_repo(tmp_path / "repo")
    commit_all(repo, "first")
    _write(repo / "src/new.py", "x = 1\n")
    source_sha = commit_all(repo, "second")
    older_short = _git(repo, "rev-parse", "--short=12", "HEAD~1").stdout.strip()
    web = make_web_root(tmp_path / "web", source_sha, generator_sha=older_short)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    git_log = tmp_path / "git-argv.log"
    wrapper = make_fake(
        hermetic.bin / "git-wrapper",
        '#!/bin/sh\nprintf \'%s\\n\' "$@" >> "$PLR_GIT_LOG"\nexec git "$@"\n',
    )
    res = run_recovery(
        ["verify", "--archive", str(archive)],
        hermetic,
        extra_env={"PLR_GIT": str(wrapper), "PLR_GIT_LOG": str(git_log)},
    )
    assert res.returncode == 0, res.stderr
    checks = json.loads(res.stdout)["checks"]
    assert checks["data_index_generator_reachable"] is True
    argv = git_log.read_text(encoding="utf-8").splitlines()
    assert ["clone", "--no-checkout"] in [
        argv[i : i + 2] for i in range(len(argv) - 1)
    ], f"git clone argv missing --no-checkout: {argv}"


def test_verify_non_object_index_json_fails_closed_with_attribute_error(hermetic, tmp_path: Path):
    """Valid non-object data/index.json must keep the legacy AttributeError.

    The pre-refactor parser called ``.get()`` on the parsed payload with only
    OSError/ValueError caught, so valid non-object JSON raised an uncaught
    AttributeError. The shared helper preserves that exact failure at the
    verify call sites. Contrast: malformed (unparseable) JSON stays
    non-fatal, yielding a None generator sha (same as missing).
    """
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr

    def mutate_index(payload: str):
        def _mutate(work: Path) -> None:
            _write(work / "static/web/data/index.json", payload)

        return _mutate

    non_object = repackage(
        archive, hermetic.tmp / "retar-nonobject-index", mutate_index("[1, 2, 3]\n"), recompute=True
    )
    res = run_recovery(["verify", "--archive", str(non_object)], hermetic)
    assert res.returncode != 0
    assert "AttributeError" in res.stderr
    assert "'list' object has no attribute 'get'" in res.stderr

    malformed = repackage(
        archive, hermetic.tmp / "retar-malformed-index", mutate_index("{not json\n"), recompute=True
    )
    res = run_recovery(["verify", "--archive", str(malformed)], hermetic)
    assert res.returncode == 0, res.stderr
    checks = json.loads(res.stdout)["checks"]
    assert checks["data_index_generator_sha"] is None
    assert checks["data_index_generator_reachable"] is None


def test_verify_non_object_release_json_fails_closed_with_attribute_error(hermetic, tmp_path: Path):
    """Valid non-object _release.json must keep the legacy AttributeError."""
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr

    def mutate(work: Path) -> None:
        _write(work / "static/web/_release.json", "42\n")

    rebuilt = repackage(archive, hermetic.tmp / "retar-nonobject-release", mutate, recompute=True)
    res = run_recovery(["verify", "--archive", str(rebuilt)], hermetic)
    assert res.returncode != 0
    assert "AttributeError" in res.stderr
    assert "'int' object has no attribute 'get'" in res.stderr


def test_create_non_object_index_json_fails_closed_with_attribute_error(hermetic, tmp_path: Path):
    """create's manifest embedding reads data/index.json via the same helper;
    a valid non-object payload keeps the legacy uncaught AttributeError."""
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    _write(web / "data/index.json", "[1, 2, 3]\n")
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode != 0
    assert "AttributeError" in res.stderr
    assert "'list' object has no attribute 'get'" in res.stderr
    assert not archive.exists()


def test_verify_generator_unreachable_reported_not_blocking(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha, generator_sha="deadbeefdead")
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    res = run_recovery(["verify", "--archive", str(archive)], hermetic)
    assert res.returncode == 0, res.stderr
    checks = json.loads(res.stdout)["checks"]
    assert checks["data_index_generator_reachable"] is False


def test_verify_generator_malformed_reported_not_blocking(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha, generator_sha="not-a-sha!")
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    res = run_recovery(["verify", "--archive", str(archive)], hermetic)
    assert res.returncode == 0, res.stderr
    checks = json.loads(res.stdout)["checks"]
    assert checks["data_index_generator_reachable"] is False


def test_verify_generator_absent_reported_not_blocking(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    res = run_recovery(["verify", "--archive", str(archive)], hermetic)
    assert res.returncode == 0, res.stderr
    checks = json.loads(res.stdout)["checks"]
    assert checks["data_index_generator_sha"] is None
    assert checks["data_index_generator_reachable"] is None


def test_verify_static_provenance_mismatch_reported_not_blocking(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha, release_sha="f" * 40)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    res = run_recovery(["verify", "--archive", str(archive)], hermetic)
    assert res.returncode == 0, res.stderr
    checks = json.loads(res.stdout)["checks"]
    assert checks["static_provenance_coherent"] is False


# ── restore ─────────────────────────────────────────────────────────────────


def test_restore_verifies_before_any_target_mutation(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    data = bytearray(archive.read_bytes())
    data[len(data) // 2] ^= 0xFF
    archive.write_bytes(bytes(data))
    app_dir, web_root, res = standard_restore(hermetic, archive, "dev")
    assert res.returncode != 0
    assert not app_dir.exists()
    assert not web_root.exists()
    # restore made no systemctl calls: the log holds only create's calls
    log = hermetic.systemctl_log.read_text(encoding="utf-8").splitlines()
    assert log == ["is-active portfolio-lab-tasker", "stop portfolio-lab-tasker", "start portfolio-lab-tasker"]


def test_restore_dev_stages_app_and_web_without_service(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha, generator_sha=source_sha[:12])
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    app_dir, web_root, res = standard_restore(hermetic, archive, "dev")
    assert res.returncode == 0, res.stderr
    report = report_of(res)
    assert report["ok"] is True
    assert report["mode"] == "dev"
    assert report["staged"] is True
    assert report["service_started"] is False
    assert report["config_restored"] is False
    assert "config" in report["config_note"].lower()
    assert _git(app_dir, "rev-parse", "HEAD").stdout.strip() == source_sha
    assert (app_dir / "data/ensemble_weights.json").is_file()
    assert (app_dir / "data/market.db").is_file()
    # dev restores never overlay the archived config; the checkout's own
    # committed config remains (target-specific non-secret config is
    # supplied separately — see test_restore_dev_does_not_overlay_archived_config)
    assert (app_dir / "config/lab-app.env").is_file()
    assert (web_root / "_release.json").is_file()
    assert (web_root / "data/prices.json").is_file()
    state = app_dir / ".portfolio-lab-recovery"
    assert (state / "recovery-manifest.json").is_file()
    assert (state / "metadata/tasker-unit.txt").is_file()
    assert (state / "metadata/caddy-portfolio-lab-block.txt").is_file()
    assert (state / "tools/portfolio_lab_recovery.py").is_file()
    restore_report = json.loads((state / "restore-report.json").read_text(encoding="utf-8"))
    assert restore_report["schema_version"] == SCHEMA
    assert restore_report["verified"] is True
    assert restore_report["source_sha"] == source_sha
    assert restore_report["target_mode"] == "dev"
    assert restore_report["config_restored"] is False
    # report binds the archive and manifest digests from canonical verify
    assert restore_report["archive_sha256"] == _sha256_bytes(archive.read_bytes())
    assert restore_report["manifest_sha256"] == _sha256_bytes(
        extract_tar_member(archive, "recovery-manifest.json")
    )
    # restore itself made no systemctl calls: the log holds only create's calls
    log = hermetic.systemctl_log.read_text(encoding="utf-8").splitlines()
    assert log == ["is-active portfolio-lab-tasker", "stop portfolio-lab-tasker", "start portfolio-lab-tasker"]


def test_restore_dev_rejects_production_paths(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    for label, flag, bad, good in (
        ("app", "--app-dir", "/root/projects/portfolio-lab", str(hermetic.tmp / "app")),
        ("web", "--web-root", "/var/www/portfolio-lab", str(hermetic.tmp / "www")),
    ):
        args = {
            "--app-dir": str(hermetic.tmp / "app"),
            "--web-root": str(hermetic.tmp / "www"),
        }
        args[flag] = bad
        res = run_recovery(
            [
                "restore",
                "--archive",
                str(archive),
                "--app-dir",
                args["--app-dir"],
                "--web-root",
                args["--web-root"],
                "--target-mode",
                "dev",
            ],
            hermetic,
        )
        assert res.returncode != 0, label
        assert "production" in res.stderr.lower(), label
        # rejected before any target mutation
        assert not (hermetic.tmp / "app").exists(), label
        assert not (hermetic.tmp / "www").exists(), label


def test_restore_requires_absolute_target_paths(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    res = run_recovery(
        [
            "restore",
            "--archive",
            str(archive),
            "--app-dir",
            "relative/app",
            "--web-root",
            str(hermetic.tmp / "www"),
            "--target-mode",
            "dev",
        ],
        hermetic,
    )
    assert res.returncode != 0
    assert "absolute" in res.stderr.lower()


def test_restore_prod_requires_allow_production_paths(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    app_dir, web_root, res = standard_restore(hermetic, archive, "prod")
    assert res.returncode != 0
    assert "allow-production-paths" in res.stderr
    assert not app_dir.exists()
    assert not web_root.exists()


def test_restore_prod_uses_archived_service_without_optional_override(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    set_systemctl_mode(hermetic, "inactive")

    app_dir, web_root, res = standard_restore(
        hermetic, archive, "prod", extra=["--allow-production-paths"]
    )

    assert res.returncode == 0, res.stderr
    assert (app_dir / "README.md").is_file()
    assert (web_root / "_release.json").is_file()
    assert hermetic.systemctl_log.read_text(encoding="utf-8").splitlines()[-1] == f"is-active {TASKER}"


def test_restore_prod_rejects_active_target_service_before_mutation(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    app_dir = hermetic.tmp / "active-app"
    web_root = hermetic.tmp / "active-www"
    _write(app_dir / "old.py", "old app\n")
    _write(web_root / "index.html", "old web\n")

    res = run_recovery(
        [
            "restore",
            "--archive",
            str(archive),
            "--app-dir",
            str(app_dir),
            "--web-root",
            str(web_root),
            "--target-mode",
            "prod",
            "--allow-production-paths",
            "--tasker-service",
            TASKER,
        ],
        hermetic,
    )

    assert res.returncode != 0
    assert "inactive" in res.stderr.lower()
    assert (app_dir / "old.py").read_text(encoding="utf-8") == "old app\n"
    assert (web_root / "index.html").read_text(encoding="utf-8") == "old web\n"


def test_restore_prod_stages_only_keeps_rollback_dirs(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    set_systemctl_mode(hermetic, "inactive")
    app_dir, web_root, res = standard_restore(hermetic, archive, "prod", extra=["--allow-production-paths"])
    assert res.returncode == 0, res.stderr
    # replace the previously restored targets: existing content must be kept in rollback dirs
    _write(app_dir / "old.py", "old\n")
    _write(web_root / "index.html", "<html>old</html>\n")
    app_dir, web_root, res = standard_restore(hermetic, archive, "prod", extra=["--allow-production-paths"])
    assert res.returncode == 0, res.stderr
    report = report_of(res)
    assert report["mode"] == "prod"
    assert report["staged"] is True
    assert report["service_started"] is False
    assert report["config_restored"] is True
    assert "activate" in report["activation_note"].lower()
    assert report["rollback_app_dir"] is not None
    assert report["rollback_web_root"] is not None
    rb_app = Path(report["rollback_app_dir"])
    rb_web = Path(report["rollback_web_root"])
    assert rb_app.is_dir() and (rb_app / "old.py").read_text(encoding="utf-8") == "old\n"
    assert rb_web.is_dir() and (rb_web / "index.html").read_text(encoding="utf-8") == "<html>old</html>\n"
    saved_report = json.loads((app_dir / ".portfolio-lab-recovery" / "restore-report.json").read_text())
    assert saved_report["rollback_app_dir"] == report["rollback_app_dir"]
    assert saved_report["rollback_web_root"] == report["rollback_web_root"]
    assert (web_root / "_release.json").is_file()
    assert (app_dir / "data/ensemble_weights.json").is_file()
    assert (app_dir / "config/lab-app.env").is_file()
    log = hermetic.systemctl_log.read_text(encoding="utf-8").splitlines()
    assert log[-2:] == [f"is-active {TASKER}", f"is-active {TASKER}"]


def test_restore_replaces_empty_target_dirs_without_nesting_staging_trees(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    set_systemctl_mode(hermetic, "inactive")
    app_dir = hermetic.tmp / "empty-app"
    web_root = hermetic.tmp / "empty-www"
    app_dir.mkdir()
    web_root.mkdir()

    app_dir, web_root, res = standard_restore(
        hermetic,
        archive,
        "prod",
        extra=["--allow-production-paths"],
        app_name=app_dir.name,
        web_name=web_root.name,
    )

    assert res.returncode == 0, res.stderr
    report = report_of(res)
    assert (app_dir / "README.md").is_file()
    assert (web_root / "_release.json").is_file()
    assert not (app_dir / "checkout").exists()
    assert not (web_root / "web").exists()
    assert report["rollback_app_dir"] is not None
    assert report["rollback_web_root"] is not None
    saved_report = json.loads((app_dir / ".portfolio-lab-recovery" / "restore-report.json").read_text())
    assert saved_report["rollback_app_dir"] == report["rollback_app_dir"]
    assert saved_report["rollback_web_root"] == report["rollback_web_root"]


def test_restore_start_dev_api_uses_no_scheduler_unit(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    app_dir, web_root, res = standard_restore(hermetic, archive, "dev", extra=["--start-dev-api"])
    assert res.returncode == 0, res.stderr
    report = report_of(res)
    assert report["service_started"] is True
    assert report["service_name"] == DEV_SERVICE
    unit = hermetic.units / f"{DEV_SERVICE}.service"
    assert unit.is_file(), list(hermetic.units.iterdir())
    content = unit.read_text(encoding="utf-8")
    assert "TASKER_DISABLE_SCHEDULER=1" in content
    assert "--no-scheduler" in content
    assert "--host 127.0.0.1" in content
    assert f"WorkingDirectory={app_dir}" in content
    assert f"Environment=PORTFOLIO_LAB_PROJECT_DIR={app_dir}" in content
    # the env-flag line must follow EnvironmentFile so .env.local cannot override it
    assert content.index("EnvironmentFile=-") < content.index("Environment=TASKER_DISABLE_SCHEDULER=1")
    log = hermetic.systemctl_log.read_text(encoding="utf-8").splitlines()
    assert "daemon-reload" in log
    assert f"enable {DEV_SERVICE}" in log
    assert f"restart {DEV_SERVICE}" in log
    # create restarted the source service; restore starts only the distinct dev unit.
    assert [line for line in log if line.startswith("start ")] == ["start portfolio-lab-tasker"]
    assert f"restart {DEV_SERVICE}" in log


def test_restore_start_dev_api_restarts_existing_dev_service(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr

    app_dir, web_root, res = standard_restore(hermetic, archive, "dev", extra=["--start-dev-api"])

    assert res.returncode == 0, res.stderr
    log = hermetic.systemctl_log.read_text(encoding="utf-8").splitlines()
    assert f"restart {DEV_SERVICE}" in log
    assert f"start {DEV_SERVICE}" not in log


def test_restore_start_dev_api_honors_tasker_service_override(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    app_dir, web_root, res = standard_restore(
        hermetic, archive, "dev", extra=["--start-dev-api", "--tasker-service", "my-recovery-dev"]
    )
    assert res.returncode == 0, res.stderr
    assert (hermetic.units / "my-recovery-dev.service").is_file()


def test_restore_start_dev_api_rejected_in_prod_mode(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    res = run_recovery(
        [
            "restore",
            "--archive",
            str(archive),
            "--app-dir",
            str(hermetic.tmp / "app"),
            "--web-root",
            str(hermetic.tmp / "www"),
            "--target-mode",
            "prod",
            "--allow-production-paths",
            "--start-dev-api",
        ],
        hermetic,
    )
    assert res.returncode != 0
    assert "dev" in res.stderr.lower()


# ── restore: bundle / staging / target safety ───────────────────────────────


def test_restore_accepts_bundle_tracked_file_symlink(hermetic, tmp_path: Path):
    """A committed Git symlink whose raw target is a relative path to an
    ordinary non-symlink file inside the checkout (git mode 120000; the
    authoritative repo legitimately contains AGENTS.md -> CLAUDE.md) must be
    accepted by restore and reproduced exactly at the target."""
    repo = make_repo(tmp_path / "repo")
    _write(repo / "CLAUDE.md", "# claude content\n")
    os.symlink("CLAUDE.md", repo / "AGENTS.md")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    app_dir, web_root, res = standard_restore(hermetic, archive, "dev")
    assert res.returncode == 0, res.stderr
    restored = app_dir / "AGENTS.md"
    assert restored.is_symlink()
    assert os.readlink(restored) == "CLAUDE.md"
    assert (app_dir / "CLAUDE.md").is_file()
    assert restored.resolve() == (app_dir / "CLAUDE.md").resolve()


def test_restore_rejects_tracked_symlink_escaping_checkout(hermetic, tmp_path: Path):
    """A committed symlink whose relative raw target escapes the checkout is
    still rejected before any target mutation."""
    repo = make_repo(tmp_path / "repo")
    _write(tmp_path / "outside.txt", "outside\n")
    os.symlink("../outside.txt", repo / "link.md")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    app_dir, web_root, res = standard_restore(hermetic, archive, "dev")
    assert res.returncode != 0
    assert "symlink" in res.stderr.lower()
    assert not app_dir.exists()
    assert not web_root.exists()


def test_restore_rejects_bundle_gitlink(hermetic, tmp_path: Path):
    """A gitlink (submodule) entry in the bundle must block restore."""
    sub = make_repo(tmp_path / "sub")
    commit_all(sub)
    repo = make_repo(tmp_path / "repo")
    subprocess.run(
        ["git", "-C", str(repo), "-c", "protocol.file.allow=always", "submodule", "add", str(sub), "submod"],
        check=True,
        capture_output=True,
    )
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    app_dir, web_root, res = standard_restore(hermetic, archive, "dev")
    assert res.returncode != 0
    assert "gitlink" in res.stderr.lower() or "symlink" in res.stderr.lower()
    assert not app_dir.exists()
    assert not web_root.exists()


# ── restore: scan_checkout_unsafe file-symlink contract ────────────────────


def test_scan_checkout_unsafe_accepts_tracked_file_symlink(monkeypatch, tmp_path: Path):
    """Scan-level acceptance: AGENTS.md -> CLAUDE.md yields no bad entries —
    not flagged as a tracked 120000 nor double-reported as an untracked
    symlink."""
    monkeypatch.setenv("PLR_GIT", "git")
    repo = make_repo(tmp_path / "repo")
    _write(repo / "CLAUDE.md", "# claude content\n")
    os.symlink("CLAUDE.md", repo / "AGENTS.md")
    commit_all(repo)
    mod = _load_recovery_module()
    assert mod.scan_checkout_unsafe(repo) == []


def test_scan_checkout_unsafe_rejects_tracked_symlink_absolute_target(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PLR_GIT", "git")
    repo = make_repo(tmp_path / "repo")
    os.symlink("/etc/hosts", repo / "abs.md")
    commit_all(repo)
    mod = _load_recovery_module()
    bad = mod.scan_checkout_unsafe(repo)
    assert any(entry.split()[0] == "abs.md" for entry in bad)


def test_scan_checkout_unsafe_rejects_tracked_symlink_broken_target(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PLR_GIT", "git")
    repo = make_repo(tmp_path / "repo")
    os.symlink("missing-target.txt", repo / "broken.md")
    commit_all(repo)
    mod = _load_recovery_module()
    bad = mod.scan_checkout_unsafe(repo)
    assert any(entry.split()[0] == "broken.md" for entry in bad)


def test_scan_checkout_unsafe_rejects_tracked_symlink_directory_target(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PLR_GIT", "git")
    repo = make_repo(tmp_path / "repo")
    (repo / "subdir").mkdir()
    _write(repo / "subdir" / "inner.txt", "x\n")
    os.symlink("subdir", repo / "dirlink")
    commit_all(repo)
    mod = _load_recovery_module()
    bad = mod.scan_checkout_unsafe(repo)
    assert any(entry.split()[0] == "dirlink" for entry in bad)


def test_scan_checkout_unsafe_rejects_tracked_symlink_chain(monkeypatch, tmp_path: Path):
    """A tracked symlink whose raw target is itself a symlink (chain) is
    rejected, while the terminal file symlink in the chain is allowed."""
    monkeypatch.setenv("PLR_GIT", "git")
    repo = make_repo(tmp_path / "repo")
    _write(repo / "f.txt", "content\n")
    os.symlink("f.txt", repo / "b")
    os.symlink("b", repo / "a")
    commit_all(repo)
    mod = _load_recovery_module()
    bad = mod.scan_checkout_unsafe(repo)
    assert any(entry.split()[0] == "a" for entry in bad)
    assert not any(entry.split()[0] == "b" for entry in bad)


def test_scan_checkout_unsafe_rejects_untracked_symlink(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PLR_GIT", "git")
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    (repo / "untracked-link").symlink_to("README.md")
    mod = _load_recovery_module()
    bad = mod.scan_checkout_unsafe(repo)
    assert any(
        entry.split()[0] == "untracked-link" and "untracked symlink" in entry for entry in bad
    )


def test_scan_checkout_unsafe_rejects_hardlinked_regular_file(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PLR_GIT", "git")
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    os.link(repo / "README.md", repo / "hard.md")
    commit_all(repo)
    mod = _load_recovery_module()
    bad = mod.scan_checkout_unsafe(repo)
    assert any(entry.split()[0] == "README.md" and "hardlinked" in entry for entry in bad)


def test_restore_clone_failure_leaves_existing_target_intact(hermetic, tmp_path: Path):
    """A bundle clone failure must leave an existing target completely
    intact: no rollback, no partial replacement, no state dir."""
    fake_git = hermetic.bin / "git-wrapper"
    make_fake(
        fake_git,
        '#!/bin/sh\nif [ "$1" = "clone" ]; then echo "clone failed (fake)" >&2; exit 1; fi\nexec git "$@"\n',
    )
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    # generator == source sha keeps verify on the fast path (no clone there)
    web = make_web_root(tmp_path / "web", source_sha, generator_sha=source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    app_dir, web_root, res = standard_restore(
        hermetic, archive, "dev", extra_env={"PLR_GIT": str(fake_git)}
    )
    assert res.returncode != 0
    assert "clone" in res.stderr.lower()
    assert not app_dir.exists()
    assert not web_root.exists()
    assert not list(hermetic.tmp.glob("*.rollback-*"))


def test_restore_placement_failure_rolls_back_moved_target(hermetic, tmp_path: Path):
    """If placement of the staged trees fails after a target was moved aside,
    the moved-aside target must be rolled back to its original content."""
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha, generator_sha=source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    app_dir = hermetic.tmp / "app"
    _write(app_dir / "existing.txt", "keep me\n")
    ro = hermetic.tmp / "ro"
    # A regular FILE where the web-root directory is required: shutil.move
    # fails with NotADirectoryError even when running as root (a chmod-based
    # read-only parent is bypassed by root, so it cannot drive this path).
    _write(ro, "x\n")
    web_root = ro / "www"
    try:
        res = run_recovery(
            [
                "restore",
                "--archive",
                str(archive),
                "--app-dir",
                str(app_dir),
                "--web-root",
                str(web_root),
                "--target-mode",
                "dev",
            ],
            hermetic,
        )
        assert res.returncode != 0
        assert "place" in res.stderr.lower() or "failed" in res.stderr.lower()
        # the moved-aside app target was rolled back with its content intact
        assert (app_dir / "existing.txt").read_text(encoding="utf-8") == "keep me\n"
        assert not (app_dir / "data").exists()
        assert not web_root.exists()
        assert not list(hermetic.tmp.glob("*.rollback-*"))
    finally:
        ro.unlink(missing_ok=True)


def test_restore_clone_failure_preserves_existing_target_content(hermetic, tmp_path: Path):
    fake_git = hermetic.bin / "git-wrapper"
    make_fake(
        fake_git,
        '#!/bin/sh\nif [ "$1" = "clone" ]; then echo "clone failed (fake)" >&2; exit 1; fi\nexec git "$@"\n',
    )
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha, generator_sha=source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    app_dir = hermetic.tmp / "app"
    _write(app_dir / "existing.txt", "keep me\n")
    web_root = hermetic.tmp / "www"
    res = run_recovery(
        [
            "restore",
            "--archive",
            str(archive),
            "--app-dir",
            str(app_dir),
            "--web-root",
            str(web_root),
            "--target-mode",
            "dev",
        ],
        hermetic,
        extra_env={"PLR_GIT": str(fake_git)},
    )
    assert res.returncode != 0
    assert (app_dir / "existing.txt").read_text(encoding="utf-8") == "keep me\n"
    assert not web_root.exists()
    assert not list(hermetic.tmp.glob("*.rollback-*"))


def test_restore_rejects_overlapping_targets(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    base = ["restore", "--archive", str(archive)]
    for label, app, www in (
        ("identical", str(hermetic.tmp / "x"), str(hermetic.tmp / "x")),
        ("nested", str(hermetic.tmp / "x"), str(hermetic.tmp / "x" / "www")),
        ("parent", str(hermetic.tmp / "x" / "app"), str(hermetic.tmp / "x")),
    ):
        res = run_recovery(
            [*base, "--app-dir", app, "--web-root", www, "--target-mode", "dev"],
            hermetic,
        )
        assert res.returncode != 0, label
        assert "overlapping" in res.stderr.lower() or "distinct" in res.stderr.lower(), label
        assert not (hermetic.tmp / "x").exists(), label


def test_restore_rejects_symlink_target(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    real = hermetic.tmp / "www-real"
    real.mkdir()
    link = hermetic.tmp / "www"
    link.symlink_to(real)
    res = run_recovery(
        [
            "restore",
            "--archive",
            str(archive),
            "--app-dir",
            str(hermetic.tmp / "app"),
            "--web-root",
            str(link),
            "--target-mode",
            "dev",
        ],
        hermetic,
    )
    assert res.returncode != 0
    assert "symlink" in res.stderr.lower()
    assert not (hermetic.tmp / "app").exists()


def test_restore_dev_api_rejects_production_service_name(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    app_dir, web_root, res = standard_restore(
        hermetic, archive, "dev", extra=["--start-dev-api", "--tasker-service", TASKER]
    )
    assert res.returncode != 0
    assert "production" in res.stderr.lower()
    # rejected before any target mutation
    assert not app_dir.exists()
    assert not web_root.exists()
    assert (hermetic.units / f"{TASKER}.service").read_text(encoding="utf-8") == UNIT_TEXT


def test_restore_rejects_control_chars_in_unit_paths(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    bad_app = hermetic.tmp / "bad\napp"
    res = run_recovery(
        [
            "restore",
            "--archive",
            str(archive),
            "--app-dir",
            str(bad_app),
            "--web-root",
            str(hermetic.tmp / "www"),
            "--target-mode",
            "dev",
            "--start-dev-api",
        ],
        hermetic,
    )
    assert res.returncode != 0
    assert "control" in res.stderr.lower()
    assert not bad_app.exists()


def test_restore_dev_does_not_overlay_archived_config(hermetic, tmp_path: Path):
    """The archived config/lab-app.env must never be written into a dev
    target; the checkout's committed config stays authoritative."""
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    committed_config = (repo / "config/lab-app.env").read_text(encoding="utf-8")
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr

    def mutate(work: Path) -> None:
        _write(work / "config/lab-app.env", "TASKER_SERVICE_NAME=tampered-config\n")

    rebuilt = repackage(archive, hermetic.tmp / "retar", mutate, recompute=True)
    res = run_recovery(["verify", "--archive", str(rebuilt)], hermetic)
    assert res.returncode == 0, res.stderr
    app_dir, web_root, res = standard_restore(hermetic, rebuilt, "dev")
    assert res.returncode == 0, res.stderr
    report = report_of(res)
    assert report["config_restored"] is False
    assert (app_dir / "config/lab-app.env").read_text(encoding="utf-8") == committed_config


def test_restore_prod_overlays_archived_config(hermetic, tmp_path: Path):
    """In prod mode the archived config/lab-app.env is restored."""
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    set_systemctl_mode(hermetic, "inactive")

    def mutate(work: Path) -> None:
        _write(work / "config/lab-app.env", "TASKER_SERVICE_NAME=prod-config\n")

    rebuilt = repackage(archive, hermetic.tmp / "retar", mutate, recompute=True)
    app_dir, web_root, res = standard_restore(
        hermetic, rebuilt, "prod", extra=["--allow-production-paths"]
    )
    assert res.returncode == 0, res.stderr
    report = report_of(res)
    assert report["config_restored"] is True
    assert (
        app_dir / "config/lab-app.env"
    ).read_text(encoding="utf-8") == "TASKER_SERVICE_NAME=prod-config\n"


def test_verify_and_restore_require_absolute_archive_path(hermetic, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    res = run_recovery(["verify", "--archive", "relative" + ARCHIVE_SUFFIX], hermetic)
    assert res.returncode != 0
    assert "absolute" in res.stderr.lower()
    res = run_recovery(
        [
            "restore",
            "--archive",
            "relative" + ARCHIVE_SUFFIX,
            "--app-dir",
            str(hermetic.tmp / "app"),
            "--web-root",
            str(hermetic.tmp / "www"),
            "--target-mode",
            "dev",
        ],
        hermetic,
    )
    assert res.returncode != 0
    assert "absolute" in res.stderr.lower()


# ── activate-prod ───────────────────────────────────────────────────────────


def test_activate_requires_confirmations(hermetic, tmp_path: Path):
    archive, app_dir, web_root, _ = _create_and_restore(hermetic, tmp_path)
    base = [
        "activate-prod",
        "--app-dir",
        str(app_dir),
        "--web-root",
        str(web_root),
        "--tasker-service",
        TASKER,
    ]
    res = run_recovery([*base, "--former-authority-confirmed-stopped", "old-host"], hermetic)
    assert res.returncode != 0
    assert "--confirm-authoritative-activation" in res.stderr
    res = run_recovery([*base, "--confirm-authoritative-activation"], hermetic)
    assert res.returncode != 0
    assert "--former-authority-confirmed-stopped" in res.stderr
    res = run_recovery(
        [*base, "--confirm-authoritative-activation", "--former-authority-confirmed-stopped", ""], hermetic
    )
    assert res.returncode != 0
    assert_no_activation_systemctl(hermetic)
    assert (hermetic.units / f"{TASKER}.service").read_text(encoding="utf-8") == UNIT_TEXT


def test_activate_requires_restore_report(hermetic, tmp_path: Path):
    app_dir = hermetic.tmp / "app"
    _write(app_dir / ".env.local", "PORTFOLIO_LAB_MODE=lab\n")
    res = run_activate(hermetic, app_dir, hermetic.tmp / "www")
    assert res.returncode != 0
    assert "report" in res.stderr.lower()
    assert_no_activation_systemctl(hermetic)


def test_activate_rejects_dev_restore_report(hermetic, tmp_path: Path):
    """A dev restore report must never be activatable, before any systemctl."""
    set_systemctl_mode(hermetic, "inactive")
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha, generator_sha=source_sha[:12])
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    app_dir, web_root, res = standard_restore(hermetic, archive, "dev")
    assert res.returncode == 0, res.stderr
    _write(app_dir / ".env.local", "PORTFOLIO_LAB_MODE=lab\n")
    res = run_activate(hermetic, app_dir, web_root)
    assert res.returncode != 0
    assert "prod" in res.stderr.lower()
    assert_no_activation_systemctl(hermetic)


def test_activate_rejects_wrong_service_name(hermetic, tmp_path: Path):
    """activate-prod service identity must bind to the archived source service."""
    set_systemctl_mode(hermetic, "inactive")
    archive, app_dir, web_root, _ = _create_and_restore(hermetic, tmp_path)
    res = run_recovery(
        [
            "activate-prod",
            "--app-dir",
            str(app_dir),
            "--web-root",
            str(web_root),
            "--tasker-service",
            "some-other-service",
            "--confirm-authoritative-activation",
            "--former-authority-confirmed-stopped",
            "old-host",
        ],
        hermetic,
    )
    assert res.returncode != 0
    assert "service" in res.stderr.lower()
    assert_no_activation_systemctl(hermetic)


def test_activate_reverifies_archive_before_trusting_report(hermetic, tmp_path: Path):
    """Tampering with the archive after restore must block activation."""
    set_systemctl_mode(hermetic, "inactive")
    archive, app_dir, web_root, _ = _create_and_restore(hermetic, tmp_path)
    data = bytearray(archive.read_bytes())
    data[len(data) // 2] ^= 0xFF
    archive.write_bytes(bytes(data))
    res = run_activate(hermetic, app_dir, web_root)
    assert res.returncode != 0
    assert "verification" in res.stderr.lower()
    assert_no_activation_systemctl(hermetic)


def test_activate_rejects_tampered_state_manifest(hermetic, tmp_path: Path):
    """The staged recovery-manifest.json must be byte-identical to the
    verified archive's manifest (empty member list cannot bypass checks)."""
    set_systemctl_mode(hermetic, "inactive")
    archive, app_dir, web_root, _ = _create_and_restore(hermetic, tmp_path)
    state_manifest = app_dir / ".portfolio-lab-recovery" / "recovery-manifest.json"
    manifest = json.loads(state_manifest.read_text(encoding="utf-8"))
    manifest["members"] = []
    state_manifest.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    res = run_activate(hermetic, app_dir, web_root)
    assert res.returncode != 0
    assert "manifest" in res.stderr.lower()
    assert_no_activation_systemctl(hermetic)


def test_activate_requires_clean_tracked_checkout(hermetic, tmp_path: Path):
    """Uncommitted tracked changes in the restored checkout block activation."""
    set_systemctl_mode(hermetic, "inactive")
    archive, app_dir, web_root, _ = _create_and_restore(hermetic, tmp_path)
    _write(app_dir / "README.md", "# modified after restore\n")
    res = run_activate(hermetic, app_dir, web_root)
    assert res.returncode != 0
    assert "clean" in res.stderr.lower() or "uncommitted" in res.stderr.lower()
    assert_no_activation_systemctl(hermetic)


def test_activate_rejects_whitespace_authority_label(hermetic, tmp_path: Path):
    set_systemctl_mode(hermetic, "inactive")
    archive, app_dir, web_root, _ = _create_and_restore(hermetic, tmp_path)
    res = run_recovery(
        [
            "activate-prod",
            "--app-dir",
            str(app_dir),
            "--web-root",
            str(web_root),
            "--tasker-service",
            TASKER,
            "--confirm-authoritative-activation",
            "--former-authority-confirmed-stopped",
            "   ",
        ],
        hermetic,
    )
    assert res.returncode != 0
    assert "former-authority" in res.stderr
    assert_no_activation_systemctl(hermetic)


def test_activate_blocks_incoherent_static_provenance(hermetic, tmp_path: Path):
    set_systemctl_mode(hermetic, "inactive")
    archive, app_dir, web_root, _ = _create_and_restore(
        hermetic, tmp_path, release_sha="f" * 40
    )
    res = run_activate(hermetic, app_dir, web_root)
    assert res.returncode != 0
    assert "provenance" in res.stderr.lower()
    assert (hermetic.units / f"{TASKER}.service").read_text(encoding="utf-8") == UNIT_TEXT
    assert_no_activation_systemctl(hermetic)


def test_activate_blocks_generator_unreachable(hermetic, tmp_path: Path):
    set_systemctl_mode(hermetic, "inactive")
    archive, app_dir, web_root, _ = _create_and_restore(hermetic, tmp_path, generator_sha="deadbeefdead")
    res = run_activate(hermetic, app_dir, web_root)
    assert res.returncode != 0
    assert "generator" in res.stderr.lower()
    assert_no_activation_systemctl(hermetic)


def test_activate_blocks_generator_absent(hermetic, tmp_path: Path):
    set_systemctl_mode(hermetic, "inactive")
    archive, app_dir, web_root, _ = _create_and_restore(hermetic, tmp_path, generator_sha=_NO_GENERATOR)
    res = run_activate(hermetic, app_dir, web_root)
    assert res.returncode != 0
    assert "generator" in res.stderr.lower()
    assert_no_activation_systemctl(hermetic)


def test_activate_requires_env_local(hermetic, tmp_path: Path):
    set_systemctl_mode(hermetic, "inactive")
    archive, app_dir, web_root, _ = _create_and_restore(hermetic, tmp_path)
    (app_dir / ".env.local").unlink()
    res = run_activate(hermetic, app_dir, web_root)
    assert res.returncode != 0
    assert ".env.local" in res.stderr
    assert_no_activation_systemctl(hermetic)


def test_activate_blocks_when_target_service_active(hermetic, tmp_path: Path):
    archive, app_dir, web_root, _ = _create_and_restore(hermetic, tmp_path)
    set_systemctl_mode(hermetic, "active")
    res = run_activate(hermetic, app_dir, web_root)
    assert res.returncode != 0
    assert "inactive" in res.stderr.lower()
    # the archived unit was not installed and nothing was enabled/started by
    # activation (create's own is-active/stop/start calls are expected)
    assert (hermetic.units / f"{TASKER}.service").read_text(encoding="utf-8") == UNIT_TEXT
    log = hermetic.systemctl_log.read_text(encoding="utf-8")
    assert "daemon-reload" not in log
    assert "enable portfolio-lab-tasker" not in log


def test_activate_blocks_when_target_service_failed(hermetic, tmp_path: Path):
    archive, app_dir, web_root, _ = _create_and_restore(hermetic, tmp_path)
    set_systemctl_mode(hermetic, "failed")
    res = run_activate(hermetic, app_dir, web_root)
    assert res.returncode != 0
    assert "inactive" in res.stderr.lower()


def test_activate_blocks_tampered_restored_tree(hermetic, tmp_path: Path):
    set_systemctl_mode(hermetic, "inactive")
    archive, app_dir, web_root, _ = _create_and_restore(hermetic, tmp_path)
    _write(web_root / "data/prices.json", '{"prices": [1]}\n')
    res = run_activate(hermetic, app_dir, web_root)
    assert res.returncode != 0
    assert "mismatch" in res.stderr.lower()
    assert_no_activation_systemctl(hermetic)


def test_activate_blocks_wrong_checkout_head(hermetic, tmp_path: Path):
    set_systemctl_mode(hermetic, "inactive")
    archive, app_dir, web_root, _ = _create_and_restore(hermetic, tmp_path)
    _git(app_dir, "config", "user.email", "test@example.com")
    _git(app_dir, "config", "user.name", "Test")
    _git(app_dir, "commit", "--allow-empty", "-m", "drift")
    res = run_activate(hermetic, app_dir, web_root)
    assert res.returncode != 0
    assert "HEAD" in res.stderr
    assert_no_activation_systemctl(hermetic)


def test_activate_promotes_archived_unit_and_starts_service(hermetic, tmp_path: Path):
    set_systemctl_mode(hermetic, "inactive")
    archive, app_dir, web_root, source_sha = _create_and_restore(
        hermetic, tmp_path
    )
    res = run_activate(hermetic, app_dir, web_root)
    assert res.returncode == 0, res.stderr
    report = report_of(res)
    assert report["ok"] is True
    assert report["service_started"] is True
    assert report["unit_installed"] is True
    assert report["source_sha"] == source_sha
    assert report["former_authority_confirmed_stopped"] == "former-host.example"
    assert report["dns_caddy_unchanged"] is True
    assert isinstance(report["post_activation_acceptance"], list) and len(report["post_activation_acceptance"]) >= 3
    installed = hermetic.units / f"{TASKER}.service"
    assert installed.read_text(encoding="utf-8") == UNIT_TEXT
    log = hermetic.systemctl_log.read_text(encoding="utf-8").splitlines()
    assert "daemon-reload" in log
    assert f"enable {TASKER}" in log
    assert f"start {TASKER}" in log
    assert not any("stop" in line for line in log)
    # activation never touches Caddy or DNS
    assert not any("caddy" in line or "dns" in line for line in log)


# ── review round 0: allowed-dirty activation, unit re-bind/re-scan, % specifiers ─


def test_activate_allows_only_the_two_allowed_dirty_data_files(hermetic, tmp_path: Path):
    """Allowed-dirty data files archived at create may be dirty after restore;
    activation must accept exactly those two ordinary modifications (after the
    manifest-digest check proved them) and still reach the service start."""
    set_systemctl_mode(hermetic, "inactive")
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    _write(repo / "data/ensemble_weights.json", '{"normal": {"spy": 0.42, "gld": 0.38}}\n')
    _write(repo / "data/vix_term_structure.json", '{"_meta": {"schema": "vix_term_structure/v1", "dirty": true}}\n')
    web = make_web_root(tmp_path / "web", source_sha, generator_sha=source_sha[:12])
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    app_dir, web_root, res = standard_restore(hermetic, archive, "prod", extra=["--allow-production-paths"])
    assert res.returncode == 0, res.stderr
    _write(app_dir / ".env.local", "PORTFOLIO_LAB_MODE=lab\n")
    res = run_activate(hermetic, app_dir, web_root)
    assert res.returncode == 0, res.stderr
    report = report_of(res)
    assert report["ok"] is True
    assert report["service_started"] is True
    assert f"start {TASKER}" in hermetic.systemctl_log.read_text(encoding="utf-8")


def test_activate_rejects_other_tracked_changes_alongside_allowed_dirty_files(hermetic, tmp_path: Path):
    """Allowing the two generated data files must not become a blanket:
    any other tracked modification still blocks activation."""
    set_systemctl_mode(hermetic, "inactive")
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    _write(repo / "data/ensemble_weights.json", '{"normal": {"spy": 0.42}}\n')
    web = make_web_root(tmp_path / "web", source_sha, generator_sha=source_sha[:12])
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    app_dir, web_root, res = standard_restore(hermetic, archive, "prod", extra=["--allow-production-paths"])
    assert res.returncode == 0, res.stderr
    _write(app_dir / ".env.local", "PORTFOLIO_LAB_MODE=lab\n")
    _write(app_dir / "README.md", "# modified after restore\n")
    res = run_activate(hermetic, app_dir, web_root)
    assert res.returncode != 0
    assert "uncommitted" in res.stderr.lower()
    assert_no_activation_systemctl(hermetic)


def test_activate_rejects_tampered_restored_unit_before_systemd_mutation(hermetic, tmp_path: Path):
    """Post-restore tampering of the staged tasker unit must block activation
    before any systemd mutation."""
    set_systemctl_mode(hermetic, "inactive")
    archive, app_dir, web_root, _ = _create_and_restore(hermetic, tmp_path)
    _write(app_dir / ".portfolio-lab-recovery/metadata/tasker-unit.txt", UNIT_TEXT + "ExecStart=/bin/evil\n")
    res = run_activate(hermetic, app_dir, web_root)
    assert res.returncode != 0
    assert "unit" in res.stderr.lower()
    assert_no_activation_systemctl(hermetic)
    assert (hermetic.units / f"{TASKER}.service").read_text(encoding="utf-8") == UNIT_TEXT


def test_activate_rescans_archived_unit_and_rejects_secret_unit(hermetic, tmp_path: Path):
    """A verify-passing archive whose unit text carries a secret must be
    refused at activation by the unit re-scan, before any systemd mutation."""
    set_systemctl_mode(hermetic, "inactive")
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha, generator_sha=source_sha[:12])
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr

    def mutate(work: Path) -> None:
        _write(work / "metadata/tasker-unit.txt", UNIT_TEXT + "Environment=ACCESS_KEY=secret\n")

    rebuilt = repackage(archive, hermetic.tmp / "unit-retar", mutate, recompute=True)
    res = run_recovery(["verify", "--archive", str(rebuilt)], hermetic)
    assert res.returncode == 0, res.stderr
    app_dir, web_root, res = standard_restore(hermetic, rebuilt, "prod", extra=["--allow-production-paths"])
    assert res.returncode == 0, res.stderr
    _write(app_dir / ".env.local", "PORTFOLIO_LAB_MODE=lab\n")
    res = run_activate(hermetic, app_dir, web_root)
    assert res.returncode != 0
    assert "secret" in res.stderr.lower()
    assert_no_activation_systemctl(hermetic)
    assert (hermetic.units / f"{TASKER}.service").read_text(encoding="utf-8") == UNIT_TEXT


def test_restore_dev_api_rejects_percent_specifier_in_paths(hermetic, tmp_path: Path):
    """'%' in dev API unit paths would be systemd-specifier interpolation;
    restore --start-dev-api must reject it before any target mutation."""
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    for label, flag in (("app", "--app-dir"), ("web", "--web-root")):
        bad = hermetic.tmp / f"bad%h-{label}"
        args = {
            "--app-dir": str(hermetic.tmp / "app"),
            "--web-root": str(hermetic.tmp / "www"),
        }
        args[flag] = str(bad)
        res = run_recovery(
            [
                "restore",
                "--archive",
                str(archive),
                "--app-dir",
                args["--app-dir"],
                "--web-root",
                args["--web-root"],
                "--target-mode",
                "dev",
                "--start-dev-api",
            ],
            hermetic,
        )
        assert res.returncode != 0, label
        assert "%" in res.stderr, label
        assert not bad.exists(), label
        assert not (hermetic.tmp / "app").exists(), label
        assert not (hermetic.tmp / "www").exists(), label


# ── final review wave: nested env secret values, archive-change guard, ──────
# ── .envrc exclusion, service-name '%' ──────────────────────────────────────


def test_create_rejects_nested_env_secret_value_in_tasker_unit_before_service_stop(hermetic, tmp_path):
    """Anchored secret-looking values inside Environment=KEY=value lines
    (benign key, secret-looking nested value) must be refused by the value
    scan before the source Tasker stop."""
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    _write(
        hermetic.units / f"{TASKER}.service",
        UNIT_TEXT
        + "Environment=WEIRD=sk-abc123def456\n"
        + "Environment=OTHER=-----BEGIN PRIVATE KEY-----\n",
    )
    _write(repo / "data/tasker_status.json", json.dumps({"state": "idle"}) + "\n")
    archive = hermetic.tmp / "backups" / ("nested-secret" + ARCHIVE_SUFFIX)
    archive.parent.mkdir(parents=True, exist_ok=True)
    res = run_recovery(
        [
            "create",
            "--app-dir",
            str(repo),
            "--web-root",
            str(web),
            "--tasker-service",
            TASKER,
            "--archive",
            str(archive),
            "--storage-encryption-attested",
        ],
        hermetic,
    )
    assert res.returncode != 0
    assert "secret" in res.stderr.lower()
    # refused before the source Tasker stop: no systemctl calls, no archive
    assert not hermetic.systemctl_log.exists()
    assert not archive.exists()


def test_activate_rejects_nested_env_secret_unit_before_systemd_mutation(hermetic, tmp_path):
    """A verify-passing archive whose unit carries a nested Environment= value
    with an anchored secret-looking value must be refused by the activation
    unit re-scan, before any systemd mutation."""
    set_systemctl_mode(hermetic, "inactive")
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha, generator_sha=source_sha[:12])
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr

    def mutate(work: Path) -> None:
        _write(work / "metadata/tasker-unit.txt", UNIT_TEXT + "Environment=WEIRD=sk-abc123def456\n")

    rebuilt = repackage(archive, hermetic.tmp / "nested-unit-retar", mutate, recompute=True)
    res = run_recovery(["verify", "--archive", str(rebuilt)], hermetic)
    assert res.returncode == 0, res.stderr
    app_dir, web_root, res = standard_restore(hermetic, rebuilt, "prod", extra=["--allow-production-paths"])
    assert res.returncode == 0, res.stderr
    _write(app_dir / ".env.local", "PORTFOLIO_LAB_MODE=lab\n")
    res = run_activate(hermetic, app_dir, web_root)
    assert res.returncode != 0
    assert "secret" in res.stderr.lower()
    assert_no_activation_systemctl(hermetic)
    assert (hermetic.units / f"{TASKER}.service").read_text(encoding="utf-8") == UNIT_TEXT


def test_restore_rejects_archive_changed_between_verify_and_staging(hermetic, tmp_path):
    """If the archive bytes change between verification and the staging pass
    (second read), restore must fail closed before any target mutation."""
    fake_git = hermetic.bin / "git-wrapper"
    make_fake(
        fake_git,
        '#!/bin/sh\n'
        'if [ "$1" = "bundle" ] && [ "$2" = "list-heads" ] && [ -f "$PLR_MUTATE_ARCHIVE" ]; then\n'
        '  printf "x" >> "$PLR_MUTATE_ARCHIVE"\n'
        'fi\n'
        'exec git "$@"\n',
    )
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    # generator == source sha keeps verify on the fast path (no clone there)
    web = make_web_root(tmp_path / "web", source_sha, generator_sha=source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    app_dir, web_root, res = standard_restore(
        hermetic,
        archive,
        "dev",
        extra_env={"PLR_GIT": str(fake_git), "PLR_MUTATE_ARCHIVE": str(archive)},
    )
    assert res.returncode != 0
    assert "changed" in res.stderr.lower()
    assert not app_dir.exists()
    assert not web_root.exists()
    assert not list(hermetic.tmp.glob("*.rollback-*"))


def test_create_excludes_envrc_files(hermetic, tmp_path):
    """.envrc may carry environment secrets and is never recovery payload:
    it must be excluded at every depth."""
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    _write(repo / ".envrc", 'export SECRET_STUFF="x"\n')
    _write(repo / "data/.envrc", 'export SECRET_STUFF="x"\n')
    _write(web / ".envrc", 'export SECRET_STUFF="x"\n')
    commit_all(repo)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    members = set(read_tar_members(archive))
    assert not any(m.endswith("/.envrc") or m == ".envrc" for m in members)


def test_create_rejects_percent_in_service_name_before_service_stop(hermetic, tmp_path):
    """'%' is outside the systemd unit-name charset; create must fail closed
    before any systemctl call."""
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    _write(hermetic.units / f"{TASKER}.service", UNIT_TEXT)
    _write(repo / "data/tasker_status.json", json.dumps({"state": "idle"}) + "\n")
    archive = hermetic.tmp / "backups" / ("pct-name" + ARCHIVE_SUFFIX)
    archive.parent.mkdir(parents=True, exist_ok=True)
    res = run_recovery(
        [
            "create",
            "--app-dir",
            str(repo),
            "--web-root",
            str(web),
            "--tasker-service",
            "bad%name",
            "--archive",
            str(archive),
            "--storage-encryption-attested",
        ],
        hermetic,
    )
    assert res.returncode != 0
    assert "%" in res.stderr
    assert not hermetic.systemctl_log.exists()
    assert not archive.exists()


def test_restore_dev_api_rejects_percent_in_service_name(hermetic, tmp_path):
    """A dev API unit name containing '%' must be rejected before any target
    mutation."""
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    app_dir = hermetic.tmp / "app"
    web_root = hermetic.tmp / "www"
    res = run_recovery(
        [
            "restore",
            "--archive",
            str(archive),
            "--app-dir",
            str(app_dir),
            "--web-root",
            str(web_root),
            "--target-mode",
            "dev",
            "--start-dev-api",
            "--tasker-service",
            "bad%name",
        ],
        hermetic,
    )
    assert res.returncode != 0
    assert "%" in res.stderr
    assert not app_dir.exists()
    assert not web_root.exists()


# ── service-controller abstraction (plan Task 2.1) ─────────────────────────
#
# The fake box-persist controller below is an executable script (argv arrays
# only) that records every invocation to PLR_BP_LOG and emits a configurable
# portfolio-lab-box-persist/v1 status. Per-action defaults:
#   start-candidate -> active/disabled/0/pid 4242
#   activate        -> active/enabled/1/pid 4242
#   status|stop     -> inactive/disabled/0/no pid
# Overrides: PLR_BP_REPLIES holds {action: payload} (payload dict merged into
# the default, or a raw string printed verbatim), plus "__all__"; PLR_BP_EXIT
# forces a nonzero exit. PLR_BOX_PERSIST_ALLOWED_ROOT pins the production
# root guard to the pytest tmp tree (test-only escape hatch).

BOX_PERSIST_FAKE = r'''#!/usr/bin/env python3
import json
import os
import sys

log_path = os.environ.get("PLR_BP_LOG")
if log_path:
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(sys.argv[1:]) + "\n")


def arg_value(name):
    for i, item in enumerate(sys.argv):
        if item == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


action = sys.argv[1] if len(sys.argv) > 1 else "?"
payload = {
    "schema": "portfolio-lab-box-persist/v1",
    "state": "inactive",
    "scheduler_mode": "disabled",
    "identity_exact": True,
    "scheduler_instances": 0,
    "pid": None,
    "service_name": arg_value("--service-name") or "",
    "mode": arg_value("--mode") or "",
    "app_dir": arg_value("--app-dir") or "",
    "web_root": arg_value("--web-root") or "",
}
if action == "start-candidate":
    payload.update({"state": "active", "pid": 4242})
elif action == "activate":
    payload.update({"state": "active", "scheduler_mode": "enabled", "scheduler_instances": 1, "pid": 4242})
replies_file = os.environ.get("PLR_BP_REPLIES")
if replies_file and os.path.exists(replies_file):
    with open(replies_file, encoding="utf-8") as fh:
        replies = json.load(fh)
    override = replies.get(action, replies.get("__all__"))
    if isinstance(override, dict):
        payload.update(override)
    elif isinstance(override, str):
        print(override)
        sys.exit(0)
exit_code = os.environ.get("PLR_BP_EXIT")
if exit_code:
    sys.exit(int(exit_code))
print(json.dumps(payload))
'''


@pytest.fixture
def box_persist_env(hermetic, tmp_path: Path):
    """Hermetic fake box-persist controller (see BOX_PERSIST_FAKE)."""
    make_fake(hermetic.bin / "portfolio-lab-box-persist", BOX_PERSIST_FAKE)
    bp_log = tmp_path / "box-persist.log"
    replies = tmp_path / "box-persist-replies.json"
    overrides = {
        "PLR_BOX_PERSIST_CONTROLLER": str(hermetic.bin / "portfolio-lab-box-persist"),
        "PLR_BOX_PERSIST_ALLOWED_ROOT": str(tmp_path),
        "PLR_BP_LOG": str(bp_log),
        "PLR_BP_REPLIES": str(replies),
    }
    return SimpleNamespace(
        tmp=tmp_path,
        env={**hermetic.env, **overrides},
        bp_log=bp_log,
        replies=replies,
        bp_controller=hermetic.bin / "portfolio-lab-box-persist",
        units=hermetic.units,
        systemctl_log=hermetic.systemctl_log,
    )


def run_box_persist_recovery(
    args: list[str],
    bp: SimpleNamespace,
    extra_env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    env = {**os.environ, **bp.env}
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(RECOVERY_SCRIPT), *args],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
        timeout=120,
    )


def box_persist_log_lines(bp: SimpleNamespace) -> list[list[str]]:
    if not bp.bp_log.exists():
        return []
    return [
        json.loads(line)
        for line in bp.bp_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def reset_bp_log(bp: SimpleNamespace) -> None:
    bp.bp_log.unlink(missing_ok=True)


def set_box_persist_replies(bp: SimpleNamespace, replies: dict) -> None:
    _write(bp.replies, json.dumps(replies))


def standard_create_bp(
    bp: SimpleNamespace,
    repo: Path,
    web: Path,
    archive_name: str = "bp-backup" + ARCHIVE_SUFFIX,
) -> tuple[Path, subprocess.CompletedProcess]:
    archive = bp.tmp / "backups" / archive_name
    archive.parent.mkdir(parents=True, exist_ok=True)
    _write(bp.units / f"{TASKER}.service", UNIT_TEXT)
    _write(repo / "data/tasker_status.json", json.dumps({"state": "idle", "version": 1}) + "\n")
    args = [
        "create",
        "--app-dir",
        str(repo),
        "--web-root",
        str(web),
        "--tasker-service",
        TASKER,
        "--archive",
        str(archive),
        "--storage-encryption-attested",
    ]
    return archive, run_box_persist_recovery(args, bp)


def standard_restore_bp(
    bp: SimpleNamespace,
    archive: Path,
    mode: str,
    extra: list[str] | None = None,
    app_name: str = "app",
    web_name: str = "www",
    extra_env: dict[str, str] | None = None,
) -> tuple[Path, Path, subprocess.CompletedProcess]:
    app_dir = bp.tmp / app_name
    web_root = bp.tmp / web_name
    args = [
        "restore",
        "--archive",
        str(archive),
        "--app-dir",
        str(app_dir),
        "--web-root",
        str(web_root),
        "--target-mode",
        mode,
        "--service-controller",
        "box-persist",
        *(extra or []),
    ]
    return app_dir, web_root, run_box_persist_recovery(args, bp, extra_env=extra_env)


def create_and_restore_box_persist(
    bp: SimpleNamespace,
    tmp_path: Path,
    archive_name: str = "bp-prod" + ARCHIVE_SUFFIX,
) -> tuple[Path, Path, Path, str]:
    """create (systemd source) + box-persist prod restore + .env.local."""
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha, generator_sha=source_sha[:12])
    archive, res = standard_create_bp(bp, repo, web, archive_name=archive_name)
    assert res.returncode == 0, res.stderr
    app_dir, web_root, res = standard_restore_bp(bp, archive, "prod", extra=["--allow-production-paths"])
    assert res.returncode == 0, res.stderr
    _write(app_dir / ".env.local", "PORTFOLIO_LAB_MODE=lab\n")
    return archive, app_dir, web_root, source_sha


def run_activate_bp(bp: SimpleNamespace, app_dir: Path, web_root: Path) -> subprocess.CompletedProcess:
    args = [
        "activate-prod",
        "--app-dir",
        str(app_dir),
        "--web-root",
        str(web_root),
        "--tasker-service",
        TASKER,
        "--confirm-authoritative-activation",
        "--former-authority-confirmed-stopped",
        "former-host.example",
        "--service-controller",
        "box-persist",
    ]
    return run_box_persist_recovery(args, bp)


def test_service_controller_defaults_to_systemd_on_create(hermetic, tmp_path: Path):
    """Omitting --service-controller keeps the systemd behavior unchanged."""
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr
    report = report_of(res)
    assert report["service_controller"] == "systemd"
    log = hermetic.systemctl_log.read_text(encoding="utf-8").splitlines()
    assert log == ["is-active portfolio-lab-tasker", "stop portfolio-lab-tasker", "start portfolio-lab-tasker"]


def test_unknown_service_controller_fails_through_parser_before_mutation(hermetic, tmp_path: Path):
    """Unknown --service-controller values must be rejected by argparse (rc 2)
    before any systemctl/controller call, source stop, or target/archive write."""
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    archive = hermetic.tmp / "backups" / ("x" + ARCHIVE_SUFFIX)
    archive.parent.mkdir(parents=True, exist_ok=True)
    _write(hermetic.units / f"{TASKER}.service", UNIT_TEXT)
    _write(repo / "data/tasker_status.json", json.dumps({"state": "idle"}) + "\n")
    cases = [
        (
            "create",
            [
                "create",
                "--app-dir",
                str(repo),
                "--web-root",
                str(web),
                "--tasker-service",
                TASKER,
                "--archive",
                str(archive),
                "--storage-encryption-attested",
                "--service-controller",
                "bogus",
            ],
        ),
        (
            "restore",
            [
                "restore",
                "--archive",
                str(archive),
                "--app-dir",
                str(hermetic.tmp / "app"),
                "--web-root",
                str(hermetic.tmp / "www"),
                "--target-mode",
                "dev",
                "--service-controller",
                "bogus",
            ],
        ),
        (
            "activate-prod",
            [
                "activate-prod",
                "--app-dir",
                str(hermetic.tmp / "app"),
                "--web-root",
                str(hermetic.tmp / "www"),
                "--tasker-service",
                TASKER,
                "--confirm-authoritative-activation",
                "--former-authority-confirmed-stopped",
                "old-host",
                "--service-controller",
                "bogus",
            ],
        ),
    ]
    for label, args in cases:
        res = run_recovery(args, hermetic)
        assert res.returncode == 2, label
        assert "invalid choice" in res.stderr, label
    assert not hermetic.systemctl_log.exists()
    assert not archive.exists()
    assert not (hermetic.tmp / "app").exists()
    assert not (hermetic.tmp / "www").exists()


def test_create_rejects_box_persist_controller_before_mutation(hermetic, tmp_path: Path):
    """create is sg01 source-side only; explicit box-persist must fail before
    any systemctl/box command, source stop, or archive write."""
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    archive = hermetic.tmp / "backups" / ("x" + ARCHIVE_SUFFIX)
    archive.parent.mkdir(parents=True, exist_ok=True)
    _write(hermetic.units / f"{TASKER}.service", UNIT_TEXT)
    _write(repo / "data/tasker_status.json", json.dumps({"state": "idle"}) + "\n")
    res = run_recovery(
        [
            "create",
            "--app-dir",
            str(repo),
            "--web-root",
            str(web),
            "--tasker-service",
            TASKER,
            "--archive",
            str(archive),
            "--storage-encryption-attested",
            "--service-controller",
            "box-persist",
        ],
        hermetic,
    )
    assert res.returncode != 0
    assert "box-persist" in res.stderr.lower()
    assert "systemd" in res.stderr.lower()
    assert not hermetic.systemctl_log.exists()
    assert not archive.exists()


def test_box_persist_controller_path_must_be_absolute(box_persist_env, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    archive, res = standard_create_bp(box_persist_env, repo, web)
    assert res.returncode == 0, res.stderr
    app_dir, web_root, res = standard_restore_bp(
        box_persist_env,
        archive,
        "dev",
        extra_env={"PLR_BOX_PERSIST_CONTROLLER": "relative/box-persist"},
    )
    assert res.returncode != 0
    assert "absolute" in res.stderr.lower()
    assert not app_dir.exists()
    assert not web_root.exists()
    assert box_persist_log_lines(box_persist_env) == []


def test_box_persist_missing_controller_rejected_before_mutation(box_persist_env, tmp_path: Path):
    """An absolute PLR_BOX_PERSIST_CONTROLLER path that does not exist must be
    rejected with an executable diagnostic before any controller invocation:
    the restore must not place targets and the fake controller must never be
    called."""
    bp = box_persist_env
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    archive, res = standard_create_bp(bp, repo, web)
    assert res.returncode == 0, res.stderr
    app_dir = bp.tmp / "app-missing"
    web_root = bp.tmp / "www-missing"
    res = run_box_persist_recovery(
        [
            "restore",
            "--archive",
            str(archive),
            "--app-dir",
            str(app_dir),
            "--web-root",
            str(web_root),
            "--target-mode",
            "dev",
            "--service-controller",
            "box-persist",
            "--start-dev-api",
        ],
        bp,
        extra_env={"PLR_BOX_PERSIST_CONTROLLER": str(bp.tmp / "bin" / "no-such-box-persist")},
    )
    assert res.returncode != 0
    assert "box-persist controller" in res.stderr
    assert "not found" in res.stderr
    assert not app_dir.exists()
    assert not web_root.exists()
    assert not bp.bp_log.exists()


def test_box_persist_non_executable_controller_rejected_before_mutation(box_persist_env, tmp_path: Path):
    """An absolute regular file without execute permission as
    PLR_BOX_PERSIST_CONTROLLER must be rejected with an executable diagnostic
    before any controller invocation or target mutation."""
    bp = box_persist_env
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    archive, res = standard_create_bp(bp, repo, web)
    assert res.returncode == 0, res.stderr
    noexec = bp.tmp / "bin" / "box-persist-noexec"
    _write(noexec, "#!/bin/sh\nexit 0\n")
    noexec.chmod(0o644)
    app_dir = bp.tmp / "app-noexec"
    web_root = bp.tmp / "www-noexec"
    res = run_box_persist_recovery(
        [
            "restore",
            "--archive",
            str(archive),
            "--app-dir",
            str(app_dir),
            "--web-root",
            str(web_root),
            "--target-mode",
            "dev",
            "--service-controller",
            "box-persist",
            "--start-dev-api",
        ],
        bp,
        extra_env={"PLR_BOX_PERSIST_CONTROLLER": str(noexec)},
    )
    assert res.returncode != 0
    assert "box-persist controller" in res.stderr
    assert "not executable" in res.stderr
    assert not app_dir.exists()
    assert not web_root.exists()
    assert not bp.bp_log.exists()


def test_box_persist_restore_rejects_paths_outside_allowed_root(box_persist_env, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    archive, res = standard_create_bp(box_persist_env, repo, web)
    assert res.returncode == 0, res.stderr
    allowed = tmp_path / "allowed-root"
    app_dir, web_root, res = standard_restore_bp(
        box_persist_env,
        archive,
        "dev",
        extra_env={"PLR_BOX_PERSIST_ALLOWED_ROOT": str(allowed)},
    )
    assert res.returncode != 0
    assert "under" in res.stderr.lower()
    assert str(allowed) in res.stderr
    assert not app_dir.exists()
    assert not web_root.exists()
    assert box_persist_log_lines(box_persist_env) == []


def test_restore_box_persist_dev_without_start_is_pure_staging(box_persist_env, tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha, generator_sha=source_sha[:12])
    archive, res = standard_create_bp(box_persist_env, repo, web)
    assert res.returncode == 0, res.stderr
    app_dir, web_root, res = standard_restore_bp(box_persist_env, archive, "dev")
    assert res.returncode == 0, res.stderr
    report = report_of(res)
    assert report["service_controller"] == "box-persist"
    assert report["service_started"] is False
    # pure staging: no controller invocation at all
    assert box_persist_log_lines(box_persist_env) == []
    assert _git(app_dir, "rev-parse", "HEAD").stdout.strip() == source_sha


def test_restore_box_persist_dev_start_candidate_accepts_active_status(box_persist_env, tmp_path: Path):
    """box-persist dev --start-dev-api delegates start-candidate and accepts
    the returned active/disabled/zero-instance status (idempotent: no inactive
    pre-condition), with no systemd unit writes or systemctl calls."""
    bp = box_persist_env
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    archive, res = standard_create_bp(bp, repo, web)
    assert res.returncode == 0, res.stderr
    app_dir, web_root, res = standard_restore_bp(bp, archive, "dev", extra=["--start-dev-api"])
    assert res.returncode == 0, res.stderr
    report = report_of(res)
    assert report["service_controller"] == "box-persist"
    assert report["service_started"] is True
    assert report["service_name"] == DEV_SERVICE
    assert report["dev_api_unit"] is None
    assert report["controller_status"] == {
        "state": "active",
        "scheduler_mode": "disabled",
        "identity_exact": True,
        "scheduler_instances": 0,
        "pid": 4242,
    }
    assert box_persist_log_lines(bp) == [
        [
            "start-candidate",
            "--mode",
            "candidate",
            "--app-dir",
            str(app_dir),
            "--web-root",
            str(web_root),
            "--service-name",
            DEV_SERVICE,
        ]
    ]
    # no systemd unit was written and systemctl saw no dev-unit operation
    assert sorted(p.name for p in bp.units.iterdir()) == [f"{TASKER}.service"]
    log = bp.systemctl_log.read_text(encoding="utf-8").splitlines()
    assert log == ["is-active portfolio-lab-tasker", "stop portfolio-lab-tasker", "start portfolio-lab-tasker"]


def test_restore_box_persist_dev_start_rejects_enabled_scheduler(box_persist_env, tmp_path: Path):
    bp = box_persist_env
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    archive, res = standard_create_bp(bp, repo, web)
    assert res.returncode == 0, res.stderr
    set_box_persist_replies(bp, {"start-candidate": {"scheduler_mode": "enabled"}})
    app_dir, web_root, res = standard_restore_bp(bp, archive, "dev", extra=["--start-dev-api"])
    assert res.returncode != 0
    assert "scheduler" in res.stderr.lower()


def test_restore_box_persist_dev_start_rejects_scheduler_instances(box_persist_env, tmp_path: Path):
    """A candidate that claims scheduler disabled while reporting running
    scheduler instances must fail closed (inconsistent scheduler counts)."""
    bp = box_persist_env
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    archive, res = standard_create_bp(bp, repo, web)
    assert res.returncode == 0, res.stderr
    set_box_persist_replies(bp, {"start-candidate": {"scheduler_instances": 1}})
    app_dir, web_root, res = standard_restore_bp(bp, archive, "dev", extra=["--start-dev-api"])
    assert res.returncode != 0
    assert "scheduler" in res.stderr.lower()


def test_restore_box_persist_rejects_malformed_and_unsupported_status(box_persist_env, tmp_path: Path):
    bp = box_persist_env
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    archive, res = standard_create_bp(bp, repo, web)
    assert res.returncode == 0, res.stderr
    set_box_persist_replies(bp, {"start-candidate": "not json at all"})
    _, _, res = standard_restore_bp(bp, archive, "dev", extra=["--start-dev-api"], app_name="app-malformed")
    assert res.returncode != 0
    assert "json" in res.stderr.lower()
    bp.bp_log.unlink(missing_ok=True)
    set_box_persist_replies(bp, {"start-candidate": {"state": "loading"}})
    _, _, res = standard_restore_bp(bp, archive, "dev", extra=["--start-dev-api"], app_name="app-unsupported")
    assert res.returncode != 0
    assert "state" in res.stderr.lower()
    bp.bp_log.unlink(missing_ok=True)
    # active state must carry a PID (validation layer, ruling 5)
    set_box_persist_replies(bp, {"start-candidate": {"pid": None}})
    _, _, res = standard_restore_bp(bp, archive, "dev", extra=["--start-dev-api"], app_name="app-nopid")
    assert res.returncode != 0
    assert "pid" in res.stderr.lower()


def test_restore_box_persist_rejects_mismatched_identity(box_persist_env, tmp_path: Path):
    """The returned status must echo the requested service/mode/app/web
    identity; any mismatch fails closed before the start is accepted."""
    bp = box_persist_env
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    archive, res = standard_create_bp(bp, repo, web)
    assert res.returncode == 0, res.stderr
    set_box_persist_replies(bp, {"start-candidate": {"service_name": "other-service"}})
    _, _, res = standard_restore_bp(bp, archive, "dev", extra=["--start-dev-api"], app_name="app-identity")
    assert res.returncode != 0
    assert "identity" in res.stderr.lower()
    bp.bp_log.unlink(missing_ok=True)
    set_box_persist_replies(bp, {"start-candidate": {"web_root": "/elsewhere/www"}})
    _, _, res = standard_restore_bp(bp, archive, "dev", extra=["--start-dev-api"], app_name="app-identity2")
    assert res.returncode != 0
    assert "identity" in res.stderr.lower()


def test_restore_box_persist_prod_preflight_requires_inactive(box_persist_env, tmp_path: Path):
    """Prod staging checks target production status before mutation and
    requires inactive/disabled/zero-instance/exact-identity/null-pid."""
    bp = box_persist_env
    active_status = {
        "state": "active",
        "scheduler_mode": "enabled",
        "identity_exact": True,
        "scheduler_instances": 1,
        "pid": 4242,
    }
    for label, replies, marker in (
        ("active", {"status": active_status}, "inactive"),
        ("enabled", {"status": {"scheduler_mode": "enabled"}}, "disable"),
        ("instances", {"status": {"scheduler_instances": 2}}, "scheduler"),
    ):
        repo = make_repo(tmp_path / f"repo-{label}")
        commit_all(repo)
        web = make_web_root(tmp_path / f"web-{label}", "x" * 40)
        archive, res = standard_create_bp(bp, repo, web, archive_name=f"{label}" + ARCHIVE_SUFFIX)
        assert res.returncode == 0, res.stderr
        set_box_persist_replies(bp, replies)
        app_dir, web_root, res = standard_restore_bp(
            bp,
            archive,
            "prod",
            extra=["--allow-production-paths"],
            app_name=f"app-{label}",
            web_name=f"www-{label}",
        )
        assert res.returncode != 0, label
        assert marker in res.stderr.lower(), (label, res.stderr)
        assert not app_dir.exists(), label
        assert not web_root.exists(), label
        assert box_persist_log_lines(bp) == [
            [
                "status",
                "--mode",
                "production",
                "--app-dir",
                str(app_dir),
                "--web-root",
                str(web_root),
                "--service-name",
                TASKER,
            ]
        ], label
        bp.bp_log.unlink(missing_ok=True)


def test_restore_box_persist_prod_stages_after_inactive_preflight(box_persist_env, tmp_path: Path):
    bp = box_persist_env
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha, generator_sha=source_sha[:12])
    archive, res = standard_create_bp(bp, repo, web)
    assert res.returncode == 0, res.stderr
    app_dir, web_root, res = standard_restore_bp(bp, archive, "prod", extra=["--allow-production-paths"])
    assert res.returncode == 0, res.stderr
    report = report_of(res)
    assert report["service_controller"] == "box-persist"
    assert report["service_started"] is False
    assert report["controller_status"] == {
        "state": "inactive",
        "scheduler_mode": "disabled",
        "identity_exact": True,
        "scheduler_instances": 0,
        "pid": None,
    }
    assert "activate" in report["activation_note"].lower()
    assert box_persist_log_lines(bp) == [
        [
            "status",
            "--mode",
            "production",
            "--app-dir",
            str(app_dir),
            "--web-root",
            str(web_root),
            "--service-name",
            TASKER,
        ]
    ]
    # no start-candidate and no systemctl/unit writes on this box-persist path
    log = bp.systemctl_log.read_text(encoding="utf-8").splitlines()
    assert log == ["is-active portfolio-lab-tasker", "stop portfolio-lab-tasker", "start portfolio-lab-tasker"]
    assert (app_dir / "README.md").is_file()


def test_activate_box_persist_preflight_blocks_before_activate_call(box_persist_env, tmp_path: Path):
    """Activation preflight must block an active target or a second scheduler
    before the activate command is ever invoked."""
    bp = box_persist_env
    active_status = {
        "state": "active",
        "scheduler_mode": "enabled",
        "identity_exact": True,
        "scheduler_instances": 1,
        "pid": 4242,
    }
    for label, replies, marker in (
        ("active", {"status": active_status}, "inactive"),
        ("second-scheduler", {"status": {"scheduler_instances": 2}}, "scheduler"),
    ):
        # stale replies from a previous subcase must not leak into the
        # helper's own restore preflight
        bp.replies.unlink(missing_ok=True)
        _, app_dir, web_root, _ = create_and_restore_box_persist(
            bp, tmp_path / f"t-{label}", archive_name=f"t-{label}" + ARCHIVE_SUFFIX
        )
        reset_bp_log(bp)
        set_box_persist_replies(bp, replies)
        res = run_activate_bp(bp, app_dir, web_root)
        assert res.returncode != 0, label
        assert marker in res.stderr.lower(), (label, res.stderr)
        lines = box_persist_log_lines(bp)
        assert len(lines) == 1, label
        assert lines[0][0] == "status", label


def test_activate_box_persist_requires_exactly_one_scheduler_instance(box_persist_env, tmp_path: Path):
    """The activate response must report exactly one scheduler instance; 0 or
    multiple instances fail closed (second-scheduler prevention gate)."""
    bp = box_persist_env
    for label, instances in (("zero", 0), ("two", 2)):
        _, app_dir, web_root, _ = create_and_restore_box_persist(
            bp, tmp_path / f"o-{label}", archive_name=f"o-{label}" + ARCHIVE_SUFFIX
        )
        reset_bp_log(bp)
        set_box_persist_replies(bp, {"activate": {"scheduler_instances": instances}})
        res = run_activate_bp(bp, app_dir, web_root)
        assert res.returncode != 0, label
        assert "exactly one" in res.stderr.lower(), (label, res.stderr)
        assert [line[0] for line in box_persist_log_lines(bp)] == ["status", "activate"], label


def test_activate_box_persist_happy_path_forwards_proof_and_reports_status(box_persist_env, tmp_path: Path):
    """box-persist activation: inactive preflight, activate invocation with
    the former-authority proof label forwarded exactly, exactly one active
    scheduler returned, no systemctl/unit writes, normalized status fields in
    the JSON recovery report."""
    bp = box_persist_env
    _, app_dir, web_root, _ = create_and_restore_box_persist(bp, tmp_path)
    reset_bp_log(bp)
    res = run_activate_bp(bp, app_dir, web_root)
    assert res.returncode == 0, res.stderr
    report = report_of(res)
    assert report["ok"] is True
    assert report["service_controller"] == "box-persist"
    assert report["unit_installed"] is False
    assert report["service_started"] is True
    assert report["former_authority_confirmed_stopped"] == "former-host.example"
    assert report["dns_caddy_unchanged"] is True
    assert report["controller_preflight_status"] == {
        "state": "inactive",
        "scheduler_mode": "disabled",
        "identity_exact": True,
        "scheduler_instances": 0,
        "pid": None,
    }
    assert report["controller_activation_status"] == {
        "state": "active",
        "scheduler_mode": "enabled",
        "identity_exact": True,
        "scheduler_instances": 1,
        "pid": 4242,
    }
    assert isinstance(report["post_activation_acceptance"], list) and len(report["post_activation_acceptance"]) >= 3
    assert box_persist_log_lines(bp) == [
        [
            "status",
            "--mode",
            "production",
            "--app-dir",
            str(app_dir),
            "--web-root",
            str(web_root),
            "--service-name",
            TASKER,
        ],
        [
            "activate",
            "--mode",
            "production",
            "--app-dir",
            str(app_dir),
            "--web-root",
            str(web_root),
            "--service-name",
            TASKER,
            "--former-authority-confirmed-stopped",
            "former-host.example",
        ],
    ]
    log = bp.systemctl_log.read_text(encoding="utf-8").splitlines()
    assert log == ["is-active portfolio-lab-tasker", "stop portfolio-lab-tasker", "start portfolio-lab-tasker"]
    assert bp.units.joinpath(f"{TASKER}.service").read_text(encoding="utf-8") == UNIT_TEXT


def test_restore_box_persist_nonzero_controller_exit_fails_closed(box_persist_env, tmp_path: Path):
    bp = box_persist_env
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    archive, res = standard_create_bp(bp, repo, web)
    assert res.returncode == 0, res.stderr
    app_dir, web_root, res = standard_restore_bp(
        bp,
        archive,
        "dev",
        extra=["--start-dev-api"],
        extra_env={"PLR_BP_EXIT": "7"},
    )
    assert res.returncode != 0
    assert "failed" in res.stderr.lower()
    assert "box-persist" in res.stderr.lower()


def test_box_persist_stop_command_contract(box_persist_env, monkeypatch):
    """The stop command contract (for the future Task 2.2 script) must be
    emitted as an argv array and its normalized status parsed."""
    bp = box_persist_env
    mod = _load_recovery_module()
    monkeypatch.setenv("PLR_BP_LOG", str(bp.bp_log))
    controller = mod.BoxPersistController(bp.bp_controller)
    app_dir = bp.tmp / "app-stop"
    web_root = bp.tmp / "www-stop"
    status = controller.stop("tasker", "production", app_dir, web_root)
    assert status["state"] == "inactive"
    assert status["scheduler_mode"] == "disabled"
    assert status["identity_exact"] is True
    assert status["scheduler_instances"] == 0
    assert status["pid"] is None
    assert box_persist_log_lines(bp) == [
        [
            "stop",
            "--mode",
            "production",
            "--app-dir",
            str(app_dir),
            "--web-root",
            str(web_root),
            "--service-name",
            "tasker",
        ]
    ]


def test_box_persist_command_timeout_fails_closed(box_persist_env, monkeypatch, capsys):
    """A controller command that exceeds the bounded subprocess ceiling must
    fail closed: the shared run() converts TimeoutExpired into SystemExit
    with a timeout diagnostic (no slow real-timeout test; the 180s bound
    stays in run())."""
    bp = box_persist_env
    mod = _load_recovery_module()

    def timeout_run(argv, cwd=None, env=None, **kwargs):
        raise subprocess.TimeoutExpired(argv[0], 180)

    monkeypatch.setattr(subprocess, "run", timeout_run)
    controller = mod.BoxPersistController(bp.bp_controller)
    with pytest.raises(SystemExit):
        controller.status("tasker", "production", bp.tmp / "app", bp.tmp / "www")
    err = capsys.readouterr().err
    assert "timed out" in err


# ── Task 2.4 generation materialization tests ───────────────────────────────


def test_generation_materialization_flag_parser_contract():
    """Parser must accept --materialize-generations-current on create only;
    verify and restore must reject the unknown argument."""
    mod = _load_recovery_module()
    parser = mod.build_parser()

    # Create accepts the flag
    args = parser.parse_args([
        "create",
        "--app-dir", "/app",
        "--web-root", "/www",
        "--tasker-service", "tasker",
        "--archive", "/dest.portfolio-lab-recovery.tar",
        "--storage-encryption-attested",
        "--materialize-generations-current",
    ])
    assert args.materialize_generations_current is True

    # Default is False
    args_default = parser.parse_args([
        "create",
        "--app-dir", "/app",
        "--web-root", "/www",
        "--tasker-service", "tasker",
        "--archive", "/dest.portfolio-lab-recovery.tar",
        "--storage-encryption-attested",
    ])
    assert args_default.materialize_generations_current is False

    # Verify rejects the flag
    with pytest.raises(SystemExit):
        parser.parse_args(["verify", "--archive", "/a.tar", "--materialize-generations-current"])

    # Restore rejects the flag
    with pytest.raises(SystemExit):
        parser.parse_args([
            "restore",
            "--archive", "/a.tar",
            "--app-dir", "/app",
            "--web-root", "/www",
            "--target-mode", "dev",
            "--materialize-generations-current",
        ])


def test_create_without_flag_rejects_current_symlink_before_stop(hermetic, tmp_path: Path):
    """Without --materialize-generations-current, existing behavior remains:
    data/generations/current directory symlink is rejected before source stop,
    and no archive is written."""
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    gen_dir = repo / "data" / "generations"
    gen_target = gen_dir / "gen-2026-09-03-001"
    gen_target.mkdir(parents=True)
    _write(gen_target / "manifest.json", '{"run_id": "gen-2026-09-03-001"}\n')
    (gen_dir / "current").symlink_to("gen-2026-09-03-001")

    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode != 0
    assert "symlink" in res.stderr.lower()
    assert not hermetic.systemctl_log.exists()
    assert not archive.exists()


def test_create_with_flag_success_and_verify_restore_reconstruction(hermetic, tmp_path: Path):
    """Safe relative current -> gen-id creates a verified archive with the flag.
    Source link bytes/type remain unchanged.
    Archive contains ordinary runtime/data/generations/current/... files and normal target files.
    Manifest has exact generation_materialization object with no absolute host paths.
    Restore reconstructs data/generations/current as the exact relative symlink.
    Prove subsequent atomic replacement of that symlink succeeds."""
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha, generator_sha=source_sha[:12])

    gen_dir = repo / "data" / "generations"
    gen_id = "gen-2026-09-03-001"
    gen_target = gen_dir / "gen-2026-09-03-001"
    gen_target.mkdir(parents=True)
    _write(gen_target / "manifest.json", '{"schema": "portfolio-lab-generation/v1", "run_id": "gen-2026-09-03-001"}\n')
    _write(gen_target / "index.json", '{"status": "ok"}\n')
    (gen_dir / "current").symlink_to(gen_id)

    # Prove source is a symlink before create
    assert (gen_dir / "current").is_symlink()
    assert os.readlink(gen_dir / "current") == gen_id

    archive, res = standard_create(
        hermetic,
        repo,
        web,
        extra=["--materialize-generations-current"],
    )
    assert res.returncode == 0, res.stderr
    report = report_of(res)
    assert report["ok"] is True
    assert report["checks"]["generation_materialization_metadata_ok"] is True
    assert report["checks"]["generation_materialization_members_match"] is True

    # Source symlink bytes and type remain untouched
    assert (gen_dir / "current").is_symlink()
    assert not (gen_dir / "current").is_file()
    assert os.readlink(gen_dir / "current") == gen_id
    assert (gen_target / "manifest.json").read_text(encoding="utf-8") == '{"schema": "portfolio-lab-generation/v1", "run_id": "gen-2026-09-03-001"}\n'
    assert (gen_target / "index.json").read_text(encoding="utf-8") == '{"status": "ok"}\n'

    # Verify tar contents: no directory/symlink members, only regular files
    with tarfile.open(str(archive), "r:") as tf:
        names = tf.getnames()
        for member in tf.getmembers():
            assert member.isfile()
            assert not member.issym()
            assert not member.isdir()

    assert "runtime/data/generations/gen-2026-09-03-001/manifest.json" in names
    assert "runtime/data/generations/gen-2026-09-03-001/index.json" in names
    assert "runtime/data/generations/current/manifest.json" in names
    assert "runtime/data/generations/current/index.json" in names

    # Manifest check
    manifest_raw = extract_tar_member(archive, "recovery-manifest.json")
    manifest = json.loads(manifest_raw)
    mat = manifest.get("generation_materialization")
    assert mat == {
        "schema_version": "portfolio-lab-generation-materialization/v1",
        "link_path": "data/generations/current",
        "original_link": gen_id,
        "target_path": f"data/generations/{gen_id}",
        "archive_path": "runtime/data/generations/current",
    }
    # Ensure no absolute host paths in manifest
    assert str(tmp_path) not in json.dumps(mat)

    # Verify command also reports the generation materialization checks
    vres = run_recovery(["verify", "--archive", str(archive)], hermetic)
    assert vres.returncode == 0, vres.stderr
    vrep = report_of(vres)
    assert vrep["ok"] is True
    assert vrep["checks"]["generation_materialization_metadata_ok"] is True
    assert vrep["checks"]["generation_materialization_members_match"] is True

    # Dev restore: prove reconstruction of current as relative symlink
    app_dir = hermetic.tmp / "app-dev"
    web_root = hermetic.tmp / "web-dev"
    rres = run_recovery(
        [
            "restore",
            "--archive", str(archive),
            "--app-dir", str(app_dir),
            "--web-root", str(web_root),
            "--target-mode", "dev",
        ],
        hermetic,
    )
    assert rres.returncode == 0, rres.stderr
    restored_link = app_dir / "data" / "generations" / "current"
    assert restored_link.is_symlink()
    assert os.readlink(restored_link) == gen_id
    assert restored_link.resolve() == (app_dir / "data" / "generations" / gen_id).resolve()
    assert (restored_link / "manifest.json").is_file()

    # Prove subsequent atomic replacement of that symlink succeeds (GenerationStore._activate pattern)
    tmp_link = restored_link.with_name("current.link.tmp")
    os.symlink("gen-next", tmp_link)
    os.replace(tmp_link, restored_link)
    assert os.readlink(restored_link) == "gen-next"

    # Prod restore: prove reconstruction also in prod mode
    set_systemctl_mode(hermetic, "inactive")
    prod_app = hermetic.tmp / "app-prod"
    prod_web = hermetic.tmp / "web-prod"
    pres = run_recovery(
        [
            "restore",
            "--archive", str(archive),
            "--app-dir", str(prod_app),
            "--web-root", str(prod_web),
            "--target-mode", "prod",
            "--allow-production-paths",
        ],
        hermetic,
    )
    assert pres.returncode == 0, pres.stderr
    prod_link = prod_app / "data" / "generations" / "current"
    assert prod_link.is_symlink()
    assert os.readlink(prod_link) == gen_id
    assert (prod_link / "manifest.json").is_file()


def test_old_archive_without_materialization_verifies_and_restores_unchanged(hermetic, tmp_path: Path):
    """Old archives without generation_materialization object remain valid,
    verify with None checks, and restore unchanged."""
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    archive, res = standard_create(hermetic, repo, web)
    assert res.returncode == 0, res.stderr

    vres = run_recovery(["verify", "--archive", str(archive)], hermetic)
    assert vres.returncode == 0, vres.stderr
    vrep = report_of(vres)
    assert vrep["checks"]["generation_materialization_metadata_ok"] is None
    assert vrep["checks"]["generation_materialization_members_match"] is None

    app_dir = hermetic.tmp / "app-old"
    web_root = hermetic.tmp / "web-old"
    rres = run_recovery(
        [
            "restore",
            "--archive", str(archive),
            "--app-dir", str(app_dir),
            "--web-root", str(web_root),
            "--target-mode", "dev",
        ],
        hermetic,
    )
    assert rres.returncode == 0, rres.stderr
    assert (app_dir / "README.md").is_file()


@pytest.mark.parametrize(
    "setup_fn,expected_error_snippet",
    [
        (lambda d: None, "missing or is not a symlink"),  # missing current
        (lambda d: (d / "current").mkdir(), "missing or is not a symlink"),  # regular directory
        (lambda d: _write(d / "current", "regular file"), "missing or is not a symlink"),  # regular file
        (lambda d: (d / "current").symlink_to("gen-missing"), "unresolvable or broken link"),  # broken link
        (lambda d: (d / "current").symlink_to("/etc/passwd"), "invalid generations/current symlink target text"),  # absolute link
        (lambda d: (d / "current").symlink_to(""), "invalid generations/current symlink target text"),  # empty component / target
        (lambda d: (d / "current").symlink_to("."), "invalid generations/current symlink target text"),  # dot component
        (lambda d: (d / "current").symlink_to(".."), "invalid generations/current symlink target text"),  # dot-dot component
        (lambda d: (d / "current").symlink_to("../escape"), "invalid generations/current symlink target text"),  # traversal
        (lambda d: (d / "current").symlink_to(r"gen\backslash"), "invalid generations/current symlink target text"),  # backslash
        (lambda d: (d / "current").symlink_to("gen\x01ctrl"), "invalid generations/current symlink target text"),  # control character
        (lambda d: (d / "current").symlink_to("current"), "target cannot be current symlink itself"),  # target equal to current
    ],
)
def test_create_pre_stop_rejection_invalid_current_links(
    hermetic, tmp_path: Path, setup_fn, expected_error_snippet
):
    """Pre-stop rejection, no service log/stop, bounded error, and no archive for invalid current links."""
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    gen_dir = repo / "data" / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)
    setup_fn(gen_dir)

    archive, res = standard_create(
        hermetic,
        repo,
        web,
        extra=["--materialize-generations-current"],
    )
    assert res.returncode != 0
    assert expected_error_snippet.lower() in res.stderr.lower(), (expected_error_snippet, res.stderr)
    assert not hermetic.systemctl_log.exists()
    assert not archive.exists()
    assert not Path(str(archive) + ".sha256").exists()


def test_create_pre_stop_rejection_target_is_file(hermetic, tmp_path: Path):
    """Target of current is a regular file instead of a directory: rejected pre-stop."""
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    gen_dir = repo / "data" / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)
    _write(gen_dir / "gen-file", "not a directory")
    (gen_dir / "current").symlink_to("gen-file")

    archive, res = standard_create(
        hermetic,
        repo,
        web,
        extra=["--materialize-generations-current"],
    )
    assert res.returncode != 0
    assert "not an ordinary directory" in res.stderr.lower()
    assert not hermetic.systemctl_log.exists()
    assert not archive.exists()
    assert not Path(str(archive) + ".sha256").exists()


def test_create_pre_stop_rejection_target_contains_unsafe_entries(hermetic, tmp_path: Path):
    """Target contains nested symlinks, hardlinks, excluded secrets, or unsafe permissions:
    fail closed before stop, do not create archive."""
    cases = [
        ("nested_dir_symlink", lambda t: (t / "nested_dir").symlink_to(t), "nested directory symlink"),
        ("nested_file_symlink", lambda t: (t / "nested_file").symlink_to("f.txt"), "nested symlink"),
        ("hardlinked_file", lambda t: os.link(t / "f.txt", t / "f_hardlink.txt"), "hardlinked regular file"),
        ("excluded_secret", lambda t: _write(t / "secrets.json", '{"key": "val"}'), "excluded secret-like path"),
        ("excluded_env", lambda t: _write(t / ".env.local", "FOO=bar"), "excluded secret-like path"),
        ("sticky_mode", lambda t: (t / "f.txt").chmod(0o1644), "sticky mode"),
        ("setuid_mode", lambda t: (t / "f.txt").chmod(0o4755), "setuid mode"),
        ("setgid_mode", lambda t: (t / "f.txt").chmod(0o2755), "setgid mode"),
        ("control_filename", lambda t: _write(t / "file\x01bad.txt", "data"), "control character"),
    ]

    for label, setup_bad, expected_snippet in cases:
        repo = make_repo(tmp_path / f"repo-{label}")
        commit_all(repo)
        web = make_web_root(tmp_path / f"web-{label}", "x" * 40)
        gen_dir = repo / "data" / "generations"
        gen_target = gen_dir / "gen-001"
        gen_target.mkdir(parents=True, exist_ok=True)
        _write(gen_target / "f.txt", "content\n")
        (gen_dir / "current").symlink_to("gen-001")

        setup_bad(gen_target)

        archive, res = standard_create(
            hermetic,
            repo,
            web,
            archive_name=f"backup-{label}" + ARCHIVE_SUFFIX,
            extra=["--materialize-generations-current"],
        )
        assert res.returncode != 0, label
        assert expected_snippet.lower() in res.stderr.lower(), (label, expected_snippet, res.stderr)
        assert not hermetic.systemctl_log.exists(), label
        assert not archive.exists(), label
        assert not Path(str(archive) + ".sha256").exists(), label


def test_create_pre_stop_rejection_empty_target_dir(hermetic, tmp_path: Path):
    """Empty generation target directory must be rejected before service stop."""
    repo = make_repo(tmp_path / "repo-empty")
    commit_all(repo)
    web = make_web_root(tmp_path / "web-empty", "x" * 40)
    gen_dir = repo / "data" / "generations"
    gen_target = gen_dir / "gen-empty"
    gen_target.mkdir(parents=True, exist_ok=True)
    (gen_dir / "current").symlink_to("gen-empty")

    archive, res = standard_create(
        hermetic,
        repo,
        web,
        extra=["--materialize-generations-current"],
    )
    assert res.returncode != 0
    assert "empty" in res.stderr.lower()
    assert not hermetic.systemctl_log.exists()
    assert not archive.exists()
    assert not Path(str(archive) + ".sha256").exists()


def test_create_pre_stop_rejection_symlinked_target_component(hermetic, tmp_path: Path):
    """Target path component containing a symlink is rejected before stop."""
    repo = make_repo(tmp_path / "repo-comp")
    commit_all(repo)
    web = make_web_root(tmp_path / "web-comp", "x" * 40)
    gen_dir = repo / "data" / "generations"
    real_target = gen_dir / "real-001"
    real_target.mkdir(parents=True, exist_ok=True)
    _write(real_target / "f.txt", "content\n")

    # Intermediate component symlink: intermediate -> real-001, current -> intermediate
    (gen_dir / "intermediate").symlink_to("real-001")
    (gen_dir / "current").symlink_to("intermediate")

    archive, res = standard_create(
        hermetic,
        repo,
        web,
        extra=["--materialize-generations-current"],
    )
    assert res.returncode != 0
    assert "symlink in generations target path component" in res.stderr.lower()
    assert not hermetic.systemctl_log.exists()
    assert not archive.exists()
    assert not Path(str(archive) + ".sha256").exists()


def test_create_pre_stop_rejection_unreadable_target(hermetic, tmp_path: Path):
    """Unreadable target directory is rejected before stop with clean diagnostic."""
    repo = make_repo(tmp_path / "repo-unreadable")
    commit_all(repo)
    web = make_web_root(tmp_path / "web-unreadable", "x" * 40)
    gen_dir = repo / "data" / "generations"
    gen_target = gen_dir / "gen-unreadable"
    gen_target.mkdir(parents=True, exist_ok=True)
    _write(gen_target / "f.txt", "content\n")
    (gen_dir / "current").symlink_to("gen-unreadable")

    # Make target unreadable
    gen_target.chmod(0o000)
    try:
        archive, res = standard_create(
            hermetic,
            repo,
            web,
            extra=["--materialize-generations-current"],
        )
        assert res.returncode != 0
        assert "cannot traverse generation target" in res.stderr.lower()
        assert not hermetic.systemctl_log.exists()
        assert not archive.exists()
        assert not Path(str(archive) + ".sha256").exists()
    finally:
        gen_target.chmod(0o755)


def test_create_pre_stop_rejection_fifo_in_target(hermetic, tmp_path: Path):
    """Non-regular file (FIFO) in generation target is rejected before stop."""
    repo = make_repo(tmp_path / "repo-fifo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web-fifo", "x" * 40)
    gen_dir = repo / "data" / "generations"
    gen_target = gen_dir / "gen-fifo"
    gen_target.mkdir(parents=True, exist_ok=True)
    _write(gen_target / "f.txt", "content\n")
    os.mkfifo(gen_target / "named_pipe")
    (gen_dir / "current").symlink_to("gen-fifo")

    archive, res = standard_create(
        hermetic,
        repo,
        web,
        extra=["--materialize-generations-current"],
    )
    assert res.returncode != 0
    assert "non-regular file" in res.stderr.lower()
    assert not hermetic.systemctl_log.exists()
    assert not archive.exists()
    assert not Path(str(archive) + ".sha256").exists()


def test_create_pre_stop_rejection_containment_escape_via_alias(hermetic, tmp_path: Path):
    """Containment escape via alias directory symlink is rejected before stop."""
    repo = make_repo(tmp_path / "repo-alias")
    commit_all(repo)
    web = make_web_root(tmp_path / "web-alias", "x" * 40)
    external = tmp_path / "external-gen"
    external.mkdir(parents=True, exist_ok=True)
    _write(external / "f.txt", "secret\n")

    gen_dir = repo / "data" / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)
    (gen_dir / "escape_alias").symlink_to(external)
    (gen_dir / "current").symlink_to("escape_alias")

    archive, res = standard_create(
        hermetic,
        repo,
        web,
        extra=["--materialize-generations-current"],
    )
    assert res.returncode != 0
    assert "escapes generations directory" in res.stderr.lower() or "symlink in generations target path component" in res.stderr.lower()
    assert not hermetic.systemctl_log.exists()
    assert not archive.exists()
    assert not Path(str(archive) + ".sha256").exists()


def test_create_secret_safe_diagnostics(hermetic, tmp_path: Path):
    """Secret tokens in invalid link text or excluded filenames must NOT appear in stderr/stdout."""
    secret_link_token = "SUPER_SECRET_LINK_TOKEN_XYZ123"
    secret_file_token = "SUPER_SECRET_FILE_TOKEN_ABC456"

    # Case 1: Secret in link target
    repo1 = make_repo(tmp_path / "repo1")
    commit_all(repo1)
    web1 = make_web_root(tmp_path / "web1", "x" * 40)
    gen_dir1 = repo1 / "data" / "generations"
    gen_dir1.mkdir(parents=True, exist_ok=True)
    # Link with control character containing secret token
    (gen_dir1 / "current").symlink_to(f"gen\x01{secret_link_token}")
    _, res1 = standard_create(hermetic, repo1, web1, extra=["--materialize-generations-current"])
    assert res1.returncode != 0
    assert secret_link_token not in res1.stderr
    assert secret_link_token not in res1.stdout
    assert str(tmp_path) not in res1.stderr

    # Case 2: Secret in excluded filename inside target
    repo2 = make_repo(tmp_path / "repo2")
    commit_all(repo2)
    web2 = make_web_root(tmp_path / "web2", "x" * 40)
    gen_dir2 = repo2 / "data" / "generations"
    gen_target2 = gen_dir2 / "gen-002"
    gen_target2.mkdir(parents=True, exist_ok=True)
    _write(gen_target2 / f"secret_{secret_file_token}.json", '{"key": 1}')
    (gen_dir2 / "current").symlink_to("gen-002")
    _, res2 = standard_create(hermetic, repo2, web2, extra=["--materialize-generations-current"])
    assert res2.returncode != 0
    assert secret_file_token not in res2.stderr
    assert secret_file_token not in res2.stdout
    assert str(tmp_path) not in res2.stderr


def test_create_with_flag_still_rejects_other_symlinks(hermetic, tmp_path: Path):
    """All non-current directory symlinks under data/ or static/web remain rejected
    even when --materialize-generations-current is passed."""
    # Symlink elsewhere in data/
    repo = make_repo(tmp_path / "repo1")
    commit_all(repo)
    web = make_web_root(tmp_path / "web1", "x" * 40)
    gen_dir = repo / "data" / "generations"
    gen_target = gen_dir / "gen-001"
    gen_target.mkdir(parents=True)
    _write(gen_target / "f.txt", "ok\n")
    (gen_dir / "current").symlink_to("gen-001")

    # another directory symlink under data/
    other = tmp_path / "other"
    other.mkdir()
    (repo / "data" / "other_link").symlink_to(other, target_is_directory=True)

    archive, res = standard_create(
        hermetic,
        repo,
        web,
        extra=["--materialize-generations-current"],
    )
    assert res.returncode != 0
    assert "symlink" in res.stderr.lower()
    assert not hermetic.systemctl_log.exists()
    assert not archive.exists()


def test_create_post_stop_tamper_fails_closed_and_restarts(hermetic, tmp_path: Path):
    """A controller stop that changes the link or target after preflight
    causes a post-stop snapshot mismatch failure, creates no archive, and attempts restart."""
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)
    gen_dir = repo / "data" / "generations"
    gen_target1 = gen_dir / "gen-001"
    gen_target1.mkdir(parents=True)
    _write(gen_target1 / "f.txt", "v1\n")
    (gen_dir / "current").symlink_to("gen-001")

    # Custom systemctl stop that mutates data/generations/current during stop
    fake_body = (
        '#!/bin/sh\n'
        f'printf \'%s\\n\' "$*" >> "{hermetic.systemctl_log}"\n'
        'case "$1" in\n'
        '  is-active) printf "active\\n"; exit 0 ;;\n'
        f'  stop) rm "{gen_dir / "current"}"; ln -s gen-002 "{gen_dir / "current"}"; exit 0 ;;\n'
        '  start) exit 0 ;;\n'
        'esac\n'
        'exit 0\n'
    )
    make_fake(hermetic.bin / "systemctl", fake_body)

    archive, res = standard_create(
        hermetic,
        repo,
        web,
        extra=["--materialize-generations-current"],
    )
    assert res.returncode != 0
    assert "snapshot" in res.stderr.lower() or "mismatch" in res.stderr.lower() or "changed" in res.stderr.lower()
    assert not archive.exists()
    log = hermetic.systemctl_log.read_text(encoding="utf-8").splitlines()
    # Ensure systemctl stop was called, then restart was attempted
    assert "stop portfolio-lab-tasker" in log
    assert "start portfolio-lab-tasker" in log


def test_verify_rejects_tampered_generation_materialization_manifest(hermetic, tmp_path: Path):
    """Tampered manifest metadata (bad schema, unsafe original link, target mismatch,
    missing/extra keys, member mismatch, digest mismatch) fails verify and fails restore before target mutation."""
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha, generator_sha=source_sha[:12])

    gen_dir = repo / "data" / "generations"
    gen_id = "gen-2026-09-03-001"
    gen_target = gen_dir / gen_id
    gen_target.mkdir(parents=True)
    _write(gen_target / "manifest.json", '{"schema": "portfolio-lab-generation/v1"}\n')
    (gen_dir / "current").symlink_to(gen_id)

    archive, res = standard_create(
        hermetic,
        repo,
        web,
        extra=["--materialize-generations-current"],
    )
    assert res.returncode == 0, res.stderr

    # Helper to repack archive with modified manifest and updated member list
    def repack_with_modified_manifest(mod_fn, output_name: str) -> Path:
        out_archive = hermetic.tmp / "backups" / (output_name + ARCHIVE_SUFFIX)
        extract_dir = hermetic.tmp / f"extract-{output_name}"
        extract_tar(archive, extract_dir)
        mpath = extract_dir / "recovery-manifest.json"
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        mod_fn(manifest, extract_dir)
        mpath.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        all_members = [
            p.relative_to(extract_dir).as_posix()
            for p in sorted(extract_dir.rglob("*"))
            if p.is_file()
        ]
        subprocess.run(
            ["tar", "-cf", str(out_archive), "-C", str(extract_dir), *all_members],
            check=True,
            env={**os.environ, "COPYFILE_DISABLE": "1"},
        )
        sidecar = Path(str(out_archive) + ".sha256")
        sidecar.write_text(f"{_sha256_bytes(out_archive.read_bytes())}  {out_archive.name}\n", encoding="utf-8")
        return out_archive

    # Test cases for metadata tampering
    tamper_cases = [
        ("bad_schema", lambda m, ed: m["generation_materialization"].update({"schema_version": "wrong-schema"})),
        ("wrong_link_path", lambda m, ed: m["generation_materialization"].update({"link_path": "data/generations/wrong_link"})),
        ("wrong_archive_path", lambda m, ed: m["generation_materialization"].update({"archive_path": "runtime/data/generations/wrong_archive"})),
        ("extra_key", lambda m, ed: m["generation_materialization"].update({"extra_key": "forbidden"})),
        ("missing_key_schema", lambda m, ed: m["generation_materialization"].pop("schema_version")),
        ("missing_key_link_path", lambda m, ed: m["generation_materialization"].pop("link_path")),
        ("missing_key_original_link", lambda m, ed: m["generation_materialization"].pop("original_link")),
        ("missing_key_target_path", lambda m, ed: m["generation_materialization"].pop("target_path")),
        ("missing_key_archive_path", lambda m, ed: m["generation_materialization"].pop("archive_path")),
        ("unsafe_link", lambda m, ed: m["generation_materialization"].update({"original_link": "../escape", "target_path": "data/generations/../escape"})),
        ("target_mismatch", lambda m, ed: m["generation_materialization"].update({"target_path": "data/generations/wrong-target"})),
    ]

    for label, tamper_fn in tamper_cases:
        bad_tar = repack_with_modified_manifest(tamper_fn, f"tamper-{label}")
        vres = run_recovery(["verify", "--archive", str(bad_tar)], hermetic)
        assert vres.returncode != 0, label

        # Also prove restore fails before target mutation
        app_dir = hermetic.tmp / f"app-{label}"
        web_root = hermetic.tmp / f"web-{label}"
        rres = run_recovery(
            [
                "restore",
                "--archive", str(bad_tar),
                "--app-dir", str(app_dir),
                "--web-root", str(web_root),
                "--target-mode", "dev",
            ],
            hermetic,
        )
        assert rres.returncode != 0, label
        assert not app_dir.exists(), label
        assert not web_root.exists(), label


def test_verify_rejects_tampered_generation_member_pairs(hermetic, tmp_path: Path):
    """Tampering member pairs (digest, size, mode mismatch, missing member, extra member)
    fails verify and fails restore before mutation."""
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha, generator_sha=source_sha[:12])

    gen_dir = repo / "data" / "generations"
    gen_id = "gen-2026-09-03-001"
    gen_target = gen_dir / gen_id
    gen_target.mkdir(parents=True)
    _write(gen_target / "manifest.json", '{"schema": "portfolio-lab-generation/v1"}\n')
    _write(gen_target / "extra.json", '{"item": 1}\n')
    (gen_dir / "current").symlink_to(gen_id)

    archive, res = standard_create(
        hermetic,
        repo,
        web,
        extra=["--materialize-generations-current"],
    )
    assert res.returncode == 0, res.stderr

    def repack_tampered_members(tamper_fn, output_name: str) -> Path:
        out_archive = hermetic.tmp / "backups" / (output_name + ARCHIVE_SUFFIX)
        extract_dir = hermetic.tmp / f"extract-{output_name}"
        extract_tar(archive, extract_dir)
        mpath = extract_dir / "recovery-manifest.json"
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        tamper_fn(manifest, extract_dir)
        mpath.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        all_members = [
            p.relative_to(extract_dir).as_posix()
            for p in sorted(extract_dir.rglob("*"))
            if p.is_file()
        ]
        subprocess.run(
            ["tar", "-cf", str(out_archive), "-C", str(extract_dir), *all_members],
            check=True,
            env={**os.environ, "COPYFILE_DISABLE": "1"},
        )
        sidecar = Path(str(out_archive) + ".sha256")
        sidecar.write_text(f"{_sha256_bytes(out_archive.read_bytes())}  {out_archive.name}\n", encoding="utf-8")
        return out_archive

    # 1. Tamper digest of current copy only (and update global member digest so global verify passes and generation check fails)
    def tamper_digest(m, ed):
        f = ed / "runtime" / "data" / "generations" / "current" / "manifest.json"
        f.write_text('{"tampered": true}\n', encoding="utf-8")
        new_sha = _sha256_bytes(f.read_bytes())
        for entry in m["members"]:
            if entry["path"] == "runtime/data/generations/current/manifest.json":
                entry["sha256"] = new_sha
                entry["bytes"] = f.stat().st_size

    # 2. Tamper mode of current copy only
    def tamper_mode(m, ed):
        for entry in m["members"]:
            if entry["path"] == "runtime/data/generations/current/manifest.json":
                entry["mode"] = 0o644 if entry["mode"] != 0o644 else 0o600
                (ed / entry["path"]).chmod(entry["mode"])

    # 3. Extra member in current not in target
    def tamper_extra_current(m, ed):
        extra_f = ed / "runtime" / "data" / "generations" / "current" / "added.json"
        extra_f.write_text('{"added": 1}\n', encoding="utf-8")
        extra_f.chmod(0o600)
        m["members"].append({
            "path": "runtime/data/generations/current/added.json",
            "sha256": _sha256_bytes(extra_f.read_bytes()),
            "bytes": extra_f.stat().st_size,
            "mode": 0o600,
        })
        m["members"] = sorted(m["members"], key=lambda e: e["path"])

    # 4. Extra member in target not in current
    def tamper_extra_target(m, ed):
        extra_f = ed / "runtime" / "data" / "generations" / gen_id / "added.json"
        extra_f.write_text('{"added": 1}\n', encoding="utf-8")
        extra_f.chmod(0o600)
        m["members"].append({
            "path": f"runtime/data/generations/{gen_id}/added.json",
            "sha256": _sha256_bytes(extra_f.read_bytes()),
            "bytes": extra_f.stat().st_size,
            "mode": 0o600,
        })
        m["members"] = sorted(m["members"], key=lambda e: e["path"])

    member_tamper_cases = [
        ("digest_mismatch", tamper_digest),
        ("mode_mismatch", tamper_mode),
        ("extra_current", tamper_extra_current),
        ("extra_target", tamper_extra_target),
    ]

    for label, tamper_fn in member_tamper_cases:
        bad_tar = repack_tampered_members(tamper_fn, f"tamper-mem-{label}")
        vres = run_recovery(["verify", "--archive", str(bad_tar)], hermetic)
        assert vres.returncode != 0, label
        vrep = json.loads(vres.stdout)
        assert vrep["checks"]["generation_materialization_members_match"] is False, (label, vrep)

        # Ensure restore fails before target mutation
        app_dir = hermetic.tmp / f"app-mem-{label}"
        web_root = hermetic.tmp / f"web-mem-{label}"
        rres = run_recovery(
            [
                "restore",
                "--archive", str(bad_tar),
                "--app-dir", str(app_dir),
                "--web-root", str(web_root),
                "--target-mode", "dev",
            ],
            hermetic,
        )
        assert rres.returncode != 0, label
        assert not app_dir.exists(), label
        assert not web_root.exists(), label


def test_restore_reconstruction_failure_leaves_target_untouched(hermetic, tmp_path: Path):
    """If staging reconstruction fails (e.g. invalid target in staged tree),
    pre-existing target app and web roots remain untouched."""
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha, generator_sha=source_sha[:12])

    gen_dir = repo / "data" / "generations"
    gen_id = "gen-2026-09-03-001"
    gen_target = gen_dir / gen_id
    gen_target.mkdir(parents=True)
    _write(gen_target / "f.txt", "data\n")
    (gen_dir / "current").symlink_to(gen_id)

    archive, res = standard_create(
        hermetic,
        repo,
        web,
        extra=["--materialize-generations-current"],
    )
    assert res.returncode == 0, res.stderr

    # Existing pre-existing targets
    app_dir = hermetic.tmp / "app-existing"
    _write(app_dir / "keep.txt", "keep app\n")
    web_root = hermetic.tmp / "web-existing"
    _write(web_root / "keep.txt", "keep web\n")

    # Tamper with the archive by removing the target member but keeping materialized current
    # So verify will fail or restore staging will fail
    bad_tar = hermetic.tmp / "backups" / ("missing-target" + ARCHIVE_SUFFIX)
    extract_dir = hermetic.tmp / "extract-missing-target"
    extract_tar(archive, extract_dir)
    # Remove target file and its manifest entry
    (extract_dir / "runtime" / "data" / "generations" / gen_id / "f.txt").unlink()
    mpath = extract_dir / "recovery-manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    manifest["members"] = [
        m for m in manifest["members"]
        if not m["path"].startswith(f"runtime/data/generations/{gen_id}/")
    ]
    mpath.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    all_members = [
        p.relative_to(extract_dir).as_posix()
        for p in sorted(extract_dir.rglob("*"))
        if p.is_file()
    ]
    subprocess.run(
        ["tar", "-cf", str(bad_tar), "-C", str(extract_dir), *all_members],
        check=True,
        env={**os.environ, "COPYFILE_DISABLE": "1"},
    )
    sidecar = Path(str(bad_tar) + ".sha256")
    sidecar.write_text(f"{_sha256_bytes(bad_tar.read_bytes())}  {bad_tar.name}\n", encoding="utf-8")

    rres = run_recovery(
        [
            "restore",
            "--archive", str(bad_tar),
            "--app-dir", str(app_dir),
            "--web-root", str(web_root),
            "--target-mode", "dev",
        ],
        hermetic,
    )
    assert rres.returncode != 0
    # Pre-existing files must be untouched
    assert (app_dir / "keep.txt").read_text(encoding="utf-8") == "keep app\n"
    assert (web_root / "keep.txt").read_text(encoding="utf-8") == "keep web\n"


def test_restore_reconstruction_staging_mutation_leaves_target_untouched(hermetic, tmp_path: Path, monkeypatch):
    """An archive that passes verify_archive but experiences a staging mutation/failure
    during reconstruction leaves pre-existing target files untouched."""
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha, generator_sha=source_sha[:12])

    gen_dir = repo / "data" / "generations"
    gen_id = "gen-2026-09-03-001"
    gen_target = gen_dir / gen_id
    gen_target.mkdir(parents=True)
    _write(gen_target / "f.txt", "data\n")
    (gen_dir / "current").symlink_to(gen_id)

    archive, res = standard_create(
        hermetic,
        repo,
        web,
        extra=["--materialize-generations-current"],
    )
    assert res.returncode == 0, res.stderr

    # Existing pre-existing targets
    app_dir = hermetic.tmp / "app-staging-fail"
    _write(app_dir / "keep.txt", "keep app\n")
    web_root = hermetic.tmp / "web-staging-fail"
    _write(web_root / "keep.txt", "keep web\n")

    # In cmd_restore, hook os.symlink when called for 'current' to raise an error
    # We can test this via a subprocess wrapper or python invocation where os.symlink is monkeypatched
    patch_script = hermetic.tmp / "run_restore_patched.py"
    patch_script.write_text(f"""
import sys, os
sys.path.insert(0, "{PROJECT_ROOT}")
import scripts.portfolio_lab_recovery as plr

orig_symlink = os.symlink
def patched_symlink(src, dst, *args, **kwargs):
    if "current" in str(dst):
        raise OSError("injected symlink failure")
    return orig_symlink(src, dst, *args, **kwargs)

os.symlink = patched_symlink
sys.argv = [
    "portfolio_lab_recovery.py",
    "restore",
    "--archive", "{archive}",
    "--app-dir", "{app_dir}",
    "--web-root", "{web_root}",
    "--target-mode", "dev",
]
raise SystemExit(plr.main())
""", encoding="utf-8")

    res = subprocess.run([sys.executable, str(patch_script)], env={**os.environ, **hermetic.env}, capture_output=True, text=True)
    assert res.returncode != 0
    assert (app_dir / "keep.txt").read_text(encoding="utf-8") == "keep app\n"
    assert (web_root / "keep.txt").read_text(encoding="utf-8") == "keep web\n"


def test_create_failure_cleans_up_destination_archive_and_sidecar(hermetic, tmp_path: Path):
    """Transactional cleanup: any failure during generation materialization create
    after archive/sidecar creation but before successful verified completion must remove both destination files."""
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    gen_dir = repo / "data" / "generations"
    gen_id = "gen-2026-09-03-001"
    gen_target = gen_dir / gen_id
    gen_target.mkdir(parents=True)
    _write(gen_target / "manifest.json", '{"schema": "portfolio-lab-generation/v1"}\n')
    _write(gen_target / "f.txt", "initial\n")
    (gen_dir / "current").symlink_to(gen_id)

    # Corrupt sqlite database will fail self-verify in verify_archive
    _write(repo / "data/market.db", b"garbage not a sqlite database" * 100)

    archive = hermetic.tmp / "backups" / ("failed-verify" + ARCHIVE_SUFFIX)
    archive.parent.mkdir(parents=True, exist_ok=True)
    _write(hermetic.units / f"{TASKER}.service", UNIT_TEXT)
    _write(repo / "data/tasker_status.json", json.dumps({"state": "idle", "version": 1}) + "\n")

    res = run_recovery(
        [
            "create",
            "--app-dir", str(repo),
            "--web-root", str(web),
            "--tasker-service", TASKER,
            "--archive", str(archive),
            "--storage-encryption-attested",
            "--materialize-generations-current",
        ],
        hermetic,
    )
    assert res.returncode != 0
    assert not archive.exists()
    assert not Path(str(archive) + ".sha256").exists()


def test_create_post_stop_staging_drift_fails_closed_restarts_and_cleans_up(hermetic, tmp_path: Path):
    """Mutating a source file between post-stop validation and staging copies
    causes staged tree snapshot validation failure, source restart, no archive, no sidecar."""
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha)

    gen_dir = repo / "data" / "generations"
    gen_id = "gen-2026-09-03-001"
    gen_target = gen_dir / gen_id
    gen_target.mkdir(parents=True)
    _write(gen_target / "manifest.json", '{"schema": "portfolio-lab-generation/v1"}\n')
    _write(gen_target / "f.txt", "initial\n")
    (gen_dir / "current").symlink_to(gen_id)

    archive = hermetic.tmp / "backups" / ("drift-archive" + ARCHIVE_SUFFIX)
    archive.parent.mkdir(parents=True, exist_ok=True)
    _write(hermetic.units / f"{TASKER}.service", UNIT_TEXT)
    _write(repo / "data/tasker_status.json", json.dumps({"state": "idle", "version": 1}) + "\n")

    # Patch copy_members_to_staging to mutate a source file in target during copy
    drift_script = hermetic.tmp / "run_drift_create.py"
    drift_script.write_text(f"""
import sys, os
sys.path.insert(0, "{PROJECT_ROOT}")
import scripts.portfolio_lab_recovery as plr

orig_copy = plr.copy_members_to_staging
def drifting_copy(members, src_root, staging, prefix):
    # mutate f.txt in source before copying
    target_f = plr.Path("{gen_target}") / "f.txt"
    if target_f.is_file():
        target_f.write_text("drifted content\\n", encoding="utf-8")
    return orig_copy(members, src_root, staging, prefix)

plr.copy_members_to_staging = drifting_copy
sys.argv = [
    "portfolio_lab_recovery.py",
    "create",
    "--app-dir", "{repo}",
    "--web-root", "{web}",
    "--tasker-service", "{TASKER}",
    "--archive", "{archive}",
    "--storage-encryption-attested",
    "--materialize-generations-current",
]
raise SystemExit(plr.main())
""", encoding="utf-8")

    res = subprocess.run(
        [sys.executable, str(drift_script)],
        env={**os.environ, **hermetic.env, "PATH": f"{hermetic.bin}:{os.environ.get('PATH', '')}"},
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "snapshot" in res.stderr.lower() or "drift" in res.stderr.lower() or "match" in res.stderr.lower(), res.stderr
    assert not archive.exists()
    assert not Path(str(archive) + ".sha256").exists()
    log = hermetic.systemctl_log.read_text(encoding="utf-8").splitlines()
    assert "stop portfolio-lab-tasker" in log
    assert "start portfolio-lab-tasker" in log


def test_create_materialized_tar_failure_cleans_up_destination_archive_and_sidecar(hermetic, tmp_path: Path):
    """Transactional cleanup: partial tar failure for materialized create cleans up archive/sidecar."""
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    gen_dir = repo / "data" / "generations"
    gen_id = "gen-2026-09-03-001"
    gen_target = gen_dir / gen_id
    gen_target.mkdir(parents=True)
    _write(gen_target / "manifest.json", '{"schema": "portfolio-lab-generation/v1"}\n')
    _write(gen_target / "f.txt", "content\n")
    (gen_dir / "current").symlink_to(gen_id)

    archive = hermetic.tmp / "backups" / ("failed-tar" + ARCHIVE_SUFFIX)
    archive.parent.mkdir(parents=True, exist_ok=True)
    _write(hermetic.units / f"{TASKER}.service", UNIT_TEXT)
    _write(repo / "data/tasker_status.json", json.dumps({"state": "idle", "version": 1}) + "\n")

    # Hook tar to create partial file and return nonzero exit
    fake_tar_script = hermetic.tmp / "fake_failing_tar.sh"
    fake_tar_script.write_text("""#!/bin/sh
# write partial archive bytes to target archive argument
for arg in "$@"; do
    case "$arg" in
        *.tar) echo "partial junk" > "$arg" ;;
    esac
done
exit 1
""", encoding="utf-8")
    fake_tar_script.chmod(0o755)

    res = run_recovery(
        [
            "create",
            "--app-dir", str(repo),
            "--web-root", str(web),
            "--tasker-service", TASKER,
            "--archive", str(archive),
            "--storage-encryption-attested",
            "--materialize-generations-current",
        ],
        hermetic,
        extra_env={"PLR_TAR": str(fake_tar_script)},
    )
    assert res.returncode != 0
    assert not archive.exists()
    assert not Path(str(archive) + ".sha256").exists()


def test_create_materialized_sidecar_write_failure_cleans_up(hermetic, tmp_path: Path):
    """Transactional cleanup: sidecar write failure cleans up archive and sidecar."""
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    gen_dir = repo / "data" / "generations"
    gen_id = "gen-2026-09-03-001"
    gen_target = gen_dir / gen_id
    gen_target.mkdir(parents=True)
    _write(gen_target / "manifest.json", '{"schema": "portfolio-lab-generation/v1"}\n')
    _write(gen_target / "f.txt", "content\n")
    (gen_dir / "current").symlink_to(gen_id)

    archive = hermetic.tmp / "backups" / ("failed-sidecar" + ARCHIVE_SUFFIX)
    archive.parent.mkdir(parents=True, exist_ok=True)
    _write(hermetic.units / f"{TASKER}.service", UNIT_TEXT)
    _write(repo / "data/tasker_status.json", json.dumps({"state": "idle", "version": 1}) + "\n")

    fail_script = hermetic.tmp / "run_sidecar_fail.py"
    fail_script.write_text(f"""
import sys, os
sys.path.insert(0, "{PROJECT_ROOT}")
import scripts.portfolio_lab_recovery as plr

orig_write_text = plr.Path.write_text
def failing_write_text(self, text, *args, **kwargs):
    if str(self).endswith(".sha256"):
        # write partial sidecar then raise
        orig_write_text(self, "partial sidecar", *args, **kwargs)
        raise OSError("injected sidecar write failure")
    return orig_write_text(self, text, *args, **kwargs)

plr.Path.write_text = failing_write_text
sys.argv = [
    "portfolio_lab_recovery.py",
    "create",
    "--app-dir", "{repo}",
    "--web-root", "{web}",
    "--tasker-service", "{TASKER}",
    "--archive", "{archive}",
    "--storage-encryption-attested",
    "--materialize-generations-current",
]
raise SystemExit(plr.main())
""", encoding="utf-8")

    res = subprocess.run([sys.executable, str(fail_script)], env={**os.environ, **hermetic.env, "PATH": f"{hermetic.bin}:{os.environ.get('PATH', '')}"}, capture_output=True, text=True)
    assert res.returncode != 0
    assert not archive.exists()
    assert not Path(str(archive) + ".sha256").exists()


def test_create_materialized_retains_verified_archive_on_restart_failure(hermetic, tmp_path: Path):
    """Verified archive and sidecar are preserved if only source service restart fails."""
    set_systemctl_mode(hermetic, "start-fail")
    repo = make_repo(tmp_path / "repo")
    commit_all(repo)
    web = make_web_root(tmp_path / "web", "x" * 40)
    gen_dir = repo / "data" / "generations"
    gen_id = "gen-2026-09-03-001"
    gen_target = gen_dir / gen_id
    gen_target.mkdir(parents=True)
    _write(gen_target / "manifest.json", '{"schema": "portfolio-lab-generation/v1"}\n')
    _write(gen_target / "f.txt", "content\n")
    (gen_dir / "current").symlink_to(gen_id)

    archive, res = standard_create(
        hermetic,
        repo,
        web,
        archive_name="restart-fail" + ARCHIVE_SUFFIX,
        extra=["--materialize-generations-current"],
    )
    assert res.returncode != 0
    report = json.loads(res.stdout)
    assert report["ok"] is False
    assert report["service_started"] is False
    assert archive.exists()
    assert Path(str(archive) + ".sha256").exists()


def test_parse_and_validate_generation_metadata_secret_safe_tamper(hermetic, tmp_path: Path):
    """Metadata validation failures must never echo raw hostile strings or secret sentinels."""
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha, generator_sha=source_sha[:12])

    gen_dir = repo / "data" / "generations"
    gen_id = "gen-2026-09-03-001"
    gen_target = gen_dir / gen_id
    gen_target.mkdir(parents=True)
    _write(gen_target / "manifest.json", '{"schema": "portfolio-lab-generation/v1"}\n')
    (gen_dir / "current").symlink_to(gen_id)

    archive, res = standard_create(hermetic, repo, web, extra=["--materialize-generations-current"])
    assert res.returncode == 0, res.stderr

    secret_sentinel = "MY_SUPER_SECRET_HOSTILE_VALUE_9999"

    def repack(mutator, name: str) -> Path:
        out_archive = hermetic.tmp / "backups" / (name + ARCHIVE_SUFFIX)
        extract_dir = hermetic.tmp / f"extract-{name}"
        extract_tar(archive, extract_dir)
        mpath = extract_dir / "recovery-manifest.json"
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        mutator(manifest)
        mpath.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        all_members = [
            p.relative_to(extract_dir).as_posix()
            for p in sorted(extract_dir.rglob("*"))
            if p.is_file()
        ]
        subprocess.run(
            ["tar", "-cf", str(out_archive), "-C", str(extract_dir), *all_members],
            check=True,
            env={**os.environ, "COPYFILE_DISABLE": "1"},
        )
        sidecar = Path(str(out_archive) + ".sha256")
        sidecar.write_text(f"{_sha256_bytes(out_archive.read_bytes())}  {out_archive.name}\n", encoding="utf-8")
        return out_archive

    cases = [
        ("schema", lambda m: m["generation_materialization"].update({"schema_version": f"bad_schema_{secret_sentinel}"})),
        ("link_path", lambda m: m["generation_materialization"].update({"link_path": f"data/generations/{secret_sentinel}"})),
        ("archive_path", lambda m: m["generation_materialization"].update({"archive_path": f"runtime/data/{secret_sentinel}"})),
        ("orig_link", lambda m: m["generation_materialization"].update({"original_link": f"../{secret_sentinel}"})),
        ("target_path", lambda m: m["generation_materialization"].update({"target_path": f"data/generations/{secret_sentinel}"})),
    ]

    for label, mutator in cases:
        bad_tar = repack(mutator, f"sentinel-{label}")
        vres = run_recovery(["verify", "--archive", str(bad_tar)], hermetic)
        assert vres.returncode != 0
        assert secret_sentinel not in vres.stderr, label
        assert secret_sentinel not in vres.stdout, label

        rres = run_recovery(
            [
                "restore",
                "--archive", str(bad_tar),
                "--app-dir", str(hermetic.tmp / f"app-sentinel-{label}"),
                "--web-root", str(hermetic.tmp / f"web-sentinel-{label}"),
                "--target-mode", "dev",
            ],
            hermetic,
        )
        assert rres.returncode != 0
        assert secret_sentinel not in rres.stderr, label
        assert secret_sentinel not in rres.stdout, label


def test_staged_validation_injected_filesystem_error_handled_cleanly(hermetic, tmp_path: Path):
    """Injected filesystem errors during staged validation emit bounded static diagnostics without traceback."""
    repo = make_repo(tmp_path / "repo")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web", source_sha, generator_sha=source_sha[:12])

    gen_dir = repo / "data" / "generations"
    gen_id = "gen-2026-09-03-001"
    gen_target = gen_dir / gen_id
    gen_target.mkdir(parents=True)
    _write(gen_target / "manifest.json", '{"schema": "portfolio-lab-generation/v1"}\n')
    (gen_dir / "current").symlink_to(gen_id)

    _write(hermetic.units / f"{TASKER}.service", UNIT_TEXT)
    _write(repo / "data/tasker_status.json", json.dumps({"state": "idle", "version": 1}) + "\n")

    # 1. Create side injected failure in validate_staged_generation_trees
    create_fail_script = hermetic.tmp / "run_create_fs_fail.py"
    create_archive = hermetic.tmp / f"fs-fail{ARCHIVE_SUFFIX}"
    create_fail_script.write_text(f"""
import sys, os
sys.path.insert(0, "{PROJECT_ROOT}")
import scripts.portfolio_lab_recovery as plr

orig_validate = plr.validate_staged_generation_trees
def failing_validate(*args, **kwargs):
    raise PermissionError("injected permission error during create staging validation")

plr.validate_staged_generation_trees = failing_validate
sys.argv = [
    "portfolio_lab_recovery.py",
    "create",
    "--app-dir", "{repo}",
    "--web-root", "{web}",
    "--tasker-service", "{TASKER}",
    "--archive", "{create_archive}",
    "--storage-encryption-attested",
    "--materialize-generations-current",
]
raise SystemExit(plr.main())
""", encoding="utf-8")

    res = subprocess.run([sys.executable, str(create_fail_script)], env={**os.environ, **hermetic.env, "PATH": f"{hermetic.bin}:{os.environ.get('PATH', '')}"}, capture_output=True, text=True)
    assert res.returncode != 0
    assert "Traceback" not in res.stderr
    assert "cannot validate staged generation trees" in res.stderr.lower()
    assert str(tmp_path) not in res.stderr
    assert not create_archive.exists()

    # 2. Restore side injected failure in staged target rglob / validation
    target_archive, c_res = standard_create(hermetic, repo, web, archive_name="restore-fs-fail" + ARCHIVE_SUFFIX, extra=["--materialize-generations-current"])
    assert c_res.returncode == 0, c_res.stderr

    restore_fail_script = hermetic.tmp / "run_restore_fs_fail.py"
    restore_fail_script.write_text(f"""
import sys, os
sys.path.insert(0, "{PROJECT_ROOT}")
import scripts.portfolio_lab_recovery as plr

orig_rglob = plr.Path.rglob
def failing_rglob(self, pattern, *args, **kwargs):
    if "generations" in str(self):
        raise PermissionError("injected permission error during restore staging validation")
    return orig_rglob(self, pattern, *args, **kwargs)

plr.Path.rglob = failing_rglob
sys.argv = [
    "portfolio_lab_recovery.py",
    "restore",
    "--archive", "{target_archive}",
    "--app-dir", "{hermetic.tmp / 'app-fs-fail'}",
    "--web-root", "{hermetic.tmp / 'web-fs-fail'}",
    "--target-mode", "dev",
]
raise SystemExit(plr.main())
""", encoding="utf-8")

    r_res = subprocess.run([sys.executable, str(restore_fail_script)], env={**os.environ, **hermetic.env, "PATH": f"{hermetic.bin}:{os.environ.get('PATH', '')}"}, capture_output=True, text=True)
    assert r_res.returncode != 0
    assert "Traceback" not in r_res.stderr
    assert "failed to reconstruct staged generations/current symlink" in r_res.stderr.lower()
    assert str(tmp_path) not in r_res.stderr
    assert not (hermetic.tmp / "app-fs-fail").exists()


def test_create_verify_restore_nested_ordinary_directories(hermetic, tmp_path: Path):
    """Target containing at least two nested directory levels and files at multiple levels
    creates a verified archive without directory members and restores with relative symlink."""
    repo = make_repo(tmp_path / "repo-nested")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web-nested", source_sha, generator_sha=source_sha[:12])

    gen_dir = repo / "data" / "generations"
    gen_id = "gen-nested-001"
    gen_target = gen_dir / gen_id
    gen_target.mkdir(parents=True)

    # File at root level
    _write(gen_target / "manifest.json", '{"schema": "portfolio-lab-generation/v1"}\n')
    # Level 1 nested dir with file
    level1 = gen_target / "nested1"
    level1.mkdir()
    _write(level1 / "file1.json", '{"level": 1}\n')
    # Level 2 nested dir with file
    level2 = level1 / "nested2"
    level2.mkdir()
    _write(level2 / "file2.json", '{"level": 2}\n')

    (gen_dir / "current").symlink_to(gen_id)

    archive = hermetic.tmp / "backups" / ("nested-archive" + ARCHIVE_SUFFIX)
    archive.parent.mkdir(parents=True, exist_ok=True)
    _write(hermetic.units / f"{TASKER}.service", UNIT_TEXT)
    _write(repo / "data/tasker_status.json", json.dumps({"state": "idle", "version": 1}) + "\n")

    res = run_recovery(
        [
            "create",
            "--app-dir", str(repo),
            "--web-root", str(web),
            "--tasker-service", TASKER,
            "--archive", str(archive),
            "--storage-encryption-attested",
            "--materialize-generations-current",
        ],
        hermetic,
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"

    # Verify no directory tar members exist in the archive
    with tarfile.open(str(archive), "r:") as tf:
        for m in tf.getmembers():
            assert not m.isdir(), f"directory member found in tar: {m.name}"
            assert m.isfile()
        names = tf.getnames()
        assert f"runtime/data/generations/{gen_id}/manifest.json" in names
        assert f"runtime/data/generations/{gen_id}/nested1/file1.json" in names
        assert f"runtime/data/generations/{gen_id}/nested1/nested2/file2.json" in names
        assert "runtime/data/generations/current/manifest.json" in names
        assert "runtime/data/generations/current/nested1/file1.json" in names
        assert "runtime/data/generations/current/nested1/nested2/file2.json" in names

    # Verify command passes
    vres = run_recovery(["verify", "--archive", str(archive)], hermetic)
    assert vres.returncode == 0, vres.stderr

    # Dev restore: prove reconstruction
    app_dir = hermetic.tmp / "app-nested-restore"
    web_root = hermetic.tmp / "web-nested-restore"
    rres = run_recovery(
        [
            "restore",
            "--archive", str(archive),
            "--app-dir", str(app_dir),
            "--web-root", str(web_root),
            "--target-mode", "dev",
        ],
        hermetic,
    )
    assert rres.returncode == 0, f"stdout={rres.stdout}\nstderr={rres.stderr}"
    restored_link = app_dir / "data" / "generations" / "current"
    assert restored_link.is_symlink()
    assert os.readlink(restored_link) == gen_id
    assert (restored_link / "nested1" / "nested2" / "file2.json").is_file()


def test_is_safe_original_link_rejects_lone_surrogate():
    """is_safe_original_link must reject surrogate code points that cannot be UTF-8 encoded."""
    mod = _load_recovery_module()
    # Lone surrogate
    surrogate_target = "gen-\ud800"
    assert mod.is_safe_original_link(surrogate_target) is False
    # Valid unicode should pass
    valid_unicode_target = "gen-2026-测试-001"
    assert mod.is_safe_original_link(valid_unicode_target) is True


def test_create_pre_stop_rejection_surrogate_link_target(hermetic, tmp_path: Path):
    """A symlink target containing a surrogate code point is rejected before stop without traceback or leak."""
    repo = make_repo(tmp_path / "repo-surrogate")
    commit_all(repo)
    web = make_web_root(tmp_path / "web-surrogate", "x" * 40)
    gen_dir = repo / "data" / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)

    # We can create a symlink with non-UTF-8 bytes or surrogate on filesystem
    # or patch os.readlink for data/generations/current
    (gen_dir / "current").symlink_to("gen-valid")

    surr_archive = hermetic.tmp / f"surrogate{ARCHIVE_SUFFIX}"
    surrogate_script = hermetic.tmp / "run_surrogate_create.py"
    surrogate_script.write_text(f"""
import sys, os
sys.path.insert(0, "{PROJECT_ROOT}")
import scripts.portfolio_lab_recovery as plr

orig_readlink = plr.os.readlink
def surrogate_readlink(path, *args, **kwargs):
    if "current" in str(path):
        return "gen-\\ud800-bad"
    return orig_readlink(path, *args, **kwargs)

plr.os.readlink = surrogate_readlink
sys.argv = [
    "portfolio_lab_recovery.py",
    "create",
    "--app-dir", "{repo}",
    "--web-root", "{web}",
    "--tasker-service", "{TASKER}",
    "--archive", "{surr_archive}",
    "--storage-encryption-attested",
    "--materialize-generations-current",
]
raise SystemExit(plr.main())
""", encoding="utf-8")

    res = subprocess.run([sys.executable, str(surrogate_script)], env={**os.environ, **hermetic.env, "PATH": f"{hermetic.bin}:{os.environ.get('PATH', '')}"}, capture_output=True, text=True)
    assert res.returncode != 0
    assert "Traceback" not in res.stderr
    assert "invalid generations/current symlink target text" in res.stderr.lower()
    assert "ud800" not in res.stderr


def test_inspect_generation_target_rejects_surrogate_filename(hermetic, tmp_path: Path):
    """Non-UTF-8 encodable surrogate filenames inside generation target are rejected before stop."""
    repo = make_repo(tmp_path / "repo-surr-file")
    commit_all(repo)
    web = make_web_root(tmp_path / "web-surr-file", "x" * 40)
    gen_dir = repo / "data" / "generations"
    gen_id = "gen-001"
    gen_target = gen_dir / gen_id
    gen_target.mkdir(parents=True)
    _write(gen_target / "manifest.json", '{"schema": "portfolio-lab-generation/v1"}\n')
    (gen_dir / "current").symlink_to(gen_id)

    surr_file_archive = hermetic.tmp / f"surr-file{ARCHIVE_SUFFIX}"
    surr_file_script = hermetic.tmp / "run_surr_file_create.py"
    surr_file_script.write_text(f"""
import sys, os
sys.path.insert(0, "{PROJECT_ROOT}")
import scripts.portfolio_lab_recovery as plr

orig_walk = plr.os.walk
def surr_walk(top, *args, **kwargs):
    for root, dirs, files in orig_walk(top, *args, **kwargs):
        if str(top) in root:
            yield root, dirs, files + ["bad-\\ud800.json"]
        else:
            yield root, dirs, files

plr.os.walk = surr_walk
sys.argv = [
    "portfolio_lab_recovery.py",
    "create",
    "--app-dir", "{repo}",
    "--web-root", "{web}",
    "--tasker-service", "{TASKER}",
    "--archive", "{surr_file_archive}",
    "--storage-encryption-attested",
    "--materialize-generations-current",
]
raise SystemExit(plr.main())
""", encoding="utf-8")

    res = subprocess.run([sys.executable, str(surr_file_script)], env={**os.environ, **hermetic.env, "PATH": f"{hermetic.bin}:{os.environ.get('PATH', '')}"}, capture_output=True, text=True)
    assert res.returncode != 0
    assert "Traceback" not in res.stderr
    assert "unsafe relative name in generation target" in res.stderr.lower() or "control character" in res.stderr.lower()
    assert "ud800" not in res.stderr


# ── Correction Round 4: Focused Tests ──────────────────────────────────────────


def test_parse_and_validate_generation_metadata_rejects_current_and_current_nested():
    """Defect 1: parse_and_validate_generation_metadata must reject original_link == 'current'
    and 'current/...' (first path component 'current') with bounded static secret-safe error."""
    mod = _load_recovery_module()
    for link in ["current", "current/nested", "current/sub/dir"]:
        payload = {
            "schema_version": mod.GENERATION_MATERIALIZATION_SCHEMA,
            "link_path": mod.GENERATION_LINK_PATH,
            "archive_path": mod.GENERATION_ARCHIVE_PATH,
            "original_link": link,
            "target_path": f"data/generations/{link}",
        }
        ok, orig_link, err = mod.parse_and_validate_generation_metadata(payload)
        assert ok is False
        assert orig_link is None
        assert err == "generation_materialization original_link cannot target current"


def test_verify_and_restore_reject_tampered_original_link_current(hermetic, tmp_path: Path):
    """Defect 1: Real-archive verify and restore tests for original_link='current' and 'current/nested',
    proving verify fails, restore fails before app/web mutation, no sentinel/path leak, and no traceback."""
    repo = make_repo(tmp_path / "repo-cur-tamper")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web-cur-tamper", source_sha, generator_sha=source_sha[:12])

    gen_dir = repo / "data" / "generations"
    gen_id = "gen-orig"
    gen_target = gen_dir / gen_id
    gen_target.mkdir(parents=True)
    _write(gen_target / "manifest.json", '{"schema": "portfolio-lab-generation/v1"}\n')
    (gen_dir / "current").symlink_to(gen_id)

    archive, res = standard_create(
        hermetic,
        repo,
        web,
        extra=["--materialize-generations-current"],
    )
    assert res.returncode == 0, res.stderr

    secret_sentinel = "SECRET_SENTINEL_CURRENT_LEAK_TEST_12345"

    def repack_tampered_current(link_value: str, name: str) -> Path:
        out_archive = hermetic.tmp / "backups" / (name + ARCHIVE_SUFFIX)
        extract_dir = hermetic.tmp / f"extract-{name}"
        extract_tar(archive, extract_dir)
        mpath = extract_dir / "recovery-manifest.json"
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        manifest["generation_materialization"]["original_link"] = link_value
        manifest["generation_materialization"]["target_path"] = f"data/generations/{link_value}"
        mpath.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        all_members = [
            p.relative_to(extract_dir).as_posix()
            for p in sorted(extract_dir.rglob("*"))
            if p.is_file()
        ]
        subprocess.run(
            ["tar", "-cf", str(out_archive), "-C", str(extract_dir), *all_members],
            check=True,
            env={**os.environ, "COPYFILE_DISABLE": "1"},
        )
        sidecar = Path(str(out_archive) + ".sha256")
        sidecar.write_text(f"{_sha256_bytes(out_archive.read_bytes())}  {out_archive.name}\n", encoding="utf-8")
        return out_archive

    for link_val, label in [
        ("current", "cur-exact"),
        (f"current/nested_{secret_sentinel}", "cur-nested"),
    ]:
        bad_archive = repack_tampered_current(link_val, f"tamper-{label}")

        # Verify must fail, no traceback, no sentinel leak
        vres = run_recovery(["verify", "--archive", str(bad_archive)], hermetic)
        assert vres.returncode != 0
        assert "Traceback" not in vres.stderr
        assert secret_sentinel not in vres.stderr
        assert secret_sentinel not in vres.stdout

        # Restore must fail before app/web mutation, no traceback, no sentinel leak
        app_dir = hermetic.tmp / f"app-{label}"
        web_root = hermetic.tmp / f"web-{label}"
        rres = run_recovery(
            [
                "restore",
                "--archive", str(bad_archive),
                "--app-dir", str(app_dir),
                "--web-root", str(web_root),
                "--target-mode", "dev",
            ],
            hermetic,
        )
        assert rres.returncode != 0
        assert "Traceback" not in rres.stderr
        assert secret_sentinel not in rres.stderr
        assert not app_dir.exists(), f"app_dir should not have been created for {label}"
        assert not web_root.exists(), f"web_root should not have been created for {label}"


def test_create_verify_dev_restore_nested_safe_original_link(hermetic, tmp_path: Path):
    """Defect 2: Support nested safe original_link like 'releases/gen-001'.
    Shipped create + verify + dev restore test proving:
    - exact manifest target_path: 'data/generations/releases/gen-001'
    - normal target/current member pairing
    - exact reconstructed raw link: 'releases/gen-001'
    - retained target files in staging and final app_dir."""
    repo = make_repo(tmp_path / "repo-nested-link")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web-nested-link", source_sha, generator_sha=source_sha[:12])

    gen_dir = repo / "data" / "generations"
    target_rel = "releases/gen-001"
    target_dir = gen_dir / target_rel
    target_dir.mkdir(parents=True)
    _write(target_dir / "manifest.json", '{"schema": "portfolio-lab-generation/v1"}\n')
    _write(target_dir / "data.csv", "a,b,c\n1,2,3\n")
    (gen_dir / "current").symlink_to(target_rel)

    archive, res = standard_create(
        hermetic,
        repo,
        web,
        extra=["--materialize-generations-current"],
    )
    assert res.returncode == 0, res.stderr

    # Verify manifest details
    extract_dir = hermetic.tmp / "extract-nested-link"
    extract_tar(archive, extract_dir)
    manifest = json.loads((extract_dir / "recovery-manifest.json").read_text(encoding="utf-8"))
    gen_mat = manifest.get("generation_materialization")
    assert gen_mat is not None
    assert gen_mat["original_link"] == target_rel
    assert gen_mat["target_path"] == f"data/generations/{target_rel}"

    # Verify pass
    vres = run_recovery(["verify", "--archive", str(archive)], hermetic)
    assert vres.returncode == 0, vres.stderr

    # Dev restore pass
    app_dir = hermetic.tmp / "app-nested-link"
    web_root = hermetic.tmp / "web-nested-link"
    rres = run_recovery(
        [
            "restore",
            "--archive", str(archive),
            "--app-dir", str(app_dir),
            "--web-root", str(web_root),
            "--target-mode", "dev",
        ],
        hermetic,
    )
    assert rres.returncode == 0, rres.stderr

    # Assert reconstructed symlink and target files
    restored_current = app_dir / "data" / "generations" / "current"
    assert restored_current.is_symlink()
    assert os.readlink(restored_current) == target_rel
    assert restored_current.resolve() == (app_dir / "data" / "generations" / target_rel).resolve()
    assert (app_dir / "data" / "generations" / target_rel / "manifest.json").read_text(encoding="utf-8") == '{"schema": "portfolio-lab-generation/v1"}\n'
    assert (app_dir / "data" / "generations" / target_rel / "data.csv").read_text(encoding="utf-8") == "a,b,c\n1,2,3\n"
    assert (restored_current / "manifest.json").read_text(encoding="utf-8") == '{"schema": "portfolio-lab-generation/v1"}\n'


def test_verify_rejects_regular_member_at_target_or_current_directory_path(hermetic, tmp_path: Path):
    """Defect 3: Verify contract says no regular member may exist directly at either target directory path
    (runtime/data/generations/<target>) or GENERATION_ARCHIVE_PATH (runtime/data/generations/current).
    Explicit checks in generation verification reject crafted/tampered real-tar with these regular members."""
    repo = make_repo(tmp_path / "repo-reg-dir-member")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web-reg-dir-member", source_sha, generator_sha=source_sha[:12])

    gen_dir = repo / "data" / "generations"
    gen_id = "gen-001"
    gen_target = gen_dir / gen_id
    gen_target.mkdir(parents=True)
    _write(gen_target / "manifest.json", '{"schema": "portfolio-lab-generation/v1"}\n')
    (gen_dir / "current").symlink_to(gen_id)

    archive, res = standard_create(
        hermetic,
        repo,
        web,
        extra=["--materialize-generations-current"],
    )
    assert res.returncode == 0, res.stderr

    # Tamper tarfile directly to add regular member at target directory path or current path
    def repack_with_regular_dir_member(target_member_name: str, remove_children_prefix: str, name: str) -> Path:
        import shutil
        out_archive = hermetic.tmp / "backups" / (name + ARCHIVE_SUFFIX)
        extract_dir = hermetic.tmp / f"extract-{name}"
        extract_tar(archive, extract_dir)
        mpath = extract_dir / "recovery-manifest.json"
        manifest = json.loads(mpath.read_text(encoding="utf-8"))

        # Remove children under remove_children_prefix from manifest and disk so topology check passes
        manifest["members"] = [
            entry for entry in manifest["members"]
            if not entry["path"].startswith(remove_children_prefix)
        ]
        target_rm = extract_dir / remove_children_prefix
        if target_rm.is_dir():
            shutil.rmtree(target_rm)

        manifest["members"].append({
            "path": target_member_name,
            "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "bytes": 0,
            "mode": 0o600,
        })
        manifest["members"] = sorted(manifest["members"], key=lambda e: e["path"])
        mpath.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        with tarfile.open(str(out_archive), "w:") as tf:
            for p in sorted(extract_dir.rglob("*")):
                if p.is_file():
                    rel = p.relative_to(extract_dir).as_posix()
                    tf.add(str(p), arcname=rel)
            ti = tarfile.TarInfo(name=target_member_name)
            ti.type = tarfile.REGTYPE
            ti.mode = 0o600
            ti.size = 0
            import io
            tf.addfile(ti, io.BytesIO(b""))

        sidecar = Path(str(out_archive) + ".sha256")
        sidecar.write_text(f"{_sha256_bytes(out_archive.read_bytes())}  {out_archive.name}\n", encoding="utf-8")
        return out_archive

    # Test exact target directory member path
    bad_tar_target = repack_with_regular_dir_member(
        f"runtime/data/generations/{gen_id}",
        f"runtime/data/generations/{gen_id}/",
        "reg-target",
    )
    vres = run_recovery(["verify", "--archive", str(bad_tar_target)], hermetic)
    assert vres.returncode != 0
    assert "Traceback" not in vres.stderr
    assert "regular member exists at generation target directory" in vres.stderr.lower() or "regular member exists at generation target directory" in vres.stdout.lower()

    # Test exact GENERATION_ARCHIVE_PATH
    bad_tar_current = repack_with_regular_dir_member(
        "runtime/data/generations/current",
        "runtime/data/generations/current/",
        "reg-current",
    )
    vres = run_recovery(["verify", "--archive", str(bad_tar_current)], hermetic)
    assert vres.returncode != 0
    assert "Traceback" not in vres.stderr
    assert "regular member exists at generation current archive path" in vres.stderr.lower() or "regular member exists at generation current archive path" in vres.stdout.lower()


def test_verify_rejects_member_pair_bytes_mismatch(hermetic, tmp_path: Path):
    """Defect 4: Member-pair bytes mismatch tamper case."""
    repo = make_repo(tmp_path / "repo-bytes-mismatch")
    source_sha = commit_all(repo)
    web = make_web_root(tmp_path / "web-bytes-mismatch", source_sha, generator_sha=source_sha[:12])

    gen_dir = repo / "data" / "generations"
    gen_id = "gen-001"
    gen_target = gen_dir / gen_id
    gen_target.mkdir(parents=True)
    _write(gen_target / "manifest.json", '{"schema": "portfolio-lab-generation/v1"}\n')
    (gen_dir / "current").symlink_to(gen_id)

    archive, res = standard_create(
        hermetic,
        repo,
        web,
        extra=["--materialize-generations-current"],
    )
    assert res.returncode == 0, res.stderr

    out_archive = hermetic.tmp / "backups" / ("tamper-bytes-mismatch" + ARCHIVE_SUFFIX)
    extract_dir = hermetic.tmp / "extract-bytes-mismatch"
    extract_tar(archive, extract_dir)
    mpath = extract_dir / "recovery-manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8"))

    # Modify bytes recorded for current copy in manifest (or change file content and keep sha same, or change recorded bytes)
    for entry in manifest["members"]:
        if entry["path"] == "runtime/data/generations/current/manifest.json":
            entry["bytes"] = entry["bytes"] + 1

    mpath.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    all_members = [
        p.relative_to(extract_dir).as_posix()
        for p in sorted(extract_dir.rglob("*"))
        if p.is_file()
    ]
    subprocess.run(
        ["tar", "-cf", str(out_archive), "-C", str(extract_dir), *all_members],
        check=True,
        env={**os.environ, "COPYFILE_DISABLE": "1"},
    )
    sidecar = Path(str(out_archive) + ".sha256")
    sidecar.write_text(f"{_sha256_bytes(out_archive.read_bytes())}  {out_archive.name}\n", encoding="utf-8")

    vres = run_recovery(["verify", "--archive", str(out_archive)], hermetic)
    assert vres.returncode != 0
    assert "Traceback" not in vres.stderr


def test_activate_prod_rejects_materialize_generations_current_flag():
    """Defect 4: Prove --materialize-generations-current is rejected by activate-prod as unknown."""
    mod = _load_recovery_module()
    parser = mod.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "activate-prod",
            "--app-dir", "/app",
            "--web-root", "/www",
            "--tasker-service", "tasker",
            "--materialize-generations-current",
        ])


def test_control_characters_ascii_del_and_c1_rejected_in_link_and_entries(hermetic, tmp_path: Path):
    """Defect 4: Treat ASCII DEL (0x7f) and C1 controls U+0080-U+009F as control characters
    in original link and generation relative paths, with focused tests."""
    mod = _load_recovery_module()

    # ASCII DEL
    assert mod.is_safe_original_link("gen-\x7f-bad") is False
    # C1 controls: 0x80 to 0x9f
    for code in [0x80, 0x85, 0x90, 0x9F]:
        assert mod.is_safe_original_link(f"gen-{chr(code)}-bad") is False

    # Non-control character at 0xA0 (NBSP) or standard unicode should pass
    assert mod.is_safe_original_link("gen-\u00a0-ok") is True
    assert mod.is_safe_original_link("gen-ok-001") is True

    # Pre-stop test for DEL in symlink target
    repo = make_repo(tmp_path / "repo-del-link")
    commit_all(repo)
    web = make_web_root(tmp_path / "web-del-link", "x" * 40)
    gen_dir = repo / "data" / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)
    (gen_dir / "current").symlink_to("gen-\x7f-target")

    archive, res = standard_create(
        hermetic,
        repo,
        web,
        extra=["--materialize-generations-current"],
    )
    assert res.returncode != 0
    assert "Traceback" not in res.stderr
    assert "invalid generations/current symlink target text" in res.stderr.lower()

    # Pre-stop test for C1 control (e.g. \x85) in symlink target
    (gen_dir / "current").unlink()
    (gen_dir / "current").symlink_to("gen-\x85-target")
    archive, res = standard_create(
        hermetic,
        repo,
        web,
        extra=["--materialize-generations-current"],
    )
    assert res.returncode != 0
    assert "Traceback" not in res.stderr
    assert "invalid generations/current symlink target text" in res.stderr.lower()
