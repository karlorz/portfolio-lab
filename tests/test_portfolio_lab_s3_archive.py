"""Tests for scripts/portfolio_lab_s3_archive.py (Track A SeaweedFS S3 publisher)."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import scripts.portfolio_lab_recovery as plr

# Import the module under test (will fail until implemented)
try:
    import scripts.portfolio_lab_s3_archive as s3_archive
except ImportError:
    s3_archive = None  # type: ignore[assignment]


class FakeS3Handler(BaseHTTPRequestHandler):
    """In-memory stub for Path-Style S3 API."""

    def log_message(self, _format: str, *_args: object) -> None:
        return

    @property
    def storage(self) -> dict[str, dict[str, bytes]]:
        # Map bucket -> key -> data
        return self.server.storage  # type: ignore[attr-defined]

    def _parse_bucket_and_key(self) -> tuple[str, str]:
        # Path-style: /<bucket>/<key...>
        path = self.path.split("?", 1)[0]
        trimmed = path.lstrip("/")
        if not trimmed:
            return "", ""
        parts = trimmed.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket, key

    def do_HEAD(self) -> None:  # noqa: N802
        bucket, key = self._parse_bucket_and_key()
        if not bucket or not key:
            self.send_response(400)
            self.end_headers()
            return
        bucket_data = self.storage.get(bucket)
        if bucket_data is None or key not in bucket_data:
            self.send_response(404)
            self.end_headers()
            return
        data = bucket_data[key]
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("ETag", f'"{hashlib.md5(data).hexdigest()}"')
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        bucket, key = self._parse_bucket_and_key()
        if not bucket:
            self.send_response(400)
            self.end_headers()
            return
        bucket_data = self.storage.get(bucket)
        if bucket_data is None:
            self.send_response(404)
            self.end_headers()
            return
        if not key:
            # List bucket objects (simple query prefix/list)
            # Check for ?prefix=...
            query = ""
            if "?" in self.path:
                query = self.path.split("?", 1)[1]
            prefix = ""
            for param in query.split("&"):
                if param.startswith("prefix="):
                    prefix = param.split("=", 1)[1]
            contents_xml = []
            for k, val in sorted(bucket_data.items()):
                if k.startswith(prefix):
                    contents_xml.append(
                        f"<Contents><Key>{k}</Key><Size>{len(val)}</Size>"
                        f"<LastModified>2026-09-04T00:00:00.000Z</LastModified></Contents>"
                    )
            xml = (
                f'<?xml version="1.0" encoding="UTF-8"?>'
                f'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                f"<Name>{bucket}</Name><Prefix>{prefix}</Prefix>"
                f"{''.join(contents_xml)}"
                f"</ListBucketResult>"
            )
            data = xml.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/xml")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if key not in bucket_data:
            self.send_response(404)
            self.end_headers()
            return
        data = bucket_data[key]
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_PUT(self) -> None:  # noqa: N802
        bucket, key = self._parse_bucket_and_key()
        if not bucket:
            self.send_response(400)
            self.end_headers()
            return
        if not key:
            # Create bucket
            if bucket not in self.storage:
                self.storage[bucket] = {}
            self.send_response(200)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else b""
        if bucket not in self.storage:
            self.storage[bucket] = {}
        self.storage[bucket][key] = body
        self.send_response(200)
        self.end_headers()

    def do_DELETE(self) -> None:  # noqa: N802
        bucket, key = self._parse_bucket_and_key()
        if not bucket or not key:
            self.send_response(400)
            self.end_headers()
            return
        if bucket in self.storage and key in self.storage[bucket]:
            del self.storage[bucket][key]
        self.send_response(204)
        self.end_headers()


@pytest.fixture
def fake_s3_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeS3Handler)
    server.storage = {}  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    endpoint = f"http://{host}:{port}"
    yield server, endpoint
    server.shutdown()
    server.server_close()


def make_valid_archive(tmp_path: Path, git_bundle: bool = True, secret_in_config: bool = False) -> tuple[Path, Path]:
    """Build a minimal tar archive with valid .sha256 sidecar using stdlib tarfile."""
    archive_path = tmp_path / "valid.portfolio-lab-recovery.tar"
    sidecar_path = tmp_path / "valid.portfolio-lab-recovery.tar.sha256"

    config_content = "VAR1=val1\nVAR2=val2\n"
    if secret_in_config:
        config_content += "API_KEY=sk-supersecret12345\n"

    # Write files to a staging directory to add to tar
    staging = tmp_path / "archive_staging"
    staging.mkdir(parents=True, exist_ok=True)

    def write_f(rel: str, data: bytes | str) -> bytes:
        p = staging / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        raw = data.encode("utf-8") if isinstance(data, str) else data
        p.write_bytes(raw)
        return raw

    dummy_sha = "a" * 40
    bundle_bytes = b"fake-git-bundle-content" if git_bundle else b"not-a-bundle"
    # Create a real mini sqlite db for runtime/data/market.db
    import sqlite3
    db_file = tmp_path / "temp_market.db"
    con = sqlite3.connect(str(db_file))
    con.execute("CREATE TABLE t (x INT);")
    con.commit()
    con.close()
    real_db_bytes = db_file.read_bytes()

    files = {
        plr.BUNDLE_MEMBER: bundle_bytes,
        plr.REVISION_MEMBER: json.dumps({
            "schema_version": plr.SCHEMA_VERSION,
            "source_sha": dummy_sha,
            "branch": "main",
            "describe": "v1.0",
            "app_dir": "/opt/portfolio-lab",
            "web_root": "/var/www/portfolio-lab",
        }).encode("utf-8"),
        plr.CONFIG_MEMBER: config_content.encode("utf-8"),
        plr.UNIT_MEMBER: b"[Service]\nExecStart=/bin/true\n",
        plr.STATUS_MEMBER: b'{"status": "ok"}',
        plr.CADDY_MEMBER: b"# caddy block\n",
        plr.CREATED_MEMBER: json.dumps({"schema_version": plr.SCHEMA_VERSION}).encode("utf-8"),
        plr.TOOLS_MEMBER: b"# recovery tool\n",
        "runtime/data/market.db": real_db_bytes,
        "static/web/index.html": b"<!DOCTYPE html><html><body>ok</body></html>",
    }

    for rel, content in files.items():
        write_f(rel, content)

    # Compute members list for manifest
    members_list = []
    for rel, content in files.items():
        members_list.append({
            "path": rel,
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "mode": 0o600,
        })

    # Add manifest member itself
    dummy_sha = "a" * 40
    manifest_bytes = json.dumps({
        "schema_version": plr.SCHEMA_VERSION,
        "source": {
            "sha": dummy_sha,
            "tasker_service": "portfolio-lab-tasker",
            "app_dir": "/opt/portfolio-lab",
            "web_root": "/var/www/portfolio-lab",
        },
        "members": members_list,
    }).encode("utf-8")
    write_f(plr.MANIFEST_MEMBER, manifest_bytes)

    with tarfile.open(str(archive_path), "w:") as tf:
        for rel in sorted([plr.MANIFEST_MEMBER, *files.keys()]):
            ti = tf.gettarinfo(str(staging / rel), arcname=rel)
            ti.mode = 0o600
            with open(staging / rel, "rb") as f:
                tf.addfile(ti, f)

    # Sidecar
    archive_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    sidecar_path.write_text(f"{archive_hash}  {archive_path.name}\n", encoding="utf-8")
    return archive_path, sidecar_path


def make_credentials_file(tmp_path: Path, mode: int = 0o600, key: str = "test-key", secret: str = "test-secret") -> Path:
    creds_file = tmp_path / "s3_credentials.env"
    creds_file.write_text(f"AWS_ACCESS_KEY_ID={key}\nAWS_SECRET_ACCESS_KEY={secret}\n", encoding="utf-8")
    creds_file.chmod(mode)
    return creds_file


def test_import_module():
    """Verify s3 archive publisher module can be imported."""
    assert s3_archive is not None, "scripts.portfolio_lab_s3_archive module must exist"


def test_defaults():
    """Verify default configurations."""
    assert s3_archive.DEFAULT_ENDPOINT == "http://10.10.1.12:8333"
    assert s3_archive.DEFAULT_BUCKET == "portfolio-lab-archives"


def test_format_object_key():
    """Object key format: daily/YYYY/MM/DD/portfolio-lab-data-<utc>.tar"""
    dt = datetime(2026, 9, 4, 15, 30, 45, tzinfo=timezone.utc)
    key = s3_archive.format_object_key(dt)
    assert key == "daily/2026/09/04/portfolio-lab-data-20260904_153045Z.tar"
    assert s3_archive.sidecar_key_for(key) == "daily/2026/09/04/portfolio-lab-data-20260904_153045Z.tar.sha256"


def test_load_credentials_from_env(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-secret")
    key, secret = s3_archive.load_credentials()
    assert key == "env-key"
    assert secret == "env-secret"


def test_load_credentials_from_file(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    creds_file = make_credentials_file(tmp_path, mode=0o600, key="file-key", secret="file-secret")
    key, secret = s3_archive.load_credentials(credentials_file=creds_file)
    assert key == "file-key"
    assert secret == "file-secret"


def test_load_credentials_rejects_insecure_file_permissions(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    creds_file = make_credentials_file(tmp_path, mode=0o644, key="file-key", secret="file-secret")
    with pytest.raises(SystemExit) as exc_info:
        s3_archive.load_credentials(credentials_file=creds_file)
    assert exc_info.value.code != 0


def test_load_credentials_never_leaks_secrets_in_exception(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    creds_file = make_credentials_file(tmp_path, mode=0o644, key="sensitive-id", secret="super-sensitive-secret")
    with pytest.raises(SystemExit):
        s3_archive.load_credentials(credentials_file=creds_file)
    captured = capsys.readouterr()
    assert "super-sensitive-secret" not in captured.out
    assert "super-sensitive-secret" not in captured.err


def test_publish_valid_archive(tmp_path: Path, fake_s3_server, monkeypatch):
    server, endpoint = fake_s3_server
    archive, sidecar = make_valid_archive(tmp_path)
    creds_file = make_credentials_file(tmp_path, mode=0o600)

    # Ensure fake recovery verification passes or is stubbed
    monkeypatch.setattr(plr, "verify_archive", lambda arch, sc: (True, {"ok": True, "archive_sha256": "dummy"}))

    fixed_dt = datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)
    result = s3_archive.publish_archive(
        archive_path=archive,
        sidecar_path=sidecar,
        endpoint=endpoint,
        bucket="portfolio-lab-archives",
        credentials_file=creds_file,
        now_dt=fixed_dt,
    )

    expected_key = "daily/2026/09/04/portfolio-lab-data-20260904_120000Z.tar"
    expected_sc_key = expected_key + ".sha256"

    assert result["ok"] is True
    assert result["object_key"] == expected_key
    assert result["sidecar_key"] == expected_sc_key

    # Check that both objects are stored in fake S3
    bucket_storage = server.storage["portfolio-lab-archives"]
    assert expected_key in bucket_storage
    assert expected_sc_key in bucket_storage
    assert bucket_storage[expected_key] == archive.read_bytes()
    assert bucket_storage[expected_sc_key] == sidecar.read_bytes()


def test_publish_refuses_overwrite_existing_key(tmp_path: Path, fake_s3_server, monkeypatch):
    server, endpoint = fake_s3_server
    archive, sidecar = make_valid_archive(tmp_path)
    creds_file = make_credentials_file(tmp_path, mode=0o600)

    monkeypatch.setattr(plr, "verify_archive", lambda arch, sc: (True, {"ok": True}))

    fixed_dt = datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)
    expected_key = "daily/2026/09/04/portfolio-lab-data-20260904_120000Z.tar"

    # Pre-populate the key in S3
    server.storage["portfolio-lab-archives"] = {expected_key: b"already-exists"}

    with pytest.raises(SystemExit) as exc:
        s3_archive.publish_archive(
            archive_path=archive,
            sidecar_path=sidecar,
            endpoint=endpoint,
            bucket="portfolio-lab-archives",
            credentials_file=creds_file,
            now_dt=fixed_dt,
        )
    assert exc.value.code != 0
    # Make sure old object was not overwritten
    assert server.storage["portfolio-lab-archives"][expected_key] == b"already-exists"


def test_publish_refuses_if_recovery_verify_fails(tmp_path: Path, fake_s3_server, monkeypatch):
    server, endpoint = fake_s3_server
    archive, sidecar = make_valid_archive(tmp_path)
    creds_file = make_credentials_file(tmp_path, mode=0o600)

    # Force recovery verify failure
    monkeypatch.setattr(plr, "verify_archive", lambda arch, sc: (False, {"error": "corrupted SQLite db"}))

    fixed_dt = datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)
    with pytest.raises(SystemExit) as exc:
        s3_archive.publish_archive(
            archive_path=archive,
            sidecar_path=sidecar,
            endpoint=endpoint,
            bucket="portfolio-lab-archives",
            credentials_file=creds_file,
            now_dt=fixed_dt,
        )
    assert exc.value.code != 0
    assert "portfolio-lab-archives" not in server.storage


def test_publish_refuses_if_contains_git_tree(tmp_path: Path, fake_s3_server, monkeypatch):
    """Daily data archive must refuse publishing if it contains a full Git tree (.git directory)."""
    server, endpoint = fake_s3_server
    archive_path = tmp_path / "has_git.portfolio-lab-recovery.tar"
    sidecar_path = tmp_path / "has_git.portfolio-lab-recovery.tar.sha256"

    # Create archive with a .git member
    with tarfile.open(str(archive_path), "w:") as tf:
        ti = tarfile.TarInfo(name=".git/HEAD")
        ti.size = 5
        ti.mode = 0o600
        tf.addfile(ti, io.BytesIO(b"ref: \n"))

    archive_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    sidecar_path.write_text(f"{archive_hash}  {archive_path.name}\n", encoding="utf-8")

    creds_file = make_credentials_file(tmp_path, mode=0o600)
    monkeypatch.setattr(plr, "verify_archive", lambda arch, sc: (True, {"ok": True}))

    with pytest.raises(SystemExit) as exc:
        s3_archive.publish_archive(
            archive_path=archive_path,
            sidecar_path=sidecar_path,
            endpoint=endpoint,
            bucket="portfolio-lab-archives",
            credentials_file=creds_file,
        )
    assert exc.value.code != 0
    assert "portfolio-lab-archives" not in server.storage


def test_publish_refuses_if_contains_secrets_in_config(tmp_path: Path, fake_s3_server, monkeypatch):
    """Archive must refuse publishing if config/lab-app.env or environment contains secrets."""
    server, endpoint = fake_s3_server
    archive, sidecar = make_valid_archive(tmp_path, secret_in_config=True)
    creds_file = make_credentials_file(tmp_path, mode=0o600)

    monkeypatch.setattr(plr, "verify_archive", lambda arch, sc: (True, {"ok": True}))

    with pytest.raises(SystemExit) as exc:
        s3_archive.publish_archive(
            archive_path=archive,
            sidecar_path=sidecar,
            endpoint=endpoint,
            bucket="portfolio-lab-archives",
            credentials_file=creds_file,
        )
    assert exc.value.code != 0
    assert "portfolio-lab-archives" not in server.storage


def test_retention_prune_never_deletes_newest_verified_object(fake_s3_server, tmp_path: Path):
    """Retention helper prunes older objects exceeding keep count but never deletes the newest verified."""
    server, endpoint = fake_s3_server
    creds_file = make_credentials_file(tmp_path, mode=0o600)

    # Populate bucket with 5 daily archives
    keys = [
        "daily/2026/09/01/portfolio-lab-data-20260901_000000Z.tar",
        "daily/2026/09/02/portfolio-lab-data-20260902_000000Z.tar",
        "daily/2026/09/03/portfolio-lab-data-20260903_000000Z.tar",
        "daily/2026/09/04/portfolio-lab-data-20260904_000000Z.tar",
        "daily/2026/09/05/portfolio-lab-data-20260905_000000Z.tar",
    ]
    bucket_storage = {}
    for k in keys:
        bucket_storage[k] = b"data"
        bucket_storage[k + ".sha256"] = b"hash  name\n"
    server.storage["portfolio-lab-archives"] = bucket_storage

    # Prune keeping 2 newest
    deleted = s3_archive.prune_archives(
        endpoint=endpoint,
        bucket="portfolio-lab-archives",
        credentials_file=creds_file,
        keep_count=2,
    )
    assert len(deleted) == 6  # 3 tar objects + 3 sha256 sidecars

    remaining = set(server.storage["portfolio-lab-archives"].keys())
    # Newest object and its sidecar must remain
    assert "daily/2026/09/05/portfolio-lab-data-20260905_000000Z.tar" in remaining
    assert "daily/2026/09/05/portfolio-lab-data-20260905_000000Z.tar.sha256" in remaining
    assert "daily/2026/09/04/portfolio-lab-data-20260904_000000Z.tar" in remaining
    assert "daily/2026/09/04/portfolio-lab-data-20260904_000000Z.tar.sha256" in remaining

    # Older objects must be deleted
    assert "daily/2026/09/01/portfolio-lab-data-20260901_000000Z.tar" not in remaining
    assert "daily/2026/09/01/portfolio-lab-data-20260901_000000Z.tar.sha256" not in remaining


def test_verify_remote_fetches_and_verifies(fake_s3_server, tmp_path: Path, monkeypatch):
    """verify-remote fetches remote archive and runs plr.verify_archive."""
    server, endpoint = fake_s3_server
    creds_file = make_credentials_file(tmp_path, mode=0o600)
    key = "daily/2026/09/04/portfolio-lab-data-20260904_120000Z.tar"
    sc_key = key + ".sha256"

    server.storage["portfolio-lab-archives"] = {
        key: b"fake-archive-content",
        sc_key: b"fake-hash  portfolio-lab-data-20260904_120000Z.tar\n",
    }

    monkeypatch.setattr(plr, "verify_archive", lambda arch, sc: (True, {"ok": True, "archive_sha256": "fake-hash"}))

    res = s3_archive.verify_remote_archive(
        object_key=key,
        endpoint=endpoint,
        bucket="portfolio-lab-archives",
        credentials_file=creds_file,
    )
    assert res["ok"] is True
    assert res["remote_key"] == key
    assert res["remote_bucket"] == "portfolio-lab-archives"


def test_cli_publish_invocation(fake_s3_server, tmp_path: Path):
    """Verify CLI publish invocation with argument flags."""
    server, endpoint = fake_s3_server
    archive, sidecar = make_valid_archive(tmp_path)

    script = Path(__file__).resolve().parents[1] / "scripts" / "portfolio_lab_s3_archive.py"
    env = dict(os.environ)
    env["AWS_ACCESS_KEY_ID"] = "cli-key"
    env["AWS_SECRET_ACCESS_KEY"] = "cli-secret"
    dummy_sha = "a" * 40
    # Create fake git command that succeeds for bundle verify and prints dummy sha for list-heads
    fake_git = tmp_path / "fake_git.sh"
    fake_git.write_text(f'#!/bin/sh\nif [ "$1" = "bundle" ] && [ "$2" = "list-heads" ]; then echo "{dummy_sha} refs/heads/main"; exit 0; fi\nexit 0\n', encoding="utf-8")
    fake_git.chmod(0o755)
    env["PLR_GIT"] = str(fake_git)

    res = subprocess.run(
        [
            sys.executable,
            str(script),
            "--endpoint",
            endpoint,
            "--bucket",
            "portfolio-lab-archives",
            "publish",
            "--archive",
            str(archive),
            "--sidecar",
            str(sidecar),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode == 0, f"stdout: {res.stdout}, stderr: {res.stderr}"
    data = json.loads(res.stdout)
    assert data["ok"] is True
    assert data["bucket"] == "portfolio-lab-archives"
    assert "daily/" in data["object_key"]


def test_cli_help_and_subcommands():
    """Verify CLI interface runs without crashing and has expected subcommands."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "portfolio_lab_s3_archive.py"
    res = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "publish" in res.stdout
    assert "prune" in res.stdout
    assert "verify-remote" in res.stdout


def test_rclone_large_archive_publish_uses_rclone(tmp_path: Path, fake_s3_server, monkeypatch):
    """Large archive (>= threshold) or transport='rclone' uses rclone copyto and not urllib PUT for tar."""
    server, endpoint = fake_s3_server
    archive, sidecar = make_valid_archive(tmp_path)
    creds_file = make_credentials_file(tmp_path, mode=0o600, key="key123", secret="secret456")
    monkeypatch.setattr(plr, "verify_archive", lambda arch, sc: (True, {"ok": True, "archive_sha256": "dummy"}))

    # Fake rclone binary
    log_file = tmp_path / "rclone_calls.log"
    fake_rclone = tmp_path / "fake_rclone.sh"
    fake_rclone.write_text(f"""#!/bin/bash
echo "$@" >> "{log_file}"
# Copy the file to local path or do whatever, or simulate copyto success
exit 0
""", encoding="utf-8")
    fake_rclone.chmod(0o755)

    monkeypatch.setenv("PLR_RCLONE", str(fake_rclone))

    # Set threshold very low so our small archive triggers rclone
    monkeypatch.setattr(s3_archive, "RCLONE_SIZE_THRESHOLD", 10)

    fixed_dt = datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)
    res = s3_archive.publish_archive(
        archive_path=archive,
        sidecar_path=sidecar,
        endpoint=endpoint,
        bucket="portfolio-lab-archives",
        credentials_file=creds_file,
        now_dt=fixed_dt,
    )
    assert res["ok"] is True
    assert res["transport"] == "rclone"

    # Verify rclone was invoked
    assert log_file.is_file()
    cmd_logged = log_file.read_text(encoding="utf-8")
    assert "copyto" in cmd_logged
    assert "--s3-provider Other" in cmd_logged
    assert "--s3-force-path-style" in cmd_logged
    assert "--s3-env-auth" in cmd_logged
    assert f"--s3-endpoint {endpoint}" in cmd_logged
    assert ":s3:portfolio-lab-archives/daily/2026/09/04/portfolio-lab-data-20260904_120000Z.tar" in cmd_logged

    # Verify secret is NOT in command argv
    assert "secret456" not in cmd_logged

    # Archive tar must not have been put via urllib directly into fake S3 server storage
    # (sidecar is still put via urllib or rclone, but tar must be handled by rclone)
    bucket_storage = server.storage.get("portfolio-lab-archives", {})
    expected_key = "daily/2026/09/04/portfolio-lab-data-20260904_120000Z.tar"
    assert expected_key not in bucket_storage, "Large archive must be published via rclone, not urllib PUT"


def test_rclone_argv_flags_and_env_credentials(tmp_path: Path, fake_s3_server, monkeypatch):
    """rclone invocation passes credentials via environment (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY) and never in argv."""
    server, endpoint = fake_s3_server
    archive, sidecar = make_valid_archive(tmp_path)
    creds_file = make_credentials_file(tmp_path, mode=0o600, key="my-access-key", secret="my-secret-key")
    monkeypatch.setattr(plr, "verify_archive", lambda arch, sc: (True, {"ok": True, "archive_sha256": "dummy"}))

    env_log = tmp_path / "rclone_env.log"
    fake_rclone = tmp_path / "fake_rclone.sh"
    fake_rclone.write_text(f"""#!/bin/bash
echo "KEY=$AWS_ACCESS_KEY_ID" >> "{env_log}"
echo "SECRET=$AWS_SECRET_ACCESS_KEY" >> "{env_log}"
echo "ARGV=$*" >> "{env_log}"
exit 0
""", encoding="utf-8")
    fake_rclone.chmod(0o755)
    monkeypatch.setenv("PLR_RCLONE", str(fake_rclone))

    res = s3_archive.publish_archive(
        archive_path=archive,
        sidecar_path=sidecar,
        endpoint=endpoint,
        bucket="portfolio-lab-archives",
        credentials_file=creds_file,
        transport="rclone",
    )
    assert res["ok"] is True
    assert res["transport"] == "rclone"

    env_contents = env_log.read_text(encoding="utf-8")
    assert "KEY=my-access-key" in env_contents
    assert "SECRET=my-secret-key" in env_contents
    assert "ARGV=" in env_contents
    argv_line = [line for line in env_contents.splitlines() if line.startswith("ARGV=")][0]
    assert "my-secret-key" not in argv_line


def test_rclone_missing_binary_fails_closed(tmp_path: Path, fake_s3_server, monkeypatch):
    """When transport='rclone' and rclone binary is missing, fail closed without urllib fallback."""
    server, endpoint = fake_s3_server
    archive, sidecar = make_valid_archive(tmp_path)
    creds_file = make_credentials_file(tmp_path, mode=0o600)
    monkeypatch.setattr(plr, "verify_archive", lambda arch, sc: (True, {"ok": True, "archive_sha256": "dummy"}))

    monkeypatch.setenv("PLR_RCLONE", "/nonexistent/path/to/rclone")

    with pytest.raises(SystemExit) as exc:
        s3_archive.publish_archive(
            archive_path=archive,
            sidecar_path=sidecar,
            endpoint=endpoint,
            bucket="portfolio-lab-archives",
            credentials_file=creds_file,
            transport="rclone",
        )
    assert exc.value.code != 0
    # Tar was not uploaded via urllib fallback
    bucket_storage = server.storage.get("portfolio-lab-archives", {})
    assert len(bucket_storage) == 0


def test_cli_publish_transport_flag(tmp_path: Path, fake_s3_server, monkeypatch):
    """CLI publish supports --transport {rclone,urllib,auto}."""
    server, endpoint = fake_s3_server
    archive, sidecar = make_valid_archive(tmp_path)

    script = Path(__file__).resolve().parents[1] / "scripts" / "portfolio_lab_s3_archive.py"
    env = dict(os.environ)
    env["AWS_ACCESS_KEY_ID"] = "cli-key"
    env["AWS_SECRET_ACCESS_KEY"] = "cli-secret"
    dummy_sha = "a" * 40
    fake_git = tmp_path / "fake_git.sh"
    fake_git.write_text(f'#!/bin/sh\nif [ "$1" = "bundle" ] && [ "$2" = "list-heads" ]; then echo "{dummy_sha} refs/heads/main"; exit 0; fi\nexit 0\n', encoding="utf-8")
    fake_git.chmod(0o755)
    env["PLR_GIT"] = str(fake_git)

    res = subprocess.run(
        [
            sys.executable,
            str(script),
            "--endpoint",
            endpoint,
            "--bucket",
            "portfolio-lab-archives",
            "publish",
            "--archive",
            str(archive),
            "--sidecar",
            str(sidecar),
            "--transport",
            "urllib",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode == 0, f"stdout: {res.stdout}, stderr: {res.stderr}"
    data = json.loads(res.stdout)
    assert data["ok"] is True
    assert data["transport"] == "urllib"


def test_wrapper_script_structure():
    """Daily wrapper script scripts/cron/portfolio-lab-s3-archive.sh exists and adheres to requirements."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "cron" / "portfolio-lab-s3-archive.sh"
    assert script_path.is_file(), f"{script_path} must exist"
    assert os.access(script_path, os.X_OK), f"{script_path} must be executable"

    content = script_path.read_text(encoding="utf-8")
    # Must source cron_guard.sh
    assert "cron_guard.sh" in content
    assert "pf-s3-archive" in content
    # Timeout 2400
    assert "2400" in content
    # Memory 3072 MB
    assert "3072" in content
    # STAMP generated inside with date -u +%Y%m%dT%H%M%SZ
    assert "date -u +%Y%m%dT%H%M%SZ" in content
    # Sidecar root default /var/backups/portfolio-lab-migration/sidecar
    assert "PLR_S3_ARCHIVE_SIDECAR" in content
    assert "/var/backups/portfolio-lab-migration/sidecar" in content
    # Archive dest
    assert "/var/backups/portfolio-lab-migration/portfolio-lab-" in content
    # Commands / flags required
    assert "--materialize-generations-current" in content
    assert "--storage-encryption-attested" in content
    assert "--service-controller systemd" in content
    assert "--tasker-service portfolio-lab-tasker" in content
    assert "verify" in content
    assert "publish" in content
    assert "--transport rclone" in content
    assert "rclone lsf" in content
    assert "prune" in content
    assert "--keep 7" in content
    assert "http://100.110.81.72:8333" in content
    assert "portfolio-lab-archives" in content
    assert 'HOME="${HOME:-/root}"' in content or "HOME:-/root" in content
    # Comment about not running as child of tasker
    assert "portfolio-lab-tasker" in content
    assert "cron_guard_end" in content
    # Must not contain set -x or echo credentials
    assert "set -x" not in content


def test_makefile_s3_archive_target():
    """Makefile defines .PHONY: s3-archive with timeout 2400, not in all."""
    makefile_path = Path(__file__).resolve().parents[1] / "Makefile"
    assert makefile_path.is_file()
    content = makefile_path.read_text(encoding="utf-8")

    assert ".PHONY: s3-archive" in content
    assert "s3-archive:" in content
    assert "timeout 2400" in content

    # Find the 'all:' target line and ensure s3-archive is not part of its prerequisites
    all_lines = [line for line in content.splitlines() if line.startswith("all:")]
    assert all_lines, "all: target must exist in Makefile"
    for line in all_lines:
        assert "s3-archive" not in line.split()


def test_systemd_units():
    """Service and timer systemd files exist and have required directives."""
    service_path = Path(__file__).resolve().parents[1] / "scripts" / "cron" / "portfolio-lab-s3-archive.service"
    timer_path = Path(__file__).resolve().parents[1] / "scripts" / "cron" / "portfolio-lab-s3-archive.timer"

    assert service_path.is_file(), f"{service_path} must exist"
    assert timer_path.is_file(), f"{timer_path} must exist"

    service_content = service_path.read_text(encoding="utf-8")
    assert "Type=oneshot" in service_content
    assert "TimeoutStartSec=2400" in service_content
    assert "EnvironmentFile=-/root/.config/portfolio-lab/s3-credentials.env" in service_content
    assert "ExecStart=/var/backups/portfolio-lab-migration/sidecar/scripts/cron/portfolio-lab-s3-archive.sh" in service_content
    assert "Conflicts=" not in service_content

    timer_content = timer_path.read_text(encoding="utf-8")
    assert "OnCalendar=*-*-* 04:17:00 UTC" in timer_content
    assert "Persistent=true" in timer_content


def test_crontab_and_cron_targets():
    """Commented line in crontab exists, CRON_TARGETS length remains 18."""
    crontab_path = Path(__file__).resolve().parents[1] / "crontab"
    content = crontab_path.read_text(encoding="utf-8")
    # Must have commented daily s3-archive line
    commented_lines = [line for line in content.splitlines() if line.strip().startswith("#") and "s3-archive" in line]
    assert len(commented_lines) >= 1, "Crontab must contain a commented s3-archive fallback line"

    from src.cron_compat import CRON_TARGETS
    assert len(CRON_TARGETS) == 18, f"CRON_TARGETS must remain length 18, found {len(CRON_TARGETS)}"
    assert "portfolio-lab-s3-archive" not in CRON_TARGETS
    assert "s3-archive" not in CRON_TARGETS
