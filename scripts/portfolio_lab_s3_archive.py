#!/usr/bin/env python3
"""Thin SeaweedFS S3 publisher for non-git Portfolio Lab material.

Track A of the Portfolio Lab sg01-to-cursor-box migration plan:
- Endpoint path-style http://10.10.1.12:8333 as default (overridable via flag or env).
- Bucket name default 'portfolio-lab-archives'.
- Object key format: daily/YYYY/MM/DD/portfolio-lab-data-<utc>.tar plus adjacent .sha256.
- Refuses overwrite of existing object key.
- Preflights local archive against recovery verify, git tree scan, and secret scan.
- Loads credentials from env or mode 0600 file; never logs/prints credentials.
- Retention helper never deletes the newest verified object.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile

import tempfile
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Reuse existing recovery verification and secret detection
try:
    import scripts.portfolio_lab_recovery as plr
except ImportError:
    # Allow running directly from scripts/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import scripts.portfolio_lab_recovery as plr

DEFAULT_ENDPOINT = "http://10.10.1.12:8333"
DEFAULT_BUCKET = "portfolio-lab-archives"
SCHEMA_VERSION = "portfolio-lab-s3-archive/v1"
RCLONE_SIZE_THRESHOLD = 8 * 1024 * 1024


def _rclone_binary() -> str:
    return os.environ.get("PLR_RCLONE", "rclone")



def die(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def format_object_key(dt: datetime | None = None) -> str:
    if dt is None:
        dt = datetime.now(timezone.utc)
    date_path = dt.strftime("%Y/%m/%d")
    timestamp = dt.strftime("%Y%m%d_%H%M%SZ")
    return f"daily/{date_path}/portfolio-lab-data-{timestamp}.tar"


def sidecar_key_for(object_key: str) -> str:
    return f"{object_key}.sha256"


def load_credentials(
    credentials_file: Path | str | None = None,
) -> tuple[str, str]:
    """Read S3 credentials from env or a mode-0600 file; never print secret values."""
    env_key = os.environ.get("AWS_ACCESS_KEY_ID")
    env_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if env_key and env_secret:
        return env_key.strip(), env_secret.strip()

    file_path: Path | None = None
    if credentials_file:
        file_path = Path(credentials_file).resolve()
    elif os.environ.get("PLR_S3_CREDENTIALS_FILE"):
        file_path = Path(os.environ["PLR_S3_CREDENTIALS_FILE"]).resolve()
    else:
        # Check standard user credentials file
        candidate = Path.home() / ".config" / "portfolio-lab" / "s3-credentials.env"
        if candidate.is_file():
            file_path = candidate

    if file_path is None:
        die("missing S3 credentials: set AWS_ACCESS_KEY_ID & AWS_SECRET_ACCESS_KEY or supply credentials file")

    if not file_path.is_file():
        die(f"credentials file does not exist: {file_path}")

    # Enforce strict 0600 / owner-only permissions
    mode = stat.S_IMODE(file_path.stat().st_mode)
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        die(
            f"credentials file {file_path} has permissions {oct(mode)}; "
            "must be 0600 (owner read/write only)"
        )

    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        die(f"cannot read credentials file: {exc}")

    key_val = ""
    secret_val = ""
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip("\"'")
        if k in ("AWS_ACCESS_KEY_ID", "S3_ACCESS_KEY", "ACCESS_KEY"):
            key_val = v
        elif k in ("AWS_SECRET_ACCESS_KEY", "S3_SECRET_KEY", "SECRET_KEY"):
            secret_val = v

    if not key_val or not secret_val:
        die(f"credentials file {file_path} missing access key or secret key")

    return key_val, secret_val


def sign_v4(
    method: str,
    url: str,
    headers: dict[str, str],
    payload: bytes,
    access_key: str,
    secret_key: str,
    region: str = "us-east-1",
    service: str = "s3",
) -> dict[str, str]:
    """Compute AWS Signature Version 4 for S3 path-style requests."""
    parsed = urllib.parse.urlparse(url)
    dt = datetime.now(timezone.utc)
    amz_date = dt.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = dt.strftime("%Y%m%d")

    out_headers = dict(headers)
    out_headers["Host"] = parsed.netloc
    out_headers["x-amz-date"] = amz_date
    payload_hash = hashlib.sha256(payload).hexdigest()
    out_headers["x-amz-content-sha256"] = payload_hash

    # Canonical headers
    sorted_header_keys = sorted(k.lower() for k in out_headers)
    canonical_headers = "".join(f"{k}:{out_headers[orig].strip()}\n" for k in sorted_header_keys for orig in out_headers if orig.lower() == k)
    signed_headers = ";".join(sorted_header_keys)

    # Canonical query string
    canonical_querystr = ""
    if parsed.query:
        query_parts = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        canonical_querystr = urllib.parse.urlencode(sorted(query_parts))

    # Path encoding
    canonical_uri = urllib.parse.quote(parsed.path, safe="/-_.~")

    canonical_request = f"{method}\n{canonical_uri}\n{canonical_querystr}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"

    algorithm = "AWS4-HMAC-SHA256"
    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = f"{algorithm}\n{amz_date}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"

    def sign(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    k_date = sign(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    k_region = sign(k_date, region)
    k_service = sign(k_region, service)
    k_signing = sign(k_service, "aws4_request")
    signature = hmac.new(k_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    authorization_header = f"{algorithm} Credential={access_key}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}"
    out_headers["Authorization"] = authorization_header
    return out_headers


class S3Client:
    """Minimal path-style S3 client using standard library urllib."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        region: str = "us-east-1",
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region

    def _request(
        self,
        method: str,
        path: str,
        data: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        if headers is None:
            headers = {}
        url = f"{self.endpoint}/{path.lstrip('/')}"
        signed_headers = sign_v4(
            method=method,
            url=url,
            headers=headers,
            payload=data,
            access_key=self.access_key,
            secret_key=self.secret_key,
            region=self.region,
        )

        req = urllib.request.Request(url, data=data if method in ("PUT", "POST") else None, headers=signed_headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_data = resp.read()
                resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                return resp.status, resp_headers, resp_data
        except urllib.error.HTTPError as exc:
            err_data = exc.read()
            err_headers = {k.lower(): v for k, v in exc.headers.items()}
            return exc.code, err_headers, err_data

    def head_object(self, bucket: str, key: str) -> bool:
        """Returns True if object exists (200), False if 404."""
        status, _, _ = self._request("HEAD", f"{bucket}/{key}")
        if status == 200:
            return True
        if status in (404, 403):
            return False
        return False

    def put_object(self, bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(data)),
        }
        status, _, resp = self._request("PUT", f"{bucket}/{key}", data=data, headers=headers)
        if status not in (200, 201, 204):
            die(f"S3 PUT {bucket}/{key} failed with HTTP {status}: {resp.decode('utf-8', errors='replace')[:200]}")

    def get_object(self, bucket: str, key: str) -> bytes:
        status, _, resp = self._request("GET", f"{bucket}/{key}")
        if status != 200:
            die(f"S3 GET {bucket}/{key} failed with HTTP {status}: {resp.decode('utf-8', errors='replace')[:200]}")
        return resp

    def delete_object(self, bucket: str, key: str) -> None:
        status, _, resp = self._request("DELETE", f"{bucket}/{key}")
        if status not in (200, 204):
            die(f"S3 DELETE {bucket}/{key} failed with HTTP {status}: {resp.decode('utf-8', errors='replace')[:200]}")

    def list_objects(self, bucket: str, prefix: str = "") -> list[dict[str, Any]]:
        path = bucket
        if prefix:
            path = f"{bucket}?prefix={urllib.parse.quote(prefix)}"
        status, _, resp = self._request("GET", path)
        if status != 200:
            return []
        try:
            root = ET.fromstring(resp)
        except ET.ParseError:
            return []

        # Strip XML namespaces for simple tag matching
        results = []
        for elem in root.iter():
            if elem.tag.endswith("Contents"):
                k_elem = elem.find("{*}Key")
                s_elem = elem.find("{*}Size")
                m_elem = elem.find("{*}LastModified")
                if k_elem is not None and k_elem.text:
                    results.append({
                        "key": k_elem.text,
                        "size": int(s_elem.text) if s_elem is not None and s_elem.text else 0,
                        "last_modified": m_elem.text if m_elem is not None else "",
                    })
        return results


def check_archive_contents(archive_path: Path) -> None:
    """Preflight inspect archive members for git trees, secrets, and safety."""
    try:
        with tarfile.open(str(archive_path), "r:") as tf:
            for info in tf:
                name = info.name.replace("\\", "/").strip("/")
                # Refuse git trees: member must not start with .git/ or have /.git/ component
                parts = name.split("/")
                if ".git" in parts:
                    die(f"archive contains Git tree member {info.name!r}; refusing to publish as daily non-git data object")

                # If member is a config or env file, check for secrets
                if parts[-1] in ("lab-app.env", ".env", "config.env") or name == plr.CONFIG_MEMBER:
                    f = tf.extractfile(info)
                    if f:
                        content = f.read().decode("utf-8", errors="replace")
                        plr.check_text_secrets(content, f"archive member {info.name}")
    except (tarfile.TarError, OSError, ValueError) as exc:
        die(f"invalid tar archive: {exc}")


def upload_via_rclone(
    local_path: Path,
    bucket: str,
    key: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
) -> None:
    """Upload large object to S3 using rclone copyto with env-auth and path style."""
    rclone_bin = _rclone_binary()
    # Check if executable exists or is resolved in PATH
    resolved = shutil.which(rclone_bin) if "/" not in rclone_bin else (rclone_bin if os.path.isfile(rclone_bin) and os.access(rclone_bin, os.X_OK) else None)
    if not resolved:
        die(f"rclone binary not found or not executable: {rclone_bin}")

    dest = f":s3:{bucket}/{key}"
    cmd = [
        resolved,
        "copyto",
        str(local_path),
        dest,
        "--s3-provider",
        "Other",
        "--s3-force-path-style",
        "--s3-env-auth",
        "--s3-endpoint",
        endpoint,
        "--s3-no-check-bucket",
        "--retries",
        "3",
    ]

    rclone_env = dict(os.environ)
    rclone_env["AWS_ACCESS_KEY_ID"] = access_key
    rclone_env["AWS_SECRET_ACCESS_KEY"] = secret_key

    try:
        proc = subprocess.run(
            cmd,
            env=rclone_env,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        die(f"failed to run rclone: {exc}")

    if proc.returncode != 0:
        err_msg = proc.stderr.strip() or proc.stdout.strip()
        die(f"rclone copyto failed (exit {proc.returncode}): {err_msg[:300]}")


def publish_archive(
    archive_path: Path,
    sidecar_path: Path | None = None,
    endpoint: str = DEFAULT_ENDPOINT,
    bucket: str = DEFAULT_BUCKET,
    credentials_file: Path | str | None = None,
    now_dt: datetime | None = None,
    transport: str | None = None,
) -> dict[str, Any]:
    """Publish verified archive to SeaweedFS S3 with immutable daily key."""
    if not archive_path.is_file():
        die(f"archive file not found: {archive_path}")

    if sidecar_path is None:
        sidecar_path = Path(str(archive_path) + ".sha256")
    if not sidecar_path.is_file():
        die(f"sidecar file not found: {sidecar_path}")

    # Validate transport selection
    archive_size = archive_path.stat().st_size
    if transport is None or transport == "auto":
        resolved_transport = "rclone" if archive_size >= RCLONE_SIZE_THRESHOLD else "urllib"
    elif transport in ("rclone", "urllib"):
        resolved_transport = transport
    else:
        die(f"invalid transport: {transport!r} (must be 'rclone', 'urllib', or 'auto')")

    # 1. Verify archive integrity using recovery tooling
    ok, verify_report = plr.verify_archive(archive_path, sidecar_path)
    if not ok:
        die(f"recovery archive verification failed: {verify_report.get('error', 'unknown verify failure')}")

    # 2. Check archive contents for git tree and secrets
    check_archive_contents(archive_path)

    # 3. Load credentials and create S3 client
    access_key, secret_key = load_credentials(credentials_file=credentials_file)
    s3 = S3Client(endpoint=endpoint, access_key=access_key, secret_key=secret_key)

    # 4. Determine object key
    object_key = format_object_key(now_dt)
    sidecar_key = sidecar_key_for(object_key)

    # 5. Check if key already exists (immutable policy: refuse overwrite)
    if s3.head_object(bucket, object_key):
        die(f"object key already exists in S3: {bucket}/{object_key}; refusing to overwrite")
    if s3.head_object(bucket, sidecar_key):
        die(f"sidecar object key already exists in S3: {bucket}/{sidecar_key}; refusing to overwrite")

    # 6. Upload archive and sidecar
    if resolved_transport == "rclone":
        upload_via_rclone(
            local_path=archive_path,
            bucket=bucket,
            key=object_key,
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
        )
    else:
        archive_bytes = archive_path.read_bytes()
        s3.put_object(bucket, object_key, archive_bytes, content_type="application/x-tar")

    sidecar_bytes = sidecar_path.read_bytes()
    s3.put_object(bucket, sidecar_key, sidecar_bytes, content_type="text/plain")

    report = {
        "ok": True,
        "schema_version": SCHEMA_VERSION,
        "endpoint": endpoint,
        "bucket": bucket,
        "object_key": object_key,
        "sidecar_key": sidecar_key,
        "archive_sha256": verify_report.get("archive_sha256") or hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "published_at": (now_dt or datetime.now(timezone.utc)).isoformat(),
        "transport": resolved_transport,
    }
    return report



def prune_archives(
    endpoint: str = DEFAULT_ENDPOINT,
    bucket: str = DEFAULT_BUCKET,
    credentials_file: Path | str | None = None,
    keep_count: int = 7,
) -> list[str]:
    """Retention helper: prunes older daily archives but never deletes the newest verified object."""
    if keep_count < 1:
        die("keep_count must be at least 1")

    access_key, secret_key = load_credentials(credentials_file=credentials_file)
    s3 = S3Client(endpoint=endpoint, access_key=access_key, secret_key=secret_key)

    objects = s3.list_objects(bucket, prefix="daily/")
    # Find all .tar archive keys (excluding .sha256 sidecars)
    archive_keys = [obj["key"] for obj in objects if obj["key"].endswith(".tar")]
    # Sorted lexicographically: format daily/YYYY/MM/DD/portfolio-lab-data-<timestamp>.tar sorts chronologically
    archive_keys.sort()

    if len(archive_keys) <= keep_count:
        return []

    # To be deleted: oldest items exceeding keep_count
    # Guarantee: the newest object(s) are strictly preserved
    to_delete = archive_keys[: len(archive_keys) - keep_count]
    deleted = []

    for key in to_delete:
        s3.delete_object(bucket, key)
        deleted.append(key)
        sc_key = key + ".sha256"
        if s3.head_object(bucket, sc_key):
            s3.delete_object(bucket, sc_key)
            deleted.append(sc_key)

    return deleted


def verify_remote_archive(
    object_key: str,
    endpoint: str = DEFAULT_ENDPOINT,
    bucket: str = DEFAULT_BUCKET,
    credentials_file: Path | str | None = None,
) -> dict[str, Any]:
    """Fetch remote object and sidecar to temporary storage, then run recovery verification."""
    access_key, secret_key = load_credentials(credentials_file=credentials_file)
    s3 = S3Client(endpoint=endpoint, access_key=access_key, secret_key=secret_key)

    sidecar_key = sidecar_key_for(object_key)
    archive_data = s3.get_object(bucket, object_key)
    sidecar_data = s3.get_object(bucket, sidecar_key)

    with tempfile.TemporaryDirectory(prefix="plr-s3-verify-") as tmpdir:
        tmppath = Path(tmpdir)
        local_archive = tmppath / Path(object_key).name
        local_sidecar = tmppath / Path(sidecar_key).name
        local_archive.write_bytes(archive_data)
        local_sidecar.write_bytes(sidecar_data)

        ok, report = plr.verify_archive(local_archive, local_sidecar)
        if not ok:
            die(f"remote archive verify failed: {report.get('error', 'unknown error')}")
        report["remote_bucket"] = bucket
        report["remote_key"] = object_key
        return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish and manage Portfolio Lab non-git archives in SeaweedFS S3."
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help=f"S3 endpoint (default: {DEFAULT_ENDPOINT})")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help=f"S3 bucket (default: {DEFAULT_BUCKET})")
    parser.add_argument("--credentials-file", help="Path to mode 0600 credentials file")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # publish subcommand
    p_pub = subparsers.add_parser("publish", help="Publish a verified archive to S3")
    p_pub.add_argument("--archive", required=True, help="Path to .tar archive")
    p_pub.add_argument("--sidecar", help="Path to .sha256 sidecar (default: <archive>.sha256)")
    p_pub.add_argument(
        "--transport",
        choices=["rclone", "urllib", "auto"],
        default="auto",
        help="Transport to use for archive upload (default: auto)",
    )


    # prune subcommand
    p_prune = subparsers.add_parser("prune", help="Prune old archives keeping the newest N")
    p_prune.add_argument("--keep", type=int, default=7, help="Number of newest archives to keep (default: 7)")

    # verify-remote subcommand
    p_vr = subparsers.add_parser("verify-remote", help="Fetch and verify a remote archive from S3")
    p_vr.add_argument("--key", required=True, help="S3 object key to verify")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "publish":
        archive_path = Path(args.archive).resolve()
        sidecar_path = Path(args.sidecar).resolve() if args.sidecar else None
        res = publish_archive(
            archive_path=archive_path,
            sidecar_path=sidecar_path,
            endpoint=args.endpoint,
            bucket=args.bucket,
            credentials_file=args.credentials_file,
            transport=args.transport,
        )
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0


    if args.command == "prune":
        deleted = prune_archives(
            endpoint=args.endpoint,
            bucket=args.bucket,
            credentials_file=args.credentials_file,
            keep_count=args.keep,
        )
        print(json.dumps({"ok": True, "deleted": deleted, "keep": args.keep}, indent=2))
        return 0

    if args.command == "verify-remote":
        res = verify_remote_archive(
            object_key=args.key,
            endpoint=args.endpoint,
            bucket=args.bucket,
            credentials_file=args.credentials_file,
        )
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0

    die(f"unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
