"""Tests for the focused static SPA origin (Task 2.3).

Tests exercise the standalone CLI:
    python portfolio_lab_static_origin.py --web-root ... --host 127.0.0.1 --port ... --max-inflight ...

Strict TDD: written and run against absent script to establish RED before implementation.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGIN_SCRIPT = PROJECT_ROOT / "scripts" / "portfolio_lab_static_origin.py"


@pytest.fixture
def web_root(tmp_path: Path) -> Path:
    """Create a minimal valid web root."""
    root = tmp_path / "www"
    root.mkdir()
    index_file = root / "index.html"
    index_file.write_text("<!DOCTYPE html><html><body>Root SPA</body></html>", encoding="utf-8")
    data_dir = root / "data"
    data_dir.mkdir()
    (data_dir / "test.json").write_text('{"status":"ok"}', encoding="utf-8")
    assets_dir = root / "assets"
    assets_dir.mkdir()
    (assets_dir / "index-a1b2c3d4.js").write_text("console.log('hashed asset');", encoding="utf-8")
    (assets_dir / "unhashed.js").write_text("console.log('unhashed asset');", encoding="utf-8")
    (root / "_release.json").write_text('{"version":"1.0.0"}', encoding="utf-8")
    return root


def run_origin_cli(args: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(ORIGIN_SCRIPT), *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env if env is not None else os.environ.copy(),
    )


def start_origin_server(
    web_root: Path,
    port: int = 0,
    max_inflight: int = 16,
    host: str = "127.0.0.1",
) -> tuple[subprocess.Popen[str], int]:
    """Start the origin server and wait for the ready JSON line on stdout."""
    cmd = [
        sys.executable,
        str(ORIGIN_SCRIPT),
        "--web-root",
        str(web_root),
        "--host",
        host,
        "--port",
        str(port),
        "--max-inflight",
        str(max_inflight),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Read the first line of stdout to extract ready port
    line = proc.stdout.readline() if proc.stdout else ""
    if not line:
        stderr = proc.stderr.read() if proc.stderr else ""
        proc.kill()
        raise RuntimeError(f"Server failed to emit ready line: stderr={stderr!r}")
    try:
        ready_data = json.loads(line)
        selected_port = ready_data["port"]
    except (json.JSONDecodeError, KeyError) as err:
        stderr = proc.stderr.read() if proc.stderr else ""
        proc.kill()
        raise RuntimeError(f"Malformed ready line {line!r}: stderr={stderr!r}") from err

    # Brief pause to ensure loop is accepting
    time.sleep(0.05)
    return proc, selected_port


def stop_origin_server(proc: subprocess.Popen[str], timeout: float = 5.0) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)


def http_request(
    port: int,
    path: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    host: str = "127.0.0.1",
) -> tuple[int, dict[str, str], bytes]:
    """Perform a low-level HTTP request and return (status, headers_dict, body)."""
    conn = socket.create_connection((host, port), timeout=5.0)
    req_headers = headers.copy() if headers else {}
    req_headers.setdefault("Host", f"{host}:{port}")
    req_headers.setdefault("Connection", "close")
    hdrs = "".join(f"{k}: {v}\r\n" for k, v in req_headers.items())
    raw_req = f"{method} {path} HTTP/1.1\r\n{hdrs}\r\n".encode("latin1")
    conn.sendall(raw_req)

    response_bytes = bytearray()
    while True:
        try:
            chunk = conn.recv(4096)
            if not chunk:
                break
            response_bytes.extend(chunk)
        except OSError:
            break
    conn.close()

    header_end = response_bytes.find(b"\r\n\r\n")
    if header_end == -1:
        raise ValueError("Incomplete HTTP response headers")
    header_raw = response_bytes[:header_end].decode("latin1")
    body = bytes(response_bytes[header_end + 4 :])

    lines = header_raw.split("\r\n")
    status_line = lines[0]
    parts = status_line.split(" ", 2)
    status_code = int(parts[1])

    resp_headers: dict[str, str] = {}
    for line in lines[1:]:
        if ": " in line:
            k, v = line.split(": ", 1)
            resp_headers[k.lower()] = v

    return status_code, resp_headers, body


# ── 1. Validation Before Bind ─────────────────────────────────────────────


def test_cli_requires_web_root() -> None:
    res = run_origin_cli(["--host", "127.0.0.1", "--port", "8001"])
    assert res.returncode != 0
    assert "web-root" in res.stderr.lower() or "required" in res.stderr.lower()


def test_cli_rejects_relative_web_root() -> None:
    res = run_origin_cli(["--web-root", "relative/path", "--host", "127.0.0.1", "--port", "8001"])
    assert res.returncode != 0
    assert "absolute" in res.stderr.lower()


def test_cli_rejects_symlink_web_root(tmp_path: Path, web_root: Path) -> None:
    symlink_root = tmp_path / "sym_www"
    symlink_root.symlink_to(web_root)
    res = run_origin_cli(["--web-root", str(symlink_root), "--host", "127.0.0.1", "--port", "8001"])
    assert res.returncode != 0
    assert "symlink" in res.stderr.lower()


def test_cli_rejects_missing_index_html(tmp_path: Path) -> None:
    bad_root = tmp_path / "no_index"
    bad_root.mkdir()
    (bad_root / "data").mkdir()
    res = run_origin_cli(["--web-root", str(bad_root), "--host", "127.0.0.1", "--port", "8001"])
    assert res.returncode != 0
    assert "index.html" in res.stderr.lower()


def test_cli_rejects_symlink_index_html(tmp_path: Path, web_root: Path) -> None:
    bad_root = tmp_path / "sym_index"
    bad_root.mkdir()
    (bad_root / "data").mkdir()
    target_index = tmp_path / "real_index.html"
    target_index.write_text("real index")
    (bad_root / "index.html").symlink_to(target_index)
    res = run_origin_cli(["--web-root", str(bad_root), "--host", "127.0.0.1", "--port", "8001"])
    assert res.returncode != 0
    assert "symlink" in res.stderr.lower() or "index.html" in res.stderr.lower()


def test_cli_rejects_missing_data_dir(tmp_path: Path) -> None:
    bad_root = tmp_path / "no_data"
    bad_root.mkdir()
    (bad_root / "index.html").write_text("ok")
    res = run_origin_cli(["--web-root", str(bad_root), "--host", "127.0.0.1", "--port", "8001"])
    assert res.returncode != 0
    assert "data" in res.stderr.lower()


def test_cli_rejects_symlink_data_dir(tmp_path: Path, web_root: Path) -> None:
    bad_root = tmp_path / "sym_data"
    bad_root.mkdir()
    (bad_root / "index.html").write_text("ok")
    target_data = tmp_path / "real_data"
    target_data.mkdir()
    (bad_root / "data").symlink_to(target_data)
    res = run_origin_cli(["--web-root", str(bad_root), "--host", "127.0.0.1", "--port", "8001"])
    assert res.returncode != 0
    assert "symlink" in res.stderr.lower() or "data" in res.stderr.lower()


@pytest.mark.parametrize("bad_host", ["0.0.0.0", "localhost", "::", "192.168.1.100"])
def test_cli_rejects_non_loopback_host(web_root: Path, bad_host: str) -> None:
    res = run_origin_cli(["--web-root", str(web_root), "--host", bad_host, "--port", "8001"])
    assert res.returncode != 0
    assert "127.0.0.1" in res.stderr


@pytest.mark.parametrize("bad_port", ["-1", "65536", "abc", "70000"])
def test_cli_rejects_invalid_port(web_root: Path, bad_port: str) -> None:
    res = run_origin_cli(["--web-root", str(web_root), "--host", "127.0.0.1", "--port", bad_port])
    assert res.returncode != 0


@pytest.mark.parametrize("bad_inflight", ["0", "-5", "65", "100", "xyz"])
def test_cli_rejects_invalid_max_inflight(web_root: Path, bad_inflight: str) -> None:
    res = run_origin_cli([
        "--web-root", str(web_root),
        "--host", "127.0.0.1",
        "--port", "8001",
        "--max-inflight", bad_inflight,
    ])
    assert res.returncode != 0
    assert "max-inflight" in res.stderr.lower()


# ── 2. GET/HEAD parity & Unsupported Methods 405 ─────────────────────────


def test_get_head_parity_and_unsupported_methods(web_root: Path) -> None:
    proc, port = start_origin_server(web_root)
    try:
        # GET index
        get_status, get_headers, get_body = http_request(port, "/")
        assert get_status == 200
        assert get_body == b"<!DOCTYPE html><html><body>Root SPA</body></html>"

        # HEAD index
        head_status, head_headers, head_body = http_request(port, "/", method="HEAD")
        assert head_status == 200
        assert head_body == b""
        assert get_headers["content-length"] == head_headers["content-length"]
        assert get_headers["content-type"] == head_headers["content-type"]
        assert get_headers.get("x-portfolio-lab-spa-fallback") == head_headers.get("x-portfolio-lab-spa-fallback")

        # Unsupported methods: POST, PUT, DELETE, PATCH, OPTIONS, CONNECT, TRACE
        for m in ("POST", "PUT", "DELETE", "PATCH", "OPTIONS", "CONNECT", "TRACE", "UNKNOWN"):
            status, headers, body = http_request(port, "/", method=m)
            assert status == 405, f"Expected 405 for method {m}"
            assert headers.get("allow") == "GET, HEAD"
    finally:
        stop_origin_server(proc)


# ── 3. Root, History SPA fallback, exact files, missing 404 ──────────────


def test_spa_fallback_and_exact_files(web_root: Path) -> None:
    proc, port = start_origin_server(web_root)
    try:
        # Root / -> fallback to index.html with SPA fallback header
        status, headers, body = http_request(port, "/")
        assert status == 200
        assert headers.get("x-portfolio-lab-spa-fallback") == "1"
        assert body == b"<!DOCTYPE html><html><body>Root SPA</body></html>"

        # Exact existing index.html -> regular file, no fallback header
        status, headers, body = http_request(port, "/index.html")
        assert status == 200
        assert headers.get("x-portfolio-lab-spa-fallback") is None
        assert body == b"<!DOCTYPE html><html><body>Root SPA</body></html>"

        # History SPA route -> non-file path falls back to index.html with fallback header
        for path in ("/design-guide", "/settings", "/portfolio/view/42"):
            status, headers, body = http_request(port, path)
            assert status == 200, f"Expected 200 fallback for {path}"
            assert headers.get("x-portfolio-lab-spa-fallback") == "1"
            assert body == b"<!DOCTYPE html><html><body>Root SPA</body></html>"

        # Missing file-like path (final segment contains a dot) -> 404, NEVER fallback
        for path in ("/missing.js", "/style.css", "/component.tsx", "/data.json"):
            status, headers, body = http_request(port, path)
            assert status == 404, f"Expected 404 for missing file-like path {path}"
            assert headers.get("x-portfolio-lab-spa-fallback") is None

        # Missing asset -> 404, NEVER fallback
        status, headers, body = http_request(port, "/assets/missing-asset.js")
        assert status == 404
        assert headers.get("x-portfolio-lab-spa-fallback") is None

        # Missing data -> 404, NEVER fallback
        status, headers, body = http_request(port, "/data/missing.json")
        assert status == 404
        assert headers.get("x-portfolio-lab-spa-fallback") is None

        # Accidental web-root file or directory under api cannot be served
        api_dir = web_root / "api"
        api_dir.mkdir()
        (api_dir / "tasker").mkdir()
        (api_dir / "tasker" / "status").write_text('{"accidental":"file"}', encoding="utf-8")

        # Direct GET and HEAD requests for /api/tasker/status and its encoded-letter
        # spelling on static origin port must return 404, no SPA fallback header,
        # and empty body for HEAD.
        for api_path in ("/api/tasker/status", "/%61pi/tasker/status"):
            get_status, get_headers, get_body = http_request(port, api_path, method="GET")
            assert get_status == 404
            assert get_headers.get("x-portfolio-lab-spa-fallback") is None

            head_status, head_headers, head_body = http_request(port, api_path, method="HEAD")
            assert head_status == 404
            assert head_headers.get("x-portfolio-lab-spa-fallback") is None
            assert head_body == b""

        # Also bare /api returns 404, no fallback
        api_bare_status, api_bare_headers, _ = http_request(port, "/api", method="GET")
        assert api_bare_status == 404
        assert api_bare_headers.get("x-portfolio-lab-spa-fallback") is None
    finally:
        stop_origin_server(proc)


# ── 4. Cache Headers ──────────────────────────────────────────────────────


def test_cache_headers(web_root: Path) -> None:
    proc, port = start_origin_server(web_root)
    try:
        # Hashed asset -> public, max-age=31536000, immutable
        status, headers, body = http_request(port, "/assets/index-a1b2c3d4.js")
        assert status == 200
        assert headers.get("cache-control") == "public, max-age=31536000, immutable"

        # Unhashed asset -> no-cache
        status, headers, body = http_request(port, "/assets/unhashed.js")
        assert status == 200
        assert headers.get("cache-control") == "no-cache"

        # Data file -> no-cache
        status, headers, body = http_request(port, "/data/test.json")
        assert status == 200
        assert headers.get("cache-control") == "no-cache"

        # Release json -> no-cache
        status, headers, body = http_request(port, "/_release.json")
        assert status == 200
        assert headers.get("cache-control") == "no-cache"

        # Fallback response -> no-cache
        status, headers, body = http_request(port, "/design-guide")
        assert status == 200
        assert headers.get("cache-control") == "no-cache"
    finally:
        stop_origin_server(proc)


# ── 5. Traversal, Malformed Encoding, Symlinks, Directory Listing 400/403/404 ─


def test_security_rejections(web_root: Path, tmp_path: Path) -> None:
    # Setup a symlink inside web_root to an outside file
    outside_file = tmp_path / "secret.txt"
    outside_file.write_text("secret outside")
    (web_root / "assets" / "sym_outside.js").symlink_to(outside_file)

    # Setup an in-root file symlink
    (web_root / "assets" / "sym_inroot.js").symlink_to(web_root / "assets" / "unhashed.js")

    # Setup a symlinked directory component inside assets
    sym_dir = web_root / "assets" / "symdir"
    real_dir = tmp_path / "real_dir"
    real_dir.mkdir()
    (real_dir / "child.js").write_text("inside symdir")
    sym_dir.symlink_to(real_dir)

    proc, port = start_origin_server(web_root)
    try:
        # Traversal attempts -> 400
        for path in ("/../index.html", "/assets/../index.html", "/%2e%2e/index.html", "/..", "/."):
            status, _, _ = http_request(port, path)
            assert status == 400, f"Expected 400 for traversal {path}"

        # Encoded separators -> 400
        for path in ("/assets%2findex.js", "/assets%2Findex.js", "/assets%5cindex.js", "/assets%5Cindex.js"):
            status, _, _ = http_request(port, path)
            assert status == 400, f"Expected 400 for encoded separator {path}"

        # Backslashes -> 400
        status, _, _ = http_request(port, r"/assets\index.js")
        assert status == 400

        # Control characters and NUL -> 400
        for path in ("/assets%00.js", "/assets%01.js", "/assets%1f.js", "/assets%7f.js"):
            status, _, _ = http_request(port, path)
            assert status == 400, f"Expected 400 for control/NUL {path}"

        # Malformed percent encoding / non-UTF8 -> 400
        for path in ("/assets%zz.js", "/assets%2.js", "/assets%ff.js"):
            status, _, _ = http_request(port, path)
            assert status == 400, f"Expected 400 for malformed percent encoding {path}"

        # Empty segments (double slashes) -> 400
        for path in ("//index.html", "/assets//unhashed.js"):
            status, _, _ = http_request(port, path)
            assert status == 400, f"Expected 400 for double slash {path}"

        # Directory listing is never served -> 404
        for path in ("/data", "/assets"):
            status, _, body = http_request(port, path)
            assert status == 404, f"Expected 404 for directory path {path}"
            assert b"<html" not in body.lower() or b"directory" not in body.lower()

        # Symlinks -> 403
        for path in ("/assets/sym_outside.js", "/assets/sym_inroot.js", "/assets/symdir/child.js"):
            status, _, _ = http_request(port, path)
            assert status == 403, f"Expected 403 for symlink path {path}"
    finally:
        stop_origin_server(proc)


# ── 6. Content Length, Types, Nosniff, and No Query Leakage ──────────────


def test_headers_and_no_query_leakage(web_root: Path) -> None:
    proc, port = start_origin_server(web_root)
    try:
        # Check nosniff and Content-Length and Content-Type on regular file
        status, headers, body = http_request(port, "/data/test.json")
        assert status == 200
        assert headers.get("x-content-type-options") == "nosniff"
        assert headers.get("content-type") == "application/json"
        assert headers.get("content-length") == str(len(body))

        # Check nosniff on JS file
        status, headers, body = http_request(port, "/assets/index-a1b2c3d4.js")
        assert status == 200
        assert headers.get("x-content-type-options") == "nosniff"
        assert "javascript" in headers.get("content-type", "")

        # Query and fragment are ignored for path selection and not leaked
        query = "?secret_token=supersecret12345"
        status, headers, body = http_request(port, f"/data/test.json{query}")
        assert status == 200
        assert b"supersecret12345" not in body

        # Query not leaked on missing file 404
        status, headers, body = http_request(port, f"/missing.js{query}")
        assert status == 404
        assert b"supersecret12345" not in body
    finally:
        stop_origin_server(proc)


# ── 7. Bounded Concurrency & Graceful Shutdown ───────────────────────────


def test_bounded_concurrency_and_graceful_shutdown(web_root: Path) -> None:
    # Start server with max-inflight=2
    proc, port = start_origin_server(web_root, max_inflight=2)
    try:
        # Open 2 slow connections that hold capacity by sending incomplete headers
        s1 = socket.create_connection(("127.0.0.1", port))
        s1.sendall(b"GET /data/test.json HTTP/1.1\r\nHost: 127.0.0.1\r\n")

        s2 = socket.create_connection(("127.0.0.1", port))
        s2.sendall(b"GET /data/test.json HTTP/1.1\r\nHost: 127.0.0.1\r\n")

        # Wait until capacity is occupied, then verify third connection gets prompt 503
        resp = b""
        for _ in range(20):
            time.sleep(0.05)
            s3 = socket.create_connection(("127.0.0.1", port))
            s3.settimeout(2.0)
            s3.sendall(b"GET /data/test.json HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
            try:
                resp = s3.recv(4096)
            except OSError:
                resp = b""
            finally:
                s3.close()
            if b"503 Service Unavailable" in resp:
                break

        assert b"503 Service Unavailable" in resp
        assert b"X-Content-Type-Options: nosniff" in resp

        # Clean up held sockets
        s1.close()
        s2.close()
    finally:
        # Test graceful SIGTERM shutdown
        stop_origin_server(proc)
        # Verify port released promptly
        time.sleep(0.1)
        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=0.5)


def test_overlong_header_rejected(web_root: Path) -> None:
    proc, port = start_origin_server(web_root)
    try:
        # Test 1: Incomplete header > 16 KiB without terminator
        s = socket.create_connection(("127.0.0.1", port))
        s.settimeout(2.0)
        junk = b"X-Junk: " + (b"A" * 17000) + b"\r\n"
        s.sendall(b"GET / HTTP/1.1\r\n" + junk)
        data = bytearray()
        while True:
            try:
                chunk = s.recv(4096)
                if not chunk:
                    break
                data.extend(chunk)
            except OSError:
                break
        s.close()
        assert b"200 OK" not in data

        # Test 2: Complete header block > 16 KiB with \r\n\r\n
        s2 = socket.create_connection(("127.0.0.1", port))
        s2.settimeout(2.0)
        s2.sendall(b"GET / HTTP/1.1\r\n" + junk + b"\r\n")
        data2 = bytearray()
        while True:
            try:
                chunk = s2.recv(4096)
                if not chunk:
                    break
                data2.extend(chunk)
            except OSError:
                break
        s2.close()
        assert b"200 OK" not in data2
        if data2:
            assert b"431" in data2 or b"400" in data2
            assert b"X-Content-Type-Options: nosniff" in data2
    finally:
        stop_origin_server(proc)


@pytest.mark.parametrize(
    ("filename", "expected_mime"),
    [
        ("test.html", "text/html; charset=utf-8"),
        ("test.css", "text/css; charset=utf-8"),
        ("test.js", "text/javascript; charset=utf-8"),
        ("test.mjs", "text/javascript; charset=utf-8"),
        ("test.json", "application/json"),
        ("test.map", "application/json"),
        ("test.svg", "image/svg+xml"),
        ("test.png", "image/png"),
        ("test.jpg", "image/jpeg"),
        ("test.jpeg", "image/jpeg"),
        ("test.gif", "image/gif"),
        ("test.ico", "image/x-icon"),
        ("test.txt", "text/plain; charset=utf-8"),
        ("test.wasm", "application/wasm"),
        ("test.woff", "font/woff"),
        ("test.woff2", "font/woff2"),
        ("test.bin", "application/octet-stream"),
    ],
)
def test_deterministic_content_types(web_root: Path, filename: str, expected_mime: str) -> None:
    # Create test file under web_root
    target = web_root / filename
    target.write_bytes(b"content")

    proc, port = start_origin_server(web_root)
    try:
        status, headers, body = http_request(port, f"/{filename}")
        assert status == 200
        assert headers.get("content-type") == expected_mime
    finally:
        stop_origin_server(proc)


def test_malformed_request_line_returns_400(web_root: Path) -> None:
    proc, port = start_origin_server(web_root)
    try:
        s = socket.create_connection(("127.0.0.1", port))
        s.settimeout(2.0)
        s.sendall(b"BROKEN\r\n\r\n")
        data = s.recv(4096)
        s.close()
        assert b"400 Bad Request" in data
        assert b"Content-Length: 0" in data
        assert b"X-Content-Type-Options: nosniff" in data
    finally:
        stop_origin_server(proc)


def test_bounded_shutdown_deadline_with_stalled_clients(web_root: Path) -> None:
    # Test with stalled clients: server shutdown must complete boundedly within overall deadline (e.g. < 2.5s)
    # even when max_inflight=8 clients hold connections
    proc, port = start_origin_server(web_root, max_inflight=8)
    sockets = []
    try:
        for _ in range(8):
            s = socket.create_connection(("127.0.0.1", port))
            s.sendall(b"GET /data/test.json HTTP/1.1\r\nHost: 127.0.0.1\r\n")
            sockets.append(s)

        # Give them a moment to be accepted
        time.sleep(0.05)

        t0 = time.monotonic()
        stop_origin_server(proc, timeout=3.0)
        elapsed = time.monotonic() - t0

        # Shutdown must have completed within 2.5s, NOT up to 8*1s sequential join
        assert elapsed < 2.5, f"Shutdown took too long: {elapsed}s"

        # Verify port released
        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=0.5)
    finally:
        for s in sockets:
            try:
                s.close()
            except OSError:
                pass
        stop_origin_server(proc)


def test_handler_fault_is_not_silently_swallowed(web_root: Path) -> None:
    # Test request processing helper with an injected handler that raises
    from scripts.portfolio_lab_static_origin import process_client_request

    class FaultyHandler:
        def handle_request(self, method: str, path: str) -> tuple[int, dict[str, str], bytes]:
            raise RuntimeError("Database connection exploded with secret=12345")

    faulty = FaultyHandler()
    status, headers, body, err = process_client_request(b"GET / HTTP/1.1\r\n\r\n", faulty)
    assert status == 500
    assert headers.get("X-Content-Type-Options") == "nosniff"
    assert err is not None
    # Bounded diagnostic must contain exception type only, never secret/path/query
    assert "RuntimeError" in err
    assert "12345" not in err
    assert "exploded" not in err
