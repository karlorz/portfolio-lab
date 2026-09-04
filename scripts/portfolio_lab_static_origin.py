#!/usr/bin/env python3
"""Standalone, standard-library-only static SPA origin for Portfolio Lab (Task 2.3).

Command:
    python portfolio_lab_static_origin.py --web-root ABSOLUTE_PATH --host 127.0.0.1 --port 8001 --max-inflight 16
"""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import signal
import socket
import stat
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote_to_bytes

SUPPORTED_METHODS = ("GET", "HEAD")


def die(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def is_hashed_asset(path: str) -> bool:
    """Check if basename contains a Vite-style hexadecimal content hash of at least 8 chars."""
    basename = os.path.basename(path)
    stem, _, ext = basename.rpartition(".")
    target = stem if stem else basename
    # Match pattern like name-a1b2c3d4 or name.a1b2c3d4
    for part in re.split(r"[-_.]", target):
        if len(part) >= 8 and re.fullmatch(r"^[0-9a-fA-F]+$", part):
            return True
    return False


def validate_web_root(raw_path: str) -> Path:
    if not os.path.isabs(raw_path):
        die(f"--web-root must be an absolute path: {raw_path!r}")
    path = Path(raw_path)
    try:
        st = os.lstat(path)
    except OSError as exc:
        die(f"cannot stat --web-root: {exc}")
    if stat.S_ISLNK(st.st_mode):
        die(f"--web-root must not be a symlink: {path}")
    if not stat.S_ISDIR(st.st_mode):
        die(f"--web-root must be a directory: {path}")

    # Check index.html
    index_path = path / "index.html"
    try:
        st_idx = os.lstat(index_path)
    except OSError:
        die(f"--web-root must contain index.html: {index_path}")
    if stat.S_ISLNK(st_idx.st_mode):
        die(f"index.html must not be a symlink: {index_path}")
    if not stat.S_ISREG(st_idx.st_mode):
        die(f"index.html must be a regular file: {index_path}")

    # Check data/
    data_path = path / "data"
    try:
        st_data = os.lstat(data_path)
    except OSError:
        die(f"--web-root must contain data/ directory: {data_path}")
    if stat.S_ISLNK(st_data.st_mode):
        die(f"data/ must not be a symlink: {data_path}")
    if not stat.S_ISDIR(st_data.st_mode):
        die(f"data/ must be a directory: {data_path}")

    return path.resolve()


def decode_url_path(raw_url: str) -> str | None:
    """Decode path exactly once as UTF-8. Reject malformed escapes, control/NUL, backslashes, empty/dot/dot-dot segments."""
    # Strip query and fragment
    path_part = raw_url.split("?", 1)[0].split("#", 1)[0]
    if not path_part.startswith("/"):
        return None
    if "\\" in path_part:
        return None

    # Check for malformed percent encoding: every '%' must be followed by 2 hex digits
    i = 0
    length = len(path_part)
    while i < length:
        if path_part[i] == "%":
            if i + 2 >= length:
                return None
            h1 = path_part[i + 1]
            h2 = path_part[i + 2]
            if h1 not in "0123456789abcdefABCDEF" or h2 not in "0123456789abcdefABCDEF":
                return None
            i += 3
        else:
            i += 1

    # Check for encoded slash (%2f / %2F) or encoded backslash (%5c / %5C)
    lower_orig = path_part.lower()
    if "%2f" in lower_orig or "%5c" in lower_orig:
        return None

    # Percent decode bytes exactly once
    try:
        decoded_bytes = unquote_to_bytes(path_part)
    except Exception:
        return None

    try:
        decoded_str = decoded_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None

    # Control chars (0-31 and 127) or NUL or backslash
    for ch in decoded_str:
        code = ord(ch)
        if code < 32 or code == 127 or ch == "\\":
            return None

    # Check path segments
    segments = decoded_str.split("/")
    # Leading segment before the first '/' is empty
    if segments[0] != "":
        return None
    # For a path like "/", segments are ["", ""]
    # For a path like "/a/b", segments are ["", "a", "b"]
    # Check intermediate or non-root segments
    for idx, seg in enumerate(segments[1:], start=1):
        if idx == len(segments) - 1 and seg == "" and len(segments) == 2:
            # This is "/"
            continue
        if seg == "" or seg == "." or seg == "..":
            return None

    return decoded_str


CONTENT_TYPES: dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".map": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
    ".wasm": "application/wasm",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
}


def guess_content_type(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    return CONTENT_TYPES.get(suffix, "application/octet-stream")


def verify_no_symlinks_in_path(root: Path, full_path: Path) -> bool:
    """Verify that every path component from root to full_path is not a symlink and stays beneath root."""
    try:
        rel = full_path.relative_to(root)
    except ValueError:
        return False

    current = root
    for part in rel.parts:
        current = current / part
        try:
            st = os.lstat(current)
            if stat.S_ISLNK(st.st_mode):
                return False
        except OSError:
            # Non-existent intermediate or target
            return True
    return True


class StaticOriginHandler:
    def __init__(self, web_root: Path) -> None:
        self.web_root = web_root
        self.index_html = web_root / "index.html"

    def handle_request(self, method: str, raw_path: str) -> tuple[int, dict[str, str], bytes]:
        headers: dict[str, str] = {
            "X-Content-Type-Options": "nosniff",
            "Connection": "close",
        }
        if method not in ("GET", "HEAD"):
            headers["Allow"] = "GET, HEAD"
            headers["Content-Length"] = "0"
            return 405, headers, b""

        path = decode_url_path(raw_path)
        if path is None:
            headers["Content-Length"] = "0"
            return 400, headers, b""

        # /api or /api/* paths are owned exclusively by the API origin and must never
        # be served by the static origin, falling back to SPA, or serving web-root files under api.
        if path == "/api" or path.startswith("/api/"):
            headers["Content-Length"] = "0"
            return 404, headers, b""

        # Normalize relative path beneath web_root
        rel_path = path.lstrip("/")
        candidate_file = (self.web_root / rel_path) if rel_path else self.index_html

        # Check symlinks along the path
        if not verify_no_symlinks_in_path(self.web_root, candidate_file):
            headers["Content-Length"] = "0"
            return 403, headers, b""

        # Check if candidate_file is an existing directory
        try:
            st = os.lstat(candidate_file)
            if stat.S_ISLNK(st.st_mode):
                headers["Content-Length"] = "0"
                return 403, headers, b""
            is_dir = stat.S_ISDIR(st.st_mode)
            is_reg = stat.S_ISREG(st.st_mode)
        except FileNotFoundError:
            is_dir = False
            is_reg = False
        except OSError:
            headers["Content-Length"] = "0"
            return 500, headers, b""

        # Directory listing is never served
        if is_dir and path != "/":
            headers["Content-Length"] = "0"
            return 404, headers, b""

        # Route-specific serving
        if path.startswith("/assets/"):
            if not is_reg:
                headers["Content-Length"] = "0"
                return 404, headers, b""
            # Asset exists and is regular file
            if is_hashed_asset(candidate_file.name):
                headers["Cache-Control"] = "public, max-age=31536000, immutable"
            else:
                headers["Cache-Control"] = "no-cache"
            return self._serve_file(candidate_file, headers)

        if path.startswith("/data/"):
            if not is_reg:
                headers["Content-Length"] = "0"
                return 404, headers, b""
            headers["Cache-Control"] = "no-cache"
            return self._serve_file(candidate_file, headers)

        if path == "/_release.json":
            if not is_reg:
                headers["Content-Length"] = "0"
                return 404, headers, b""
            headers["Cache-Control"] = "no-cache"
            return self._serve_file(candidate_file, headers)

        # Existing exact regular file (other than root SPA fallback)
        if is_reg and path != "/":
            headers["Cache-Control"] = "no-cache"
            return self._serve_file(candidate_file, headers)

        # Non-file paths / root / history routes
        # If the path looks like a missing file (final segment contains a dot), return 404
        last_segment = path.rpartition("/")[2]
        if "." in last_segment:
            headers["Content-Length"] = "0"
            return 404, headers, b""

        # Serve index.html as SPA fallback
        headers["Cache-Control"] = "no-cache"
        headers["X-Portfolio-Lab-SPA-Fallback"] = "1"
        return self._serve_file(self.index_html, headers)

    def _serve_file(self, file_path: Path, headers: dict[str, str]) -> tuple[int, dict[str, str], bytes]:
        try:
            body = file_path.read_bytes()
        except OSError:
            headers["Content-Length"] = "0"
            return 500, headers, b""
        headers["Content-Type"] = guess_content_type(file_path)
        headers["Content-Length"] = str(len(body))
        return 200, headers, body


def process_client_request(
    raw_data: bytes,
    handler: Any,
) -> tuple[int, dict[str, str], bytes, str | None]:
    """Process a raw HTTP request block. Returns (status_code, headers, body, error_diagnostic)."""
    header_idx = raw_data.find(b"\r\n\r\n")
    if header_idx != -1 and header_idx + 4 > 16384:
        return 431, {"Content-Length": "0", "Connection": "close", "X-Content-Type-Options": "nosniff"}, b"", None
    if len(raw_data) >= 16384 and header_idx == -1:
        return 431, {"Content-Length": "0", "Connection": "close", "X-Content-Type-Options": "nosniff"}, b"", None

    first_line_end = raw_data.find(b"\r\n")
    if first_line_end == -1:
        return 400, {"Content-Length": "0", "Connection": "close", "X-Content-Type-Options": "nosniff"}, b"", None

    req_line = bytes(raw_data[:first_line_end]).decode("latin1", errors="replace")
    parts = req_line.split()
    if len(parts) < 2:
        return 400, {"Content-Length": "0", "Connection": "close", "X-Content-Type-Options": "nosniff"}, b"", None

    method = parts[0]
    raw_path = parts[1]
    try:
        status_code, headers, body = handler.handle_request(method, raw_path)
        return status_code, headers, body, None
    except Exception as exc:
        err = f"ERROR: internal request handler defect: {type(exc).__name__}"
        return 500, {"Content-Length": "0", "Connection": "close", "X-Content-Type-Options": "nosniff"}, b"", err


def run_server(
    web_root: Path,
    host: str = "127.0.0.1",
    port: int = 8001,
    max_inflight: int = 16,
) -> None:
    handler = StaticOriginHandler(web_root)
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server_sock.bind((host, port))
        selected_port = server_sock.getsockname()[1]
        server_sock.listen(128)
    except OSError as exc:
        die(f"cannot bind/listen on {host}:{port}: {exc}")
    server_sock.setblocking(False)

    # Report ready line
    ready_obj = {"ready": True, "host": host, "port": selected_port, "max_inflight": max_inflight}
    print(json.dumps(ready_obj, separators=(",", ":")), flush=True)

    semaphore = threading.Semaphore(max_inflight)
    stop_event = threading.Event()

    def signal_handler(signum: int, frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    def handle_client(client_sock: socket.socket) -> None:
        try:
            client_sock.settimeout(5.0)
            raw_data = bytearray()
            while True:
                header_idx = raw_data.find(b"\r\n\r\n")
                if header_idx != -1 or len(raw_data) >= 16384:
                    break
                chunk = client_sock.recv(4096)
                if not chunk:
                    break
                raw_data.extend(chunk)

            status_code, headers, body, err = process_client_request(bytes(raw_data), handler)
            if err is not None:
                print(err, file=sys.stderr)

            status_text = {
                200: "OK",
                400: "Bad Request",
                403: "Forbidden",
                404: "Not Found",
                405: "Method Not Allowed",
                431: "Request Header Fields Too Large",
                500: "Internal Server Error",
                503: "Service Unavailable",
            }.get(status_code, "Unknown")

            resp_lines = [f"HTTP/1.1 {status_code} {status_text}\r\n"]
            for k, v in headers.items():
                resp_lines.append(f"{k}: {v}\r\n")
            resp_lines.append("\r\n")
            resp_bytes = "".join(resp_lines).encode("latin1")
            req_method = bytes(raw_data[:raw_data.find(b"\r\n")]).decode("latin1", errors="replace").split()[0] if raw_data.find(b"\r\n") != -1 and raw_data[:raw_data.find(b"\r\n")].split() else ""
            if req_method != "HEAD":
                resp_bytes += body

            client_sock.sendall(resp_bytes)
        except (socket.timeout, OSError):
            pass
        finally:
            try:
                client_sock.close()
            except OSError:
                pass
            semaphore.release()

    active_threads: list[threading.Thread] = []

    while not stop_event.is_set():
        try:
            r, _, _ = select.select([server_sock], [], [], 0.2)
            if not r:
                continue
            client_sock, _ = server_sock.accept()
        except OSError:
            break

        # Attempt to acquire inflight semaphore
        if not semaphore.acquire(blocking=False):
            try:
                client_sock.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\n"
                    b"Content-Length: 0\r\n"
                    b"X-Content-Type-Options: nosniff\r\n"
                    b"Connection: close\r\n\r\n"
                )
                client_sock.close()
            except OSError:
                pass
            continue

        th = threading.Thread(target=handle_client, args=(client_sock,), daemon=True)
        th.start()
        active_threads.append(th)
        active_threads = [t for t in active_threads if t.is_alive()]

    server_sock.close()
    shutdown_deadline = time.monotonic() + 2.0
    for th in active_threads:
        remain = max(0.0, shutdown_deadline - time.monotonic())
        th.join(timeout=remain)


def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio Lab Static SPA Origin")
    parser.add_argument("--web-root", required=True, help="Absolute path to web root")
    parser.add_argument("--host", default="127.0.0.1", help="Loopback host (127.0.0.1 only)")
    parser.add_argument("--port", type=int, default=8001, help="Listen port (8001 production; 0 for test)")
    parser.add_argument("--max-inflight", type=int, default=16, help="Max concurrent requests (1-64)")
    args = parser.parse_args()

    if args.host != "127.0.0.1":
        die(f"invalid host {args.host!r}; only 127.0.0.1 is accepted")

    if args.port < 0 or args.port > 65535:
        die(f"port must be between 0 and 65535; got {args.port}")

    if args.max_inflight < 1 or args.max_inflight > 64:
        die(f"max-inflight must be between 1 and 64; got {args.max_inflight}")

    root = validate_web_root(args.web_root)
    run_server(root, host=args.host, port=args.port, max_inflight=args.max_inflight)


if __name__ == "__main__":
    main()
