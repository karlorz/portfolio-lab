#!/home/box/.local/bin/python3
"""Attended sg01 former-authority proof refresh for the cursor-box Portfolio Lab.

A one-shot, **attended** CLI (never scheduled, never wired into cron/tasker):
it probes the former authority host sg01 and, only when both legacy units are
exactly inactive and disabled, atomically rewrites the former-authority proof
consumed by the read-only daily evidence collector
(``portfolio_lab_daily_evidence.py``, ``authority`` category).

Probes (read-only, nothing else)
    ``systemctl is-active`` / ``systemctl is-enabled`` for
    ``portfolio-lab-tasker`` and ``portfolio-lab-s3-archive.timer``, each run
    as one explicit-argv SSH invocation with ``-o BatchMode=yes`` and a bounded
    ``-o ConnectTimeout``. A probe is accepted only on the canonical systemd
    answer for the desired state: stdout exactly ``inactive\\n`` with exit 3,
    or stdout exactly ``disabled\\n`` with exit 1.

Missing units (explicit policy)
    An absent unit file is NOT accepted as "not enabled". The answer systemd
    gives for a missing unit is not stable across versions (``not-found`` on
    stdout with exit 1, 4 or 5, or an error line on stderr with empty stdout),
    so no single (token, exit) pair can be pinned as canonical without
    probing the host itself, which this attended tool must not broaden.
    Decommissioning therefore keeps both unit files present and disabled; a
    genuinely absent unit fails closed and is an attended gate documented in
    ``PORTFOLIO_LAB_CURSOR_BOX_RECYCLE.md``.

Fail-closed contract
    Missing/unusable ssh, probe timeout, bounded-read overrun, any nonzero or
    unexpected exit code, any unexpected or malformed token, any active or
    enabled state, an unusable config value (relative/symlinked paths, an ssh
    override that is not absolute, a non-sg01 host, an out-of-range timeout, a
    naive or unconvertible ``--now``), a probe lifecycle error (fcntl/poll/kill)
    or a closed reporting pipe means: no new proof, the existing proof
    preserved byte-for-byte, temporary files cleaned, child processes reaped and
    their pipes closed, empty stdout, one static sanitized line on stderr, exit
    1. Nothing is ever written on failure and no remote output, host, or path
    is echoed.

Output (success only)
    ``PROOF`` holds exactly the ``portfolio-lab-former-authority-proof/v1``
    object ``{schema, host_label, collected_at, tasker, archive_timer}`` with
    ``host_label`` ``sg01``, an aware UTC ``collected_at``, and all four nested
    booleans ``false``. The file is committed atomically: a unique temp in the
    target directory (the directory is never created) is written, chmod'ed
    0600, and verified as a regular non-symlink file owned by the running uid
    at exactly mode 0600 and the exact size *before* the rename, so a failed
    verification can never displace an existing proof. The committed target is
    then confirmed; a mismatch is reported statically and never deletes an
    artifact (the collector independently rejects a non-conforming proof).
    Under a pinned ``--now`` reruns are byte-identical. Exactly one compact,
    secret-free JSON summary line goes to stdout; if that pipe is already
    closed the CLI still exits 1 with one static stderr line and no
    interpreter-shutdown noise, leaving the already-committed proof in place.

Exit codes
    0  proof refreshed (summary on stdout)
    1  invalid CLI/config, probe failure, or write/verify failure

Overrides (flag > env > default)
    --proof              PLSPR_PROOF              <root>/run/former-authority-proof.json
    --ssh                PLSPR_SSH                ssh (any override must be absolute)
    --host               PLSPR_HOST               sg01 (fixed; no other host is accepted)
    --connect-timeout    PLSPR_CONNECT_TIMEOUT    10 (whole seconds, 1..30)
    --now                PLSPR_NOW                (actual time; ISO-8601 with a UTC
                                                   offset or Z; naive input rejected)
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROOF_SCHEMA = "portfolio-lab-former-authority-proof/v1"
DEFAULT_HOST = "sg01"
DEFAULT_SSH = "ssh"
DEFAULT_ROOT = Path("/home/box/.local/share/portfolio-lab")
DEFAULT_PROOF = DEFAULT_ROOT / "run" / "former-authority-proof.json"
DEFAULT_CONNECT_TIMEOUT = 10.0
MIN_CONNECT_TIMEOUT = 1.0
MAX_CONNECT_TIMEOUT = 30.0
PROBE_WALL_GRACE = 5.0  # fixed slack beyond ConnectTimeout for the remote command

TASKER_SERVICE = "portfolio-lab-tasker"
ARCHIVE_TIMER_SERVICE = "portfolio-lab-s3-archive.timer"

# (subcommand, unit, expected stdout token, canonical exit code for that token)
PROBE_SPECS: tuple[tuple[str, str, str, int], ...] = (
    ("is-active", TASKER_SERVICE, "inactive", 3),
    ("is-enabled", TASKER_SERVICE, "disabled", 1),
    ("is-active", ARCHIVE_TIMER_SERVICE, "inactive", 3),
    ("is-enabled", ARCHIVE_TIMER_SERVICE, "disabled", 1),
)

MAX_PROBE_BYTES = 4096  # per probe stdout/stderr stream

_ENV_PREFIX = "PLSPR_"


def die(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    # Flush now so a closed reporting pipe raises here instead of producing an
    # "Exception ignored" message and exit code 120 at interpreter shutdown.
    sys.stdout.flush()


# ── config ────────────────────────────────────────────────────────────────


def _opt(args_value: str | None, env_name: str, default: str) -> str:
    if args_value is not None:
        return args_value
    raw = os.environ.get(f"{_ENV_PREFIX}{env_name}")
    if raw is not None and raw.strip() != "":
        return raw
    return default


def _abs(path_raw: str, what: str) -> Path:
    if not path_raw:
        die(f"{_ENV_PREFIX}{what} must be an absolute path")
    path = Path(path_raw)
    if not path.is_absolute():
        die(f"{_ENV_PREFIX}{what} must be an absolute path; got a relative path")
    if path.is_symlink():
        die(f"{_ENV_PREFIX}{what} must not be a symlink")
    return path


def _bounded_timeout(raw: str, what: str) -> float:
    """ssh ConnectTimeout is a whole number of seconds inside a fixed range."""
    try:
        value = float(raw)
    except ValueError:
        die(f"{_ENV_PREFIX}{what} must be a numeric value in seconds")
    if not math.isfinite(value):
        die(f"{_ENV_PREFIX}{what} must be a finite numeric value in seconds")
    if value != int(value) or not MIN_CONNECT_TIMEOUT <= value <= MAX_CONNECT_TIMEOUT:
        die(
            f"{_ENV_PREFIX}{what} must be a whole number of seconds between "
            f"{MIN_CONNECT_TIMEOUT:g} and {MAX_CONNECT_TIMEOUT:g}"
        )
    return value


def _parse_now(raw: str) -> datetime:
    """Aware UTC-offset input only. A naive timestamp has no defined zone for
    this contract and is rejected; extreme aware dates can overflow during
    conversion, which is also a static config failure."""
    text = raw.strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        die(f"{_ENV_PREFIX}NOW must be an ISO-8601 timestamp with a UTC offset or Z")
    if dt.tzinfo is None:
        die(f"{_ENV_PREFIX}NOW must be an ISO-8601 timestamp with a UTC offset or Z")
    try:
        return dt.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        die(f"{_ENV_PREFIX}NOW is outside the representable UTC range")


class Config:
    def __init__(self, args: argparse.Namespace) -> None:
        # The default is used only when no override is supplied. Any override
        # — including one that repeats the default name — must be absolute, so
        # PATH resolution can never be smuggled in through a flag or the env.
        if args.ssh is not None:
            self.ssh = str(_abs(args.ssh, "SSH"))
        else:
            env_ssh = os.environ.get(f"{_ENV_PREFIX}SSH", "").strip()
            self.ssh = str(_abs(env_ssh, "SSH")) if env_ssh else DEFAULT_SSH
        self.proof = _abs(_opt(args.proof, "PROOF", str(DEFAULT_PROOF)), "PROOF")
        host = _opt(args.host, "HOST", DEFAULT_HOST)
        if host != DEFAULT_HOST:
            die(f"{_ENV_PREFIX}HOST must be exactly {DEFAULT_HOST}")
        self.host = host
        self.connect_timeout = _bounded_timeout(
            _opt(args.connect_timeout, "CONNECT_TIMEOUT", str(DEFAULT_CONNECT_TIMEOUT)),
            "CONNECT_TIMEOUT",
        )
        self.now = _parse_now(
            _opt(args.now, "NOW", datetime.now(timezone.utc).isoformat())
        )


class _ArgumentParser(argparse.ArgumentParser):
    """argparse exits 2 on usage errors by default; this contract is exit 1
    for invalid CLI usage (stdout stays empty)."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _ArgumentParser(
        prog="portfolio_lab_sg01_proof_refresh.py",
        description="Attended one-shot refresh of the sg01 former-authority proof.",
    )
    parser.add_argument("--proof", default=None)
    parser.add_argument("--ssh", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--connect-timeout", default=None)
    parser.add_argument("--now", default=None)
    return parser.parse_args(argv)


# ── probes ────────────────────────────────────────────────────────────────


def probe_argv(cfg: Config, subcommand: str, unit: str) -> list[str]:
    """The only remote interface: explicit argv, batch mode, bounded connect
    timeout, fixed read-only systemctl probe. No shell, no credentials, no
    host-key-check bypass, nothing mutable."""
    return [
        cfg.ssh,
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={int(cfg.connect_timeout)}",
        cfg.host,
        "systemctl", subcommand, unit,
    ]


def _reap_probe(proc: subprocess.Popen[bytes]) -> None:
    """Best-effort kill, reap and stream close; never raises, so a poll/kill
    race can neither leak a child nor leak a pipe fd."""
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        pass
    for stream in (proc.stdout, proc.stderr):
        try:
            stream.close()
        except OSError:
            pass


def run_probe(cfg: Config, subcommand: str, unit: str) -> tuple[int | None, str]:
    """One bounded probe: capped non-blocking reads, hard wall deadline,
    kill-and-reap. Static failure exits only; remote bytes are never echoed."""
    try:
        proc = subprocess.Popen(
            probe_argv(cfg, subcommand, unit),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError:
        die("ssh is not available")
    chunks: list[bytes] = []
    exceeded = False
    timed_out = False
    still_running = False
    try:
        for stream in (proc.stdout, proc.stderr):
            flags = fcntl.fcntl(stream.fileno(), fcntl.F_GETFL)
            fcntl.fcntl(stream.fileno(), fcntl.F_SETFL, flags | os.O_NONBLOCK)
        deadline = time.monotonic() + cfg.connect_timeout + PROBE_WALL_GRACE
        totals = {"out": 0, "err": 0}
        eof = {"out": False, "err": False}
        streams = (("out", proc.stdout), ("err", proc.stderr))
        while not exceeded and not timed_out:
            if time.monotonic() >= deadline:
                timed_out = True
                break
            if proc.poll() is not None and all(eof.values()):
                break
            drained = True
            for name, stream in streams:
                if eof[name]:
                    continue
                try:
                    chunk = os.read(stream.fileno(), 65536)
                except BlockingIOError:
                    continue  # nothing buffered yet
                except OSError as exc:
                    if exc.errno == errno.EINTR:
                        continue
                    chunk = b""  # broken stream: treat as closed
                if chunk:
                    drained = False
                    totals[name] += len(chunk)
                    if name == "out":
                        chunks.append(chunk)
                    if totals[name] > MAX_PROBE_BYTES:
                        exceeded = True
                else:
                    eof[name] = True
            if drained and proc.poll() is None:
                # Idle live child: never busy-spin the poll loop to the deadline.
                time.sleep(0.01)
        still_running = proc.poll() is None
    except OSError:
        _reap_probe(proc)
        die("probe failed")
    _reap_probe(proc)
    if exceeded:
        die("probe output exceeded the bounded read")
    if timed_out or still_running:
        die("probe timed out")
    return proc.returncode, b"".join(chunks).decode("utf-8", "replace")


def probe_is_expected_state(cfg: Config, spec: tuple[str, str, str, int]) -> bool:
    """True only for the canonical (stdout, exit) pair of the desired state."""
    subcommand, unit, expected, expected_exit = spec
    exit_code, stdout = run_probe(cfg, subcommand, unit)
    return exit_code == expected_exit and stdout == f"{expected}\n"


# ── proof write ───────────────────────────────────────────────────────────


def build_proof(cfg: Config) -> bytes:
    """Canonical v1 proof bytes: compact, key-sorted, aware UTC collected_at,
    all four nested booleans false. Deterministic under a pinned ``--now``."""
    payload = {
        "schema": PROOF_SCHEMA,
        "host_label": cfg.host,
        "collected_at": cfg.now.isoformat(),
        "tasker": {"active": False, "enabled": False},
        "archive_timer": {"active": False, "enabled": False},
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def verify_artifact(path_str: str, expected_size: int) -> None:
    """Strict shape check for a proof artifact: regular, non-symlink, owned by
    the running uid, exactly mode 0600, exact size. Raises OSError otherwise,
    so an unverified temp is never renamed into place."""
    st = os.lstat(path_str)
    if (
        stat.S_ISLNK(st.st_mode)
        or not stat.S_ISREG(st.st_mode)
        or st.st_uid != os.getuid()
        or stat.S_IMODE(st.st_mode) != 0o600
        or st.st_size != expected_size
    ):
        raise OSError("proof artifact does not match the required shape")


def replace_proof(path: Path, data: bytes) -> None:
    """Atomic commit: unique temp in the target directory (never created),
    mode 0600 set and verified before the rename, temp removed on any failure,
    so an existing proof survives every unsuccessful write byte-for-byte."""
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".pspr-")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.chmod(tmp_name, 0o600)
        verify_artifact(tmp_name, len(data))
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def verify_committed(path: Path, expected_size: int) -> None:
    """Confirm the committed target. A mismatch is reported statically and
    never deletes: an existing proof artifact is never destroyed here, and the
    collector independently rejects a non-conforming proof."""
    try:
        st = os.lstat(path)
    except OSError:
        die("failed to write proof")
    if (
        stat.S_ISLNK(st.st_mode)
        or not stat.S_ISREG(st.st_mode)
        or st.st_uid != os.getuid()
    ):
        die("failed to write proof")
    if stat.S_IMODE(st.st_mode) != 0o600 or st.st_size != expected_size:
        try:
            os.chmod(path, 0o600)  # safe: regular, owned, non-symlink
            verify_artifact(str(path), expected_size)
        except OSError:
            die("failed to write proof")


# ── main ──────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = Config(args)

    for spec in PROBE_SPECS:
        if not probe_is_expected_state(cfg, spec):
            die("former authority units are not in the expected stopped state")

    data = build_proof(cfg)
    try:
        replace_proof(cfg.proof, data)
    except OSError:
        die("failed to write proof")
    verify_committed(cfg.proof, len(data))

    emit(
        {
            "schema": PROOF_SCHEMA,
            "category": "authority",
            "status": "pass",
            "host_label": cfg.host,
            "collected_at": cfg.now.isoformat(),
            "tasker": {"active": False, "enabled": False},
            "archive_timer": {"active": False, "enabled": False},
        }
    )
    return 0


def _silence_broken_stdout() -> None:
    """Redirect stdout to devnull, but only when stdout is the broken stream:
    shutdown would otherwise re-raise the error ('Exception ignored', exit
    120). A broken stderr must never make this touch a healthy stdout fd."""
    try:
        sys.stdout.flush()
        return
    except BrokenPipeError:
        pass
    except (OSError, ValueError):
        return
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
    except OSError:
        return
    try:
        os.dup2(devnull, sys.stdout.fileno())
    except (OSError, ValueError):
        pass
    finally:
        os.close(devnull)


def _static_stderr(reason: str) -> None:
    """One static line on stderr; a closed stderr cannot be reported on."""
    try:
        print(f"ERROR: {reason}", file=sys.stderr)
    except (OSError, ValueError):
        pass


def run(argv: list[str] | None = None) -> int:
    """Top level for the documented static-error contract: reachable failures
    are one static stderr line, empty stdout, exit 1. SystemExit (including
    die()) and KeyboardInterrupt are never intercepted; only unexpected OS
    errors and a closed reporting pipe are translated here."""
    try:
        return main(argv)
    except BrokenPipeError:
        _silence_broken_stdout()
        _static_stderr("stdout is closed")
        return 1
    except OSError:
        _static_stderr("operation failed")
        return 1


if __name__ == "__main__":
    sys.exit(run())
