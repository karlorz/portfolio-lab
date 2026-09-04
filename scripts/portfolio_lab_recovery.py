#!/usr/bin/env python3
"""Portfolio Lab recovery: create/verify/restore/activate-prod self-contained archives.

Storage-layer encryption is attested by the operator (``--storage-encryption-
attested`` is required on create only); the archive itself is a plaintext tar.
Commands use subprocess argument arrays only (never a shell).

Usage::

  portfolio_lab_recovery.py create --app-dir PATH --web-root PATH \\
      --tasker-service NAME --archive PATH --storage-encryption-attested \\
      [--service-controller systemd|box-persist]
  portfolio_lab_recovery.py verify --archive PATH
  portfolio_lab_recovery.py restore --archive PATH --app-dir PATH --web-root PATH \\
      --target-mode dev|prod [--allow-production-paths] [--start-dev-api] \\
      [--tasker-service NAME] [--service-controller systemd|box-persist]
  portfolio_lab_recovery.py activate-prod --app-dir PATH --web-root PATH \\
      --tasker-service NAME --confirm-authoritative-activation \\
      --former-authority-confirmed-stopped LABEL \\
      [--service-controller systemd|box-persist]

Archive layout (member names)::

  recovery-manifest.json                # embedded self-description, schema .../v2
  source/repository.bundle              # git bundle --all of the source checkout
  source/revision.json                  # recorded source revision + target paths
  runtime/data/...                      # quiesced app data incl. SQLite -wal/-shm
  runtime/logs/research-implement.md    # optional; the only log file archived
  static/web/...                        # served static release + public data
  config/lab-app.env                    # deploy config, only after a secret scan
  metadata/tasker-unit.txt              # tasker systemd unit text (fail closed)
  metadata/tasker-status.json           # live Tasker API capture (fail closed)
  metadata/caddy-portfolio-lab-block.txt  # ONLY the managed Caddy block (fail closed)
  metadata/created.json                 # creation timestamp
  tools/portfolio_lab_recovery.py       # this tool as bootstrap

The only sidecar is ``<archive>.sha256``; archive and sidecar are ``0600``.
Creation validates the source and destination (source service state must be
exactly ``active`` or exactly ``inactive``), refuses symlinked roots (app
data dir and web root), symlinked directories, and unreadable subtrees under
the archived trees (fail closed before the service is quiesced),
drains the tasker service with ``systemctl stop`` only when it is initially
active (always attempting restart in finally; if inactive it is never
started), preflights the bundle, and self-verifies the archive. Verify
reads the uncompressed tar with Python
stdlib ``tarfile`` only: member index (names/types/duplicates/topology) is
validated before any extraction, extraction is manual via ``extractfile()``
into a fresh controlled temp dir, and sidecar/member/schema/digest/path/
type/mode, the git bundle revision, SQLite integrity (stdlib sqlite3), and
static release provenance are all checked before any target is mutated.

Dev restores reject the production paths, never restore ``config/lab-app.env``
(``config_restored: false``), and stay staged unless ``--start-dev-api``
installs the distinct no-scheduler API-only unit (which rejects the archived/
production service name). Prod restores require ``--allow-production-paths``,
stage the full tree in safe staging first (rejecting symlinks/gitlinks in the
bundle checkout), and only replace targets after staging succeeds, keeping
rollback dirs. Production activation re-verifies the original archive, binds
the restore report to it, requires a prod report, matching service identity,
a clean tracked checkout (only the two allowed-dirty generated data files
may differ, after manifest-digest proof), a re-scanned archived unit bound
to the manifest, an inactive target service, explicit confirmations, and
never touches DNS or Caddy.

Command overrides (tests use fakes; production defaults): PLR_GIT, PLR_TAR,
PLR_SYSTEMCTL, PLR_SYSTEMD_UNIT_DIR, PLR_WIKI_DIR, PLR_CADDY_CONFIG,
PLR_DEV_SERVICE_NAME, PLR_BOX_PERSIST_CONTROLLER (default
/home/box/.local/bin/portfolio-lab-box-persist), PLR_BOX_PERSIST_ALLOWED_ROOT
(test-only escape hatch for the box-persist target-root guard). PLR_ALLOW_TMP_DEST (default unset): test-only escape
hatch permitting archive destinations under pytest tmp trees
(/tmp/pytest-of-*); every other /tmp destination stays forbidden.

The optional ``--service-controller`` selection (default ``systemd``, the
unchanged sg01 behavior) makes service operations explicit: ``systemd`` for
the sg01 source create/restart and systemd targets; ``box-persist`` for
cursor-box native processes, valid only for target restore and
activate-prod. create is sg01 source-side and rejects box-persist. All
box-persist commands run the focused controller executable with argv arrays
only (never a shell) and must return the same normalized status JSON
(schema portfolio-lab-box-persist/v1: state active|inactive,
scheduler_mode enabled|disabled, identity_exact bool, scheduler_instances
nonnegative int, pid positive-int|null, plus the requested
service_name/mode/app_dir/web_root) with exit zero. Malformed JSON,
unsupported states, service/mode/app/web identity mismatch, active state
with inexact identity or missing PID, inactive state with a PID, and
inconsistent scheduler counts fail closed; dev candidate start requires
active/disabled/zero-instance/exact/PID, production staging and activation
preflight require inactive/disabled/zero-instance/exact/no-PID, and
activation requires exactly one returned scheduler instance (the explicit
second-scheduler prevention gate).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.error import URLError
from urllib.request import urlopen

SCHEMA_VERSION = "portfolio-lab-recovery/v2"
GENERATION_MATERIALIZATION_SCHEMA = "portfolio-lab-generation-materialization/v1"
GENERATION_LINK_PATH = "data/generations/current"
GENERATION_ARCHIVE_PATH = "runtime/data/generations/current"
DEFAULT_DEV_SERVICE_NAME = "portfolio-lab-tasker-recovery-dev"
BUNDLE_MEMBER = "source/repository.bundle"
REVISION_MEMBER = "source/revision.json"
MANIFEST_MEMBER = "recovery-manifest.json"
CONFIG_MEMBER = "config/lab-app.env"
UNIT_MEMBER = "metadata/tasker-unit.txt"
STATUS_MEMBER = "metadata/tasker-status.json"
CADDY_MEMBER = "metadata/caddy-portfolio-lab-block.txt"
CREATED_MEMBER = "metadata/created.json"
TOOLS_MEMBER = "tools/portfolio_lab_recovery.py"
RESEARCH_MEMBER = "runtime/logs/research-implement.md"
CADDY_BEGIN = "# BEGIN portfolio-lab managed"
CADDY_END = "# END portfolio-lab managed"
SQLITE_SUFFIXES = (".db", ".sqlite", ".sqlite3")

# Dirty-source policy: only ordinary (unstaged, worktree) modifications of
# these tracked data files are allowed; anything else is rejected before the
# tasker service is stopped.
ALLOWED_DIRTY_FILES = frozenset({"data/ensemble_weights.json", "data/vix_term_structure.json"})

# Secrets/identity/agents/Caddy cert state are never archived.
EXCLUDED_DIR_COMPONENTS = frozenset(
    {
        ".git",
        ".hermes",
        ".claude",
        ".grok",
        ".ssh",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        "secrets",
        "credentials",
        "tokens",
        "certs",
    }
)
EXCLUDED_FILE_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".crt")
EXCLUDED_FILE_NAMES = frozenset(
    {
        ".env",
        ".envrc",
        ".netrc",
        ".npmrc",
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "secrets.json",
        "credentials.json",
        "tokens.json",
        "token.json",
        "secret.json",
        "rclone.conf",
        "machine-id",
    }
)
# Config keys that make config/lab-app.env ineligible for archiving.
SECRET_KEY_PATTERN = re.compile(
    r"(secret|token|password|passwd|credential|private|api[-_]?key|access[-_]?key|auth)", re.IGNORECASE
)
# Credential-bearing config values (beyond key names): URL user:password@,
# common secret query parameters, and well-known secret value prefixes.
SECRET_VALUE_PATTERNS = (
    re.compile(r"://[^\s/@:]+:[^\s/@:]+@"),
    re.compile(r"[\?&](?:secret|token|password|passwd|api[_-]?key|key|credential|auth)=[^\s&]+", re.IGNORECASE),
    re.compile(r"^(?:sk-|pk-|ghp_|gho_|xox[baprs]-|AKIA[0-9A-Z]{16}|-----BEGIN )", re.IGNORECASE),
)

REQUIRED_MEMBERS = frozenset(
    {
        MANIFEST_MEMBER,
        BUNDLE_MEMBER,
        REVISION_MEMBER,
        CONFIG_MEMBER,
        UNIT_MEMBER,
        STATUS_MEMBER,
        CADDY_MEMBER,
        CREATED_MEMBER,
        TOOLS_MEMBER,
    }
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _cmd(name: str) -> str:
    return os.environ.get(f"PLR_{name.upper()}", name)


def die(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def run(
    argv: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            env={**os.environ, **env} if env else None,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except FileNotFoundError:
        die(f"command not found: {argv[0]}")
    except subprocess.TimeoutExpired:
        die(f"command timed out: {argv[0]}")
    raise AssertionError("unreachable")


def run_checked(argv: list[str], what: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    res = run(argv, cwd=cwd)
    if res.returncode != 0:
        detail = (res.stderr or res.stdout).strip()
        die(f"{what} failed: {detail or 'exit %d' % res.returncode}")
    return res


def _check_root() -> None:
    # Test fakes (and operator-provided wrappers) override the commands and/or
    # unit dir; real systemctl/systemd writes require root.
    if os.environ.get("PLR_SYSTEMCTL") or os.environ.get("PLR_SYSTEMD_UNIT_DIR"):
        return
    if getattr(os, "geteuid", lambda: 0)() != 0:
        die("run as root for systemctl/systemd operations")


def require_attestation(args: argparse.Namespace) -> None:
    if not args.storage_encryption_attested:
        die("requires --storage-encryption-attested (storage-layer encryption must be attested)")


def valid_full_sha(value: str) -> bool:
    return len(value) == 40 and all(c in "0123456789abcdef" for c in value)


def valid_sha_prefix(value: str) -> bool:
    return 7 <= len(value) <= 40 and all(c in "0123456789abcdef" for c in value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vault_dir() -> Path:
    return Path(
        os.environ.get("PLR_WIKI_DIR", os.environ.get("WIKI_DIR", str(Path.home() / "wiki")))
    ).expanduser()


def forbidden_roots(extra: list[Path]) -> list[Path]:
    roots = [PROJECT_ROOT, Path("/tmp"), _vault_dir(), *extra]
    return sorted({path.resolve() for path in roots})


def check_not_forbidden(path: Path, role: str, extra: list[Path]) -> Path:
    dest = path.resolve()
    for root in forbidden_roots(extra):
        if dest == root or dest.is_relative_to(root):
            if (
                root == Path("/tmp").resolve()
                and os.environ.get("PLR_ALLOW_TMP_DEST") == "1"
                and str(dest).startswith(f"{root}/pytest-of-")
            ):
                continue
            die(f"{role} is forbidden (under {root}); choose an explicit absolute path outside repo/app/web/vault/tmp")
    return dest


def check_archive_destination(path: str, extra: list[Path]) -> Path:
    raw = Path(path)
    if not raw.is_absolute():
        die("archive destination must be an absolute path")
    if any(ord(c) < 32 for c in str(raw)):
        die("archive destination path contains control characters")
    dest = check_not_forbidden(raw, "archive destination", extra)
    if dest.exists():
        die(f"archive already exists: {dest}")
    if not dest.parent.is_dir():
        die(f"archive destination parent directory does not exist: {dest.parent}")
    return dest


def validate_service_name(name: str) -> None:
    if not name or name in (".", "..") or "/" in name or any(ord(c) < 32 for c in name):
        die(f"invalid systemd service name: {name!r}")
    # systemd unit names allow only [A-Za-z0-9:_.-] (plus C-style escapes);
    # '%' is outside that charset and would make every systemctl call fail.
    # No expected unit name (portfolio-lab-tasker, *-recovery-dev) contains
    # '%', so fail closed early instead of at the first systemctl call.
    if "%" in name:
        die(f"invalid systemd service name (unit names cannot contain '%'): {name!r}")


def validate_unit_value(value: str, what: str) -> None:
    if any(ord(c) < 32 for c in value):
        die(f"{what} contains control characters; refusing unit interpolation: {value!r}")
    if "%" in value:
        die(f"{what} contains '%' (systemd specifier); refusing unit interpolation: {value!r}")


# ── service controllers ────────────────────────────────────────────────────

BOX_PERSIST_SCHEMA = "portfolio-lab-box-persist/v1"
DEFAULT_BOX_PERSIST_CONTROLLER = "/home/box/.local/bin/portfolio-lab-box-persist"
BOX_PERSIST_PRODUCTION_ROOT = Path("/home/box/.local/share/portfolio-lab")
NORMALIZED_STATUS_FIELDS = ("state", "scheduler_mode", "identity_exact", "scheduler_instances", "pid")


class ServiceController(Protocol):
    """Explicit service-controller surface used by the recovery CLI.

    ``status`` returns a normalized status with exactly: state
    (active|inactive), scheduler_mode (enabled|disabled), identity_exact
    (bool; must be true before an active state is treated as healthy),
    scheduler_instances (nonnegative int), pid (positive int when active,
    null when inactive)."""

    name: str

    def status(self, service_name: str, mode: str, app_dir: Path, web_root: Path) -> dict[str, Any]: ...
    def stop(self, service_name: str, mode: str, app_dir: Path, web_root: Path) -> Any: ...
    def start(self, service_name: str, mode: str, app_dir: Path, web_root: Path) -> dict[str, Any]: ...


class SystemdController:
    """systemd controller (sg01 source + systemd targets).

    Unit-name identity: ``systemctl is-active`` yields only the normalized
    active/inactive states (failed/unknown fail closed). Status reports
    scheduler mode enabled for the production service and disabled for the
    no-scheduler dev unit; scheduler_instances and pid are not tracked at
    unit level (systemd owns process identity), and recovery gates never
    consult them on systemd paths. stop/start keep the existing bounded
    180-second subprocess ceiling and exact systemctl argv."""

    name = "systemd"

    def status(
        self,
        service_name: str,
        mode: str = "production",
        app_dir: Path | None = None,
        web_root: Path | None = None,
    ) -> dict[str, Any]:
        res = run([_cmd("systemctl"), "is-active", service_name])
        state = res.stdout.strip()
        if res.returncode not in (0, 3):
            die(f"systemctl is-active {service_name} failed (rc={res.returncode}); failing closed")
        if state not in ("active", "inactive"):
            die(
                f"systemctl service {service_name} is in unexpected state {state!r}; "
                "failing closed (expected active or inactive)"
            )
        return {
            "state": state,
            "scheduler_mode": "enabled" if mode == "production" else "disabled",
            "identity_exact": True,
            "scheduler_instances": 0,
            "pid": None,
        }

    def stop(
        self,
        service_name: str,
        mode: str = "production",
        app_dir: Path | None = None,
        web_root: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        res = run([_cmd("systemctl"), "stop", service_name])
        if res.returncode != 0:
            die(f"systemctl stop {service_name} failed: {(res.stderr or res.stdout).strip()}")
        return res

    def start(
        self,
        service_name: str,
        mode: str = "production",
        app_dir: Path | None = None,
        web_root: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        res = run([_cmd("systemctl"), "start", service_name])
        if res.returncode != 0:
            die(f"systemctl start {service_name} failed: {(res.stderr or res.stdout).strip()}")
        return res


def get_service_controller(name: str) -> ServiceController:
    if name == "systemd":
        return SystemdController()
    if name == "box-persist":
        return BoxPersistController()
    die(f"unknown service controller: {name}")
    raise AssertionError("unreachable")


def box_persist_allowed_root() -> Path:
    """Production box-persist target root; PLR_BOX_PERSIST_ALLOWED_ROOT is a
    test-only escape hatch overriding it (absolute when set)."""
    override = os.environ.get("PLR_BOX_PERSIST_ALLOWED_ROOT")
    if override:
        if not os.path.isabs(override):
            die("PLR_BOX_PERSIST_ALLOWED_ROOT must be an absolute path")
        return Path(override).resolve()
    return BOX_PERSIST_PRODUCTION_ROOT


def validate_box_persist_targets(app_dir: Path, web_root: Path) -> None:
    """box-persist restore/activate targets (resolved) must both sit under
    the box-persist allowed root; rejected before any target mutation or
    controller call."""
    root = box_persist_allowed_root()
    for path, role in ((app_dir, "--app-dir"), (web_root, "--web-root")):
        if not path.is_absolute():
            die(f"box-persist {role} must be an absolute path")
        if not path.is_relative_to(root):
            die(f"box-persist {role} must be under {root}; got {path}")


def normalized_status(status: dict[str, Any]) -> dict[str, Any]:
    """The five normalized controller status fields, for JSON reports."""
    return {field: status[field] for field in NORMALIZED_STATUS_FIELDS}


def parse_box_persist_status(
    stdout: str,
    *,
    mode: str,
    app_dir: Path,
    web_root: Path,
    service_name: str,
) -> dict[str, Any]:
    """Validate a box-persist command's stdout as normalized status JSON.

    Rejects malformed JSON, unsupported states, mismatched
    service/mode/app/web identity, active state with inexact identity or
    missing PID, inactive state with a PID, and inconsistent scheduler
    counts (disabled with running instances)."""
    try:
        payload = json.loads(stdout)
    except ValueError:
        die(f"box-persist returned malformed JSON on stdout: {stdout.strip()[:160]!r}; failing closed")
    if not isinstance(payload, dict):
        die("box-persist returned non-object JSON on stdout; failing closed")
    if payload.get("schema") != BOX_PERSIST_SCHEMA:
        die(f"box-persist status schema mismatch: {payload.get('schema')!r}")
    state = payload.get("state")
    if state not in ("active", "inactive"):
        die(f"box-persist status has unsupported state {state!r}; failing closed")
    scheduler_mode = payload.get("scheduler_mode")
    if scheduler_mode not in ("enabled", "disabled"):
        die(f"box-persist status has unsupported scheduler_mode {scheduler_mode!r}; failing closed")
    identity_exact = payload.get("identity_exact")
    if not isinstance(identity_exact, bool):
        die("box-persist status identity_exact must be a boolean; failing closed")
    instances = payload.get("scheduler_instances")
    if isinstance(instances, bool) or not isinstance(instances, int) or instances < 0:
        die("box-persist status scheduler_instances must be a nonnegative integer; failing closed")
    pid = payload.get("pid")
    if pid is not None and (isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0):
        die(f"box-persist status pid must be a positive integer or null: {pid!r}; failing closed")
    if state == "active":
        if pid is None:
            die("box-persist status reports active state without a PID; failing closed")
        if not identity_exact:
            die("box-persist status reports active state with inexact process identity; failing closed")
    elif pid is not None:
        die("box-persist status reports inactive state with a PID; failing closed")
    if scheduler_mode == "disabled" and instances > 0:
        die(
            "box-persist status is inconsistent: scheduler_mode disabled with "
            f"scheduler_instances {instances}; failing closed"
        )
    expected = {
        "service_name": service_name,
        "mode": mode,
        "app_dir": str(app_dir),
        "web_root": str(web_root),
    }
    mismatched = [
        field
        for field, wanted in expected.items()
        if not isinstance(payload.get(field), str) or payload.get(field) != wanted
    ]
    if mismatched:
        shown = ", ".join(
            f"{field}={payload.get(field)!r} (expected {expected[field]!r})" for field in mismatched
        )
        die(f"box-persist status identity mismatch: {shown}; failing closed")
    return payload


class BoxPersistController:
    """box-persist controller for cursor-box native (non-systemd) processes.

    Every command runs the focused controller executable as an argv array
    (never a shell) and must return the same normalized
    portfolio-lab-box-persist/v1 status JSON on stdout with exit zero.
    Nonzero exit, timeout, malformed output, unsupported states, identity
    mismatch, active-without-PID, inactive-with-PID, and inconsistent
    scheduler counts all fail closed. ``start`` maps to ``start-candidate``
    (mode candidate); ``activate`` always uses mode production and forwards
    the former-authority proof label."""

    name = "box-persist"

    def __init__(self, executable: Path | None = None) -> None:
        raw = (
            str(executable)
            if executable is not None
            else os.environ.get("PLR_BOX_PERSIST_CONTROLLER", DEFAULT_BOX_PERSIST_CONTROLLER)
        )
        if not os.path.isabs(raw):
            die(f"box-persist controller path must be absolute: {raw!r}")
        path = Path(raw)
        if not path.is_file():
            die(f"box-persist controller executable not found: {path}")
        if not os.access(path, os.X_OK):
            die(f"box-persist controller executable is not executable: {path}")
        self.executable = path

    def _invoke(
        self,
        action: str,
        *,
        mode: str,
        app_dir: Path,
        web_root: Path,
        service_name: str,
        label: str | None = None,
    ) -> dict[str, Any]:
        argv = [
            str(self.executable),
            action,
            "--mode",
            mode,
            "--app-dir",
            str(app_dir),
            "--web-root",
            str(web_root),
            "--service-name",
            service_name,
        ]
        if label is not None:
            argv += ["--former-authority-confirmed-stopped", label]
        # Timeouts are converted by the shared run() (fail-closed SystemExit),
        # so no controller-level TimeoutExpired handling is reachable here.
        res = run(argv)
        if res.returncode != 0:
            detail = (res.stderr or res.stdout).strip()
            die(f"box-persist {action} failed: {detail or f'exit {res.returncode}'}")
        return parse_box_persist_status(
            res.stdout,
            mode=mode,
            app_dir=app_dir,
            web_root=web_root,
            service_name=service_name,
        )

    def status(self, service_name: str, mode: str, app_dir: Path, web_root: Path) -> dict[str, Any]:
        return self._invoke(
            "status", mode=mode, app_dir=app_dir, web_root=web_root, service_name=service_name
        )

    def stop(self, service_name: str, mode: str, app_dir: Path, web_root: Path) -> dict[str, Any]:
        return self._invoke(
            "stop", mode=mode, app_dir=app_dir, web_root=web_root, service_name=service_name
        )

    def start(self, service_name: str, mode: str, app_dir: Path, web_root: Path) -> dict[str, Any]:
        if mode != "candidate":
            die(f"box-persist start maps to start-candidate and requires --mode candidate; got {mode!r}")
        return self._invoke(
            "start-candidate",
            mode="candidate",
            app_dir=app_dir,
            web_root=web_root,
            service_name=service_name,
        )

    def activate(self, service_name: str, app_dir: Path, web_root: Path, label: str) -> dict[str, Any]:
        return self._invoke(
            "activate",
            mode="production",
            app_dir=app_dir,
            web_root=web_root,
            service_name=service_name,
            label=label,
        )


def require_prod_inactive_status(status: dict[str, Any], service_name: str, context: str) -> None:
    """box-persist production preflight: inactive, scheduler disabled, zero
    scheduler instances, exact identity, no PID."""
    ok = (
        status["state"] == "inactive"
        and status["scheduler_mode"] == "disabled"
        and status["identity_exact"] is True
        and status["scheduler_instances"] == 0
        and status["pid"] is None
    )
    if not ok:
        die(
            f"box-persist {context}: target service {service_name} preflight requires "
            "inactive state with the scheduler disabled, zero scheduler instances, exact "
            "identity, and no PID "
            f"(state={status['state']}, scheduler_mode={status['scheduler_mode']}, "
            f"identity_exact={status['identity_exact']}, "
            f"scheduler_instances={status['scheduler_instances']}, pid={status['pid']})"
        )


def require_candidate_started(status: dict[str, Any], service_name: str) -> None:
    """box-persist dev candidate start: active, scheduler disabled, zero
    scheduler instances, exact identity, positive PID."""
    ok = (
        status["state"] == "active"
        and status["scheduler_mode"] == "disabled"
        and status["identity_exact"] is True
        and status["scheduler_instances"] == 0
        and status["pid"] is not None
    )
    if not ok:
        die(
            f"box-persist start-candidate for {service_name} did not yield a "
            "scheduler-disabled active process "
            f"(state={status['state']}, scheduler_mode={status['scheduler_mode']}, "
            f"identity_exact={status['identity_exact']}, "
            f"scheduler_instances={status['scheduler_instances']}, pid={status['pid']})"
        )


def require_activation_status(status: dict[str, Any], service_name: str) -> None:
    """box-persist activation: exactly one active scheduler (enabled mode,
    exact identity, positive PID, one scheduler instance)."""
    ok = (
        status["state"] == "active"
        and status["scheduler_mode"] == "enabled"
        and status["identity_exact"] is True
        and isinstance(status["pid"], int)
        and status["pid"] > 0
        and status["scheduler_instances"] == 1
    )
    if not ok:
        die(
            f"box-persist activation of {service_name} did not yield exactly one active "
            "scheduler "
            f"(state={status['state']}, scheduler_mode={status['scheduler_mode']}, "
            f"identity_exact={status['identity_exact']}, "
            f"scheduler_instances={status['scheduler_instances']}, pid={status['pid']})"
        )


# ── git + dirty-source policy ───────────────────────────────────────────────


def git_head(repo: Path) -> str:
    res = run([_cmd("git"), "rev-parse", "--verify", "HEAD"], cwd=repo)
    if res.returncode != 0:
        die(f"not a git repository or no commits at {repo}")
    return res.stdout.strip()


def _porcelain_offending(stdout: str) -> list[str]:
    """Classify `git status --porcelain` lines against ALLOWED_DIRTY_FILES.

    Only ordinary (unstaged) modifications of the two allowed-dirty
    generated data files pass; everything else (staged/untracked/renamed/
    deleted entries and malformed lines) is returned as offending. Shared
    by the create and activate-prod gates, which keep their own git
    invocation flags and diagnostics."""
    offending = []
    for line in stdout.splitlines():
        if len(line) < 4:
            offending.append(line)
            continue
        status, path = line[:2], line[3:]
        if not (status == " M" and path in ALLOWED_DIRTY_FILES):
            offending.append(line)
    return offending


def check_clean_enough(repo: Path) -> None:
    res = run([_cmd("git"), "status", "--porcelain"], cwd=repo)
    if res.returncode != 0:
        die(f"git status failed in {repo}: {(res.stderr or res.stdout).strip()}")
    offending = _porcelain_offending(res.stdout)
    if offending:
        die(
            "dirty working tree: only ordinary modifications to data/ensemble_weights.json and "
            f"data/vix_term_structure.json are allowed; offending entries: {offending[:10]}"
        )


def create_bundle(repo: Path, bundle_path: Path) -> None:
    run_checked(
        [_cmd("git"), "bundle", "create", str(bundle_path), "--all"],
        "git bundle create",
        cwd=repo,
    )
    run_checked(
        [_cmd("git"), "bundle", "verify", str(bundle_path)],
        "git bundle verify",
        cwd=repo,
    )


# ── member selection (secrets excluded, symlinks never archived) ────────────


def is_excluded(rel: str) -> bool:
    parts = [part.lower() for part in rel.split("/")]
    name = parts[-1]
    if any(part in EXCLUDED_DIR_COMPONENTS for part in parts):
        return True
    if name in EXCLUDED_FILE_NAMES or name.startswith(".env."):
        return True
    if "secret" in name or "token" in name or "credential" in name:
        return True
    if "session" in name and ("broker" in name or "alpaca" in name):
        return True
    if name.endswith(EXCLUDED_FILE_SUFFIXES):
        return True
    return False


def is_control_char(c: str) -> bool:
    """Check if character is ASCII C0 control (<32), ASCII DEL (0x7F), or C1 control (0x80-0x9F)."""
    o = ord(c)
    return o < 32 or o == 0x7F or (0x80 <= o <= 0x9F)


def is_safe_original_link(raw_target: str) -> bool:
    """Validate raw original link text according to generation link source contract:
    relative, non-empty UTF-8-safe path string with no absolute root, backslash,
    control characters, empty component, '.' component, or '..' component."""
    if not isinstance(raw_target, str) or not raw_target:
        return False
    try:
        raw_target.encode("utf-8")
    except UnicodeEncodeError:
        return False
    if raw_target.startswith("/") or "\\" in raw_target:
        return False
    if any(is_control_char(c) for c in raw_target):
        return False
    parts = raw_target.split("/")
    if any(p in ("", ".", "..") for p in parts):
        return False
    if ":" in parts[0]:
        return False
    return True


def inspect_generation_target(
    generations_dir: Path,
    raw_link: str,
) -> tuple[Path, str, list[dict[str, Any]]]:
    """Validate generations/current link and inspect target tree according to contract.

    Returns (resolved_target_path, target_rel_to_generations, snapshot_entries).
    Fails closed on any contract violation with bounded, secret-safe diagnostics."""
    if not is_safe_original_link(raw_link):
        die("invalid generations/current symlink target text")

    # Target cannot be 'current' or any path resolving to the 'current' symlink itself
    if raw_link == "current" or Path(raw_link).parts[0] == "current":
        die("generations/current target cannot be current symlink itself")

    try:
        gen_dir_resolved = generations_dir.resolve(strict=True)
    except OSError:
        die("cannot resolve generations directory")

    try:
        raw_target_path = generations_dir / raw_link
        target_path = raw_target_path.resolve(strict=True)
    except OSError:
        die("generations/current target unresolvable or broken link")

    # Must remain strictly beneath resolved generations_dir; not generations_dir itself or current
    if not target_path.is_relative_to(gen_dir_resolved):
        die("generations/current target escapes generations directory")
    if target_path == gen_dir_resolved:
        die("generations/current target cannot be generations directory itself")
    if target_path == (generations_dir / "current"):
        die("generations/current target cannot be current symlink itself")

    # Every component from generations_dir to target must be ordinary and non-symlink
    rel_parts = Path(raw_link).parts
    curr = generations_dir
    for part in rel_parts:
        curr = curr / part
        try:
            if curr.is_symlink():
                if curr != target_path or target_path.is_symlink():
                    die("symlink in generations target path component")
            if not curr.exists():
                die("generations target path component does not exist")
        except OSError:
            die("generations target path component inaccessible")

    if not target_path.is_dir() or target_path.is_symlink():
        die("generations target is not an ordinary directory")

    # Inspect target tree
    target_rel = target_path.relative_to(gen_dir_resolved).as_posix()
    entries: list[dict[str, Any]] = []

    def _onerror(error: OSError) -> None:
        die("cannot traverse generation target")

    try:
        for dirpath, dirnames, filenames in os.walk(
            target_path, followlinks=False, onerror=_onerror
        ):
            dp = Path(dirpath)
            for dname in sorted(dirnames):
                try:
                    dname.encode("utf-8")
                except UnicodeEncodeError:
                    die("unsafe relative name in generation target")
                candidate = dp / dname
                try:
                    if candidate.is_symlink():
                        die("nested directory symlink in generation target")
                except OSError:
                    die("cannot inspect directory in generation target")
            for fname in sorted(filenames):
                try:
                    fname.encode("utf-8")
                except UnicodeEncodeError:
                    die("unsafe relative name in generation target")
                p = dp / fname
                try:
                    if p.is_symlink():
                        die("nested symlink in generation target")
                    st = p.lstat()
                except OSError:
                    die("cannot stat file in generation target")
                if stat.S_ISDIR(st.st_mode):
                    continue
                if not stat.S_ISREG(st.st_mode):
                    die("non-regular file in generation target")
                if st.st_nlink != 1:
                    die("hardlinked regular file in generation target")
                mode = st.st_mode & 0o7777
                if mode & 0o4000:
                    die("setuid mode in generation target")
                if mode & 0o2000:
                    die("setgid mode in generation target")
                if mode & 0o1000:
                    die("sticky mode in generation target")
                try:
                    rel_file = p.relative_to(target_path).as_posix()
                    rel_file.encode("utf-8")
                except (UnicodeEncodeError, ValueError):
                    die("unsafe relative name in generation target")
                if any(is_control_char(c) for c in rel_file):
                    die("control character in generation file path")
                if not path_safe(rel_file):
                    die("unsafe relative name in generation target")
                # Excluded secret-like path check
                if is_excluded(f"generations/{target_rel}/{rel_file}"):
                    die("generation target contains excluded secret-like path")
                digest = sha256_file(p)
                entries.append({
                    "rel_path": rel_file,
                    "sha256": digest,
                    "bytes": st.st_size,
                    "mode": mode,
                })
    except OSError:
        die("cannot traverse generation target")

    if not entries:
        die("generation target directory is empty")

    return target_path, target_rel, sorted(entries, key=lambda e: e["rel_path"])


def validate_generation_snapshot(
    generations_dir: Path,
    expected_link: str,
    expected_target: Path,
    expected_entries: list[dict[str, Any]],
) -> None:
    """Validate that current symlink and target tree exactly match pre-stop snapshot."""
    current_link = generations_dir / "current"
    if not current_link.is_symlink():
        die("generations/current changed from symlink during quiesce")
    try:
        raw_link = os.readlink(current_link)
    except OSError:
        die("cannot read generations/current symlink after stop")
    if raw_link != expected_link:
        die("generations/current link changed after stop")
    target_path, _, entries = inspect_generation_target(generations_dir, raw_link)
    if target_path != expected_target:
        die("generations target path changed after stop")
    if entries != expected_entries:
        die("generations target snapshot changed after stop")


def _walk_tree_files(
    root: Path,
    prefix: str,
    allowed_dir_symlink: Path | None = None,
):
    """Yield regular non-symlink files under ``root`` without ever descending
    into symlinked directories; fail closed on any symlinked directory or
    traversal error encountered — unless allowed_dir_symlink is explicitly
    permitted (checked by identity)."""

    def _onerror(error: OSError) -> None:
        die(f"refusing to archive: cannot traverse {prefix}: {error}")

    try:
        for dirpath, dirnames, filenames in os.walk(
            root, followlinks=False, onerror=_onerror
        ):
            dp = Path(dirpath)
            for name in sorted(dirnames):
                candidate = dp / name
                if candidate.is_symlink():
                    if allowed_dir_symlink is not None and candidate.resolve() == allowed_dir_symlink.resolve() and candidate == allowed_dir_symlink:
                        # Permitted directory symlink; do not descend into it via os.walk
                        continue
                    die(f"refusing to archive: symlinked directory {candidate} under {prefix}")
            for name in sorted(filenames):
                path = dp / name
                if path.is_symlink():
                    if allowed_dir_symlink is not None and path == allowed_dir_symlink:
                        continue
                    continue
                if not path.is_file():
                    continue
                yield path
    except OSError as error:
        die(f"refusing to archive: cannot traverse {prefix}: {error}")


def iter_tree_files(
    root: Path,
    prefix: str,
    allowed_dir_symlink: Path | None = None,
) -> list[str]:
    members = []
    for path in _walk_tree_files(root, prefix, allowed_dir_symlink=allowed_dir_symlink):
        rel = path.relative_to(root).as_posix()
        if is_excluded(rel):
            continue
        if any(ord(c) < 32 for c in rel):
            die(f"refusing to archive file with control characters in its name: {path}")
        if not path_safe(rel):
            die(f"refusing to archive file with unsafe relative name: {path}")
        members.append(f"{prefix}/{rel}")
    return sorted(members)


def check_no_symlinked_dirs(
    root: Path,
    prefix: str,
    allowed_dir_symlink: Path | None = None,
) -> None:
    """Pre-quiesce preflight: refuse to proceed if any symlinked directory
    sits under an archived tree (except allowed_dir_symlink if provided)."""
    for _ in _walk_tree_files(root, prefix, allowed_dir_symlink=allowed_dir_symlink):
        pass


def copy_members_to_staging(members: list[str], src_root: Path, staging: Path, prefix: str) -> None:
    for member in members:
        rel = Path(member[len(prefix) + 1 :])
        source = src_root / rel
        target = staging / member
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def member_entries(members: list[str], staging: Path) -> list[dict[str, Any]]:
    entries = []
    for member in members:
        path = staging / member
        st = path.stat()
        mode = st.st_mode & 0o7777
        if mode & 0o7000:
            die(f"refusing to archive file with setuid/setgid/sticky mode: {path} ({oct(mode)})")
        entries.append({"path": member, "sha256": sha256_file(path), "bytes": st.st_size, "mode": mode})
    return entries


def validate_staged_generation_trees(
    staging: Path,
    expected_link: str,
    expected_entries: list[dict[str, Any]],
) -> None:
    """Validate both staged target tree and staged materialized-current tree
    against snapshot entries and against each other before tar creation."""
    target_staged_root = staging / "runtime" / "data" / "generations" / expected_link
    current_staged_root = staging / GENERATION_ARCHIVE_PATH

    try:
        for root, name in ((target_staged_root, "target"), (current_staged_root, "current duplicate")):
            if not root.is_dir() or root.is_symlink():
                die(f"staged {name} is not an ordinary directory")
            for p in root.rglob("*"):
                if p.is_symlink():
                    die(f"staged {name} contains symlink")
                st = p.lstat()
                if stat.S_ISDIR(st.st_mode):
                    continue
                if not stat.S_ISREG(st.st_mode):
                    die(f"staged {name} contains non-regular file")

        target_files = {
            p.relative_to(target_staged_root).as_posix(): p
            for p in target_staged_root.rglob("*")
            if p.is_file()
        }
        current_files = {
            p.relative_to(current_staged_root).as_posix(): p
            for p in current_staged_root.rglob("*")
            if p.is_file()
        }

        expected_map = {e["rel_path"]: e for e in expected_entries}

        if set(target_files.keys()) != set(expected_map.keys()):
            die("staged generation target files do not match snapshot")
        if set(current_files.keys()) != set(expected_map.keys()):
            die("staged generation current duplicate files do not match snapshot")

        for rel_f, exp in expected_map.items():
            t_p = target_files[rel_f]
            c_p = current_files[rel_f]
            t_st = t_p.stat()
            c_st = c_p.stat()
            t_sha = sha256_file(t_p)
            c_sha = sha256_file(c_p)

            if t_sha != exp["sha256"] or t_st.st_size != exp["bytes"] or (t_st.st_mode & 0o7777) != exp["mode"]:
                die("staged generation target file does not match snapshot")
            if c_sha != exp["sha256"] or c_st.st_size != exp["bytes"] or (c_st.st_mode & 0o7777) != exp["mode"]:
                die("staged generation current duplicate does not match snapshot")
    except OSError:
        die("cannot validate staged generation trees")


def parse_and_validate_generation_metadata(
    gen_mat: Any,
) -> tuple[bool, str | None, str | None]:
    """Validate generation materialization metadata shape and content.
    Returns (ok, original_link, error_reason)."""
    if not isinstance(gen_mat, dict):
        return False, None, "generation_materialization is not an object"

    # Exactly five keys required
    expected_keys = {
        "schema_version",
        "link_path",
        "original_link",
        "target_path",
        "archive_path",
    }
    if set(gen_mat.keys()) != expected_keys:
        return False, None, "generation_materialization keys mismatch"

    if gen_mat.get("schema_version") != GENERATION_MATERIALIZATION_SCHEMA:
        return False, None, "generation_materialization schema mismatch"
    if gen_mat.get("link_path") != GENERATION_LINK_PATH:
        return False, None, "generation_materialization link_path mismatch"
    if gen_mat.get("archive_path") != GENERATION_ARCHIVE_PATH:
        return False, None, "generation_materialization archive_path mismatch"

    orig_link = gen_mat.get("original_link")
    if not isinstance(orig_link, str) or not is_safe_original_link(orig_link):
        return False, None, "generation_materialization original_link is unsafe"
    if orig_link == "current" or Path(orig_link).parts[0] == "current":
        return False, None, "generation_materialization original_link cannot target current"

    expected_target_path = f"data/generations/{orig_link}"
    if gen_mat.get("target_path") != expected_target_path:
        return False, None, "generation_materialization target_path mismatch"

    return True, orig_link, None


def _write_staging_text(staging: Path, member: str, text: str) -> None:
    target = staging / member
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _write_staging_json(staging: Path, member: str, payload: dict[str, Any]) -> None:
    target = staging / member
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ── metadata capture (fail closed) ──────────────────────────────────────────


def read_required_text(path: Path, what: str) -> str:
    try:
        if not path.is_file():
            raise OSError("not a file")
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        die(f"{what} unavailable at {path} ({exc}); failing closed")


def extract_caddy_managed_block(config: Path) -> str | None:
    if not config.is_file():
        return None
    text = config.read_text(encoding="utf-8", errors="replace")
    start = text.find(CADDY_BEGIN)
    end = text.find(CADDY_END, start)
    if start == -1 or end == -1:
        return None
    return text[start : end + len(CADDY_END)]


def check_text_secrets(text: str, what: str) -> None:
    bad = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, eq, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        candidate_keys = [key]
        candidate_values = [value]
        if key.casefold() == "environment":
            nested_key, nested_eq, nested_value = value.strip("\"'").partition("=")
            if nested_eq:
                candidate_keys.append(nested_key.strip())
                candidate_values.append(nested_value.strip())
        if any(SECRET_KEY_PATTERN.search(candidate) for candidate in candidate_keys):
            bad.append(f"key {candidate_keys[-1]!r}")
        elif any(
            v and any(pattern.search(v) for pattern in SECRET_VALUE_PATTERNS)
            for v in candidate_values
        ):
            bad.append(f"value of {key!r}")
    if bad:
        die(f"{what} contains secret-like key(s)/value(s): {', '.join(bad[:5])}; refusing to archive secrets")


def check_config_secrets(config_file: Path) -> None:
    try:
        text = config_file.read_text(encoding="utf-8")
    except OSError:
        die(f"config file unreadable: {config_file}")
    check_text_secrets(text, "config/lab-app.env")


def parse_env_values(config_file: Path) -> dict[str, str]:
    """Read simple non-secret KEY=VALUE deployment settings without sourcing them."""

    try:
        lines = config_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        die(f"config file unreadable: {config_file}")
    values: dict[str, str] = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            values[key] = value.strip()
    return values


def capture_live_tasker_status(config_file: Path) -> str:
    """Return the source Tasker status API response before service quiescing."""

    values = parse_env_values(config_file)
    url = os.environ.get("PLR_TASKER_STATUS_URL")
    if not url:
        host = values.get("TASKER_HOST", "127.0.0.1")
        port = values.get("TASKER_PORT", "8000")
        url = f"http://{host}:{port}/api/tasker/status"
    try:
        with urlopen(url, timeout=5) as response:  # noqa: S310 -- explicit loopback/configured Tasker API
            if response.status != 200:
                die(f"source Tasker API returned HTTP {response.status}: {url}")
            body = response.read()
    except (OSError, URLError) as exc:
        die(f"source Tasker API unavailable at {url}: {exc}")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        die(f"source Tasker API returned invalid JSON at {url}: {exc}")
    if not isinstance(payload, dict):
        die(f"source Tasker API returned non-object JSON at {url}")
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _read_json_string_field(path: Path, field: str) -> str | None:
    """Read a non-empty string field from a JSON file, or None.

    Shared by the create/verify/activate paths: unreadable or malformed
    JSON and non-string or empty values all yield None; callers apply
    their own fail-closed validation with their own error messages.
    Valid non-object JSON keeps the legacy behavior: the unguarded
    ``.get()`` raises AttributeError (only OSError/ValueError are caught)."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = payload.get(field)
    return value if isinstance(value, str) and value else None


def data_index_generator_sha(web_root: Path) -> str | None:
    return _read_json_string_field(web_root / "data" / "index.json", "generator_git_sha")


# ── safe tar reading (stdlib tarfile; never extract* on untrusted archives) ─


def path_safe(name: str) -> bool:
    if name.startswith("/") or name.startswith("./") or "\\" in name:
        return False
    parts = name.split("/")
    if any(part in ("", "..", ".") for part in parts):
        return False
    if ":" in parts[0]:
        return False
    return True


def allowed_member(name: str) -> bool:
    if name in REQUIRED_MEMBERS or name == RESEARCH_MEMBER:
        return True
    return name.startswith("runtime/data/") or name.startswith("static/web/")


def read_archive_members(archive: Path) -> tuple[list[tarfile.TarInfo], list[str]]:
    """Read the uncompressed tar member index and validate it fully.

    Rejects non-regular member types (symlink/hardlink/dir/device/FIFO/
    sparse/unknown), unsafe names (absolute, ``./``, dot/dotdot/empty
    components, backslashes, drive-like roots, control characters),
    duplicates, and path-topology collisions (``x`` and ``x/y``). Nothing
    is extracted here; extraction happens only after this returns."""
    try:
        with tarfile.open(str(archive), "r:") as tf:
            infos = list(tf)
    except (tarfile.TarError, OSError, ValueError) as exc:
        raise ValueError(f"tar read failed: {exc}") from exc
    names = [info.name for info in infos]
    problems: list[str] = []
    for info in infos:
        name = info.name
        if not path_safe(name):
            problems.append(f"{name!r} (unsafe path)")
            continue
        if any(ord(c) < 32 for c in name):
            problems.append(f"{name!r} (control characters)")
            continue
        if name.rsplit("/", 1)[-1].startswith("._"):
            problems.append(f"{name!r} (AppleDouble metadata member)")
            continue
        if info.type not in (tarfile.REGTYPE, tarfile.AREGTYPE) or info.issparse() or info.linkname:
            problems.append(f"{name!r} (member type {info.type})")
    if len(set(names)) != len(names):
        problems.append("duplicate member names")
    for a, b in zip(sorted(names), sorted(names)[1:]):
        if b.startswith(a + "/"):
            problems.append(f"path topology collision: {a!r} and {b!r}")
            break
    if problems:
        raise ValueError("unsafe archive members: " + "; ".join(problems[:8]))
    return infos, names


def extract_validated_members(archive: Path, infos: list[tarfile.TarInfo], dest: Path) -> None:
    """Manual extraction of pre-validated regular members.

    Never uses ``TarFile.extract*``. All members were validated by
    ``read_archive_members`` (regular files, safe names, no duplicates, no
    topology collisions), so extraction cannot escape ``dest``. Files are
    written with restrictive ``0600``; validated modes are applied later by
    the caller."""
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(str(archive), "r:") as tf:
            for info in infos:
                src = tf.extractfile(info)
                if src is None:
                    raise OSError(f"cannot read member {info.name}")
                target = dest / info.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
                os.chmod(target, 0o600)
    except (tarfile.TarError, OSError, ValueError) as exc:
        raise ValueError(f"archive extraction failed: {exc}") from exc


def sqlite_integrity_ok(db_path: Path) -> bool:
    import sqlite3

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=rw", uri=True, timeout=10)
    except sqlite3.Error:
        return False
    try:
        try:
            row = conn.execute("PRAGMA integrity_check;").fetchone()
        except sqlite3.Error:
            return False
        return bool(row) and row[0] == "ok"
    finally:
        conn.close()


# ── verification (before any target mutation) ───────────────────────────────


def verify_archive(archive: Path, sidecar: Path) -> tuple[bool, dict[str, Any]]:
    checks: dict[str, Any] = {
        "sidecar_ok": False,
        "archive_type": "unknown",
        "schema_ok": False,
        "path_safe": False,
        "members_match": False,
        "allowed_members": False,
        "required_members": False,
        "digests_match": False,
        "modes_match": False,
        "bundle_ok": False,
        "bundle_revision_matches": False,
        "revision_consistent": False,
        "sqlite_ok": None,
        "static_provenance_coherent": None,
        "static_provenance_note": None,
        "data_index_generator_sha": None,
        "data_index_generator_reachable": None,
        "generation_materialization_metadata_ok": None,
        "generation_materialization_members_match": None,
    }
    report: dict[str, Any] = {
        "command": "verify",
        "schema_version": SCHEMA_VERSION,
        "archive": str(archive),
        "sidecar": str(sidecar),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sha": None,
        "source_tasker_service": None,
        "archive_sha256": None,
        "manifest_sha256": None,
        "checks": checks,
        "ok": False,
    }

    def fail(reason: str) -> tuple[bool, dict[str, Any]]:
        report["error"] = reason
        return False, report

    if not archive.is_file():
        return fail(f"archive file missing: {archive}")
    if not sidecar.is_file():
        return fail(f"sidecar missing: {sidecar}")
    # Strict sidecar grammar: exactly "<64 lowercase hex>  <archive basename>".
    try:
        raw = sidecar.read_text(encoding="utf-8")
    except OSError as exc:
        return fail(f"sidecar unreadable: {exc}")
    if not raw.endswith("\n") or "\n" in raw[:-1]:
        return fail("sidecar digest/format invalid (must be exactly one line): " + str(sidecar))
    hex_token, sep, name_token = raw[:-1].partition("  ")
    if (
        sep != "  "
        or len(hex_token) != 64
        or any(c not in "0123456789abcdef" for c in hex_token)
        or name_token != archive.name
    ):
        return fail(f"sidecar digest/format invalid (expected '<sha256>  {archive.name}'): {sidecar}")
    if hex_token != sha256_file(archive):
        return fail("archive digest mismatch")
    checks["sidecar_ok"] = True

    try:
        infos, names = read_archive_members(archive)
    except ValueError as exc:
        checks["path_safe"] = False
        return fail(str(exc))
    if not names:
        return fail("archive has no members")
    checks["archive_type"] = "tar"
    checks["path_safe"] = True
    report["archive_sha256"] = hex_token

    tmp = Path(tempfile.mkdtemp(prefix="pl-recovery-verify-"))
    try:
        # `git bundle verify` requires a repository context; init the
        # throwaway dir before extraction so a hostile ".git" member cannot
        # pre-create one (extraction then fails closed).
        init_res = run([_cmd("git"), "init", "-q", str(tmp)])
        if init_res.returncode != 0:
            return fail("git init failed for bundle verification context")
        try:
            extract_validated_members(archive, infos, tmp)
        except ValueError as exc:
            return fail(str(exc))

        manifest_path = tmp / MANIFEST_MEMBER
        if not manifest_path.is_file():
            return fail(f"embedded manifest missing: {MANIFEST_MEMBER}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return fail("embedded manifest is not valid JSON")
        if manifest.get("schema_version") != SCHEMA_VERSION:
            return fail(f"embedded manifest schema mismatch: {manifest.get('schema_version')!r}")
        checks["schema_ok"] = True
        report["manifest_sha256"] = sha256_file(manifest_path)

        members = manifest.get("members")
        if not isinstance(members, list):
            return fail("embedded manifest members missing")
        member_paths = [str(entry.get("path")) for entry in members]
        if sorted(names) != sorted([*member_paths, MANIFEST_MEMBER]):
            checks["members_match"] = False
            return fail("archive members do not match embedded manifest")
        checks["members_match"] = True

        bad = [name for name in member_paths if not allowed_member(name)]
        if bad:
            checks["allowed_members"] = False
            return fail("disallowed member in archive: " + "; ".join(bad[:5]))
        checks["allowed_members"] = True

        member_set = set(member_paths)
        missing = sorted((REQUIRED_MEMBERS - {MANIFEST_MEMBER}) - member_set)
        if missing:
            checks["required_members"] = False
            return fail("required member missing: " + ", ".join(missing))
        if not any(name.startswith("runtime/data/") for name in member_paths):
            checks["required_members"] = False
            return fail("archive has no runtime/data members")
        if not any(name.startswith("static/web/") for name in member_paths):
            checks["required_members"] = False
            return fail("archive has no static/web members")
        checks["required_members"] = True

        infos_by_name = {info.name: info for info in infos}
        digests_ok = True
        modes_ok = True
        for entry in members:
            member = str(entry.get("path"))
            path = tmp / member
            sha = entry.get("sha256")
            size = entry.get("bytes")
            mode = entry.get("mode")
            if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                return fail(f"malformed manifest entry digest: {member}")
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                return fail(f"malformed manifest entry size: {member}")
            if not isinstance(mode, int) or isinstance(mode, bool) or mode < 0 or mode > 0o7777 or mode & 0o7000:
                return fail(f"unsafe recorded member mode: {member} ({mode!r})")
            header = infos_by_name.get(member)
            if header is None or (header.mode & 0o7777) != mode:
                modes_ok = False
            if not path.is_file() or path.is_symlink():
                return fail(f"member missing or not a regular file after extract: {member}")
            os.chmod(path, mode)
            if sha256_file(path) != sha or path.stat().st_size != size:
                digests_ok = False
                break
            if path.stat().st_mode & 0o7777 != mode:
                modes_ok = False
        if not digests_ok:
            checks["digests_match"] = False
            return fail("member digest/size mismatch (archive tampered)")
        checks["digests_match"] = True
        checks["modes_match"] = modes_ok

        bundle = tmp / BUNDLE_MEMBER
        bundle_res = run([_cmd("git"), "bundle", "verify", str(bundle)], cwd=tmp)
        checks["bundle_ok"] = bundle_res.returncode == 0
        if not checks["bundle_ok"]:
            return fail("git bundle verify failed: " + (bundle_res.stderr or bundle_res.stdout).strip())

        source = manifest.get("source") or {}
        source_sha = source.get("sha")
        source_service = source.get("tasker_service")
        if not isinstance(source_sha, str) or not valid_full_sha(source_sha):
            return fail(f"recorded source revision is not a valid 40-hex sha: {source_sha!r}")
        report["source_sha"] = source_sha
        report["source_tasker_service"] = source_service if isinstance(source_service, str) else None
        heads = run([_cmd("git"), "bundle", "list-heads", str(bundle)], cwd=tmp)
        checks["bundle_revision_matches"] = source_sha in heads.stdout.split()
        if not checks["bundle_revision_matches"]:
            return fail("bundle missing recorded source revision")

        rev_ok = True
        try:
            revision = json.loads((tmp / REVISION_MEMBER).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            revision = {}
        if revision.get("source_sha") != source_sha:
            rev_ok = False
        if revision.get("app_dir") != source.get("app_dir") or revision.get("web_root") != source.get("web_root"):
            rev_ok = False
        try:
            created = json.loads((tmp / CREATED_MEMBER).read_text(encoding="utf-8"))
            if created.get("schema_version") != SCHEMA_VERSION:
                rev_ok = False
        except (OSError, ValueError):
            rev_ok = False
        checks["revision_consistent"] = rev_ok
        if not rev_ok:
            return fail("revision/metadata inconsistent with manifest")

        db_members = [
            name
            for name in member_paths
            if name.startswith("runtime/data/") and Path(name).suffix in SQLITE_SUFFIXES
        ]
        if db_members:
            checks["sqlite_ok"] = True
            for member in db_members:
                if not sqlite_integrity_ok(tmp / member):
                    checks["sqlite_ok"] = False
                    return fail(f"sqlite integrity failed: {member}")
        else:
            checks["sqlite_ok"] = None

        release = tmp / "static" / "web" / "_release.json"
        release_sha = _read_json_string_field(release, "source_git_sha")
        coherent = bool(release_sha) and release_sha == source_sha
        checks["static_provenance_coherent"] = coherent
        checks["static_provenance_note"] = (
            "ok"
            if coherent
            else f"_release source_git_sha {release_sha!r} != archived source {source_sha!r}"
        )

        generator_sha = _read_json_string_field(
            tmp / "static" / "web" / "data" / "index.json", "generator_git_sha"
        )
        checks["data_index_generator_sha"] = generator_sha
        if generator_sha is None:
            checks["data_index_generator_reachable"] = None
        elif not valid_sha_prefix(generator_sha):
            checks["data_index_generator_reachable"] = False
        elif generator_sha == source_sha:
            checks["data_index_generator_reachable"] = True
        else:
            clone_dir = tmp / "clone"
            # --no-checkout: only the object graph is needed for rev-list --all,
            # so skip materializing a worktree.
            clone_res = run([_cmd("git"), "clone", "--no-checkout", str(bundle), str(clone_dir)])
            if clone_res.returncode != 0:
                checks["data_index_generator_reachable"] = False
            else:
                revs = run([_cmd("git"), "-C", str(clone_dir), "rev-list", "--all"])
                if revs.returncode != 0:
                    checks["data_index_generator_reachable"] = False
                else:
                    matches = [rev for rev in revs.stdout.split() if rev.startswith(generator_sha)]
                    checks["data_index_generator_reachable"] = len(matches) == 1

        # Optional generation link materialization validation
        gen_mat = manifest.get("generation_materialization")
        if gen_mat is not None:
            mat_ok, orig_link, mat_err = parse_and_validate_generation_metadata(gen_mat)
            if not mat_ok:
                checks["generation_materialization_metadata_ok"] = False
                return fail(mat_err or "generation_materialization metadata invalid")
            assert orig_link is not None
            checks["generation_materialization_metadata_ok"] = True

            # Member sets validation
            target_dir_member = f"runtime/data/generations/{orig_link}"
            curr_dir_member = GENERATION_ARCHIVE_PATH
            target_prefix = f"runtime/data/generations/{orig_link}/"
            curr_prefix = f"{GENERATION_ARCHIVE_PATH}/"

            entries_by_path = {str(entry.get("path")): entry for entry in members}
            if target_dir_member in entries_by_path:
                checks["generation_materialization_members_match"] = False
                return fail("regular member exists at generation target directory")
            if curr_dir_member in entries_by_path:
                checks["generation_materialization_members_match"] = False
                return fail("regular member exists at generation current archive path")

            target_files = {p[len(target_prefix):]: p for p in entries_by_path if p.startswith(target_prefix)}
            curr_files = {p[len(curr_prefix):]: p for p in entries_by_path if p.startswith(curr_prefix)}

            if not target_files or not curr_files:
                checks["generation_materialization_members_match"] = False
                return fail("generation_materialization member sets must be non-empty")

            if set(target_files.keys()) != set(curr_files.keys()):
                checks["generation_materialization_members_match"] = False
                return fail("generation_materialization target and current members relative paths do not match")

            pairs_ok = True
            for rel_file, tpath in target_files.items():
                cpath = curr_files[rel_file]
                t_entry = entries_by_path[tpath]
                c_entry = entries_by_path[cpath]
                if (
                    t_entry.get("sha256") != c_entry.get("sha256")
                    or t_entry.get("bytes") != c_entry.get("bytes")
                    or t_entry.get("mode") != c_entry.get("mode")
                ):
                    pairs_ok = False
                    break

            checks["generation_materialization_members_match"] = pairs_ok
            if not pairs_ok:
                return fail("generation_materialization member metadata mismatch between target and current")
        else:
            checks["generation_materialization_metadata_ok"] = None
            checks["generation_materialization_members_match"] = None

        if not modes_ok:
            return fail("member mode mismatch (archive tampered)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    report["ok"] = True
    return True, report


def print_failed_verify(report: dict[str, Any], command: str) -> None:
    out = dict(report)
    out["command"] = command
    out["ok"] = False
    print(json.dumps(out, indent=2, sort_keys=True))
    die("archive verification failed: " + str(report.get("error", "unknown")))


# ── create ──────────────────────────────────────────────────────────────────


def cmd_create(args: argparse.Namespace) -> int:
    # create is sg01 source-side and supports only systemd: an explicit
    # box-persist selection fails before bundle creation, any controller
    # call, source stop, or archive write.
    if args.service_controller != "systemd":
        die(
            "create is sg01 source-side and supports only --service-controller systemd; "
            "box-persist is invalid for create"
        )
    require_attestation(args)
    if not str(args.archive).endswith(".portfolio-lab-recovery.tar"):
        die("archive path must end with .portfolio-lab-recovery.tar")
    raw_app_dir = Path(args.app_dir)
    raw_web_root = Path(args.web_root)
    # Reject raw root symlinks before resolve(): resolve() would silently
    # follow a symlinked root and collect outside the intended tree.
    if (raw_app_dir / "data").is_symlink():
        die(f"refusing to archive: app data directory is a symlink: {raw_app_dir / 'data'}")
    if raw_web_root.is_symlink():
        die(f"refusing to archive: web root is a symlink: {raw_web_root}")
    source = raw_app_dir.resolve()
    web_root = raw_web_root.resolve()
    data_dir = (source / "data").resolve()
    config_file = (source / "config" / "lab-app.env").resolve()
    archive = check_archive_destination(args.archive, [source, web_root])

    if not (source / ".git").is_dir():
        die(f"app dir is not a git repository: {source}")
    if not data_dir.is_dir():
        die(f"app data directory missing: {data_dir}")
    if not web_root.is_dir():
        die(f"web root missing: {web_root}")
    if not config_file.is_file():
        die(f"config file missing: {config_file}")
    source_sha = git_head(source)
    if not valid_full_sha(source_sha):
        die(f"source HEAD is not a valid 40-hex sha: {source_sha!r}")
    branch = run([_cmd("git"), "rev-parse", "--abbrev-ref", "HEAD"], cwd=source).stdout.strip()
    describe = run([_cmd("git"), "describe", "--tags", "--always"], cwd=source).stdout.strip()

    # Source-side validation happens entirely before the tasker stop.
    check_clean_enough(source)
    check_config_secrets(config_file)

    mat_flag = getattr(args, "materialize_generations_current", False)
    allowed_dir_symlink: Path | None = None
    gen_snapshot: tuple[str, Path, list[dict[str, Any]]] | None = None

    if mat_flag:
        gen_dir = data_dir / "generations"
        if not gen_dir.exists() or not gen_dir.is_dir() or gen_dir.is_symlink():
            die(f"generations directory missing or is a symlink: {gen_dir}")
        curr_link = gen_dir / "current"
        if not curr_link.is_symlink():
            die(f"generations/current missing or is not a symlink: {curr_link}")
        try:
            raw_link = os.readlink(curr_link)
        except OSError as exc:
            die(f"cannot readlink generations/current: {exc}")
        target_path, target_rel, entries = inspect_generation_target(gen_dir, raw_link)
        allowed_dir_symlink = curr_link
        gen_snapshot = (raw_link, target_path, entries)

    # Symlinked directories under an archived tree are refused before the
    # source service is quiesced: member collection must never follow links
    # out of the archived tree, and a refusal must never drain Tasker or
    # write an archive.
    check_no_symlinked_dirs(data_dir, "runtime/data", allowed_dir_symlink=allowed_dir_symlink)
    check_no_symlinked_dirs(web_root, "static/web")

    service_name = args.tasker_service
    validate_service_name(service_name)
    unit_dir = Path(os.environ.get("PLR_SYSTEMD_UNIT_DIR", "/etc/systemd/system"))
    unit_text = read_required_text(unit_dir / f"{service_name}.service", "tasker unit")
    check_text_secrets(unit_text, "tasker unit")
    # The live endpoint is the capture of record. Also require the runtime
    # mirror as a separate fail-closed source-state prerequisite.
    live_status_text = capture_live_tasker_status(config_file)
    status_text = read_required_text(data_dir / "tasker_status.json", "tasker status mirror")
    try:
        json.loads(status_text)
    except ValueError:
        die(f"tasker status mirror is not valid JSON: {data_dir / 'tasker_status.json'}")
    caddy_config = Path(os.environ.get("PLR_CADDY_CONFIG", "/etc/caddy/Caddyfile"))
    caddy_block = extract_caddy_managed_block(caddy_config)
    if caddy_block is None:
        die(f"managed Caddy block not found in {caddy_config}; refusing to create archive without it")

    _check_root()
    controller = get_service_controller("systemd")
    tmp = Path(tempfile.mkdtemp(prefix="pl-recovery-create-"))
    initially_active = False
    stopped = False
    restarted = False
    try:
        bundle_tmp = tmp / "repository.bundle"
        create_bundle(source, bundle_tmp)
        # Bundle preflight: the recorded revision must be a bundle head before
        # the source service is quiesced.
        heads = run([_cmd("git"), "bundle", "list-heads", str(bundle_tmp)], cwd=source)
        if source_sha not in heads.stdout.split():
            die("bundle preflight failed: recorded source revision not in bundle heads")

        initially_active = controller.status(service_name, "production", source, web_root)["state"] == "active"
        if initially_active:
            controller.stop(service_name, "production", source, web_root)
            stopped = True

        # Re-read and re-validate generation link and target snapshot post-stop, before staging
        if mat_flag:
            assert gen_snapshot is not None
            expected_link, expected_target, expected_entries = gen_snapshot
            validate_generation_snapshot(
                data_dir / "generations",
                expected_link,
                expected_target,
                expected_entries,
            )

        staging = tmp / "staging"
        staging.mkdir()
        (staging / "source").mkdir()
        shutil.copy2(bundle_tmp, staging / BUNDLE_MEMBER)
        _write_staging_json(
            staging,
            REVISION_MEMBER,
            {
                "schema_version": SCHEMA_VERSION,
                "source_sha": source_sha,
                "branch": branch,
                "describe": describe,
                "app_dir": str(source),
                "web_root": str(web_root),
                "tasker_service": service_name,
            },
        )
        data_members = iter_tree_files(data_dir, "runtime/data", allowed_dir_symlink=allowed_dir_symlink)
        copy_members_to_staging(data_members, data_dir, staging, "runtime/data")

        # Materialize generations/current in staging if requested
        mat_members: list[str] = []
        if mat_flag:
            assert gen_snapshot is not None
            expected_link, expected_target, expected_entries = gen_snapshot
            for entry in expected_entries:
                rel_f = entry["rel_path"]
                src_f = expected_target / rel_f
                dst_f = staging / GENERATION_ARCHIVE_PATH / rel_f
                dst_f.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_f, dst_f)
                mat_members.append(f"{GENERATION_ARCHIVE_PATH}/{rel_f}")
            # Validate staged trees against snapshot and each other before tar
            try:
                validate_staged_generation_trees(staging, expected_link, expected_entries)
            except (OSError, PermissionError):
                die("cannot validate staged generation trees")

        web_members = iter_tree_files(web_root, "static/web")
        copy_members_to_staging(web_members, web_root, staging, "static/web")
        (staging / "config").mkdir()
        shutil.copy2(config_file, staging / CONFIG_MEMBER)
        optional_members: list[str] = []
        research = source / "logs" / "research-implement.md"
        if research.is_file():
            optional_members.append(RESEARCH_MEMBER)
            (staging / "runtime" / "logs").mkdir(parents=True, exist_ok=True)
            shutil.copy2(research, staging / RESEARCH_MEMBER)
        (staging / "metadata").mkdir()
        _write_staging_text(staging, UNIT_MEMBER, unit_text)
        _write_staging_text(staging, STATUS_MEMBER, live_status_text)
        _write_staging_text(staging, CADDY_MEMBER, caddy_block)
        _write_staging_json(
            staging,
            CREATED_MEMBER,
            {
                "schema_version": SCHEMA_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "archive": archive.name,
            },
        )
        (staging / "tools").mkdir()
        shutil.copy2(Path(__file__), staging / TOOLS_MEMBER)

        member_list = sorted(
            [
                BUNDLE_MEMBER,
                REVISION_MEMBER,
                CONFIG_MEMBER,
                UNIT_MEMBER,
                STATUS_MEMBER,
                CADDY_MEMBER,
                CREATED_MEMBER,
                TOOLS_MEMBER,
                *data_members,
                *mat_members,
                *web_members,
                *optional_members,
            ]
        )
        embedded: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "sha": source_sha,
                "branch": branch,
                "describe": describe,
                "app_dir": str(source),
                "web_root": str(web_root),
                "tasker_service": service_name,
            },
            "data_index_generator_sha": data_index_generator_sha(staging / "static" / "web"),
            "storage_encryption_attested": True,
            "members": member_entries([m for m in member_list if m != MANIFEST_MEMBER], staging),
        }
        if mat_flag:
            assert gen_snapshot is not None
            expected_link, _, _ = gen_snapshot
            embedded["generation_materialization"] = {
                "schema_version": GENERATION_MATERIALIZATION_SCHEMA,
                "link_path": GENERATION_LINK_PATH,
                "original_link": expected_link,
                "target_path": f"data/generations/{expected_link}",
                "archive_path": GENERATION_ARCHIVE_PATH,
            }
        _write_staging_json(staging, MANIFEST_MEMBER, embedded)
        all_members = sorted([*member_list, MANIFEST_MEMBER])

        # COPYFILE_DISABLE keeps macOS tar from embedding AppleDouble "._*"
        # metadata members; the archive member set must be exactly the
        # validated member list.
        sidecar = Path(str(archive) + ".sha256")
        try:
            tar_res = run(
                [_cmd("tar"), "-cf", str(archive), "-C", str(staging), *all_members],
                env={"COPYFILE_DISABLE": "1"},
            )
            if tar_res.returncode != 0:
                die(f"tar create failed: {(tar_res.stderr or tar_res.stdout).strip()}")

            sidecar.write_text(f"{sha256_file(archive)}  {archive.name}\n", encoding="utf-8")
            archive.chmod(0o600)
            sidecar.chmod(0o600)

            ok, vreport = verify_archive(archive, sidecar)
        except (Exception, SystemExit):
            if mat_flag:
                archive.unlink(missing_ok=True)
                sidecar.unlink(missing_ok=True)
            raise

        report = dict(vreport)
        report["command"] = "create"
        report["service_name"] = service_name
        report["service_controller"] = controller.name
        report["initially_active"] = initially_active
        report["service_stopped"] = stopped
        report["service_started"] = False
        report["storage_encryption_attested"] = True
        report["member_count"] = len(all_members)
        if not ok:
            if mat_flag:
                archive.unlink(missing_ok=True)
                sidecar.unlink(missing_ok=True)
            print_failed_verify(report, "create")

        if initially_active:
            try:
                controller.start(service_name, "production", source, web_root)
            except SystemExit:
                # Backward-compatible: the archive exists and was verified, so
                # a failed restart still emits the create report (ok false,
                # service_started false, archive fields) before stderr reports
                # the failure; the finally-block fallback start still runs.
                report["ok"] = False
                print(json.dumps(report, indent=2, sort_keys=True))
                raise
            restarted = True
            report["service_started"] = True
        report["ok"] = True
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    finally:
        # The drain phase, once entered, always attempts the source restart;
        # deliberately a plain run() (never controller.start) so a failed
        # restart attempt cannot mask the original error.
        if initially_active and not restarted:
            run([_cmd("systemctl"), "start", service_name])
        shutil.rmtree(tmp, ignore_errors=True)


# ── verify ──────────────────────────────────────────────────────────────────


def cmd_verify(args: argparse.Namespace) -> int:
    if not Path(args.archive).is_absolute():
        die("verify --archive must be an absolute path")
    archive = Path(args.archive).resolve()
    sidecar = Path(str(archive) + ".sha256")
    ok, report = verify_archive(archive, sidecar)
    if not ok:
        print_failed_verify(report, "verify")
    report["ok"] = True
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


# ── restore ─────────────────────────────────────────────────────────────────


def _copy_tree_members(src_root: Path, dst_root: Path) -> None:
    if not src_root.is_dir():
        return
    for path in sorted(src_root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(src_root)
        target = dst_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        # Restore original mode from source file
        os.chmod(target, path.stat().st_mode & 0o7777)


def rollback_existing(path: Path) -> Path | None:
    """Move any existing target aside; the rollback dir is kept."""
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    rollback = path.with_name(f"{path.name}.rollback-{stamp}")
    n = 1
    while rollback.exists():
        rollback = path.with_name(f"{path.name}.rollback-{stamp}-{n}")
        n += 1
    path.rename(rollback)
    return rollback


def _tracked_symlink_ok(checkout: Path, rel: str) -> bool:
    """Tracked symlink (git mode 120000) restore contract: raw target must
    be a non-empty relative path that resolves strictly inside the checkout
    to an ordinary non-symlink regular file, with no symlinked component in
    the raw target path (no chains, no directory targets, no broken
    targets)."""
    path = checkout / rel
    try:
        raw = os.readlink(path)
    except OSError:
        return False
    if not raw or os.path.isabs(raw):
        return False
    try:
        root = checkout.resolve(strict=True)
    except OSError:
        return False
    target = (path.parent / raw).resolve(strict=False)
    if not target.is_relative_to(root) or target == root:
        return False
    cursor = path.parent
    try:
        for part in Path(raw).parts:
            cursor = cursor / part
            if cursor.is_symlink():
                return False
        if cursor.is_symlink() or not cursor.is_file():
            return False
    except OSError:
        return False
    return True


def scan_checkout_unsafe(checkout: Path) -> list[str]:
    """Reject gitlinks, hardlinked regular files, untracked symlinks, and
    tracked symlinks that violate the file-symlink contract in the staged
    source checkout before any recovery content is copied through them.
    Tracked symlinks proven safe by _tracked_symlink_ok are allowed (the
    authoritative repo legitimately contains AGENTS.md -> CLAUDE.md)."""
    bad: list[str] = []
    tracked_symlinks: set[str] = set()
    res = run([_cmd("git"), "-C", str(checkout), "ls-files", "-s"])
    if res.returncode != 0:
        die(f"git ls-files failed in staged checkout: {(res.stderr or res.stdout).strip()}")
    for line in res.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) != 2:
            continue
        mode = fields[0].split()[0] if fields[0].split() else ""
        rel = fields[1]
        if mode == "120000":
            tracked_symlinks.add(rel)
            if not _tracked_symlink_ok(checkout, rel):
                bad.append(f"{rel} (tracked symlink, git mode 120000)")
            continue
        if mode == "160000":
            bad.append(f"{rel} (gitlink, git mode 160000)")
            continue
        path = checkout / rel
        if path.is_symlink():
            bad.append(f"{rel} (working-tree symlink)")
        elif path.is_file():
            nlink = 1
            try:
                nlink = path.stat().st_nlink
            except OSError:
                pass
            if nlink > 1:
                bad.append(f"{rel} (hardlinked, nlink={nlink})")
    for path in checkout.rglob("*"):
        if ".git" in path.parts:
            continue
        if path.is_symlink():
            rel = path.relative_to(checkout).as_posix()
            if rel in tracked_symlinks:
                continue
            bad.append(f"{rel} (untracked symlink)")
    return bad


def _dev_api_unit(app_dir: Path, web_root: Path) -> str:
    return f"""[Unit]
Description=Portfolio Lab tasker API (dev recovery candidate, scheduler disabled)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={app_dir}
Environment=PORTFOLIO_LAB_ENABLE_ML=0
Environment=PORTFOLIO_LAB_MODE=lab
Environment=CRON_BACKEND=tasker
Environment=PORTFOLIO_LAB_PROJECT_DIR={app_dir}
Environment=PUBLIC_DATA_DIR={web_root}/data
Environment=PYTHONPATH={app_dir}
Environment=LOG_LEVEL=INFO
Environment=JSON_LOGS=1
Environment=PATH=/usr/local/bin:/usr/bin:/bin
EnvironmentFile=-{app_dir}/.env.local
Environment=TASKER_DISABLE_SCHEDULER=1
ExecStart={app_dir}/scripts/python_runtime.sh -m src.tasker.service --host 127.0.0.1 --port 8000 --no-scheduler
Restart=always
RestartSec=10
TimeoutStopSec=30
KillSignal=SIGTERM

[Install]
WantedBy=multi-user.target
"""


def cmd_restore(args: argparse.Namespace) -> int:
    if args.target_mode == "prod" and args.start_dev_api:
        die("--start-dev-api is only valid for dev mode")
    archive_raw = Path(args.archive)
    if not archive_raw.is_absolute():
        die("restore --archive must be an absolute path")
    archive = archive_raw.resolve()
    sidecar = Path(str(archive) + ".sha256")
    raw_app_dir = Path(args.app_dir)
    raw_web_root = Path(args.web_root)
    if not raw_app_dir.is_absolute() or not raw_web_root.is_absolute():
        die("restore --app-dir and --web-root must be absolute paths")
    if args.target_mode == "dev":
        # Dev-mode production-path guard runs BEFORE resolve(): resolve() can
        # raise PermissionError traversing unreadable parents (CI runner home
        # dirs), while this check is pure string containment — no fs access.
        # Resolved forms are re-checked after resolve() so symlink aliases
        # (/var -> /private/var on macOS) still match.
        if "/root/projects/portfolio-lab" in str(raw_app_dir) or "/var/www/portfolio-lab" in str(raw_web_root):  # portability-scan allowlist: intentional production-path guard
            die("dev mode rejects production paths (/root/projects/portfolio-lab, /var/www/portfolio-lab)")  # portability-scan allowlist: intentional production-path guard
    app_dir = raw_app_dir.resolve()
    web_root = raw_web_root.resolve()
    if app_dir == web_root or app_dir.is_relative_to(web_root) or web_root.is_relative_to(app_dir):
        die("restore --app-dir and --web-root must be distinct, non-overlapping paths")
    # Raw-path symlink check: resolve() would follow the link and hide it.
    if raw_app_dir.is_symlink():
        die(f"restore target is a symlink: {raw_app_dir}")
    if raw_web_root.is_symlink():
        die(f"restore target is a symlink: {raw_web_root}")

    # Explicit controller selection; box-persist targets must sit under the
    # box-persist root, rejected before any target mutation or controller call.
    controller = get_service_controller(args.service_controller)
    if args.service_controller == "box-persist":
        validate_box_persist_targets(app_dir, web_root)

    # Verify fully before any target mutation.
    ok, report = verify_archive(archive, sidecar)
    if not ok:
        print_failed_verify(report, "restore")

    if args.target_mode == "dev":
        # Match both the raw and resolved forms: /var resolves to /private/var
        # on macOS, and the operator-facing contract names the prod paths
        # literally. Raw strings were already rejected pre-resolve() above;
        # this second pass keeps the guard's dual-form match for symlink
        # aliases (and the exact literal is pinned by
        # tests/test_cron_python_runtime.py).
        app_forbidden = "/root/projects/portfolio-lab" in (str(app_dir), str(raw_app_dir))  # portability-scan allowlist: intentional production-path guard
        web_forbidden = "/var/www/portfolio-lab" in (str(web_root), str(raw_web_root))
        if app_forbidden or web_forbidden:
            die("dev mode rejects production paths (/root/projects/portfolio-lab, /var/www/portfolio-lab)")  # portability-scan allowlist: intentional production-path guard
    elif not args.allow_production_paths:
        die("prod target mode requires --allow-production-paths")

    tmp = Path(tempfile.mkdtemp(prefix="pl-recovery-restore-"))
    try:
        extracted = tmp / "members"
        try:
            infos, _names = read_archive_members(archive)
            extract_validated_members(archive, infos, extracted)
        except ValueError as exc:
            die(str(exc))
        # Contained fail-closed guard: the archive was verified above; if its
        # bytes changed between verification and this staging pass, refuse
        # rather than trust the re-read data. A residual race window remains
        # between this re-check and the reads below (recorded in the task
        # report; no broader refactor in this fix wave).
        if sha256_file(archive) != report.get("archive_sha256"):
            die("archive changed between verification and staging; refusing restore")
        try:
            manifest = json.loads((extracted / MANIFEST_MEMBER).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            die("embedded manifest is not valid JSON; refusing restore")
        source_info = manifest.get("source") or {}
        source_sha = source_info.get("sha")
        if not isinstance(source_sha, str) or not valid_full_sha(source_sha):
            die(f"recorded source revision is not a valid 40-hex sha: {source_sha!r}")
        archived_service = source_info.get("tasker_service")
        if not isinstance(archived_service, str):
            die("recorded source tasker service is missing or invalid")
        validate_service_name(archived_service)
        if args.target_mode == "prod":
            if args.tasker_service is not None and args.tasker_service != archived_service:
                die(
                    f"--tasker-service {args.tasker_service!r} does not match the archived source service "
                    f"{archived_service!r}"
                )
            if args.service_controller == "systemd":
                _check_root()
                target_status = controller.status(archived_service, "production", app_dir, web_root)
                if target_status["state"] != "inactive":
                    die(
                        f"target service {archived_service} must be inactive before production restore "
                        f"(state: {target_status['state'] or 'unknown'})"
                    )
            else:
                # box-persist prod staging preflights the target production
                # status before any mutation; it remains staged only.
                target_status = controller.status(archived_service, "production", app_dir, web_root)
                require_prod_inactive_status(target_status, archived_service, "before production restore")

        # Dev API identity is validated before any target mutation: it must be
        # a distinct unit name, never the archived/production service.
        dev_name: str | None = None
        if args.start_dev_api:
            dev_name = args.tasker_service or os.environ.get("PLR_DEV_SERVICE_NAME", DEFAULT_DEV_SERVICE_NAME)
            if dev_name == archived_service:
                die(
                    f"--tasker-service {dev_name!r} is the archived/production service; "
                    "the dev API needs a distinct unit name"
                )
            validate_service_name(dev_name)
            validate_unit_value(str(app_dir), "app dir")
            validate_unit_value(str(web_root), "web root")

        # Staged source checkout from the bundle, pinned to the recorded sha.
        checkout = tmp / "checkout"
        run_checked(
            [_cmd("git"), "clone", str(extracted / BUNDLE_MEMBER), str(checkout)],
            "git clone from bundle",
        )
        pin = run([_cmd("git"), "-C", str(checkout), "rev-parse", "--verify", f"{source_sha}^{{commit}}"])
        if pin.returncode != 0:
            die(f"recorded source revision {source_sha} is not a commit in the bundle")
        run_checked(
            [_cmd("git"), "-C", str(checkout), "checkout", "--detach", "--quiet", source_sha],
            "git checkout recorded source revision",
        )
        head = run([_cmd("git"), "-C", str(checkout), "rev-parse", "--verify", "HEAD"])
        if head.returncode != 0 or head.stdout.strip() != source_sha:
            die("staged checkout HEAD does not match recorded source revision")
        bad_entries = scan_checkout_unsafe(checkout)
        if bad_entries:
            die(
                "restored source checkout contains symlink/gitlink/hardlink entries; "
                "refusing to copy recovery content through them: " + "; ".join(bad_entries[:10])
            )

        # Build the full replacement trees in safe staging before touching
        # any target.
        staging_app = checkout
        _copy_tree_members(extracted / "runtime" / "data", staging_app / "data")

        # Generation link materialization reconstruction
        gen_mat = manifest.get("generation_materialization")
        if gen_mat is not None:
            mat_ok, orig_link, mat_err = parse_and_validate_generation_metadata(gen_mat)
            if not mat_ok:
                die(f"restore manifest validation failed: {mat_err}")
            assert orig_link is not None

            staged_generations = staging_app / "data" / "generations"
            staged_target = staged_generations / orig_link
            staged_current = staged_generations / "current"

            # Validate both staged target and staged current against manifest member sets
            target_prefix = f"runtime/data/generations/{orig_link}/"
            curr_prefix = f"{GENERATION_ARCHIVE_PATH}/"
            manifest_members = manifest.get("members") or []
            entries_by_path = {str(entry.get("path")): entry for entry in manifest_members}
            expected_target_files = {
                p[len(target_prefix):]: entries_by_path[p]
                for p in entries_by_path
                if p.startswith(target_prefix)
            }
            expected_curr_files = {
                p[len(curr_prefix):]: entries_by_path[p]
                for p in entries_by_path
                if p.startswith(curr_prefix)
            }

            if not expected_target_files or not expected_curr_files:
                die("manifest generation member sets must be non-empty")

            # 2. Prove staged target is an ordinary non-symlink directory beneath staged generations
            try:
                if staged_generations.is_symlink() or not staged_generations.is_dir():
                    die("staged generations dir is not an ordinary directory")
                # Explicitly validate every intermediate component from staged_generations through staged_target
                walk_comp = staged_generations
                for part in Path(orig_link).parts:
                    walk_comp = walk_comp / part
                    if walk_comp.is_symlink() or not walk_comp.is_dir():
                        die("staged generation target component is not an ordinary directory")
                if staged_target.is_symlink() or not staged_target.is_dir():
                    die("staged generation target is not an ordinary directory")
                if not staged_target.resolve().is_relative_to(staged_generations.resolve()):
                    die("staged generation target escapes generations directory")

                # Validate files in staged_target
                for p in staged_target.rglob("*"):
                    if p.is_symlink():
                        die("symlink found inside staged generation target")
                    st = p.lstat()
                    if stat.S_ISDIR(st.st_mode):
                        continue
                    if not stat.S_ISREG(st.st_mode):
                        die("non-regular file found inside staged generation target")

                staged_target_map = {
                    p.relative_to(staged_target).as_posix(): p
                    for p in staged_target.rglob("*")
                    if p.is_file()
                }
                if set(staged_target_map.keys()) != set(expected_target_files.keys()):
                    die("staged generation target files do not match manifest")
                for rel_f, exp_entry in expected_target_files.items():
                    p = staged_target_map[rel_f]
                    st = p.stat()
                    exp_mode = exp_entry.get("mode")
                    # When extracted into staging checkout from tar via extractfile,
                    # restore applies manifest mode to match archived mode.
                    os.chmod(p, exp_mode)
                    if (
                        sha256_file(p) != exp_entry.get("sha256")
                        or st.st_size != exp_entry.get("bytes")
                        or (p.stat().st_mode & 0o7777) != exp_mode
                    ):
                        die("staged generation target file metadata does not match manifest")

                # 3. Prove staged current is an ordinary non-symlink directory
                if staged_current.is_symlink() or not staged_current.is_dir():
                    die("staged generations/current is not an ordinary directory")
                for p in staged_current.rglob("*"):
                    if p.is_symlink():
                        die("symlink found inside staged generations/current")
                    st = p.lstat()
                    if stat.S_ISDIR(st.st_mode):
                        continue
                    if not stat.S_ISREG(st.st_mode):
                        die("non-regular file found inside staged generations/current")

                staged_current_map = {
                    p.relative_to(staged_current).as_posix(): p
                    for p in staged_current.rglob("*")
                    if p.is_file()
                }
                if set(staged_current_map.keys()) != set(expected_curr_files.keys()):
                    die("staged generations/current files do not match manifest")
                for rel_f, exp_entry in expected_curr_files.items():
                    p = staged_current_map[rel_f]
                    st = p.stat()
                    exp_mode = exp_entry.get("mode")
                    os.chmod(p, exp_mode)
                    if (
                        sha256_file(p) != exp_entry.get("sha256")
                        or st.st_size != exp_entry.get("bytes")
                        or (p.stat().st_mode & 0o7777) != exp_mode
                    ):
                        die("staged generations/current file metadata does not match manifest")

                # 4. Remove only the staged ordinary current directory
                shutil.rmtree(staged_current)

                # 5. Recreate staged current as a relative symlink with exact original_link
                os.symlink(orig_link, staged_current)

                # 6. Prove reconstructed link resolves to staged target beneath generations
                if not staged_current.is_symlink():
                    die("failed to create staged generations/current symlink")
                if os.readlink(staged_current) != orig_link:
                    die("staged generations/current raw link mismatch")
                if staged_current.resolve() != staged_target.resolve():
                    die("staged generations/current does not resolve to target")
                if not staged_current.resolve().is_relative_to(staged_generations.resolve()):
                    die("staged generations/current resolved target escapes generations directory")
            except OSError:
                die("failed to reconstruct staged generations/current symlink")

        config_restored = False
        if args.target_mode == "prod" and (extracted / CONFIG_MEMBER).is_file():
            (staging_app / "config").mkdir(parents=True, exist_ok=True)
            shutil.copy2(extracted / CONFIG_MEMBER, staging_app / "config" / "lab-app.env")
            config_restored = True
        if (extracted / RESEARCH_MEMBER).is_file():
            (staging_app / "logs").mkdir(parents=True, exist_ok=True)
            shutil.copy2(extracted / RESEARCH_MEMBER, staging_app / "logs" / "research-implement.md")
        staging_web = tmp / "web"
        _copy_tree_members(extracted / "static" / "web", staging_web)

        state_dir = staging_app / ".portfolio-lab-recovery"
        shutil.copytree(extracted / "metadata", state_dir / "metadata")
        shutil.copytree(extracted / "tools", state_dir / "tools")
        shutil.copy2(extracted / MANIFEST_MEMBER, state_dir / "recovery-manifest.json")
        restore_report = {
            "schema_version": SCHEMA_VERSION,
            "archive": str(archive),
            "archive_sha256": report.get("archive_sha256"),
            "manifest_sha256": report.get("manifest_sha256"),
            "source_sha": source_sha,
            "target_mode": args.target_mode,
            "config_restored": config_restored,
            "app_dir": str(app_dir),
            "web_root": str(web_root),
            "tasker_service": archived_service if args.target_mode == "prod" else args.tasker_service,
            "verified": True,
            "checks": report.get("checks"),
            "rollback_app_dir": None,
            "rollback_web_root": None,
            "restored_at": datetime.now(timezone.utc).isoformat(),
        }
        (state_dir / "restore-report.json").write_text(
            json.dumps(restore_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        # Replace targets only after all staging work succeeded; keep rollback
        # dirs and restore moved-aside targets if placement fails partway.
        rollback_app: Path | None = None
        rollback_web: Path | None = None
        moved: list[tuple[Path, Path]] = []
        try:
            rollback_app = rollback_existing(app_dir)
            if rollback_app is not None:
                moved.append((app_dir, rollback_app))
            rollback_web = rollback_existing(web_root)
            if rollback_web is not None:
                moved.append((web_root, rollback_web))
            shutil.move(str(staging_app), str(app_dir))
            shutil.move(str(staging_web), str(web_root))
        except OSError as exc:
            for original, rollback in reversed(moved):
                if original.exists():
                    shutil.rmtree(original, ignore_errors=True)
                try:
                    rollback.rename(original)
                except OSError:
                    pass
            die(f"failed to place restored trees at target: {exc}")
        restore_report["rollback_app_dir"] = str(rollback_app) if rollback_app else None
        restore_report["rollback_web_root"] = str(rollback_web) if rollback_web else None
        (app_dir / ".portfolio-lab-recovery" / "restore-report.json").write_text(
            json.dumps(restore_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        report["command"] = "restore"
        report["service_controller"] = controller.name
        report["staged"] = True
        report["app_dir"] = str(app_dir)
        report["web_root"] = str(web_root)
        report["mode"] = args.target_mode
        report["config_restored"] = config_restored
        report["service_started"] = False
        report["service_name"] = None
        report["dev_api_unit"] = None
        report["rollback_app_dir"] = str(rollback_app) if rollback_app else None
        report["rollback_web_root"] = str(rollback_web) if rollback_web else None
        if args.target_mode == "prod":
            report["activation_note"] = "staged only; production activation is a separate activate-prod step"
            if args.service_controller == "box-persist":
                report["controller_status"] = normalized_status(target_status)
            report["ok"] = True
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0

        report["config_note"] = (
            "config/lab-app.env is not restored in dev mode; supply target-specific "
            "non-secret config separately"
        )
        if args.start_dev_api:
            assert dev_name is not None
            if args.service_controller == "systemd":
                _check_root()
                unit_dir = Path(os.environ.get("PLR_SYSTEMD_UNIT_DIR", "/etc/systemd/system"))
                unit_dir.mkdir(parents=True, exist_ok=True)
                unit_path = unit_dir / f"{dev_name}.service"
                unit_path.write_text(_dev_api_unit(app_dir, web_root), encoding="utf-8")
                run_checked([_cmd("systemctl"), "daemon-reload"], "systemctl daemon-reload")
                run_checked([_cmd("systemctl"), "enable", dev_name], f"systemctl enable {dev_name}")
                run_checked([_cmd("systemctl"), "restart", dev_name], f"systemctl restart {dev_name}")
                report["service_started"] = True
                report["service_name"] = dev_name
                report["dev_api_unit"] = str(unit_path)
            else:
                # box-persist dev: delegate to start-candidate (no systemd
                # unit write, daemon-reload, enable, or restart) and require
                # the returned active/disabled/zero-instance/exact/PID status.
                started = controller.start(dev_name, "candidate", app_dir, web_root)
                require_candidate_started(started, dev_name)
                report["service_started"] = True
                report["service_name"] = dev_name
                report["dev_api_unit"] = None
                report["controller_status"] = normalized_status(started)
        report["ok"] = True
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── activate-prod (separate from restore) ───────────────────────────────────


def cmd_activate(args: argparse.Namespace) -> int:
    if not args.confirm_authoritative_activation:
        die("activate-prod requires --confirm-authoritative-activation")
    if not args.former_authority_confirmed_stopped.strip():
        die("activate-prod requires --former-authority-confirmed-stopped LABEL (non-whitespace)")
    raw_app_dir = Path(args.app_dir)
    raw_web_root = Path(args.web_root)
    if not raw_app_dir.is_absolute() or not raw_web_root.is_absolute():
        die("activate-prod --app-dir and --web-root must be absolute paths")
    app_dir = raw_app_dir.resolve()
    web_root = raw_web_root.resolve()
    controller = get_service_controller(args.service_controller)
    if args.service_controller == "box-persist":
        validate_box_persist_targets(app_dir, web_root)
    service_name = args.tasker_service
    validate_service_name(service_name)

    state_dir = app_dir / ".portfolio-lab-recovery"
    report_path = state_dir / "restore-report.json"
    if not report_path.is_file():
        die(f"restore report missing at {report_path}; run restore first")
    try:
        restore_report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        die(f"restore report is not valid JSON: {report_path}")
    if restore_report.get("schema_version") != SCHEMA_VERSION:
        die(f"restore report schema mismatch: {restore_report.get('schema_version')!r}")
    if restore_report.get("verified") is not True:
        die("restore report is not verified; refusing activation")
    if restore_report.get("target_mode") != "prod":
        die("activate-prod requires a prod restore report; refusing to activate a dev restore")
    if restore_report.get("app_dir") != str(app_dir) or restore_report.get("web_root") != str(web_root):
        die("restore report paths do not match --app-dir/--web-root")
    source_sha = restore_report.get("source_sha")
    if not isinstance(source_sha, str) or not valid_full_sha(source_sha):
        die("restore report has no valid 40-hex source_sha")

    # Re-verify the original archive and bind the report to it before
    # trusting any staged state.
    archive_raw = Path(str(restore_report.get("archive") or ""))
    if not str(archive_raw).startswith("/") or not archive_raw.is_file():
        die("restore report archive path is not an absolute existing archive")
    ok, fresh = verify_archive(archive_raw, Path(str(archive_raw) + ".sha256"))
    if not ok:
        print_failed_verify(fresh, "activate-prod")
    if fresh.get("archive_sha256") != restore_report.get("archive_sha256"):
        die("restore report archive digest does not match the archive at the recorded path")
    if fresh.get("manifest_sha256") != restore_report.get("manifest_sha256"):
        die("restore report manifest digest does not match the verified archive")
    if fresh.get("source_sha") != source_sha:
        die("restore report source revision does not match the verified archive")
    if not isinstance(fresh.get("source_tasker_service"), str) or fresh["source_tasker_service"] != service_name:
        die(
            f"activate-prod service name {service_name!r} does not match the archived source service "
            f"{fresh.get('source_tasker_service')!r}"
        )

    # State manifest must be byte-identical to the verified archive's manifest.
    state_manifest = state_dir / "recovery-manifest.json"
    if not state_manifest.is_file():
        die(f"recovery manifest missing at {state_manifest}")
    if sha256_file(state_manifest) != fresh.get("manifest_sha256"):
        die("restored recovery-manifest.json does not match the verified archive")
    try:
        manifest = json.loads(state_manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        die("recovery manifest is not valid JSON")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        die("recovery manifest schema mismatch")
    members = manifest.get("members")
    if not isinstance(members, list) or not members:
        die("recovery manifest member list is empty or malformed")

    # Report provenance coherence: checkout HEAD, static release sha, generator.
    head_res = run([_cmd("git"), "-C", str(app_dir), "rev-parse", "--verify", "HEAD"])
    if head_res.returncode != 0 or head_res.stdout.strip() != source_sha:
        die(f"restored app HEAD does not match report source revision {source_sha}")
    release_sha = _read_json_string_field(web_root / "_release.json", "source_git_sha")
    if release_sha != source_sha:
        die(f"static release provenance incoherent: _release.json source_git_sha {release_sha!r} != {source_sha!r}")
    generator_sha = _read_json_string_field(web_root / "data" / "index.json", "generator_git_sha")
    if generator_sha is None:
        die("public data index has no generator_git_sha; refusing activation")
    if not valid_sha_prefix(generator_sha):
        die(f"generator_git_sha is not a valid short sha: {generator_sha!r}")
    revs = run([_cmd("git"), "-C", str(app_dir), "rev-list", "--all"])
    if revs.returncode != 0:
        die("git rev-list --all failed in restored checkout")
    matches = [rev for rev in revs.stdout.split() if rev.startswith(generator_sha)]
    if len(matches) != 1:
        die(
            f"generator_git_sha {generator_sha!r} must resolve to exactly one reachable commit "
            f"(matches: {len(matches)})"
        )

    # Restored tree digests must match the archived manifest.
    for entry in members:
        member = str(entry.get("path"))
        if member.startswith("static/web/"):
            target = web_root / member[len("static/web/") :]
        elif member.startswith("runtime/data/"):
            target = app_dir / "data" / member[len("runtime/data/") :]
        elif member == CONFIG_MEMBER:
            target = app_dir / "config" / "lab-app.env"
        else:
            continue
        if not target.is_file() or sha256_file(target) != entry.get("sha256"):
            die(f"restored tree mismatch: {member} (missing or digest mismatch)")

    # The tracked source checkout must be clean (data/ is gitignored and the
    # recovery state dir is untracked, so neither trips this gate). Ordinary
    # modifications of the two allowed-dirty generated data files are
    # expected here: create archives them as-is, restore overlays the
    # archived runtime data, and the manifest-digest check above already
    # proved they match the archive. Any other tracked change blocks.
    status_res = run([_cmd("git"), "-C", str(app_dir), "status", "--porcelain", "--untracked-files=no"])
    if status_res.returncode != 0:
        die("git status failed in restored checkout: " + (status_res.stderr or status_res.stdout).strip())
    offending = _porcelain_offending(status_res.stdout)
    if offending:
        die(
            "restored checkout has uncommitted tracked changes beyond the allowed generated data files; "
            "refusing activation: " + "; ".join(offending[:10])
        )

    if not (app_dir / ".env.local").is_file():
        die("restored app dir has no .env.local; provision secrets before activation")

    # All gates above pass before any target mutation or controller call.
    label = args.former_authority_confirmed_stopped.strip()
    report = {
        "command": "activate-prod",
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "app_dir": str(app_dir),
        "web_root": str(web_root),
        "service_name": service_name,
        "service_controller": controller.name,
        "unit_installed": False,
        "service_started": True,
        "source_sha": source_sha,
        "generator_git_sha": generator_sha,
        "archive_sha256": fresh.get("archive_sha256"),
        "manifest_sha256": fresh.get("manifest_sha256"),
        "former_authority_confirmed_stopped": label,
        "dns_caddy_unchanged": True,
        "post_activation_acceptance": [
            "desktop and mobile site load",
            "SPA route fallback",
            "/_release.json",
            "/data/index.json",
            "/data/signals.json",
            "/api/tasker/status",
            "data freshness",
            "scheduler status",
            "expected kill/halt condition",
        ],
    }
    if args.service_controller == "systemd":
        _check_root()
        target_status = controller.status(service_name, "production", app_dir, web_root)
        if target_status["state"] != "inactive":
            die(
                f"target service {service_name} must be inactive before activation "
                f"(state: {target_status['state'] or 'unknown'})"
            )

        unit_src = state_dir / "metadata" / "tasker-unit.txt"
        if not unit_src.is_file():
            die("restored state missing metadata/tasker-unit.txt; activation requires the archived unit")
        # Re-bind the restored unit to the verified archive manifest and re-run
        # the secret scan before any systemd mutation; the manifest itself was
        # proven byte-identical to the verified archive above.
        unit_sha: str | None = None
        for entry in members:
            if str(entry.get("path")) == UNIT_MEMBER:
                candidate = entry.get("sha256")
                if isinstance(candidate, str) and len(candidate) == 64:
                    unit_sha = candidate
                break
        if unit_sha is None or sha256_file(unit_src) != unit_sha:
            die("restored tasker unit does not match the verified archive manifest; refusing activation")
        check_text_secrets(unit_src.read_text(encoding="utf-8", errors="replace"), "tasker unit")
        unit_dir = Path(os.environ.get("PLR_SYSTEMD_UNIT_DIR", "/etc/systemd/system"))
        unit_dir.mkdir(parents=True, exist_ok=True)
        unit_path = unit_dir / f"{service_name}.service"
        shutil.copy2(unit_src, unit_path)
        run_checked([_cmd("systemctl"), "daemon-reload"], "systemctl daemon-reload")
        run_checked([_cmd("systemctl"), "enable", service_name], f"systemctl enable {service_name}")
        run_checked([_cmd("systemctl"), "start", service_name], f"systemctl start {service_name}")
        report["unit_installed"] = True
        report["unit_path"] = str(unit_path)
    else:
        # box-persist: preflight the production status before any controller
        # activation call; the returned status is the explicit
        # second-scheduler prevention gate. No systemd units are written.
        preflight = controller.status(service_name, "production", app_dir, web_root)
        require_prod_inactive_status(preflight, service_name, "before activation")
        activated = controller.activate(service_name, app_dir, web_root, label)
        require_activation_status(activated, service_name)
        report["controller_preflight_status"] = normalized_status(preflight)
        report["controller_activation_status"] = normalized_status(activated)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


# ── CLI ─────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="create a verified recovery archive")
    p_create.add_argument("--app-dir", required=True, help="source app checkout (git repo)")
    p_create.add_argument("--web-root", required=True, help="served static web tree (release + public data)")
    p_create.add_argument("--tasker-service", required=True, help="tasker systemd unit name")
    p_create.add_argument("--archive", required=True, help="explicit absolute <name>.portfolio-lab-recovery.tar path")
    p_create.add_argument("--storage-encryption-attested", action="store_true", help="attest storage-layer encryption")
    p_create.add_argument(
        "--service-controller",
        choices=("systemd", "box-persist"),
        default="systemd",
        help="service controller (create is sg01 source-side: systemd only)",
    )
    p_create.add_argument(
        "--materialize-generations-current",
        action="store_true",
        default=False,
        help="materialize data/generations/current directory symlink in recovery archive",
    )

    p_verify = sub.add_parser("verify", help="verify an archive against its sidecar")
    p_verify.add_argument("--archive", required=True)

    p_restore = sub.add_parser("restore", help="stage a verified archive (dev default: no service)")
    p_restore.add_argument("--archive", required=True)
    p_restore.add_argument("--app-dir", required=True, help="restore target app checkout")
    p_restore.add_argument("--web-root", required=True, help="restore target web tree")
    p_restore.add_argument("--target-mode", choices=("dev", "prod"), required=True)
    p_restore.add_argument("--allow-production-paths", action="store_true", help="prod: allow production paths")
    p_restore.add_argument("--start-dev-api", action="store_true", help="dev: start distinct no-scheduler tasker unit")
    p_restore.add_argument("--tasker-service", default=None, help="dev API unit name (default: portfolio-lab-tasker-recovery-dev)")
    p_restore.add_argument(
        "--service-controller",
        choices=("systemd", "box-persist"),
        default="systemd",
        help="service controller (box-persist only for cursor-box targets)",
    )

    p_activate = sub.add_parser("activate-prod", help="promote a staged prod restore (separate from restore)")
    p_activate.add_argument("--app-dir", required=True)
    p_activate.add_argument("--web-root", required=True)
    p_activate.add_argument("--tasker-service", required=True)
    p_activate.add_argument("--confirm-authoritative-activation", action="store_true", help="explicit confirmation")
    p_activate.add_argument(
        "--former-authority-confirmed-stopped",
        default="",
        help="label of the former authority confirmed stopped",
    )
    p_activate.add_argument(
        "--service-controller",
        choices=("systemd", "box-persist"),
        default="systemd",
        help="service controller (box-persist only for cursor-box targets)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "create":
        return cmd_create(args)
    if args.command == "verify":
        return cmd_verify(args)
    if args.command == "restore":
        return cmd_restore(args)
    if args.command == "activate-prod":
        return cmd_activate(args)
    die(f"unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
