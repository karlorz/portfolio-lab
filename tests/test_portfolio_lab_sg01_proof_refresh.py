"""Strict-TDD tests for the attended sg01 former-authority proof refresh CLI.

Every test drives the shipped CLI (``scripts/portfolio_lab_sg01_proof_refresh.py``)
as ``sys.executable SCRIPT ...`` with a fake ``ssh`` binary (env-driven
responses + argv log) so no real SSH, network, or remote host is ever
touched — the default ``ssh`` name is never passed as an override. The
utility is one-shot and attended: it must never be scheduled, so a repository
drift guard asserts the script is referenced in no scheduling surface
(Makefile, crontab, ``config/tasker.yaml``, and every cron executable under
``scripts/cron/``).

Coverage: happy path (exact schema, host sg01, aware UTC collected_at,
nested booleans all false, mode 0600, owner uid, regular non-symlink);
byte-deterministic reruns under a pinned ``--now``; one compact secret-free
stdout summary only on success; fail-closed on each active/enabled true
state, missing ssh, probe timeout, nonzero/unexpected probe exits, bounded
stdout/stderr overrun, and unexpected/malformed tokens (failed/unknown/
masked/not-found/static, empty, embedded/trailing/leading whitespace, wrong
case, missing newline); the explicit missing-unit policy (``not-found`` and
the stderr-error shape stay fail-closed, with the attended gate documented in
the runbook); relative and symlink proof paths; every ``--ssh`` override
(including the literal default name) must be absolute; invalid connect
timeouts; naive or unconvertible ``--now``; probe fcntl/poll/kill lifecycle
errors (child reaped, pipes closed, one static line, no traceback); a closed
reporting pipe (exit 1, no interpreter-shutdown anomaly); existing proof
preserved byte-for-byte on every failure including a pre-rename verification
failure; temp-file cleanup; explicit-argv/BatchMode safety and leak
assertions; and integration proving the existing daily evidence collector
accepts the generated proof with a pinned ``--now`` (and warns when the
pinned ``--now`` is earlier than the proof's ``collected_at``).
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI = PROJECT_ROOT / "scripts" / "portfolio_lab_sg01_proof_refresh.py"
DAILY_CLI = PROJECT_ROOT / "scripts" / "portfolio_lab_daily_evidence.py"
RECYCLE_MD = PROJECT_ROOT / "scripts" / "PORTFOLIO_LAB_CURSOR_BOX_RECYCLE.md"

SCHEMA = "portfolio-lab-former-authority-proof/v1"
DEFAULT_SSH = "ssh"
DEFAULT_HOST = "sg01"
DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_PROOF = Path(
    "/home/box/.local/share/portfolio-lab/run/former-authority-proof.json"
)
TASKER_SERVICE = "portfolio-lab-tasker"
ARCHIVE_TIMER_SERVICE = "portfolio-lab-s3-archive.timer"

NOW = "2026-09-05T04:00:00Z"
NOW_ISO = "2026-09-05T04:00:00+00:00"
NOW_DT = datetime(2026, 9, 5, 4, 0, 0, tzinfo=timezone.utc)

PROBE_KEYS = (
    ("is-active", TASKER_SERVICE),
    ("is-enabled", TASKER_SERVICE),
    ("is-active", ARCHIVE_TIMER_SERVICE),
    ("is-enabled", ARCHIVE_TIMER_SERVICE),
)


# ── fake ssh (env-driven, no secrets, argv-logged) ────────────────────────


def write_fake_ssh(path: Path) -> Path:
    """Fake ssh: answers per ``FAKE_SSH_SCRIPT`` JSON keyed by
    "subcommand unit" and logs argv to ``FAKE_SSH_ARGV_LOG`` when set."""
    source = f"""#!{sys.executable}
import json, os, sys, time
script = json.load(open(os.environ["FAKE_SSH_SCRIPT"], encoding="utf-8"))
log = os.environ.get("FAKE_SSH_ARGV_LOG", "")
if log:
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(sys.argv[1:]) + "\\n")
entry = script.get(" ".join(sys.argv[-2:]), script.get("*", {{"exit": 255}}))
delay = entry.get("delay", 0)
if delay:
    time.sleep(delay)
sys.stderr.write(entry.get("err", ""))
sys.stdout.write(entry.get("out", ""))
sys.exit(entry.get("exit", 0))
"""
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)
    return path


def inactive_responses() -> dict[str, dict[str, object]]:
    """Canonical systemd answers for both units exactly inactive/disabled."""
    return {
        "is-active portfolio-lab-tasker": {"exit": 3, "out": "inactive\n"},
        "is-enabled portfolio-lab-tasker": {"exit": 1, "out": "disabled\n"},
        "is-active portfolio-lab-s3-archive.timer": {"exit": 3, "out": "inactive\n"},
        "is-enabled portfolio-lab-s3-archive.timer": {"exit": 1, "out": "disabled\n"},
    }


# ── fixtures and helpers ──────────────────────────────────────────────────


@pytest.fixture
def box(tmp_path: Path) -> SimpleNamespace:
    ssh_script = tmp_path / "ssh-responses.json"
    ssh_script.write_text(json.dumps(inactive_responses()), encoding="utf-8")
    return SimpleNamespace(
        tmp=tmp_path,
        proof=tmp_path / "former-authority-proof.json",
        fake_ssh=write_fake_ssh(tmp_path / "fake-ssh"),
        ssh_script=ssh_script,
        argv_log=tmp_path / "ssh-argv.jsonl",
    )


def set_ssh_responses(
    box: SimpleNamespace, responses: dict[str, dict[str, object]]
) -> None:
    box.ssh_script.write_text(json.dumps(responses), encoding="utf-8")


def ssh_env(box: SimpleNamespace) -> dict[str, str]:
    return {
        "FAKE_SSH_SCRIPT": str(box.ssh_script),
        "FAKE_SSH_ARGV_LOG": str(box.argv_log),
    }


def run_cli(
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    cli: Path = CLI,
) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(cli)]
    if args:
        cmd.extend(args)
    run_env = dict(os.environ)
    run_env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    if env:
        run_env.update(env)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=run_env,
        timeout=120,
    )


def base_args(box: SimpleNamespace, **over: str) -> list[str]:
    args = {
        "proof": str(box.proof),
        "ssh": str(box.fake_ssh),
        "connect_timeout": "5",
        "now": NOW,
    }
    args.update(over)
    out: list[str] = []
    for flag, value in args.items():
        out += [f"--{flag.replace('_', '-')}", value]
    return out


def run_refresh(
    box: SimpleNamespace, env: dict[str, str] | None = None, **over: str
) -> subprocess.CompletedProcess[str]:
    run_env = ssh_env(box)
    if env:
        run_env.update(env)
    return run_cli(base_args(box, **over), env=run_env)


def expected_proof_bytes(
    collected_at: str = NOW_ISO, host_label: str = DEFAULT_HOST
) -> bytes:
    payload = {
        "schema": SCHEMA,
        "host_label": host_label,
        "collected_at": collected_at,
        "tasker": {"active": False, "enabled": False},
        "archive_timer": {"active": False, "enabled": False},
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def assert_closed_failure(result: subprocess.CompletedProcess[str]) -> None:
    """The universal fail-closed contract: exit 1, empty stdout, static
    sanitized stderr (no tracebacks, no host/token/remote output)."""
    assert result.returncode == 1
    assert result.stdout == ""
    assert "Traceback" not in result.stderr
    for forbidden in ("inactive", "disabled", "active", "enabled", "systemctl"):
        assert forbidden not in result.stderr, f"stderr leaked {forbidden!r}"


def assert_no_temp_litter(box: SimpleNamespace) -> None:
    leftovers = [p for p in box.tmp.rglob(".pspr-*")]
    assert leftovers == [], f"temp files left behind: {leftovers}"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("pspr_under_test", CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── production defaults and never-scheduled contract ──────────────────────


SCHEDULING_SURFACES = (
    "Makefile",
    "crontab",
    "config/tasker.yaml",
)
EXECUTABLE_CRON_SUFFIXES = (".sh", ".py", ".service", ".timer")


def test_production_defaults_and_not_scheduled_anywhere():
    mod = _load_cli_module()
    assert mod.DEFAULT_PROOF == DEFAULT_PROOF
    assert mod.DEFAULT_SSH == DEFAULT_SSH
    assert mod.DEFAULT_HOST == DEFAULT_HOST
    assert mod.DEFAULT_CONNECT_TIMEOUT == DEFAULT_CONNECT_TIMEOUT
    name = CLI.name
    surfaces = [PROJECT_ROOT / rel for rel in SCHEDULING_SURFACES]
    cron_dir = PROJECT_ROOT / "scripts" / "cron"
    executable = sorted(
        path for path in cron_dir.rglob("*")
        if path.is_file() and path.suffix in EXECUTABLE_CRON_SUFFIXES
    )
    assert executable, "the drift guard must actually see cron executables"
    surfaces += executable
    checked = 0
    for source in surfaces:
        text = source.read_text(encoding="utf-8", errors="replace")
        assert name not in text, f"{name} must never be scheduled; found in {source}"
        checked += 1
    assert checked >= 4


def test_env_overrides_match_flag_overrides(box):
    env = ssh_env(box)
    env.update({
        "PLSPR_PROOF": str(box.proof),
        "PLSPR_SSH": str(box.fake_ssh),
        "PLSPR_CONNECT_TIMEOUT": "5",
        "PLSPR_NOW": NOW,
    })
    result = run_cli(env=env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() != ""
    assert box.proof.read_bytes() == expected_proof_bytes()


# ── happy path / determinism / mode / schema ──────────────────────────────


def test_happy_path_writes_canonical_proof(box):
    result = run_refresh(box)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 1, "exactly one compact JSON summary line on stdout"
    assert json.loads(lines[0]) == {
        "schema": SCHEMA,
        "category": "authority",
        "status": "pass",
        "host_label": DEFAULT_HOST,
        "collected_at": NOW_ISO,
        "tasker": {"active": False, "enabled": False},
        "archive_timer": {"active": False, "enabled": False},
    }
    st = box.proof.lstat()
    assert stat.S_ISREG(st.st_mode) and not stat.S_ISLNK(st.st_mode)
    assert st.st_uid == os.getuid()
    assert stat.S_IMODE(st.st_mode) == 0o600
    assert box.proof.read_bytes() == expected_proof_bytes()
    assert_no_temp_litter(box)


def test_pinned_now_reruns_are_byte_identical(box):
    first = run_refresh(box)
    assert first.returncode == 0, first.stderr
    first_stdout = first.stdout
    first_bytes = box.proof.read_bytes()
    time.sleep(1.1)  # wall clock must not leak into the artifact
    second = run_refresh(box)
    assert second.returncode == 0, second.stderr
    assert second.stdout == first_stdout
    assert box.proof.read_bytes() == first_bytes
    assert json.loads(box.proof.read_bytes())["collected_at"] == NOW_ISO


def test_probe_argv_is_explicit_batchmode_and_bounded(box):
    result = run_refresh(box)
    assert result.returncode == 0, result.stderr
    calls = [
        json.loads(line)
        for line in box.argv_log.read_text(encoding="utf-8").splitlines()
    ]
    assert len(calls) == 4
    for (subcommand, unit), argv in zip(PROBE_KEYS, calls):
        assert argv == [
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=5",
            DEFAULT_HOST,
            "systemctl", subcommand, unit,
        ], argv
    all_argv = " ".join(json.dumps(call) for call in calls)
    for forbidden in (
        "StrictHostKeyChecking=no", "UserKnownHostsFile=/dev/null",
        "PasswordAuthentication", "shell=True", ";", "|",
    ):
        assert forbidden not in all_argv, f"argv must not contain {forbidden!r}"


# ── fail closed: active/enabled, missing ssh, timeout, nonzero, tokens ────


@pytest.mark.parametrize(
    ("subcommand", "unit", "out", "exit"),
    [
        ("is-active", TASKER_SERVICE, "active\n", 0),
        ("is-enabled", TASKER_SERVICE, "enabled\n", 0),
        ("is-active", ARCHIVE_TIMER_SERVICE, "active\n", 0),
        ("is-enabled", ARCHIVE_TIMER_SERVICE, "enabled\n", 0),
    ],
)
def test_any_active_or_enabled_fails_closed(box, subcommand, unit, out, exit):
    responses = inactive_responses()
    responses[f"{subcommand} {unit}"] = {"exit": exit, "out": out}
    set_ssh_responses(box, responses)
    existing = expected_proof_bytes(collected_at="2026-09-04T00:00:00+00:00")
    box.proof.write_bytes(existing)
    box.proof.chmod(0o600)
    result = run_refresh(box)
    assert_closed_failure(result)
    assert box.proof.read_bytes() == existing, "existing proof must be preserved"
    assert_no_temp_litter(box)


def test_missing_ssh_fails_closed(box):
    existing = expected_proof_bytes()
    box.proof.write_bytes(existing)
    box.proof.chmod(0o600)
    result = run_refresh(box, ssh=str(box.tmp / "no-such-ssh"))
    assert_closed_failure(result)
    assert "ssh" in result.stderr.lower()
    assert box.proof.read_bytes() == existing
    assert_no_temp_litter(box)


def test_probe_timeout_fails_closed(box):
    responses = inactive_responses()
    responses["is-active portfolio-lab-tasker"] = {
        "exit": 3, "out": "inactive\n", "delay": 12,
    }
    set_ssh_responses(box, responses)
    result = run_refresh(box, connect_timeout="1")
    assert_closed_failure(result)
    assert "timed out" in result.stderr.lower()
    assert not box.proof.exists(), "no new proof on timeout"
    assert_no_temp_litter(box)


@pytest.mark.parametrize(
    ("subcommand", "unit", "out", "exit"),
    [
        # Right token, wrong exit code (nonzero, outside the canonical set).
        ("is-active", TASKER_SERVICE, "inactive\n", 2),
        ("is-active", TASKER_SERVICE, "inactive\n", 0),
        ("is-enabled", TASKER_SERVICE, "disabled\n", 2),
        ("is-enabled", TASKER_SERVICE, "disabled\n", 0),
        ("is-active", ARCHIVE_TIMER_SERVICE, "inactive\n", 4),
        ("is-enabled", ARCHIVE_TIMER_SERVICE, "disabled\n", 3),
        # Generic nonzero failure with no output at all.
        ("is-active", TASKER_SERVICE, "", 255),
    ],
)
def test_nonzero_or_unexpected_exit_fails_closed(box, subcommand, unit, out, exit):
    responses = inactive_responses()
    responses[f"{subcommand} {unit}"] = {"exit": exit, "out": out}
    set_ssh_responses(box, responses)
    result = run_refresh(box)
    assert_closed_failure(result)
    assert not box.proof.exists(), "no new proof on probe failure"
    assert_no_temp_litter(box)


@pytest.mark.parametrize(
    ("token", "exit"),
    [
        ("failed\n", 3),
        ("unknown\n", 4),
        ("masked\n", 1),
        ("not-found\n", 5),
        ("static\n", 0),
        ("indirect\n", 0),
        ("activating\n", 0),
        ("deactivating\n", 0),
        # malformed tokens
        ("inactive", 3),           # missing trailing newline
        ("inactive\n\n", 3),       # extra newline
        ("inactive\r\n", 3),       # CRLF
        (" inactive\n", 3),        # leading whitespace
        ("inactive \n", 3),        # trailing whitespace
        ("inactive active\n", 0),  # embedded token
        ("inactive\nactive\n", 0),
        ("INACTIVE\n", 3),         # wrong case
        ("", 0),                   # empty stdout
    ],
)
def test_unexpected_or_malformed_tokens_fail_closed(box, token, exit):
    responses = inactive_responses()
    responses["is-active portfolio-lab-tasker"] = {"exit": exit, "out": token}
    set_ssh_responses(box, responses)
    result = run_refresh(box)
    assert_closed_failure(result)
    assert not box.proof.exists(), "no new proof on malformed token"
    assert_no_temp_litter(box)


# ── config validation: paths, timeout, host ───────────────────────────────


def test_relative_proof_path_fails_closed(box):
    result = run_cli(
        ["--proof", "relative/former-authority-proof.json",
         "--ssh", str(box.fake_ssh), "--connect-timeout", "5", "--now", NOW],
        env=ssh_env(box),
    )
    assert_closed_failure(result)
    assert "absolute" in result.stderr.lower()
    assert not (box.tmp / "relative").exists()


def test_symlink_proof_path_fails_closed(box):
    victim = box.tmp / "victim-proof.json"
    victim.write_bytes(expected_proof_bytes())
    box.proof.symlink_to(victim)
    result = run_refresh(box)
    assert_closed_failure(result)
    assert "symlink" in result.stderr.lower()
    assert box.proof.is_symlink(), "rejected symlink must be left in place"
    assert victim.read_bytes() == expected_proof_bytes(), "symlink target untouched"


@pytest.mark.parametrize("value", ["0", "-1", "0.00001", "abc", "nan", "inf", "31"])
def test_invalid_connect_timeout_fails_closed(box, value):
    result = run_refresh(box, connect_timeout=value)
    assert_closed_failure(result)
    assert not box.proof.exists()


def test_ssh_override_must_be_absolute(box):
    result = run_refresh(box, ssh="fake-ssh")
    assert_closed_failure(result)
    assert "absolute" in result.stderr.lower()


def test_host_must_be_exactly_sg01(box):
    result = run_refresh(box, host="sg02")
    assert result.returncode == 1
    assert result.stdout == ""
    assert "Traceback" not in result.stderr
    # The rejection names only the fixed literal host, never the supplied one.
    assert result.stderr.splitlines() == ["ERROR: PLSPR_HOST must be exactly sg01"]
    assert "sg02" not in result.stderr
    assert not box.proof.exists()


def test_unknown_flag_exits_one_with_empty_stdout(box):
    result = run_cli(base_args(box) + ["--schedule", "daily"], env=ssh_env(box))
    assert result.returncode == 1
    assert result.stdout == ""
    assert "Traceback" not in result.stderr
    assert not box.proof.exists()


# ── ssh override resolution: every override must be absolute ──────────────


def test_default_ssh_is_retained_only_when_the_override_is_omitted():
    mod = _load_cli_module()
    cfg = mod.Config(mod.parse_args(["--now", NOW]))
    assert cfg.ssh == DEFAULT_SSH


def test_ssh_override_naming_the_default_must_still_be_absolute(box):
    result = run_refresh(box, ssh=DEFAULT_SSH)
    assert_closed_failure(result)
    assert "absolute" in result.stderr.lower()
    assert not box.proof.exists()


def test_ssh_env_override_naming_the_default_must_still_be_absolute(box):
    env = ssh_env(box)
    env.update({
        "PLSPR_SSH": DEFAULT_SSH,
        "PLSPR_PROOF": str(box.proof),
        "PLSPR_NOW": NOW,
    })
    result = run_cli(env=env)
    assert_closed_failure(result)
    assert "absolute" in result.stderr.lower()
    assert not box.proof.exists()


# ── static-error contract: reachable failures never traceback ─────────────


@pytest.mark.parametrize("value", ["0001-01-01T00:00:00+23:59", "0001-01-01T00:00:01+00:02"])
def test_timezone_underflow_fails_closed_without_traceback(box, value):
    """``astimezone`` raises OverflowError on extreme aware dates; that must
    never escape as a traceback."""
    result = run_refresh(box, now=value)
    assert result.returncode == 1
    assert result.stdout == ""
    assert "Traceback" not in result.stderr
    assert "OverflowError" not in result.stderr
    assert result.stderr.splitlines()[0].startswith("ERROR: PLSPR_NOW")
    assert len(result.stderr.splitlines()) == 1
    assert not box.proof.exists()


def test_extreme_but_convertible_now_is_accepted(box):
    result = run_refresh(box, now="9999-12-31T23:00:00+23:59")
    assert result.returncode == 0, result.stderr
    assert json.loads(box.proof.read_bytes())["collected_at"] == (
        "9999-12-30T23:01:00+00:00"
    )


@pytest.mark.parametrize("value", ["2026-09-05T04:00:00", "2026-09-05 04:00:00"])
def test_naive_now_is_rejected(box, value):
    """A naive timestamp has no defined zone here: rejected, never assumed."""
    result = run_refresh(box, now=value)
    assert result.returncode == 1
    assert result.stdout == ""
    assert "Traceback" not in result.stderr
    assert "offset" in result.stderr
    assert not box.proof.exists()


def test_probe_output_overrun_fails_closed(box):
    responses = inactive_responses()
    responses["is-active portfolio-lab-tasker"] = {
        "exit": 3, "out": "inactive\n" + "x" * 5000,
    }
    set_ssh_responses(box, responses)
    result = run_refresh(box)
    assert_closed_failure(result)
    assert "bounded" in result.stderr
    assert "inactive" not in result.stderr
    assert not box.proof.exists()
    assert_no_temp_litter(box)


def test_probe_stderr_overrun_fails_closed(box):
    responses = inactive_responses()
    responses["is-enabled portfolio-lab-s3-archive.timer"] = {
        "exit": 1, "out": "disabled\n", "err": "y" * 5000,
    }
    set_ssh_responses(box, responses)
    result = run_refresh(box)
    assert_closed_failure(result)
    assert "bounded" in result.stderr
    assert not box.proof.exists()
    assert_no_temp_litter(box)


class _StubProbe:
    """Popen-shaped stub over real, already-EOF fds that raises on one
    lifecycle call so cleanup paths can be exercised deterministically."""

    def __init__(self, raise_on: str) -> None:
        out_read, out_write = os.pipe()
        err_read, err_write = os.pipe()
        self.stdout = os.fdopen(out_read, "rb")
        self.stderr = os.fdopen(err_read, "rb")
        os.close(out_write)
        os.close(err_write)
        self.raise_on = raise_on
        self.kill_attempted = False
        self.returncode = None

    def poll(self) -> int | None:
        if self.raise_on == "poll":
            raise OSError("simulated poll failure")
        return 0

    def kill(self) -> None:
        self.kill_attempted = True
        if self.raise_on == "kill":
            raise OSError("simulated kill failure")

    def wait(self, timeout: float | None = None) -> int:
        return 0


@pytest.mark.parametrize("raise_on", ["poll", "kill", "fcntl"])
def test_probe_lifecycle_errors_are_cleaned_and_fail_closed(
    raise_on, monkeypatch, capsys, tmp_path
):
    """fcntl/poll/kill errors must reap the child, close both fds, and exit 1
    with one static line — never a traceback."""
    mod = _load_cli_module()
    stub = _StubProbe(raise_on)
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **k: stub)
    if raise_on == "fcntl":
        def boom(*_args: object) -> None:
            raise OSError("simulated fcntl failure")
        monkeypatch.setattr(mod.fcntl, "fcntl", boom)
    proof = tmp_path / "stub-proof.json"
    with pytest.raises(SystemExit) as excinfo:
        mod.main([
            "--proof", str(proof), "--ssh", "/bin/true",
            "--connect-timeout", "5", "--now", NOW,
        ])
    captured = capsys.readouterr()
    assert excinfo.value.code == 1
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert len(captured.err.splitlines()) == 1
    assert captured.err.startswith("ERROR: ")
    assert stub.kill_attempted is True, "child must be killed best-effort"
    assert stub.stdout.closed and stub.stderr.closed, "probe fds must be closed"
    assert not proof.exists(), "no proof may be written on a probe error"


# ── reporting pipe ────────────────────────────────────────────────────────


def test_closed_stderr_still_exits_one_without_anomaly(monkeypatch, box):
    """A dead diagnostic stream cannot break the exit-1 contract: the static
    line is dropped, nothing escapes, and no proof is written."""
    mod = _load_cli_module()

    class _Broken:
        def write(self, *_args: object) -> int:
            raise BrokenPipeError("stderr is closed")

        def flush(self) -> None:
            raise BrokenPipeError("stderr is closed")

    monkeypatch.setattr(sys, "stderr", _Broken())
    assert mod.run([
        "--proof", "relative.json", "--ssh", str(box.fake_ssh), "--now", NOW,
    ]) == 1


def test_broken_stdout_exits_one_without_interpreter_shutdown_anomaly(box):
    """A closed stdout must not produce 'Exception ignored' noise or the
    shutdown exit code; the already-committed proof is left valid."""
    read_fd, write_fd = os.pipe()
    os.close(read_fd)  # every write to the write end is EPIPE
    env = dict(os.environ)
    env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    env.update(ssh_env(box))
    proc = subprocess.Popen(
        [sys.executable, str(CLI), *base_args(box)],
        stdout=write_fd,
        stderr=subprocess.PIPE,
        pass_fds=(write_fd,),
        env=env,
        text=True,
    )
    try:
        _, stderr = proc.communicate(timeout=120)
    finally:
        os.close(write_fd)
    assert proc.returncode == 1, stderr
    assert "Exception ignored" not in stderr
    assert "Traceback" not in stderr
    assert stderr.splitlines() == ["ERROR: stdout is closed"]
    # The atomic commit already succeeded; a dead reporting pipe must not
    # destroy a verified artifact (the collector validates it independently).
    assert box.proof.read_bytes() == expected_proof_bytes()


# ── missing-unit semantics: explicit, documented, fail-closed ─────────────


@pytest.mark.parametrize("exit_code", [1, 4, 5])
def test_missing_unit_not_found_stays_fail_closed(box, exit_code):
    """``not-found`` is deliberately NOT accepted: the (token, exit) pair for
    an absent unit file is not stable across systemd versions, so accepting
    any one guess would either dead-end or over-accept. The recycle runbook
    documents the attended gate instead."""
    responses = inactive_responses()
    responses["is-enabled portfolio-lab-tasker"] = {"exit": exit_code, "out": "not-found\n"}
    set_ssh_responses(box, responses)
    result = run_refresh(box)
    assert_closed_failure(result)
    assert not box.proof.exists()
    assert_no_temp_litter(box)


def test_missing_unit_dbus_error_shape_stays_fail_closed(box):
    """Other systemd versions answer the missing-unit case with an error on
    stderr and empty stdout; that must also fail closed and never be echoed."""
    responses = inactive_responses()
    responses["is-enabled portfolio-lab-s3-archive.timer"] = {
        "exit": 1,
        "out": "",
        "err": "Failed to get unit file state for portfolio-lab-s3-archive.timer: "
               "No such file or directory\n",
    }
    set_ssh_responses(box, responses)
    result = run_refresh(box)
    assert_closed_failure(result)
    assert "Failed to get unit file state" not in result.stderr
    assert "No such file" not in result.stderr
    assert not box.proof.exists()


def test_not_found_token_never_proves_disabled_state(box):
    """A not-found answer on the active probe is equally unacceptable."""
    responses = inactive_responses()
    responses["is-active portfolio-lab-s3-archive.timer"] = {"exit": 4, "out": "not-found\n"}
    set_ssh_responses(box, responses)
    result = run_refresh(box)
    assert_closed_failure(result)
    assert not box.proof.exists()


# ── proof preservation, temp cleanup, write failure ───────────────────────


def test_existing_proof_preserved_byte_for_byte_on_every_failure(box):
    responses = inactive_responses()
    responses["is-enabled portfolio-lab-s3-archive.timer"] = {
        "exit": 1, "out": "masked\n",
    }
    set_ssh_responses(box, responses)
    existing = expected_proof_bytes(collected_at="2026-09-04T00:00:00+00:00")
    box.proof.write_bytes(existing)
    box.proof.chmod(0o600)
    before = box.proof.lstat()
    result = run_refresh(box)
    assert_closed_failure(result)
    assert box.proof.read_bytes() == existing
    assert box.proof.lstat().st_mode == before.st_mode
    assert_no_temp_litter(box)


def test_write_failure_is_clean_and_fails_closed(box):
    missing_parent = box.tmp / "no-such-dir"
    result = run_refresh(box, proof=str(missing_parent / "proof.json"))
    assert_closed_failure(result)
    assert not missing_parent.exists(), "no directories may be created"
    assert_no_temp_litter(box)


def test_temp_artifact_verified_before_replace_so_existing_proof_survives(monkeypatch, box):
    """The temp artifact is verified (regular, non-symlink, owner uid, exact
    0600, exact size) before the rename: a verification failure must leave the
    existing proof untouched byte-for-byte."""
    mod = _load_cli_module()
    existing = expected_proof_bytes(collected_at="2026-09-04T00:00:00+00:00")
    box.proof.write_bytes(existing)
    box.proof.chmod(0o600)

    def boom(_path: str, _size: int) -> None:
        raise OSError("simulated artifact verification failure")

    monkeypatch.setattr(mod, "verify_artifact", boom)
    with pytest.raises(OSError):
        mod.replace_proof(box.proof, expected_proof_bytes())
    assert box.proof.read_bytes() == existing
    assert stat.S_IMODE(box.proof.lstat().st_mode) == 0o600
    assert_no_temp_litter(box)


def test_verify_artifact_rejects_wrong_mode_size_and_type(box):
    mod = _load_cli_module()
    good = box.tmp / "good.json"
    good.write_bytes(b"{}")
    good.chmod(0o600)
    mod.verify_artifact(str(good), 2)  # regular, 0600, owner uid, exact size
    with pytest.raises(OSError):
        mod.verify_artifact(str(good), 3)  # wrong size
    wide = box.tmp / "wide.json"
    wide.write_bytes(b"{}")
    wide.chmod(0o644)
    with pytest.raises(OSError):
        mod.verify_artifact(str(wide), 2)  # wrong mode
    link = box.tmp / "link.json"
    link.symlink_to(good)
    with pytest.raises(OSError):
        mod.verify_artifact(str(link), 2)  # symlink
    with pytest.raises(OSError):
        mod.verify_artifact(str(box.tmp / "missing.json"), 2)  # unreadable


def test_temp_cleanup_on_write_failure(monkeypatch, tmp_path):
    mod = _load_cli_module()
    parent = Path(tempfile.mkdtemp(dir=str(tmp_path)))
    target = parent / "former-authority-proof.json"

    def boom_chmod(_path: str, _mode: int) -> None:
        raise OSError("simulated chmod failure")

    monkeypatch.setattr(os, "chmod", boom_chmod)
    with pytest.raises(OSError):
        mod.replace_proof(target, expected_proof_bytes())
    assert not target.exists()
    assert [p for p in parent.iterdir() if p.name.startswith(".pspr-")] == []


# ── safety / leak assertions ──────────────────────────────────────────────


def test_failure_stderr_is_static_and_remote_output_never_leaks(box):
    """Noisy remote stderr/stdout (ssh banner + fake secret) must never
    appear on our stderr; failure stderr is one static line only."""
    responses = inactive_responses()
    responses["is-active portfolio-lab-tasker"] = {
        "exit": 0,
        "out": "active\nSUPERSECRETREMOTE42\n",
        "err": "Warning: Permanently added the RSA host key.\r\n",
    }
    set_ssh_responses(box, responses)
    result = run_refresh(box)
    assert_closed_failure(result)
    for forbidden in (
        "SUPERSECRETREMOTE42", "Warning:", "RSA", "host key",
        str(box.proof), str(box.fake_ssh), str(box.tmp), "sg01",
    ):
        assert forbidden not in result.stderr, f"stderr leaked {forbidden!r}"
    assert len(result.stderr.splitlines()) == 1, "stderr must be one static line"


def test_success_stdout_is_secret_free(box):
    result = run_refresh(box)
    assert result.returncode == 0, result.stderr
    for forbidden in (str(box.proof), str(box.fake_ssh), str(box.tmp), "ssh"):
        assert forbidden not in result.stdout, f"stdout leaked {forbidden!r}"


# ── integration: existing daily collector accepts the generated proof ─────


def _write_exec(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)
    return path


def write_fake_controller(path: Path) -> Path:
    """One env-driven fake for both persist controllers (the tasker role is
    detected by ``--app-dir`` in argv; FT_/FS_ env prefixes)."""
    source = f"""#!{sys.executable}
import json, os, sys
tasker_role = "--app-dir" in sys.argv
prefix = "FT" if tasker_role else "FS"
exit_code = os.environ.get(prefix + "_EXIT", "")
if exit_code:
    sys.exit(int(exit_code))
payload = os.environ.get(prefix + "_PAYLOAD", "")
if payload:
    sys.stdout.write(payload)
sys.exit(0)
"""
    return _write_exec(path, source)


@contextmanager
def http_pair(api_body: bytes) -> tuple[str, str]:
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (stdlib name)
            body = api_body if self.path == "/api/tasker/status" else b"<html>ok</html>"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        yield f"http://127.0.0.1:{port}/api/tasker/status", f"http://127.0.0.1:{port}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


@pytest.fixture
def lab(tmp_path: Path) -> SimpleNamespace:
    """A production-shaped cursor-box layout for the daily collector."""
    root = tmp_path / "lab-root"
    app = root / "app"
    www = root / "www"
    (app / "data").mkdir(parents=True)
    (www / "data").mkdir(parents=True)
    run = root / "run"
    run.mkdir()
    ref = NOW_DT
    for rel in ("app/data/signals.json", "app/data/tasker_status.json",
                "www/data/signals.json", "www/data/tasker_status.json"):
        path = root / rel
        path.write_text(
            json.dumps({"allocation": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}}),
            encoding="utf-8",
        )
        os.utime(path, (ref.timestamp() - 600, ref.timestamp() - 600))
    (run / "s3-archive-last-utc-day").write_text(
        ref.strftime("%Y%m%d") + "\n", encoding="utf-8"
    )
    (run / "s3-archive.log").write_text(
        f"[{ref.strftime('%Y-%m-%dT%H:%M:%SZ')}] published "
        "s3://portfolio-lab-archives/daily/2026/09/05/"
        f"portfolio-lab-data-20260905_035500Z.tar sha256={'a' * 64}\n",
        encoding="utf-8",
    )
    api_payload = {
        "service": TASKER_SERVICE,
        "backend": "tasker",
        "timestamp": "2026-09-05T04:00:00Z",
        "tasks": [{
            "id": "portfolio-lab-data", "label": "Data pipeline",
            "command": "make fetch-secrets", "schedule": "5 * * * *",
            "enabled": True, "manual_only": False, "timeout_seconds": 3600,
            "paused": False, "pause_reason": None, "last_status": "success",
            "last_run_id": "run-20260905t035500-abcdef12",
            "last_finished_at": "2026-09-05T04:00:00Z",
            "last_duration_seconds": 42.5, "failure_count": 0,
            "consecutive_failures": 0,
        }],
        "recent_runs": [{
            "run_id": "run-20260905t035500-abcdef12",
            "task_id": "portfolio-lab-data", "command": ["make", "data"],
            "trigger": "schedule", "retry_of": None, "status": "success",
            "pid": 9999, "started_at": "2026-09-05T03:55:00Z",
            "finished_at": "2026-09-05T04:00:00Z", "duration_seconds": 42.5,
            "exit_code": 0, "error": None, "termination_cause": None,
            "termination_detail": None,
            "log_path": "run-20260905t035500-abcdef12.log",
            "created_at": "2026-09-05T03:55:00Z",
            "updated_at": "2026-09-05T04:00:00Z",
        }],
    }
    tasker_payload = {
        "schema": "portfolio-lab-box-persist/v1", "state": "active",
        "scheduler_mode": "enabled", "identity_exact": True,
        "scheduler_instances": 1, "pid": 1234,
        "service_name": TASKER_SERVICE, "mode": "production",
        "app_dir": str(app.resolve()), "web_root": str(www.resolve()),
    }
    static_payload = {
        "schema": "portfolio-lab-static-persist/v1", "state": "active",
        "identity_exact": True, "service_name": "portfolio-lab-static",
        "mode": "production", "web_root": str(www.resolve()),
    }
    return SimpleNamespace(
        root=root,
        app=app,
        www=www,
        run=run,
        out=tmp_path / "evidence-out",
        tasker_controller=write_fake_controller(tmp_path / "fake-tasker"),
        static_controller=write_fake_controller(tmp_path / "fake-static"),
        api_body=json.dumps(api_payload).encode(),
        tasker_payload=json.dumps(tasker_payload),
        static_payload=json.dumps(static_payload),
    )


def run_daily(
    lab: SimpleNamespace,
    proof_path: Path,
    now: str,
    api_url: str,
    static_url: str,
) -> subprocess.CompletedProcess[str]:
    env = {
        "PLDE_ROOT": str(lab.root),
        "PLDE_OUTPUT_ROOT": str(lab.out),
        "PLDE_TASKER_CONTROLLER": str(lab.tasker_controller),
        "PLDE_STATIC_CONTROLLER": str(lab.static_controller),
        "PLDE_API_URL": api_url,
        "PLDE_STATIC_URL": static_url,
        "PLDE_AUTHORITY_PROOF": str(proof_path),
        "FT_PAYLOAD": lab.tasker_payload,
        "FS_PAYLOAD": lab.static_payload,
    }
    return run_cli(
        ["--now", now, "--timeout", "10", "--freshness-max-age", "3600"],
        env=env,
        cli=DAILY_CLI,
    )


def test_daily_collector_accepts_generated_proof_with_pinned_now(lab, box):
    generated = box.tmp / "integ-proof.json"
    refreshed = run_refresh(box, proof=str(generated))
    assert refreshed.returncode == 0, refreshed.stderr

    with http_pair(lab.api_body) as (api_url, static_url):
        result = run_daily(lab, generated, NOW, api_url, static_url)
    assert result.returncode == 0, result.stderr
    day_dir = lab.out / "2026-09-05"
    assert day_dir.is_dir()
    authority = json.loads((day_dir / "authority.json").read_text(encoding="utf-8"))
    details = authority["details"]
    assert authority["status"] == "pass"
    assert details["status"] == "pass"
    assert details["present"] is True
    assert details["host_label"] == DEFAULT_HOST
    assert details["collected_at"] == NOW_ISO
    assert details["tasker_active"] is False
    assert details["tasker_enabled"] is False
    assert details["archive_timer_active"] is False
    assert details["archive_timer_enabled"] is False
    summary = json.loads((day_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["details"]["categories"]["authority"] == "pass"


def test_daily_collector_warns_when_pinned_now_earlier_than_proof(lab, box):
    generated = box.tmp / "integ-proof.json"
    refreshed = run_refresh(box, proof=str(generated), now="2026-09-05T06:00:00Z")
    assert refreshed.returncode == 0, refreshed.stderr
    assert json.loads(generated.read_bytes())["collected_at"] == (
        "2026-09-05T06:00:00+00:00"
    )

    with http_pair(lab.api_body) as (api_url, static_url):
        # A pinned --now earlier than collected_at future-dates the proof.
        result = run_daily(lab, generated, NOW, api_url, static_url)
    assert result.returncode == 0, result.stderr  # warning is exit 0
    authority = json.loads(
        (lab.out / "2026-09-05" / "authority.json").read_text(encoding="utf-8")
    )
    assert authority["status"] == "warning"
    assert "future" in authority["details"]["reason"]


def test_recycle_runbook_documents_attended_refresh_command():
    text = RECYCLE_MD.read_text(encoding="utf-8")
    assert f"python3 app/scripts/{CLI.name}" in text, (
        "runbook must document the attended refresh command"
    )
    assert SCHEMA in text
    assert "attended" in text.lower()
    # The missing-unit (not-found) case must be an explicit attended gate.
    assert "not-found" in text, "runbook must document the missing-unit gate"
