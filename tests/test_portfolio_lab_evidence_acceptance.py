"""Strict-TDD tests for the bounded seven-day evidence acceptance checker.

Every test drives the shipped CLI module
(``scripts/portfolio_lab_evidence_acceptance.py``) in-process against
isolated temp evidence roots built exactly like the daily evidence
collector's output (dirs 0700, files 0600, envelope schema
``portfolio-lab-daily-evidence/v1``). No network, SSH, Docker, subprocess,
or service mutation.

Coverage: pass seven days with a valid recycle proof (accept); seven pass
days without recycle (ready_except_recycle); absent recycle with
--require-recycle-proof (extend); missing day and absent evidence root
(extend); archive/daily warning (extend); category fail, summary map/overall/
notify mismatch, perms/symlink/extra-entry/oversize layouts, duplicate
archive sha256, wrong archive utc day / latest_success day / sha format,
scheduler count != 1, endpoint/freshness/disk/authority criterion violations,
malformed envelopes (schema/category/status/collected_at/JSON), invalid and
malformed recycle proofs (investigate); malformed CLI and unsafe output-json
placement (exit 1); output-json mode 0600 and idempotence.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import stat
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXE = PROJECT_ROOT / "scripts" / "portfolio_lab_evidence_acceptance.py"

DAILY_SCHEMA = "portfolio-lab-daily-evidence/v1"
RECYCLE_SCHEMA = "portfolio-lab-recycle-proof/v1"
REPORT_SCHEMA = "portfolio-lab-evidence-acceptance/v1"
EXPECTED_HOST = "sg01"

END_DAY = "2026-09-05"
START_DAY = "2026-08-30"
CATEGORIES = ("tasker", "jobs", "freshness", "archive", "resources", "authority")
GIB = 1024**3
MAX_EVIDENCE_BYTES = 262144


@pytest.fixture(scope="module")
def checker() -> object:
    spec = importlib.util.spec_from_file_location("pl_evidence_acceptance", EXE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module  # dataclasses introspection needs the module registered
    spec.loader.exec_module(module)
    return module


def day_span(end_day: str = END_DAY) -> list[str]:
    end = datetime.strptime(end_day, "%Y-%m-%d").date()
    return [(end - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]


def envelope(category: str, status: str, details: object, day: str) -> dict[str, object]:
    return {
        "schema": DAILY_SCHEMA,
        "category": category,
        "collected_at": f"{day}T04:00:00+00:00",
        "status": status,
        "details": details,
    }


def _details(category: str, day: str, sha: str | None = None) -> dict[str, object]:
    if category == "tasker":
        return {
            "tasker_controller": {
                "status": "pass", "state": "active", "mode": "production",
                "identity_exact": True, "service_name_exact": True,
                "scheduler_instances": 1, "paths_exact": True,
            },
            "static_controller": {
                "status": "pass", "state": "active", "mode": "production",
                "identity_exact": True, "service_name_exact": True, "paths_exact": True,
            },
        }
    if category == "jobs":
        return {
            "api": {"status": "pass", "http_status": 200, "tasks_retained": 2,
                    "runs_retained": 1, "tasks": [], "recent_runs": []},
            "static_root": {"status": "pass", "http_status": 200,
                            "content_type": "text/html; charset=utf-8", "bytes": 4},
        }
    if category == "freshness":
        return {
            "files": [
                {"path": rel, "status": "pass", "present": True, "age_seconds": 5,
                 "mtime": f"{day}T03:00:00+00:00", "size_bytes": 5}
                for rel in ("app/data/signals.json", "app/data/tasker_status.json",
                            "www/data/signals.json", "www/data/tasker_status.json")
            ]
        }
    if category == "archive":
        digest = sha or hashlib.sha256(day.encode("ascii")).hexdigest()
        return {"utc_day": day, "latest_success": f"{day}T03:00:00Z", "sha256": digest}
    if category == "resources":
        return {
            "disk": {"status": "pass", "total_bytes": 500 * GIB,
                     "used_bytes": 400 * GIB, "free_bytes": 100 * GIB, "percent": 80.0},
            "memory": {"mem_total_kb": 8388608, "mem_available_kb": 2097152},
        }
    return {
        "status": "pass", "present": True, "host_label": EXPECTED_HOST,
        "collected_at": f"{day}T03:30:00+00:00",
        "tasker_active": False, "tasker_enabled": False,
        "archive_timer_active": False, "archive_timer_enabled": False,
    }


def day_payloads(
    day: str, statuses: dict[str, str] | None = None, sha: str | None = None
) -> dict[str, dict[str, object]]:
    statuses = dict.fromkeys(CATEGORIES, "pass") if statuses is None else dict(statuses)
    payloads = {
        f"{category}.json": envelope(category, statuses[category], _details(category, day, sha), day)
        for category in CATEGORIES
    }
    overall = "fail" if "fail" in statuses.values() else ("warning" if "warning" in statuses.values() else "pass")
    summary_details = {
        "utc_day": day, "overall": overall,
        "notify_grok_bot": overall == "fail", "categories": dict(statuses),
    }
    payloads["summary.json"] = envelope("summary", overall, summary_details, day)
    return payloads


def write_day(
    root: Path,
    day: str,
    payloads: dict[str, dict[str, object]] | None = None,
    statuses: dict[str, str] | None = None,
    *,
    dir_mode: int = 0o700,
    file_mode: int = 0o600,
    extra: tuple[str, ...] = (),
) -> None:
    payloads = day_payloads(day, statuses=statuses) if payloads is None else payloads
    day_dir = root / day
    day_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    for name, payload in payloads.items():
        path = day_dir / name
        path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True), encoding="utf-8")
        os.chmod(path, file_mode)
    for name in extra:
        path = day_dir / name
        path.write_text("unexpected", encoding="utf-8")
        os.chmod(path, 0o600)
    os.chmod(day_dir, dir_mode)


def write_window(root: Path) -> None:
    for day in day_span():
        write_day(root, day)


def recycle_payload(
    *,
    performed_at: str = "2026-09-04T05:30:00+00:00",
    schema: str = RECYCLE_SCHEMA,
    category: str = "recycle",
    status: str = "pass",
    pre_scheduler: int = 1,
    post_scheduler: int = 1,
    components: str = "pass",
) -> dict[str, object]:
    comp = {"tasker": components, "static": components, "tunnel": components, "api": components}
    return {
        "schema": schema, "category": category, "status": status,
        "collected_at": "2026-09-04T06:00:00+00:00",
        "details": {
            "performed_at": performed_at,
            "pre_recycle": {"scheduler_instances": pre_scheduler, **comp},
            "post_recycle": {"scheduler_instances": post_scheduler, **comp},
        },
    }


def write_recycle(root: Path, payload: object = None, mode: int = 0o600) -> None:
    payload = recycle_payload() if payload is None else payload
    path = root / "recycle.json"
    path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    os.chmod(path, mode)


def run_main(mod: object, root: Path, *extra: str) -> int:
    return mod.main(["--evidence-root", str(root), "--end-day", END_DAY, *extra])


def read_report(capsys: object) -> dict[str, object]:
    return json.loads(capsys.readouterr().out)


# ── accept and ready_except_recycle ───────────────────────────────────────


def test_accept_seven_days_with_valid_recycle(checker: object, tmp_path: Path, capsys: object) -> None:
    root = tmp_path / "evidence"
    write_window(root)
    write_recycle(root)
    assert run_main(checker, root, "--days", "7") == 0
    report = read_report(capsys)
    assert report["schema"] == REPORT_SCHEMA
    assert report["window"] == {"start_day": START_DAY, "end_day": END_DAY, "days": 7}
    assert report["days_present"] == 7
    assert report["verdict"] == "accept"
    assert [day["status"] for day in report["per_day"]] == ["pass"] * 7
    assert all(day["reasons"] == [] for day in report["per_day"])
    assert all(
        day["categories"] == dict.fromkeys(CATEGORIES, "pass") for day in report["per_day"]
    )
    assert report["archive_continuity"]["duplicate_sha"] is False
    assert len(report["archive_continuity"]["days"]) == 7
    assert {entry["day"] for entry in report["archive_continuity"]["days"]} == set(day_span())
    assert len({entry["sha256"] for entry in report["archive_continuity"]["days"]}) == 7
    assert report["recycle"]["status"] == "proven"
    assert report["recycle"]["present"] is True
    assert report["blockers"] == []
    assert report["warnings"] == []
    assert report["attended_decisions"] != []


def test_ready_except_recycle_when_recycle_absent(checker: object, tmp_path: Path, capsys: object) -> None:
    root = tmp_path / "evidence"
    write_window(root)
    assert run_main(checker, root) == 0
    report = read_report(capsys)
    assert report["verdict"] == "ready_except_recycle"
    assert report["recycle"] == {"present": False, "status": "unproven", "reasons": []}
    assert report["blockers"] == []
    assert report["warnings"] == ["recycle proof absent: recycle persistence remains an attended unproven gate"]


# ── extend: missing days and warnings ─────────────────────────────────────


def test_missing_day_extends(checker: object, tmp_path: Path, capsys: object) -> None:
    root = tmp_path / "evidence"
    write_window(root)
    shutil.rmtree(root / "2026-09-03")
    assert run_main(checker, root) == 2
    report = read_report(capsys)
    assert report["verdict"] == "extend"
    assert report["days_present"] == 6
    assert report["blockers"] == []
    assert "day 2026-09-03 directory missing" in report["warnings"]
    by_day = {day["day"]: day["status"] for day in report["per_day"]}
    assert by_day["2026-09-03"] == "missing"
    assert len(report["archive_continuity"]["days"]) == 6


def test_absent_evidence_root_extends(checker: object, tmp_path: Path, capsys: object) -> None:
    root = tmp_path / "never-created"
    assert run_main(checker, root) == 2
    report = read_report(capsys)
    assert report["verdict"] == "extend"
    assert report["days_present"] == 0
    assert sum("directory missing" in warning for warning in report["warnings"]) == 7


def test_archive_warning_extends(checker: object, tmp_path: Path, capsys: object) -> None:
    root = tmp_path / "evidence"
    days = day_span()
    for day in days[:-1]:
        write_day(root, day)
    statuses = dict.fromkeys(CATEGORIES, "pass")
    statuses["archive"] = "warning"
    write_day(root, days[-1], statuses=statuses)
    assert run_main(checker, root) == 2
    report = read_report(capsys)
    assert report["verdict"] == "extend"
    assert report["blockers"] == []
    assert f"day {days[-1]}: archive category is warning" in report["warnings"]
    assert report["per_day"][-1]["status"] == "warning"


# ── investigate: fails, mismatch, layout, criteria ────────────────────────


def test_category_fail_investigates(checker: object, tmp_path: Path, capsys: object) -> None:
    root = tmp_path / "evidence"
    days = day_span()
    for day in days[:-1]:
        write_day(root, day)
    statuses = dict.fromkeys(CATEGORIES, "pass")
    statuses["authority"] = "fail"
    write_day(root, days[-1], statuses=statuses)
    assert run_main(checker, root) == 2
    report = read_report(capsys)
    assert report["verdict"] == "investigate"
    assert f"day {days[-1]}: authority category is fail" in report["blockers"]


@pytest.mark.parametrize(
    ("kind", "mutate"),
    [
        ("categories", lambda p: p["summary.json"]["details"]["categories"].update(authority="warning")),
        ("overall", lambda p: p["summary.json"]["details"].update(overall="warning")),
        ("notify", lambda p: p["summary.json"]["details"].update(notify_grok_bot=True)),
    ],
)
def test_summary_mismatch_investigates(
    checker: object, tmp_path: Path, capsys: object, kind: str, mutate: object
) -> None:
    root = tmp_path / "evidence"
    payloads = day_payloads(day_span()[0])
    mutate(payloads)
    write_day(root, day_span()[0], payloads)
    for day in day_span()[1:]:
        write_day(root, day)
    assert run_main(checker, root) == 2
    report = read_report(capsys)
    assert report["verdict"] == "investigate", kind
    assert report["blockers"] != []


@pytest.mark.parametrize(
    ("kind", "setup"),
    [
        ("day-dir-mode", lambda r, d: os.chmod(r / d, 0o755)),
        ("file-mode", lambda r, d: os.chmod(r / d / "freshness.json", 0o644)),
        ("day-symlink", lambda r, d: (shutil.rmtree(r / d), os.symlink(r / d, r / d))),
        ("file-symlink", lambda r, d: ((r / d / "tasker.json").unlink(), os.symlink(r / d / "archive.json", r / d / "tasker.json"))),
        ("day-extra-entry", lambda r, d: (r / d / "notes.txt").write_text("x")),
        ("missing-file", lambda r, d: (r / d / "jobs.json").unlink()),
    ],
)
def test_security_layout_investigates(
    checker: object, tmp_path: Path, capsys: object, kind: str, setup: object
) -> None:
    root = tmp_path / "evidence"
    write_window(root)
    setup(root, day_span()[2])
    assert run_main(checker, root) == 2
    report = read_report(capsys)
    assert report["verdict"] == "investigate", kind
    assert report["blockers"] != []
    assert report["per_day"][2]["status"] == "fail"


def test_root_extra_entry_investigates(checker: object, tmp_path: Path, capsys: object) -> None:
    root = tmp_path / "evidence"
    write_window(root)
    (root / "notes.txt").write_text("x", encoding="utf-8")
    assert run_main(checker, root) == 2
    report = read_report(capsys)
    assert report["verdict"] == "investigate"
    assert all(day["status"] == "pass" for day in report["per_day"])
    assert "unexpected entry in evidence root" in report["blockers"][0]


def test_oversize_evidence_file_investigates(checker: object, tmp_path: Path, capsys: object) -> None:
    root = tmp_path / "evidence"
    write_window(root)
    path = root / day_span()[2] / "resources.json"
    blob = path.read_text(encoding="utf-8")
    path.write_text(blob + " " * (MAX_EVIDENCE_BYTES + 1 - len(blob)), encoding="utf-8")
    assert run_main(checker, root) == 2
    report = read_report(capsys)
    assert report["verdict"] == "investigate"
    assert f"day {day_span()[2]}: evidence file exceeds the {MAX_EVIDENCE_BYTES} byte bound" in report["blockers"]


def test_duplicate_archive_sha_investigates(checker: object, tmp_path: Path, capsys: object) -> None:
    root = tmp_path / "evidence"
    days = day_span()
    for day in days:
        write_day(root, day)
    sha_of_first = hashlib.sha256(days[0].encode("ascii")).hexdigest()
    write_day(root, days[1], day_payloads(days[1], sha=sha_of_first))
    assert run_main(checker, root) == 2
    report = read_report(capsys)
    assert report["verdict"] == "investigate"
    assert report["archive_continuity"]["duplicate_sha"] is True
    assert "duplicate archive sha256 across the window" in report["blockers"]


@pytest.mark.parametrize(
    ("kind", "mutate"),
    [
        ("utc-day", lambda d: d["archive.json"]["details"].update(utc_day="2026-09-04")),
        ("latest-success-day", lambda d: d["archive.json"]["details"].update(latest_success="2026-09-04T03:00:00Z")),
        ("sha-format", lambda d: d["archive.json"]["details"].update(sha256="not-hex")),
    ],
)
def test_wrong_archive_day_investigates(
    checker: object, tmp_path: Path, capsys: object, kind: str, mutate: object
) -> None:
    root = tmp_path / "evidence"
    days = day_span()
    payloads = day_payloads(days[3])
    mutate(payloads)
    write_day(root, days[3], payloads)
    for day in days[:3] + days[4:]:
        write_day(root, day)
    assert run_main(checker, root) == 2
    report = read_report(capsys)
    assert report["verdict"] == "investigate", kind
    assert report["per_day"][3]["status"] == "fail"


@pytest.mark.parametrize(
    ("kind", "mutate"),
    [
        ("scheduler-count", lambda d: d["tasker.json"]["details"]["tasker_controller"].update(scheduler_instances=2)),
        ("tasker-state", lambda d: d["tasker.json"]["details"]["tasker_controller"].update(state="inactive")),
        ("static-identity", lambda d: d["tasker.json"]["details"]["static_controller"].update(identity_exact=False)),
        ("api-status", lambda d: d["jobs.json"]["details"]["api"].update(status="fail")),
        ("api-http", lambda d: d["jobs.json"]["details"]["api"].update(http_status=500)),
        ("static-http", lambda d: d["jobs.json"]["details"]["static_root"].update(http_status=503)),
        ("freshness-entry", lambda d: d["freshness.json"]["details"]["files"][0].update(status="fail")),
        ("disk-free", lambda d: d["resources.json"]["details"]["disk"].update(free_bytes=10 * GIB)),
        ("disk-status", lambda d: d["resources.json"]["details"]["disk"].update(status="warning")),
        ("authority-present", lambda d: d["authority.json"]["details"].update(present=False)),
        ("authority-host", lambda d: d["authority.json"]["details"].update(host_label="other-host")),
        ("authority-flags", lambda d: d["authority.json"]["details"].update(tasker_active=True)),
    ],
)
def test_criterion_violation_investigates(
    checker: object, tmp_path: Path, capsys: object, kind: str, mutate: object
) -> None:
    root = tmp_path / "evidence"
    days = day_span()
    payloads = day_payloads(days[4])
    mutate(payloads)
    write_day(root, days[4], payloads)
    for day in days[:4] + days[5:]:
        write_day(root, day)
    assert run_main(checker, root) == 2
    report = read_report(capsys)
    assert report["verdict"] == "investigate", kind
    assert report["per_day"][4]["status"] == "fail"


@pytest.mark.parametrize(
    ("kind", "mutate"),
    [
        ("schema", lambda d: d["tasker.json"].update(schema="portfolio-lab-daily-evidence/v2")),
        ("category-name", lambda d: d["jobs.json"].update(category="authority")),
        ("status", lambda d: d["freshness.json"].update(status="bogus")),
        ("collected-at-day", lambda d: d["authority.json"].update(collected_at="2026-09-02T04:00:00+00:00")),
        ("collected-at-naive", lambda d: d["tasker.json"].update(collected_at="2026-09-05T04:00:00")),
    ],
)
def test_malformed_envelope_investigates(
    checker: object, tmp_path: Path, capsys: object, kind: str, mutate: object
) -> None:
    root = tmp_path / "evidence"
    days = day_span()
    payloads = day_payloads(days[5])
    mutate(payloads)
    write_day(root, days[5], payloads)
    for day in days[:5]:
        write_day(root, day)
    assert run_main(checker, root) == 2
    report = read_report(capsys)
    assert report["verdict"] == "investigate", kind
    assert report["per_day"][5]["status"] == "fail"


@pytest.mark.parametrize(
    "kind",
    ["garbage-json", "non-utf8"],
)
def test_unparseable_evidence_investigates(
    checker: object, tmp_path: Path, capsys: object, kind: str
) -> None:
    root = tmp_path / "evidence"
    write_window(root)
    path = root / day_span()[1] / "tasker.json"
    path.write_bytes(b"not json at all" if kind == "garbage-json" else b"\xff\xfe\x00")
    assert run_main(checker, root) == 2
    report = read_report(capsys)
    assert report["verdict"] == "investigate", kind
    assert report["per_day"][1]["status"] == "fail"


# ── recycle proof: required, absent, invalid ──────────────────────────────


def test_recycle_required_absent_extends(checker: object, tmp_path: Path, capsys: object) -> None:
    root = tmp_path / "evidence"
    write_window(root)
    assert run_main(checker, root, "--require-recycle-proof") == 2
    report = read_report(capsys)
    assert report["verdict"] == "extend"
    assert report["blockers"] == []
    assert (
        "attended recycle proof required (--require-recycle-proof) but unproven; acceptance blocked"
        in report["warnings"]
    )


@pytest.mark.parametrize(
    ("kind", "overrides", "mode"),
    [
        ("schema", {"schema": "portfolio-lab-recycle-proof/v2"}, 0o600),
        ("category", {"category": "other"}, 0o600),
        ("status", {"status": "fail"}, 0o600),
        ("outside-window", {"performed_at": "2026-08-29T05:30:00+00:00"}, 0o600),
        ("pre-scheduler", {"pre_scheduler": 0}, 0o600),
        ("post-scheduler", {"post_scheduler": 2}, 0o600),
        ("component", {"components": "fail"}, 0o600),
        ("mode", {}, 0o644),
    ],
)
def test_invalid_recycle_investigates(
    checker: object, tmp_path: Path, capsys: object, kind: str, overrides: dict[str, object], mode: int
) -> None:
    root = tmp_path / "evidence"
    write_window(root)
    write_recycle(root, recycle_payload(**overrides), mode=mode)
    assert run_main(checker, root) == 2
    report = read_report(capsys)
    assert report["verdict"] == "investigate", kind
    assert report["recycle"]["status"] == "invalid"
    assert report["blockers"] != []
    assert "recycle:" in report["blockers"][0]


@pytest.mark.parametrize("kind", ["garbage-json", "symlink"])
def test_malformed_recycle_file_investigates(
    checker: object, tmp_path: Path, capsys: object, kind: str
) -> None:
    root = tmp_path / "evidence"
    write_window(root)
    path = root / "recycle.json"
    if kind == "garbage-json":
        path.write_text("garbage", encoding="utf-8")
        os.chmod(path, 0o600)
    else:
        os.symlink(root / "archive.json", path)
    assert run_main(checker, root) == 2
    report = read_report(capsys)
    assert report["verdict"] == "investigate", kind
    assert report["recycle"]["status"] == "invalid"


def test_recycle_boundary_within_window_still_accepts(
    checker: object, tmp_path: Path, capsys: object
) -> None:
    root = tmp_path / "evidence"
    write_window(root)
    write_recycle(root, recycle_payload(performed_at="2026-08-30T00:00:00+00:00"))
    assert run_main(checker, root) == 0
    assert read_report(capsys)["verdict"] == "accept"


# ── CLI validation, output safety, permissions, idempotence ───────────────


@pytest.mark.parametrize(
    "kind",
    [
        "no-root", "relative-root", "bad-end-day", "bad-calendar",
        "days-6", "days-abc", "relative-output", "output-inside-root",
    ],
)
def test_invalid_cli_exit_1(checker: object, tmp_path: Path, capsys: object, kind: str) -> None:
    root = tmp_path / "evidence"
    write_window(root)
    if kind == "no-root":
        args = ["--end-day", END_DAY]
    elif kind == "relative-root":
        args = ["--evidence-root", "relative/root"]
    elif kind == "bad-end-day":
        args = ["--evidence-root", str(root), "--end-day", "2026-9-5"]
    elif kind == "bad-calendar":
        args = ["--evidence-root", str(root), "--end-day", "2026-13-01"]
    elif kind == "days-6":
        args = ["--evidence-root", str(root), "--days", "6"]
    elif kind == "days-abc":
        args = ["--evidence-root", str(root), "--days", "abc"]
    elif kind == "relative-output":
        args = ["--evidence-root", str(root), "--output-json", "out/report.json"]
    else:
        args = ["--evidence-root", str(root), "--output-json", str(root / "report.json")]
    with pytest.raises(SystemExit) as exc:
        checker.main(args)
    assert exc.value.code == 1
    assert capsys.readouterr().out == ""


def test_output_json_mode_and_idempotence(checker: object, tmp_path: Path, capsys: object) -> None:
    root = tmp_path / "evidence"
    out_dir = tmp_path / "reports"
    out_dir.mkdir()
    report_path = out_dir / "acceptance.json"
    write_window(root)
    write_recycle(root)
    assert run_main(checker, root, "--output-json", str(report_path)) == 0
    first_stdout = capsys.readouterr().out
    assert stat.S_IMODE(report_path.stat().st_mode) == 0o600
    first_bytes = report_path.read_bytes()
    assert first_bytes.decode("utf-8") == first_stdout.strip()
    assert run_main(checker, root, "--output-json", str(report_path)) == 0
    assert report_path.read_bytes() == first_bytes
    assert json.loads(first_bytes)["verdict"] == "accept"