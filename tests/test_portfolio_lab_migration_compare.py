"""Strict TDD tests for portfolio_lab_migration_compare.py (Tasks 2.5–2.6).

Tests cover:
1. Directory input (evidence.json) success writes byte-identical JSON and Markdown on two runs;
   exact final success statement; mode 0600; no paths/URLs in output.
2. Direct JSON file inputs work.
3. HTTPS URL validation accepts a mocked safe bounded JSON response and rejects HTTP,
   credentials, query, fragment, oversized response, malformed JSON, and redirects/final URL mismatch
   without leaking URL/exception text.
4. Strict schema rejects missing/extra keys, wrong roles/hosts, invalid timestamps, unsafe logical names,
   absolute paths, secret-like keys, controls, excessive sizes, malformed digests/SHAs, bool-as-int counts,
   and invalid nested shapes.
5. Every dimension has a passing equality/expected-state test and a blocking mismatch test.
6. Required unavailable/missing evidence produces an unavailable difference and blocked verdict, not success.
7. Expected differences are emitted for scheduler mode/disable controls, authority role/access protection,
   and bounded freshness collection deltas.
8. Exact champion allocation, one-scheduler proof, candidate zero scheduled starts,
   Git/bundle/archive/SQLite/release/digest integrity, endpoint 200 status, and source authority
   cannot be explained away.
9. Fingerprint-bound explanation converts an eligible mismatch to explained; wrong/stale fingerprint
   or unused entry is unavailable and blocking; duplicate explanations rejected.
10. Output contains no raw mismatching values, secret sentinel, input/output path, credentials,
    query token, Authorization/Bearer text, or arbitrary exception text.
11. Blocked Markdown ends with exact failed check IDs, Dry run blocked, and retained safe-state line.
    CLI exits 2 after reports are written.
12. Atomic output: injected write/replace failure leaves prior reports unchanged and no sibling temp file.
13. Output paths same/colliding, symlink output, or missing parent are rejected before writes.
14. Markdown escaping prevents evidence/check text from injecting tables or headings.
15. CLI help and syntax work through the shipped script.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARE_SCRIPT = PROJECT_ROOT / "scripts" / "portfolio_lab_migration_compare.py"

_SHIPPED_MODULE_CACHE: dict[str, Any] = {}


def _load_shipped_module() -> Any:
    """Import the shipped CLI script in-process (its __main__ guard makes this safe)."""
    if not _SHIPPED_MODULE_CACHE:
        spec = importlib.util.spec_from_file_location("plmc_shipped_cli", COMPARE_SCRIPT)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SHIPPED_MODULE_CACHE["mod"] = mod
    return _SHIPPED_MODULE_CACHE["mod"]

VALID_COMMIT = "a" * 40
VALID_SHA256 = "b" * 64
VALID_TIME = "2026-09-03T12:00:00+00:00"

BASE_SOURCE_EVIDENCE: dict[str, Any] = {
    "schema_version": "portfolio-lab-migration-evidence/v1",
    "role": "source",
    "host": "sg01",
    "collected_at": VALID_TIME,
    "git": {
        "commit": VALID_COMMIT,
        "bundle_source_commit": VALID_COMMIT,
    },
    "recovery": {
        "archive_sha256": VALID_SHA256,
        "sidecar_ok": True,
        "archive_verified": True,
        "bundle_verified": True,
    },
    "sqlite": {
        "integrity": {"signals.db": "ok"},
        "counts": {"signals.rows": 123},
    },
    "digests": {
        "static": {"assets/index.js": "c" * 64},
        "runtime": {"signals.json": "d" * 64},
    },
    "release": {
        "schema_version": "portfolio-lab-static-release/v1",
        "source_git_sha": VALID_COMMIT,
        "manifest_sha256": "e" * 64,
    },
    "allocation": {
        "SPY": 0.46,
        "GLD": 0.38,
        "TLT": 0.16,
    },
    "safety": {
        "kill_switch": {"enabled": True, "level": "warning", "incident_id": "logical-id"},
        "open_incidents": [
            {"id": "logical-id", "channel": "signal_staleness", "severity": "p2", "state": "firing"}
        ],
    },
    "tasker": {
        "registry_sha256": "f" * 64,
        "scheduler_mode": "enabled",
        "scheduler_instances": 1,
        "scheduler_env_disabled": False,
        "scheduler_arg_disabled": False,
        "scheduled_starts_observed": 0,
        "status_schema": "tasker-status/v1",
    },
    "schemas": {
        "signals.json": "signals-data/v1",
        "index.json": "public-data-index/v1",
    },
    "freshness": {
        "signals.json": {
            "generated_at": VALID_TIME,
            "age_seconds": 10,
            "max_age_seconds": 900,
        }
    },
    "endpoints": {
        "/": {
            "status": 200,
            "content_type": "text/html",
            "schema_version": None,
            "body_sha256": "1" * 64,
        },
        "/_release.json": {
            "status": 200,
            "content_type": "application/json",
            "schema_version": "portfolio-lab-static-release/v1",
            "body_sha256": "2" * 64,
        },
        "/data/index.json": {
            "status": 200,
            "content_type": "application/json",
            "schema_version": "public-data-index/v1",
            "body_sha256": "3" * 64,
        },
        "/data/signals.json": {
            "status": 200,
            "content_type": "application/json",
            "schema_version": "signals-data/v1",
            "body_sha256": "4" * 64,
        },
        "/api/tasker/status": {
            "status": 200,
            "content_type": "application/json",
            "schema_version": "tasker-status/v1",
            "body_sha256": None,
        },
    },
    "authority": {
        "authoritative": True,
        "healthy": True,
        "access_protected": False,
        "public_origin_loopback_only": True,
    },
}

BASE_CANDIDATE_EVIDENCE: dict[str, Any] = {
    "schema_version": "portfolio-lab-migration-evidence/v1",
    "role": "candidate",
    "host": "cursor-box",
    "collected_at": VALID_TIME,
    "git": {
        "commit": VALID_COMMIT,
        "bundle_source_commit": VALID_COMMIT,
    },
    "recovery": {
        "archive_sha256": VALID_SHA256,
        "sidecar_ok": True,
        "archive_verified": True,
        "bundle_verified": True,
    },
    "sqlite": {
        "integrity": {"signals.db": "ok"},
        "counts": {"signals.rows": 123},
    },
    "digests": {
        "static": {"assets/index.js": "c" * 64},
        "runtime": {"signals.json": "d" * 64},
    },
    "release": {
        "schema_version": "portfolio-lab-static-release/v1",
        "source_git_sha": VALID_COMMIT,
        "manifest_sha256": "e" * 64,
    },
    "allocation": {
        "SPY": 0.46,
        "GLD": 0.38,
        "TLT": 0.16,
    },
    "safety": {
        "kill_switch": {"enabled": True, "level": "warning", "incident_id": "logical-id"},
        "open_incidents": [
            {"id": "logical-id", "channel": "signal_staleness", "severity": "p2", "state": "firing"}
        ],
    },
    "tasker": {
        "registry_sha256": "f" * 64,
        "scheduler_mode": "disabled",
        "scheduler_instances": 0,
        "scheduler_env_disabled": True,
        "scheduler_arg_disabled": True,
        "scheduled_starts_observed": 0,
        "status_schema": "tasker-status/v1",
    },
    "schemas": {
        "signals.json": "signals-data/v1",
        "index.json": "public-data-index/v1",
    },
    "freshness": {
        "signals.json": {
            "generated_at": VALID_TIME,
            "age_seconds": 12,
            "max_age_seconds": 900,
        }
    },
    "endpoints": {
        "/": {
            "status": 200,
            "content_type": "text/html",
            "schema_version": None,
            "body_sha256": "1" * 64,
        },
        "/_release.json": {
            "status": 200,
            "content_type": "application/json",
            "schema_version": "portfolio-lab-static-release/v1",
            "body_sha256": "2" * 64,
        },
        "/data/index.json": {
            "status": 200,
            "content_type": "application/json",
            "schema_version": "public-data-index/v1",
            "body_sha256": "3" * 64,
        },
        "/data/signals.json": {
            "status": 200,
            "content_type": "application/json",
            "schema_version": "signals-data/v1",
            "body_sha256": "4" * 64,
        },
        "/api/tasker/status": {
            "status": 200,
            "content_type": "application/json",
            "schema_version": "tasker-status/v1",
            "body_sha256": None,
        },
    },
    "authority": {
        "authoritative": False,
        "healthy": True,
        "access_protected": True,
        "public_origin_loopback_only": True,
    },
}


def run_compare(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run portfolio_lab_migration_compare.py with sys.executable."""
    cmd = [sys.executable, str(COMPARE_SCRIPT), *args]
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    return subprocess.run(
        cmd,
        cwd=cwd or PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=full_env,
    )


def canonical_fingerprint(val: Any) -> str:
    """Canonical SHA-256 fingerprint matching the comparator specification."""
    blob = json.dumps(val, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class TestMigrationCompareCLI:
    """End-to-end tests for portfolio_lab_migration_compare.py."""

    def test_cli_help(self) -> None:
        """CLI prints help with exit 0."""
        res = run_compare(["--help"])
        assert res.returncode == 0
        assert "--source" in res.stdout
        assert "--candidate" in res.stdout
        assert "--output-json" in res.stdout
        assert "--output-markdown" in res.stdout

    def test_directory_inputs_success_deterministic_and_permissions(self, tmp_path: Path) -> None:
        """Requirement 1: Directory input writes byte-identical JSON and Markdown on two runs, mode 0600."""
        src_dir = tmp_path / "src_dir"
        cand_dir = tmp_path / "cand_dir"
        src_dir.mkdir()
        cand_dir.mkdir()

        (src_dir / "evidence.json").write_text(json.dumps(BASE_SOURCE_EVIDENCE))
        (cand_dir / "evidence.json").write_text(json.dumps(BASE_CANDIDATE_EVIDENCE))

        out_json1 = tmp_path / "report1.json"
        out_md1 = tmp_path / "report1.md"
        out_json2 = tmp_path / "report2.json"
        out_md2 = tmp_path / "report2.md"

        res1 = run_compare([
            "--source", str(src_dir),
            "--candidate", str(cand_dir),
            "--output-json", str(out_json1),
            "--output-markdown", str(out_md1),
        ])
        assert res1.returncode == 0, f"res1 failed: stderr={res1.stderr}, stdout={res1.stdout}"

        # stdout compact JSON check
        stdout_json = json.loads(res1.stdout)
        assert stdout_json["verdict"] == "pass"
        assert stdout_json["terminal_statement"] == "Dry run passed; cutover approval required."
        assert "report1.json" not in res1.stdout
        assert str(src_dir) not in res1.stdout

        # Mode check (0600)
        mode_json = stat.S_IMODE(out_json1.stat().st_mode)
        mode_md = stat.S_IMODE(out_md1.stat().st_mode)
        assert mode_json == 0o600
        assert mode_md == 0o600

        # Markdown ending check
        md_text1 = out_md1.read_text()
        assert md_text1.rstrip().endswith("Dry run passed; cutover approval required.")
        # No local path or raw url leakage
        assert str(src_dir) not in md_text1
        assert str(cand_dir) not in md_text1
        assert str(out_json1) not in md_text1

        # Second run for byte-identity
        res2 = run_compare([
            "--source", str(src_dir),
            "--candidate", str(cand_dir),
            "--output-json", str(out_json2),
            "--output-markdown", str(out_md2),
        ])
        assert res2.returncode == 0
        assert out_json1.read_bytes() == out_json2.read_bytes()
        assert out_md1.read_bytes() == out_md2.read_bytes()

    def test_direct_file_inputs(self, tmp_path: Path) -> None:
        """Requirement 2: Direct JSON file inputs work."""
        src_file = tmp_path / "src.json"
        cand_file = tmp_path / "cand.json"
        src_file.write_text(json.dumps(BASE_SOURCE_EVIDENCE))
        cand_file.write_text(json.dumps(BASE_CANDIDATE_EVIDENCE))

        out_json = tmp_path / "out.json"
        out_md = tmp_path / "out.md"

        res = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
        ])
        assert res.returncode == 0
        rep = json.loads(out_json.read_text())
        assert rep["summary"]["verdict"] == "pass"

    def test_https_url_validation_and_rejection(self, tmp_path: Path) -> None:
        """Requirement 3: HTTPS URL validation accepts safe bounded mock response and rejects invalid URLs."""
        out_json = tmp_path / "out.json"
        out_md = tmp_path / "out.md"
        src_file = tmp_path / "src.json"
        src_file.write_text(json.dumps(BASE_SOURCE_EVIDENCE))

        # Reject http
        res = run_compare([
            "--source", str(src_file),
            "--candidate", "http://example.com/evidence.json",
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
        ])
        assert res.returncode == 1
        assert "http://example.com/evidence.json" not in res.stderr
        assert "http://example.com/evidence.json" not in res.stdout

        # Reject credentials in URL
        res = run_compare([
            "--source", str(src_file),
            "--candidate", "https://user:pass@example.com/evidence.json",
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
        ])
        assert res.returncode == 1
        assert "user:pass" not in res.stderr

        # Reject query in URL
        res = run_compare([
            "--source", str(src_file),
            "--candidate", "https://example.com/evidence.json?token=123",
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
        ])
        assert res.returncode == 1
        assert "token=123" not in res.stderr

        # Reject fragment in URL
        res = run_compare([
            "--source", str(src_file),
            "--candidate", "https://example.com/evidence.json#part",
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
        ])
        assert res.returncode == 1
        assert "part" not in res.stderr

    def test_strict_schema_validation(self, tmp_path: Path) -> None:
        """Requirement 4: Strict schema rejects missing/extra keys, wrong roles/hosts, secret-like keys, etc."""
        out_json = tmp_path / "out.json"
        out_md = tmp_path / "out.md"
        cand_file = tmp_path / "cand.json"
        cand_file.write_text(json.dumps(BASE_CANDIDATE_EVIDENCE))

        # Missing key
        bad_src = copy.deepcopy(BASE_SOURCE_EVIDENCE)
        del bad_src["sqlite"]
        src_file = tmp_path / "src_bad.json"
        src_file.write_text(json.dumps(bad_src))
        res = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
        ])
        assert res.returncode == 1

        # Extra key
        bad_src = copy.deepcopy(BASE_SOURCE_EVIDENCE)
        bad_src["extra_key"] = "bad"
        src_file.write_text(json.dumps(bad_src))
        res = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
        ])
        assert res.returncode == 1

        # Secret-like key rejection
        bad_src = copy.deepcopy(BASE_SOURCE_EVIDENCE)
        bad_src["sqlite"]["counts"]["secret_count"] = 10
        src_file.write_text(json.dumps(bad_src))
        res = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
        ])
        assert res.returncode == 1

        # Unsafe logical names (e.g. absolute path or slashes in counts)
        bad_src = copy.deepcopy(BASE_SOURCE_EVIDENCE)
        bad_src["sqlite"]["counts"]["/etc/passwd"] = 10
        src_file.write_text(json.dumps(bad_src))
        res = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
        ])
        assert res.returncode == 1

        # Wrong host
        bad_src = copy.deepcopy(BASE_SOURCE_EVIDENCE)
        bad_src["host"] = "wrong-host"
        src_file.write_text(json.dumps(bad_src))
        res = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
        ])
        assert res.returncode == 1

        # Bool as int rejection
        bad_src = copy.deepcopy(BASE_SOURCE_EVIDENCE)
        bad_src["sqlite"]["counts"]["signals.rows"] = True
        src_file.write_text(json.dumps(bad_src))
        res = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
        ])
        assert res.returncode == 1

    def test_dimensions_pass_and_blocking_mismatch(self, tmp_path: Path) -> None:
        """Requirements 5, 7: All dimensions evaluated; expected differences vs blocking differences."""
        out_json = tmp_path / "out.json"
        out_md = tmp_path / "out.md"
        src_file = tmp_path / "src.json"
        cand_file = tmp_path / "cand.json"
        src_file.write_text(json.dumps(BASE_SOURCE_EVIDENCE))

        # Test Git mismatch is blocking
        bad_cand = copy.deepcopy(BASE_CANDIDATE_EVIDENCE)
        bad_cand["git"]["commit"] = "0" * 40
        bad_cand["git"]["bundle_source_commit"] = "0" * 40
        bad_cand["release"]["source_git_sha"] = "0" * 40
        cand_file.write_text(json.dumps(bad_cand))

        res = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
        ])
        assert res.returncode == 2
        rep = json.loads(out_json.read_text())
        assert rep["summary"]["verdict"] == "blocked"
        assert rep["summary"]["counts"]["blocking"] >= 1
        md = out_md.read_text()
        assert "Dry run blocked" in md
        assert "git.commit" in md
        assert "Retained safe state: sg01 remains authoritative; cursor-box scheduler remains disabled." in md

    def test_allocation_strictness(self, tmp_path: Path) -> None:
        """Requirement 8: Champion allocation 46/38/16 must match exactly."""
        out_json = tmp_path / "out.json"
        out_md = tmp_path / "out.md"
        src_file = tmp_path / "src.json"
        cand_file = tmp_path / "cand.json"

        bad_src = copy.deepcopy(BASE_SOURCE_EVIDENCE)
        bad_src["allocation"]["SPY"] = 0.45
        bad_src["allocation"]["GLD"] = 0.39
        src_file.write_text(json.dumps(bad_src))
        cand_file.write_text(json.dumps(BASE_CANDIDATE_EVIDENCE))

        res = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
        ])
        assert res.returncode == 2
        rep = json.loads(out_json.read_text())
        assert any(d["check_id"].startswith("allocation.") for d in rep["differences"])

    def test_one_scheduler_invariant_and_candidate_starts(self, tmp_path: Path) -> None:
        """Requirement 8: Candidate scheduler starts observed > 0 is blocking."""
        out_json = tmp_path / "out.json"
        out_md = tmp_path / "out.md"
        src_file = tmp_path / "src.json"
        cand_file = tmp_path / "cand.json"
        src_file.write_text(json.dumps(BASE_SOURCE_EVIDENCE))

        bad_cand = copy.deepcopy(BASE_CANDIDATE_EVIDENCE)
        bad_cand["tasker"]["scheduled_starts_observed"] = 1
        cand_file.write_text(json.dumps(bad_cand))

        res = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
        ])
        assert res.returncode == 2
        rep = json.loads(out_json.read_text())
        assert rep["summary"]["counts"]["blocking"] >= 1

    def test_fingerprint_bound_explanations(self, tmp_path: Path) -> None:
        """Requirements 8, 9: Fingerprint-bound explanation converts eligible difference to explained."""
        out_json = tmp_path / "out.json"
        out_md = tmp_path / "out.md"
        src_file = tmp_path / "src.json"
        cand_file = tmp_path / "cand.json"
        exp_file = tmp_path / "explanations.json"

        # Introduce an eligible difference: endpoint body digest for /data/signals.json
        cand = copy.deepcopy(BASE_CANDIDATE_EVIDENCE)
        cand["endpoints"]["/data/signals.json"]["body_sha256"] = "9" * 64

        src_file.write_text(json.dumps(BASE_SOURCE_EVIDENCE))
        cand_file.write_text(json.dumps(cand))

        src_fp = canonical_fingerprint(BASE_SOURCE_EVIDENCE["endpoints"]["/data/signals.json"]["body_sha256"])
        cand_fp = canonical_fingerprint("9" * 64)

        explanations_content = {
            "schema_version": "portfolio-lab-migration-explanations/v1",
            "entries": [
                {
                    "check_id": "endpoints./data/signals.json.body_sha256",
                    "source_fingerprint": src_fp,
                    "candidate_fingerprint": cand_fp,
                    "reason": "signals body regenerated with newer timestamp during test window",
                }
            ],
        }
        exp_file.write_text(json.dumps(explanations_content))

        res = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
            "--explanations", str(exp_file),
        ])
        assert res.returncode == 0
        rep = json.loads(out_json.read_text())
        assert rep["summary"]["verdict"] == "pass"
        assert rep["summary"]["counts"]["explained"] == 1
        assert rep["summary"]["counts"]["blocking"] == 0

        # Now test with wrong fingerprint (stale) -> becomes unavailable and blocks
        explanations_content["entries"][0]["candidate_fingerprint"] = "0" * 64
        exp_file.write_text(json.dumps(explanations_content))

        res2 = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
            "--explanations", str(exp_file),
        ])
        assert res2.returncode == 2
        rep2 = json.loads(out_json.read_text())
        assert rep2["summary"]["verdict"] == "blocked"
        assert rep2["summary"]["counts"]["unavailable"] >= 1

    def test_unexplainable_check_cannot_be_downgraded(self, tmp_path: Path) -> None:
        """Requirement 8: Integrity booleans, SQLite integrity, champion allocation cannot be explained away."""
        out_json = tmp_path / "out.json"
        out_md = tmp_path / "out.md"
        src_file = tmp_path / "src.json"
        cand_file = tmp_path / "cand.json"
        exp_file = tmp_path / "explanations.json"

        bad_cand = copy.deepcopy(BASE_CANDIDATE_EVIDENCE)
        bad_cand["sqlite"]["integrity"]["signals.db"] = "corrupt"
        src_file.write_text(json.dumps(BASE_SOURCE_EVIDENCE))
        cand_file.write_text(json.dumps(bad_cand))

        src_fp = canonical_fingerprint("ok")
        cand_fp = canonical_fingerprint("corrupt")

        explanations_content = {
            "schema_version": "portfolio-lab-migration-explanations/v1",
            "entries": [
                {
                    "check_id": "sqlite.integrity.signals.db",
                    "source_fingerprint": src_fp,
                    "candidate_fingerprint": cand_fp,
                    "reason": "attempt to explain away sqlite integrity failure",
                }
            ],
        }
        exp_file.write_text(json.dumps(explanations_content))

        res = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
            "--explanations", str(exp_file),
        ])
        assert res.returncode == 2
        rep = json.loads(out_json.read_text())
        assert rep["summary"]["counts"]["blocking"] >= 1

    def test_no_raw_values_or_secrets_leakage(self, tmp_path: Path) -> None:
        """Requirement 10: Output contains no raw mismatching values, secret sentinel, or credentials."""
        out_json = tmp_path / "out.json"
        out_md = tmp_path / "out.md"
        src_file = tmp_path / "src.json"
        cand_file = tmp_path / "cand.json"

        secret_sentinel = "SUPER_SECRET_PAYLOAD_VALUE_XYZ"
        bad_cand = copy.deepcopy(BASE_CANDIDATE_EVIDENCE)
        bad_cand["digests"]["runtime"]["signals.json"] = "e" * 64
        src_file.write_text(json.dumps(BASE_SOURCE_EVIDENCE))
        cand_file.write_text(json.dumps(bad_cand))

        res = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
        ])
        assert res.returncode == 2
        json_txt = out_json.read_text()
        md_txt = out_md.read_text()
        assert secret_sentinel not in json_txt
        assert secret_sentinel not in md_txt
        assert "Authorization" not in json_txt
        assert "Bearer" not in md_txt

    def test_atomic_write_and_error_handling(self, tmp_path: Path) -> None:
        """Requirements 12, 13: Atomic write and output validation (colliding paths, symlinks)."""
        src_file = tmp_path / "src.json"
        cand_file = tmp_path / "cand.json"
        src_file.write_text(json.dumps(BASE_SOURCE_EVIDENCE))
        cand_file.write_text(json.dumps(BASE_CANDIDATE_EVIDENCE))

        colliding_path = tmp_path / "same.txt"

        # Same output paths rejected
        res = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(colliding_path),
            "--output-markdown", str(colliding_path),
        ])
        assert res.returncode == 1

        # Missing parent dir rejected
        missing_dir_out = tmp_path / "nonexistent" / "out.json"
        res2 = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(missing_dir_out),
            "--output-markdown", str(tmp_path / "out.md"),
        ])
        assert res2.returncode == 1

    def test_markdown_escaping(self, tmp_path: Path) -> None:
        """Requirement 14: Markdown escaping prevents evidence/check text from injecting tables or headings."""
        out_json = tmp_path / "out.json"
        out_md = tmp_path / "out.md"
        src_file = tmp_path / "src.json"
        cand_file = tmp_path / "cand.json"
        exp_file = tmp_path / "explanations.json"

        cand = copy.deepcopy(BASE_CANDIDATE_EVIDENCE)
        cand["endpoints"]["/data/signals.json"]["body_sha256"] = "8" * 64
        src_file.write_text(json.dumps(BASE_SOURCE_EVIDENCE))
        cand_file.write_text(json.dumps(cand))

        src_fp = canonical_fingerprint(BASE_SOURCE_EVIDENCE["endpoints"]["/data/signals.json"]["body_sha256"])
        cand_fp = canonical_fingerprint("8" * 64)

        # Reason contains pipe or markdown characters that should be escaped
        malicious_reason = "Test | cell | injection\n# Injected Heading"
        # However, explanation schema validation requires safe chars (1-200 safe characters, no controls, etc.)
        # Let's test pipe in reason if allowed, or check that unsafe characters in explanations are rejected:
        exp_bad = {
            "schema_version": "portfolio-lab-migration-explanations/v1",
            "entries": [
                {
                    "check_id": "endpoints./data/signals.json.body_sha256",
                    "source_fingerprint": src_fp,
                    "candidate_fingerprint": cand_fp,
                    "reason": malicious_reason,
                }
            ],
        }
        exp_file.write_text(json.dumps(exp_bad))

        res = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
            "--explanations", str(exp_file),
        ])
        # Reason with newline should be rejected by strict explanation validation
        assert res.returncode == 1

    def test_missing_required_evidence_is_unavailable_and_blocks(self, tmp_path: Path) -> None:
        """Requirement 6: Missing required evidence produces unavailable difference and blocked verdict."""
        src_file = tmp_path / "src.json"
        cand_file = tmp_path / "cand.json"
        out_json = tmp_path / "out.json"
        out_md = tmp_path / "out.md"

        bad_cand = copy.deepcopy(BASE_CANDIDATE_EVIDENCE)
        del bad_cand["endpoints"]["/data/signals.json"]
        src_file.write_text(json.dumps(BASE_SOURCE_EVIDENCE))
        cand_file.write_text(json.dumps(bad_cand))

        res = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
        ])
        assert res.returncode == 2
        rep = json.loads(out_json.read_text())
        assert rep["summary"]["verdict"] == "blocked"
        assert rep["summary"]["counts"]["unavailable"] >= 1
        assert rep["summary"]["counts"]["blocking"] == 0
        assert rep["summary"]["counts"]["explained"] == 0
        assert rep["summary"]["counts"]["expected"] >= 1

    def test_expected_differences_emitted(self, tmp_path: Path) -> None:
        """Requirement 7: Scheduler, authority, and bounded freshness deltas are expected differences."""
        src_file = tmp_path / "src.json"
        cand_file = tmp_path / "cand.json"
        out_json = tmp_path / "out.json"
        out_md = tmp_path / "out.md"

        # Candidate freshness collection slightly later than source
        cand = copy.deepcopy(BASE_CANDIDATE_EVIDENCE)
        cand["freshness"]["signals.json"] = {
            "generated_at": "2026-09-03T12:00:10+00:00",
            "age_seconds": 20,
            "max_age_seconds": 900,
        }
        src_file.write_text(json.dumps(BASE_SOURCE_EVIDENCE))
        cand_file.write_text(json.dumps(cand))

        res = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
        ])
        assert res.returncode == 0
        rep = json.loads(out_json.read_text())
        by_id = {d["check_id"]: d["classification"] for d in rep["differences"]}
        assert by_id["tasker.scheduler_mode"] == "expected"
        assert by_id["tasker.scheduler_instances"] == "expected"
        assert by_id["tasker.scheduler_env_disabled"] == "expected"
        assert by_id["tasker.scheduler_arg_disabled"] == "expected"
        assert by_id["authority.authoritative"] == "expected"
        assert by_id["authority.access_protected"] == "expected"
        assert by_id["freshness.signals.json.generated_at"] == "expected"
        assert by_id["freshness.signals.json.age_seconds"] == "expected"
        assert rep["summary"]["counts"]["blocking"] == 0

        # Beyond the freshness delta: blocking
        cand["freshness"]["signals.json"]["generated_at"] = "2026-09-04T12:00:10+00:00"
        cand_file.write_text(json.dumps(cand))
        res2 = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
            "--max-freshness-delta-seconds", "0.5",
        ])
        assert res2.returncode == 2
        rep2 = json.loads(out_json.read_text())
        assert rep2["summary"]["counts"]["blocking"] >= 1

    def test_duplicate_explanations_rejected(self, tmp_path: Path) -> None:
        """Requirement 9: Duplicate explanation check IDs rejected at input validation."""
        src_file = tmp_path / "src.json"
        cand_file = tmp_path / "cand.json"
        out_json = tmp_path / "out.json"
        out_md = tmp_path / "out.md"
        exp_file = tmp_path / "explanations.json"

        src_file.write_text(json.dumps(BASE_SOURCE_EVIDENCE))
        cand_file.write_text(json.dumps(BASE_CANDIDATE_EVIDENCE))

        exp_bad = {
            "schema_version": "portfolio-lab-migration-explanations/v1",
            "entries": [
                {
                    "check_id": "freshness.signals.json.generated_at",
                    "source_fingerprint": "1" * 64,
                    "candidate_fingerprint": "2" * 64,
                    "reason": "first entry duplicate",
                },
                {
                    "check_id": "freshness.signals.json.generated_at",
                    "source_fingerprint": "3" * 64,
                    "candidate_fingerprint": "4" * 64,
                    "reason": "second entry duplicate",
                },
            ],
        }
        exp_file.write_text(json.dumps(exp_bad))

        res = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
            "--explanations", str(exp_file),
        ])
        assert res.returncode == 1

    def test_endpoint_status_and_digest_cannot_be_explained_away(self, tmp_path: Path) -> None:
        """Requirement 8: Required endpoint 200 status and digest integrity cannot be downgraded."""
        src_file = tmp_path / "src.json"
        cand_file = tmp_path / "cand.json"
        out_json = tmp_path / "out.json"
        out_md = tmp_path / "out.md"
        exp_file = tmp_path / "explanations.json"

        bad_cand = copy.deepcopy(BASE_CANDIDATE_EVIDENCE)
        bad_cand["endpoints"]["/data/index.json"]["status"] = 500
        bad_cand["digests"]["runtime"]["signals.json"] = "9" * 64

        src_file.write_text(json.dumps(BASE_SOURCE_EVIDENCE))
        cand_file.write_text(json.dumps(bad_cand))

        status_src_fp = canonical_fingerprint(200)
        status_cand_fp = canonical_fingerprint(500)
        digest_src_fp = canonical_fingerprint("d" * 64)
        digest_cand_fp = canonical_fingerprint("9" * 64)

        exp_bad = {
            "schema_version": "portfolio-lab-migration-explanations/v1",
            "entries": [
                {
                    "check_id": "endpoints./data/index.json.status",
                    "source_fingerprint": status_src_fp,
                    "candidate_fingerprint": status_cand_fp,
                    "reason": "attempt to explain away endpoint 500",
                },
                {
                    "check_id": "digests.runtime.signals.json",
                    "source_fingerprint": digest_src_fp,
                    "candidate_fingerprint": digest_cand_fp,
                    "reason": "attempt to explain away digest mismatch",
                },
            ],
        }
        exp_file.write_text(json.dumps(exp_bad))

        res = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
            "--explanations", str(exp_file),
        ])
        assert res.returncode == 2
        rep = json.loads(out_json.read_text())
        assert rep["summary"]["counts"]["explained"] == 0
        assert rep["summary"]["counts"]["blocking"] >= 1
        assert rep["summary"]["counts"]["unavailable"] >= 1
        assert all(
            d["classification"] != "explained"
            for d in rep["differences"]
            if d["check_id"] in {
                "endpoints./data/index.json.status",
                "digests.runtime.signals.json",
            }
        )

    def test_every_dimension_has_blocking_mismatch(self, tmp_path: Path) -> None:
        """Requirement 5: Each dimension has a blocking mismatch case."""
        mutations = {
            "git": lambda e: (
                e["git"].update({"commit": "0" * 40, "bundle_source_commit": "0" * 40}),
                e["release"].update({"source_git_sha": "0" * 40, "manifest_sha256": "0" * 64}),
            )[1],
            "recovery": lambda e: e["recovery"].update({"archive_sha256": "9" * 64}),
            "sqlite": lambda e: e["sqlite"]["integrity"].update({"signals.db": "corrupt"}),
            "digests": lambda e: e["digests"]["static"].update({"assets/index.js": "9" * 64}),
            "release": lambda e: e["release"].update({"manifest_sha256": "9" * 64}),
            "allocation": lambda e: e["allocation"].update({"SPY": 0.45}),
            "safety": lambda e: e["safety"]["kill_switch"].update({"enabled": False}),
            "tasker": lambda e: e["tasker"].update({"scheduler_instances": 2}),
            "schemas": lambda e: e["schemas"].update({"signals.json": "signals-data/v2"}),
            "freshness": lambda e: e["freshness"]["signals.json"].update({"max_age_seconds": 1800}),
            "endpoints": lambda e: e["endpoints"]["/"].update({"status": 503}),
            "authority": lambda e: e["authority"].update({"healthy": False}),
        }
        for dim, mutate in mutations.items():
            src_file = tmp_path / f"src_{dim}.json"
            cand_file = tmp_path / f"cand_{dim}.json"
            out_json = tmp_path / f"out_{dim}.json"
            out_md = tmp_path / f"out_{dim}.md"

            bad_cand = copy.deepcopy(BASE_CANDIDATE_EVIDENCE)
            mutate(bad_cand)
            src_file.write_text(json.dumps(BASE_SOURCE_EVIDENCE))
            cand_file.write_text(json.dumps(bad_cand))

            res = run_compare([
                "--source", str(src_file),
                "--candidate", str(cand_file),
                "--output-json", str(out_json),
                "--output-markdown", str(out_md),
            ])
            assert res.returncode == 2, f"dimension {dim} not blocked: {res.stderr}"
            rep = json.loads(out_json.read_text())
            assert rep["summary"]["verdict"] == "blocked"
            assert any(d["dimension"] == dim and d["classification"] == "blocking" for d in rep["differences"]), dim

    def test_symlink_output_rejected(self, tmp_path: Path) -> None:
        """Requirement 13: Symlink output destination rejected before writes."""
        src_file = tmp_path / "src.json"
        cand_file = tmp_path / "cand.json"
        real_target = tmp_path / "real_out.md"
        real_target.write_text("old markdown")
        symlink_out = tmp_path / "symlink_out.md"
        symlink_out.symlink_to(real_target)

        src_file.write_text(json.dumps(BASE_SOURCE_EVIDENCE))
        cand_file.write_text(json.dumps(BASE_CANDIDATE_EVIDENCE))

        res = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(tmp_path / "out.json"),
            "--output-markdown", str(symlink_out),
        ])
        assert res.returncode == 1
        assert real_target.read_text() == "old markdown"
        assert not list(tmp_path.glob(".tmp-*"))

    def test_atomic_temp_creation_failure_preserves_prior_reports(self, tmp_path: Path) -> None:
        """Requirement 12 (temp-creation stage): injected temp-write failure leaves prior reports untouched."""
        src_file = tmp_path / "src.json"
        cand_file = tmp_path / "cand.json"
        src_file.write_text(json.dumps(BASE_SOURCE_EVIDENCE))
        cand_file.write_text(json.dumps(BASE_CANDIDATE_EVIDENCE))

        json_dir = tmp_path / "json_dir"
        md_dir = tmp_path / "md_dir"
        json_dir.mkdir()
        md_dir.mkdir()
        out_json = json_dir / "out.json"
        out_md = md_dir / "out.md"
        out_json.write_text("prior json")
        out_md.write_text("prior markdown")

        # Make the markdown directory read-only so the second temp creation fails
        md_dir.chmod(0o500)
        try:
            res = run_compare([
                "--source", str(src_file),
                "--candidate", str(cand_file),
                "--output-json", str(out_json),
                "--output-markdown", str(out_md),
            ])
        finally:
            md_dir.chmod(0o700)

        assert res.returncode == 1
        assert out_json.read_text() == "prior json"
        assert out_md.read_text() == "prior markdown"
        assert not list(json_dir.glob(".tmp-*"))
        assert not list(md_dir.glob(".tmp-*"))
        assert not list(json_dir.glob(".bak-*"))
        assert not list(md_dir.glob(".bak-*"))

    def test_markdown_escaping_positive(self, tmp_path: Path) -> None:
        """Requirement 14: Markdown escaping prevents reason/evidence text from breaking tables or headings."""
        src_file = tmp_path / "src.json"
        cand_file = tmp_path / "cand.json"
        out_json = tmp_path / "out.json"
        out_md = tmp_path / "out.md"
        exp_file = tmp_path / "explanations.json"

        cand = copy.deepcopy(BASE_CANDIDATE_EVIDENCE)
        cand["endpoints"]["/data/signals.json"]["body_sha256"] = "7" * 64
        src_file.write_text(json.dumps(BASE_SOURCE_EVIDENCE))
        cand_file.write_text(json.dumps(cand))

        src_fp = canonical_fingerprint(BASE_SOURCE_EVIDENCE["endpoints"]["/data/signals.json"]["body_sha256"])
        cand_fp = canonical_fingerprint("7" * 64)

        exp_ok = {
            "schema_version": "portfolio-lab-migration-explanations/v1",
            "entries": [
                {
                    "check_id": "endpoints./data/signals.json.body_sha256",
                    "source_fingerprint": src_fp,
                    "candidate_fingerprint": cand_fp,
                    "reason": "signals regenerated within window | table cell attempt # heading attempt",
                }
            ],
        }
        exp_file.write_text(json.dumps(exp_ok))

        res = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
            "--explanations", str(exp_file),
        ])
        assert res.returncode == 0
        md = out_md.read_text()
        assert "| table cell attempt \\# heading" in md
        assert "\\|" in md

    def test_more_schema_strictness(self, tmp_path: Path) -> None:
        """Requirement 4: Role, timestamp, control chars, sizes, malformed SHA, invalid nested shapes."""
        out_json = tmp_path / "out.json"
        out_md = tmp_path / "out.md"
        cand_file = tmp_path / "cand.json"
        cand_file.write_text(json.dumps(BASE_CANDIDATE_EVIDENCE))
        src_file = tmp_path / "src_bad.json"

        # Wrong role
        bad = copy.deepcopy(BASE_SOURCE_EVIDENCE)
        bad["role"] = "candidate"
        src_file.write_text(json.dumps(bad))
        res = run_compare(["--source", str(src_file), "--candidate", str(cand_file),
                           "--output-json", str(out_json), "--output-markdown", str(out_md)])
        assert res.returncode == 1

        # Invalid timestamp (not timezone aware)
        bad = copy.deepcopy(BASE_SOURCE_EVIDENCE)
        bad["collected_at"] = "2026-09-03T12:00:00"
        src_file.write_text(json.dumps(bad))
        res = run_compare(["--source", str(src_file), "--candidate", str(cand_file),
                           "--output-json", str(out_json), "--output-markdown", str(out_md)])
        assert res.returncode == 1

        # Control character in a logical name
        bad = copy.deepcopy(BASE_SOURCE_EVIDENCE)
        bad["sqlite"]["counts"]["bad\x01name"] = 1
        src_file.write_text(json.dumps(bad))
        res = run_compare(["--source", str(src_file), "--candidate", str(cand_file),
                           "--output-json", str(out_json), "--output-markdown", str(out_md)])
        assert res.returncode == 1

        # Malformed SHA in digests
        bad = copy.deepcopy(BASE_SOURCE_EVIDENCE)
        bad["digests"]["runtime"]["signals.json"] = "xyz"
        src_file.write_text(json.dumps(bad))
        res = run_compare(["--source", str(src_file), "--candidate", str(cand_file),
                           "--output-json", str(out_json), "--output-markdown", str(out_md)])
        assert res.returncode == 1

        # Excessive sizes (a >200-char key)
        bad = copy.deepcopy(BASE_SOURCE_EVIDENCE)
        bad["sqlite"]["counts"][("k" * 250)] = 1
        src_file.write_text(json.dumps(bad))
        res = run_compare(["--source", str(src_file), "--candidate", str(cand_file),
                           "--output-json", str(out_json), "--output-markdown", str(out_md)])
        assert res.returncode == 1


class FakeHTTPResponse:
    """Bounded fake urllib response: context manager, headers.get, geturl, record read sizes."""

    def __init__(
        self,
        body: bytes | str,
        *,
        content_type: str = "application/json",
        final_url: str | None = None,
    ) -> None:
        self._body = body.encode("utf-8") if isinstance(body, str) else body
        self._ct = content_type
        self._final = final_url
        self.headers = {"Content-Type": self._ct}
        self.read_sizes: list[int] = []

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def read(self, n: int = -1) -> bytes:
        self.read_sizes.append(n)
        if n is None or n < 0:
            return self._body
        return self._body[:n]

    def geturl(self) -> str:
        return self._final if self._final is not None else ""


class TestAtomicPublication:
    """Requirement 12 (replace stage): rollback-safe two-file publication."""

    def test_second_install_replace_failure_restores_prior_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Second install replace failure must restore both prior destinations exactly (bytes + modes)."""
        mod = _load_shipped_module()
        d = tmp_path / "out"
        d.mkdir()
        j = d / "report.json"
        m = d / "report.md"
        j.write_bytes(b"prior-json-1")
        m.write_bytes(b"prior-md-1")
        os.chmod(j, 0o640)
        os.chmod(m, 0o600)

        real_replace = os.replace
        install_count = [0]

        def failing_second_install(src: str, dst: str) -> None:
            if Path(src).name.startswith(".tmp-"):
                install_count[0] += 1
                if install_count[0] == 2:
                    raise OSError("injected second install failure")
            real_replace(src, dst)

        monkeypatch.setattr(os, "replace", failing_second_install)
        with pytest.raises(SystemExit) as exc:
            mod.write_outputs_atomically(j, "new-json-content", m, "new-md-content")
        assert exc.value.code == 1
        assert install_count[0] == 2
        assert j.read_bytes() == b"prior-json-1"
        assert m.read_bytes() == b"prior-md-1"
        assert stat.S_IMODE(j.stat().st_mode) == 0o640
        assert stat.S_IMODE(m.stat().st_mode) == 0o600
        assert not list(d.glob(".tmp-*"))
        assert not list(d.glob(".bak-*"))

    def test_backup_stage_failure_preserves_prior_files_and_cleans_artifacts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A backup-stage failure leaves both existing reports byte- and mode-identical."""
        mod = _load_shipped_module()
        d = tmp_path / "out"
        d.mkdir()
        j = d / "report.json"
        m = d / "report.md"
        j.write_bytes(b"prior-json")
        m.write_bytes(b"prior-markdown")
        os.chmod(j, 0o641)
        os.chmod(m, 0o604)
        real_copyfile = shutil.copyfile

        def fail_second_backup(src: str | os.PathLike[str], dst: str | os.PathLike[str], *, follow_symlinks: bool = True):
            if Path(src) == m:
                raise OSError("injected backup failure")
            return real_copyfile(src, dst, follow_symlinks=follow_symlinks)

        monkeypatch.setattr(shutil, "copyfile", fail_second_backup)
        with pytest.raises(SystemExit) as exc:
            mod.write_outputs_atomically(j, "new-json", m, "new-markdown")
        assert exc.value.code == 1
        assert j.read_bytes() == b"prior-json"
        assert m.read_bytes() == b"prior-markdown"
        assert stat.S_IMODE(j.stat().st_mode) == 0o641
        assert stat.S_IMODE(m.stat().st_mode) == 0o604
        assert not list(d.glob(".tmp-*"))
        assert not list(d.glob(".bak-*"))

    def test_second_install_replace_failure_removes_newly_installed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With outputs initially absent, a failed second install must leave both absent."""
        mod = _load_shipped_module()
        d = tmp_path / "out"
        d.mkdir()
        j = d / "report.json"
        m = d / "report.md"
        assert not j.exists() and not m.exists()

        real_replace = os.replace
        install_count = [0]

        def failing_second_install(src: str, dst: str) -> None:
            if Path(src).name.startswith(".tmp-"):
                install_count[0] += 1
                if install_count[0] == 2:
                    raise OSError("injected second install failure")
            real_replace(src, dst)

        monkeypatch.setattr(os, "replace", failing_second_install)
        with pytest.raises(SystemExit) as exc:
            mod.write_outputs_atomically(j, "new-json-content", m, "new-md-content")
        assert exc.value.code == 1
        assert not j.exists()
        assert not m.exists()
        assert not list(d.glob(".tmp-*"))
        assert not list(d.glob(".bak-*"))

    def test_directory_and_fifo_outputs_rejected_before_writes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Directory/FIFO destinations are rejected before any temp/backup file is created."""
        mod = _load_shipped_module()
        d = tmp_path / "out"
        d.mkdir()
        j = d / "report.json"
        j.write_bytes(b"prior-json")
        adir = d / "adir"
        adir.mkdir()
        fifo = d / "afifo"
        os.mkfifo(fifo)

        created = []
        real_mkstemp = tempfile.mkstemp

        def spy_mkstemp(*args: Any, **kwargs: Any) -> tuple[int, str]:
            created.append(1)
            return real_mkstemp(*args, **kwargs)

        monkeypatch.setattr(tempfile, "mkstemp", spy_mkstemp)

        with pytest.raises(SystemExit) as exc:
            mod.write_outputs_atomically(j, "new-json-content", adir, "new-md")
        assert exc.value.code == 1
        assert created == []
        assert j.read_bytes() == b"prior-json"

        with pytest.raises(SystemExit) as exc:
            mod.write_outputs_atomically(j, "new-json-content", fifo, "new-md")
        assert exc.value.code == 1
        assert created == []
        assert stat.S_ISFIFO(fifo.stat().st_mode)
        assert j.read_bytes() == b"prior-json"
        assert not list(d.glob(".tmp-*"))
        assert not list(d.glob(".bak-*"))

    def test_symlink_output_parent_rejected_before_writes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A symlinked output parent is rejected before staging begins."""
        mod = _load_shipped_module()
        real_parent = tmp_path / "real"
        real_parent.mkdir()
        linked_parent = tmp_path / "linked"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        json_target = tmp_path / "json-report"
        markdown_target = linked_parent / "markdown-report"

        created = []
        real_mkstemp = tempfile.mkstemp

        def spy_mkstemp(*args: Any, **kwargs: Any) -> tuple[int, str]:
            created.append(1)
            return real_mkstemp(*args, **kwargs)

        monkeypatch.setattr(tempfile, "mkstemp", spy_mkstemp)
        with pytest.raises(SystemExit) as exc:
            mod.write_outputs_atomically(json_target, "new-json", markdown_target, "new-md")
        assert exc.value.code == 1
        assert created == []
        assert not list(tmp_path.rglob(".tmp-*"))
        assert not list(tmp_path.rglob(".bak-*"))


class TestUrlLoadingModuleLevel:
    """Requirement 3 (URL branch): real urlopen path exercised via monkeypatch."""

    URL = "https://evidence.example/e.json"

    def test_accept_and_bounded_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _load_shipped_module()
        body = json.dumps(BASE_SOURCE_EVIDENCE).encode("utf-8")
        resp = FakeHTTPResponse(body, final_url=self.URL)
        monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=10: resp)

        data = mod.load_from_url(self.URL)
        assert data["host"] == "sg01"
        assert data["role"] == "source"
        # read must be strictly bounded at 1 MiB + 1
        assert resp.read_sizes == [1048576 + 1]

    def test_load_evidence_https_validates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _load_shipped_module()
        body = json.dumps(BASE_CANDIDATE_EVIDENCE).encode("utf-8")
        resp = FakeHTTPResponse(body, final_url=self.URL)
        monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=10: resp)

        ev = mod.load_evidence(self.URL, role="candidate", host="cursor-box")
        assert ev["host"] == "cursor-box"

    def test_url_loader_uses_direct_urlopen(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The URL loader uses the standard-library opener directly."""
        mod = _load_shipped_module()
        body = json.dumps(BASE_SOURCE_EVIDENCE).encode("utf-8")
        resp = FakeHTTPResponse(body, final_url=self.URL)
        monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=10: resp)

        data = mod.load_from_url(self.URL)
        assert data["host"] == "sg01"

    def test_redirect_mismatch_rejected(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        mod = _load_shipped_module()
        body = json.dumps(BASE_SOURCE_EVIDENCE).encode("utf-8")
        resp = FakeHTTPResponse(body, final_url="https://elsewhere.example/x.json")
        monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=10: resp)

        with pytest.raises(SystemExit) as exc:
            mod.load_from_url(self.URL)
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "redirect" in err
        assert "evidence.example" not in err

    def test_content_type_must_be_exact_json_media_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _load_shipped_module()
        body = json.dumps(BASE_SOURCE_EVIDENCE).encode("utf-8")

        # Parameterized JSON media type accepted (case-insensitive media type only)
        resp = FakeHTTPResponse(body, content_type="Application/JSON; charset=utf-8", final_url=self.URL)
        monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=10: resp)
        assert mod.load_from_url(self.URL)["host"] == "sg01"

        # A complex type containing application/json is NOT acceptable
        resp2 = FakeHTTPResponse(body, content_type="text/application/json", final_url=self.URL)
        monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=10: resp2)
        with pytest.raises(SystemExit) as exc:
            mod.load_from_url(self.URL)
        assert exc.value.code == 1

        # Plain HTML rejected
        resp3 = FakeHTTPResponse(body, content_type="text/html", final_url=self.URL)
        monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=10: resp3)
        with pytest.raises(SystemExit) as exc:
            mod.load_from_url(self.URL)
        assert exc.value.code == 1

    def test_malformed_json_rejected(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        mod = _load_shipped_module()
        resp = FakeHTTPResponse(b"{not json", final_url=self.URL)
        monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=10: resp)

        with pytest.raises(SystemExit) as exc:
            mod.load_from_url(self.URL)
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "evidence.example" not in err

    def test_oversized_response_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _load_shipped_module()
        resp = FakeHTTPResponse(b"x" * (1048576 + 100), final_url=self.URL)
        monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=10: resp)

        with pytest.raises(SystemExit) as exc:
            mod.load_from_url(self.URL)
        assert exc.value.code == 1
        assert resp.read_sizes == [1048576 + 1]

    def test_grammar_rejected_before_any_fetch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """HTTP/credentials/query/fragment URLs are rejected without ever calling urlopen."""
        mod = _load_shipped_module()
        called = []

        def boom(*args: Any, **kwargs: Any) -> Any:
            called.append(1)
            raise AssertionError("urlopen must not be called")

        monkeypatch.setattr(urllib.request, "urlopen", boom)
        for bad_url in (
            "http://example.com/e.json",
            "https://user:pass@example.com/e.json",
            "https://example.com/e.json?token=1",
            "https://example.com/e.json#frag",
        ):
            with pytest.raises(SystemExit) as exc:
                mod.load_from_url(bad_url)
            assert exc.value.code == 1
        assert called == []


class TestSchedulerAndAccessAnomaliesUnexplainable:
    """Requirement 8: scheduler mode/env/arg and Access-control anomalies can never be explained."""

    CASES = [
        ("tasker.scheduler_mode", "enabled", "paused"),
        ("tasker.scheduler_env_disabled", True, False),
        ("tasker.scheduler_arg_disabled", True, False),
        ("authority.access_protected", True, False),
    ]

    def test_attempted_explanation_stays_blocking(self, tmp_path: Path) -> None:
        for check_id, s_val, c_val in self.CASES:
            src_file = tmp_path / "src.json"
            cand_file = tmp_path / "cand.json"
            out_json = tmp_path / "out.json"
            out_md = tmp_path / "out.md"
            exp_file = tmp_path / "explanations.json"

            src_ev = copy.deepcopy(BASE_SOURCE_EVIDENCE)
            cand_ev = copy.deepcopy(BASE_CANDIDATE_EVIDENCE)
            if check_id.startswith("authority."):
                src_ev["authority"][check_id[len("authority."):]] = s_val
                cand_ev["authority"][check_id[len("authority."):]] = c_val
            else:
                src_ev["tasker"][check_id[len("tasker."):]] = s_val
                cand_ev["tasker"][check_id[len("tasker."):]] = c_val
            src_file.write_text(json.dumps(src_ev))
            cand_file.write_text(json.dumps(cand_ev))

            exp = {
                "schema_version": "portfolio-lab-migration-explanations/v1",
                "entries": [{
                    "check_id": check_id,
                    "source_fingerprint": canonical_fingerprint(s_val),
                    "candidate_fingerprint": canonical_fingerprint(c_val),
                    "reason": "attempted operator downgrade of scheduler or access anomaly",
                }],
            }
            exp_file.write_text(json.dumps(exp))

            res = run_compare([
                "--source", str(src_file),
                "--candidate", str(cand_file),
                "--output-json", str(out_json),
                "--output-markdown", str(out_md),
                "--explanations", str(exp_file),
            ])
            assert res.returncode == 2, f"{check_id} was downgraded: {res.returncode}"
            rep = json.loads(out_json.read_text())
            assert rep["summary"]["counts"]["explained"] == 0, check_id
            diff = next(d for d in rep["differences"] if d["check_id"] == check_id)
            assert diff["classification"] == "blocking", check_id
            assert rep["summary"]["counts"]["unavailable"] >= 1, check_id


class TestSameSideSafetyInvariants:
    """Requirement: equal unsafe states block; healthy/loopback must be true on both sides."""

    @staticmethod
    def _run(tmp_path: Path, src_mut, cand_mut, label: str) -> dict[str, Any]:
        src_file = tmp_path / "src.json"
        cand_file = tmp_path / "cand.json"
        out_json = tmp_path / "out.json"
        out_md = tmp_path / "out.md"

        src_ev = copy.deepcopy(BASE_SOURCE_EVIDENCE)
        cand_ev = copy.deepcopy(BASE_CANDIDATE_EVIDENCE)
        src_mut(src_ev)
        cand_mut(cand_ev)
        src_file.write_text(json.dumps(src_ev))
        cand_file.write_text(json.dumps(cand_ev))

        res = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
        ])
        assert res.returncode == 2, f"{label} not blocked"
        return json.loads(out_json.read_text())

    def test_healthy_false_on_both_sides_blocks(self, tmp_path: Path) -> None:
        rep = self._run(
            tmp_path,
            lambda e: e["authority"].update({"healthy": False}),
            lambda e: e["authority"].update({"healthy": False}),
            "healthy False/False",
        )
        assert any(d["check_id"] == "authority.healthy" and d["classification"] == "blocking" for d in rep["differences"])

    def test_loopback_false_on_both_sides_blocks(self, tmp_path: Path) -> None:
        rep = self._run(
            tmp_path,
            lambda e: e["authority"].update({"public_origin_loopback_only": False}),
            lambda e: e["authority"].update({"public_origin_loopback_only": False}),
            "loopback False/False",
        )
        assert any(
            d["check_id"] == "authority.public_origin_loopback_only" and d["classification"] == "blocking"
            for d in rep["differences"]
        )

    def test_equal_unsafe_scheduler_and_authority_states_block(self, tmp_path: Path) -> None:
        rep = self._run(
            tmp_path,
            lambda e: e["tasker"].update({"scheduler_mode": "disabled"}),
            lambda e: e["tasker"].update({"scheduler_mode": "disabled"}),
            "both schedulers disabled",
        )
        assert any(d["check_id"] == "tasker.scheduler_mode" and d["classification"] == "blocking" for d in rep["differences"])

        rep = self._run(
            tmp_path,
            lambda e: e["authority"].update({"authoritative": True}),
            lambda e: e["authority"].update({"authoritative": True}),
            "both authoritative",
        )
        assert any(d["check_id"] == "authority.authoritative" and d["classification"] == "blocking" for d in rep["differences"])

        rep = self._run(
            tmp_path,
            lambda e: e["authority"].update({"access_protected": True}),
            lambda e: e["authority"].update({"access_protected": True}),
            "both access protected",
        )
        assert any(d["check_id"] == "authority.access_protected" and d["classification"] == "blocking" for d in rep["differences"])
    def test_expected_scheduler_fields_require_complete_safe_pair(self, tmp_path: Path) -> None:
        """An unsafe tasker field blocks every intended scheduler asymmetry."""
        rep = self._run(
            tmp_path,
            lambda e: None,
            lambda e: e["tasker"].update({"scheduler_instances": 1}),
            "candidate scheduler instance count is unsafe",
        )
        for check_id in (
            "tasker.scheduler_mode",
            "tasker.scheduler_instances",
            "tasker.scheduler_env_disabled",
            "tasker.scheduler_arg_disabled",
        ):
            diff = next(d for d in rep["differences"] if d["check_id"] == check_id)
            assert diff["classification"] == "blocking", check_id

    def test_expected_authority_fields_require_complete_safe_pair(self, tmp_path: Path) -> None:
        """An unsafe authority field blocks every intended authority asymmetry."""
        rep = self._run(
            tmp_path,
            lambda e: e["authority"].update({"healthy": False}),
            lambda e: None,
            "source authority health is unsafe",
        )
        for check_id in ("authority.authoritative", "authority.access_protected"):
            diff = next(d for d in rep["differences"] if d["check_id"] == check_id)
            assert diff["classification"] == "blocking", check_id


class TestPathGrammarHostileCases:
    """Requirement 4: hostile slash-bearing fields are rejected by strict validators."""

    SLASH_FIELD_CASES = {
        "digests.static": lambda e, k: e["digests"]["static"].update({k: "c" * 64}),
        "digests.runtime": lambda e, k: e["digests"]["runtime"].update({k: "d" * 64}),
        "schemas.key": lambda e, k: e["schemas"].update({k: "signals-data/v1"}),
        "schemas.value": lambda e, k: e["schemas"].update({"extra.json": k}),
        "freshness.key": lambda e, k: e["freshness"].update({k: {"generated_at": VALID_TIME, "age_seconds": 1, "max_age_seconds": 900}}),
        "endpoints.key": lambda e, k: e["endpoints"].update({k: {"status": 200, "content_type": "application/json", "schema_version": None, "body_sha256": "1" * 64}}),
        "endpoints.schema": lambda e, k: e["endpoints"]["/"].update({"schema_version": k}),
        "tasker.status_schema": lambda e, k: e["tasker"].update({"status_schema": k}),
    }

    HOSTILE_NAMES = [
        "/etc/passwd",
        "../escape",
        "a/../b",
        "a//b",
        "a/",  # trailing slash
        "\\evil",
        "usr:pass@host/x",
        "bad\x01name",
        "./dot",
    ]

    HOSTILE_ENDPOINT_KEYS = [
        "//double",
        "/a/../b",
        "/a//b",
        "http://example.com/x",
        "/a?q=1",
        "/a#frag",
        "/a\\b",
        "/a@b",
        "no-leading-slash",
    ]

    HOSTILE_SCHEMA_STRINGS = [
        "/leading",
        "../v1",
        "a\\b",
        "a@b",
        "a=b",
        "us:er@host/v1",
    ]

    HOSTILE_CHECK_IDS = [
        "https://x/y",
        "endpoints./a/../b.status",
        "sqlite..x",
        "git.commit.extra",
        "freshness./x.generated_at",
        "schemas./abs",
        "unknown.dim",
        "tasker.scheduler_mode.extra",
        "recovery.nope",
    ]

    @staticmethod
    def _resolve_case(which: str, k: str):
        def mutate(e):
            TestPathGrammarHostileCases.SLASH_FIELD_CASES[which](e, k)
        return mutate

    def test_hostile_relative_and_endpoint_fields(self, tmp_path: Path) -> None:
        for which in ("digests.static", "digests.runtime", "schemas.key", "freshness.key"):
            for k in self.HOSTILE_NAMES:
                src_file = tmp_path / "src.json"
                cand_file = tmp_path / "cand.json"
                src_ev = copy.deepcopy(BASE_SOURCE_EVIDENCE)
                self._resolve_case(which, k)(src_ev)
                src_file.write_text(json.dumps(src_ev))
                cand_file.write_text(json.dumps(BASE_CANDIDATE_EVIDENCE))
                res = run_compare(["--source", str(src_file), "--candidate", str(cand_file),
                                   "--output-json", str(tmp_path / "o.json"), "--output-markdown", str(tmp_path / "o.md")])
                assert res.returncode == 1, f"{which} allowed {k!r}"

        for k in self.HOSTILE_ENDPOINT_KEYS:
            src_file = tmp_path / "src.json"
            cand_file = tmp_path / "cand.json"
            src_ev = copy.deepcopy(BASE_SOURCE_EVIDENCE)
            self._resolve_case("endpoints.key", k)(src_ev)
            src_file.write_text(json.dumps(src_ev))
            cand_file.write_text(json.dumps(BASE_CANDIDATE_EVIDENCE))
            res = run_compare(["--source", str(src_file), "--candidate", str(cand_file),
                               "--output-json", str(tmp_path / "o.json"), "--output-markdown", str(tmp_path / "o.md")])
            assert res.returncode == 1, f"endpoints allowed {k!r}"

        for which in ("schemas.value", "endpoints.schema", "tasker.status_schema"):
            for k in self.HOSTILE_SCHEMA_STRINGS:
                src_file = tmp_path / "src.json"
                cand_file = tmp_path / "cand.json"
                src_ev = copy.deepcopy(BASE_SOURCE_EVIDENCE)
                self._resolve_case(which, k)(src_ev)
                src_file.write_text(json.dumps(src_ev))
                cand_file.write_text(json.dumps(BASE_CANDIDATE_EVIDENCE))
                res = run_compare(["--source", str(src_file), "--candidate", str(cand_file),
                                   "--output-json", str(tmp_path / "o.json"), "--output-markdown", str(tmp_path / "o.md")])
                assert res.returncode == 1, f"{which} allowed {k!r}"

    def test_recursive_sensitive_key_in_explanations_rejected(self, tmp_path: Path) -> None:
        """Sensitive explanation keys are rejected recursively before comparison."""
        src_file = tmp_path / "src.json"
        cand_file = tmp_path / "cand.json"
        out_json = tmp_path / "o.json"
        out_md = tmp_path / "o.md"
        exp_file = tmp_path / "exp.json"
        src_file.write_text(json.dumps(BASE_SOURCE_EVIDENCE))
        cand_file.write_text(json.dumps(BASE_CANDIDATE_EVIDENCE))
        exp_file.write_text(json.dumps({
            "schema_version": "portfolio-lab-migration-explanations/v1",
            "entries": [{
                "check_id": "git.commit",
                "source_fingerprint": "1" * 64,
                "candidate_fingerprint": "2" * 64,
                "reason": "safe reason",
                "nested": {"token_value": "hidden"},
            }],
        }))
        res = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
            "--explanations", str(exp_file),
        ])
        assert res.returncode == 1

        src_file = tmp_path / "src.json"
        cand_file = tmp_path / "cand.json"
        out_json = tmp_path / "o.json"
        out_md = tmp_path / "o.md"
        src_file.write_text(json.dumps(BASE_SOURCE_EVIDENCE))
        cand_file.write_text(json.dumps(BASE_CANDIDATE_EVIDENCE))

        for cid in self.HOSTILE_CHECK_IDS:
            exp_file = tmp_path / "exp.json"
            exp_file.write_text(json.dumps({
                "schema_version": "portfolio-lab-migration-explanations/v1",
                "entries": [{
                    "check_id": cid,
                    "source_fingerprint": "1" * 64,
                    "candidate_fingerprint": "2" * 64,
                    "reason": "hostile check id attempt",
                }],
            }))
            res = run_compare(["--source", str(src_file), "--candidate", str(cand_file),
                               "--output-json", str(out_json), "--output-markdown", str(out_md),
                               "--explanations", str(exp_file)])
            assert res.returncode == 1, f"explanations allowed check_id {cid!r}"

    def test_legitimate_slash_forms_preserved(self, tmp_path: Path) -> None:
        """assets/index.js, /data/index.json, portfolio-lab-static-release/v1, route check ids work."""
        out_json = tmp_path / "o.json"
        out_md = tmp_path / "o.md"
        exp_file = tmp_path / "exp.json"

        src_ev = copy.deepcopy(BASE_SOURCE_EVIDENCE)
        cand_ev = copy.deepcopy(BASE_CANDIDATE_EVIDENCE)
        for ev in (src_ev, cand_ev):
            ev["digests"]["static"]["assets/deep/app.js"] = "c" * 64
            ev["digests"]["runtime"]["state/current.json"] = "d" * 64
            ev["schemas"]["charts/overview.json"] = "portfolio-lab-static-release/v1"
            ev["freshness"]["data/signals.json"] = {
                "generated_at": VALID_TIME,
                "age_seconds": 5,
                "max_age_seconds": 900,
            }
        # Explainable freshness delta on the slash-bearing key, explained via a route-form check id
        cand_ev["freshness"]["data/signals.json"]["generated_at"] = "2026-09-03T12:00:10+00:00"
        src_file = tmp_path / "src.json"
        cand_file = tmp_path / "cand.json"
        src_file.write_text(json.dumps(src_ev))
        cand_file.write_text(json.dumps(cand_ev))

        exp_file.write_text(json.dumps({
            "schema_version": "portfolio-lab-migration-explanations/v1",
            "entries": [{
                "check_id": "freshness.data/signals.json.generated_at",
                "source_fingerprint": canonical_fingerprint(VALID_TIME),
                "candidate_fingerprint": canonical_fingerprint("2026-09-03T12:00:10+00:00"),
                "reason": "freshness regenerated within window",
            }],
        }))

        res = run_compare(["--source", str(src_file), "--candidate", str(cand_file),
                           "--output-json", str(out_json), "--output-markdown", str(out_md),
                           "--explanations", str(exp_file)])
        assert res.returncode == 0, f"legitimate slash forms rejected: {res.stderr}"
        rep = json.loads(out_json.read_text())
        assert rep["summary"]["verdict"] == "pass"
        assert rep["summary"]["counts"]["explained"] == 1


class TestFreshnessDeltaAndFiniteNumbers:
    """Requirement 5: delta is finite/bounded; NaN/Infinity never enters evidence."""

    def test_max_freshness_delta_bounds(self, tmp_path: Path) -> None:
        src_file = tmp_path / "src.json"
        cand_file = tmp_path / "cand.json"
        out_json = tmp_path / "o.json"
        out_md = tmp_path / "o.md"
        src_file.write_text(json.dumps(BASE_SOURCE_EVIDENCE))
        cand_file.write_text(json.dumps(BASE_CANDIDATE_EVIDENCE))

        for bad in ("nan", "inf", "-inf", "-1", "86401"):
            res = run_compare(["--source", str(src_file), "--candidate", str(cand_file),
                               "--output-json", str(out_json), "--output-markdown", str(out_md),
                               "--max-freshness-delta-seconds", bad])
            assert res.returncode == 1, f"delta {bad} accepted"

        # Exactly the upper bound works
        res = run_compare(["--source", str(src_file), "--candidate", str(cand_file),
                           "--output-json", str(out_json), "--output-markdown", str(out_md),
                           "--max-freshness-delta-seconds", "86400"])
        assert res.returncode == 0

        # Zero delta works when freshness is identical
        cand_equal = copy.deepcopy(BASE_CANDIDATE_EVIDENCE)
        cand_equal["freshness"]["signals.json"]["age_seconds"] = 10
        cand_file.write_text(json.dumps(cand_equal))
        res = run_compare(["--source", str(src_file), "--candidate", str(cand_file),
                           "--output-json", str(out_json), "--output-markdown", str(out_md),
                           "--max-freshness-delta-seconds", "0"])
        assert res.returncode == 0

    def test_non_finite_numbers_rejected(self, tmp_path: Path) -> None:
        src_file = tmp_path / "src.json"
        cand_file = tmp_path / "cand.json"
        out_json = tmp_path / "o.json"
        out_md = tmp_path / "o.md"
        cand_file.write_text(json.dumps(BASE_CANDIDATE_EVIDENCE))

        # Literal NaN constant
        bad = copy.deepcopy(BASE_SOURCE_EVIDENCE)
        bad["allocation"]["SPY"] = float("nan")
        src_file.write_text(json.dumps(bad))
        res = run_compare(["--source", str(src_file), "--candidate", str(cand_file),
                           "--output-json", str(out_json), "--output-markdown", str(out_md)])
        assert res.returncode == 1

        # Literal Infinity constant
        bad = copy.deepcopy(BASE_SOURCE_EVIDENCE)
        bad["allocation"]["GLD"] = float("inf")
        src_file.write_text(json.dumps(bad))
        res = run_compare(["--source", str(src_file), "--candidate", str(cand_file),
                           "--output-json", str(out_json), "--output-markdown", str(out_md)])
        assert res.returncode == 1

        # Exponent overflow parses to inf and must be rejected by finite checks
        text = json.dumps(BASE_SOURCE_EVIDENCE).replace('"GLD": 0.38', '"GLD": 1e999')
        src_file.write_text(text)
        res = run_compare(["--source", str(src_file), "--candidate", str(cand_file),
                           "--output-json", str(out_json), "--output-markdown", str(out_md)])
        assert res.returncode == 1

        # Infinity freshness age
        text = json.dumps(BASE_SOURCE_EVIDENCE).replace('"age_seconds": 10', '"age_seconds": Infinity')
        src_file.write_text(text)
        res = run_compare(["--source", str(src_file), "--candidate", str(cand_file),
                           "--output-json", str(out_json), "--output-markdown", str(out_md)])
        assert res.returncode == 1


class TestStaleExplanationDimensionAndBlockedSuffix:
    """Requirement 6: stale explanations use the 'explanations' dimension; blocked suffix is exact and ordered."""

    def test_stale_explanation_uses_explanations_dimension(self, tmp_path: Path) -> None:
        src_file = tmp_path / "src.json"
        cand_file = tmp_path / "cand.json"
        out_json = tmp_path / "o.json"
        out_md = tmp_path / "o.md"
        exp_file = tmp_path / "exp.json"

        src_file.write_text(json.dumps(BASE_SOURCE_EVIDENCE))
        cand_file.write_text(json.dumps(BASE_CANDIDATE_EVIDENCE))
        exp_file.write_text(json.dumps({
            "schema_version": "portfolio-lab-migration-explanations/v1",
            "entries": [{
                "check_id": "endpoints./data/signals.json.body_sha256",
                "source_fingerprint": "1" * 64,
                "candidate_fingerprint": "2" * 64,
                "reason": "stale fingerprint entry",
            }],
        }))

        res = run_compare(["--source", str(src_file), "--candidate", str(cand_file),
                           "--output-json", str(out_json), "--output-markdown", str(out_md),
                           "--explanations", str(exp_file)])
        assert res.returncode == 2
        rep = json.loads(out_json.read_text())
        stale = [d for d in rep["differences"] if d["check_id"].startswith("explanation.")]
        assert len(stale) == 1
        assert stale[0]["dimension"] == "explanations"
        md = out_md.read_text()
        assert "**explanations:**" in md

    def test_blocked_markdown_ordered_suffix(self, tmp_path: Path) -> None:
        src_file = tmp_path / "src.json"
        cand_file = tmp_path / "cand.json"
        out_json = tmp_path / "o.json"
        out_md = tmp_path / "o.md"

        bad_cand = copy.deepcopy(BASE_CANDIDATE_EVIDENCE)
        bad_cand["git"]["commit"] = "0" * 40
        bad_cand["git"]["bundle_source_commit"] = "0" * 40
        bad_cand["release"]["source_git_sha"] = "0" * 40
        bad_cand["release"]["manifest_sha256"] = "0" * 64
        src_file.write_text(json.dumps(BASE_SOURCE_EVIDENCE))
        cand_file.write_text(json.dumps(bad_cand))

        res = run_compare(["--source", str(src_file), "--candidate", str(cand_file),
                           "--output-json", str(out_json), "--output-markdown", str(out_md)])
        assert res.returncode == 2
        md = out_md.read_text()
        expected_suffix = (
            "Dry run blocked\n"
            "- `git.bundle_source_commit`\n"
            "- `git.commit`\n"
            "- `release.manifest_sha256`\n"
            "- `release.source_git_sha`\n"
            "\n"
            "Retained safe state: sg01 remains authoritative; cursor-box scheduler remains disabled."
        )
        assert md.rstrip().endswith(expected_suffix)
        assert md.rstrip().splitlines()[-1] == (
            "Retained safe state: sg01 remains authoritative; cursor-box scheduler remains disabled."
        )

    def test_freshness_expected_reasons_static(self, tmp_path: Path) -> None:
        src_file = tmp_path / "src.json"
        cand_file = tmp_path / "cand.json"
        out_json = tmp_path / "o.json"
        out_md = tmp_path / "o.md"

        cand = copy.deepcopy(BASE_CANDIDATE_EVIDENCE)
        cand["freshness"]["signals.json"] = {
            "generated_at": "2026-09-03T12:00:10+00:00",
            "age_seconds": 20,
            "max_age_seconds": 900,
        }
        src_file.write_text(json.dumps(BASE_SOURCE_EVIDENCE))
        cand_file.write_text(json.dumps(cand))

        res = run_compare(["--source", str(src_file), "--candidate", str(cand_file),
                           "--output-json", str(out_json), "--output-markdown", str(out_md)])
        assert res.returncode == 0
        rep = json.loads(out_json.read_text())
        by_id = {d["check_id"]: d["reason"] for d in rep["differences"]}
        assert by_id["freshness.signals.json.generated_at"] == "generated_at collection delta within configured maximum"
        assert by_id["freshness.signals.json.age_seconds"] == "age_seconds collection delta within configured maximum"


class TestFixRoundRegressions:
    """Targeted regression tests for the 5 fix requirements in Tasks 2.5-2.6."""

    def test_source_scheduled_starts_observed_positive_allowed_and_decoupled_gates(self, tmp_path: Path) -> None:
        """Fix 1: Source may have positive scheduled_starts_observed; candidate 0; decoupled safety gates."""
        src_file = tmp_path / "src.json"
        cand_file = tmp_path / "cand.json"
        out_json = tmp_path / "out.json"
        out_md = tmp_path / "out.md"

        src = copy.deepcopy(BASE_SOURCE_EVIDENCE)
        src["tasker"]["scheduled_starts_observed"] = 42
        src_file.write_text(json.dumps(src))
        cand_file.write_text(json.dumps(BASE_CANDIDATE_EVIDENCE))

        res = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
        ])
        assert res.returncode == 0
        rep = json.loads(out_json.read_text())
        assert rep["summary"]["verdict"] == "pass"
        diff_map = {d["check_id"]: d for d in rep["differences"]}
        assert "tasker.scheduled_starts_observed" in diff_map
        assert diff_map["tasker.scheduled_starts_observed"]["classification"] == "expected"
        assert diff_map["tasker.scheduled_starts_observed"]["reason"] == "source scheduler is authoritative; candidate scheduler has zero observed starts"

        # Candidate > 0 must be blocking and unexplainable
        cand_bad = copy.deepcopy(BASE_CANDIDATE_EVIDENCE)
        cand_bad["tasker"]["scheduled_starts_observed"] = 1
        cand_file.write_text(json.dumps(cand_bad))
        res = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
        ])
        assert res.returncode == 2
        rep = json.loads(out_json.read_text())
        diff_bad = next(d for d in rep["differences"] if d["check_id"] == "tasker.scheduled_starts_observed")
        assert diff_bad["classification"] == "blocking"

        # Independent gating: Unsafe tasker field does NOT relabel correct authority asymmetries as blocking
        cand_bad_tasker = copy.deepcopy(BASE_CANDIDATE_EVIDENCE)
        cand_bad_tasker["tasker"]["scheduler_instances"] = 1  # unsafe tasker
        cand_file.write_text(json.dumps(cand_bad_tasker))
        src_file.write_text(json.dumps(BASE_SOURCE_EVIDENCE))  # normal source
        res = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
        ])
        assert res.returncode == 2
        rep = json.loads(out_json.read_text())
        diffs = {d["check_id"]: d["classification"] for d in rep["differences"]}
        assert diffs["authority.authoritative"] == "expected"
        assert diffs["authority.access_protected"] == "expected"
        assert diffs["tasker.scheduler_instances"] == "blocking"

        # Independent gating: Unsafe authority field does NOT relabel correct tasker asymmetries as blocking
        cand_bad_auth = copy.deepcopy(BASE_CANDIDATE_EVIDENCE)
        cand_bad_auth["authority"]["healthy"] = False  # unsafe authority
        cand_file.write_text(json.dumps(cand_bad_auth))
        res = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
        ])
        assert res.returncode == 2
        rep = json.loads(out_json.read_text())
        diffs = {d["check_id"]: d["classification"] for d in rep["differences"]}
        assert diffs["tasker.scheduler_mode"] == "expected"
        assert diffs["tasker.scheduler_instances"] == "expected"
        assert diffs["authority.healthy"] == "blocking"

    def test_explanation_reason_grammar_c0_c1_del_and_credential_reject(self, tmp_path: Path) -> None:
        """Fix 2: Reason rejects all C0, DEL, C1 controls and credential/scheme punctuation (: and @)."""
        src_file = tmp_path / "src.json"
        cand_file = tmp_path / "cand.json"
        out_json = tmp_path / "out.json"
        out_md = tmp_path / "out.md"
        exp_file = tmp_path / "exp.json"

        cand = copy.deepcopy(BASE_CANDIDATE_EVIDENCE)
        cand["endpoints"]["/data/signals.json"]["body_sha256"] = "9" * 64
        src_file.write_text(json.dumps(BASE_SOURCE_EVIDENCE))
        cand_file.write_text(json.dumps(cand))

        s_fp = hashlib.sha256(json.dumps("4" * 64).encode("utf-8")).hexdigest()
        c_fp = hashlib.sha256(json.dumps("9" * 64).encode("utf-8")).hexdigest()

        for bad_reason in [
            "bad reason with \x1b escape",
            "bad reason with \x7f del",
            "bad reason with \x80 c1 control",
            "bad reason with \x9f c1 control",
            "bad reason with user:pass@host",
            "bad reason with colon: value",
            "bad reason with at@sign",
        ]:
            exp_file.write_text(json.dumps({
                "schema_version": "portfolio-lab-migration-explanations/v1",
                "entries": [{
                    "check_id": "endpoints./data/signals.json.body_sha256",
                    "source_fingerprint": s_fp,
                    "candidate_fingerprint": c_fp,
                    "reason": bad_reason,
                }],
            }))
            res = run_compare([
                "--source", str(src_file),
                "--candidate", str(cand_file),
                "--output-json", str(out_json),
                "--output-markdown", str(out_md),
                "--explanations", str(exp_file),
            ])
            assert res.returncode == 1, f"bad reason accepted: {bad_reason}"
            assert bad_reason not in res.stderr

    def test_endpoint_evidence_content_type_no_parameters_null_schema_and_null_digest(self, tmp_path: Path) -> None:
        """Fix 3: Endpoint content_type without parameters, null schema only for HTML, null digest only for status."""
        src_file = tmp_path / "src.json"
        cand_file = tmp_path / "cand.json"
        out_json = tmp_path / "out.json"
        out_md = tmp_path / "out.md"
        cand_file.write_text(json.dumps(BASE_CANDIDATE_EVIDENCE))

        # content_type with parameters must be rejected in evidence
        bad_src = copy.deepcopy(BASE_SOURCE_EVIDENCE)
        bad_src["endpoints"]["/data/signals.json"]["content_type"] = "application/json; charset=utf-8"
        src_file.write_text(json.dumps(bad_src))
        res = run_compare(["--source", str(src_file), "--candidate", str(cand_file),
                           "--output-json", str(out_json), "--output-markdown", str(out_md)])
        assert res.returncode == 1

        # content_type with controls must be rejected
        bad_src = copy.deepcopy(BASE_SOURCE_EVIDENCE)
        bad_src["endpoints"]["/data/signals.json"]["content_type"] = "application/json\r\n"
        src_file.write_text(json.dumps(bad_src))
        res = run_compare(["--source", str(src_file), "--candidate", str(cand_file),
                           "--output-json", str(out_json), "--output-markdown", str(out_md)])
        assert res.returncode == 1

        # JSON endpoint with schema_version: null must be rejected
        bad_src = copy.deepcopy(BASE_SOURCE_EVIDENCE)
        bad_src["endpoints"]["/data/signals.json"]["schema_version"] = None
        src_file.write_text(json.dumps(bad_src))
        res = run_compare(["--source", str(src_file), "--candidate", str(cand_file),
                           "--output-json", str(out_json), "--output-markdown", str(out_md)])
        assert res.returncode == 1

        # Non-status endpoint with body_sha256: null must be rejected in validation
        bad_src = copy.deepcopy(BASE_SOURCE_EVIDENCE)
        bad_src["endpoints"]["/data/signals.json"]["body_sha256"] = None
        src_file.write_text(json.dumps(bad_src))
        res = run_compare(["--source", str(src_file), "--candidate", str(cand_file),
                           "--output-json", str(out_json), "--output-markdown", str(out_md)])
        assert res.returncode == 1

    def test_cli_diagnostic_redaction_argparse_errors_never_echo_token(self, tmp_path: Path) -> None:
        """Fix 4: All argparse errors exit 1 with bounded static diagnostic that never echoes raw token."""
        sentinel = "SECRET_SENTINEL_TOKEN_12345"
        res = run_compare(["--max-freshness-delta-seconds", sentinel])
        assert res.returncode == 1
        assert sentinel not in res.stderr
        assert sentinel not in res.stdout
        assert "error: " in res.stderr

    def test_atomic_backup_toctou_symlink_hardening(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fix 5: write_outputs_atomically uses follow_symlinks=False for backup copy."""
        mod = _load_shipped_module()
        called_with_kwargs: list[dict[str, Any]] = []
        real_copyfile = shutil.copyfile

        def tracked_copyfile(src: Any, dst: Any, *, follow_symlinks: bool = True) -> Any:
            called_with_kwargs.append({"src": src, "dst": dst, "follow_symlinks": follow_symlinks})
            return real_copyfile(src, dst, follow_symlinks=follow_symlinks)

        monkeypatch.setattr(mod.shutil, "copyfile", tracked_copyfile)

        out_json = tmp_path / "out.json"
        out_md = tmp_path / "out.md"
        # Pre-create files so backup copy is triggered on overwrite
        out_json.write_text("old json")
        out_md.write_text("old md")

        mod.write_outputs_atomically(out_json, "new json", out_md, "new md")
        assert len(called_with_kwargs) == 2
        for call in called_with_kwargs:
            assert call["follow_symlinks"] is False

    def test_endpoint_schema_version_rule_is_media_type_based_not_route_based(self, tmp_path: Path) -> None:
        """Fix Round 3: schema_version: null is permitted only for HTML (case-insensitively), regardless of route.

        - Optional endpoint /about with text/html and schema_version: null is accepted.
        - Route / with application/json and schema_version: null is rejected.
        """
        src_file = tmp_path / "src.json"
        cand_file = tmp_path / "cand.json"
        out_json = tmp_path / "out.json"
        out_md = tmp_path / "out.md"

        # 1. Valid optional HTML endpoint with schema_version: null present in both source and candidate
        src_html = copy.deepcopy(BASE_SOURCE_EVIDENCE)
        cand_html = copy.deepcopy(BASE_CANDIDATE_EVIDENCE)
        about_ep = {
            "status": 200,
            "content_type": "text/html",
            "schema_version": None,
            "body_sha256": "5" * 64,
        }
        src_html["endpoints"]["/about"] = about_ep
        cand_html["endpoints"]["/about"] = about_ep
        src_file.write_text(json.dumps(src_html))
        cand_file.write_text(json.dumps(cand_html))

        res = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
        ])
        assert res.returncode == 0, f"Valid optional HTML endpoint rejected: {res.stderr}"
        rep = json.loads(out_json.read_text())
        assert rep["summary"]["verdict"] == "pass"

        # 2. Route / changed to application/json with schema_version: null must be rejected
        src_json_root = copy.deepcopy(BASE_SOURCE_EVIDENCE)
        cand_json_root = copy.deepcopy(BASE_CANDIDATE_EVIDENCE)
        src_json_root["endpoints"]["/"]["content_type"] = "application/json"
        src_json_root["endpoints"]["/"]["schema_version"] = None
        cand_json_root["endpoints"]["/"]["content_type"] = "application/json"
        cand_json_root["endpoints"]["/"]["schema_version"] = None
        src_file.write_text(json.dumps(src_json_root))
        cand_file.write_text(json.dumps(cand_json_root))

        res = run_compare([
            "--source", str(src_file),
            "--candidate", str(cand_file),
            "--output-json", str(out_json),
            "--output-markdown", str(out_md),
        ])
        assert res.returncode == 1, "Route / with application/json and schema_version: null was unexpectedly accepted"
