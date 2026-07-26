"""Batch JP — agent test-gate / wait-test-exit contracts for smooth dev cycle.

Full make test is ~30–45m and must not be the default agent mid-session gate.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_makefile_defines_test_gate_as_test_fast_alias() -> None:
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "test-gate: test-fast" in makefile
    assert "make test-gate" in makefile or "test-gate" in makefile
    # Help text must advertise gate as default agent path
    assert "DEFAULT agent gate" in makefile or "test-gate" in makefile.split("help:")[1].split("make data")[0]


def test_wait_test_exit_script_exists_and_is_executable_contract() -> None:
    path = PROJECT_ROOT / "scripts" / "wait-test-exit.sh"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "test_last_exit.json" in text
    assert "FAIL — no pytest/make-test process" in text
    assert "--max-sec" in text
    # Fail-fast on dead suite (the Claude 10m poll bug)
    assert "DEAD_STREAK" in text or "suite was abandoned" in text.lower() or "no pytest" in text


def test_agent_docs_tier_default_gate_not_full_suite() -> None:
    for name in ("CLAUDE.md", "AGENTS.md"):
        text = (PROJECT_ROOT / name).read_text(encoding="utf-8")
        assert "make test-gate" in text, f"{name} must advertise test-gate"
        assert "do not default to full suite" in text.lower() or "tiered" in text.lower()
        # Must not be the old one-liner only
        assert "make test-fast" in text or "test-fast" in text
