"""Fixture-backed parallel coverage for the Session A/B contract fences.

Encodes the same assertions as ``test_research_implement_loop_contract.py``
against ``tests/fixtures/research_implement/contract_spec.md``, so the
six-field OPEN / idle-fire vs ``scheduler_delete`` / A vs B role fences run
without a host ``logs/research-implement.md``. The skip-if-missing host
contract file is intentionally unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "research_implement"
SPEC = FIXTURES / "contract_spec.md"


def _fences(text: str) -> list[str]:
    return re.findall(r"```\n(.*?)```", text, flags=re.S)


def _session_a(text: str) -> str:
    for block in _fences(text):
        if "Session A" in block and "do not implement" in block:
            return block
    raise AssertionError("Session A fence missing from fixture contract_spec.md")


def _session_b(text: str) -> str:
    for block in _fences(text):
        if "nothing to implement" in block:
            return block
    raise AssertionError("Session B fence missing from fixture contract_spec.md")


def test_beat9_fixture_session_a_six_field_open_and_producer_role():
    """Parallel to host contract: Session A fence — six-field OPEN + producer role."""
    text = SPEC.read_text(encoding="utf-8")
    session_a = _session_a(text)
    assert "HARD GATE" not in session_a
    assert "must-search" not in session_a
    assert "| ops-followup-waitress |" not in session_a
    assert "1. title" in session_a
    assert "2. acceptance" in session_a
    assert "3. risks" in session_a
    assert "4. file_touch" in session_a
    assert "5. breaking_change" in session_a
    assert "6. redeploy_notes" in session_a
    assert "ready-for-implement: yes" in session_a
    assert "If OPEN is 0" in session_a
    assert "brainstorm" in session_a
    assert "Empty Queue + no new item = failed fire" in session_a
    assert "subprocess smoke" in session_a
    assert "Writing one = failed fire" in session_a
    assert "Skip a script that already has a dedicated tests/test_*.py" in session_a
    assert "portfolio-lab-*.sh" in session_a
    assert "PROMOTE" in session_a
    assert "recount only" in session_a
    assert "do not implement" in session_a
    assert len(session_a) < 4000


def test_beat9_fixture_session_b_queue_only_consumer_role():
    """Parallel to host contract: Session B fence — Queue-only consumer role."""
    session_b = _session_b(SPEC.read_text(encoding="utf-8"))
    assert "read ## Queue and ## Heartbeat" in session_b
    assert "nothing to implement" in session_b
    assert "Ignore ## Watch and ## Project Work" in session_b
    assert "Pick: first OPEN Queue item with all six fields" in session_b
    assert "SHIPPED does not count" in session_b
    assert "do not re-research" in session_b


def test_beat9_fixture_session_b_idle_fire_never_scheduler_delete():
    """Parallel to host contract: idle fire (queue 0/10) vs never scheduler_delete."""
    session_b = _session_b(SPEC.read_text(encoding="utf-8"))
    assert "Never call scheduler_delete" in session_b
    assert "Leave the schedule running" in session_b
    assert "does not apply" in session_b
    assert "queue 0/10" in session_b


def test_beat9_fixture_a_vs_b_roles_are_distinct():
    """A is producer (do not implement); B is consumer (do not re-research)."""
    text = SPEC.read_text(encoding="utf-8")
    session_a = _session_a(text)
    session_b = _session_b(text)
    # Producer A: brainstorm / plan; never implement.
    assert "do not implement" in session_a
    assert "brainstorm" in session_a
    assert "You produce. B implements only ## Queue." in session_a
    # Consumer B: implement; do not re-research; idle never deletes.
    assert "do not re-research" in session_b
    assert "Never call scheduler_delete" in session_b
    assert "Pick: first OPEN Queue item with all six fields" in session_b
    assert "Never call scheduler_delete" not in session_a
    # Producer vs consumer fences must be different blocks.
    assert session_a != session_b
