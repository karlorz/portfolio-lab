"""Contract tests for the Session A/B loop spec.

Reads the shipped file logs/research-implement.md. A real task is a
six-field OPEN Queue item. Session A must brainstorm when OPEN is 0.
Session B implements only from ## Queue.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = REPO_ROOT / "logs" / "research-implement.md"


def _fences(text: str) -> list[str]:
    return re.findall(r"```\n(.*?)```", text, flags=re.S)


def _session_a(text: str) -> str:
    for block in _fences(text):
        if "Session A" in block and "do not implement" in block:
            return block
    raise AssertionError("Session A fence missing")


def _session_b(text: str) -> str:
    for block in _fences(text):
        if "nothing to implement" in block:
            return block
    raise AssertionError("Session B fence missing")


def test_session_a_is_compact_and_brainstorms_when_queue_empty():
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
    assert len(session_a) < 4000


def test_session_b_picks_only_queue():
    session_b = _session_b(SPEC.read_text(encoding="utf-8"))
    assert "read ## Queue and ## Heartbeat" in session_b
    assert "nothing to implement" in session_b
    assert "Ignore ## Watch and ## Project Work" in session_b
    assert "Pick: first OPEN Queue item with all six fields" in session_b
    assert "SHIPPED does not count" in session_b


def test_session_b_never_deletes_schedule_on_empty_queue():
    session_b = _session_b(SPEC.read_text(encoding="utf-8"))
    assert "Never call scheduler_delete" in session_b
    assert "Leave the schedule running" in session_b
    assert "does not apply" in session_b
    assert "queue 0/10" in session_b
