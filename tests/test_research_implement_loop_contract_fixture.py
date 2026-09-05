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

import pytest

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


def test_beat9_fixture_six_field_open_runtime_without_host_log():
    """Queue fixture: complete six-field OPEN is B-pickable (no host log required)."""
    from unittest.mock import patch

    from src.research_implement.queue import (
        REQUIRED_FIELDS,
        count_open,
        first_b_pick,
        is_b_pickable,
        is_complete_six_field,
        parse_queue_items,
    )
    from src.research_implement.session_b import (
        SchedulerDeleteForbidden,
        run_session_b,
        scheduler_delete,
    )

    # Host logs/research-implement.md is optional; this path uses queue fixtures only.
    plan = (FIXTURES / "one_open_ready.md").read_text(encoding="utf-8")
    items = parse_queue_items(plan)
    assert len(items) == 1
    assert is_complete_six_field(items[0])
    assert is_b_pickable(items[0])
    assert count_open(items) == 1
    assert first_b_pick(items) is items[0]
    for name in REQUIRED_FIELDS:
        assert getattr(items[0], name)

    calls = {"n": 0}

    def _spy(*_a, **_k):
        calls["n"] += 1
        raise SchedulerDeleteForbidden("spy")

    with patch("src.research_implement.session_b.scheduler_delete", _spy):
        result = run_session_b(plan, decode_only=True)
    assert result.ok and result.verdict == "picked"
    assert result.item is not None and result.item.item_id == "Q1"
    assert result.keep_schedule is True
    assert result.scheduler_delete_called is False
    assert calls["n"] == 0
    with pytest.raises(SchedulerDeleteForbidden):
        scheduler_delete("task-id")


def test_beat9_fixture_idle_fire_vs_scheduler_delete_runtime():
    """Empty Queue fixture: idle queue 0/10; never calls scheduler_delete."""
    from unittest.mock import patch

    from src.research_implement.session_b import (
        SchedulerDeleteForbidden,
        run_session_b,
        scheduler_delete,
    )

    # Host logs/research-implement.md is optional; idle path uses empty_queue fixture.
    plan = (FIXTURES / "empty_queue.md").read_text(encoding="utf-8")
    calls = {"n": 0}

    def _spy(*_a, **_k):
        calls["n"] += 1
        raise SchedulerDeleteForbidden("spy")

    with patch("src.research_implement.session_b.scheduler_delete", _spy):
        result = run_session_b(plan, plan_path="fixture://empty")
    assert result.ok and result.verdict == "idle"
    assert result.keep_schedule is True
    assert result.scheduler_delete_called is False
    assert calls["n"] == 0
    assert "nothing to implement" in result.message
    assert "queue 0/10" in result.message
    with pytest.raises(SchedulerDeleteForbidden):
        scheduler_delete("task-id")


def test_beat10_fixture_malformed_incomplete_and_broken_ready_idle():
    """Beat 10 fixtures: missing fields / broken ready → idle; never scheduler_delete."""
    from unittest.mock import patch

    from src.research_implement.queue import (
        first_b_pick,
        is_b_pickable,
        is_complete_six_field,
        parse_queue_items,
    )
    from src.research_implement.session_b import (
        SESSION_RESULT_JSON_KEYS,
        SchedulerDeleteForbidden,
        run_session_b,
    )

    calls = {"n": 0}

    def _spy(*_a, **_k):
        calls["n"] += 1
        raise SchedulerDeleteForbidden("spy")

    incomplete = (FIXTURES / "incomplete_open.md").read_text(encoding="utf-8")
    items = parse_queue_items(incomplete)
    assert len(items) == 1
    assert not is_complete_six_field(items[0])
    assert not is_b_pickable(items[0])
    assert first_b_pick(items) is None

    broken = (FIXTURES / "broken_ready_flag.md").read_text(encoding="utf-8")
    broken_items = parse_queue_items(broken)
    assert len(broken_items) == 2
    assert all(is_complete_six_field(i) for i in broken_items)
    assert all(not is_b_pickable(i) for i in broken_items)
    assert first_b_pick(broken_items) is None

    for plan in (incomplete, broken):
        with patch("src.research_implement.session_b.scheduler_delete", _spy):
            result = run_session_b(plan, decode_only=True)
        assert result.ok and result.verdict == "idle"
        assert result.item is None
        assert result.keep_schedule is True
        assert result.scheduler_delete_called is False
        payload = result.to_dict()
        assert tuple(payload.keys()) == SESSION_RESULT_JSON_KEYS
        assert payload["verdict"] == "idle"
        assert payload["scheduler_delete_called"] is False
    assert calls["n"] == 0


def test_beat10_fixture_watch_lookalike_skip_and_idle_decode_json_shape(capsys):
    """Beat 10: Watch lookalike skipped; idle-decode --json shares Session B shape."""
    import json
    from unittest.mock import patch

    from src.research_implement.__main__ import main
    from src.research_implement.queue import first_b_pick, parse_queue_items
    from src.research_implement.session_b import (
        SESSION_RESULT_JSON_KEYS,
        SchedulerDeleteForbidden,
        run_session_b,
        session_b_result_dict,
    )

    calls = {"n": 0}

    def _spy(*_a, **_k):
        calls["n"] += 1
        raise SchedulerDeleteForbidden("spy")

    lookalike = (FIXTURES / "watch_lookalike.md").read_text(encoding="utf-8")
    assert [i.item_id for i in parse_queue_items(lookalike)] == ["Q3"]
    with patch("src.research_implement.session_b.scheduler_delete", _spy):
        picked = run_session_b(lookalike, decode_only=True)
    assert picked.verdict == "picked" and picked.item is not None
    assert picked.item.item_id == "Q3"
    assert picked.scheduler_delete_called is False

    watch_only = (FIXTURES / "watch_only_lookalike.md").read_text(encoding="utf-8")
    assert parse_queue_items(watch_only) == []
    assert first_b_pick([]) is None
    with patch("src.research_implement.session_b.scheduler_delete", _spy):
        idle = run_session_b(watch_only, decode_only=True)
    assert idle.verdict == "idle"
    assert idle.scheduler_delete_called is False
    assert calls["n"] == 0

    # idle-decode --json uses the same SessionResult.to_dict keys (no plan write).
    import tempfile
    from pathlib import Path as P

    with tempfile.TemporaryDirectory() as td:
        plan = P(td) / "watch_only.md"
        plan.write_text(watch_only, encoding="utf-8")
        rc = main(["idle-decode", "--plan", str(plan), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    # CLI dumps with sort_keys=True; contract is key set + values, not order.
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)
    assert payload == session_b_result_dict(idle)
    assert payload["verdict"] == "idle"
    assert payload["item"] is None
    assert payload["implement_result"] is None
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
