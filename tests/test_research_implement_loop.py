"""Fixture-based unit tests for Session A/B research-implement loop.

These pass WITHOUT a host ``logs/research-implement.md``. The skippable
contract tests in ``test_research_implement_loop_contract.py`` remain unchanged.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.research_implement.queue import (
    QUEUE_CAPACITY,
    REQUIRED_FIELDS,
    append_queue_item,
    count_open,
    first_b_pick,
    format_queue_item,
    is_b_pickable,
    is_complete_six_field,
    parse_queue_items,
    render_queue_count,
)
from src.research_implement.session_a import run_session_a
from src.research_implement.session_b import (
    SchedulerDeleteForbidden,
    decode_fields,
    format_decode_report,
    run_session_b,
    scheduler_delete,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "research_implement"


def _load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_required_fields_are_exactly_six():
    assert REQUIRED_FIELDS == (
        "title",
        "acceptance",
        "risks",
        "file_touch",
        "breaking_change",
        "redeploy_notes",
    )
    assert QUEUE_CAPACITY == 10


def test_parse_one_open_ready_item():
    items = parse_queue_items(_load("one_open_ready.md"))
    assert len(items) == 1
    item = items[0]
    assert item.item_id == "Q1"
    assert is_complete_six_field(item)
    assert is_b_pickable(item)
    assert item.ready_for_implement.lower() == "yes"
    assert count_open(items) == 1
    assert first_b_pick(items) is item


def test_shipped_does_not_count_as_open():
    items = parse_queue_items(_load("shipped_only.md"))
    assert len(items) == 1
    assert not is_b_pickable(items[0])
    assert count_open(items) == 0
    assert first_b_pick(items) is None


def test_incomplete_open_is_not_b_pickable():
    items = parse_queue_items(_load("incomplete_open.md"))
    assert len(items) == 1
    assert items[0].status.upper().startswith("OPEN")
    assert not is_complete_six_field(items[0])
    assert not is_b_pickable(items[0])
    assert count_open(items) == 0


def test_parser_ignores_watch_and_project_work_sections():
    items = parse_queue_items(_load("watch_lookalike.md"))
    assert [i.item_id for i in items] == ["Q3"]
    pick = first_b_pick(items)
    assert pick is not None
    assert pick.title == "Real ready Queue item"
    # Watch lookalike title must not appear in Queue parse.
    assert all("Watch lookalike" not in i.title for i in items)


def test_format_and_append_roundtrip():
    base = _load("empty_queue.md")
    assert count_open(parse_queue_items(base)) == 0
    block = format_queue_item(
        item_id="Q1",
        heading="Appended ready item",
        title="Appended ready item",
        acceptance="pytest EXIT=0",
        risks="do not touch kill_switch",
        file_touch="write tests/x.py",
        breaking_change=False,
        redeploy_notes="none",
        status="OPEN",
        ready_for_implement="yes",
    )
    assert "ready-for-implement: yes" in block
    for i, name in enumerate(REQUIRED_FIELDS, start=1):
        assert f"{i}. **{name}**:" in block
    updated = append_queue_item(base, block)
    items = parse_queue_items(updated)
    assert count_open(items) == 1
    assert "### Q1. Appended ready item" in updated
    # Watch section still present after Queue.
    assert "## Watch" in updated
    assert updated.index("### Q1.") < updated.index("## Watch")


def test_session_a_brainstorms_at_most_one_open_when_empty():
    plan = _load("empty_queue.md")
    before = len(parse_queue_items(plan))

    def brainstorm(_items):
        return {
            "heading": "New shippable change",
            "title": "New shippable change",
            "acceptance": "pytest EXIT=0",
            "risks": "do not touch order_router",
            "file_touch": "write tests/test_new.py",
            "breaking_change": False,
            "redeploy_notes": "none",
        }

    result = run_session_a(plan, brainstorm=brainstorm)
    assert result.ok
    assert result.verdict == "queued"
    assert result.wrote_item
    assert result.open_count == 1
    assert "queued" in result.message
    after_items = parse_queue_items(result.plan_text)
    assert len(after_items) - before == 1  # appends ≤1 OPEN
    assert count_open(after_items) == 1
    assert "ready-for-implement: yes" in result.plan_text


def test_session_a_failed_fire_when_empty_and_no_candidate():
    result = run_session_a(_load("empty_queue.md"), brainstorm=None)
    assert not result.ok
    assert result.verdict == "failed"
    assert "failed fire" in result.message
    assert render_queue_count(0) in result.message


def test_session_a_light_recount_when_open_already_present():
    called = {"n": 0}

    def brainstorm(_items):
        called["n"] += 1
        return {"title": "should-not-run"}

    result = run_session_a(_load("one_open_ready.md"), brainstorm=brainstorm)
    assert result.ok
    assert result.verdict == "light"
    assert not result.wrote_item
    assert result.open_count == 1
    assert "recount only" in result.message
    assert called["n"] == 0  # must not research/brainstorm
    # Plan text unchanged (recount only).
    assert result.plan_text == _load("one_open_ready.md")


def test_session_b_idle_fire_on_empty_queue_never_deletes():
    calls = {"n": 0}

    def _spy(*_a, **_k):
        calls["n"] += 1
        raise SchedulerDeleteForbidden("spy")

    with patch("src.research_implement.session_b.scheduler_delete", _spy):
        result = run_session_b(_load("empty_queue.md"), plan_path="fixture://empty")
    assert result.ok
    assert result.verdict == "idle"
    assert result.keep_schedule is True
    assert result.scheduler_delete_called is False
    assert calls["n"] == 0  # empty-queue path never calls scheduler_delete
    assert "nothing to implement" in result.message
    assert "queue 0/10" in result.message
    assert "keep_schedule" in result.message
    # Hard guard still armed on the real symbol.
    with pytest.raises(SchedulerDeleteForbidden):
        scheduler_delete("task-id")


def test_session_b_idle_on_shipped_only():
    result = run_session_b(_load("shipped_only.md"))
    assert result.ok
    assert result.verdict == "idle"
    assert "queue 0/10" in result.message
    assert result.keep_schedule
    assert not result.scheduler_delete_called


def test_session_b_picks_first_complete_open_ignoring_watch():
    result = run_session_b(_load("watch_lookalike.md"), decode_only=True)
    assert result.ok
    assert result.verdict == "picked"
    assert result.item is not None
    assert result.item.item_id == "Q3"
    assert result.item.title == "Real ready Queue item"
    assert result.keep_schedule
    assert not result.scheduler_delete_called
    # Decode path reports Q id + six fields.
    assert "decode pick Q3:" in result.message
    assert result.decode_report is not None
    assert "1. title:" in result.decode_report
    fields = decode_fields(result.item)
    assert fields["item_id"] == "Q3"
    assert all(fields[name] for name in REQUIRED_FIELDS)


def test_session_b_picks_first_complete_skips_incomplete_and_watch():
    result = run_session_b(_load("mixed_priority.md"), decode_only=True)
    assert result.ok
    assert result.verdict == "picked"
    assert result.item is not None
    assert result.item.item_id == "Q2"
    assert result.item.title == "Second ready complete item"
    assert "Watch lookalike" not in (result.item.title or "")
    assert "Incomplete first row" not in (result.item.title or "")
    report = format_decode_report(result.item)
    assert "decode pick Q2:" in report
    for i, name in enumerate(REQUIRED_FIELDS, start=1):
        assert f"{i}. {name}:" in report


def test_session_b_refuses_incomplete_open_as_idle():
    result = run_session_b(_load("incomplete_open.md"))
    assert result.ok
    assert result.verdict == "idle"
    assert result.item is None
    assert result.decode_report is None
    assert result.keep_schedule
    assert "queue 0/10" in result.message


def test_session_b_implement_callback_can_ship_optional(tmp_path: Path):
    plan = _load("one_open_ready.md")
    seen: list[str] = []

    def implement(item):
        seen.append(item.item_id)
        return {"sha": "deadbeef", "note": "fixture ship"}

    result = run_session_b(plan, implement=implement, decode_only=False, ship_sha="deadbeef")
    assert result.ok
    assert result.verdict == "shipped"
    assert seen == ["Q1"]
    assert count_open(parse_queue_items(result.plan_text)) == 0
    assert "SHIPPED `deadbeef`" in result.plan_text
    assert result.keep_schedule
    assert not result.scheduler_delete_called


def test_session_b_cli_decode_json_path(tmp_path: Path):
    from src.research_implement.__main__ import main

    plan = tmp_path / "plan.md"
    plan.write_text(_load("one_open_ready.md"), encoding="utf-8")
    rc = main(["session-b", "--plan", str(plan), "--json"])
    assert rc == 0


def test_modules_are_distinct_from_legacy_research_agent():
    # Import alias sanity: A/B package must not pull legacy agent.
    import src.research_implement as loop
    import src.research.agent as legacy

    assert loop.__name__ == "src.research_implement"
    assert legacy.__name__ == "src.research.agent"
    assert "run_session_a" in dir(loop)
    assert "run_session_b" in dir(loop)
    assert "format_decode_report" in dir(loop)
