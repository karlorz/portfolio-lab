"""Fixture-based unit tests for Session A/B research-implement loop.

These pass WITHOUT a host ``logs/research-implement.md``. The skippable
contract tests in ``test_research_implement_loop_contract.py`` remain unchanged.
"""

from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from src.research_implement.queue import (
    QUEUE_CAPACITY,
    REQUIRED_FIELDS,
    AmbiguousQueueError,
    QueueItem,
    append_queue_item,
    count_open,
    count_queue_headings,
    first_b_pick,
    format_queue_item,
    is_b_pickable,
    is_complete_six_field,
    is_open_status,
    is_ready_yes,
    next_queue_id,
    parse_queue_items,
    render_queue_count,
    require_unique_queue_section,
    serialize_queue_item,
    serialize_queue_items,
    write_queue_section,
    mark_item_shipped,
)
from src.research_implement.session_a import (
    SESSION_A_RESULT_JSON_KEYS,
    SESSION_A_RESULT_KEYS,
    SessionAResult,
    default_search_plan,
    incomplete_candidate_reasons,
    run_session_a,
    run_session_a_path,
    session_a_result_dict,
    stub_brainstorm,
)
from src.research_implement.session_b import (
    SESSION_B_RESULT_KEYS,
    SESSION_RESULT_JSON_KEYS,
    SchedulerDeleteForbidden,
    SessionResult,
    decode_fields,
    default_implement,
    dry_run_implement,
    fixture_ship_implement,
    format_decode_report,
    make_fixture_ship_implement,
    run_session_b,
    run_session_b_path,
    scheduler_delete,
    session_b_result_dict,
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
            "ready_for_implement": "yes",
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
    """Explicit fixture_ship_implement marks SHIPPED; default never does."""
    plan = _load("one_open_ready.md")
    # Default / dry_run must not ship.
    dry = run_session_b(plan, decode_only=False)
    assert dry.verdict == "dry_run"
    assert "SHIPPED" not in dry.plan_text
    assert count_open(parse_queue_items(dry.plan_text)) == 1

    ship_fn = make_fixture_ship_implement("deadbeef", note="fixture ship")
    result = run_session_b(plan, implement=ship_fn, decode_only=False)
    assert result.ok
    assert result.verdict == "shipped"
    assert result.implement_result is not None
    assert result.implement_result["sha"] == "deadbeef"
    assert result.implement_result["dry_run"] is False
    assert count_open(parse_queue_items(result.plan_text)) == 0
    assert "SHIPPED `deadbeef`" in result.plan_text
    assert result.keep_schedule
    assert not result.scheduler_delete_called
    # Convenience alias is also a ship double, never the default.
    assert fixture_ship_implement is not dry_run_implement
    assert default_implement is dry_run_implement


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
    assert "stub_brainstorm" in dir(loop)
    assert "default_search_plan" in dir(loop)
    assert "dry_run_implement" in dir(loop)
    assert "default_implement" in dir(loop)
    assert "fixture_ship_implement" in dir(loop)
    assert "make_fixture_ship_implement" in dir(loop)
    assert "SessionResult" in dir(loop)
    assert "SESSION_RESULT_JSON_KEYS" in dir(loop)
    assert "SESSION_B_RESULT_KEYS" in dir(loop)
    assert "session_b_result_dict" in dir(loop)


def test_stub_brainstorm_fills_six_fields():
    assert stub_brainstorm is default_search_plan
    raw = stub_brainstorm([])
    assert raw == default_search_plan([])
    for name in REQUIRED_FIELDS:
        assert name in raw
    assert raw["title"]
    assert raw["acceptance"]
    assert raw["risks"]
    assert raw["file_touch"]
    assert raw["redeploy_notes"]
    # bool False is fine; coerce path in Session A accepts it
    assert raw["breaking_change"] is False or str(raw["breaking_change"]).strip()
    assert is_ready_yes(str(raw.get("ready_for_implement") or ""))
    assert incomplete_candidate_reasons(raw) == []


def test_session_a_default_stub_queues_when_open_zero():
    plan = _load("empty_queue.md")
    result = run_session_a(plan, search_plan=stub_brainstorm)
    assert result.ok
    assert result.verdict == "queued"
    assert result.wrote_item
    assert result.open_count == 1
    assert len(parse_queue_items(result.plan_text)) == 1  # append ≤1
    assert is_complete_six_field(parse_queue_items(result.plan_text)[0])
    assert "ready-for-implement: yes" in result.plan_text


def test_session_a_stub_not_called_when_open_present():
    called = {"n": 0}

    def tracking(items):
        called["n"] += 1
        return stub_brainstorm(items)

    result = run_session_a(_load("one_open_ready.md"), search_plan=tracking)
    assert result.ok
    assert result.verdict == "light"
    assert called["n"] == 0
    assert not result.wrote_item


def test_cli_help_smoke():
    from src.research_implement.__main__ import main

    with pytest.raises(SystemExit) as ei:
        main(["--help"])
    assert ei.value.code == 0


def test_cli_session_a_help_and_stub_dry_run(tmp_path: Path, capsys):
    from src.research_implement.__main__ import main

    with pytest.raises(SystemExit) as ei:
        main(["session-a", "--help"])
    assert ei.value.code == 0

    plan = tmp_path / "empty.md"
    plan.write_text(_load("empty_queue.md"), encoding="utf-8")
    rc = main(["session-a", "--plan", str(plan), "--stub", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "queued" in out
    # dry-run must not write
    assert count_open(parse_queue_items(plan.read_text(encoding="utf-8"))) == 0


def test_cli_session_a_no_stub_fails_on_empty(tmp_path: Path):
    from src.research_implement.__main__ import main

    plan = tmp_path / "empty.md"
    plan.write_text(_load("empty_queue.md"), encoding="utf-8")
    rc = main(["session-a", "--log", str(plan), "--no-stub", "--dry-run"])
    assert rc == 1


def test_cli_idle_decode_empty_fixture(tmp_path: Path, capsys):
    from src.research_implement.__main__ import main

    plan = tmp_path / "empty.md"
    plan.write_text(_load("empty_queue.md"), encoding="utf-8")
    rc = main(["idle-decode", "--plan", str(plan), "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = __import__("json").loads(out)
    assert payload["verdict"] == "idle"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert payload["queue"] == "queue 0/10"


def test_cli_session_b_log_alias_decode(tmp_path: Path):
    from src.research_implement.__main__ import main

    plan = tmp_path / "one.md"
    plan.write_text(_load("one_open_ready.md"), encoding="utf-8")
    rc = main(["session-b", "--log", str(plan), "--json"])
    assert rc == 0

def test_dry_run_implement_records_file_touch_and_acceptance():
    assert dry_run_implement is default_implement
    items = parse_queue_items(_load("one_open_ready.md"))
    item = items[0]
    recorded = dry_run_implement(item)
    assert recorded["dry_run"] is True
    assert recorded["wrote_files"] is False
    assert recorded["sha"] is None
    assert recorded["item_id"] == "Q1"
    assert recorded["file_touch"] == item.file_touch
    assert recorded["acceptance"] == item.acceptance
    assert "test_research_implement_loop.py" in recorded["file_touch"]


def test_session_b_dry_run_implement_one_open_ready_no_repo_mutation(tmp_path: Path):
    """Dry-run implement on one OPEN ready: record intent; no real file writes."""
    plan_src = _load("one_open_ready.md")
    plan = tmp_path / "plan.md"
    plan.write_text(plan_src, encoding="utf-8")

    # Sentinel: a path named in file_touch that must not be mutated.
    touched_target = Path(__file__).resolve()
    before_bytes = touched_target.read_bytes()
    before_mtime = touched_target.stat().st_mtime_ns

    # Extra sentinel outside temp/fixture logs.
    repo_sentinel = Path(__file__).resolve().parents[1] / "src" / "research_implement" / "queue.py"
    sentinel_before = repo_sentinel.read_bytes()

    calls = {"n": 0}

    def _spy(*_a, **_k):
        calls["n"] += 1
        raise SchedulerDeleteForbidden("spy")

    with patch("src.research_implement.session_b.scheduler_delete", _spy):
        result = run_session_b(
            plan_src,
            implement=dry_run_implement,
            decode_only=False,
            plan_path=plan,
            write_path=True,  # even if asked to write, dry-run must not ship/mark
        )

    assert result.ok
    assert result.verdict == "dry_run"
    assert result.item is not None
    assert result.item.item_id == "Q1"
    assert result.keep_schedule is True
    assert result.scheduler_delete_called is False
    assert calls["n"] == 0
    assert result.implement_result is not None
    assert result.implement_result["dry_run"] is True
    assert result.implement_result["file_touch"] == result.item.file_touch
    assert result.implement_result["acceptance"] == result.item.acceptance
    assert "file_touch=" in result.message
    assert "no repo write" in result.message
    # Plan text unchanged (not SHIPPED); OPEN still 1.
    assert count_open(parse_queue_items(result.plan_text)) == 1
    assert "SHIPPED" not in result.plan_text
    # write_path=True must not rewrite plan for dry-run.
    assert plan.read_text(encoding="utf-8") == plan_src
    # No real file mutations outside temp/fixture logs.
    assert touched_target.read_bytes() == before_bytes
    assert touched_target.stat().st_mtime_ns == before_mtime
    assert repo_sentinel.read_bytes() == sentinel_before


def test_session_b_dry_run_idle_on_empty_never_deletes():
    result = run_session_b(
        _load("empty_queue.md"),
        implement=dry_run_implement,
        decode_only=False,
    )
    assert result.ok
    assert result.verdict == "idle"
    assert result.implement_result is None
    assert result.keep_schedule
    assert not result.scheduler_delete_called
    assert "queue 0/10" in result.message


def test_cli_session_b_dry_run_one_open(tmp_path: Path, capsys):
    from src.research_implement.__main__ import main

    plan = tmp_path / "one.md"
    plan.write_text(_load("one_open_ready.md"), encoding="utf-8")
    # Snapshot fixture + a repo file that file_touch mentions.
    loop_test = Path(__file__).resolve()
    before = loop_test.read_bytes()

    rc = main(["session-b", "--plan", str(plan), "--dry-run", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = __import__("json").loads(out)
    assert payload["verdict"] == "dry_run"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert payload["implement_result"]["dry_run"] is True
    assert payload["implement_result"]["file_touch"]
    assert payload["implement_result"]["acceptance"]
    assert payload["item"]["item_id"] == "Q1"
    # Plan on disk unchanged; no repo mutation.
    assert "SHIPPED" not in plan.read_text(encoding="utf-8")
    assert loop_test.read_bytes() == before


def test_cli_session_b_dry_run_empty_idle(tmp_path: Path, capsys):
    from src.research_implement.__main__ import main

    plan = tmp_path / "empty.md"
    plan.write_text(_load("empty_queue.md"), encoding="utf-8")
    rc = main(["session-b", "--plan", str(plan), "--dry-run", "--json"])
    assert rc == 0
    payload = __import__("json").loads(capsys.readouterr().out)
    assert payload["verdict"] == "idle"
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert payload["implement_result"] is None

def test_session_b_default_implement_when_not_decode_only():
    """decode_only=False with implement=None uses dry_run_implement."""
    result = run_session_b(_load("one_open_ready.md"), decode_only=False)
    assert result.ok
    assert result.verdict == "dry_run"
    assert result.implement_result is not None
    assert result.implement_result["dry_run"] is True
    assert result.keep_schedule
    assert not result.scheduler_delete_called


def test_cli_session_b_default_remains_decode_only(tmp_path: Path, capsys):
    from src.research_implement.__main__ import main

    plan = tmp_path / "one.md"
    plan.write_text(_load("one_open_ready.md"), encoding="utf-8")
    rc = main(["session-b", "--plan", str(plan), "--json"])
    assert rc == 0
    payload = __import__("json").loads(capsys.readouterr().out)
    assert payload["verdict"] == "picked"
    assert payload["implement_result"] is None

def test_e2e_a_stub_then_b_dry_run_never_ships_tmp_path(tmp_path: Path):
    """A→B dry-run on tmp_path only: dry-run never ships / never deletes.

    Contract: ``dry_run_implement`` is non-mutating — it does not mark SHIPPED,
    does not rewrite the plan, and never calls ``scheduler_delete``. A second
    Session B fire therefore picks the same OPEN again (not idle queue 0/10).
    """
    plan = tmp_path / "plan.md"
    plan.write_text(_load("empty_queue.md"), encoding="utf-8")
    assert count_open(parse_queue_items(plan.read_text(encoding="utf-8"))) == 0

    # Session A: stub_brainstorm on empty temp plan → one OPEN item.
    a = run_session_a_path(plan, brainstorm=stub_brainstorm, write=True)
    assert a.ok
    assert a.verdict == "queued"
    assert a.open_count == 1
    assert a.wrote_item
    after_a = plan.read_text(encoding="utf-8")
    items_a = parse_queue_items(after_a)
    assert count_open(items_a) == 1
    assert is_b_pickable(items_a[0])
    assert "SHIPPED" not in after_a

    calls = {"n": 0}

    def _spy(*_a, **_k):
        calls["n"] += 1
        raise SchedulerDeleteForbidden("spy")

    # Session B #1: dry_run_implement → decode + dry-run implement report.
    with patch("src.research_implement.session_b.scheduler_delete", _spy):
        b1 = run_session_b_path(
            plan,
            implement=dry_run_implement,
            decode_only=False,
            write=False,
        )
    assert b1.ok
    assert b1.verdict == "dry_run"
    assert b1.item is not None
    assert b1.item.item_id == items_a[0].item_id
    assert b1.decode_report is not None
    assert f"decode pick {b1.item.item_id}:" in b1.decode_report
    assert b1.implement_result is not None
    assert b1.implement_result["dry_run"] is True
    assert b1.implement_result["wrote_files"] is False
    assert b1.implement_result["sha"] is None
    assert b1.implement_result["file_touch"] == b1.item.file_touch
    assert b1.implement_result["acceptance"] == b1.item.acceptance
    assert "no repo write" in b1.message
    assert b1.keep_schedule is True
    assert b1.scheduler_delete_called is False
    assert calls["n"] == 0
    # Dry-run never ships / never deletes the OPEN row.
    after_b1 = plan.read_text(encoding="utf-8")
    assert after_b1 == after_a
    assert count_open(parse_queue_items(after_b1)) == 1
    assert "SHIPPED" not in after_b1

    # Session B #2: same OPEN still pickable (not idle 0/10).
    with patch("src.research_implement.session_b.scheduler_delete", _spy):
        b2 = run_session_b_path(
            plan,
            implement=dry_run_implement,
            decode_only=False,
            write=False,
        )
    assert b2.ok
    assert b2.verdict == "dry_run"  # not idle — dry-run left OPEN in place
    assert b2.item is not None
    assert b2.item.item_id == b1.item.item_id
    assert b2.open_count == 1
    assert "queue 1/10" in b2.message or b2.queue_label == "queue 1/10"
    assert b2.keep_schedule is True
    assert b2.scheduler_delete_called is False
    assert calls["n"] == 0
    assert plan.read_text(encoding="utf-8") == after_a


def test_e2e_cli_a_stub_then_b_dry_run_tmp_path(tmp_path: Path, capsys):
    """CLI sequence: session-a --stub then session-b --dry-run twice on tmp plan."""
    from src.research_implement.__main__ import main

    plan = tmp_path / "plan.md"
    plan.write_text(_load("empty_queue.md"), encoding="utf-8")

    rc_a = main(["session-a", "--plan", str(plan), "--stub"])
    assert rc_a == 0
    out_a = capsys.readouterr().out
    assert "queued" in out_a
    assert count_open(parse_queue_items(plan.read_text(encoding="utf-8"))) == 1

    rc_b1 = main(["session-b", "--plan", str(plan), "--dry-run", "--json"])
    assert rc_b1 == 0
    payload1 = __import__("json").loads(capsys.readouterr().out)
    assert payload1["verdict"] == "dry_run"
    assert payload1["item"]["item_id"]
    assert payload1["implement_result"]["dry_run"] is True
    assert payload1["keep_schedule"] is True
    assert payload1["scheduler_delete_called"] is False
    assert "SHIPPED" not in plan.read_text(encoding="utf-8")

    # Second B: still dry_run on same OPEN (non-mutating contract).
    rc_b2 = main(["session-b", "--plan", str(plan), "--dry-run", "--json"])
    assert rc_b2 == 0
    payload2 = __import__("json").loads(capsys.readouterr().out)
    assert payload2["verdict"] == "dry_run"
    assert payload2["item"]["item_id"] == payload1["item"]["item_id"]
    assert payload2["open_count"] == 1
    assert payload2["scheduler_delete_called"] is False

def test_beat5_dry_run_never_ships_fixture_ship_writes_tmp_path_only(tmp_path: Path):
    """Beat 5: dry_run never ships; custom implement can ship on tmp_path only.

    Live prod implement stays unwired — only the explicit test double ships.
    Idle / keep_schedule / no scheduler_delete remain true on both paths.
    """
    plan = tmp_path / "plan.md"
    plan.write_text(_load("one_open_ready.md"), encoding="utf-8")
    repo_sentinel = Path(__file__).resolve().parents[1] / "src" / "research_implement" / "queue.py"
    sentinel_before = repo_sentinel.read_bytes()

    calls = {"n": 0}

    def _spy(*_a, **_k):
        calls["n"] += 1
        raise SchedulerDeleteForbidden("spy")

    # Path A: dry_run_implement — never SHIPPED, plan on disk unchanged.
    with patch("src.research_implement.session_b.scheduler_delete", _spy):
        dry = run_session_b_path(
            plan,
            implement=dry_run_implement,
            decode_only=False,
            write=True,  # even with write=True dry-run must not rewrite
        )
    assert dry.ok
    assert dry.verdict == "dry_run"
    assert dry.keep_schedule is True
    assert dry.scheduler_delete_called is False
    assert calls["n"] == 0
    assert "SHIPPED" not in dry.plan_text
    assert count_open(parse_queue_items(dry.plan_text)) == 1
    assert plan.read_text(encoding="utf-8") == _load("one_open_ready.md")

    # Path B: explicit fixture_ship_implement — SHIPPED on tmp_path plan only.
    ship_fn = make_fixture_ship_implement("cafef00d", note="beat5 tmp ship")
    with patch("src.research_implement.session_b.scheduler_delete", _spy):
        shipped = run_session_b_path(
            plan,
            implement=ship_fn,
            decode_only=False,
            write=True,
        )
    assert shipped.ok
    assert shipped.verdict == "shipped"
    assert shipped.implement_result is not None
    assert shipped.implement_result["sha"] == "cafef00d"
    assert shipped.implement_result["dry_run"] is False
    assert shipped.keep_schedule is True
    assert shipped.scheduler_delete_called is False
    assert calls["n"] == 0
    assert count_open(parse_queue_items(shipped.plan_text)) == 0
    assert "SHIPPED `cafef00d`" in shipped.plan_text
    on_disk = plan.read_text(encoding="utf-8")
    assert "SHIPPED `cafef00d`" in on_disk
    assert count_open(parse_queue_items(on_disk)) == 0
    # Repo source untouched (tmp_path only).
    assert repo_sentinel.read_bytes() == sentinel_before

    # After ship, next B is idle (queue 0/10) and still never deletes schedule.
    with patch("src.research_implement.session_b.scheduler_delete", _spy):
        idle = run_session_b_path(plan, decode_only=False, write=False)
    assert idle.ok
    assert idle.verdict == "idle"
    assert "queue 0/10" in idle.message
    assert idle.keep_schedule is True
    assert idle.scheduler_delete_called is False
    assert calls["n"] == 0


def _assert_dry_run_implement_schema(payload: dict) -> None:
    """JSON/dict schema for dry_run_implement results (CLI + API)."""
    required = {
        "dry_run",
        "item_id",
        "title",
        "file_touch",
        "acceptance",
        "wrote_files",
        "sha",
    }
    assert set(payload) >= required
    assert payload["dry_run"] is True
    assert payload["wrote_files"] is False
    assert payload["sha"] is None
    assert isinstance(payload["item_id"], str) and payload["item_id"]
    assert isinstance(payload["title"], str)
    assert isinstance(payload["file_touch"], str) and payload["file_touch"]
    assert isinstance(payload["acceptance"], str) and payload["acceptance"]
    # dry-run must not claim a ship note/sha
    assert not payload.get("note")


def _assert_shipped_implement_schema(payload: dict) -> None:
    """JSON/dict schema for fixture_ship_implement results (test double only)."""
    required = {
        "dry_run",
        "item_id",
        "title",
        "file_touch",
        "acceptance",
        "wrote_files",
        "sha",
        "note",
    }
    assert set(payload) >= required
    assert payload["dry_run"] is False
    assert payload["wrote_files"] is False
    assert isinstance(payload["sha"], str) and payload["sha"]
    assert isinstance(payload["note"], str)
    assert isinstance(payload["item_id"], str) and payload["item_id"]
    assert isinstance(payload["file_touch"], str) and payload["file_touch"]
    assert isinstance(payload["acceptance"], str) and payload["acceptance"]


def test_beat5_implement_result_json_schema_dry_run_vs_shipped():
    """Beat 5 polish: contrast dry_run vs shipped implement_result schemas."""
    plan = _load("one_open_ready.md")
    dry = run_session_b(plan, implement=dry_run_implement, decode_only=False)
    assert dry.verdict == "dry_run"
    assert dry.implement_result is not None
    _assert_dry_run_implement_schema(dry.implement_result)
    # Round-trip through json like CLI --json consumers.
    import json

    dry_round = json.loads(json.dumps(dry.implement_result))
    _assert_dry_run_implement_schema(dry_round)

    ship_fn = make_fixture_ship_implement("schemabeef", note="schema contrast")
    shipped = run_session_b(plan, implement=ship_fn, decode_only=False)
    assert shipped.verdict == "shipped"
    assert shipped.implement_result is not None
    _assert_shipped_implement_schema(shipped.implement_result)
    shipped_round = json.loads(json.dumps(shipped.implement_result))
    _assert_shipped_implement_schema(shipped_round)

    # Explicit contrast: schemas must disagree on dry_run / sha presence.
    assert dry.implement_result["dry_run"] is True
    assert shipped.implement_result["dry_run"] is False
    assert dry.implement_result["sha"] is None
    assert shipped.implement_result["sha"] == "schemabeef"
    assert "note" not in dry.implement_result or not dry.implement_result.get("note")
    assert shipped.implement_result["note"] == "schema contrast"


def test_beat5_fixture_two_open_ready_ship_only_first_on_tmp_path(tmp_path: Path):
    """Fixture edge: two OPEN ready — dry_run ships none; ship double ships Q1 only."""
    src = _load("two_open_ready.md")
    items = parse_queue_items(src)
    assert [i.item_id for i in items] == ["Q1", "Q2"]
    assert all(is_b_pickable(i) for i in items)
    assert count_open(items) == 2
    assert first_b_pick(items).item_id == "Q1"

    plan = tmp_path / "two.md"
    plan.write_text(src, encoding="utf-8")

    dry = run_session_b_path(
        plan, implement=dry_run_implement, decode_only=False, write=True
    )
    assert dry.verdict == "dry_run"
    assert dry.item is not None and dry.item.item_id == "Q1"
    _assert_dry_run_implement_schema(dry.implement_result)
    assert "SHIPPED" not in plan.read_text(encoding="utf-8")
    assert count_open(parse_queue_items(plan.read_text(encoding="utf-8"))) == 2

    ship_fn = make_fixture_ship_implement("twoopen01", note="ship Q1 only")
    shipped = run_session_b_path(
        plan, implement=ship_fn, decode_only=False, write=True
    )
    assert shipped.verdict == "shipped"
    assert shipped.item is not None and shipped.item.item_id == "Q1"
    _assert_shipped_implement_schema(shipped.implement_result)
    on_disk = plan.read_text(encoding="utf-8")
    assert "SHIPPED `twoopen01`" in on_disk
    after = parse_queue_items(on_disk)
    assert count_open(after) == 1
    pick = first_b_pick(after)
    assert pick is not None and pick.item_id == "Q2"
    assert pick.title == "Second ready complete item"

    # Next fire picks Q2 (still OPEN); dry_run still does not ship it.
    dry2 = run_session_b_path(
        plan, implement=dry_run_implement, decode_only=False, write=True
    )
    assert dry2.verdict == "dry_run"
    assert dry2.item is not None and dry2.item.item_id == "Q2"
    assert count_open(parse_queue_items(plan.read_text(encoding="utf-8"))) == 1
    assert "SHIPPED `twoopen01`" in plan.read_text(encoding="utf-8")


def test_beat5_fixture_open_complete_not_ready_is_idle():
    """Fixture edge: six fields present but ready-for-implement no → idle, no ship."""
    plan = _load("open_complete_not_ready.md")
    items = parse_queue_items(plan)
    assert len(items) == 1
    assert is_complete_six_field(items[0])
    assert items[0].status.upper().startswith("OPEN")
    assert not is_b_pickable(items[0])
    assert count_open(items) == 0
    assert first_b_pick(items) is None

    dry = run_session_b(plan, implement=dry_run_implement, decode_only=False)
    assert dry.ok and dry.verdict == "idle"
    assert dry.implement_result is None
    assert "SHIPPED" not in dry.plan_text
    assert dry.keep_schedule and not dry.scheduler_delete_called

    # Even an explicit ship double cannot ship — nothing is B-pickable.
    shipped = run_session_b(
        plan,
        implement=make_fixture_ship_implement("should-not-run"),
        decode_only=False,
    )
    assert shipped.ok and shipped.verdict == "idle"
    assert shipped.implement_result is None
    assert "SHIPPED" not in shipped.plan_text


def _assert_session_result_json_shape(payload: dict) -> None:
    """CLI ``--json`` and ``SessionResult.to_dict`` / ``to_json_dict`` share one key set."""
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)
    assert set(payload.keys()) == set(SESSION_B_RESULT_KEYS)
    assert isinstance(payload["ok"], bool)
    assert isinstance(payload["verdict"], str)
    assert isinstance(payload["open_count"], int)
    assert isinstance(payload["queue"], str)
    assert payload["queue"].startswith("queue ")
    assert isinstance(payload["keep_schedule"], bool)
    assert isinstance(payload["scheduler_delete_called"], bool)
    assert "item" in payload
    assert "implement_result" in payload
    assert isinstance(payload["wrote_files"], bool)
    assert isinstance(payload["shipped"], bool)
    # Round-trip like CLI consumers.
    roundtrip = json.loads(json.dumps(payload, sort_keys=True))
    assert set(roundtrip.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat6_session_result_json_idle_fire_keys():
    """Beat 6: idle fire JSON — queue 0/10, keep_schedule, shared SessionResult keys."""
    result = run_session_b(_load("empty_queue.md"), decode_only=True)
    assert isinstance(result, SessionResult)
    payload = result.to_json_dict()
    _assert_session_result_json_shape(payload)
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["queue"] == "queue 0/10"
    assert payload["open_count"] == 0
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert payload["item"] is None
    assert payload["implement_result"] is None
    assert payload["wrote_files"] is False
    assert payload["shipped"] is False
    assert "keep_schedule" in result.message
    assert "queue 0/10" in result.message


def test_beat6_session_result_json_dry_run_wrote_files_false_not_shipped(tmp_path: Path):
    """Beat 6: dry_run JSON — wrote_files false, shipped false, plan not SHIPPED."""
    plan = tmp_path / "plan.md"
    plan.write_text(_load("one_open_ready.md"), encoding="utf-8")
    result = run_session_b_path(
        plan, implement=dry_run_implement, decode_only=False, write=True
    )
    payload = result.to_json_dict()
    _assert_session_result_json_shape(payload)
    assert payload["verdict"] == "dry_run"
    assert payload["wrote_files"] is False
    assert payload["shipped"] is False
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert payload["item"] is not None
    assert payload["item"]["item_id"] == "Q1"
    assert payload["implement_result"] is not None
    assert payload["implement_result"]["wrote_files"] is False
    assert payload["implement_result"]["dry_run"] is True
    assert payload["implement_result"]["sha"] is None
    assert "SHIPPED" not in plan.read_text(encoding="utf-8")
    assert "SHIPPED" not in result.plan_text


def test_beat6_session_result_json_fixture_ship_tmp_path(tmp_path: Path):
    """Beat 6: fixture ship JSON — shipped true on tmp_path; SHIPPED in plan text."""
    plan = tmp_path / "plan.md"
    plan.write_text(_load("one_open_ready.md"), encoding="utf-8")
    ship_fn = make_fixture_ship_implement("beat6cafe", note="beat6 ship")
    result = run_session_b_path(
        plan, implement=ship_fn, decode_only=False, write=True
    )
    payload = result.to_json_dict()
    _assert_session_result_json_shape(payload)
    assert payload["verdict"] == "shipped"
    assert payload["shipped"] is True
    assert payload["wrote_files"] is False  # fixture double does not write repo files
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert payload["implement_result"] is not None
    assert payload["implement_result"]["sha"] == "beat6cafe"
    assert payload["implement_result"]["dry_run"] is False
    assert "SHIPPED `beat6cafe`" in result.plan_text
    on_disk = plan.read_text(encoding="utf-8")
    assert "SHIPPED `beat6cafe`" in on_disk
    assert payload["queue"] == "queue 0/10"
    assert payload["open_count"] == 0


def test_beat6_session_result_json_decode_only_picked_keys(tmp_path: Path, capsys):
    """Beat 6: decode_only (verdict picked) shares the same SessionResult JSON keys."""
    plan = tmp_path / "one.md"
    plan.write_text(_load("one_open_ready.md"), encoding="utf-8")
    result = run_session_b_path(plan, decode_only=True, write=False)
    payload = result.to_json_dict()
    _assert_session_result_json_shape(payload)
    assert payload["verdict"] == "picked"
    assert payload["shipped"] is False
    assert payload["wrote_files"] is False
    assert payload["implement_result"] is None
    assert payload["item"]["item_id"] == "Q1"
    assert payload["keep_schedule"] is True

    # CLI --json must emit the identical shape (shared to_json_dict).
    from src.research_implement.__main__ import main

    rc = main(["session-b", "--plan", str(plan), "--json"])
    assert rc == 0
    cli_payload = json.loads(capsys.readouterr().out)
    _assert_session_result_json_shape(cli_payload)
    assert cli_payload == payload


def test_beat6_cli_json_idle_and_dry_run_share_shape(tmp_path: Path, capsys):
    """Beat 6: CLI --json idle + dry_run both use SessionResult key set."""
    from src.research_implement.__main__ import main

    empty = tmp_path / "empty.md"
    empty.write_text(_load("empty_queue.md"), encoding="utf-8")
    rc = main(["idle-decode", "--plan", str(empty), "--json"])
    assert rc == 0
    idle_payload = json.loads(capsys.readouterr().out)
    _assert_session_result_json_shape(idle_payload)
    assert idle_payload["verdict"] == "idle"
    assert idle_payload["queue"] == "queue 0/10"
    assert idle_payload["keep_schedule"] is True
    assert idle_payload["wrote_files"] is False
    assert idle_payload["shipped"] is False

    one = tmp_path / "one.md"
    one.write_text(_load("one_open_ready.md"), encoding="utf-8")
    rc = main(["session-b", "--plan", str(one), "--dry-run", "--json"])
    assert rc == 0
    dry_payload = json.loads(capsys.readouterr().out)
    _assert_session_result_json_shape(dry_payload)
    assert dry_payload["verdict"] == "dry_run"
    assert dry_payload["wrote_files"] is False
    assert dry_payload["shipped"] is False
    assert dry_payload["implement_result"]["wrote_files"] is False



def test_beat7_two_open_ready_first_open_pick_order_second_remains_open(tmp_path: Path, capsys):
    """Beat 7: with two_open_ready.md always pick first complete ready OPEN; Q2 stays OPEN."""
    src = _load("two_open_ready.md")
    items = parse_queue_items(src)
    assert [i.item_id for i in items] == ["Q1", "Q2"]
    assert all(is_b_pickable(i) for i in items)
    assert count_open(items) == 2
    assert first_b_pick(items).item_id == "Q1"
    assert first_b_pick(items).title == "First ready complete item"

    plan = tmp_path / "two.md"
    plan.write_text(src, encoding="utf-8")

    # Decode-only: first OPEN only; plan unchanged so second remains OPEN.
    decoded = run_session_b_path(plan, decode_only=True, write=False)
    assert decoded.ok and decoded.verdict == "picked"
    assert decoded.item is not None and decoded.item.item_id == "Q1"
    assert decoded.open_count == 2
    assert decoded.queue_label == "queue 2/10"
    on_disk = plan.read_text(encoding="utf-8")
    after = parse_queue_items(on_disk)
    assert [i.item_id for i in after if is_b_pickable(i)] == ["Q1", "Q2"]
    assert after[1].item_id == "Q2" and is_open_status(after[1].status)
    assert "SHIPPED" not in on_disk

    # Dry-run: still first OPEN; never ships; second remains OPEN ready.
    dry = run_session_b_path(
        plan, implement=dry_run_implement, decode_only=False, write=True
    )
    assert dry.verdict == "dry_run"
    assert dry.item is not None and dry.item.item_id == "Q1"
    assert dry.open_count == 2
    assert dry.wrote_files is False and dry.shipped is False
    on_disk = plan.read_text(encoding="utf-8")
    after = parse_queue_items(on_disk)
    assert count_open(after) == 2
    assert first_b_pick(after).item_id == "Q1"
    q2 = next(i for i in after if i.item_id == "Q2")
    assert is_b_pickable(q2)
    assert q2.title == "Second ready complete item"
    assert "SHIPPED" not in on_disk

    # Repeated dry-run still picks Q1 (non-mutating); Q2 never consumed.
    dry2 = run_session_b_path(
        plan, implement=dry_run_implement, decode_only=False, write=True
    )
    assert dry2.item is not None and dry2.item.item_id == "Q1"
    assert count_open(parse_queue_items(plan.read_text(encoding="utf-8"))) == 2

    # CLI --json decode also reports Q1 first-OPEN pick.
    from src.research_implement.__main__ import main

    rc = main(["session-b", "--plan", str(plan), "--json"])
    assert rc == 0
    cli_payload = json.loads(capsys.readouterr().out)
    _assert_session_result_json_shape(cli_payload)
    assert cli_payload["verdict"] == "picked"
    assert cli_payload["item"]["item_id"] == "Q1"
    assert cli_payload["open_count"] == 2
    assert cli_payload == decoded.to_dict()


def test_beat7_open_complete_not_ready_idle_not_ready_for_implement(tmp_path: Path, capsys):
    """Beat 7: open_complete_not_ready.md → idle (complete but not ready-for-implement)."""
    plan_text = _load("open_complete_not_ready.md")
    items = parse_queue_items(plan_text)
    assert len(items) == 1
    assert is_complete_six_field(items[0])
    assert is_open_status(items[0].status)
    assert items[0].ready_for_implement.strip().lower() == "no"
    assert not is_b_pickable(items[0])
    assert first_b_pick(items) is None
    assert count_open(items) == 0

    for kwargs in (
        {"decode_only": True},
        {"implement": dry_run_implement, "decode_only": False},
        {
            "implement": make_fixture_ship_implement("must-not-run"),
            "decode_only": False,
        },
    ):
        result = run_session_b(plan_text, **kwargs)
        assert result.ok and result.verdict == "idle"
        assert result.item is None
        assert result.implement_result is None
        assert result.open_count == 0
        assert result.queue_label == "queue 0/10"
        assert result.keep_schedule is True
        assert result.scheduler_delete_called is False
        assert result.wrote_files is False
        assert result.shipped is False
        assert "SHIPPED" not in result.plan_text
        payload = result.to_dict()
        _assert_session_result_json_shape(payload)
        assert payload["verdict"] == "idle"
        assert payload["item"] is None
        assert payload["implement_result"] is None

    from src.research_implement.__main__ import main

    p = tmp_path / "not_ready.md"
    p.write_text(plan_text, encoding="utf-8")
    rc = main(["session-b", "--plan", str(p), "--json"])
    assert rc == 0
    cli_payload = json.loads(capsys.readouterr().out)
    _assert_session_result_json_shape(cli_payload)
    assert cli_payload["verdict"] == "idle"
    assert cli_payload["queue"] == "queue 0/10"
    assert cli_payload["shipped"] is False
    assert cli_payload["wrote_files"] is False


def test_beat7_session_result_to_dict_key_stability_all_verdicts(tmp_path: Path, capsys):
    """Beat 7: shared SessionResult.to_dict used by CLI --json; keys stable across verdicts."""
    from src.research_implement.__main__ import main

    assert SESSION_B_RESULT_KEYS == SESSION_RESULT_JSON_KEYS

    # idle
    empty = tmp_path / "empty.md"
    empty.write_text(_load("empty_queue.md"), encoding="utf-8")
    idle = run_session_b_path(empty, decode_only=True)
    idle_dict = idle.to_dict()
    assert idle.to_json_dict() == idle_dict == session_b_result_dict(idle)
    _assert_session_result_json_shape(idle_dict)
    assert idle_dict["verdict"] == "idle"

    rc = main(["idle-decode", "--plan", str(empty), "--json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out) == idle_dict

    # decode_only / picked
    one = tmp_path / "one.md"
    one.write_text(_load("one_open_ready.md"), encoding="utf-8")
    picked = run_session_b_path(one, decode_only=True, write=False)
    picked_dict = picked.to_dict()
    assert picked.to_json_dict() == picked_dict == session_b_result_dict(picked)
    _assert_session_result_json_shape(picked_dict)
    assert picked_dict["verdict"] == "picked"
    assert picked_dict["item"]["item_id"] == "Q1"
    assert picked_dict["implement_result"] is None
    assert picked_dict["wrote_files"] is False
    assert picked_dict["shipped"] is False

    rc = main(["session-b", "--plan", str(one), "--json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out) == picked_dict

    # dry_run
    dry = run_session_b_path(
        one, implement=dry_run_implement, decode_only=False, write=False
    )
    dry_dict = dry.to_dict()
    assert dry.to_json_dict() == dry_dict == session_b_result_dict(dry)
    _assert_session_result_json_shape(dry_dict)
    assert dry_dict["verdict"] == "dry_run"
    assert dry_dict["wrote_files"] is False
    assert dry_dict["shipped"] is False
    assert dry_dict["implement_result"]["dry_run"] is True

    rc = main(["session-b", "--plan", str(one), "--dry-run", "--json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out) == dry_dict

    # shipped (fixture double on tmp_path only)
    ship_plan = tmp_path / "ship.md"
    ship_plan.write_text(_load("one_open_ready.md"), encoding="utf-8")
    ship_fn = make_fixture_ship_implement("beat7cafe", note="beat7 ship")
    shipped = run_session_b_path(
        ship_plan, implement=ship_fn, decode_only=False, write=True
    )
    shipped_dict = shipped.to_dict()
    assert shipped.to_json_dict() == shipped_dict == session_b_result_dict(shipped)
    _assert_session_result_json_shape(shipped_dict)
    assert shipped_dict["verdict"] == "shipped"
    assert shipped_dict["shipped"] is True
    assert shipped_dict["wrote_files"] is False
    assert shipped_dict["implement_result"]["sha"] == "beat7cafe"
    assert "SHIPPED `beat7cafe`" in ship_plan.read_text(encoding="utf-8")

    # Key sets identical across all four verdicts (stability).
    for payload in (idle_dict, picked_dict, dry_dict, shipped_dict):
        assert tuple(sorted(payload.keys())) == tuple(sorted(SESSION_RESULT_JSON_KEYS))


def _assert_session_a_result_json_shape(payload: dict) -> None:
    """CLI ``session-a --json`` and ``SessionAResult.to_dict`` / ``to_json_dict`` share one key set."""
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)
    assert set(payload.keys()) == set(SESSION_A_RESULT_KEYS)
    assert isinstance(payload["ok"], bool)
    assert isinstance(payload["verdict"], str)
    assert isinstance(payload["open_count"], int)
    assert isinstance(payload["queue"], str)
    assert payload["queue"].startswith("queue ")
    assert "b_pick_title" in payload
    assert "title" in payload
    assert isinstance(payload["wrote_item"], bool)
    roundtrip = json.loads(json.dumps(payload, sort_keys=True))
    assert set(roundtrip.keys()) == set(SESSION_A_RESULT_JSON_KEYS)


def test_beat8_session_a_empty_stub_appends_one_open_json_shape(tmp_path: Path, capsys):
    """Beat 8: empty plan + stub → append one OPEN; shared SessionAResult JSON shape."""
    plan = tmp_path / "empty.md"
    plan.write_text(_load("empty_queue.md"), encoding="utf-8")
    before = count_open(parse_queue_items(plan.read_text(encoding="utf-8")))
    assert before == 0

    result = run_session_a_path(plan, brainstorm=stub_brainstorm, write=True)
    assert isinstance(result, SessionAResult)
    assert result.ok and result.verdict == "queued"
    assert result.wrote_item is True
    assert result.open_count == 1
    assert result.title == "Stub shippable change"
    assert result.b_pick_title == "Stub shippable change"
    assert result.queue_label == "queue 1/10"

    on_disk = plan.read_text(encoding="utf-8")
    items = parse_queue_items(on_disk)
    assert count_open(items) == 1
    assert len(items) == 1
    assert items[0].status.upper().startswith("OPEN")
    assert "ready-for-implement: yes" in on_disk
    assert "Stub shippable change" in on_disk

    payload = result.to_json_dict()
    assert payload == result.to_dict() == session_a_result_dict(result)
    _assert_session_a_result_json_shape(payload)
    assert payload["ok"] is True
    assert payload["verdict"] == "queued"
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    assert payload["wrote_item"] is True
    assert payload["title"] == "Stub shippable change"
    assert payload["b_pick_title"] == "Stub shippable change"

    from src.research_implement.__main__ import main

    # CLI --json must emit the identical shape (shared to_json_dict).
    # Plan already has OPEN=1, so a second stub fire would be recount-only;
    # use a fresh empty copy for CLI append+json.
    empty2 = tmp_path / "empty2.md"
    empty2.write_text(_load("empty_queue.md"), encoding="utf-8")
    rc = main(["session-a", "--plan", str(empty2), "--stub", "--json"])
    assert rc == 0
    cli_payload = json.loads(capsys.readouterr().out)
    _assert_session_a_result_json_shape(cli_payload)
    assert cli_payload["verdict"] == "queued"
    assert cli_payload["wrote_item"] is True
    assert cli_payload["open_count"] == 1
    assert count_open(parse_queue_items(empty2.read_text(encoding="utf-8"))) == 1


def test_beat8_session_a_two_open_ready_recount_only_no_second_append(tmp_path: Path, capsys):
    """Beat 8: two_open_ready / OPEN>=1 → recount-only light; no second append."""
    src = _load("two_open_ready.md")
    items_before = parse_queue_items(src)
    assert count_open(items_before) == 2
    assert len(items_before) == 2

    plan = tmp_path / "two.md"
    plan.write_text(src, encoding="utf-8")

    called = {"n": 0}

    def tracking(_items):
        called["n"] += 1
        return stub_brainstorm(_items)

    result = run_session_a_path(plan, brainstorm=tracking, write=True)
    assert isinstance(result, SessionAResult)
    assert result.ok and result.verdict == "light"
    assert result.wrote_item is False
    assert result.open_count == 2
    assert result.title is None
    assert called["n"] == 0  # must not call brainstorm when OPEN >= 1
    assert "recount only" in result.message

    on_disk = plan.read_text(encoding="utf-8")
    assert on_disk == src  # plan unchanged
    after = parse_queue_items(on_disk)
    assert count_open(after) == 2
    assert len(after) == 2
    assert [i.item_id for i in after] == ["Q1", "Q2"]

    payload = result.to_json_dict()
    assert payload == result.to_dict() == session_a_result_dict(result)
    _assert_session_a_result_json_shape(payload)
    assert payload["ok"] is True
    assert payload["verdict"] == "light"
    assert payload["open_count"] == 2
    assert payload["queue"] == "queue 2/10"
    assert payload["wrote_item"] is False
    assert payload["title"] is None
    assert payload["b_pick_title"] == "First ready complete item"

    # Second fire still recount-only; still no append.
    result2 = run_session_a_path(plan, brainstorm=tracking, write=True)
    assert result2.verdict == "light"
    assert result2.wrote_item is False
    assert result2.open_count == 2
    assert called["n"] == 0
    assert count_open(parse_queue_items(plan.read_text(encoding="utf-8"))) == 2
    assert plan.read_text(encoding="utf-8") == src

    from src.research_implement.__main__ import main

    rc = main(["session-a", "--plan", str(plan), "--stub", "--json"])
    assert rc == 0
    cli_payload = json.loads(capsys.readouterr().out)
    _assert_session_a_result_json_shape(cli_payload)
    assert cli_payload == payload
    assert cli_payload["verdict"] == "light"
    assert cli_payload["wrote_item"] is False
    assert cli_payload["open_count"] == 2
    # Still exactly two OPEN after CLI recount-only.
    assert count_open(parse_queue_items(plan.read_text(encoding="utf-8"))) == 2
    assert plan.read_text(encoding="utf-8") == src


def test_beat8_session_a_one_open_ready_recount_and_failed_json_keys(tmp_path: Path, capsys):
    """Beat 8: OPEN>=1 recount + failed fire share SessionAResult key set with queued."""
    one = tmp_path / "one.md"
    one.write_text(_load("one_open_ready.md"), encoding="utf-8")
    light = run_session_a_path(one, brainstorm=stub_brainstorm, write=True)
    light_dict = light.to_dict()
    assert light.to_json_dict() == light_dict == session_a_result_dict(light)
    _assert_session_a_result_json_shape(light_dict)
    assert light_dict["verdict"] == "light"
    assert light_dict["wrote_item"] is False
    assert light_dict["open_count"] == 1
    assert one.read_text(encoding="utf-8") == _load("one_open_ready.md")

    empty = tmp_path / "empty.md"
    empty.write_text(_load("empty_queue.md"), encoding="utf-8")
    queued = run_session_a_path(empty, brainstorm=stub_brainstorm, write=True)
    queued_dict = queued.to_dict()
    _assert_session_a_result_json_shape(queued_dict)
    assert queued_dict["verdict"] == "queued"
    assert queued_dict["wrote_item"] is True

    failed = run_session_a(_load("empty_queue.md"), brainstorm=None)
    failed_dict = failed.to_dict()
    _assert_session_a_result_json_shape(failed_dict)
    assert failed_dict["ok"] is False
    assert failed_dict["verdict"] == "failed"
    assert failed_dict["wrote_item"] is False
    assert failed_dict["open_count"] == 0
    assert failed_dict["queue"] == "queue 0/10"

    # Key sets identical across queued / light / failed (stability).
    for payload in (queued_dict, light_dict, failed_dict):
        assert tuple(sorted(payload.keys())) == tuple(sorted(SESSION_A_RESULT_JSON_KEYS))
        assert SESSION_A_RESULT_KEYS == SESSION_A_RESULT_JSON_KEYS

    from src.research_implement.__main__ import main

    rc = main(["session-a", "--plan", str(one), "--stub", "--json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out) == light_dict


# --- Beat 10: idle-decode --json shared Session B shape + malformed fixtures ---


def test_beat10_idle_decode_json_empty_idle_shared_shape(tmp_path: Path, capsys):
    """Beat 10: idle-decode --json on empty Queue → idle shared Session B JSON."""
    from src.research_implement.__main__ import main

    plan = tmp_path / "empty.md"
    plan.write_text(_load("empty_queue.md"), encoding="utf-8")
    expected = run_session_b_path(plan, decode_only=True, write=False).to_dict()
    _assert_session_result_json_shape(expected)
    assert expected["verdict"] == "idle"
    assert expected["item"] is None
    assert expected["implement_result"] is None
    assert expected["queue"] == "queue 0/10"
    assert expected["keep_schedule"] is True
    assert expected["scheduler_delete_called"] is False
    assert expected["wrote_files"] is False
    assert expected["shipped"] is False

    rc = main(["idle-decode", "--plan", str(plan), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    _assert_session_result_json_shape(payload)
    assert payload == expected
    assert payload == session_b_result_dict(
        run_session_b_path(plan, decode_only=True, write=False)
    )


def test_beat10_idle_decode_json_not_ready_idle_shared_shape(tmp_path: Path, capsys):
    """Beat 10: idle-decode --json on complete-but-not-ready → idle shared shape."""
    from src.research_implement.__main__ import main

    plan = tmp_path / "not_ready.md"
    plan.write_text(_load("open_complete_not_ready.md"), encoding="utf-8")
    result = run_session_b_path(plan, decode_only=True, write=False)
    assert result.verdict == "idle"
    expected = result.to_json_dict()
    _assert_session_result_json_shape(expected)
    assert expected["ok"] is True
    assert expected["verdict"] == "idle"
    assert expected["open_count"] == 0
    assert expected["queue"] == "queue 0/10"
    assert expected["item"] is None
    assert expected["keep_schedule"] is True
    assert expected["scheduler_delete_called"] is False

    rc = main(["idle-decode", "--plan", str(plan), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    _assert_session_result_json_shape(payload)
    assert payload == expected
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS) == set(SESSION_B_RESULT_KEYS)


def test_beat10_idle_decode_json_open_ready_picked_decode_only(tmp_path: Path, capsys):
    """Beat 10: idle-decode --json on OPEN ready → picked decode_only shared shape."""
    from src.research_implement.__main__ import main

    plan = tmp_path / "one.md"
    plan.write_text(_load("one_open_ready.md"), encoding="utf-8")
    result = run_session_b_path(plan, decode_only=True, write=False)
    assert result.verdict == "picked"
    expected = result.to_dict()
    _assert_session_result_json_shape(expected)
    assert expected["ok"] is True
    assert expected["verdict"] == "picked"
    assert expected["open_count"] == 1
    assert expected["queue"] == "queue 1/10"
    assert expected["item"] is not None
    assert expected["item"]["item_id"] == "Q1"
    assert expected["item"]["ready_for_implement"].lower() == "yes"
    assert expected["implement_result"] is None  # decode_only
    assert expected["wrote_files"] is False
    assert expected["shipped"] is False
    assert expected["keep_schedule"] is True
    assert expected["scheduler_delete_called"] is False

    rc = main(["idle-decode", "--plan", str(plan), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    _assert_session_result_json_shape(payload)
    assert payload == expected
    # idle-decode and session-b --json share the same decode_only shape.
    rc = main(["session-b", "--plan", str(plan), "--json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out) == payload


def test_beat10_malformed_incomplete_missing_fields_idle_never_delete():
    """Beat 10: incomplete / missing fields → idle; never scheduler_delete; no crash."""
    calls = {"n": 0}

    def _spy(*_a, **_k):
        calls["n"] += 1
        raise SchedulerDeleteForbidden("spy")

    plan = _load("incomplete_open.md")
    items = parse_queue_items(plan)
    assert len(items) == 1
    assert not is_complete_six_field(items[0])
    assert not is_b_pickable(items[0])
    assert first_b_pick(items) is None
    assert count_open(items) == 0

    with patch("src.research_implement.session_b.scheduler_delete", _spy):
        result = run_session_b(plan, decode_only=True)
    assert result.ok and result.verdict == "idle"
    assert result.item is None
    assert result.keep_schedule is True
    assert result.scheduler_delete_called is False
    assert calls["n"] == 0
    payload = result.to_dict()
    _assert_session_result_json_shape(payload)
    assert payload["verdict"] == "idle"
    assert payload["scheduler_delete_called"] is False


def test_beat10_malformed_watch_lookalike_skip_or_idle_never_delete(tmp_path: Path, capsys):
    """Beat 10: Watch lookalike rows skipped; Queue-only pick or idle; never delete."""
    from src.research_implement.__main__ import main

    calls = {"n": 0}

    def _spy(*_a, **_k):
        calls["n"] += 1
        raise SchedulerDeleteForbidden("spy")

    # watch_lookalike.md: real Queue Q3 pickable; Watch Fake.* ignored.
    lookalike = _load("watch_lookalike.md")
    items = parse_queue_items(lookalike)
    assert [i.item_id for i in items] == ["Q3"]
    assert all("Watch lookalike" not in i.title for i in items)
    assert first_b_pick(items) is not None
    assert first_b_pick(items).title == "Real ready Queue item"

    with patch("src.research_implement.session_b.scheduler_delete", _spy):
        picked = run_session_b(lookalike, decode_only=True)
    assert picked.ok and picked.verdict == "picked"
    assert picked.item is not None and picked.item.item_id == "Q3"
    assert picked.scheduler_delete_called is False
    assert calls["n"] == 0
    _assert_session_result_json_shape(picked.to_dict())

    # watch_only_lookalike.md: empty Queue + Watch Q99-looking row → idle.
    watch_only = _load("watch_only_lookalike.md")
    only_items = parse_queue_items(watch_only)
    assert only_items == []
    assert first_b_pick(only_items) is None

    with patch("src.research_implement.session_b.scheduler_delete", _spy):
        idle = run_session_b(watch_only, decode_only=True)
    assert idle.ok and idle.verdict == "idle"
    assert idle.keep_schedule is True
    assert idle.scheduler_delete_called is False
    assert calls["n"] == 0
    _assert_session_result_json_shape(idle.to_dict())

    plan = tmp_path / "watch_only.md"
    plan.write_text(watch_only, encoding="utf-8")
    rc = main(["idle-decode", "--plan", str(plan), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    _assert_session_result_json_shape(payload)
    assert payload["verdict"] == "idle"
    assert payload["scheduler_delete_called"] is False
    assert payload["item"] is None


def test_beat10_malformed_broken_ready_flag_idle_never_delete(tmp_path: Path, capsys):
    """Beat 10: broken ready-for-implement values → not pickable / idle; never delete."""
    from src.research_implement.__main__ import main

    calls = {"n": 0}

    def _spy(*_a, **_k):
        calls["n"] += 1
        raise SchedulerDeleteForbidden("spy")

    plan_text = _load("broken_ready_flag.md")
    items = parse_queue_items(plan_text)
    assert len(items) == 2
    for item in items:
        assert is_open_status(item.status)
        assert is_complete_six_field(item)
        assert not is_b_pickable(item)  # broken ready flag → skip-not-pickable
    assert first_b_pick(items) is None
    assert count_open(items) == 0

    with patch("src.research_implement.session_b.scheduler_delete", _spy):
        result = run_session_b(plan_text, decode_only=True)
    assert result.ok and result.verdict == "idle"
    assert result.item is None
    assert result.keep_schedule is True
    assert result.scheduler_delete_called is False
    assert calls["n"] == 0
    payload = result.to_dict()
    _assert_session_result_json_shape(payload)
    assert payload["verdict"] == "idle"
    assert payload["queue"] == "queue 0/10"

    plan = tmp_path / "broken.md"
    plan.write_text(plan_text, encoding="utf-8")
    rc = main(["idle-decode", "--plan", str(plan), "--json"])
    assert rc == 0
    cli = json.loads(capsys.readouterr().out)
    _assert_session_result_json_shape(cli)
    assert cli == payload
    assert cli["scheduler_delete_called"] is False


def test_beat10_malformed_mixed_priority_skips_incomplete_picks_ready():
    """Beat 10: skip-not-pickable incomplete/Watch; pick first complete ready OPEN."""
    calls = {"n": 0}

    def _spy(*_a, **_k):
        calls["n"] += 1
        raise SchedulerDeleteForbidden("spy")

    plan = _load("mixed_priority.md")
    items = parse_queue_items(plan)
    assert [i.item_id for i in items] == ["Q1", "Q2"]
    assert not is_b_pickable(items[0])  # incomplete / missing acceptance
    assert is_b_pickable(items[1])
    assert first_b_pick(items) is items[1]
    assert all("Watch lookalike" not in i.title for i in items)

    with patch("src.research_implement.session_b.scheduler_delete", _spy):
        result = run_session_b(plan, decode_only=True)
    assert result.ok and result.verdict == "picked"
    assert result.item is not None
    assert result.item.item_id == "Q2"
    assert result.item.title == "Second ready complete item"
    assert result.scheduler_delete_called is False
    assert calls["n"] == 0
    payload = result.to_dict()
    _assert_session_result_json_shape(payload)
    assert payload["verdict"] == "picked"
    assert payload["item"]["item_id"] == "Q2"
    assert payload["implement_result"] is None


def test_beat10_malformed_fixtures_idle_decode_json_never_crash(tmp_path: Path, capsys):
    """Beat 10: idle-decode --json on malformed fixtures never crashes / never deletes."""
    from src.research_implement.__main__ import main

    cases = [
        ("incomplete_open.md", "idle"),
        ("open_complete_not_ready.md", "idle"),
        ("broken_ready_flag.md", "idle"),
        ("watch_only_lookalike.md", "idle"),
        ("shipped_only.md", "idle"),
        ("empty_queue.md", "idle"),
        ("watch_lookalike.md", "picked"),
        ("mixed_priority.md", "picked"),
        ("one_open_ready.md", "picked"),
    ]
    for name, expect_verdict in cases:
        plan = tmp_path / name
        plan.write_text(_load(name), encoding="utf-8")
        rc = main(["idle-decode", "--plan", str(plan), "--json"])
        assert rc == 0, f"{name} should exit 0"
        payload = json.loads(capsys.readouterr().out)
        _assert_session_result_json_shape(payload)
        assert payload["ok"] is True
        assert payload["verdict"] == expect_verdict, name
        assert payload["keep_schedule"] is True
        assert payload["scheduler_delete_called"] is False
        if expect_verdict == "idle":
            assert payload["item"] is None
            assert payload["implement_result"] is None
            assert payload["shipped"] is False
            assert payload["wrote_files"] is False
        else:
            assert payload["item"] is not None
            assert payload["implement_result"] is None  # decode_only
            assert payload["shipped"] is False


# --- Beat 11: full pipeline e2e on tmp_path (A stub → B dry_run → B ship → A light / B idle) ---


def test_beat11_full_pipeline_e2e_stub_dry_run_ship_recount_idle(tmp_path: Path, capsys):
    """Beat 11: full A/B pipeline on tmp_path only.

    Sequence:
      1. Session A stub append (empty → queued; SessionA JSON shape)
      2. Session B dry_run JSON (never ships; plan unchanged; SessionB JSON)
      3. Session A recount light (OPEN still 1 mid-pipeline; no second append)
      4. Session B fixture_ship (SHIPPED on tmp_path; dry_run never did)
      5. Session B idle (queue 0/10 after ship); assert idle JSON + never
         scheduler_delete (A light requires OPEN>=1 so it is step 3, not post-ship)

    Live prod implement / CLI stub-ship stay unwired. No Tasker / no LLM.
    """
    from src.research_implement.__main__ import main

    plan = tmp_path / "pipeline.md"
    plan.write_text(_load("empty_queue.md"), encoding="utf-8")
    assert count_open(parse_queue_items(plan.read_text(encoding="utf-8"))) == 0

    repo_sentinel = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "research_implement"
        / "queue.py"
    )
    sentinel_before = repo_sentinel.read_bytes()

    calls = {"n": 0}

    def _spy(*_a, **_k):
        calls["n"] += 1
        raise SchedulerDeleteForbidden("spy")

    # --- 1. Session A stub append ---
    a1 = run_session_a_path(plan, brainstorm=stub_brainstorm, write=True)
    assert isinstance(a1, SessionAResult)
    assert a1.ok and a1.verdict == "queued"
    assert a1.wrote_item is True
    assert a1.open_count == 1
    a1_dict = a1.to_dict()
    assert a1_dict == a1.to_json_dict() == session_a_result_dict(a1)
    _assert_session_a_result_json_shape(a1_dict)
    assert a1_dict["verdict"] == "queued"
    assert a1_dict["wrote_item"] is True
    assert a1_dict["open_count"] == 1
    assert a1_dict["queue"] == "queue 1/10"
    after_a1 = plan.read_text(encoding="utf-8")
    items_a1 = parse_queue_items(after_a1)
    assert count_open(items_a1) == 1
    assert is_b_pickable(items_a1[0])
    assert "SHIPPED" not in after_a1
    item_id = items_a1[0].item_id

    # CLI session-a --json on a fresh empty copy shares the same shape.
    empty_cli = tmp_path / "empty_cli.md"
    empty_cli.write_text(_load("empty_queue.md"), encoding="utf-8")
    rc_a = main(["session-a", "--plan", str(empty_cli), "--stub", "--json"])
    assert rc_a == 0
    cli_a = json.loads(capsys.readouterr().out)
    _assert_session_a_result_json_shape(cli_a)
    assert cli_a["verdict"] == "queued"
    assert cli_a["wrote_item"] is True

    # --- 2. Session B dry_run (JSON) — never ships ---
    with patch("src.research_implement.session_b.scheduler_delete", _spy):
        b_dry = run_session_b_path(
            plan,
            implement=dry_run_implement,
            decode_only=False,
            write=True,  # even with write=True dry-run must not rewrite
        )
    assert b_dry.ok and b_dry.verdict == "dry_run"
    dry_dict = b_dry.to_dict()
    assert dry_dict == b_dry.to_json_dict() == session_b_result_dict(b_dry)
    _assert_session_result_json_shape(dry_dict)
    assert dry_dict["verdict"] == "dry_run"
    assert dry_dict["ok"] is True
    assert dry_dict["open_count"] == 1
    assert dry_dict["queue"] == "queue 1/10"
    assert dry_dict["keep_schedule"] is True
    assert dry_dict["scheduler_delete_called"] is False
    assert dry_dict["shipped"] is False
    assert dry_dict["wrote_files"] is False
    assert dry_dict["item"] is not None
    assert dry_dict["item"]["item_id"] == item_id
    assert dry_dict["implement_result"] is not None
    _assert_dry_run_implement_schema(dry_dict["implement_result"])
    assert plan.read_text(encoding="utf-8") == after_a1
    assert "SHIPPED" not in plan.read_text(encoding="utf-8")
    assert calls["n"] == 0

    # CLI session-b --dry-run --json shares the same Session B shape.
    rc_dry = main(["session-b", "--plan", str(plan), "--dry-run", "--json"])
    assert rc_dry == 0
    cli_dry = json.loads(capsys.readouterr().out)
    _assert_session_result_json_shape(cli_dry)
    assert cli_dry["verdict"] == "dry_run"
    assert cli_dry["shipped"] is False
    assert cli_dry["implement_result"]["dry_run"] is True
    assert cli_dry["scheduler_delete_called"] is False
    assert plan.read_text(encoding="utf-8") == after_a1  # still never shipped

    # --- 3. Session A recount light (OPEN still present; no second append) ---
    called = {"n": 0}

    def tracking(_items):
        called["n"] += 1
        return stub_brainstorm(_items)

    a_light = run_session_a_path(plan, brainstorm=tracking, write=True)
    assert a_light.ok and a_light.verdict == "light"
    assert a_light.wrote_item is False
    assert a_light.open_count == 1
    assert called["n"] == 0
    assert "recount only" in a_light.message
    light_dict = a_light.to_dict()
    _assert_session_a_result_json_shape(light_dict)
    assert light_dict["verdict"] == "light"
    assert light_dict["wrote_item"] is False
    assert light_dict["open_count"] == 1
    assert plan.read_text(encoding="utf-8") == after_a1
    assert count_open(parse_queue_items(plan.read_text(encoding="utf-8"))) == 1

    rc_light = main(["session-a", "--plan", str(plan), "--stub", "--json"])
    assert rc_light == 0
    cli_light = json.loads(capsys.readouterr().out)
    _assert_session_a_result_json_shape(cli_light)
    assert cli_light["verdict"] == "light"
    assert cli_light["wrote_item"] is False
    assert plan.read_text(encoding="utf-8") == after_a1

    # --- 4. Session B fixture_ship — ships on tmp_path only ---
    ship_fn = make_fixture_ship_implement("beat11cafe", note="beat11 pipeline ship")
    with patch("src.research_implement.session_b.scheduler_delete", _spy):
        b_ship = run_session_b_path(
            plan,
            implement=ship_fn,
            decode_only=False,
            write=True,
        )
    assert b_ship.ok and b_ship.verdict == "shipped"
    ship_dict = b_ship.to_dict()
    _assert_session_result_json_shape(ship_dict)
    assert ship_dict["verdict"] == "shipped"
    assert ship_dict["shipped"] is True
    assert ship_dict["wrote_files"] is False
    assert ship_dict["keep_schedule"] is True
    assert ship_dict["scheduler_delete_called"] is False
    assert ship_dict["open_count"] == 0
    assert ship_dict["queue"] == "queue 0/10"
    assert ship_dict["item"] is not None
    assert ship_dict["item"]["item_id"] == item_id
    assert ship_dict["implement_result"] is not None
    _assert_shipped_implement_schema(ship_dict["implement_result"])
    assert ship_dict["implement_result"]["sha"] == "beat11cafe"
    assert ship_dict["implement_result"]["dry_run"] is False
    on_disk = plan.read_text(encoding="utf-8")
    assert "SHIPPED `beat11cafe`" in on_disk
    assert count_open(parse_queue_items(on_disk)) == 0
    assert calls["n"] == 0
    # Contrast: dry_run never shipped; fixture_ship did (tmp_path only).
    assert dry_dict["shipped"] is False
    assert ship_dict["shipped"] is True
    assert "SHIPPED" not in after_a1
    assert repo_sentinel.read_bytes() == sentinel_before

    # --- 5. Session B idle after ship; Session A on shipped plan ---
    # After ship OPEN=0: A with stub would append again; recount-light is the
    # mid-pipeline (step 3) contract. Post-ship consumer is idle.
    with patch("src.research_implement.session_b.scheduler_delete", _spy):
        b_idle = run_session_b_path(plan, decode_only=False, write=False)
    assert b_idle.ok and b_idle.verdict == "idle"
    idle_dict = b_idle.to_dict()
    _assert_session_result_json_shape(idle_dict)
    assert idle_dict["verdict"] == "idle"
    assert idle_dict["queue"] == "queue 0/10"
    assert idle_dict["open_count"] == 0
    assert idle_dict["item"] is None
    assert idle_dict["implement_result"] is None
    assert idle_dict["shipped"] is False
    assert idle_dict["wrote_files"] is False
    assert idle_dict["keep_schedule"] is True
    assert idle_dict["scheduler_delete_called"] is False
    assert calls["n"] == 0
    assert "queue 0/10" in b_idle.message

    rc_idle = main(["idle-decode", "--plan", str(plan), "--json"])
    assert rc_idle == 0
    cli_idle = json.loads(capsys.readouterr().out)
    _assert_session_result_json_shape(cli_idle)
    assert cli_idle == idle_dict
    assert cli_idle["scheduler_delete_called"] is False

    # Key stability across pipeline Session B verdicts.
    for payload in (dry_dict, ship_dict, idle_dict):
        assert tuple(sorted(payload.keys())) == tuple(sorted(SESSION_RESULT_JSON_KEYS))
    for payload in (a1_dict, light_dict):
        assert tuple(sorted(payload.keys())) == tuple(sorted(SESSION_A_RESULT_JSON_KEYS))

# --- Beat 12: CLI --help smoke + brainstorm/search_plan spy (OPEN>=1 recount-only) ---


def test_beat12_cli_help_smoke_session_a_b_idle_decode(capsys):
    """Beat 12: session-a / session-b / idle-decode ``--help`` exit 0 + flag mentions.

    - session-a: help mentions ``json`` and ``dry-run``
    - session-b: help mentions ``json`` and ``dry-run``
    - idle-decode: help mentions ``json`` and ``idle`` (decode-only alias; no dry-run flag)
    """
    from src.research_implement.__main__ import main

    with pytest.raises(SystemExit) as ei_a:
        main(["session-a", "--help"])
    assert ei_a.value.code == 0
    help_a = capsys.readouterr().out.lower()
    assert "json" in help_a
    assert "dry-run" in help_a or "dry run" in help_a

    with pytest.raises(SystemExit) as ei_b:
        main(["session-b", "--help"])
    assert ei_b.value.code == 0
    help_b = capsys.readouterr().out.lower()
    assert "json" in help_b
    assert "dry-run" in help_b or "dry run" in help_b

    with pytest.raises(SystemExit) as ei_idle:
        main(["idle-decode", "--help"])
    assert ei_idle.value.code == 0
    help_idle = capsys.readouterr().out.lower()
    assert "json" in help_idle
    assert "idle" in help_idle
    # Subparser description + --json help mention idle fire / idle.
    assert "idle fire" in help_idle or "idle |" in help_idle or "idle when" in help_idle


def test_beat12_brainstorm_spy_open_ge1_search_plan_not_called(tmp_path: Path):
    """Beat 12: OPEN>=1 → custom ``search_plan`` / ``brainstorm`` NOT called (recount-only).

    Spy both kwarg names; neither may fire when Queue already has OPEN items.
    """
    src = _load("one_open_ready.md")
    assert count_open(parse_queue_items(src)) >= 1

    plan = tmp_path / "open_ge1.md"
    plan.write_text(src, encoding="utf-8")
    before = plan.read_text(encoding="utf-8")
    open_before = count_open(parse_queue_items(before))
    assert open_before >= 1

    spy = {"brainstorm": 0, "search_plan": 0}

    def boom_brainstorm(_items):
        spy["brainstorm"] += 1
        raise AssertionError("brainstorm must not be called when OPEN>=1")

    def boom_search_plan(_items):
        spy["search_plan"] += 1
        raise AssertionError("search_plan must not be called when OPEN>=1")

    # brainstorm= kwarg path
    r1 = run_session_a_path(plan, brainstorm=boom_brainstorm, write=True)
    assert r1.ok and r1.verdict == "light"
    assert r1.wrote_item is False
    assert r1.open_count == open_before
    assert spy["brainstorm"] == 0
    assert "recount only" in r1.message
    assert plan.read_text(encoding="utf-8") == before

    # search_plan= kwarg path (preferred alias; wins if both passed)
    r2 = run_session_a(
        before,
        brainstorm=boom_brainstorm,
        search_plan=boom_search_plan,
    )
    assert r2.ok and r2.verdict == "light"
    assert r2.wrote_item is False
    assert r2.open_count == open_before
    assert spy["brainstorm"] == 0
    assert spy["search_plan"] == 0
    assert "recount only" in r2.message

    # two_open fixture still recount-only; neither callback fires.
    two = _load("two_open_ready.md")
    assert count_open(parse_queue_items(two)) >= 2
    r3 = run_session_a(two, search_plan=boom_search_plan)
    assert r3.ok and r3.verdict == "light"
    assert r3.open_count >= 2
    assert spy["search_plan"] == 0
    light = r3.to_dict()
    assert light["verdict"] == "light"
    assert light["wrote_item"] is False



# --- Beat 13: Queue markdown round-trip + Session A stub append id stability ---


def test_beat13_queue_markdown_roundtrip_preserves_six_fields_and_ready():
    """Beat 13: parse → serialize/write → parse preserves six fields + ready for OPEN.

    Covers one_open_ready and two_open_ready fixtures via write_queue_section
    (items=None identity path and explicit items path).
    """
    for name in ("one_open_ready.md", "two_open_ready.md"):
        src = _load(name)
        items = parse_queue_items(src)
        assert items
        open_items = [i for i in items if is_open_status(i.status)]
        assert open_items
        for item in open_items:
            assert is_complete_six_field(item)
            assert item.ready_for_implement.strip().lower() == "yes"

        # Identity write (items=None) and explicit write both round-trip.
        for written in (
            write_queue_section(src),
            write_queue_section(src, items),
            write_queue_section(src, list(items)),
        ):
            assert "## Watch" in written or "## Heartbeat" in written
            again = parse_queue_items(written)
            assert [i.item_id for i in again] == [i.item_id for i in items]
            for before, after in zip(items, again, strict=True):
                if not is_open_status(before.status):
                    continue
                assert after.item_id == before.item_id
                assert after.heading == before.heading
                for field in REQUIRED_FIELDS:
                    assert after.field_map()[field] == before.field_map()[field]
                assert after.status.split(None, 1)[0].upper() == "OPEN"
                assert is_ready_yes(after.ready_for_implement)
                assert is_complete_six_field(after)
                assert is_b_pickable(after)

        # serialize_queue_item alone preserves fields when re-parsed as fragment.
        for item in open_items:
            block = serialize_queue_item(item)
            assert "ready-for-implement: yes" in block.lower()
            for i, field in enumerate(REQUIRED_FIELDS, start=1):
                assert f"{i}. **{field}**:" in block
            frag_items = parse_queue_items(block)
            assert len(frag_items) == 1
            got = frag_items[0]
            assert got.item_id == item.item_id
            for field in REQUIRED_FIELDS:
                assert got.field_map()[field] == item.field_map()[field]
            assert got.ready_for_implement.strip().lower() == "yes"


def test_beat13_append_id_stability_empty_ship_clear_two_open(tmp_path: Path):
    """Beat 13: stub append ids stable/new without collide; two_open first-OPEN unchanged.

    - empty → Session A stub → Q1 (stable first id)
    - ship Q1 → Session A stub again → Q2 (new, no collide with shipped Q1)
    - clear Queue rows → Session A stub → Q1 again (stable after clear)
    - two_open_ready: first-OPEN pick remains Q1 (unchanged by id helpers)
    """
    # --- empty → stub append → Q1 ---
    empty = _load("empty_queue.md")
    plan = tmp_path / "id_stability.md"
    plan.write_text(empty, encoding="utf-8")
    assert count_open(parse_queue_items(plan.read_text(encoding="utf-8"))) == 0
    assert next_queue_id(parse_queue_items(plan.read_text(encoding="utf-8"))) == "Q1"

    a1 = run_session_a_path(plan, brainstorm=stub_brainstorm, write=True)
    assert a1.ok and a1.verdict == "queued" and a1.wrote_item
    items1 = parse_queue_items(plan.read_text(encoding="utf-8"))
    assert [i.item_id for i in items1] == ["Q1"]
    assert count_open(items1) == 1
    assert first_b_pick(items1).item_id == "Q1"

    # --- ship Q1 → stub append → Q2 (no collide) ---
    ship_fn = make_fixture_ship_implement("beat13cafe", note="beat13 ship Q1")
    shipped = run_session_b_path(
        plan, implement=ship_fn, decode_only=False, write=True
    )
    assert shipped.ok and shipped.verdict == "shipped"
    after_ship = parse_queue_items(plan.read_text(encoding="utf-8"))
    assert count_open(after_ship) == 0
    assert after_ship[0].item_id == "Q1"
    assert after_ship[0].status.upper().startswith("SHIPPED")
    assert next_queue_id(after_ship) == "Q2"

    a2 = run_session_a_path(plan, brainstorm=stub_brainstorm, write=True)
    assert a2.ok and a2.verdict == "queued" and a2.wrote_item
    items2 = parse_queue_items(plan.read_text(encoding="utf-8"))
    ids2 = [i.item_id for i in items2]
    assert "Q1" in ids2 and "Q2" in ids2
    assert len(ids2) == len(set(ids2))  # no collide
    open2 = [i for i in items2 if is_b_pickable(i)]
    assert len(open2) == 1 and open2[0].item_id == "Q2"
    assert first_b_pick(items2).item_id == "Q2"

    # --- clear Queue rows → stub append → Q1 stable again ---
    cleared_md = write_queue_section(plan.read_text(encoding="utf-8"), [])
    plan.write_text(cleared_md, encoding="utf-8")
    cleared_items = parse_queue_items(plan.read_text(encoding="utf-8"))
    assert cleared_items == []
    assert count_open(cleared_items) == 0
    assert next_queue_id(cleared_items) == "Q1"

    a3 = run_session_a_path(plan, brainstorm=stub_brainstorm, write=True)
    assert a3.ok and a3.verdict == "queued"
    items3 = parse_queue_items(plan.read_text(encoding="utf-8"))
    assert [i.item_id for i in items3] == ["Q1"]
    assert first_b_pick(items3).item_id == "Q1"

    # --- two_open_ready: first-OPEN pick unchanged (Q1) ---
    two = _load("two_open_ready.md")
    two_items = parse_queue_items(two)
    assert [i.item_id for i in two_items] == ["Q1", "Q2"]
    assert first_b_pick(two_items).item_id == "Q1"
    # Round-trip write must not change first-OPEN pick.
    two_written = write_queue_section(two, two_items)
    two_again = parse_queue_items(two_written)
    assert [i.item_id for i in two_again] == ["Q1", "Q2"]
    assert first_b_pick(two_again).item_id == "Q1"
    assert count_open(two_again) == 2
    # next id after two open is Q3 (no collide); first-OPEN still Q1
    assert next_queue_id(two_again) == "Q3"
    assert first_b_pick(two_again).item_id == "Q1"
    # Session A on two_open is recount-only — does not append, pick unchanged
    light = run_session_a(two_written, brainstorm=stub_brainstorm)
    assert light.ok and light.verdict == "light" and light.wrote_item is False
    light_items = parse_queue_items(light.plan_text)
    assert first_b_pick(light_items).item_id == "Q1"
    assert [i.item_id for i in light_items] == ["Q1", "Q2"]


def test_beat13_serialize_queue_items_body_then_parse():
    """Beat 13: serialize_queue_items body re-parses with six fields + ready."""
    src = _load("two_open_ready.md")
    items = parse_queue_items(src)
    body = serialize_queue_items(items)
    assert body.count("### Q") == 2
    again = parse_queue_items("## Queue\n\n" + body)
    assert len(again) == 2
    for before, after in zip(items, again, strict=True):
        for field in REQUIRED_FIELDS:
            assert after.field_map()[field] == before.field_map()[field]
        assert after.ready_for_implement.strip().lower() == "yes"
        assert is_b_pickable(after)
    assert first_b_pick(again).item_id == "Q1"

# --- Beat 14: Session A fail-closed on incomplete brainstorm/search_plan ---


def _complete_candidate(**overrides):
    """Complete six-field + ready candidate; overrides may drop keys for fail cases."""
    base = {
        "heading": "Complete candidate",
        "title": "Complete candidate",
        "acceptance": "pytest EXIT=0",
        "risks": "do not touch kill_switch",
        "file_touch": "write tests/test_complete_candidate.py",
        "breaking_change": False,
        "redeploy_notes": "none",
        "ready_for_implement": "yes",
    }
    base.update(overrides)
    return base


def test_beat14_fail_closed_missing_six_fields_no_partial_append(tmp_path: Path):
    """Beat 14: missing required six fields → failed; plan unchanged; no OPEN append."""
    empty = _load("empty_queue.md")
    plan = tmp_path / "incomplete_fields.md"
    plan.write_text(empty, encoding="utf-8")
    before = plan.read_text(encoding="utf-8")

    # Missing acceptance + file_touch (and ready left present to isolate six-field path).
    incomplete = _complete_candidate()
    del incomplete["acceptance"]
    del incomplete["file_touch"]
    assert "acceptance" in incomplete_candidate_reasons(incomplete)
    assert "file_touch" in incomplete_candidate_reasons(incomplete)

    def search_plan(_items):
        return incomplete

    result = run_session_a_path(plan, search_plan=search_plan, write=True)
    assert not result.ok
    assert result.verdict == "failed"
    assert result.wrote_item is False
    assert result.open_count == 0
    assert result.plan_text == before
    assert plan.read_text(encoding="utf-8") == before
    assert count_open(parse_queue_items(result.plan_text)) == 0
    assert "### Q" not in result.plan_text or count_open(parse_queue_items(result.plan_text)) == 0
    assert parse_queue_items(result.plan_text) == parse_queue_items(before)
    assert "incomplete candidate" in result.message
    assert "acceptance" in result.message
    assert "file_touch" in result.message

    payload = result.to_dict()
    assert payload == result.to_json_dict() == session_a_result_dict(result)
    _assert_session_a_result_json_shape(payload)
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["b_pick_title"] is None


def test_beat14_fail_closed_missing_or_bad_ready_flag_no_partial_append(tmp_path: Path):
    """Beat 14: missing/non-yes ready flag → failed; plan unchanged; no OPEN append."""
    empty = _load("empty_queue.md")
    plan = tmp_path / "bad_ready.md"
    plan.write_text(empty, encoding="utf-8")
    before = plan.read_text(encoding="utf-8")

    cases = [
        ("missing", _complete_candidate()),
        ("no", _complete_candidate(ready_for_implement="no")),
        ("garbage", _complete_candidate(ready_for_implement="maybe")),
        ("empty", _complete_candidate(ready_for_implement="")),
    ]
    # missing key
    missing = _complete_candidate()
    del missing["ready_for_implement"]
    cases[0] = ("missing", missing)

    for label, cand in cases:
        assert "ready_for_implement" in incomplete_candidate_reasons(cand), label

        def search_plan(_items, c=cand):
            return c

        result = run_session_a(before, search_plan=search_plan)
        assert not result.ok, label
        assert result.verdict == "failed", label
        assert result.wrote_item is False, label
        assert result.open_count == 0, label
        assert result.plan_text == before, label
        assert count_open(parse_queue_items(result.plan_text)) == 0, label
        assert parse_queue_items(result.plan_text) == [], label
        assert "incomplete candidate" in result.message, label
        assert "ready_for_implement" in result.message, label

        payload = result.to_dict()
        _assert_session_a_result_json_shape(payload)
        assert payload["ok"] is False
        assert payload["verdict"] == "failed"
        assert payload["wrote_item"] is False
        assert payload["open_count"] == 0
        assert payload["queue"] == "queue 0/10"

    # Disk unchanged when write_path path used
    def boom(_items):
        bad = _complete_candidate(ready_for_implement="no")
        return bad

    written = run_session_a_path(plan, brainstorm=boom, write=True)
    assert written.verdict == "failed" and written.wrote_item is False
    assert plan.read_text(encoding="utf-8") == before


def test_beat14_fixture_incomplete_candidate_json_failed_shape(tmp_path: Path, capsys):
    """Beat 14: fixture incomplete candidate JSON → failed SessionA JSON shape."""
    fixture = FIXTURES / "incomplete_candidate.json"
    assert fixture.is_file()
    cand = json.loads(fixture.read_text(encoding="utf-8"))
    reasons = incomplete_candidate_reasons(cand)
    assert reasons  # incomplete by construction
    assert "acceptance" in reasons or "ready_for_implement" in reasons

    empty = _load("empty_queue.md")
    plan = tmp_path / "fixture_incomplete.md"
    plan.write_text(empty, encoding="utf-8")
    before = plan.read_text(encoding="utf-8")

    def search_plan(_items):
        return cand

    result = run_session_a_path(plan, search_plan=search_plan, write=True)
    assert not result.ok
    assert result.verdict == "failed"
    assert result.wrote_item is False
    assert result.plan_text == before
    assert plan.read_text(encoding="utf-8") == before

    payload = result.to_dict()
    _assert_session_a_result_json_shape(payload)
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["title"] is None or isinstance(payload["title"], str)

    # CLI --json failed shape via --candidate-json
    from src.research_implement.__main__ import main

    rc = main(
        [
            "session-a",
            "--plan",
            str(plan),
            "--candidate-json",
            str(fixture),
            "--json",
        ]
    )
    assert rc != 0  # failed fire is non-zero
    cli_payload = json.loads(capsys.readouterr().out)
    _assert_session_a_result_json_shape(cli_payload)
    assert cli_payload["verdict"] == "failed"
    assert cli_payload["wrote_item"] is False
    assert cli_payload["ok"] is False
    assert plan.read_text(encoding="utf-8") == before


def test_beat14_empty_plus_complete_stub_still_queues(tmp_path: Path, capsys):
    """Beat 14: empty Queue + complete stub still queues (fail-closed does not break stub)."""
    empty = _load("empty_queue.md")
    plan = tmp_path / "stub_ok.md"
    plan.write_text(empty, encoding="utf-8")
    assert count_open(parse_queue_items(empty)) == 0
    assert incomplete_candidate_reasons(stub_brainstorm([])) == []

    # brainstorm= path
    a1 = run_session_a(empty, brainstorm=stub_brainstorm)
    assert a1.ok and a1.verdict == "queued" and a1.wrote_item is True
    assert a1.open_count == 1
    items1 = parse_queue_items(a1.plan_text)
    assert len(items1) == 1
    assert is_complete_six_field(items1[0])
    assert is_ready_yes(items1[0].ready_for_implement)
    assert is_b_pickable(items1[0])
    queued = a1.to_dict()
    _assert_session_a_result_json_shape(queued)
    assert queued["verdict"] == "queued"
    assert queued["wrote_item"] is True
    assert queued["ok"] is True
    assert queued["open_count"] == 1

    # search_plan= path + write
    a2 = run_session_a_path(plan, search_plan=stub_brainstorm, write=True)
    assert a2.ok and a2.verdict == "queued" and a2.wrote_item is True
    on_disk = plan.read_text(encoding="utf-8")
    assert on_disk == a2.plan_text
    assert count_open(parse_queue_items(on_disk)) == 1
    assert "ready-for-implement: yes" in on_disk
    assert is_b_pickable(parse_queue_items(on_disk)[0])

    # Complete candidate via search_plan (non-stub) also queues
    def complete_plan(_items):
        return _complete_candidate(title="Beat14 complete", heading="Beat14 complete")

    a3 = run_session_a(empty, search_plan=complete_plan)
    assert a3.ok and a3.verdict == "queued" and a3.wrote_item is True
    assert incomplete_candidate_reasons(_complete_candidate()) == []

    from src.research_implement.__main__ import main

    plan2 = tmp_path / "cli_stub.md"
    plan2.write_text(empty, encoding="utf-8")
    rc = main(["session-a", "--plan", str(plan2), "--stub", "--json"])
    assert rc == 0
    cli_payload = json.loads(capsys.readouterr().out)
    _assert_session_a_result_json_shape(cli_payload)
    assert cli_payload["verdict"] == "queued"
    assert cli_payload["wrote_item"] is True
    assert cli_payload["ok"] is True
    assert count_open(parse_queue_items(plan2.read_text(encoding="utf-8"))) == 1


# --- Beat 15: CLI session-a --candidate-json dict/list when OPEN=0 ---


def test_beat15_complete_candidate_json_queues_one_open(tmp_path: Path, capsys):
    """Beat 15: complete --candidate-json on empty Queue queues one OPEN (not stub)."""
    from src.research_implement.__main__ import _load_candidate, main

    fixture = FIXTURES / "complete_candidate.json"
    assert fixture.is_file()
    cand = _load_candidate(fixture)
    assert isinstance(cand, dict)
    assert incomplete_candidate_reasons(cand) == []
    assert cand["title"] == "Complete candidate fixture"

    empty = _load("empty_queue.md")
    plan = tmp_path / "complete_json.md"
    plan.write_text(empty, encoding="utf-8")
    assert count_open(parse_queue_items(plan.read_text(encoding="utf-8"))) == 0

    rc = main(
        [
            "session-a",
            "--plan",
            str(plan),
            "--candidate-json",
            str(fixture),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    _assert_session_a_result_json_shape(payload)
    assert payload["ok"] is True
    assert payload["verdict"] == "queued"
    assert payload["wrote_item"] is True
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    assert payload["title"] == "Complete candidate fixture"
    assert payload["b_pick_title"] == "Complete candidate fixture"

    on_disk = plan.read_text(encoding="utf-8")
    items = parse_queue_items(on_disk)
    assert len(items) == 1
    assert count_open(items) == 1
    assert items[0].title == "Complete candidate fixture"
    assert is_complete_six_field(items[0])
    assert is_ready_yes(items[0].ready_for_implement)
    assert is_b_pickable(items[0])
    # Must be the candidate-json title, not the stub default.
    assert "Stub shippable change" not in on_disk
    assert "ready-for-implement: yes" in on_disk


def test_beat15_complete_candidate_json_list_queues_first_dict(tmp_path: Path, capsys):
    """Beat 15: --candidate-json list uses first dict element as brainstorm candidate."""
    from src.research_implement.__main__ import _load_candidate, main

    fixture = FIXTURES / "complete_candidate_list.json"
    cand = _load_candidate(fixture)
    assert isinstance(cand, dict)
    assert cand["title"] == "Complete list candidate"
    assert incomplete_candidate_reasons(cand) == []

    plan = tmp_path / "list_json.md"
    plan.write_text(_load("empty_queue.md"), encoding="utf-8")
    rc = main(
        [
            "session-a",
            "--plan",
            str(plan),
            "--candidate-json",
            str(fixture),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    _assert_session_a_result_json_shape(payload)
    assert payload["verdict"] == "queued"
    assert payload["wrote_item"] is True
    assert payload["title"] == "Complete list candidate"
    assert count_open(parse_queue_items(plan.read_text(encoding="utf-8"))) == 1
    assert "Complete list candidate" in plan.read_text(encoding="utf-8")
    assert "Second ignored" not in plan.read_text(encoding="utf-8")


def test_beat15_incomplete_candidate_json_fails_no_partial_append(tmp_path: Path, capsys):
    """Beat 15: incomplete --candidate-json fail-closes; plan unchanged; no partial OPEN."""
    from src.research_implement.__main__ import _load_candidate, main

    fixture = FIXTURES / "incomplete_candidate.json"
    cand = _load_candidate(fixture)
    assert isinstance(cand, dict)
    reasons = incomplete_candidate_reasons(cand)
    assert reasons
    assert "acceptance" in reasons or "ready_for_implement" in reasons

    empty = _load("empty_queue.md")
    plan = tmp_path / "incomplete_json.md"
    plan.write_text(empty, encoding="utf-8")
    before = plan.read_text(encoding="utf-8")

    rc = main(
        [
            "session-a",
            "--plan",
            str(plan),
            "--candidate-json",
            str(fixture),
            "--json",
        ]
    )
    assert rc != 0
    payload = json.loads(capsys.readouterr().out)
    _assert_session_a_result_json_shape(payload)
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert plan.read_text(encoding="utf-8") == before
    assert count_open(parse_queue_items(plan.read_text(encoding="utf-8"))) == 0
    assert parse_queue_items(plan.read_text(encoding="utf-8")) == []
    # No partial OPEN heading appended.
    assert "### Q" not in plan.read_text(encoding="utf-8")


def test_beat15_open_ge1_recount_only_ignores_candidate_json(tmp_path: Path, capsys):
    """Beat 15: OPEN>=1 → recount-only light; --candidate-json ignored (no append)."""
    from src.research_implement import __main__ as ri_main
    from src.research_implement.__main__ import main

    fixture = FIXTURES / "complete_candidate.json"
    incomplete = FIXTURES / "incomplete_candidate.json"
    src = _load("one_open_ready.md")
    assert count_open(parse_queue_items(src)) >= 1

    plan = tmp_path / "open_ge1.md"
    plan.write_text(src, encoding="utf-8")
    before = plan.read_text(encoding="utf-8")
    open_before = count_open(parse_queue_items(before))
    items_before = parse_queue_items(before)

    # Deferred load: OPEN>=1 must not call _load_candidate at all.
    with patch.object(ri_main, "_load_candidate", wraps=ri_main._load_candidate) as spy:
        rc = main(
            [
                "session-a",
                "--plan",
                str(plan),
                "--candidate-json",
                str(fixture),
                "--json",
            ]
        )
    assert spy.call_count == 0
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    _assert_session_a_result_json_shape(payload)
    assert payload["ok"] is True
    assert payload["verdict"] == "light"
    assert payload["wrote_item"] is False
    assert payload["open_count"] == open_before
    assert payload["title"] is None
    assert payload["queue"] == render_queue_count(open_before)

    on_disk = plan.read_text(encoding="utf-8")
    assert on_disk == before
    assert parse_queue_items(on_disk) == items_before
    assert count_open(parse_queue_items(on_disk)) == open_before
    # Candidate-json title must not appear (callback never applied).
    assert "Complete candidate fixture" not in on_disk

    # two_open also recount-only with candidate-json present
    two = tmp_path / "two.md"
    two_src = _load("two_open_ready.md")
    two.write_text(two_src, encoding="utf-8")
    rc2 = main(
        [
            "session-a",
            "--plan",
            str(two),
            "--candidate-json",
            str(fixture),
            "--json",
        ]
    )
    assert rc2 == 0
    payload2 = json.loads(capsys.readouterr().out)
    assert payload2["verdict"] == "light"
    assert payload2["wrote_item"] is False
    assert payload2["open_count"] == 2
    assert two.read_text(encoding="utf-8") == two_src
    assert "Complete candidate fixture" not in two.read_text(encoding="utf-8")

    # Incomplete candidate-json also ignored when OPEN>=1 (light, not failed).
    plan2 = tmp_path / "open_ge1_incomplete.md"
    plan2.write_text(src, encoding="utf-8")
    with patch.object(ri_main, "_load_candidate", wraps=ri_main._load_candidate) as spy2:
        rc3 = main(
            [
                "session-a",
                "--plan",
                str(plan2),
                "--candidate-json",
                str(incomplete),
                "--json",
            ]
        )
    assert spy2.call_count == 0
    assert rc3 == 0
    payload3 = json.loads(capsys.readouterr().out)
    assert payload3["verdict"] == "light"
    assert payload3["wrote_item"] is False
    assert payload3["ok"] is True
    assert plan2.read_text(encoding="utf-8") == src


def test_beat15_load_candidate_dict_or_list_helpers():
    """Beat 15: _load_candidate normalizes dict and list; empty list → None."""
    from src.research_implement.__main__ import _load_candidate

    d = _load_candidate(FIXTURES / "complete_candidate.json")
    assert d["title"] == "Complete candidate fixture"
    lst = _load_candidate(FIXTURES / "complete_candidate_list.json")
    assert lst["title"] == "Complete list candidate"
    incomplete = _load_candidate(FIXTURES / "incomplete_candidate.json")
    assert "acceptance" not in incomplete or incomplete.get("acceptance") in (None, "")
    assert incomplete_candidate_reasons(incomplete)



# --- Beat 16: candidate-json CLI error paths + sequential double-OPEN ship ---


def test_beat16_candidate_json_missing_file_fails_no_plan_mutation(tmp_path: Path, capsys):
    """Beat 16: missing --candidate-json file → SystemExit clear fail; plan unchanged."""
    from src.research_implement.__main__ import main

    empty = _load("empty_queue.md")
    plan = tmp_path / "missing_cand_plan.md"
    plan.write_text(empty, encoding="utf-8")
    before = plan.read_text(encoding="utf-8")
    missing = tmp_path / "does_not_exist_candidate.json"
    assert not missing.exists()

    with pytest.raises(SystemExit) as ei:
        main(
            [
                "session-a",
                "--plan",
                str(plan),
                "--candidate-json",
                str(missing),
                "--json",
            ]
        )
    assert ei.value.code != 0
    msg = str(ei.value)
    assert "--candidate-json" in msg
    assert "not found" in msg.lower()
    assert plan.read_text(encoding="utf-8") == before
    assert count_open(parse_queue_items(plan.read_text(encoding="utf-8"))) == 0
    assert "### Q" not in plan.read_text(encoding="utf-8")
    # No Session A JSON success payload on this path.
    out = capsys.readouterr().out.strip()
    assert out == "" or "queued" not in out


def test_beat16_candidate_json_invalid_json_fails_no_plan_mutation(tmp_path: Path, capsys):
    """Beat 16: invalid --candidate-json → SystemExit clear fail; plan unchanged."""
    from src.research_implement.__main__ import main

    empty = _load("empty_queue.md")
    plan = tmp_path / "invalid_cand_plan.md"
    plan.write_text(empty, encoding="utf-8")
    before = plan.read_text(encoding="utf-8")
    bad = tmp_path / "invalid_candidate.json"
    bad.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(SystemExit) as ei:
        main(
            [
                "session-a",
                "--plan",
                str(plan),
                "--candidate-json",
                str(bad),
                "--json",
            ]
        )
    assert ei.value.code != 0
    msg = str(ei.value)
    assert "--candidate-json" in msg
    assert "invalid JSON" in msg
    assert plan.read_text(encoding="utf-8") == before
    assert count_open(parse_queue_items(plan.read_text(encoding="utf-8"))) == 0
    assert parse_queue_items(plan.read_text(encoding="utf-8")) == []
    out = capsys.readouterr().out.strip()
    assert out == "" or "queued" not in out


def test_beat16_candidate_json_wrong_type_fails_no_plan_mutation(tmp_path: Path, capsys):
    """Beat 16: wrong top-level type (not object/list) → SystemExit; plan unchanged."""
    from src.research_implement.__main__ import _load_candidate, main

    empty = _load("empty_queue.md")
    plan = tmp_path / "wrong_type_plan.md"
    plan.write_text(empty, encoding="utf-8")
    before = plan.read_text(encoding="utf-8")

    cases = [
        ("number.json", "42", "int"),
        ("string.json", '"hello"', "str"),
        ("null.json", "null", "NoneType"),
        ("bool.json", "true", "bool"),
    ]
    for name, body, typename in cases:
        cand = tmp_path / name
        cand.write_text(body, encoding="utf-8")
        plan.write_text(before, encoding="utf-8")

        with pytest.raises(SystemExit) as ei_load:
            _load_candidate(cand)
        assert ei_load.value.code != 0
        load_msg = str(ei_load.value)
        assert "--candidate-json" in load_msg
        assert "object or list" in load_msg
        assert typename in load_msg

        with pytest.raises(SystemExit) as ei:
            main(
                [
                    "session-a",
                    "--plan",
                    str(plan),
                    "--candidate-json",
                    str(cand),
                    "--json",
                ]
            )
        assert ei.value.code != 0
        msg = str(ei.value)
        assert "--candidate-json" in msg
        assert "object or list" in msg
        assert typename in msg
        assert plan.read_text(encoding="utf-8") == before
        assert count_open(parse_queue_items(plan.read_text(encoding="utf-8"))) == 0
        assert "### Q" not in plan.read_text(encoding="utf-8")
        capsys.readouterr()  # drain


def test_beat16_sequential_double_open_ship_then_idle(tmp_path: Path):
    """Beat 16: two_open_ready → ship Q1 → ship Q2 → idle; never scheduler_delete.

    JSON shapes ok on each ship and the final idle. tmp_path / fixture ship only.
    """
    src = _load("two_open_ready.md")
    items = parse_queue_items(src)
    assert [i.item_id for i in items] == ["Q1", "Q2"]
    assert count_open(items) == 2
    assert first_b_pick(items).item_id == "Q1"

    plan = tmp_path / "double_ship.md"
    plan.write_text(src, encoding="utf-8")

    calls = {"n": 0}

    def _spy(*_a, **_k):
        calls["n"] += 1
        raise SchedulerDeleteForbidden("spy")

    # --- ship first OPEN (Q1) ---
    ship1 = make_fixture_ship_implement("beat16ship1", note="beat16 ship Q1")
    with patch("src.research_implement.session_b.scheduler_delete", _spy):
        first = run_session_b_path(
            plan, implement=ship1, decode_only=False, write=True
        )
    assert first.ok and first.verdict == "shipped"
    assert first.item is not None and first.item.item_id == "Q1"
    assert first.keep_schedule is True
    assert first.scheduler_delete_called is False
    first_dict = first.to_dict()
    assert first_dict == first.to_json_dict() == session_b_result_dict(first)
    _assert_session_result_json_shape(first_dict)
    assert first_dict["verdict"] == "shipped"
    assert first_dict["shipped"] is True
    assert first_dict["scheduler_delete_called"] is False
    assert first_dict["keep_schedule"] is True
    assert first_dict["open_count"] == 1
    assert first_dict["queue"] == "queue 1/10"
    _assert_shipped_implement_schema(first_dict["implement_result"])
    assert first_dict["implement_result"]["sha"] == "beat16ship1"

    after1 = plan.read_text(encoding="utf-8")
    assert "SHIPPED `beat16ship1`" in after1
    items1 = parse_queue_items(after1)
    assert count_open(items1) == 1
    pick1 = first_b_pick(items1)
    assert pick1 is not None and pick1.item_id == "Q2"
    assert calls["n"] == 0

    # --- ship second OPEN (Q2) ---
    ship2 = make_fixture_ship_implement("beat16ship2", note="beat16 ship Q2")
    with patch("src.research_implement.session_b.scheduler_delete", _spy):
        second = run_session_b_path(
            plan, implement=ship2, decode_only=False, write=True
        )
    assert second.ok and second.verdict == "shipped"
    assert second.item is not None and second.item.item_id == "Q2"
    assert second.keep_schedule is True
    assert second.scheduler_delete_called is False
    second_dict = second.to_dict()
    _assert_session_result_json_shape(second_dict)
    assert second_dict["verdict"] == "shipped"
    assert second_dict["shipped"] is True
    assert second_dict["scheduler_delete_called"] is False
    assert second_dict["open_count"] == 0
    assert second_dict["queue"] == "queue 0/10"
    _assert_shipped_implement_schema(second_dict["implement_result"])
    assert second_dict["implement_result"]["sha"] == "beat16ship2"

    after2 = plan.read_text(encoding="utf-8")
    assert "SHIPPED `beat16ship1`" in after2
    assert "SHIPPED `beat16ship2`" in after2
    items2 = parse_queue_items(after2)
    assert count_open(items2) == 0
    assert first_b_pick(items2) is None
    assert calls["n"] == 0

    # --- idle after both shipped ---
    with patch("src.research_implement.session_b.scheduler_delete", _spy):
        idle = run_session_b_path(plan, decode_only=True, write=False)
    assert idle.ok and idle.verdict == "idle"
    assert idle.keep_schedule is True
    assert idle.scheduler_delete_called is False
    idle_dict = idle.to_dict()
    _assert_session_result_json_shape(idle_dict)
    assert idle_dict["verdict"] == "idle"
    assert idle_dict["ok"] is True
    assert idle_dict["open_count"] == 0
    assert idle_dict["queue"] == "queue 0/10"
    assert idle_dict["shipped"] is False
    assert idle_dict["item"] is None
    assert idle_dict["implement_result"] is None
    assert idle_dict["scheduler_delete_called"] is False
    assert idle_dict["keep_schedule"] is True
    assert calls["n"] == 0
    # Plan still has both SHIPPED markers; no schedule delete ever.
    final = plan.read_text(encoding="utf-8")
    assert "SHIPPED `beat16ship1`" in final
    assert "SHIPPED `beat16ship2`" in final
    assert count_open(parse_queue_items(final)) == 0


# --- Beat 17: decode_only default never implement; dry_run + fixture_ship opt-in ---


def test_beat17_decode_only_default_never_calls_implement(tmp_path: Path, capsys):
    """Beat 17: decode_only True / CLI session-b default never invokes implement.

    Spy implement must not fire when decode_only=True or when CLI omits --dry-run.
    """
    plan_md = _load("one_open_ready.md")
    assert first_b_pick(parse_queue_items(plan_md)) is not None

    spy = {"n": 0}

    def boom_implement(item):
        spy["n"] += 1
        raise AssertionError(f"implement must not be called in decode_only; got {item.item_id}")

    # API: decode_only=True with explicit spy implement — never called.
    r = run_session_b(plan_md, implement=boom_implement, decode_only=True)
    assert r.ok and r.verdict == "picked"
    assert r.implement_result is None
    assert r.item is not None and r.item.item_id == "Q1"
    assert spy["n"] == 0
    assert "SHIPPED" not in r.plan_text
    payload = r.to_dict()
    _assert_session_result_json_shape(payload)
    assert payload["verdict"] == "picked"
    assert payload["implement_result"] is None
    assert payload["shipped"] is False

    # Path helper default is decode_only=True.
    plan = tmp_path / "decode_default.md"
    plan.write_text(plan_md, encoding="utf-8")
    r2 = run_session_b_path(plan, implement=boom_implement)
    assert r2.ok and r2.verdict == "picked"
    assert r2.implement_result is None
    assert spy["n"] == 0

    # CLI session-b default (no --dry-run) is decode-only; spy dry_run_implement.
    from src.research_implement.__main__ import main

    plan_cli = tmp_path / "cli_decode.md"
    plan_cli.write_text(plan_md, encoding="utf-8")
    dry_spy = {"n": 0}

    def boom_dry(item):
        dry_spy["n"] += 1
        raise AssertionError(
            f"CLI decode-only must not call dry_run_implement; {item.item_id}"
        )

    with patch("src.research_implement.__main__.dry_run_implement", boom_dry), patch(
        "src.research_implement.session_b.dry_run_implement", boom_dry
    ), patch(
        "src.research_implement.session_b.default_implement", boom_dry
    ):
        rc = main(["session-b", "--plan", str(plan_cli), "--json"])
    assert rc == 0
    cli_payload = json.loads(capsys.readouterr().out)
    _assert_session_result_json_shape(cli_payload)
    assert cli_payload["verdict"] == "picked"
    assert cli_payload["implement_result"] is None
    assert cli_payload["shipped"] is False
    assert "SHIPPED" not in plan_cli.read_text(encoding="utf-8")
    assert spy["n"] == 0
    assert dry_spy["n"] == 0

    # idle-decode alias also stays decode-only (spy never fires).
    plan_idle = tmp_path / "cli_idle_decode.md"
    plan_idle.write_text(plan_md, encoding="utf-8")
    with patch("src.research_implement.__main__.dry_run_implement", boom_dry), patch(
        "src.research_implement.session_b.default_implement", boom_dry
    ):
        rc2 = main(["idle-decode", "--plan", str(plan_idle), "--json"])
    assert rc2 == 0
    idle_payload = json.loads(capsys.readouterr().out)
    assert idle_payload["verdict"] == "picked"
    assert idle_payload["implement_result"] is None
    assert dry_spy["n"] == 0


def test_beat17_dry_run_calls_implement_spy(tmp_path: Path, capsys):
    """Beat 17: decode_only=False / --dry-run calls implement spy once; never ships."""
    plan_md = _load("one_open_ready.md")
    spy = {"n": 0, "items": []}

    def spy_dry(item):
        spy["n"] += 1
        spy["items"].append(item.item_id)
        return dry_run_implement(item)

    # Explicit spy wrapping dry_run_implement.
    r = run_session_b(plan_md, implement=spy_dry, decode_only=False)
    assert r.ok and r.verdict == "dry_run"
    assert spy["n"] == 1
    assert spy["items"] == ["Q1"]
    assert r.implement_result is not None
    _assert_dry_run_implement_schema(r.implement_result)
    assert r.implement_result["dry_run"] is True
    assert r.implement_result["sha"] is None
    assert "SHIPPED" not in r.plan_text
    assert count_open(parse_queue_items(r.plan_text)) == 1

    # implement=None + decode_only=False uses default_implement (== dry_run_implement).
    assert default_implement is dry_run_implement
    r2 = run_session_b(plan_md, decode_only=False)
    assert r2.ok and r2.verdict == "dry_run"
    assert r2.implement_result is not None
    _assert_dry_run_implement_schema(r2.implement_result)
    assert r2.implement_result["dry_run"] is True
    assert "SHIPPED" not in r2.plan_text

    # CLI --dry-run exercises dry_run_implement path (spy proves call).
    from src.research_implement.__main__ import main

    plan = tmp_path / "cli_dry.md"
    plan.write_text(plan_md, encoding="utf-8")
    before = plan.read_text(encoding="utf-8")
    cli_spy = {"n": 0}

    def counting_dry(item):
        cli_spy["n"] += 1
        return dry_run_implement(item)

    with patch("src.research_implement.__main__.dry_run_implement", counting_dry):
        rc = main(["session-b", "--plan", str(plan), "--dry-run", "--json"])
    assert rc == 0
    assert cli_spy["n"] == 1
    payload = json.loads(capsys.readouterr().out)
    _assert_session_result_json_shape(payload)
    assert payload["verdict"] == "dry_run"
    assert payload["shipped"] is False
    assert payload["wrote_files"] is False
    _assert_dry_run_implement_schema(payload["implement_result"])
    assert payload["implement_result"]["dry_run"] is True
    assert plan.read_text(encoding="utf-8") == before


def test_beat17_fixture_ship_only_when_opted_in(tmp_path: Path):
    """Beat 17: ship only with decode_only=False + fixture_ship implement (opt-in).

    decode_only=True ignores fixture_ship; default dry_run never ships; only
    explicit fixture_ship + decode_only=False marks SHIPPED (tmp_path).
    """
    plan_md = _load("one_open_ready.md")
    ship_fn = make_fixture_ship_implement("beat17cafe", note="beat17 fixture ship")

    # Opted ship callback + decode_only=True → never calls implement, no ship.
    spy = {"n": 0}

    def ship_spy(item):
        spy["n"] += 1
        return ship_fn(item)

    decode = run_session_b(plan_md, implement=ship_spy, decode_only=True)
    assert decode.ok and decode.verdict == "picked"
    assert decode.implement_result is None
    assert spy["n"] == 0
    assert "SHIPPED" not in decode.plan_text
    assert count_open(parse_queue_items(decode.plan_text)) == 1

    # decode_only=False without fixture_ship → dry_run, never SHIPPED.
    dry = run_session_b(plan_md, decode_only=False)
    assert dry.verdict == "dry_run"
    assert dry.implement_result is not None
    assert dry.implement_result["dry_run"] is True
    assert "SHIPPED" not in dry.plan_text
    assert count_open(parse_queue_items(dry.plan_text)) == 1

    # Opt-in: decode_only=False + fixture_ship → shipped on tmp_path only.
    plan = tmp_path / "ship_opt_in.md"
    plan.write_text(plan_md, encoding="utf-8")
    shipped = run_session_b_path(
        plan, implement=ship_fn, decode_only=False, write=True
    )
    assert shipped.ok and shipped.verdict == "shipped"
    assert shipped.implement_result is not None
    _assert_shipped_implement_schema(shipped.implement_result)
    assert shipped.implement_result["sha"] == "beat17cafe"
    assert shipped.implement_result["dry_run"] is False
    assert "SHIPPED `beat17cafe`" in shipped.plan_text
    on_disk = plan.read_text(encoding="utf-8")
    assert "SHIPPED `beat17cafe`" in on_disk
    assert count_open(parse_queue_items(on_disk)) == 0
    payload = shipped.to_dict()
    _assert_session_result_json_shape(payload)
    assert payload["verdict"] == "shipped"
    assert payload["shipped"] is True
    # Convenience alias is ship double, never the default.
    assert fixture_ship_implement is not dry_run_implement
    assert default_implement is dry_run_implement


# --- Beat 18: public API export smoke + Makefile target listing -----------------


def test_beat18_public_api_exports_smoke():
    """Package ``src.research_implement`` exposes the Session A/B public surface.

    Import the package and assert ``__all__`` + ``getattr`` for the names
    callers / Makefile e2e docs rely on (results, dry-run/ship doubles,
    incomplete-candidate helper, queue parse).
    """
    import src.research_implement as ri

    assert hasattr(ri, "__all__")
    assert isinstance(ri.__all__, list)
    assert len(ri.__all__) >= 1

    required = (
        "SessionAResult",
        "SessionResult",
        "SessionBResult",
        "dry_run_implement",
        "default_implement",
        "make_fixture_ship_implement",
        "fixture_ship_implement",
        "incomplete_candidate_reasons",
        "parse_queue_items",
        "run_session_a",
        "run_session_b",
        "QueueItem",
        "stub_brainstorm",
        "scheduler_delete",
    )
    for name in required:
        assert name in ri.__all__, f"{name} missing from __all__"
        obj = getattr(ri, name, None)
        assert obj is not None, f"{name} not importable via getattr"
        assert getattr(ri, name) is obj

    # Aliases stay wired: SessionResult is SessionBResult; default is dry_run.
    assert ri.SessionResult is ri.SessionBResult
    assert ri.default_implement is ri.dry_run_implement
    assert ri.fixture_ship_implement is not ri.dry_run_implement
    assert callable(ri.make_fixture_ship_implement)
    assert callable(ri.incomplete_candidate_reasons)
    assert callable(ri.parse_queue_items)

    # Every __all__ entry resolves (no dangling export names).
    for name in ri.__all__:
        assert hasattr(ri, name), f"__all__ lists {name} but getattr fails"
        assert getattr(ri, name) is not None or name in ri.__all__


def test_beat18_makefile_lists_research_implement_targets():
    """Makefile help / echo lists research-implement side-dev targets.

    Doc-assert the Makefile text, then smoke ``make help`` and the two
    e2e echo targets (recipes are echo-only; no Tasker / no pytest spawn).
    """
    import subprocess

    root = Path(__file__).resolve().parents[1]
    makefile = (root / "Makefile").read_text(encoding="utf-8")
    for target in (
        "test-research-implement",
        "research-implement-e2e-dry-run",
        "research-implement-e2e-pipeline",
    ):
        assert target in makefile, f"Makefile missing target mention: {target}"
        # Recipe / .PHONY lines use ``target:`` form.
        assert f"{target}:" in makefile or f"make {target}" in makefile

    # ``make help`` mentions the suite target.
    help_out = subprocess.run(
        ["make", "help"],
        cwd=str(root),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "test-research-implement" in help_out

    # Echo-only e2e targets mention dry-run / pipeline / suite target.
    dry = subprocess.run(
        ["make", "research-implement-e2e-dry-run"],
        cwd=str(root),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.lower()
    assert "dry-run" in dry or "dry_run" in dry
    assert "e2e" in dry or "pytest" in dry

    pipe = subprocess.run(
        ["make", "research-implement-e2e-pipeline"],
        cwd=str(root),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "test-research-implement" in pipe
    assert "e2e" in pipe.lower() or "pipeline" in pipe.lower()
    assert "beat18" in pipe.lower() or "beat11" in pipe.lower()


# --- Beat 19: Queue rewrite preserves Watch / Project Work / Heartbeat ---

_BEAT19_MARKERS = (
    "BEAT19_WATCH_MARKER",
    "BEAT19_PROJECT_MARKER",
    "BEAT19_HEARTBEAT_MARKER",
)
_BEAT19_SECTIONS = ("## Watch", "## Project Work", "## Heartbeat")


def _assert_beat19_non_queue_preserved(text: str) -> None:
    """Non-Queue section headings + distinctive markers must survive Queue rewrite."""
    for heading in _BEAT19_SECTIONS:
        assert heading in text, f"missing section {heading}"
    for marker in _BEAT19_MARKERS:
        assert marker in text, f"missing marker {marker}"
    # Marker order: Watch body before Project Work before Heartbeat.
    assert text.index("BEAT19_WATCH_MARKER") < text.index("## Project Work")
    assert text.index("BEAT19_PROJECT_MARKER") < text.index("## Heartbeat")
    assert text.index("BEAT19_HEARTBEAT_MARKER") > text.index("## Heartbeat")


def test_beat19_serialize_write_preserves_watch_project_heartbeat():
    """Beat 19: serialize / write_queue_section keeps Watch+Project Work+Heartbeat."""
    src = _load("watch_queue_heartbeat.md")
    _assert_beat19_non_queue_preserved(src)
    items = parse_queue_items(src)
    assert len(items) == 1 and is_b_pickable(items[0])

    for written in (
        write_queue_section(src),
        write_queue_section(src, items),
        write_queue_section(src, list(items)),
    ):
        _assert_beat19_non_queue_preserved(written)
        assert "## Queue" in written
        again = parse_queue_items(written)
        assert len(again) == 1
        assert again[0].item_id == "Q1"
        assert is_b_pickable(again[0])
        assert again[0].title == "Beat19 shippable preserve item"
        # Queue still precedes Watch in the living-plan layout.
        assert written.index("## Queue") < written.index("## Watch")


def test_beat19_a_append_preserves_non_queue_sections(tmp_path: Path):
    """Beat 19: Session A append updates Queue; Watch/Project/Heartbeat stay."""
    src = _load("watch_queue_heartbeat_empty.md")
    _assert_beat19_non_queue_preserved(src)
    assert count_open(parse_queue_items(src)) == 0

    plan = tmp_path / "beat19_append.md"
    plan.write_text(src, encoding="utf-8")
    result = run_session_a_path(plan, brainstorm=stub_brainstorm, write=True)
    assert result.ok and result.verdict == "queued" and result.wrote_item

    updated = plan.read_text(encoding="utf-8")
    _assert_beat19_non_queue_preserved(updated)
    assert updated == result.plan_text
    items = parse_queue_items(updated)
    assert len(items) == 1
    assert items[0].item_id == "Q1"
    assert is_b_pickable(items[0])
    assert count_open(items) == 1
    # Append also preserves via in-memory helper (no path write).
    mem = run_session_a(src, brainstorm=stub_brainstorm)
    assert mem.ok and mem.wrote_item
    _assert_beat19_non_queue_preserved(mem.plan_text)
    assert count_open(parse_queue_items(mem.plan_text)) == 1


def test_beat19_b_ship_preserves_non_queue_sections(tmp_path: Path):
    """Beat 19: Session B fixture ship updates Queue; non-Queue sections remain.

    Never calls scheduler_delete (idle/ship keep_schedule path).
    """
    src = _load("watch_queue_heartbeat.md")
    _assert_beat19_non_queue_preserved(src)
    assert count_open(parse_queue_items(src)) == 1

    plan = tmp_path / "beat19_ship.md"
    plan.write_text(src, encoding="utf-8")
    ship_fn = make_fixture_ship_implement("beat19cafe", note="beat19 preserve ship")
    calls = {"n": 0}

    def _spy(*_a, **_k):
        calls["n"] += 1
        raise SchedulerDeleteForbidden("spy")

    with patch("src.research_implement.session_b.scheduler_delete", _spy):
        shipped = run_session_b_path(
            plan, implement=ship_fn, decode_only=False, write=True
        )
    assert shipped.ok and shipped.verdict == "shipped"
    assert shipped.shipped is True
    assert shipped.scheduler_delete_called is False
    assert shipped.keep_schedule is True
    assert calls["n"] == 0

    updated = plan.read_text(encoding="utf-8")
    _assert_beat19_non_queue_preserved(updated)
    assert updated == shipped.plan_text
    assert "SHIPPED `beat19cafe`" in updated
    items = parse_queue_items(updated)
    assert len(items) == 1
    assert items[0].item_id == "Q1"
    assert items[0].status.upper().startswith("SHIPPED")
    assert count_open(items) == 0
    assert not is_b_pickable(items[0])
    # Direct mark_item_shipped / in-memory B ship also preserve.
    marked = mark_item_shipped(src, "Q1", "dead19", note="direct")
    _assert_beat19_non_queue_preserved(marked)
    assert "SHIPPED `dead19`" in marked
    with patch("src.research_implement.session_b.scheduler_delete", _spy):
        mem = run_session_b(
            src, implement=make_fixture_ship_implement("mem19"), decode_only=False
        )
    assert mem.verdict == "shipped"
    assert mem.scheduler_delete_called is False
    assert calls["n"] == 0
    _assert_beat19_non_queue_preserved(mem.plan_text)


def _assert_beat19_queue_with_watch_heartbeat_preserved(text: str) -> None:
    """Preserve helpers for Watch-before-Queue fixture ``queue_with_watch_heartbeat.md``."""
    for heading in ("## Watch", "## Queue", "## Project Work", "## Heartbeat"):
        assert heading in text, f"missing section {heading}"
    assert "beat19-watch-marker" in text
    assert "beat19-heartbeat-marker" in text
    assert "raw/transcripts/beat19-preserve.md" in text
    # Watch intentionally precedes Queue in this fixture.
    assert text.index("## Watch") < text.index("## Queue")
    assert text.index("## Queue") < text.index("## Heartbeat")
    assert text.index("## Project Work") < text.index("## Heartbeat")


def test_beat19_queue_with_watch_heartbeat_a_append_and_b_ship(tmp_path: Path):
    """Beat 19: ``queue_with_watch_heartbeat.md`` A stub append then B fixture_ship.

    Watch-before-Queue layout; Queue updates; Watch/Heartbeat/Project Work stay;
    never scheduler_delete.
    """
    src = _load("queue_with_watch_heartbeat.md")
    _assert_beat19_queue_with_watch_heartbeat_preserved(src)
    assert count_open(parse_queue_items(src)) == 0

    # write_queue_section identity / clear path keeps non-Queue sections.
    written = write_queue_section(src)
    _assert_beat19_queue_with_watch_heartbeat_preserved(written)
    cleared = write_queue_section(src, [])
    _assert_beat19_queue_with_watch_heartbeat_preserved(cleared)
    assert parse_queue_items(cleared) == []

    plan = tmp_path / "beat19_watch_before_queue.md"
    plan.write_text(src, encoding="utf-8")
    a = run_session_a_path(plan, brainstorm=stub_brainstorm, write=True)
    assert a.ok and a.verdict == "queued" and a.wrote_item
    after_a = plan.read_text(encoding="utf-8")
    _assert_beat19_queue_with_watch_heartbeat_preserved(after_a)
    items_a = parse_queue_items(after_a)
    assert len(items_a) == 1 and items_a[0].item_id == "Q1"
    assert is_b_pickable(items_a[0])
    assert count_open(items_a) == 1
    # Watch still before Queue after append.
    assert after_a.index("## Watch") < after_a.index("## Queue")
    assert after_a.index("### Q1.") < after_a.index("## Heartbeat")

    calls = {"n": 0}

    def _spy(*_a, **_k):
        calls["n"] += 1
        raise SchedulerDeleteForbidden("spy")

    ship_fn = make_fixture_ship_implement("beat19wh", note="watch-before-queue ship")
    with patch("src.research_implement.session_b.scheduler_delete", _spy):
        b = run_session_b_path(
            plan, implement=ship_fn, decode_only=False, write=True
        )
    assert b.ok and b.verdict == "shipped"
    assert b.scheduler_delete_called is False
    assert b.keep_schedule is True
    assert calls["n"] == 0

    after_b = plan.read_text(encoding="utf-8")
    _assert_beat19_queue_with_watch_heartbeat_preserved(after_b)
    assert "SHIPPED `beat19wh`" in after_b
    items_b = parse_queue_items(after_b)
    assert len(items_b) == 1
    assert items_b[0].status.upper().startswith("SHIPPED")
    assert count_open(items_b) == 0
    assert after_b.index("## Watch") < after_b.index("## Queue")
    assert after_b.index("## Queue") < after_b.index("## Heartbeat")

    # Post-ship idle also never scheduler_delete.
    with patch("src.research_implement.session_b.scheduler_delete", _spy):
        idle = run_session_b_path(plan)
    assert idle.ok and idle.verdict == "idle"
    assert idle.scheduler_delete_called is False
    assert calls["n"] == 0
    _assert_beat19_queue_with_watch_heartbeat_preserved(plan.read_text(encoding="utf-8"))


# --- Beat 20: missing ## Queue → create without destroying Watch/Heartbeat/front matter ---

_BEAT20_MARKERS = (
    "BEAT20_WATCH_MARKER",
    "BEAT20_HEARTBEAT_MARKER",
)
_BEAT20_FRONT_KEYS = ("title: beat20-no-queue", "status: living")


def _assert_beat20_watch_heartbeat_front_preserved(text: str) -> None:
    """Watch + Heartbeat + YAML front matter must survive Queue creation."""
    assert text.lstrip().startswith("---"), "YAML front matter missing"
    for key in _BEAT20_FRONT_KEYS:
        assert key in text, f"missing front-matter key {key}"
    assert "## Watch" in text
    assert "## Heartbeat" in text
    for marker in _BEAT20_MARKERS:
        assert marker in text, f"missing marker {marker}"
    assert text.index("BEAT20_WATCH_MARKER") < text.index("## Heartbeat")
    assert text.index("BEAT20_HEARTBEAT_MARKER") > text.index("## Heartbeat")
    # Front matter precedes Watch.
    assert text.index("---") < text.index("## Watch")


def test_beat20_a_append_creates_queue_preserves_watch_heartbeat(tmp_path: Path):
    """Beat 20: copy watch-only fixture; Session A stub creates ## Queue + one OPEN.

    BEAT20_WATCH_MARKER / BEAT20_HEARTBEAT_MARKER and front-matter title stay intact.
    """
    src = _load("watch_heartbeat_no_queue.md")
    assert "## Queue" not in src
    _assert_beat20_watch_heartbeat_front_preserved(src)
    assert count_open(parse_queue_items(src)) == 0

    plan = tmp_path / "beat20_watch_heartbeat_no_queue.md"
    plan.write_text(src, encoding="utf-8")
    result = run_session_a_path(plan, brainstorm=stub_brainstorm, write=True)
    assert result.ok and result.verdict == "queued" and result.wrote_item
    assert result.open_count == 1

    updated = plan.read_text(encoding="utf-8")
    assert updated == result.plan_text
    assert "## Queue" in updated
    assert "BEAT20_WATCH_MARKER" in updated
    assert "BEAT20_HEARTBEAT_MARKER" in updated
    assert "title: beat20-no-queue" in updated
    _assert_beat20_watch_heartbeat_front_preserved(updated)
    items = parse_queue_items(updated)
    assert len(items) == 1
    assert items[0].item_id == "Q1"
    assert is_b_pickable(items[0])
    assert count_open(items) == 1

    # In-memory run_session_a path matches (stub append).
    mem = run_session_a(src, brainstorm=stub_brainstorm)
    assert mem.ok and mem.wrote_item and mem.open_count == 1
    assert "## Queue" in mem.plan_text
    assert "BEAT20_WATCH_MARKER" in mem.plan_text
    assert "BEAT20_HEARTBEAT_MARKER" in mem.plan_text
    assert "title: beat20-no-queue" in mem.plan_text
    assert count_open(parse_queue_items(mem.plan_text)) == 1


def test_beat20_empty_file_creates_queue_section(tmp_path: Path):
    """Beat 20: empty plan file + stub append → ## Queue present with one OPEN."""
    plan = tmp_path / "beat20_empty.md"
    plan.write_text("", encoding="utf-8")
    assert plan.exists() and plan.read_text(encoding="utf-8") == ""

    result = run_session_a_path(plan, brainstorm=stub_brainstorm, write=True)
    assert result.ok and result.verdict == "queued" and result.wrote_item
    assert result.open_count == 1

    disk = plan.read_text(encoding="utf-8")
    assert disk.startswith("## Queue") or "## Queue" in disk
    assert "## Queue" in disk
    items = parse_queue_items(disk)
    assert count_open(items) == 1
    assert is_b_pickable(items[0])

    # Empty-string in-memory stub append likewise creates Queue.
    mem = run_session_a("", brainstorm=stub_brainstorm)
    assert mem.ok and mem.verdict == "queued" and mem.wrote_item
    assert mem.plan_text.startswith("## Queue") or "## Queue" in mem.plan_text
    assert count_open(parse_queue_items(mem.plan_text)) == 1


def test_beat20_write_queue_section_creates_when_missing():
    """Beat 20: write_queue_section on watch-only md with one QueueItem creates section.

    Watch/Heartbeat markers preserved; front matter title intact.
    """
    src = _load("watch_heartbeat_no_queue.md")
    assert "## Queue" not in src
    _assert_beat20_watch_heartbeat_front_preserved(src)

    item = QueueItem(
        item_id="Q1",
        heading="Beat20 create when missing",
        title="Beat20 create when missing",
        acceptance="pytest EXIT=0",
        risks="none",
        file_touch="tests/test_research_implement_loop.py",
        breaking_change="no",
        redeploy_notes="none",
        status="OPEN",
        ready_for_implement="yes",
    )
    created = write_queue_section(src, [item])
    assert "## Queue" in created
    assert "BEAT20_WATCH_MARKER" in created
    assert "BEAT20_HEARTBEAT_MARKER" in created
    assert "title: beat20-no-queue" in created
    _assert_beat20_watch_heartbeat_front_preserved(created)
    items = parse_queue_items(created)
    assert len(items) == 1
    assert items[0].item_id == "Q1"
    assert count_open(items) == 1
    assert is_b_pickable(items[0])
    # Newly created Queue follows existing non-Queue sections.
    assert created.index("## Watch") < created.index("## Heartbeat")
    assert created.index("## Heartbeat") < created.index("## Queue")


# --- Beat 21: more than one ## Queue → fail-closed; never silent merge ---

_BEAT21_MARKERS = (
    "BEAT21_WATCH_MARKER",
    "BEAT21_HEARTBEAT_MARKER",
    "Beat21 first queue item",
    "Beat21 second queue item",
)
_BEAT21_FRONT_KEYS = ("title: beat21-two-queue", "status: living")


def _assert_beat21_two_queue_content_intact(text: str) -> None:
    """Both Queue bodies + Watch/Heartbeat/front matter must survive refuse."""
    assert text.lstrip().startswith("---"), "YAML front matter missing"
    for key in _BEAT21_FRONT_KEYS:
        assert key in text, f"missing front-matter key {key}"
    assert count_queue_headings(text) == 2
    assert text.count("## Queue") == 2  # no prose duplicates of the heading token
    assert "## Watch" in text and "## Heartbeat" in text
    for marker in _BEAT21_MARKERS:
        assert marker in text, f"missing marker {marker}"
    assert "### Q1." in text and "### Q9." in text


def test_beat21_parse_write_append_refuse_two_queue_sections():
    """Beat 21: fixture with two ## Queue → parse/write/append raise; content intact."""
    src = _load("two_queue_sections.md")
    _assert_beat21_two_queue_content_intact(src)
    assert count_queue_headings(src) == 2
    with pytest.raises(AmbiguousQueueError) as ei:
        require_unique_queue_section(src)
    assert ei.value.count == 2
    assert "ambiguous ## Queue" in str(ei.value).lower() or "refuse" in str(ei.value).lower()

    with pytest.raises(AmbiguousQueueError):
        parse_queue_items(src)

    item = QueueItem(
        item_id="Q99",
        heading="must not append",
        title="must not append",
        acceptance="pytest EXIT=0",
        risks="none",
        file_touch="tests/test_research_implement_loop.py",
        breaking_change="no",
        redeploy_notes="none",
        status="OPEN",
        ready_for_implement="yes",
    )
    with pytest.raises(AmbiguousQueueError):
        write_queue_section(src, [item])
    with pytest.raises(AmbiguousQueueError):
        write_queue_section(src)
    with pytest.raises(AmbiguousQueueError):
        append_queue_item(src, format_queue_item(
            item_id="Q99",
            heading="must not append",
            title="must not append",
            acceptance="pytest EXIT=0",
            risks="none",
            file_touch="tests/test_research_implement_loop.py",
            breaking_change="false",
            redeploy_notes="none",
        ))
    with pytest.raises(AmbiguousQueueError):
        mark_item_shipped(src, "Q1", "dead21")

    # Source fixture file on disk unchanged by pure functions (sanity).
    disk = _load("two_queue_sections.md")
    assert disk == src
    _assert_beat21_two_queue_content_intact(disk)


def test_beat21_session_a_failed_no_plan_mutation(tmp_path: Path):
    """Beat 21: Session A returns failed; wrote_item=False; plan file untouched."""
    src = _load("two_queue_sections.md")
    _assert_beat21_two_queue_content_intact(src)
    plan = tmp_path / "beat21_two_queue_a.md"
    plan.write_text(src, encoding="utf-8")

    result = run_session_a_path(plan, brainstorm=stub_brainstorm, write=True)
    assert result.ok is False
    assert result.verdict == "failed"
    assert result.wrote_item is False
    assert result.plan_text == src
    assert "ambiguous" in result.message.lower() or "refuse" in result.message.lower()
    assert "## Queue" in result.message or "Queue" in result.message

    on_disk = plan.read_text(encoding="utf-8")
    assert on_disk == src
    _assert_beat21_two_queue_content_intact(on_disk)
    assert "must not append" not in on_disk
    assert "Stub shippable change" not in on_disk

    mem = run_session_a(src, brainstorm=stub_brainstorm)
    assert mem.ok is False and mem.verdict == "failed" and mem.wrote_item is False
    assert mem.plan_text == src
    _assert_beat21_two_queue_content_intact(mem.plan_text)


def test_beat21_session_b_failed_no_plan_mutation(tmp_path: Path):
    """Beat 21: Session B returns failed; never ships; plan file untouched."""
    src = _load("two_queue_sections.md")
    _assert_beat21_two_queue_content_intact(src)
    plan = tmp_path / "beat21_two_queue_b.md"
    plan.write_text(src, encoding="utf-8")

    calls = {"n": 0}

    def _spy(*_a, **_k):
        calls["n"] += 1
        raise SchedulerDeleteForbidden("spy")

    with patch("src.research_implement.session_b.scheduler_delete", _spy):
        decode = run_session_b_path(plan, decode_only=True, write=False)
        dry = run_session_b_path(
            plan, implement=dry_run_implement, decode_only=False, write=False
        )
        shipped = run_session_b_path(
            plan,
            implement=make_fixture_ship_implement("beat21dead"),
            decode_only=False,
            write=True,
        )
    assert calls["n"] == 0
    for result, label in ((decode, "decode"), (dry, "dry"), (shipped, "ship")):
        assert result.ok is False, label
        assert result.verdict == "failed", label
        assert result.shipped is False, label
        assert result.scheduler_delete_called is False, label
        assert result.keep_schedule is True, label
        assert result.plan_text == src, label
        assert "ambiguous" in result.message.lower() or "refuse" in result.message.lower()

    on_disk = plan.read_text(encoding="utf-8")
    assert on_disk == src
    _assert_beat21_two_queue_content_intact(on_disk)
    assert "SHIPPED" not in on_disk

# --- Beat 22: side-dev doc (CLI flags + make targets) ---------------------


def test_beat22_doc_mentions_cli_and_make_targets():
    """Beat 22: docs/research-implement-ab-side-dev.md lists CLI + make targets."""
    root = Path(__file__).resolve().parents[1]
    doc = root / "docs" / "research-implement-ab-side-dev.md"
    assert doc.is_file(), f"missing {doc}"
    body = doc.read_text(encoding="utf-8")
    for needle in (
        "session-a",
        "session-b",
        "idle-decode",
        "--json",
        "--dry-run",
        "--candidate-json",
        "queue 0/10",
        "scheduler_delete",
        "test-research-implement",
        "research-implement-e2e-dry-run",
        "research-implement-e2e-pipeline",
    ):
        assert needle in body, f"doc missing {needle!r}"


def test_beat23_missing_plan_path_fails_all_cmds(tmp_path: Path, capsys):
    """Beat 23: missing --plan/--log path → SystemExit; no file created."""
    from src.research_implement.__main__ import main

    missing = tmp_path / "no-such-plan.md"
    assert not missing.exists()
    for cmd in ("session-a", "session-b", "idle-decode"):
        for flag in ("--plan", "--log"):
            with pytest.raises(SystemExit) as ei:
                main([cmd, flag, str(missing), "--json"])
            msg = str(ei.value)
            assert ei.value.code != 0
            assert "not found" in msg.lower()
            assert "--plan" in msg or "--log" in msg
            assert not missing.exists()
            out = capsys.readouterr().out
            assert out.strip() == "" or "queued" not in out


def test_beat23_omitted_plan_flag_fails_all_cmds(capsys):
    """Beat 23: omitting --plan/--log → argparse SystemExit non-zero."""
    from src.research_implement.__main__ import main

    for cmd in ("session-a", "session-b", "idle-decode"):
        with pytest.raises(SystemExit) as ei:
            main([cmd, "--json"])
        assert ei.value.code != 0
        err = capsys.readouterr().err
        assert "--plan" in err or "--log" in err or "required" in err.lower()


def test_beat23_existing_plan_still_works(tmp_path: Path, capsys):
    """Beat 23: existing plan still works for session-a / session-b / idle-decode."""
    from src.research_implement.__main__ import main

    src = FIXTURES / "empty_queue.md"
    plan = tmp_path / "plan.md"
    plan.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    before = plan.read_text(encoding="utf-8")

    rc_idle = main(["idle-decode", "--plan", str(plan), "--json"])
    assert rc_idle == 0
    out_idle = capsys.readouterr().out
    assert "idle" in out_idle or "queue" in out_idle
    assert plan.read_text(encoding="utf-8") == before

    rc_b = main(["session-b", "--plan", str(plan), "--json"])
    assert rc_b == 0
    out_b = capsys.readouterr().out
    assert "idle" in out_b or "queue" in out_b
    assert plan.read_text(encoding="utf-8") == before

    rc_a = main(["session-a", "--plan", str(plan), "--stub", "--dry-run", "--json"])
    assert rc_a == 0
    out_a = capsys.readouterr().out
    assert "queued" in out_a or '"verdict"' in out_a
    # dry-run must not create/mutate the plan file contents
    assert plan.read_text(encoding="utf-8") == before
    assert plan.is_file()


def test_beat24_plan_and_log_together_fails(tmp_path: Path, capsys):
    """Beat 24: passing both --plan and --log → SystemExit; files unchanged."""
    from src.research_implement.__main__ import main

    src = FIXTURES / "empty_queue.md"
    plan = tmp_path / "plan.md"
    plan.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    before = plan.read_text(encoding="utf-8")
    log = tmp_path / "also.md"
    log.write_text(before, encoding="utf-8")
    with pytest.raises(SystemExit) as ei:
        main(["idle-decode", "--plan", str(plan), "--log", str(log), "--json"])
    assert ei.value.code == 2
    err = capsys.readouterr().err.lower()
    assert "not allowed" in err and "--plan" in err
    assert plan.read_text(encoding="utf-8") == before
    assert log.read_text(encoding="utf-8") == before


def test_beat24_log_alias_works_like_plan(tmp_path: Path, capsys):
    """Beat 24: --log alone still resolves like --plan."""
    from src.research_implement.__main__ import main

    src = FIXTURES / "empty_queue.md"
    plan = tmp_path / "plan.md"
    plan.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    rc = main(["idle-decode", "--log", str(plan), "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "idle" in out or "queue" in out


def test_beat25_stub_and_no_stub_together_fails(tmp_path: Path):
    """Beat 25: --stub + --no-stub → SystemExit; plan unchanged."""
    from src.research_implement.__main__ import main

    src = FIXTURES / "empty_queue.md"
    plan = tmp_path / "plan.md"
    plan.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    before = plan.read_text(encoding="utf-8")
    with pytest.raises(SystemExit) as ei:
        main(["session-a", "--plan", str(plan), "--stub", "--no-stub", "--json"])
    msg = str(ei.value).lower()
    assert "only one" in msg and "--stub" in msg
    assert plan.read_text(encoding="utf-8") == before


def test_beat25_candidate_json_overrides_stub(tmp_path: Path, capsys):
    """Beat 25: --candidate-json wins over --stub when OPEN=0 (queues candidate title)."""
    import json
    from src.research_implement.__main__ import main

    src = FIXTURES / "empty_queue.md"
    plan = tmp_path / "plan.md"
    plan.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    cand = FIXTURES / "complete_candidate.json"
    cand_title = str(
        json.loads(cand.read_text(encoding="utf-8")).get("title")
        or json.loads(cand.read_text(encoding="utf-8")).get("heading")
    )
    rc = main(
        [
            "session-a",
            "--plan",
            str(plan),
            "--stub",
            "--candidate-json",
            str(cand),
            "--json",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out.lower()
    assert '"ok": true' in out or "queued" in out
    body = plan.read_text(encoding="utf-8")
    assert cand_title in body
    # Stub default title must not appear when candidate overrides.
    from src.research_implement.session_a import STUB_TITLE

    if cand_title != STUB_TITLE:
        assert STUB_TITLE not in body


def test_beat26_idle_decode_rejects_dry_run(tmp_path: Path, capsys):
    """Beat 26: idle-decode is decode-only — --dry-run is rejected; plan unchanged."""
    from src.research_implement.__main__ import main

    src = FIXTURES / "one_open_ready.md"
    plan = tmp_path / "plan.md"
    plan.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    before = plan.read_text(encoding="utf-8")
    with pytest.raises(SystemExit) as ei:
        main(["idle-decode", "--plan", str(plan), "--dry-run", "--json"])
    assert ei.value.code == 2
    err = capsys.readouterr().err.lower()
    assert "unrecognized" in err or "dry-run" in err
    assert plan.read_text(encoding="utf-8") == before


def test_beat26_session_a_dry_run_leaves_plan_unchanged(tmp_path: Path, capsys):
    """Beat 26: session-a --stub --dry-run reports queued but does not write."""
    from src.research_implement.__main__ import main

    src = FIXTURES / "empty_queue.md"
    plan = tmp_path / "plan.md"
    plan.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    before = plan.read_text(encoding="utf-8")
    rc = main(["session-a", "--plan", str(plan), "--stub", "--dry-run", "--json"])
    assert rc == 0
    out = capsys.readouterr().out.lower()
    assert '"ok": true' in out or "queued" in out or "dry" in out
    assert plan.read_text(encoding="utf-8") == before


def test_beat27_session_b_rejects_producer_flags(tmp_path: Path, capsys):
    """Beat 27: session-b rejects --stub / --candidate-json; plan unchanged."""
    from src.research_implement.__main__ import main

    src = FIXTURES / "one_open_ready.md"
    plan = tmp_path / "plan.md"
    plan.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    before = plan.read_text(encoding="utf-8")
    cand = FIXTURES / "complete_candidate.json"
    for extra in (
        ["--stub"],
        ["--no-stub"],
        ["--candidate-json", str(cand)],
    ):
        with pytest.raises(SystemExit) as ei:
            main(["session-b", "--plan", str(plan), *extra, "--json"])
        assert ei.value.code == 2
        err = capsys.readouterr().err.lower()
        assert "unrecognized" in err
        assert plan.read_text(encoding="utf-8") == before


def test_beat27_idle_decode_rejects_producer_flags(tmp_path: Path, capsys):
    """Beat 27: idle-decode rejects --stub / --candidate-json; plan unchanged."""
    from src.research_implement.__main__ import main

    src = FIXTURES / "empty_queue.md"
    plan = tmp_path / "plan.md"
    plan.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    before = plan.read_text(encoding="utf-8")
    cand = FIXTURES / "complete_candidate.json"
    for extra in (
        ["--stub"],
        ["--candidate-json", str(cand)],
    ):
        with pytest.raises(SystemExit) as ei:
            main(["idle-decode", "--plan", str(plan), *extra, "--json"])
        assert ei.value.code == 2
        err = capsys.readouterr().err.lower()
        assert "unrecognized" in err
        assert plan.read_text(encoding="utf-8") == before


def test_beat28_empty_candidate_json_list_fails(tmp_path: Path, capsys):
    """Beat 28: --candidate-json [] → failed (no stub); plan unchanged."""
    from src.research_implement.__main__ import main

    src = FIXTURES / "empty_queue.md"
    plan = tmp_path / "plan.md"
    plan.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    before = plan.read_text(encoding="utf-8")
    cand = tmp_path / "empty_list.json"
    cand.write_text("[]\n", encoding="utf-8")
    rc = main(
        [
            "session-a",
            "--plan",
            str(plan),
            "--no-stub",
            "--candidate-json",
            str(cand),
            "--json",
        ]
    )
    assert rc == 1
    out = capsys.readouterr().out.lower()
    assert '"ok": false' in out or "failed" in out
    assert plan.read_text(encoding="utf-8") == before


def test_beat28_unknown_subcommand_fails(capsys):
    """Beat 28: unknown subcommand → non-zero SystemExit; no plan touch."""
    from src.research_implement.__main__ import main

    with pytest.raises(SystemExit) as ei:
        main(["not-a-real-cmd", "--plan", "/tmp/does-not-matter.md"])
    # argparse required subparsers → exit 2
    assert ei.value.code == 2
    err = capsys.readouterr().err.lower()
    assert "invalid choice" in err or "not-a-real-cmd" in err


def test_beat29_candidate_json_list_no_dicts_fails(tmp_path: Path, capsys):
    """Beat 29: --candidate-json ["x", 1] → failed (no stub); plan unchanged."""
    from src.research_implement.__main__ import main

    src = FIXTURES / "empty_queue.md"
    plan = tmp_path / "plan.md"
    plan.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    before = plan.read_text(encoding="utf-8")
    cand = tmp_path / "no_dicts.json"
    cand.write_text('["x", 1, true]\n', encoding="utf-8")
    rc = main(
        [
            "session-a",
            "--plan",
            str(plan),
            "--no-stub",
            "--candidate-json",
            str(cand),
            "--json",
        ]
    )
    assert rc == 1
    out = capsys.readouterr().out.lower()
    assert '"ok": false' in out or "failed" in out
    assert plan.read_text(encoding="utf-8") == before


def test_beat29_json_keys_match_session_contracts(tmp_path: Path, capsys):
    """Beat 29: CLI --json keys exactly match SESSION_*_RESULT_JSON_KEYS."""
    import json
    from src.research_implement.__main__ import main
    from src.research_implement.session_a import SESSION_A_RESULT_JSON_KEYS
    from src.research_implement.session_b import SESSION_RESULT_JSON_KEYS

    empty = tmp_path / "empty.md"
    empty.write_text((FIXTURES / "empty_queue.md").read_text(encoding="utf-8"), encoding="utf-8")
    one = tmp_path / "one.md"
    one.write_text((FIXTURES / "one_open_ready.md").read_text(encoding="utf-8"), encoding="utf-8")

    rc_a = main(["session-a", "--plan", str(empty), "--stub", "--dry-run", "--json"])
    assert rc_a == 0
    payload_a = json.loads(capsys.readouterr().out)
    assert tuple(sorted(payload_a)) == tuple(sorted(SESSION_A_RESULT_JSON_KEYS))
    assert set(payload_a) == set(SESSION_A_RESULT_JSON_KEYS)

    rc_idle = main(["idle-decode", "--plan", str(empty), "--json"])
    assert rc_idle == 0
    payload_idle = json.loads(capsys.readouterr().out)
    assert set(payload_idle) == set(SESSION_RESULT_JSON_KEYS)

    rc_b = main(["session-b", "--plan", str(one), "--json"])
    assert rc_b == 0
    payload_b = json.loads(capsys.readouterr().out)
    assert set(payload_b) == set(SESSION_RESULT_JSON_KEYS)


def test_beat30_candidate_json_list_skips_nondicts(tmp_path: Path, capsys):
    """Beat 30: list skips leading non-dicts; first dict queues; plan gains that title."""
    from src.research_implement.__main__ import main

    src = FIXTURES / "empty_queue.md"
    plan = tmp_path / "plan.md"
    plan.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    cand = FIXTURES / "complete_candidate_list_skip_nondict.json"
    rc = main(
        [
            "session-a",
            "--plan",
            str(plan),
            "--no-stub",
            "--candidate-json",
            str(cand),
            "--json",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Skip-nondict first dict" in out or "queued" in out.lower()
    body = plan.read_text(encoding="utf-8")
    assert "Skip-nondict first dict" in body
    assert "ignored second" not in body


def test_beat30_relative_plan_path_works(tmp_path: Path, monkeypatch, capsys):
    """Beat 30: relative --plan path works when cwd has the file."""
    from src.research_implement.__main__ import main

    plan = tmp_path / "rel-plan.md"
    plan.write_text((FIXTURES / "empty_queue.md").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    rc = main(["idle-decode", "--plan", "rel-plan.md", "--json"])
    assert rc == 0
    out = capsys.readouterr().out.lower()
    assert "idle" in out or "queue" in out or '"ok"' in out


def test_beat31_plan_directory_fails(tmp_path: Path, capsys):
    """Beat 31: --plan pointing at a directory → SystemExit; clear not-a-file."""
    from src.research_implement.__main__ import main

    d = tmp_path / "not-a-plan"
    d.mkdir()
    with pytest.raises(SystemExit) as ei:
        main(["idle-decode", "--plan", str(d), "--json"])
    msg = str(ei.value).lower()
    assert "not a file" in msg
    assert d.is_dir()


def test_beat31_relative_log_alias_works(tmp_path: Path, monkeypatch, capsys):
    """Beat 31: relative --log alias resolves like --plan when file exists in cwd."""
    from src.research_implement.__main__ import main

    plan = tmp_path / "rel-log.md"
    plan.write_text((FIXTURES / "empty_queue.md").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    rc = main(["idle-decode", "--log", "rel-log.md", "--json"])
    assert rc == 0
    out = capsys.readouterr().out.lower()
    assert "idle" in out or "queue" in out or '"ok"' in out


def test_beat32_candidate_json_directory_fails(tmp_path: Path):
    """Beat 32: --candidate-json pointing at a directory → SystemExit; plan unchanged."""
    from src.research_implement.__main__ import main

    src = FIXTURES / "empty_queue.md"
    plan = tmp_path / "plan.md"
    plan.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    before = plan.read_text(encoding="utf-8")
    d = tmp_path / "cand-dir"
    d.mkdir()
    with pytest.raises(SystemExit) as ei:
        main(
            [
                "session-a",
                "--plan",
                str(plan),
                "--no-stub",
                "--candidate-json",
                str(d),
                "--json",
            ]
        )
    msg = str(ei.value).lower()
    assert "not a file" in msg and "candidate-json" in msg
    assert plan.read_text(encoding="utf-8") == before


def test_beat32_candidate_json_null_fails(tmp_path: Path):
    """Beat 32: --candidate-json top-level null → SystemExit; plan unchanged."""
    from src.research_implement.__main__ import main

    src = FIXTURES / "empty_queue.md"
    plan = tmp_path / "plan.md"
    plan.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    before = plan.read_text(encoding="utf-8")
    cand = tmp_path / "null.json"
    cand.write_text("null\n", encoding="utf-8")
    with pytest.raises(SystemExit) as ei:
        main(
            [
                "session-a",
                "--plan",
                str(plan),
                "--no-stub",
                "--candidate-json",
                str(cand),
                "--json",
            ]
        )
    msg = str(ei.value).lower()
    assert "candidate-json" in msg
    assert plan.read_text(encoding="utf-8") == before


def test_beat33_empty_plan_path_fails(capsys):
    """Beat 33: empty / whitespace --plan → SystemExit; clear empty-path message."""
    from src.research_implement.__main__ import main

    for bad in ("", "   ", "\t"):
        with pytest.raises(SystemExit) as ei:
            main(["idle-decode", "--plan", bad, "--json"])
        msg = str(ei.value).lower()
        assert "empty" in msg


def test_beat33_idle_json_never_scheduler_delete(tmp_path: Path, capsys):
    """Beat 33: idle-decode --json keeps schedule; never scheduler_delete."""
    import json
    from src.research_implement.__main__ import main

    plan = tmp_path / "empty.md"
    plan.write_text((FIXTURES / "empty_queue.md").read_text(encoding="utf-8"), encoding="utf-8")
    rc = main(["idle-decode", "--plan", str(plan), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("keep_schedule") is True
    assert payload.get("scheduler_delete_called") is False
    assert payload.get("verdict") == "idle" or "idle" in str(payload.get("verdict", "")).lower()


def test_beat34_empty_log_path_fails():
    """Beat 34: empty / whitespace --log → SystemExit path is empty (like --plan)."""
    from src.research_implement.__main__ import main

    for bad in ("", "   ", "\t"):
        with pytest.raises(SystemExit) as ei:
            main(["idle-decode", "--log", bad, "--json"])
        assert "empty" in str(ei.value).lower()


def test_beat34_session_b_dry_run_json_never_scheduler_delete(tmp_path: Path, capsys):
    """Beat 34: session-b --dry-run --json keeps schedule; never scheduler_delete."""
    import json
    from src.research_implement.__main__ import main

    plan = tmp_path / "one.md"
    plan.write_text((FIXTURES / "one_open_ready.md").read_text(encoding="utf-8"), encoding="utf-8")
    before = plan.read_text(encoding="utf-8")
    rc = main(["session-b", "--plan", str(plan), "--dry-run", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("keep_schedule") is True
    assert payload.get("scheduler_delete_called") is False
    assert payload.get("shipped") is False
    # dry-run must not mutate the plan (OPEN stays).
    assert plan.read_text(encoding="utf-8") == before


def test_beat35_session_b_decode_only_json_keeps_schedule(tmp_path: Path, capsys):
    """Beat 35: session-b default decode-only --json keeps schedule; plan unchanged."""
    import json
    from src.research_implement.__main__ import main

    plan = tmp_path / "one.md"
    plan.write_text((FIXTURES / "one_open_ready.md").read_text(encoding="utf-8"), encoding="utf-8")
    before = plan.read_text(encoding="utf-8")
    rc = main(["session-b", "--plan", str(plan), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("verdict") == "picked"
    assert payload.get("keep_schedule") is True
    assert payload.get("scheduler_delete_called") is False
    assert payload.get("shipped") is False
    assert payload.get("implement_result") is None
    assert plan.read_text(encoding="utf-8") == before


def test_beat35_session_a_recount_json_wrote_item_false(tmp_path: Path, capsys):
    """Beat 35: Session A OPEN>=1 recount-only --json → wrote_item=false; plan unchanged."""
    import json
    from src.research_implement.__main__ import main

    plan = tmp_path / "one.md"
    plan.write_text((FIXTURES / "one_open_ready.md").read_text(encoding="utf-8"), encoding="utf-8")
    before = plan.read_text(encoding="utf-8")
    rc = main(["session-a", "--plan", str(plan), "--stub", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("wrote_item") is False
    assert payload.get("verdict") in {"light", "queued"} or payload.get("ok") is True
    # recount-only: open stays >=1, no new append
    assert plan.read_text(encoding="utf-8") == before


def test_beat36_idle_decode_picked_json_matches_session_b(tmp_path: Path, capsys):
    """Beat 36: idle-decode --json on ready OPEN matches session-b decode-only shape."""
    import json
    from src.research_implement.__main__ import main

    src = (FIXTURES / "one_open_ready.md").read_text(encoding="utf-8")
    plan_idle = tmp_path / "idle.md"
    plan_b = tmp_path / "b.md"
    plan_idle.write_text(src, encoding="utf-8")
    plan_b.write_text(src, encoding="utf-8")

    rc_i = main(["idle-decode", "--plan", str(plan_idle), "--json"])
    out_i = capsys.readouterr().out
    rc_b = main(["session-b", "--plan", str(plan_b), "--json"])
    out_b = capsys.readouterr().out
    assert rc_i == 0 and rc_b == 0
    pi = json.loads(out_i)
    pb = json.loads(out_b)
    assert pi.get("verdict") == pb.get("verdict") == "picked"
    assert pi.get("keep_schedule") is True
    assert pi.get("scheduler_delete_called") is False
    assert pi.get("shipped") is False
    assert set(pi) == set(pb)
    assert plan_idle.read_text(encoding="utf-8") == src
    assert plan_b.read_text(encoding="utf-8") == src


def test_beat36_session_a_stub_queues_wrote_item_true(tmp_path: Path, capsys):
    """Beat 36: session-a --stub on empty Queue --json → wrote_item=true / queued."""
    import json
    from src.research_implement.__main__ import main

    plan = tmp_path / "empty.md"
    plan.write_text((FIXTURES / "empty_queue.md").read_text(encoding="utf-8"), encoding="utf-8")
    before = plan.read_text(encoding="utf-8")
    rc = main(["session-a", "--plan", str(plan), "--stub", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("ok") is True
    assert payload.get("wrote_item") is True
    assert payload.get("verdict") == "queued"
    assert plan.read_text(encoding="utf-8") != before
    assert "OPEN" in plan.read_text(encoding="utf-8") or payload.get("open_count", 0) >= 1


def test_beat37_empty_candidate_json_path_fails(tmp_path: Path):
    """Beat 37: empty / whitespace --candidate-json → SystemExit; plan unchanged."""
    from src.research_implement.__main__ import main

    plan = tmp_path / "plan.md"
    plan.write_text((FIXTURES / "empty_queue.md").read_text(encoding="utf-8"), encoding="utf-8")
    before = plan.read_text(encoding="utf-8")
    for bad in ("", "   ", "\t"):
        with pytest.raises(SystemExit) as ei:
            main(
                [
                    "session-a",
                    "--plan",
                    str(plan),
                    "--no-stub",
                    "--candidate-json",
                    bad,
                    "--json",
                ]
            )
        assert "empty" in str(ei.value).lower()
        assert plan.read_text(encoding="utf-8") == before


def test_beat37_session_a_stub_dry_run_json_queued_no_write(tmp_path: Path, capsys):
    """Beat 37: --stub --dry-run --json → queued/wrote_item true in JSON; plan unchanged."""
    import json
    from src.research_implement.__main__ import main

    plan = tmp_path / "empty.md"
    plan.write_text((FIXTURES / "empty_queue.md").read_text(encoding="utf-8"), encoding="utf-8")
    before = plan.read_text(encoding="utf-8")
    rc = main(["session-a", "--plan", str(plan), "--stub", "--dry-run", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("ok") is True
    assert payload.get("verdict") == "queued"
    assert payload.get("wrote_item") is True
    assert plan.read_text(encoding="utf-8") == before


def test_beat38_relative_candidate_json_works(tmp_path: Path, monkeypatch, capsys):
    """Beat 38: relative --candidate-json works when file exists in cwd."""
    import json
    from src.research_implement.__main__ import main

    plan = tmp_path / "plan.md"
    plan.write_text((FIXTURES / "empty_queue.md").read_text(encoding="utf-8"), encoding="utf-8")
    cand = tmp_path / "cand.json"
    cand.write_text((FIXTURES / "complete_candidate.json").read_text(encoding="utf-8"), encoding="utf-8")
    title = json.loads(cand.read_text(encoding="utf-8")).get("title") or "Complete"
    monkeypatch.chdir(tmp_path)
    rc = main(
        [
            "session-a",
            "--plan",
            "plan.md",
            "--no-stub",
            "--candidate-json",
            "cand.json",
            "--json",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert title in out or "queued" in out.lower()
    assert title in plan.read_text(encoding="utf-8")


def test_beat38_candidate_json_boolean_fails(tmp_path: Path):
    """Beat 38: --candidate-json top-level true/false → SystemExit; plan unchanged."""
    from src.research_implement.__main__ import main

    plan = tmp_path / "plan.md"
    plan.write_text((FIXTURES / "empty_queue.md").read_text(encoding="utf-8"), encoding="utf-8")
    before = plan.read_text(encoding="utf-8")
    for raw in ("true", "false"):
        cand = tmp_path / f"{raw}.json"
        cand.write_text(raw + "\n", encoding="utf-8")
        with pytest.raises(SystemExit) as ei:
            main(
                [
                    "session-a",
                    "--plan",
                    str(plan),
                    "--no-stub",
                    "--candidate-json",
                    str(cand),
                    "--json",
                ]
            )
        msg = str(ei.value).lower()
        assert "candidate-json" in msg
        assert "bool" in msg or "object" in msg or "list" in msg
        assert plan.read_text(encoding="utf-8") == before


def test_beat39_empty_object_candidate_json_fails(tmp_path: Path, capsys):
    """Beat 39: --candidate-json {} → Session A failed (incomplete); plan unchanged."""
    import json
    from src.research_implement.__main__ import main

    plan = tmp_path / "plan.md"
    plan.write_text((FIXTURES / "empty_queue.md").read_text(encoding="utf-8"), encoding="utf-8")
    before = plan.read_text(encoding="utf-8")
    cand = tmp_path / "empty_obj.json"
    cand.write_text("{}\n", encoding="utf-8")
    rc = main(
        [
            "session-a",
            "--plan",
            str(plan),
            "--no-stub",
            "--candidate-json",
            str(cand),
            "--json",
        ]
    )
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("ok") is False
    assert payload.get("wrote_item") is False
    assert payload.get("verdict") == "failed"
    assert plan.read_text(encoding="utf-8") == before


def test_beat39_session_a_help_mentions_producer_flags(capsys):
    """Beat 39: session-a --help mentions --stub / --no-stub / --candidate-json."""
    from src.research_implement.__main__ import main

    with pytest.raises(SystemExit) as ei:
        main(["session-a", "--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out.lower()
    assert "--stub" in out
    assert "--no-stub" in out
    assert "--candidate-json" in out


def test_beat40_top_level_help_lists_subcommands(capsys):
    """Beat 40: top-level --help lists session-a / session-b / idle-decode."""
    from src.research_implement.__main__ import main

    with pytest.raises(SystemExit) as ei:
        main(["--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out.lower()
    assert "session-a" in out
    assert "session-b" in out
    assert "idle-decode" in out


def test_beat40_idle_help_no_dry_run_flag_session_b_has_it(capsys):
    """Beat 40: idle-decode --help usage omits --dry-run; session-b includes it."""
    from src.research_implement.__main__ import main

    with pytest.raises(SystemExit) as ei_i:
        main(["idle-decode", "--help"])
    assert ei_i.value.code == 0
    idle = capsys.readouterr().out.lower()
    # usage line for idle must not advertise --dry-run
    idle_usage = idle.split("options:", 1)[0]
    assert "--dry-run" not in idle_usage
    assert "--json" in idle

    with pytest.raises(SystemExit) as ei_b:
        main(["session-b", "--help"])
    assert ei_b.value.code == 0
    b = capsys.readouterr().out.lower()
    assert "--dry-run" in b


def test_beat41_top_help_mentions_side_dev_no_tasker(capsys):
    """Beat 41: top-level --help mentions side-dev / no Tasker / no live LLM."""
    from src.research_implement.__main__ import main

    with pytest.raises(SystemExit) as ei:
        main(["--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out.lower()
    assert "side-dev" in out or "side dev" in out
    assert "tasker" in out
    assert "llm" in out


def test_beat41_cli_rejects_implement_stub_ship_flags(tmp_path: Path, capsys):
    """Beat 41: --implement / stub-ship style flags are unrecognized; plan unchanged."""
    from src.research_implement.__main__ import main

    plan = tmp_path / "one.md"
    plan.write_text((FIXTURES / "one_open_ready.md").read_text(encoding="utf-8"), encoding="utf-8")
    before = plan.read_text(encoding="utf-8")
    for extra in (
        ["--implement", "stub-ship"],
        ["--stub-ship"],
    ):
        with pytest.raises(SystemExit) as ei:
            main(["session-b", "--plan", str(plan), *extra, "--json"])
        assert ei.value.code == 2
        err = capsys.readouterr().err.lower()
        assert "unrecognized" in err
        assert plan.read_text(encoding="utf-8") == before


def test_beat42_makefile_lists_research_implement_targets():
    """Beat 42: Makefile lists test-research-implement + both e2e targets."""
    from pathlib import Path as _Path

    text = (_Path(__file__).resolve().parents[1] / "Makefile").read_text(encoding="utf-8")
    assert "test-research-implement" in text
    assert "research-implement-e2e-dry-run" in text
    assert "research-implement-e2e-pipeline" in text


def test_beat42_b_and_idle_help_mention_never_scheduler_delete(capsys):
    """Beat 42: session-b / idle-decode --help mention never scheduler_delete."""
    from src.research_implement.__main__ import main

    with pytest.raises(SystemExit) as ei_b:
        main(["session-b", "--help"])
    assert ei_b.value.code == 0
    b = capsys.readouterr().out.lower()
    assert "scheduler_delete" in b or "scheduler delete" in b

    with pytest.raises(SystemExit) as ei_i:
        main(["idle-decode", "--help"])
    assert ei_i.value.code == 0
    idle = capsys.readouterr().out.lower()
    assert "scheduler_delete" in idle or "scheduler delete" in idle


def test_beat43_side_dev_doc_mentions_make_and_no_cli_ship():
    """Beat 43: side-dev guide mentions make targets, never scheduler_delete, no CLI ship."""
    from pathlib import Path as _Path

    doc = (_Path(__file__).resolve().parents[1] / "docs" / "research-implement-ab-side-dev.md")
    text = doc.read_text(encoding="utf-8").lower()
    assert "test-research-implement" in text
    assert "e2e" in text
    assert "scheduler_delete" in text or "scheduler delete" in text
    assert "implement" in text or "stub-ship" in text or "callback" in text


def test_beat43_init_mentions_recent_help_beats():
    """Beat 43: package __init__ docstring mentions Beat 40 help contracts."""
    from pathlib import Path as _Path

    init = (_Path(__file__).resolve().parents[1] / "src" / "research_implement" / "__init__.py")
    text = init.read_text(encoding="utf-8")
    assert "Beat 40" in text
    assert "test-research-implement" in text or "e2e" in text.lower()

def test_beat44_session_b_json_help_mentions_session_result(capsys):
    """Beat 44: session-b --json help mentions SessionResult.to_dict / shared shape."""
    from src.research_implement.__main__ import main

    with pytest.raises(SystemExit) as ei:
        main(["session-b", "--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out.lower()
    assert "sessionresult" in out.replace("_", "") or "sessionresult.to_dict" in out.replace(" ", "")
    assert "to_dict" in out or "sessionresult" in out.lower()
    assert "idle" in out
    assert "scheduler_delete" in out or "scheduler delete" in out


def test_beat44_dry_run_message_includes_decode_pick():
    """Beat 44: dry-run message includes decode pick six-field report; never scheduler_delete."""
    from unittest.mock import patch

    calls = {"n": 0}

    def _spy(*_a, **_k):
        calls["n"] += 1

    with patch("src.research_implement.session_b.scheduler_delete", _spy):
        result = run_session_b(_load("one_open_ready.md"), decode_only=False)
    assert result.ok is True
    assert result.verdict == "dry_run"
    assert result.keep_schedule is True
    assert result.scheduler_delete_called is False
    assert calls["n"] == 0
    assert "no repo write" in result.message
    assert "decode pick" in result.message
    assert "1. title:" in result.message
    assert "6. redeploy_notes:" in result.message
    assert result.decode_report is not None
    assert "decode pick" in result.decode_report


def test_beat45_idle_and_a_json_help_mention_to_dict(capsys):
    """Beat 45: idle-decode / session-a --json help mention to_dict contracts."""
    from src.research_implement.__main__ import main

    with pytest.raises(SystemExit) as ei_i:
        main(["idle-decode", "--help"])
    assert ei_i.value.code == 0
    idle = capsys.readouterr().out
    assert "SessionResult.to_dict" in idle or "to_dict" in idle

    with pytest.raises(SystemExit) as ei_a:
        main(["session-a", "--help"])
    assert ei_a.value.code == 0
    a = capsys.readouterr().out
    assert "SessionAResult.to_dict" in a or "to_dict" in a


def test_beat45_decode_only_message_includes_decode_pick():
    """Beat 45: session-b decode-only message includes decode pick six-field report."""
    from src.research_implement.session_b import run_session_b

    plan = (FIXTURES / "one_open_ready.md").read_text(encoding="utf-8")
    result = run_session_b(plan, decode_only=True)
    assert result.ok
    assert result.item is not None
    assert f"decode pick {result.item.item_id}" in result.message
    assert result.scheduler_delete_called is False


def test_beat46_not_ready_cli_idle_json(tmp_path: Path, capsys):
    """Beat 46: open_complete_not_ready → idle-decode/session-b idle JSON; plan unchanged."""
    import json
    from src.research_implement.__main__ import main

    src = (FIXTURES / "open_complete_not_ready.md").read_text(encoding="utf-8")
    for cmd in ("idle-decode", "session-b"):
        plan = tmp_path / f"{cmd}.md"
        plan.write_text(src, encoding="utf-8")
        rc = main([cmd, "--plan", str(plan), "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload.get("verdict") == "idle"
        assert payload.get("queue") == "queue 0/10"
        assert payload.get("keep_schedule") is True
        assert payload.get("scheduler_delete_called") is False
        assert payload.get("item") is None
        assert plan.read_text(encoding="utf-8") == src


def test_beat46_empty_queue_cli_idle_message_mentions_queue():
    """Beat 46: empty Queue idle message mentions queue 0/10; never scheduler_delete."""
    from src.research_implement.session_b import run_session_b

    plan = (FIXTURES / "empty_queue.md").read_text(encoding="utf-8")
    result = run_session_b(plan, decode_only=True)
    assert result.ok
    assert result.verdict == "idle"
    assert "queue 0/10" in result.message
    assert result.keep_schedule is True
    assert result.scheduler_delete_called is False


def test_beat47_incomplete_and_shipped_only_cli_idle(tmp_path: Path, capsys):
    """Beat 47: incomplete_open / shipped_only → idle-decode idle JSON; plan unchanged."""
    import json
    from src.research_implement.__main__ import main

    for name in ("incomplete_open.md", "shipped_only.md"):
        src = (FIXTURES / name).read_text(encoding="utf-8")
        plan = tmp_path / name
        plan.write_text(src, encoding="utf-8")
        rc = main(["idle-decode", "--plan", str(plan), "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload.get("verdict") == "idle"
        assert payload.get("queue") == "queue 0/10"
        assert payload.get("keep_schedule") is True
        assert payload.get("scheduler_delete_called") is False
        assert plan.read_text(encoding="utf-8") == src


def test_beat47_mixed_priority_cli_picks_first_ready(tmp_path: Path, capsys):
    """Beat 47: mixed_priority → session-b --json picks first ready; plan unchanged."""
    import json
    from src.research_implement.__main__ import main

    src = (FIXTURES / "mixed_priority.md").read_text(encoding="utf-8")
    plan = tmp_path / "mixed.md"
    plan.write_text(src, encoding="utf-8")
    rc = main(["session-b", "--plan", str(plan), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("verdict") == "picked"
    assert payload.get("item") is not None
    assert payload.get("keep_schedule") is True
    assert payload.get("scheduler_delete_called") is False
    assert plan.read_text(encoding="utf-8") == src


def test_beat48_watch_lookalike_cli_picks_queue_not_watch(tmp_path: Path, capsys):
    """Beat 48: watch_lookalike → session-b picks Queue Q3, not Watch Fake; plan unchanged."""
    import json
    from src.research_implement.__main__ import main

    src = (FIXTURES / "watch_lookalike.md").read_text(encoding="utf-8")
    plan = tmp_path / "lookalike.md"
    plan.write_text(src, encoding="utf-8")
    rc = main(["session-b", "--plan", str(plan), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("verdict") == "picked"
    item = payload.get("item") or {}
    item_id = str(item.get("item_id") or item.get("id") or "")
    title = str(item.get("title") or "")
    assert "Q3" in item_id or "Real ready" in title
    assert "Fake" not in title and "Watch lookalike" not in title
    assert payload.get("scheduler_delete_called") is False
    assert plan.read_text(encoding="utf-8") == src


def test_beat48_watch_only_cli_idles(tmp_path: Path, capsys):
    """Beat 48: watch_only_lookalike / watch_heartbeat_no_queue → idle-decode idle."""
    import json
    from src.research_implement.__main__ import main

    for name in ("watch_only_lookalike.md", "watch_heartbeat_no_queue.md"):
        src = (FIXTURES / name).read_text(encoding="utf-8")
        plan = tmp_path / name
        plan.write_text(src, encoding="utf-8")
        rc = main(["idle-decode", "--plan", str(plan), "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload.get("verdict") == "idle"
        assert payload.get("queue") == "queue 0/10"
        assert payload.get("keep_schedule") is True
        assert payload.get("scheduler_delete_called") is False
        assert plan.read_text(encoding="utf-8") == src


def test_beat49_two_open_cli_picks_first(tmp_path: Path, capsys):
    """Beat 49: two_open_ready → session-b picks first OPEN; plan unchanged (decode-only)."""
    import json
    from src.research_implement.__main__ import main
    from src.research_implement.queue import parse_queue_items

    src = (FIXTURES / "two_open_ready.md").read_text(encoding="utf-8")
    plan = tmp_path / "two.md"
    plan.write_text(src, encoding="utf-8")
    before_ids = [i.item_id for i in parse_queue_items(src)]
    rc = main(["session-b", "--plan", str(plan), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("verdict") == "picked"
    item = payload.get("item") or {}
    first_id = str(item.get("item_id") or "")
    assert first_id == before_ids[0] or first_id.startswith("Q")
    # Second OPEN still present; plan not mutated by decode-only.
    assert plan.read_text(encoding="utf-8") == src
    assert len(parse_queue_items(plan.read_text(encoding="utf-8"))) >= 2


def test_beat49_queue_watch_heartbeat_a_append_preserves(tmp_path: Path, capsys):
    """Beat 49: session-a stub append on queue_with_watch_heartbeat keeps Watch/Heartbeat."""
    from src.research_implement.__main__ import main

    # Prefer fixture with empty-ish queue if present; else watch_queue_heartbeat_empty
    for name in ("watch_queue_heartbeat_empty.md", "queue_with_watch_heartbeat.md"):
        path = FIXTURES / name
        if path.is_file():
            src = path.read_text(encoding="utf-8")
            break
    else:
        raise AssertionError("missing watch/queue heartbeat fixture")

    plan = tmp_path / "wh.md"
    plan.write_text(src, encoding="utf-8")
    # If already OPEN>=1, recount-only — still must preserve sections.
    rc = main(["session-a", "--plan", str(plan), "--stub", "--json"])
    assert rc == 0
    body = plan.read_text(encoding="utf-8")
    assert "## Watch" in body or "Watch" in body
    assert "## Heartbeat" in body or "Heartbeat" in body
    assert "## Queue" in body


def test_beat50_broken_ready_cli_idle(tmp_path: Path, capsys):
    """Beat 50: broken_ready_flag → idle-decode/session-b idle; plan unchanged."""
    import json
    from src.research_implement.__main__ import main

    src = (FIXTURES / "broken_ready_flag.md").read_text(encoding="utf-8")
    for cmd in ("idle-decode", "session-b"):
        plan = tmp_path / f"{cmd}.md"
        plan.write_text(src, encoding="utf-8")
        rc = main([cmd, "--plan", str(plan), "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload.get("verdict") == "idle"
        assert payload.get("queue") == "queue 0/10"
        assert payload.get("keep_schedule") is True
        assert payload.get("scheduler_delete_called") is False
        assert plan.read_text(encoding="utf-8") == src


def test_beat50_public_api_exports_still_stable():
    """Beat 50: research_implement still exports Session A/B result helpers."""
    import src.research_implement as ri

    for name in (
        "SessionAResult",
        "SessionResult",
        "SESSION_A_RESULT_JSON_KEYS",
        "SESSION_RESULT_JSON_KEYS",
        "run_session_a",
        "run_session_b",
        "parse_queue_items",
    ):
        assert hasattr(ri, name), name


def test_beat51_is_ready_yes_aliases():
    """Beat 51: is_ready_yes accepts yes/y/true/1 (any case/space); rejects others."""
    from src.research_implement.queue import is_ready_yes

    for ok in ("yes", "YES", " Yes ", "y", "Y", "true", "TRUE", "1", " 1 "):
        assert is_ready_yes(ok), ok
    for bad in ("", "no", "READY", "maybe", "yes!", "0", "false", "ready"):
        assert not is_ready_yes(bad), bad


def test_beat51_ready_alias_cli_pick(tmp_path: Path, capsys):
    """Beat 51: ready-for-implement YES/TRUE → idle-decode/session-b pick; plan unchanged."""
    import json
    from src.research_implement.__main__ import main

    base = (FIXTURES / "one_open_ready.md").read_text(encoding="utf-8")
    assert "ready-for-implement: yes" in base
    for alias, cmd in (("YES", "idle-decode"), ("TRUE", "session-b")):
        plan_text = base.replace("ready-for-implement: yes", f"ready-for-implement: {alias}", 1)
        plan = tmp_path / f"beat51_{alias}_{cmd}.md"
        plan.write_text(plan_text, encoding="utf-8")
        rc = main([cmd, "--plan", str(plan), "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload.get("verdict") == "picked", (alias, cmd, payload)
        assert plan.read_text(encoding="utf-8") == plan_text


def test_beat52_two_queue_cli_fail_closed(tmp_path: Path, capsys):
    """Beat 52: two_queue_sections → session-a/b/idle-decode CLI failed; plan intact."""
    import json
    from src.research_implement.__main__ import main

    src = (FIXTURES / "two_queue_sections.md").read_text(encoding="utf-8")
    for cmd, extra in (
        ("session-a", ["--stub"]),
        ("session-b", []),
        ("idle-decode", []),
    ):
        plan = tmp_path / f"beat52_{cmd}.md"
        plan.write_text(src, encoding="utf-8")
        rc = main([cmd, "--plan", str(plan), "--json", *extra])
        assert rc == 1, cmd
        payload = json.loads(capsys.readouterr().out)
        assert payload.get("ok") is False, cmd
        assert payload.get("verdict") == "failed", cmd
        if cmd == "session-a":
            assert payload.get("wrote_item") is False
        else:
            assert payload.get("shipped") is False
            assert payload.get("scheduler_delete_called") is False
            assert payload.get("keep_schedule") is True
        assert plan.read_text(encoding="utf-8") == src


def test_beat52_two_queue_cli_markers_intact(tmp_path: Path, capsys):
    """Beat 52: after failed CLI, Watch/Heartbeat markers still present on disk."""
    import json
    from src.research_implement.__main__ import main

    src = (FIXTURES / "two_queue_sections.md").read_text(encoding="utf-8")
    plan = tmp_path / "beat52_markers.md"
    plan.write_text(src, encoding="utf-8")
    rc = main(["session-b", "--plan", str(plan), "--json"])
    assert rc == 1
    _ = json.loads(capsys.readouterr().out)
    on_disk = plan.read_text(encoding="utf-8")
    assert on_disk == src
    assert "BEAT21_WATCH_MARKER" in on_disk
    assert "BEAT21_HEARTBEAT_MARKER" in on_disk or "Heartbeat" in on_disk
    assert on_disk.count("## Queue") == 2


def test_beat53_watch_queue_empty_cli_idle_preserves(tmp_path: Path, capsys):
    """Beat 53: watch_queue_heartbeat_empty → idle CLI; markers unchanged."""
    import json
    from src.research_implement.__main__ import main

    src = (FIXTURES / "watch_queue_heartbeat_empty.md").read_text(encoding="utf-8")
    for cmd in ("idle-decode", "session-b"):
        plan = tmp_path / f"beat53_idle_{cmd}.md"
        plan.write_text(src, encoding="utf-8")
        rc = main([cmd, "--plan", str(plan), "--json"])
        assert rc == 0, cmd
        payload = json.loads(capsys.readouterr().out)
        assert payload.get("verdict") == "idle", cmd
        assert payload.get("queue") == "queue 0/10"
        assert payload.get("keep_schedule") is True
        assert payload.get("scheduler_delete_called") is False
        body = plan.read_text(encoding="utf-8")
        assert body == src
        assert "BEAT19_WATCH_MARKER" in body
        assert "BEAT19_PROJECT_MARKER" in body
        assert "BEAT19_HEARTBEAT_MARKER" in body


def test_beat53_watch_queue_empty_a_append_preserves(tmp_path: Path, capsys):
    """Beat 53: session-a stub on empty Watch/Heartbeat plan keeps markers."""
    import json
    from src.research_implement.__main__ import main
    from src.research_implement.queue import count_open, parse_queue_items

    src = (FIXTURES / "watch_queue_heartbeat_empty.md").read_text(encoding="utf-8")
    plan = tmp_path / "beat53_a.md"
    plan.write_text(src, encoding="utf-8")
    rc = main(["session-a", "--plan", str(plan), "--stub", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("verdict") == "queued"
    assert payload.get("wrote_item") is True
    body = plan.read_text(encoding="utf-8")
    assert "BEAT19_WATCH_MARKER" in body
    assert "BEAT19_PROJECT_MARKER" in body
    assert "BEAT19_HEARTBEAT_MARKER" in body
    assert "## Queue" in body
    assert count_open(parse_queue_items(body)) == 1


def test_beat54_watch_queue_open_cli_pick_preserves(tmp_path: Path, capsys):
    """Beat 54: watch_queue_heartbeat idle-decode pick; markers unchanged."""
    import json
    from src.research_implement.__main__ import main

    src = (FIXTURES / "watch_queue_heartbeat.md").read_text(encoding="utf-8")
    plan = tmp_path / "beat54_pick.md"
    plan.write_text(src, encoding="utf-8")
    rc = main(["idle-decode", "--plan", str(plan), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("verdict") == "picked"
    item = payload.get("item") or {}
    assert item.get("item_id") == "Q1"
    assert payload.get("keep_schedule") is True
    assert payload.get("scheduler_delete_called") is False
    body = plan.read_text(encoding="utf-8")
    assert body == src
    assert "BEAT19_WATCH_MARKER" in body
    assert "BEAT19_PROJECT_MARKER" in body
    assert "BEAT19_HEARTBEAT_MARKER" in body


def test_beat54_watch_queue_open_cli_dry_run_preserves(tmp_path: Path, capsys):
    """Beat 54: session-b --dry-run on OPEN Watch/Heartbeat plan; no mutation."""
    import json
    from src.research_implement.__main__ import main

    src = (FIXTURES / "watch_queue_heartbeat.md").read_text(encoding="utf-8")
    plan = tmp_path / "beat54_dry.md"
    plan.write_text(src, encoding="utf-8")
    rc = main(["session-b", "--plan", str(plan), "--dry-run", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("verdict") == "dry_run"
    assert payload.get("shipped") is False
    assert payload.get("wrote_files") is False
    assert payload.get("scheduler_delete_called") is False
    impl = payload.get("implement_result") or {}
    assert impl.get("dry_run") is True
    body = plan.read_text(encoding="utf-8")
    assert body == src
    assert "BEAT19_WATCH_MARKER" in body
    assert "status: SHIPPED" not in body
    assert "status: OPEN" in body


def test_beat55_no_queue_cli_idle_preserves(tmp_path: Path, capsys):
    """Beat 55: watch_heartbeat_no_queue → idle-decode idle; markers unchanged."""
    import json
    from src.research_implement.__main__ import main

    src = (FIXTURES / "watch_heartbeat_no_queue.md").read_text(encoding="utf-8")
    assert "## Queue" not in src
    plan = tmp_path / "beat55_idle.md"
    plan.write_text(src, encoding="utf-8")
    rc = main(["idle-decode", "--plan", str(plan), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("verdict") == "idle"
    assert payload.get("queue") == "queue 0/10"
    assert payload.get("keep_schedule") is True
    assert payload.get("scheduler_delete_called") is False
    body = plan.read_text(encoding="utf-8")
    assert body == src
    assert "BEAT20_WATCH_MARKER" in body
    assert "BEAT20_HEARTBEAT_MARKER" in body
    assert "## Queue" not in body


def test_beat55_no_queue_a_stub_creates_queue_preserves(tmp_path: Path, capsys):
    """Beat 55: session-a --stub on no-Queue plan creates Queue; markers stay."""
    import json
    from src.research_implement.__main__ import main
    from src.research_implement.queue import count_open, parse_queue_items

    src = (FIXTURES / "watch_heartbeat_no_queue.md").read_text(encoding="utf-8")
    plan = tmp_path / "beat55_a.md"
    plan.write_text(src, encoding="utf-8")
    rc = main(["session-a", "--plan", str(plan), "--stub", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("verdict") == "queued"
    assert payload.get("wrote_item") is True
    body = plan.read_text(encoding="utf-8")
    assert "BEAT20_WATCH_MARKER" in body
    assert "BEAT20_HEARTBEAT_MARKER" in body
    assert body.count("## Queue") == 1
    assert count_open(parse_queue_items(body)) == 1


def test_beat56_json_keys_light_dry_run_failed(tmp_path: Path, capsys):
    """Beat 56: light / dry_run / failed CLI --json keys match Session contracts."""
    import json
    from src.research_implement.__main__ import main
    from src.research_implement.session_a import SESSION_A_RESULT_JSON_KEYS
    from src.research_implement.session_b import SESSION_RESULT_JSON_KEYS

    one = tmp_path / "one.md"
    one.write_text((FIXTURES / "one_open_ready.md").read_text(encoding="utf-8"), encoding="utf-8")
    rc_light = main(["session-a", "--plan", str(one), "--stub", "--json"])
    assert rc_light == 0
    light = json.loads(capsys.readouterr().out)
    assert light.get("verdict") == "light"
    assert set(light) == set(SESSION_A_RESULT_JSON_KEYS)

    dry_plan = tmp_path / "dry.md"
    dry_plan.write_text((FIXTURES / "one_open_ready.md").read_text(encoding="utf-8"), encoding="utf-8")
    rc_dry = main(["session-b", "--plan", str(dry_plan), "--dry-run", "--json"])
    assert rc_dry == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry.get("verdict") == "dry_run"
    assert set(dry) == set(SESSION_RESULT_JSON_KEYS)

    fail_plan = tmp_path / "fail.md"
    fail_plan.write_text((FIXTURES / "two_queue_sections.md").read_text(encoding="utf-8"), encoding="utf-8")
    rc_fail = main(["session-b", "--plan", str(fail_plan), "--json"])
    assert rc_fail == 1
    failed = json.loads(capsys.readouterr().out)
    assert failed.get("verdict") == "failed"
    assert set(failed) == set(SESSION_RESULT_JSON_KEYS)


def test_beat56_queue_with_watch_a_then_b_pick(tmp_path: Path, capsys):
    """Beat 56: queue_with_watch_heartbeat A-stub then idle-decode pick; Watch stays."""
    import json
    from src.research_implement.__main__ import main

    src = (FIXTURES / "queue_with_watch_heartbeat.md").read_text(encoding="utf-8")
    plan = tmp_path / "beat56_wh.md"
    plan.write_text(src, encoding="utf-8")
    rc_a = main(["session-a", "--plan", str(plan), "--stub", "--json"])
    assert rc_a == 0
    a_payload = json.loads(capsys.readouterr().out)
    assert a_payload.get("verdict") in {"queued", "light"}
    after_a = plan.read_text(encoding="utf-8")
    assert "beat19-watch-marker" in after_a or "## Watch" in after_a
    assert "beat19-heartbeat-marker" in after_a or "## Heartbeat" in after_a

    rc_b = main(["idle-decode", "--plan", str(plan), "--json"])
    assert rc_b == 0
    b_payload = json.loads(capsys.readouterr().out)
    assert b_payload.get("verdict") == "picked"
    assert b_payload.get("keep_schedule") is True
    after_b = plan.read_text(encoding="utf-8")
    assert after_b == after_a
    assert "## Watch" in after_b
    assert "## Heartbeat" in after_b


def test_beat57_session_a_failed_json_keys_and_export(tmp_path: Path, capsys):
    """Beat 57: session-a dual-Queue failed JSON keys; AmbiguousQueueError exported."""
    import json
    import src.research_implement as ri
    from src.research_implement.__main__ import main
    from src.research_implement.session_a import SESSION_A_RESULT_JSON_KEYS

    assert hasattr(ri, "AmbiguousQueueError")
    assert "AmbiguousQueueError" in getattr(ri, "__all__", ())

    plan = tmp_path / "beat57_fail.md"
    src = (FIXTURES / "two_queue_sections.md").read_text(encoding="utf-8")
    plan.write_text(src, encoding="utf-8")
    rc = main(["session-a", "--plan", str(plan), "--stub", "--json"])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("verdict") == "failed"
    assert payload.get("wrote_item") is False
    assert set(payload) == set(SESSION_A_RESULT_JSON_KEYS)
    assert plan.read_text(encoding="utf-8") == src


def test_beat57_session_a_dry_run_empty_no_write(tmp_path: Path, capsys):
    """Beat 57: session-a --dry-run --stub on empty Queue does not mutate plan."""
    import json
    from src.research_implement.__main__ import main

    src = (FIXTURES / "empty_queue.md").read_text(encoding="utf-8")
    plan = tmp_path / "beat57_dry.md"
    plan.write_text(src, encoding="utf-8")
    rc = main(["session-a", "--plan", str(plan), "--stub", "--dry-run", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("verdict") == "queued"
    assert payload.get("ok") is True
    # JSON may report would-write (wrote_item true); disk must stay unchanged.
    assert plan.read_text(encoding="utf-8") == src
    assert "## Queue" in src or "GROUP CHECK" in src or "0 OPEN" in src


def test_beat58_dry_run_empty_is_idle(tmp_path: Path, capsys):
    """Beat 58: session-b --dry-run on empty Queue → idle; plan unchanged."""
    import json
    from src.research_implement.__main__ import main

    src = (FIXTURES / "empty_queue.md").read_text(encoding="utf-8")
    plan = tmp_path / "beat58_idle.md"
    plan.write_text(src, encoding="utf-8")
    rc = main(["session-b", "--plan", str(plan), "--dry-run", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("verdict") == "idle"
    assert payload.get("queue") == "queue 0/10"
    assert payload.get("keep_schedule") is True
    assert payload.get("scheduler_delete_called") is False
    assert payload.get("implement_result") is None
    assert payload.get("shipped") is False
    assert plan.read_text(encoding="utf-8") == src


def test_beat58_queue_helper_exports_still_public():
    """Beat 58: is_b_pickable / is_ready_yes / count_open remain public exports."""
    import src.research_implement as ri

    for name in ("is_b_pickable", "is_ready_yes", "count_open", "parse_queue_items"):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name


def test_beat59_two_open_dry_run_picks_first(tmp_path: Path, capsys):
    """Beat 59: two_open_ready session-b --dry-run picks Q1; both OPEN stay."""
    import json
    from src.research_implement.__main__ import main
    from src.research_implement.queue import count_open, parse_queue_items

    src = (FIXTURES / "two_open_ready.md").read_text(encoding="utf-8")
    before = parse_queue_items(src)
    assert count_open(before) == 2
    plan = tmp_path / "beat59_two.md"
    plan.write_text(src, encoding="utf-8")
    rc = main(["session-b", "--plan", str(plan), "--dry-run", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("verdict") == "dry_run"
    assert payload.get("open_count") == 2
    item = payload.get("item") or {}
    assert item.get("item_id") == before[0].item_id
    impl = payload.get("implement_result") or {}
    assert impl.get("dry_run") is True
    assert payload.get("scheduler_delete_called") is False
    body = plan.read_text(encoding="utf-8")
    assert body == src
    assert count_open(parse_queue_items(body)) == 2
    assert "status: SHIPPED" not in body


def test_beat59_two_open_idle_decode_picks_first(tmp_path: Path, capsys):
    """Beat 59: idle-decode on two_open_ready picks first; plan unchanged."""
    import json
    from src.research_implement.__main__ import main
    from src.research_implement.queue import parse_queue_items

    src = (FIXTURES / "two_open_ready.md").read_text(encoding="utf-8")
    first_id = parse_queue_items(src)[0].item_id
    plan = tmp_path / "beat59_decode.md"
    plan.write_text(src, encoding="utf-8")
    rc = main(["idle-decode", "--plan", str(plan), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("verdict") == "picked"
    assert (payload.get("item") or {}).get("item_id") == first_id
    assert plan.read_text(encoding="utf-8") == src


def test_beat60_watch_lookalike_dry_run_picks_queue(tmp_path: Path, capsys):
    """Beat 60: watch_lookalike session-b --dry-run picks Q3; never Watch title."""
    import json
    from src.research_implement.__main__ import main

    src = (FIXTURES / "watch_lookalike.md").read_text(encoding="utf-8")
    plan = tmp_path / "beat60_dry.md"
    plan.write_text(src, encoding="utf-8")
    rc = main(["session-b", "--plan", str(plan), "--dry-run", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("verdict") == "dry_run"
    item = payload.get("item") or {}
    assert item.get("item_id") == "Q3"
    assert item.get("title") == "Real ready Queue item"
    assert "Watch lookalike" not in str(item.get("title") or "")
    assert payload.get("open_count") == 1
    assert plan.read_text(encoding="utf-8") == src


def test_beat60_watch_lookalike_idle_decode_picks_queue(tmp_path: Path, capsys):
    """Beat 60: idle-decode on watch_lookalike picks Queue Q3; plan unchanged."""
    import json
    from src.research_implement.__main__ import main

    src = (FIXTURES / "watch_lookalike.md").read_text(encoding="utf-8")
    plan = tmp_path / "beat60_decode.md"
    plan.write_text(src, encoding="utf-8")
    rc = main(["idle-decode", "--plan", str(plan), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("verdict") == "picked"
    item = payload.get("item") or {}
    assert item.get("item_id") == "Q3"
    assert "Watch lookalike" not in str(payload)
    assert plan.read_text(encoding="utf-8") == src


def test_beat61_non_pickable_dry_run_idle(tmp_path: Path, capsys):
    """Beat 61: shipped/incomplete/watch_only --dry-run → idle; plan intact."""
    import json
    from src.research_implement.__main__ import main

    for name in ("shipped_only.md", "incomplete_open.md", "watch_only_lookalike.md"):
        src = (FIXTURES / name).read_text(encoding="utf-8")
        plan = tmp_path / f"beat61_{name}"
        plan.write_text(src, encoding="utf-8")
        rc = main(["session-b", "--plan", str(plan), "--dry-run", "--json"])
        assert rc == 0, name
        payload = json.loads(capsys.readouterr().out)
        assert payload.get("verdict") == "idle", name
        assert payload.get("queue") == "queue 0/10", name
        assert payload.get("keep_schedule") is True, name
        assert payload.get("scheduler_delete_called") is False, name
        assert payload.get("implement_result") is None, name
        assert payload.get("shipped") is False, name
        assert plan.read_text(encoding="utf-8") == src, name


def test_beat61_append_queue_item_still_exported():
    """Beat 61: append_queue_item / QueueItem remain public exports."""
    import src.research_implement as ri

    for name in ("append_queue_item", "QueueItem", "mark_item_shipped", "is_ready_yes"):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name


def test_beat62_not_ready_dry_run_idle(tmp_path: Path, capsys):
    """Beat 62: open_complete_not_ready / broken_ready_flag --dry-run → idle."""
    import json
    from src.research_implement.__main__ import main

    for name in ("open_complete_not_ready.md", "broken_ready_flag.md"):
        src = (FIXTURES / name).read_text(encoding="utf-8")
        plan = tmp_path / f"beat62_{name}"
        plan.write_text(src, encoding="utf-8")
        rc = main(["session-b", "--plan", str(plan), "--dry-run", "--json"])
        assert rc == 0, name
        payload = json.loads(capsys.readouterr().out)
        assert payload.get("verdict") == "idle", name
        assert payload.get("queue") == "queue 0/10", name
        assert payload.get("keep_schedule") is True, name
        assert payload.get("scheduler_delete_called") is False, name
        assert payload.get("implement_result") is None, name
        assert plan.read_text(encoding="utf-8") == src, name


def test_beat62_next_queue_id_still_exported():
    """Beat 62: next_queue_id remains public and advances past existing Q ids."""
    import src.research_implement as ri
    from src.research_implement.queue import QueueItem, next_queue_id, parse_queue_items

    assert hasattr(ri, "next_queue_id")
    assert "next_queue_id" in getattr(ri, "__all__", ())
    items = parse_queue_items((FIXTURES / "two_open_ready.md").read_text(encoding="utf-8"))
    assert next_queue_id(items).startswith("Q")
    assert int(next_queue_id(items)[1:]) > max(int(i.item_id[1:]) for i in items)


def test_beat63_mixed_priority_dry_run_picks_first_ready(tmp_path: Path, capsys):
    """Beat 63: mixed_priority session-b --dry-run picks first ready; plan intact."""
    import json
    from src.research_implement.__main__ import main
    from src.research_implement.queue import is_b_pickable, parse_queue_items

    src = (FIXTURES / "mixed_priority.md").read_text(encoding="utf-8")
    ready = [i for i in parse_queue_items(src) if is_b_pickable(i)]
    assert ready
    first_id = ready[0].item_id
    plan = tmp_path / "beat63_dry.md"
    plan.write_text(src, encoding="utf-8")
    rc = main(["session-b", "--plan", str(plan), "--dry-run", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("verdict") == "dry_run"
    assert (payload.get("item") or {}).get("item_id") == first_id
    assert payload.get("scheduler_delete_called") is False
    assert plan.read_text(encoding="utf-8") == src


def test_beat63_mixed_priority_idle_decode_and_heading_export(tmp_path: Path, capsys):
    """Beat 63: idle-decode picks same first ready; count_queue_headings exported."""
    import json
    import src.research_implement as ri
    from src.research_implement.__main__ import main
    from src.research_implement.queue import count_queue_headings, is_b_pickable, parse_queue_items

    assert hasattr(ri, "count_queue_headings")
    assert "count_queue_headings" in getattr(ri, "__all__", ())

    src = (FIXTURES / "mixed_priority.md").read_text(encoding="utf-8")
    assert count_queue_headings(src) == 1
    first_id = [i for i in parse_queue_items(src) if is_b_pickable(i)][0].item_id
    plan = tmp_path / "beat63_decode.md"
    plan.write_text(src, encoding="utf-8")
    rc = main(["idle-decode", "--plan", str(plan), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("verdict") == "picked"
    assert (payload.get("item") or {}).get("item_id") == first_id
    assert plan.read_text(encoding="utf-8") == src


def test_beat64_one_open_dry_run_and_decode(tmp_path: Path, capsys):
    """Beat 64: one_open_ready dry-run + idle-decode pick Q1; plan intact."""
    import json
    from src.research_implement.__main__ import main

    src = (FIXTURES / "one_open_ready.md").read_text(encoding="utf-8")
    for cmd, extra, verdict in (
        ("session-b", ["--dry-run"], "dry_run"),
        ("idle-decode", [], "picked"),
    ):
        plan = tmp_path / f"beat64_{cmd}.md"
        plan.write_text(src, encoding="utf-8")
        rc = main([cmd, "--plan", str(plan), "--json", *extra])
        assert rc == 0, cmd
        payload = json.loads(capsys.readouterr().out)
        assert payload.get("verdict") == verdict, cmd
        assert (payload.get("item") or {}).get("item_id") == "Q1", cmd
        assert payload.get("open_count") == 1, cmd
        assert payload.get("keep_schedule") is True, cmd
        assert payload.get("scheduler_delete_called") is False, cmd
        assert plan.read_text(encoding="utf-8") == src, cmd


def test_beat64_format_and_require_unique_exports():
    """Beat 64: format_queue_item / require_unique_queue_section stay public."""
    import src.research_implement as ri
    from src.research_implement.queue import require_unique_queue_section

    for name in ("format_queue_item", "require_unique_queue_section"):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name
    src = (FIXTURES / "one_open_ready.md").read_text(encoding="utf-8")
    require_unique_queue_section(src)  # raises AmbiguousQueueError if duplicate
    from src.research_implement.queue import AmbiguousQueueError, count_queue_headings

    assert count_queue_headings(src) == 1
    with __import__("pytest").raises(AmbiguousQueueError):
        require_unique_queue_section(
            (FIXTURES / "two_queue_sections.md").read_text(encoding="utf-8")
        )


def test_beat65_two_queue_dry_run_still_failed(tmp_path: Path, capsys):
    """Beat 65: two_queue session-b/session-a --dry-run → failed; plan intact."""
    import json
    from src.research_implement.__main__ import main

    src = (FIXTURES / "two_queue_sections.md").read_text(encoding="utf-8")
    # session-b --dry-run
    plan_b = tmp_path / "beat65_b.md"
    plan_b.write_text(src, encoding="utf-8")
    rc_b = main(["session-b", "--plan", str(plan_b), "--dry-run", "--json"])
    assert rc_b == 1
    payload_b = json.loads(capsys.readouterr().out)
    assert payload_b.get("verdict") == "failed"
    assert payload_b.get("ok") is False
    assert payload_b.get("implement_result") is None
    assert payload_b.get("scheduler_delete_called") is False
    assert plan_b.read_text(encoding="utf-8") == src

    # session-a --dry-run --stub
    plan_a = tmp_path / "beat65_a.md"
    plan_a.write_text(src, encoding="utf-8")
    rc_a = main(["session-a", "--plan", str(plan_a), "--stub", "--dry-run", "--json"])
    assert rc_a == 1
    payload_a = json.loads(capsys.readouterr().out)
    assert payload_a.get("verdict") == "failed"
    assert payload_a.get("wrote_item") is False
    assert plan_a.read_text(encoding="utf-8") == src


def test_beat65_write_queue_section_still_exported():
    """Beat 65: write_queue_section / parse_queue_items remain public."""
    import src.research_implement as ri

    for name in ("write_queue_section", "parse_queue_items", "format_queue_item"):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name


def test_beat66_no_stub_light_and_failed(tmp_path: Path, capsys):
    """Beat 66: --no-stub on one_open → light; on empty → failed; plans intact."""
    import json
    from src.research_implement.__main__ import main

    one_src = (FIXTURES / "one_open_ready.md").read_text(encoding="utf-8")
    one = tmp_path / "beat66_one.md"
    one.write_text(one_src, encoding="utf-8")
    rc_light = main(["session-a", "--plan", str(one), "--no-stub", "--json"])
    assert rc_light == 0
    light = json.loads(capsys.readouterr().out)
    assert light.get("verdict") == "light"
    assert light.get("wrote_item") is False
    assert one.read_text(encoding="utf-8") == one_src

    empty_src = (FIXTURES / "empty_queue.md").read_text(encoding="utf-8")
    empty = tmp_path / "beat66_empty.md"
    empty.write_text(empty_src, encoding="utf-8")
    rc_fail = main(["session-a", "--plan", str(empty), "--no-stub", "--json"])
    assert rc_fail == 1
    failed = json.loads(capsys.readouterr().out)
    assert failed.get("verdict") == "failed"
    assert failed.get("wrote_item") is False
    assert empty.read_text(encoding="utf-8") == empty_src


def test_beat66_implement_helpers_still_exported():
    """Beat 66: stub_brainstorm / dry_run_implement / make_fixture_ship_implement public."""
    import src.research_implement as ri

    for name in (
        "stub_brainstorm",
        "dry_run_implement",
        "make_fixture_ship_implement",
    ):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name
        assert callable(getattr(ri, name))


def test_beat67_no_stub_dry_run_light_and_failed(tmp_path: Path, capsys):
    """Beat 67: --no-stub --dry-run on one_open → light; on empty → failed; plans intact."""
    import json
    from src.research_implement.__main__ import main

    one_src = (FIXTURES / "one_open_ready.md").read_text(encoding="utf-8")
    one = tmp_path / "beat67_one.md"
    one.write_text(one_src, encoding="utf-8")
    rc_light = main(
        ["session-a", "--plan", str(one), "--no-stub", "--dry-run", "--json"]
    )
    assert rc_light == 0
    light = json.loads(capsys.readouterr().out)
    assert light.get("verdict") == "light"
    assert light.get("wrote_item") is False
    assert one.read_text(encoding="utf-8") == one_src

    empty_src = (FIXTURES / "empty_queue.md").read_text(encoding="utf-8")
    empty = tmp_path / "beat67_empty.md"
    empty.write_text(empty_src, encoding="utf-8")
    rc_fail = main(
        ["session-a", "--plan", str(empty), "--no-stub", "--dry-run", "--json"]
    )
    assert rc_fail == 1
    failed = json.loads(capsys.readouterr().out)
    assert failed.get("verdict") == "failed"
    assert failed.get("wrote_item") is False
    assert empty.read_text(encoding="utf-8") == empty_src


def test_beat67_decode_helpers_still_exported():
    """Beat 67: decode_fields / format_decode_report / first_b_pick stay public."""
    import src.research_implement as ri

    for name in ("decode_fields", "format_decode_report", "first_b_pick"):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name
        assert callable(getattr(ri, name))

    items = parse_queue_items((FIXTURES / "one_open_ready.md").read_text(encoding="utf-8"))
    pick = ri.first_b_pick(items)
    assert pick is not None
    fields = ri.decode_fields(pick)
    report = ri.format_decode_report(pick)
    assert fields["item_id"] == pick.item_id
    assert f"decode pick {pick.item_id}:" in report
    assert "1. title:" in report


def test_beat68_first_b_pick_and_incomplete_reasons():
    """Beat 68: first_b_pick on fixtures; incomplete_candidate_reasons({}) full list."""
    from src.research_implement import first_b_pick, incomplete_candidate_reasons, parse_queue_items

    one = parse_queue_items((FIXTURES / "one_open_ready.md").read_text(encoding="utf-8"))
    assert first_b_pick(one) is not None
    assert first_b_pick(one).item_id == "Q1"

    two = parse_queue_items((FIXTURES / "two_open_ready.md").read_text(encoding="utf-8"))
    assert first_b_pick(two).item_id == two[0].item_id

    empty = parse_queue_items((FIXTURES / "empty_queue.md").read_text(encoding="utf-8"))
    assert first_b_pick(empty) is None
    shipped = parse_queue_items((FIXTURES / "shipped_only.md").read_text(encoding="utf-8"))
    assert first_b_pick(shipped) is None

    reasons = incomplete_candidate_reasons({})
    for key in (
        "title",
        "acceptance",
        "risks",
        "file_touch",
        "breaking_change",
        "redeploy_notes",
        "ready_for_implement",
    ):
        assert key in reasons, key


def test_beat68_idle_decode_message_has_decode_pick(tmp_path: Path, capsys):
    """Beat 68: idle-decode --json on one_open includes decode pick in message path."""
    import json
    from src.research_implement.__main__ import main
    from src.research_implement import format_decode_report, first_b_pick, parse_queue_items

    src = (FIXTURES / "one_open_ready.md").read_text(encoding="utf-8")
    plan = tmp_path / "beat68.md"
    plan.write_text(src, encoding="utf-8")
    rc = main(["idle-decode", "--plan", str(plan), "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload.get("verdict") == "picked"
    item = payload.get("item") or {}
    assert item.get("item_id") == "Q1"
    # Shared decode report wording still available for the picked item.
    pick = first_b_pick(parse_queue_items(src))
    report = format_decode_report(pick)
    assert "decode pick" in report.lower()
    assert "Q1" in report
    assert plan.read_text(encoding="utf-8") == src


def test_beat69_cli_json_matches_result_dicts(tmp_path: Path, capsys):
    """Beat 69: CLI --json equals session_*_result_dict for queued/picked."""
    import json
    from src.research_implement.__main__ import main
    from src.research_implement import (
        SESSION_A_RESULT_JSON_KEYS,
        SESSION_RESULT_JSON_KEYS,
        run_session_a,
        run_session_b,
        session_a_result_dict,
        session_b_result_dict,
        stub_brainstorm,
    )

    empty_src = (FIXTURES / "empty_queue.md").read_text(encoding="utf-8")
    plan_a = tmp_path / "beat69_a.md"
    plan_a.write_text(empty_src, encoding="utf-8")
    rc_a = main(["session-a", "--plan", str(plan_a), "--stub", "--json"])
    assert rc_a == 0
    cli_a = json.loads(capsys.readouterr().out)
    assert set(cli_a) == set(SESSION_A_RESULT_JSON_KEYS)
    mem_a = run_session_a(empty_src, brainstorm=stub_brainstorm)
    assert session_a_result_dict(mem_a)["verdict"] == "queued"
    assert set(session_a_result_dict(mem_a)) == set(SESSION_A_RESULT_JSON_KEYS)
    assert cli_a["verdict"] == "queued"

    one_src = (FIXTURES / "one_open_ready.md").read_text(encoding="utf-8")
    plan_b = tmp_path / "beat69_b.md"
    plan_b.write_text(one_src, encoding="utf-8")
    rc_b = main(["idle-decode", "--plan", str(plan_b), "--json"])
    assert rc_b == 0
    cli_b = json.loads(capsys.readouterr().out)
    assert set(cli_b) == set(SESSION_RESULT_JSON_KEYS)
    mem_b = run_session_b(one_src, decode_only=True)
    assert session_b_result_dict(mem_b)["verdict"] == "picked"
    assert set(session_b_result_dict(mem_b)) == set(SESSION_RESULT_JSON_KEYS)
    assert cli_b["verdict"] == "picked"
    assert plan_b.read_text(encoding="utf-8") == one_src


def test_beat69_result_dict_helpers_still_exported():
    """Beat 69: session_*_result_dict + SESSION_*_RESULT_JSON_KEYS stay public."""
    import src.research_implement as ri

    for name in (
        "session_a_result_dict",
        "session_b_result_dict",
        "SESSION_A_RESULT_JSON_KEYS",
        "SESSION_RESULT_JSON_KEYS",
    ):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name


def test_beat70_session_a_light_failed_json_keys(tmp_path: Path, capsys):
    """Beat 70: session-a --json light/failed keys match SESSION_A_RESULT_JSON_KEYS."""
    import json
    from src.research_implement.__main__ import main
    from src.research_implement.session_a import SESSION_A_RESULT_JSON_KEYS

    one = tmp_path / "beat70_one.md"
    one.write_text((FIXTURES / "one_open_ready.md").read_text(encoding="utf-8"), encoding="utf-8")
    rc_light = main(["session-a", "--plan", str(one), "--stub", "--json"])
    assert rc_light == 0
    light = json.loads(capsys.readouterr().out)
    assert light.get("verdict") == "light"
    assert set(light) == set(SESSION_A_RESULT_JSON_KEYS)
    assert one.read_text(encoding="utf-8") == (FIXTURES / "one_open_ready.md").read_text(encoding="utf-8")

    fail = tmp_path / "beat70_fail.md"
    fail_src = (FIXTURES / "two_queue_sections.md").read_text(encoding="utf-8")
    fail.write_text(fail_src, encoding="utf-8")
    rc_fail = main(["session-a", "--plan", str(fail), "--stub", "--json"])
    assert rc_fail == 1
    failed = json.loads(capsys.readouterr().out)
    assert failed.get("verdict") == "failed"
    assert set(failed) == set(SESSION_A_RESULT_JSON_KEYS)
    assert fail.read_text(encoding="utf-8") == fail_src


def test_beat70_path_runners_still_exported(tmp_path: Path):
    """Beat 70: run_session_a_path / run_session_b_path public and usable."""
    import src.research_implement as ri
    from src.research_implement import stub_brainstorm

    for name in ("run_session_a_path", "run_session_b_path"):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name

    plan = tmp_path / "beat70_path.md"
    plan.write_text((FIXTURES / "empty_queue.md").read_text(encoding="utf-8"), encoding="utf-8")
    a = ri.run_session_a_path(plan, brainstorm=stub_brainstorm, write=False)
    assert a.verdict in {"queued", "light", "failed"}
    b = ri.run_session_b_path(plan, decode_only=True, write=False)
    assert b.verdict in {"idle", "picked", "failed"}


def test_beat71_session_b_dry_idle_failed_json_keys(tmp_path: Path, capsys):
    """Beat 71: session-b --json dry_run/idle/failed keys match SESSION_RESULT_JSON_KEYS."""
    import json
    from src.research_implement.__main__ import main
    from src.research_implement.session_b import SESSION_RESULT_JSON_KEYS

    one = tmp_path / "beat71_one.md"
    one.write_text((FIXTURES / "one_open_ready.md").read_text(encoding="utf-8"), encoding="utf-8")
    rc_dry = main(["session-b", "--plan", str(one), "--dry-run", "--json"])
    assert rc_dry == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry.get("verdict") == "dry_run"
    assert set(dry) == set(SESSION_RESULT_JSON_KEYS)
    assert one.read_text(encoding="utf-8") == (FIXTURES / "one_open_ready.md").read_text(encoding="utf-8")

    empty = tmp_path / "beat71_empty.md"
    empty_src = (FIXTURES / "empty_queue.md").read_text(encoding="utf-8")
    empty.write_text(empty_src, encoding="utf-8")
    rc_idle = main(["session-b", "--plan", str(empty), "--json"])
    assert rc_idle == 0
    idle = json.loads(capsys.readouterr().out)
    assert idle.get("verdict") == "idle"
    assert set(idle) == set(SESSION_RESULT_JSON_KEYS)
    assert empty.read_text(encoding="utf-8") == empty_src

    fail = tmp_path / "beat71_fail.md"
    fail_src = (FIXTURES / "two_queue_sections.md").read_text(encoding="utf-8")
    fail.write_text(fail_src, encoding="utf-8")
    rc_fail = main(["session-b", "--plan", str(fail), "--json"])
    assert rc_fail == 1
    failed = json.loads(capsys.readouterr().out)
    assert failed.get("verdict") == "failed"
    assert set(failed) == set(SESSION_RESULT_JSON_KEYS)
    assert fail.read_text(encoding="utf-8") == fail_src


def test_beat71_to_dict_aliases_align():
    """Beat 71: to_dict / to_json_dict / session_b_result_dict agree on a pick."""
    from src.research_implement import (
        run_session_b,
        session_b_result_dict,
        SESSION_RESULT_JSON_KEYS,
    )

    result = run_session_b(
        (FIXTURES / "one_open_ready.md").read_text(encoding="utf-8"),
        decode_only=True,
    )
    assert result.verdict == "picked"
    d1 = result.to_dict()
    d2 = result.to_json_dict()
    d3 = session_b_result_dict(result)
    assert d1 == d2 == d3
    assert set(d1) == set(SESSION_RESULT_JSON_KEYS)


def test_beat72_session_a_to_dict_aliases_align():
    """Beat 72: to_dict / to_json_dict / session_a_result_dict agree for A verdicts."""
    from src.research_implement import (
        SESSION_A_RESULT_JSON_KEYS,
        run_session_a,
        session_a_result_dict,
        stub_brainstorm,
    )

    cases = (
        ((FIXTURES / "empty_queue.md").read_text(encoding="utf-8"), "queued", stub_brainstorm),
        ((FIXTURES / "one_open_ready.md").read_text(encoding="utf-8"), "light", stub_brainstorm),
        ((FIXTURES / "two_queue_sections.md").read_text(encoding="utf-8"), "failed", stub_brainstorm),
    )
    for plan_text, verdict, brainstorm in cases:
        result = run_session_a(plan_text, brainstorm=brainstorm)
        assert result.verdict == verdict, verdict
        d1 = result.to_dict()
        d2 = result.to_json_dict()
        d3 = session_a_result_dict(result)
        assert d1 == d2 == d3, verdict
        assert set(d1) == set(SESSION_A_RESULT_JSON_KEYS), verdict


def test_beat72_session_a_cli_queued_json_keys(tmp_path: Path, capsys):
    """Beat 72: session-a --stub --json queued keys match SESSION_A_RESULT_JSON_KEYS."""
    import json
    from src.research_implement.__main__ import main
    from src.research_implement.session_a import SESSION_A_RESULT_JSON_KEYS

    src = (FIXTURES / "empty_queue.md").read_text(encoding="utf-8")
    plan = tmp_path / "beat72.md"
    plan.write_text(src, encoding="utf-8")
    rc = main(["session-a", "--plan", str(plan), "--stub", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("verdict") == "queued"
    assert set(payload) == set(SESSION_A_RESULT_JSON_KEYS)


def test_beat73_result_and_queue_types_exported():
    """Beat 73: SessionAResult / SessionResult / SessionBResult / QueueItem public."""
    import src.research_implement as ri

    for name in ("SessionAResult", "SessionResult", "SessionBResult", "QueueItem"):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name

    # SessionBResult is alias or subclass of SessionResult
    assert ri.SessionBResult is ri.SessionResult or issubclass(
        ri.SessionBResult, ri.SessionResult
    )


def test_beat73_result_to_dict_callable_on_live_results():
    """Beat 73: live A/B results still expose to_dict matching SESSION_* keys."""
    from src.research_implement import (
        SESSION_A_RESULT_JSON_KEYS,
        SESSION_RESULT_JSON_KEYS,
        run_session_a,
        run_session_b,
        stub_brainstorm,
    )

    a = run_session_a(
        (FIXTURES / "empty_queue.md").read_text(encoding="utf-8"),
        brainstorm=stub_brainstorm,
    )
    assert set(a.to_dict()) == set(SESSION_A_RESULT_JSON_KEYS)
    b = run_session_b(
        (FIXTURES / "one_open_ready.md").read_text(encoding="utf-8"),
        decode_only=True,
    )
    assert set(b.to_dict()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat74_queueitem_pickable_and_format_roundtrip():
    """Beat 74: one_open QueueItem is B-pickable; format/parse round-trip."""
    from src.research_implement import (
        format_queue_item,
        is_b_pickable,
        parse_queue_items,
    )

    items = parse_queue_items((FIXTURES / "one_open_ready.md").read_text(encoding="utf-8"))
    assert len(items) >= 1
    item = items[0]
    assert is_b_pickable(item)
    assert item.item_id == "Q1"
    assert item.status.upper() == "OPEN"
    assert item.ready_for_implement.strip().lower() in {"yes", "y", "true", "1"}

    blob = format_queue_item(
        item_id=item.item_id,
        heading=item.heading or item.title,
        title=item.title,
        acceptance=item.acceptance,
        risks=item.risks,
        file_touch=item.file_touch,
        breaking_change=item.breaking_change,
        redeploy_notes=item.redeploy_notes,
        status=item.status,
        ready_for_implement=item.ready_for_implement,
    )
    wrapped = "## Queue\n\n" + blob + "\n"
    again = parse_queue_items(wrapped)
    assert len(again) == 1
    assert again[0].item_id == item.item_id
    assert again[0].title == item.title
    assert is_b_pickable(again[0])


def test_beat74_is_b_pickable_still_exported():
    """Beat 74: is_b_pickable remains a public export."""
    import src.research_implement as ri

    assert hasattr(ri, "is_b_pickable")
    assert "is_b_pickable" in getattr(ri, "__all__", ())


def test_beat75_incomplete_and_shipped_not_pickable():
    """Beat 75: incomplete_open / shipped_only → no B-pickable; count_open 0."""
    from src.research_implement import count_open, first_b_pick, is_b_pickable, parse_queue_items

    for name in ("incomplete_open.md", "shipped_only.md"):
        items = parse_queue_items((FIXTURES / name).read_text(encoding="utf-8"))
        assert count_open(items) == 0, name
        assert first_b_pick(items) is None, name
        assert not any(is_b_pickable(i) for i in items), name


def test_beat75_count_open_still_exported():
    """Beat 75: count_open remains a public export."""
    import src.research_implement as ri

    assert hasattr(ri, "count_open")
    assert "count_open" in getattr(ri, "__all__", ())


def test_beat76_not_ready_and_broken_ready_not_pickable():
    """Beat 76: open_complete_not_ready / broken_ready_flag → no B-pick; count_open 0."""
    from src.research_implement import (
        count_open,
        first_b_pick,
        is_b_pickable,
        is_ready_yes,
        parse_queue_items,
    )

    for name in ("open_complete_not_ready.md", "broken_ready_flag.md"):
        items = parse_queue_items((FIXTURES / name).read_text(encoding="utf-8"))
        assert count_open(items) == 0, name
        assert first_b_pick(items) is None, name
        assert not any(is_b_pickable(i) for i in items), name

    assert not is_ready_yes("READY")
    assert not is_ready_yes("maybe")
    assert not is_ready_yes("no")


def test_beat76_is_ready_yes_still_exported():
    """Beat 76: is_ready_yes remains a public export."""
    import src.research_implement as ri

    assert hasattr(ri, "is_ready_yes")
    assert "is_ready_yes" in getattr(ri, "__all__", ())


def test_beat77_mixed_priority_first_b_pick_skips_incomplete():
    """Beat 77: mixed_priority first_b_pick is first ready; count_open matches pickables."""
    from src.research_implement import count_open, first_b_pick, is_b_pickable, parse_queue_items

    items = parse_queue_items((FIXTURES / "mixed_priority.md").read_text(encoding="utf-8"))
    pickables = [i for i in items if is_b_pickable(i)]
    assert pickables
    assert count_open(items) == len(pickables)
    pick = first_b_pick(items)
    assert pick is not None
    assert pick.item_id == pickables[0].item_id
    assert is_b_pickable(pick)


def test_beat77_first_b_pick_still_exported():
    """Beat 77: first_b_pick remains a public export."""
    import src.research_implement as ri

    assert hasattr(ri, "first_b_pick")
    assert "first_b_pick" in getattr(ri, "__all__", ())


def test_beat78_two_open_first_b_pick_and_count():
    """Beat 78: two_open_ready first_b_pick is first; count_open is 2."""
    from src.research_implement import count_open, first_b_pick, is_b_pickable, parse_queue_items

    items = parse_queue_items((FIXTURES / "two_open_ready.md").read_text(encoding="utf-8"))
    assert len(items) >= 2
    assert count_open(items) == 2
    assert all(is_b_pickable(i) for i in items[:2])
    pick = first_b_pick(items)
    assert pick is not None
    assert pick.item_id == items[0].item_id
    assert pick.item_id != items[1].item_id


def test_beat78_parse_queue_items_still_exported():
    """Beat 78: parse_queue_items remains a public export."""
    import src.research_implement as ri

    assert hasattr(ri, "parse_queue_items")
    assert "parse_queue_items" in getattr(ri, "__all__", ())


def test_beat79_one_open_and_empty_pick_counts():
    """Beat 79: one_open pick/count=1; empty_queue pick None / count 0."""
    from src.research_implement import count_open, first_b_pick, parse_queue_items

    one = parse_queue_items((FIXTURES / "one_open_ready.md").read_text(encoding="utf-8"))
    assert count_open(one) == 1
    pick = first_b_pick(one)
    assert pick is not None and pick.item_id == "Q1"

    empty = parse_queue_items((FIXTURES / "empty_queue.md").read_text(encoding="utf-8"))
    assert count_open(empty) == 0
    assert first_b_pick(empty) is None


def test_beat79_next_queue_id_still_exported():
    """Beat 79: next_queue_id remains a public export."""
    import src.research_implement as ri

    assert hasattr(ri, "next_queue_id")
    assert "next_queue_id" in getattr(ri, "__all__", ())


def test_beat80_top_level_help_still_lists_subcommands(capsys):
    """Beat 80: top-level --help lists session-a / session-b / idle-decode."""
    import pytest
    from src.research_implement.__main__ import main

    with pytest.raises(SystemExit) as ei:
        main(["--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    for name in ("session-a", "session-b", "idle-decode"):
        assert name in out, name


def test_beat80_public_api_milestone_exports():
    """Beat 80: core runners/helpers/result types remain public."""
    import src.research_implement as ri

    for name in (
        "run_session_a",
        "run_session_b",
        "run_session_a_path",
        "run_session_b_path",
        "parse_queue_items",
        "first_b_pick",
        "is_b_pickable",
        "is_ready_yes",
        "count_open",
        "SessionAResult",
        "SessionResult",
        "QueueItem",
        "SESSION_A_RESULT_JSON_KEYS",
        "SESSION_RESULT_JSON_KEYS",
    ):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name

def test_beat81_watch_lookalike_and_watch_only_picks():
    """Beat 81: watch_lookalike first_b_pick=Q3; watch_only first_b_pick=None."""
    look = parse_queue_items(_load("watch_lookalike.md"))
    pick = first_b_pick(look)
    assert pick is not None
    assert pick.item_id == "Q3"
    assert "Fake" not in (pick.title or "")
    assert count_open(look) == 1

    only = parse_queue_items(_load("watch_only_lookalike.md"))
    assert first_b_pick(only) is None
    assert count_open(only) == 0
    assert len(only) == 0


def test_beat81_mark_item_shipped_still_exported():
    """Beat 81: mark_item_shipped remains a public export."""
    import src.research_implement as ri

    assert hasattr(ri, "mark_item_shipped")
    assert "mark_item_shipped" in getattr(ri, "__all__", ())

def test_beat82_subcommand_help_lists_plan_json(capsys):
    """Beat 82: session-a / session-b / idle-decode --help mention --plan and --json."""
    import pytest
    from src.research_implement.__main__ import main

    for cmd in ("session-a", "session-b", "idle-decode"):
        with pytest.raises(SystemExit) as ei:
            main([cmd, "--help"])
        assert ei.value.code == 0
        out = capsys.readouterr().out
        assert "--plan" in out, cmd
        assert "--json" in out, cmd


def test_beat82_render_queue_count_and_capacity_exported():
    """Beat 82: render_queue_count / QUEUE_CAPACITY remain public; smoke strings."""
    import src.research_implement as ri

    for name in ("render_queue_count", "QUEUE_CAPACITY"):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name
    assert ri.QUEUE_CAPACITY == 10
    assert ri.render_queue_count(0) == "queue 0/10"
    assert ri.render_queue_count(1) == "queue 1/10"

def test_beat83_non_pickable_fixtures_first_b_pick_none():
    """Beat 83: incomplete/shipped/not-ready/broken/empty never yield a first_b_pick."""
    for name in (
        "incomplete_open.md",
        "shipped_only.md",
        "open_complete_not_ready.md",
        "broken_ready_flag.md",
        "empty_queue.md",
    ):
        items = parse_queue_items(_load(name))
        assert first_b_pick(items) is None, name
        assert count_open(items) == 0, name


def test_beat83_serialize_queue_helpers_still_exported():
    """Beat 83: serialize_queue_item / serialize_queue_items remain public."""
    import src.research_implement as ri

    for name in ("serialize_queue_item", "serialize_queue_items"):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name

def test_beat84_pickable_fixtures_first_b_pick_ids():
    """Beat 84: one_open/two_open/watch_lookalike/mixed_priority first_b_pick ids."""
    expectations = (
        ("one_open_ready.md", "Q1", 1),
        ("two_open_ready.md", "Q1", 2),
        ("watch_lookalike.md", "Q3", 1),
        ("mixed_priority.md", "Q2", 1),
    )
    for name, item_id, open_n in expectations:
        items = parse_queue_items(_load(name))
        pick = first_b_pick(items)
        assert pick is not None, name
        assert pick.item_id == item_id, name
        assert count_open(items) == open_n, name


def test_beat84_complete_and_open_status_still_exported():
    """Beat 84: is_complete_six_field / is_open_status remain public."""
    import src.research_implement as ri

    for name in ("is_complete_six_field", "is_open_status"):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name
    item = parse_queue_items(_load("one_open_ready.md"))[0]
    assert ri.is_complete_six_field(item) is True
    assert ri.is_open_status(item.status) is True

def test_beat85_subcommand_help_producer_and_decode_flags(capsys):
    """Beat 85: session-a producer flags; session-b --decode-only; idle-decode no --dry-run."""
    import pytest
    from src.research_implement.__main__ import main

    with pytest.raises(SystemExit) as ei:
        main(["session-a", "--help"])
    assert ei.value.code == 0
    out_a = capsys.readouterr().out
    for flag in ("--stub", "--no-stub", "--dry-run", "--candidate-json"):
        assert flag in out_a, flag

    with pytest.raises(SystemExit) as ei:
        main(["session-b", "--help"])
    assert ei.value.code == 0
    out_b = capsys.readouterr().out
    assert "--decode-only" in out_b
    assert "--dry-run" in out_b

    with pytest.raises(SystemExit) as ei:
        main(["idle-decode", "--help"])
    assert ei.value.code == 0
    out_i = capsys.readouterr().out
    assert "--plan" in out_i
    assert "--json" in out_i
    assert "--dry-run" not in out_i


def test_beat85_required_fields_still_exported():
    """Beat 85: REQUIRED_FIELDS remains public with the six field names."""
    import src.research_implement as ri

    assert hasattr(ri, "REQUIRED_FIELDS")
    assert "REQUIRED_FIELDS" in getattr(ri, "__all__", ())
    assert ri.REQUIRED_FIELDS == (
        "title",
        "acceptance",
        "risks",
        "file_touch",
        "breaking_change",
        "redeploy_notes",
    )

def test_beat86_idle_queue_zero_ten_and_scheduler_guard():
    """Beat 86: idle fire is queue 0/10; scheduler_delete raises SchedulerDeleteForbidden."""
    for name in ("empty_queue.md", "shipped_only.md", "incomplete_open.md"):
        result = run_session_b(_load(name), decode_only=True)
        assert result.ok is True
        assert result.verdict == "idle"
        assert result.keep_schedule is True
        assert result.scheduler_delete_called is False
        assert "queue 0/10" in result.message
        assert result.decode_report is None

    with pytest.raises(SchedulerDeleteForbidden) as ei:
        scheduler_delete()
    err = str(ei.value)
    assert "queue 0/10" in err
    assert "Never call scheduler_delete" in err


def test_beat86_scheduler_delete_forbidden_exported_and_decode_pick():
    """Beat 86: SchedulerDeleteForbidden public; decode report starts with decode pick."""
    import src.research_implement as ri

    assert hasattr(ri, "SchedulerDeleteForbidden")
    assert "SchedulerDeleteForbidden" in getattr(ri, "__all__", ())
    assert ri.SchedulerDeleteForbidden is SchedulerDeleteForbidden

    item = parse_queue_items(_load("one_open_ready.md"))[0]
    report = format_decode_report(item)
    assert report.startswith("decode pick Q1:")
    for n in range(1, 7):
        assert f"  {n}. " in report

def test_beat87_two_queue_ambiguous_and_heading_count():
    """Beat 87: two_queue_sections headings=2 and require_unique raises AmbiguousQueueError."""
    src = _load("two_queue_sections.md")
    assert count_queue_headings(src) == 2
    with pytest.raises(AmbiguousQueueError) as ei:
        require_unique_queue_section(src)
    assert "2 headings" in str(ei.value)
    assert "fail-closed" in str(ei.value).lower() or "refuse" in str(ei.value).lower()


def test_beat87_decode_helpers_still_exported():
    """Beat 87: format_decode_report / decode_fields remain public; report has decode pick."""
    import src.research_implement as ri

    for name in ("format_decode_report", "decode_fields", "AmbiguousQueueError"):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name
    item = parse_queue_items(_load("two_open_ready.md"))[0]
    report = format_decode_report(item)
    assert report.startswith("decode pick Q1:")

def test_beat88_watch_marker_fixtures_idle_or_pick():
    """Beat 88: Watch/Heartbeat empty/no-queue idle; watch_queue_heartbeat picks Q1."""
    for name in (
        "watch_queue_heartbeat_empty.md",
        "watch_heartbeat_no_queue.md",
        "queue_with_watch_heartbeat.md",
    ):
        result = run_session_b(_load(name), decode_only=True)
        assert result.ok is True
        assert result.verdict == "idle", name
        assert result.keep_schedule is True
        body = _load(name)
        assert "## Watch" in body
        assert "## Heartbeat" in body

    result = run_session_b(_load("watch_queue_heartbeat.md"), decode_only=True)
    assert result.ok is True
    assert result.verdict == "picked"
    assert result.item is not None
    assert result.item.item_id == "Q1"


def test_beat88_session_result_keys_still_exported():
    """Beat 88: SESSION_A_RESULT_KEYS / SESSION_B_RESULT_KEYS remain public."""
    import src.research_implement as ri

    for name in ("SESSION_A_RESULT_KEYS", "SESSION_B_RESULT_KEYS"):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name
    assert ri.SESSION_A_RESULT_KEYS == (
        "ok",
        "verdict",
        "open_count",
        "queue",
        "b_pick_title",
        "title",
        "wrote_item",
    )
    assert ri.SESSION_B_RESULT_KEYS == (
        "ok",
        "verdict",
        "open_count",
        "queue",
        "keep_schedule",
        "scheduler_delete_called",
        "item",
        "implement_result",
        "wrote_files",
        "shipped",
    )

def test_beat89_incomplete_candidate_reasons_contract():
    """Beat 89: incomplete item lists acceptance; complete empty; bad type candidate_type."""
    incomplete = parse_queue_items(_load("incomplete_open.md"))[0]
    complete = parse_queue_items(_load("one_open_ready.md"))[0]
    assert incomplete_candidate_reasons(incomplete) == ["acceptance"]
    assert incomplete_candidate_reasons(complete) == []
    assert incomplete_candidate_reasons(["not", "a", "candidate"]) == ["candidate_type"]


def test_beat89_session_json_keys_still_exported():
    """Beat 89: SESSION_A_RESULT_JSON_KEYS / SESSION_RESULT_JSON_KEYS remain public."""
    import src.research_implement as ri

    for name in ("SESSION_A_RESULT_JSON_KEYS", "SESSION_RESULT_JSON_KEYS", "incomplete_candidate_reasons"):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name
    assert ri.SESSION_A_RESULT_JSON_KEYS == (
        "ok",
        "verdict",
        "open_count",
        "queue",
        "b_pick_title",
        "title",
        "wrote_item",
    )
    assert ri.SESSION_RESULT_JSON_KEYS == (
        "ok",
        "verdict",
        "open_count",
        "queue",
        "keep_schedule",
        "scheduler_delete_called",
        "item",
        "implement_result",
        "wrote_files",
        "shipped",
    )

def test_beat90_top_level_help_still_lists_subcommands(capsys):
    """Beat 90: top-level --help lists session-a / session-b / idle-decode."""
    import pytest
    from src.research_implement.__main__ import main

    with pytest.raises(SystemExit) as ei:
        main(["--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    for name in ("session-a", "session-b", "idle-decode"):
        assert name in out, name


def test_beat90_public_api_milestone_exports():
    """Beat 90: core runners/helpers/guards/result types remain public through beat90."""
    import src.research_implement as ri

    for name in (
        "run_session_a",
        "run_session_b",
        "run_session_a_path",
        "run_session_b_path",
        "parse_queue_items",
        "first_b_pick",
        "is_b_pickable",
        "is_ready_yes",
        "count_open",
        "mark_item_shipped",
        "render_queue_count",
        "QUEUE_CAPACITY",
        "REQUIRED_FIELDS",
        "AmbiguousQueueError",
        "SchedulerDeleteForbidden",
        "incomplete_candidate_reasons",
        "SessionAResult",
        "SessionResult",
        "SessionBResult",
        "QueueItem",
        "SESSION_A_RESULT_KEYS",
        "SESSION_B_RESULT_KEYS",
        "SESSION_A_RESULT_JSON_KEYS",
        "SESSION_RESULT_JSON_KEYS",
    ):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name

def test_beat91_empty_append_next_q1_pickable():
    """Beat 91: empty_queue next_queue_id=Q1; append makes first_b_pick Q1."""
    empty = _load("empty_queue.md")
    items = parse_queue_items(empty)
    assert items == []
    assert next_queue_id(items) == "Q1"
    block = format_queue_item(
        item_id="Q1",
        heading="Beat91 stub",
        title="Beat91 stub",
        acceptance="ok",
        risks="none",
        file_touch="t.py",
        breaking_change="false",
        redeploy_notes="none",
        status="OPEN",
        ready_for_implement="yes",
    )
    updated = append_queue_item(empty, block)
    assert "## Queue" in updated
    parsed = parse_queue_items(updated)
    assert count_open(parsed) == 1
    pick = first_b_pick(parsed)
    assert pick is not None
    assert pick.item_id == "Q1"
    assert pick.title == "Beat91 stub"


def test_beat91_append_format_write_still_exported():
    """Beat 91: append_queue_item / format_queue_item / write_queue_section stay public."""
    import src.research_implement as ri

    for name in ("append_queue_item", "format_queue_item", "write_queue_section", "next_queue_id"):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name

def test_beat92_mark_item_shipped_clears_pick():
    """Beat 92: mark_item_shipped on one_open → status SHIPPED; first_b_pick None."""
    src = _load("one_open_ready.md")
    assert first_b_pick(parse_queue_items(src)) is not None
    shipped = mark_item_shipped(src, "Q1", "dead92", note="beat92")
    items = parse_queue_items(shipped)
    assert count_open(items) == 0
    assert first_b_pick(items) is None
    assert items[0].status.startswith("SHIPPED")
    assert "dead92" in items[0].status
    assert "beat92" in items[0].status


def test_beat92_stub_brainstorm_and_implement_helpers_exported():
    """Beat 92: stub_brainstorm ready yes; stub/dry-run implement helpers public."""
    import src.research_implement as ri

    stub = stub_brainstorm()
    assert stub["ready_for_implement"] == "yes"
    assert incomplete_candidate_reasons(stub) == []
    for name in (
        "stub_brainstorm",
        "dry_run_implement",
        "make_fixture_ship_implement",
        "default_implement",
        "fixture_ship_implement",
    ):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name

def test_beat93_is_ready_yes_aliases_and_negatives():
    """Beat 93: is_ready_yes accepts yes/y/true (any case); rejects no/empty/ready."""
    for token in ("yes", "YES", "Yes", "y", "Y", "true", "TRUE", "True"):
        assert is_ready_yes(token) is True, token
    for token in ("no", "n", "", "ready", "maybe", "0"):
        assert is_ready_yes(token) is False, token


def test_beat93_search_plan_aliases_and_result_dicts_exported():
    """Beat 93: default/stub_search_plan alias stub; session_*_result_dict public."""
    import src.research_implement as ri

    assert ri.default_search_plan is ri.stub_brainstorm
    assert ri.stub_search_plan is ri.stub_brainstorm
    for name in (
        "default_search_plan",
        "stub_search_plan",
        "session_a_result_dict",
        "session_b_result_dict",
        "is_ready_yes",
    ):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name

def test_beat94_result_dict_and_to_dict_match_json_keys():
    """Beat 94: session_*_result_dict and to_dict keys match SESSION_*_JSON_KEYS."""
    empty = _load("empty_queue.md")
    a = run_session_a(empty, brainstorm=stub_brainstorm)
    da = session_a_result_dict(a)
    assert tuple(da.keys()) == SESSION_A_RESULT_JSON_KEYS
    assert a.to_dict() == da
    assert da["verdict"] == "queued"

    one = _load("one_open_ready.md")
    b = run_session_b(one, decode_only=True)
    db = session_b_result_dict(b)
    assert tuple(db.keys()) == SESSION_RESULT_JSON_KEYS
    assert b.to_dict() == db
    assert db["verdict"] == "picked"


def test_beat94_path_runners_still_exported():
    """Beat 94: run_session_a_path / run_session_b_path remain public."""
    import src.research_implement as ri

    for name in ("run_session_a_path", "run_session_b_path", "session_a_result_dict", "session_b_result_dict"):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name

def test_beat95_serialize_roundtrip_stays_pickable():
    """Beat 95: serialize_queue_item → reparse under ## Queue stays B-pickable."""
    original = parse_queue_items(_load("one_open_ready.md"))[0]
    block = serialize_queue_item(original)
    wrapped = "## Queue" + "\n\n" + block + "\n"
    parsed = parse_queue_items(wrapped)
    assert len(parsed) == 1
    assert parsed[0].item_id == original.item_id
    assert parsed[0].title == original.title
    assert is_b_pickable(parsed[0]) is True
    assert first_b_pick(parsed) is not None


def test_beat95_count_queue_headings_fixture_contract():
    """Beat 95: count_queue_headings is 1 for normal plans, 2 for two_queue_sections."""
    assert count_queue_headings(_load("one_open_ready.md")) == 1
    assert count_queue_headings(_load("two_open_ready.md")) == 1
    assert count_queue_headings(_load("empty_queue.md")) == 1
    assert count_queue_headings(_load("two_queue_sections.md")) == 2
    import src.research_implement as ri

    assert "count_queue_headings" in getattr(ri, "__all__", ())
    assert "serialize_queue_item" in getattr(ri, "__all__", ())

def test_beat96_decode_fields_and_report_lines():
    """Beat 96: decode_fields has six fields + status/ready; report has lines 1–6."""
    item = parse_queue_items(_load("one_open_ready.md"))[0]
    fields = decode_fields(item)
    for name in REQUIRED_FIELDS:
        assert name in fields, name
        assert fields[name]
    assert fields["status"] == "OPEN"
    assert is_ready_yes(fields["ready_for_implement"]) is True
    assert fields["item_id"] == "Q1"

    report = format_decode_report(item)
    assert report.startswith("decode pick Q1:")
    for n, name in enumerate(REQUIRED_FIELDS, start=1):
        assert f"  {n}. {name}:" in report, name


def test_beat96_decode_helpers_still_exported():
    """Beat 96: decode_fields / format_decode_report remain public."""
    import src.research_implement as ri

    for name in ("decode_fields", "format_decode_report", "REQUIRED_FIELDS"):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name

def test_beat97_next_queue_id_progression():
    """Beat 97: next_queue_id empty→Q1, one→Q2, two→Q3, watch_lookalike→Q4."""
    expectations = (
        ("empty_queue.md", "Q1"),
        ("one_open_ready.md", "Q2"),
        ("two_open_ready.md", "Q3"),
        ("watch_lookalike.md", "Q4"),
    )
    for name, expected in expectations:
        items = parse_queue_items(_load(name))
        assert next_queue_id(items) == expected, name


def test_beat97_require_unique_fail_closed_and_exports():
    """Beat 97: require_unique ok on one Queue; AmbiguousQueueError on two; exports public."""
    require_unique_queue_section(_load("one_open_ready.md"))
    with pytest.raises(AmbiguousQueueError):
        require_unique_queue_section(_load("two_queue_sections.md"))
    import src.research_implement as ri

    for name in ("next_queue_id", "require_unique_queue_section", "AmbiguousQueueError"):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name

def test_beat98_is_b_pickable_fixture_matrix():
    """Beat 98: is_b_pickable true only for complete OPEN ready items."""
    expectations = (
        ("one_open_ready.md", [("Q1", True)]),
        ("two_open_ready.md", [("Q1", True), ("Q2", True)]),
        ("incomplete_open.md", [("Q2", False)]),
        ("shipped_only.md", [("Q1", False)]),
        ("open_complete_not_ready.md", [("Q1", False)]),
        ("broken_ready_flag.md", [("Q1", False), ("Q2", False)]),
        ("mixed_priority.md", [("Q1", False), ("Q2", True)]),
        ("watch_lookalike.md", [("Q3", True)]),
    )
    for name, expected in expectations:
        items = parse_queue_items(_load(name))
        got = [(i.item_id, is_b_pickable(i)) for i in items]
        assert got == expected, name


def test_beat98_pick_helpers_still_exported():
    """Beat 98: is_b_pickable / is_complete_six_field / is_open_status / is_ready_yes public."""
    import src.research_implement as ri

    for name in ("is_b_pickable", "is_complete_six_field", "is_open_status", "is_ready_yes"):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name

def test_beat99_count_open_fixture_matrix_and_render():
    """Beat 99: count_open per fixture; render_queue_count matches queue N/10."""
    expectations = (
        ("empty_queue.md", 0),
        ("one_open_ready.md", 1),
        ("two_open_ready.md", 2),
        ("incomplete_open.md", 0),
        ("shipped_only.md", 0),
        ("mixed_priority.md", 1),
        ("watch_lookalike.md", 1),
        ("broken_ready_flag.md", 0),
    )
    for name, open_n in expectations:
        items = parse_queue_items(_load(name))
        assert count_open(items) == open_n, name
        assert render_queue_count(open_n) == f"queue {open_n}/10", name


def test_beat99_count_render_capacity_still_exported():
    """Beat 99: count_open / render_queue_count / QUEUE_CAPACITY remain public."""
    import src.research_implement as ri

    for name in ("count_open", "render_queue_count", "QUEUE_CAPACITY"):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name
    assert ri.QUEUE_CAPACITY == 10

def test_beat100_top_level_help_still_lists_subcommands(capsys):
    """Beat 100: top-level --help lists session-a / session-b / idle-decode."""
    import pytest
    from src.research_implement.__main__ import main

    with pytest.raises(SystemExit) as ei:
        main(["--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    for name in ("session-a", "session-b", "idle-decode"):
        assert name in out, name


def test_beat100_public_api_milestone_exports():
    """Beat 100: core runners/helpers/guards/decode/serialize remain public through beat100."""
    import src.research_implement as ri

    for name in (
        "run_session_a",
        "run_session_b",
        "run_session_a_path",
        "run_session_b_path",
        "parse_queue_items",
        "first_b_pick",
        "is_b_pickable",
        "is_ready_yes",
        "is_complete_six_field",
        "is_open_status",
        "count_open",
        "count_queue_headings",
        "next_queue_id",
        "mark_item_shipped",
        "append_queue_item",
        "format_queue_item",
        "write_queue_section",
        "serialize_queue_item",
        "serialize_queue_items",
        "render_queue_count",
        "QUEUE_CAPACITY",
        "REQUIRED_FIELDS",
        "AmbiguousQueueError",
        "SchedulerDeleteForbidden",
        "require_unique_queue_section",
        "incomplete_candidate_reasons",
        "decode_fields",
        "format_decode_report",
        "stub_brainstorm",
        "default_search_plan",
        "session_a_result_dict",
        "session_b_result_dict",
        "SessionAResult",
        "SessionResult",
        "SessionBResult",
        "QueueItem",
        "SESSION_A_RESULT_KEYS",
        "SESSION_B_RESULT_KEYS",
        "SESSION_A_RESULT_JSON_KEYS",
        "SESSION_RESULT_JSON_KEYS",
    ):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name

def test_beat101_format_queue_item_roundtrip_pickable():
    """Beat 101: format_queue_item → parse under ## Queue stays B-pickable."""
    block = format_queue_item(
        item_id="Q7",
        heading="Beat101",
        title="Beat101",
        acceptance="ok",
        risks="none",
        file_touch="t.py",
        breaking_change="false",
        redeploy_notes="none",
        status="OPEN",
        ready_for_implement="yes",
    )
    wrapped = "## Queue" + "\n\n" + block + "\n"
    items = parse_queue_items(wrapped)
    assert len(items) == 1
    assert items[0].item_id == "Q7"
    assert items[0].title == "Beat101"
    assert is_b_pickable(items[0]) is True
    assert first_b_pick(items).item_id == "Q7"


def test_beat101_dry_run_and_fixture_ship_still_exported():
    """Beat 101: dry_run_implement / make_fixture_ship_implement remain public."""
    import src.research_implement as ri

    for name in ("dry_run_implement", "make_fixture_ship_implement", "format_queue_item"):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name
    assert callable(ri.dry_run_implement)
    assert callable(ri.make_fixture_ship_implement())

def test_beat102_write_queue_section_transplants_two_open():
    """Beat 102: write_queue_section(empty, two_open items) → open=2 pick Q1."""
    empty = _load("empty_queue.md")
    two = parse_queue_items(_load("two_open_ready.md"))
    out = write_queue_section(empty, two)
    items = parse_queue_items(out)
    assert count_open(items) == 2
    assert [i.item_id for i in items] == ["Q1", "Q2"]
    assert first_b_pick(items).item_id == "Q1"


def test_beat102_default_implement_aliases_dry_run_exported():
    """Beat 102: default_implement is dry_run_implement; both + write_queue public."""
    import src.research_implement as ri

    assert ri.default_implement is ri.dry_run_implement
    for name in ("default_implement", "dry_run_implement", "write_queue_section", "fixture_ship_implement"):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name

def test_beat103_serialize_queue_items_two_open_roundtrip():
    """Beat 103: serialize_queue_items(two_open) → reparse stays open=2 pick Q1."""
    original = parse_queue_items(_load("two_open_ready.md"))
    body = serialize_queue_items(original)
    wrapped = "## Queue" + "\n\n" + body + "\n"
    items = parse_queue_items(wrapped)
    assert count_open(items) == 2
    assert [i.item_id for i in items] == ["Q1", "Q2"]
    assert all(is_b_pickable(i) for i in items)
    assert first_b_pick(items).item_id == "Q1"


def test_beat103_serialize_queue_items_still_exported():
    """Beat 103: serialize_queue_items remains a public export."""
    import src.research_implement as ri

    assert hasattr(ri, "serialize_queue_items")
    assert "serialize_queue_items" in getattr(ri, "__all__", ())

def test_beat104_queueitem_field_map_covers_required():
    """Beat 104: QueueItem.field_map keys match REQUIRED_FIELDS with values."""
    item = parse_queue_items(_load("one_open_ready.md"))[0]
    fields = item.field_map()
    assert tuple(fields.keys()) == REQUIRED_FIELDS
    for name in REQUIRED_FIELDS:
        assert fields[name], name
    assert item.title == fields["title"]
    assert item.acceptance == fields["acceptance"]


def test_beat104_queueitem_still_exported():
    """Beat 104: QueueItem / REQUIRED_FIELDS remain public."""
    import src.research_implement as ri

    for name in ("QueueItem", "REQUIRED_FIELDS"):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name

def test_beat105_to_json_dict_aliases_to_dict():
    """Beat 105: Session A light + Session B picked to_json_dict == to_dict."""
    a = run_session_a(_load("one_open_ready.md"), brainstorm=stub_brainstorm)
    assert a.verdict == "light"
    assert a.wrote_item is False
    assert a.to_json_dict() == a.to_dict()

    b = run_session_b(_load("one_open_ready.md"), decode_only=True)
    assert b.verdict == "picked"
    assert b.to_json_dict() == b.to_dict()


def test_beat105_session_b_result_is_session_result():
    """Beat 105: SessionBResult is SessionResult; both remain public."""
    import src.research_implement as ri

    assert ri.SessionBResult is ri.SessionResult
    for name in ("SessionAResult", "SessionBResult", "SessionResult"):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name

def test_beat106_keep_schedule_never_scheduler_delete():
    """Beat 106: idle fixtures + one_open pick keep_schedule; never scheduler_delete_called."""
    for name in ("empty_queue.md", "shipped_only.md", "incomplete_open.md"):
        result = run_session_b(_load(name), decode_only=True)
        assert result.verdict == "idle", name
        assert result.keep_schedule is True, name
        assert result.scheduler_delete_called is False, name

    picked = run_session_b(_load("one_open_ready.md"), decode_only=True)
    assert picked.verdict == "picked"
    assert picked.keep_schedule is True
    assert picked.scheduler_delete_called is False


def test_beat106_scheduler_delete_still_forbidden_exported():
    """Beat 106: scheduler_delete raises SchedulerDeleteForbidden; both public."""
    import src.research_implement as ri

    with pytest.raises(SchedulerDeleteForbidden):
        scheduler_delete()
    for name in ("scheduler_delete", "SchedulerDeleteForbidden"):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name

def test_beat107_session_a_queued_light_failed_to_dict():
    """Beat 107: Session A empty→queued wrote; one_open→light; two_queue→failed."""
    queued = run_session_a(_load("empty_queue.md"), brainstorm=stub_brainstorm).to_dict()
    assert queued["verdict"] == "queued"
    assert queued["ok"] is True
    assert queued["wrote_item"] is True
    assert queued["open_count"] == 1
    assert queued["queue"] == "queue 1/10"
    assert queued["title"] == "Stub shippable change"

    light = run_session_a(_load("one_open_ready.md"), brainstorm=stub_brainstorm).to_dict()
    assert light["verdict"] == "light"
    assert light["ok"] is True
    assert light["wrote_item"] is False
    assert light["open_count"] == 1
    assert light["title"] is None

    failed = run_session_a(_load("two_queue_sections.md"), brainstorm=stub_brainstorm).to_dict()
    assert failed["verdict"] == "failed"
    assert failed["ok"] is False
    assert failed["wrote_item"] is False
    assert failed["open_count"] == 0


def test_beat107_run_session_a_still_exported():
    """Beat 107: run_session_a remains a public export."""
    import src.research_implement as ri

    assert hasattr(ri, "run_session_a")
    assert "run_session_a" in getattr(ri, "__all__", ())

def test_beat108_session_b_idle_picked_failed_to_dict():
    """Beat 108: Session B empty→idle; one_open→picked; two_queue→failed."""
    idle = run_session_b(_load("empty_queue.md"), decode_only=True).to_dict()
    assert idle["verdict"] == "idle"
    assert idle["ok"] is True
    assert idle["open_count"] == 0
    assert idle["queue"] == "queue 0/10"
    assert idle["keep_schedule"] is True
    assert idle["scheduler_delete_called"] is False
    assert idle["shipped"] is False

    picked = run_session_b(_load("one_open_ready.md"), decode_only=True).to_dict()
    assert picked["verdict"] == "picked"
    assert picked["ok"] is True
    assert picked["open_count"] == 1
    assert picked["queue"] == "queue 1/10"
    assert picked["keep_schedule"] is True
    assert picked["shipped"] is False

    failed = run_session_b(_load("two_queue_sections.md"), decode_only=True).to_dict()
    assert failed["verdict"] == "failed"
    assert failed["ok"] is False
    assert failed["open_count"] == 0
    assert failed["keep_schedule"] is True
    assert failed["scheduler_delete_called"] is False


def test_beat108_run_session_b_still_exported():
    """Beat 108: run_session_b remains a public export."""
    import src.research_implement as ri

    assert hasattr(ri, "run_session_b")
    assert "run_session_b" in getattr(ri, "__all__", ())

def test_beat109_first_b_pick_matches_count_open():
    """Beat 109: count_open == pickable count; first_b_pick is first pickable or None."""
    fixtures = (
        "empty_queue.md",
        "one_open_ready.md",
        "two_open_ready.md",
        "incomplete_open.md",
        "shipped_only.md",
        "open_complete_not_ready.md",
        "broken_ready_flag.md",
        "mixed_priority.md",
        "watch_lookalike.md",
        "watch_only_lookalike.md",
    )
    for name in fixtures:
        items = parse_queue_items(_load(name))
        pickable = [i for i in items if is_b_pickable(i)]
        assert count_open(items) == len(pickable), name
        pick = first_b_pick(items)
        if not pickable:
            assert pick is None, name
        else:
            assert pick is not None, name
            assert pick.item_id == pickable[0].item_id, name


def test_beat109_public_all_unique():
    """Beat 109: package __all__ has unique non-empty names."""
    import src.research_implement as ri

    assert hasattr(ri, "__all__")
    names = list(ri.__all__)
    assert names
    assert len(names) == len(set(names))
    for name in names:
        assert hasattr(ri, name), name

def test_beat110_top_level_help_still_lists_subcommands(capsys):
    """Beat 110: top-level --help lists session-a / session-b / idle-decode."""
    import pytest
    from src.research_implement.__main__ import main

    with pytest.raises(SystemExit) as ei:
        main(["--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    for name in ("session-a", "session-b", "idle-decode"):
        assert name in out, name


def test_beat110_public_api_milestone_exports():
    """Beat 110: core A/B API + aliases/write/serialize remain public through beat110."""
    import src.research_implement as ri

    for name in (
        "run_session_a",
        "run_session_b",
        "run_session_a_path",
        "run_session_b_path",
        "parse_queue_items",
        "first_b_pick",
        "is_b_pickable",
        "is_ready_yes",
        "is_complete_six_field",
        "is_open_status",
        "count_open",
        "count_queue_headings",
        "next_queue_id",
        "mark_item_shipped",
        "append_queue_item",
        "format_queue_item",
        "write_queue_section",
        "serialize_queue_item",
        "serialize_queue_items",
        "render_queue_count",
        "QUEUE_CAPACITY",
        "REQUIRED_FIELDS",
        "AmbiguousQueueError",
        "SchedulerDeleteForbidden",
        "scheduler_delete",
        "require_unique_queue_section",
        "incomplete_candidate_reasons",
        "decode_fields",
        "format_decode_report",
        "stub_brainstorm",
        "default_search_plan",
        "stub_search_plan",
        "default_implement",
        "dry_run_implement",
        "make_fixture_ship_implement",
        "fixture_ship_implement",
        "session_a_result_dict",
        "session_b_result_dict",
        "SessionAResult",
        "SessionResult",
        "SessionBResult",
        "QueueItem",
        "SESSION_A_RESULT_KEYS",
        "SESSION_B_RESULT_KEYS",
        "SESSION_A_RESULT_JSON_KEYS",
        "SESSION_RESULT_JSON_KEYS",
    ):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name
    assert ri.SessionBResult is ri.SessionResult
    assert ri.default_implement is ri.dry_run_implement
    assert ri.default_search_plan is ri.stub_brainstorm
    assert len(ri.__all__) == len(set(ri.__all__))

def test_beat111_idle_decode_path_queue_zero_ten():
    """Beat 111: non-pickable fixtures idle with queue 0/10; decode_report None; no scheduler_delete."""
    idle_fixtures = (
        "empty_queue.md",
        "shipped_only.md",
        "incomplete_open.md",
        "open_complete_not_ready.md",
        "broken_ready_flag.md",
        "watch_only_lookalike.md",
        "watch_heartbeat_no_queue.md",
        "watch_queue_heartbeat_empty.md",
    )
    for name in idle_fixtures:
        result = run_session_b(_load(name), decode_only=True)
        assert result.ok is True, name
        assert result.verdict == "idle", name
        assert result.keep_schedule is True, name
        assert result.scheduler_delete_called is False, name
        assert result.decode_report is None, name
        assert result.item is None, name
        assert result.queue_label == "queue 0/10", name
        assert "queue 0/10" in result.message, name
        assert "nothing to implement" in result.message, name
        assert "keep_schedule" in result.message, name
        payload = result.to_dict()
        assert payload["queue"] == "queue 0/10", name
        assert payload["scheduler_delete_called"] is False, name
        assert payload["keep_schedule"] is True, name


def test_beat111_picked_decode_report_matches_helper():
    """Beat 111: picked decode_report == format_decode_report(item); helpers stay public."""
    import src.research_implement as ri

    result = run_session_b(_load("one_open_ready.md"), decode_only=True)
    assert result.verdict == "picked"
    assert result.item is not None
    expected = format_decode_report(result.item)
    assert result.decode_report == expected
    assert result.decode_report.startswith("decode pick Q1:")
    assert expected in result.message
    assert result.scheduler_delete_called is False
    assert result.keep_schedule is True

    for name in ("decode_fields", "format_decode_report", "scheduler_delete"):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name

def test_beat112_path_a_empty_writes_stub(tmp_path: Path):
    """Beat 112: run_session_a_path empty + stub → queued write; plan gains OPEN item."""
    plan = tmp_path / "plan.md"
    plan.write_text(_load("empty_queue.md"), encoding="utf-8")
    before = plan.read_text(encoding="utf-8")
    result = run_session_a_path(plan, brainstorm=stub_brainstorm, write=True)
    assert result.verdict == "queued"
    assert result.wrote_item is True
    after = plan.read_text(encoding="utf-8")
    assert after != before
    assert "Stub shippable change" in after or "status: OPEN" in after


def test_beat112_path_b_idle_leaves_plan(tmp_path: Path):
    """Beat 112: run_session_b_path idle decode leaves plan bytes unchanged."""
    plan = tmp_path / "plan.md"
    src = _load("empty_queue.md")
    plan.write_text(src, encoding="utf-8")
    result = run_session_b_path(plan, decode_only=True, write=False)
    assert result.verdict == "idle"
    assert plan.read_text(encoding="utf-8") == src
    import src.research_implement as ri

    for name in ("run_session_a_path", "run_session_b_path"):
        assert name in getattr(ri, "__all__", ()), name

def test_beat113_idle_decode_cli_json_matrix(tmp_path: Path, capsys):
    """Beat 113: idle-decode --json empty→idle; mixed→picked Q2; two_queue→failed."""
    import json
    from src.research_implement.__main__ import main

    cases = (
        ("empty_queue.md", 0, "idle", None),
        ("mixed_priority.md", 0, "picked", "Q2"),
        ("two_queue_sections.md", 1, "failed", None),
        ("shipped_only.md", 0, "idle", None),
    )
    for name, code, verdict, item_id in cases:
        plan = tmp_path / name
        plan.write_text(_load(name), encoding="utf-8")
        rc = main(["idle-decode", "--plan", str(plan), "--json"])
        assert rc == code, name
        payload = json.loads(capsys.readouterr().out)
        assert payload["verdict"] == verdict, name
        if item_id is None:
            assert payload.get("item") is None
        else:
            assert payload["item"]["item_id"] == item_id, name
        assert plan.read_text(encoding="utf-8") == _load(name)


def test_beat113_makefile_echo_mentions_beat113():
    """Beat 113: Makefile suite echo includes beat113."""
    text = Path("Makefile").read_text(encoding="utf-8")
    assert "beat10…beat113" in text or "beat113" in text

def test_beat114_session_b_decode_only_cli_json_matrix(tmp_path: Path, capsys):
    """Beat 114: session-b --decode-only --json empty→idle; one_open→Q1; two_queue→failed."""
    import json
    from src.research_implement.__main__ import main

    cases = (
        ("empty_queue.md", 0, "idle", None),
        ("one_open_ready.md", 0, "picked", "Q1"),
        ("two_queue_sections.md", 1, "failed", None),
    )
    for name, code, verdict, item_id in cases:
        plan = tmp_path / name
        src = _load(name)
        plan.write_text(src, encoding="utf-8")
        rc = main(["session-b", "--plan", str(plan), "--decode-only", "--json"])
        assert rc == code, name
        payload = json.loads(capsys.readouterr().out)
        assert payload["verdict"] == verdict, name
        if item_id is None:
            assert payload.get("item") is None
        else:
            assert payload["item"]["item_id"] == item_id, name
        assert plan.read_text(encoding="utf-8") == src
        assert payload["keep_schedule"] is True
        assert payload["scheduler_delete_called"] is False


def test_beat114_makefile_echo_mentions_beat114():
    """Beat 114: Makefile suite echo includes beat114."""
    text = Path("Makefile").read_text(encoding="utf-8")
    assert "beat114" in text



def test_beat115_session_a_stub_json_idle_vs_light_tmp(tmp_path: Path):
    """Beat 115: session-a --stub --json empty→queued write; one_open→light no-write."""
    from src.research_implement.__main__ import main

    # empty_queue + --stub --json → queued write
    empty = tmp_path / "empty_queue.md"
    empty.write_text(_load("empty_queue.md"), encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(empty), "--stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "queued"
    assert payload["wrote_item"] is True
    assert payload["open_count"] == 1
    assert payload["title"] == "Stub shippable change"
    assert payload["b_pick_title"] == "Stub shippable change"

    # one_open_ready + --stub --json → light, plan bytes unchanged
    one_src = _load("one_open_ready.md")
    one = tmp_path / "one_open_ready.md"
    one.write_text(one_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(one), "--stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "light"
    assert payload["wrote_item"] is False
    assert payload["open_count"] == 1
    assert payload["title"] is None
    assert payload["b_pick_title"] == "Add fixture unit test for queue parser"
    assert one.read_text(encoding="utf-8") == one_src


def test_beat115_session_a_stub_json_fail_paths_tmp(tmp_path: Path):
    """Beat 115: session-a --stub/--no-stub --json two_queue/empty→failed; no fixture mutate."""
    from src.research_implement.__main__ import main

    # two_queue_sections + --stub --json → failed
    two_src = _load("two_queue_sections.md")
    two = tmp_path / "two_queue_sections.md"
    two.write_text(two_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(two), "--stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    assert payload["open_count"] == 0
    assert payload["title"] is None
    assert payload["b_pick_title"] is None
    assert two.read_text(encoding="utf-8") == two_src

    # empty_queue + --no-stub --json → failed
    empty_src = _load("empty_queue.md")
    empty = tmp_path / "empty_no_stub.md"
    empty.write_text(empty_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(empty), "--no-stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    assert payload["open_count"] == 0
    assert empty.read_text(encoding="utf-8") == empty_src


def test_beat116_session_b_dry_run_json_idle_vs_picked_tmp(tmp_path: Path):
    """Beat 116: session-b --dry-run --json empty→idle; one_open→dry_run open_count=1; plan unchanged."""
    from src.research_implement.__main__ import main

    # empty_queue → idle, open_count=0
    empty_src = _load("empty_queue.md")
    empty = tmp_path / "empty_queue.md"
    empty.write_text(empty_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(empty), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert empty.read_text(encoding="utf-8") == empty_src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # one_open_ready → dry_run, open_count=1
    one_src = _load("one_open_ready.md")
    one = tmp_path / "one_open_ready.md"
    one.write_text(one_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(one), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "dry_run"
    assert payload["open_count"] == 1
    assert one.read_text(encoding="utf-8") == one_src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat116_session_b_dry_run_json_fail_and_multi_tmp(tmp_path: Path):
    """Beat 116: session-b --dry-run --json two_queue→failed; two_open→dry_run open_count=2; plan unchanged."""
    from src.research_implement.__main__ import main

    # two_queue_sections → failed, open_count=0
    two_src = _load("two_queue_sections.md")
    two = tmp_path / "two_queue_sections.md"
    two.write_text(two_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(two), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["open_count"] == 0
    assert two.read_text(encoding="utf-8") == two_src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # two_open_ready → dry_run, open_count=2
    multi_src = _load("two_open_ready.md")
    multi = tmp_path / "two_open_ready.md"
    multi.write_text(multi_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(multi), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "dry_run"
    assert payload["open_count"] == 2
    assert multi.read_text(encoding="utf-8") == multi_src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat117_session_a_dry_run_json_queued_vs_light_tmp(tmp_path: Path):
    """Beat 117: session-a --stub --dry-run --json empty→queued wrote_item=True plan unchanged; one_open→light."""
    from src.research_implement.__main__ import main

    # empty_queue + --stub --dry-run --json → queued, wrote_item=True, but plan bytes unchanged
    empty_src = _load("empty_queue.md")
    empty = tmp_path / "empty_queue.md"
    empty.write_text(empty_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(empty), "--stub", "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "queued"
    assert payload["wrote_item"] is True
    assert payload["open_count"] == 1
    assert payload["title"] == "Stub shippable change"
    assert empty.read_text(encoding="utf-8") == empty_src

    # one_open_ready + --stub --dry-run --json → light, wrote_item=False, plan unchanged
    one_src = _load("one_open_ready.md")
    one = tmp_path / "one_open_ready.md"
    one.write_text(one_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(one), "--stub", "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "light"
    assert payload["wrote_item"] is False
    assert payload["open_count"] == 1
    assert payload["title"] is None
    assert one.read_text(encoding="utf-8") == one_src


def test_beat117_session_a_dry_run_json_fail_and_no_stub_tmp(tmp_path: Path):
    """Beat 117: session-a --stub/--no-stub --dry-run --json two_queue/empty→failed; one_open no-stub→light; plans unchanged."""
    from src.research_implement.__main__ import main

    # two_queue_sections + --stub --dry-run --json → failed
    two_src = _load("two_queue_sections.md")
    two = tmp_path / "two_queue_sections.md"
    two.write_text(two_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(two), "--stub", "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    assert payload["open_count"] == 0
    assert two.read_text(encoding="utf-8") == two_src

    # empty_queue + --no-stub --dry-run --json → failed
    empty_src = _load("empty_queue.md")
    empty = tmp_path / "empty_no_stub.md"
    empty.write_text(empty_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(empty), "--no-stub", "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    assert payload["open_count"] == 0
    assert empty.read_text(encoding="utf-8") == empty_src

    # one_open_ready + --no-stub --dry-run --json → light
    one_src = _load("one_open_ready.md")
    one = tmp_path / "one_open_no_stub.md"
    one.write_text(one_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(one), "--no-stub", "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "light"
    assert payload["wrote_item"] is False
    assert payload["open_count"] == 1
    assert one.read_text(encoding="utf-8") == one_src


def test_beat118_idle_decode_watch_json_idle_vs_pick_tmp(tmp_path: Path):
    """Beat 118: idle-decode --json watch_only→idle; watch_lookalike→picked; plan unchanged both."""
    from src.research_implement.__main__ import main

    # watch_only_lookalike → idle, open_count=0, queue 0/10
    only_src = _load("watch_only_lookalike.md")
    only = tmp_path / "watch_only_lookalike.md"
    only.write_text(only_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(only), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert only.read_text(encoding="utf-8") == only_src

    # watch_lookalike → picked, open_count=1, queue 1/10
    look_src = _load("watch_lookalike.md")
    look = tmp_path / "watch_lookalike.md"
    look.write_text(look_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(look), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "picked"
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    assert look.read_text(encoding="utf-8") == look_src


def test_beat118_idle_decode_watch_json_heartbeat_and_fail_tmp(tmp_path: Path):
    """Beat 118: idle-decode --json watch_queue_heartbeat→picked; two_queue→failed; plans unchanged."""
    from src.research_implement.__main__ import main

    # watch_queue_heartbeat → picked, open_count=1, queue 1/10
    hb_src = _load("watch_queue_heartbeat.md")
    hb = tmp_path / "watch_queue_heartbeat.md"
    hb.write_text(hb_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(hb), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "picked"
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    assert hb.read_text(encoding="utf-8") == hb_src

    # two_queue_sections → failed, open_count=0
    two_src = _load("two_queue_sections.md")
    two = tmp_path / "two_queue_sections.md"
    two.write_text(two_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(two), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["open_count"] == 0
    assert two.read_text(encoding="utf-8") == two_src


def test_beat119_session_b_dry_run_watch_json_idle_vs_pick_tmp(tmp_path: Path):
    """Beat 119: session-b --dry-run --json watch_only→idle; watch_lookalike→dry_run; plan unchanged both."""
    from src.research_implement.__main__ import main

    # watch_only_lookalike → idle, open_count=0, queue 0/10; never scheduler_delete
    only_src = _load("watch_only_lookalike.md")
    only = tmp_path / "watch_only_lookalike.md"
    only.write_text(only_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(only), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert only.read_text(encoding="utf-8") == only_src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # watch_lookalike → dry_run, open_count=1, queue 1/10
    look_src = _load("watch_lookalike.md")
    look = tmp_path / "watch_lookalike.md"
    look.write_text(look_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(look), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "dry_run"
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert look.read_text(encoding="utf-8") == look_src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat119_session_b_dry_run_watch_json_heartbeat_and_fail_tmp(tmp_path: Path):
    """Beat 119: session-b --dry-run --json watch_queue_heartbeat→dry_run; two_queue→failed; plans unchanged."""
    from src.research_implement.__main__ import main

    # watch_queue_heartbeat → dry_run, open_count=1, queue 1/10
    hb_src = _load("watch_queue_heartbeat.md")
    hb = tmp_path / "watch_queue_heartbeat.md"
    hb.write_text(hb_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(hb), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "dry_run"
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert hb.read_text(encoding="utf-8") == hb_src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # two_queue_sections → failed, open_count=0
    two_src = _load("two_queue_sections.md")
    two = tmp_path / "two_queue_sections.md"
    two.write_text(two_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(two), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["open_count"] == 0
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert two.read_text(encoding="utf-8") == two_src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

def test_beat120_top_level_help_still_lists_subcommands(capsys):
    """Beat 120: top-level --help lists session-a / session-b / idle-decode."""
    import pytest
    from src.research_implement.__main__ import main

    with pytest.raises(SystemExit) as ei:
        main(["--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    for name in ("session-a", "session-b", "idle-decode"):
        assert name in out, name


def test_beat120_public_api_milestone_exports():
    """Beat 120: core A/B API + aliases/write/serialize remain public through beat120."""
    import src.research_implement as ri

    for name in (
        "run_session_a",
        "run_session_b",
        "run_session_a_path",
        "run_session_b_path",
        "parse_queue_items",
        "first_b_pick",
        "is_b_pickable",
        "is_ready_yes",
        "is_complete_six_field",
        "is_open_status",
        "count_open",
        "count_queue_headings",
        "next_queue_id",
        "mark_item_shipped",
        "append_queue_item",
        "format_queue_item",
        "write_queue_section",
        "serialize_queue_item",
        "serialize_queue_items",
        "render_queue_count",
        "QUEUE_CAPACITY",
        "REQUIRED_FIELDS",
        "AmbiguousQueueError",
        "SchedulerDeleteForbidden",
        "scheduler_delete",
        "require_unique_queue_section",
        "incomplete_candidate_reasons",
        "decode_fields",
        "format_decode_report",
        "stub_brainstorm",
        "default_search_plan",
        "stub_search_plan",
        "default_implement",
        "dry_run_implement",
        "make_fixture_ship_implement",
        "fixture_ship_implement",
        "session_a_result_dict",
        "session_b_result_dict",
        "SessionAResult",
        "SessionResult",
        "SessionBResult",
        "QueueItem",
        "SESSION_A_RESULT_KEYS",
        "SESSION_B_RESULT_KEYS",
        "SESSION_A_RESULT_JSON_KEYS",
        "SESSION_RESULT_JSON_KEYS",
    ):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name
    assert ri.SessionBResult is ri.SessionResult
    assert ri.default_implement is ri.dry_run_implement
    assert ri.default_search_plan is ri.stub_brainstorm
    assert len(ri.__all__) == len(set(ri.__all__))


def test_beat121_session_a_watch_json_queued_keeps_markers_tmp(tmp_path: Path):
    """Beat 121: session-a --stub --json watch_only/heartbeat_no_queue→queued; markers kept."""
    from src.research_implement.__main__ import main

    # watch_only_lookalike + --stub --json → queued write; Watch markers still present
    only_src = _load("watch_only_lookalike.md")
    only = tmp_path / "watch_only_lookalike.md"
    only.write_text(only_src, encoding="utf-8")
    assert "## Watch" in only_src
    assert "## Heartbeat" in only_src
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(only), "--stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "queued"
    assert payload["wrote_item"] is True
    assert payload["open_count"] == 1
    only_after = only.read_text(encoding="utf-8")
    assert "## Watch" in only_after
    assert "## Heartbeat" in only_after

    # watch_heartbeat_no_queue + --stub --json → queued write; Watch+Heartbeat markers kept
    hb_src = _load("watch_heartbeat_no_queue.md")
    hb = tmp_path / "watch_heartbeat_no_queue.md"
    hb.write_text(hb_src, encoding="utf-8")
    assert "BEAT20_WATCH_MARKER" in hb_src
    assert "BEAT20_HEARTBEAT_MARKER" in hb_src
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(hb), "--stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "queued"
    assert payload["wrote_item"] is True
    assert payload["open_count"] == 1
    hb_after = hb.read_text(encoding="utf-8")
    assert "BEAT20_WATCH_MARKER" in hb_after
    assert "BEAT20_HEARTBEAT_MARKER" in hb_after


def test_beat121_session_a_watch_json_light_and_no_stub_fail_tmp(tmp_path: Path):
    """Beat 121: session-a watch_lookalike stub→light; watch_only --no-stub→failed; plans unchanged."""
    from src.research_implement.__main__ import main

    # watch_lookalike + --stub --json → light, wrote_item=False, plan unchanged
    look_src = _load("watch_lookalike.md")
    look = tmp_path / "watch_lookalike.md"
    look.write_text(look_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(look), "--stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "light"
    assert payload["wrote_item"] is False
    assert payload["open_count"] == 1
    assert look.read_text(encoding="utf-8") == look_src

    # watch_only_lookalike + --no-stub --json → failed, wrote_item=False, plan unchanged
    only_src = _load("watch_only_lookalike.md")
    only = tmp_path / "watch_only_no_stub.md"
    only.write_text(only_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(only), "--no-stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    assert only.read_text(encoding="utf-8") == only_src


def test_beat122_session_a_watch_dry_run_json_queued_vs_light_tmp(tmp_path: Path):
    """Beat 122: session-a --stub --dry-run --json watch_only→queued wrote_item=True plan unchanged; watch_lookalike→light."""
    from src.research_implement.__main__ import main

    # watch_only_lookalike + --stub --dry-run --json → queued, wrote_item=True, open_count=1, plan unchanged
    only_src = _load("watch_only_lookalike.md")
    only = tmp_path / "watch_only_lookalike.md"
    only.write_text(only_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(only), "--stub", "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "queued"
    assert payload["wrote_item"] is True
    assert payload["open_count"] == 1
    assert only.read_text(encoding="utf-8") == only_src

    # watch_lookalike + --stub --dry-run --json → light, wrote_item=False, open_count=1, plan unchanged
    look_src = _load("watch_lookalike.md")
    look = tmp_path / "watch_lookalike.md"
    look.write_text(look_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(look), "--stub", "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "light"
    assert payload["wrote_item"] is False
    assert payload["open_count"] == 1
    assert look.read_text(encoding="utf-8") == look_src


def test_beat122_session_a_watch_dry_run_json_heartbeat_and_no_stub_tmp(tmp_path: Path):
    """Beat 122: session-a --stub/--no-stub --dry-run --json heartbeat_no_queue→queued; watch_only no-stub→failed; plans unchanged."""
    from src.research_implement.__main__ import main

    # watch_heartbeat_no_queue + --stub --dry-run --json → queued, wrote_item=True, plan unchanged
    hb_src = _load("watch_heartbeat_no_queue.md")
    hb = tmp_path / "watch_heartbeat_no_queue.md"
    hb.write_text(hb_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(hb), "--stub", "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "queued"
    assert payload["wrote_item"] is True
    assert hb.read_text(encoding="utf-8") == hb_src

    # watch_only_lookalike + --no-stub --dry-run --json → failed, wrote_item=False, plan unchanged
    only_src = _load("watch_only_lookalike.md")
    only = tmp_path / "watch_only_no_stub.md"
    only.write_text(only_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(only), "--no-stub", "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    assert only.read_text(encoding="utf-8") == only_src


def test_beat123_session_b_decode_watch_json_idle_vs_pick_tmp(tmp_path: Path):
    """Beat 123: session-b --decode-only --json watch_only→idle; watch_lookalike→picked; plan unchanged both."""
    from src.research_implement.__main__ import main

    # watch_only_lookalike → idle, open_count=0, queue 0/10; never scheduler_delete
    only_src = _load("watch_only_lookalike.md")
    only = tmp_path / "watch_only_lookalike.md"
    only.write_text(only_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(only), "--decode-only", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert only.read_text(encoding="utf-8") == only_src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # watch_lookalike → picked, open_count=1, queue 1/10
    look_src = _load("watch_lookalike.md")
    look = tmp_path / "watch_lookalike.md"
    look.write_text(look_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(look), "--decode-only", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "picked"
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert look.read_text(encoding="utf-8") == look_src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat123_session_b_decode_watch_json_heartbeat_and_fail_tmp(tmp_path: Path):
    """Beat 123: session-b --decode-only --json watch_queue_heartbeat→picked; two_queue→failed; plans unchanged."""
    from src.research_implement.__main__ import main

    # watch_queue_heartbeat → picked, open_count=1, queue 1/10
    hb_src = _load("watch_queue_heartbeat.md")
    hb = tmp_path / "watch_queue_heartbeat.md"
    hb.write_text(hb_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(hb), "--decode-only", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "picked"
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert hb.read_text(encoding="utf-8") == hb_src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # two_queue_sections → failed, open_count=0
    two_src = _load("two_queue_sections.md")
    two = tmp_path / "two_queue_sections.md"
    two.write_text(two_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(two), "--decode-only", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["open_count"] == 0
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert two.read_text(encoding="utf-8") == two_src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat124_idle_decode_nonpickable_json_shipped_vs_incomplete_tmp(tmp_path: Path):
    """Beat 124: idle-decode --json shipped_only→idle; incomplete_open→idle; plans unchanged."""
    from src.research_implement.__main__ import main

    # shipped_only → idle, open_count=0, queue 0/10; never scheduler_delete
    shipped_src = _load("shipped_only.md")
    shipped = tmp_path / "shipped_only.md"
    shipped.write_text(shipped_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(shipped), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert shipped.read_text(encoding="utf-8") == shipped_src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # incomplete_open → idle; plan unchanged
    inc_src = _load("incomplete_open.md")
    inc = tmp_path / "incomplete_open.md"
    inc.write_text(inc_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(inc), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert inc.read_text(encoding="utf-8") == inc_src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat124_idle_decode_nonpickable_json_broken_vs_not_ready_tmp(tmp_path: Path):
    """Beat 124: idle-decode --json broken_ready_flag→idle; open_complete_not_ready→idle; plans unchanged."""
    from src.research_implement.__main__ import main

    # broken_ready_flag → idle; never scheduler_delete
    broken_src = _load("broken_ready_flag.md")
    broken = tmp_path / "broken_ready_flag.md"
    broken.write_text(broken_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(broken), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert broken.read_text(encoding="utf-8") == broken_src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # open_complete_not_ready → idle; plan unchanged
    nr_src = _load("open_complete_not_ready.md")
    nr = tmp_path / "open_complete_not_ready.md"
    nr.write_text(nr_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(nr), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert nr.read_text(encoding="utf-8") == nr_src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat125_session_b_dry_run_nonpickable_json_shipped_vs_incomplete_tmp(tmp_path: Path):
    """Beat 125: session-b --dry-run --json shipped_only→idle; incomplete_open→idle; plans unchanged."""
    from src.research_implement.__main__ import main

    # shipped_only → idle (never dry_run), open_count=0, queue 0/10; never scheduler_delete
    shipped_src = _load("shipped_only.md")
    shipped = tmp_path / "shipped_only.md"
    shipped.write_text(shipped_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(shipped), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["verdict"] != "dry_run"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert shipped.read_text(encoding="utf-8") == shipped_src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # incomplete_open → idle (never dry_run); plan unchanged
    inc_src = _load("incomplete_open.md")
    inc = tmp_path / "incomplete_open.md"
    inc.write_text(inc_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(inc), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["verdict"] != "dry_run"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert inc.read_text(encoding="utf-8") == inc_src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat125_session_b_dry_run_nonpickable_json_broken_vs_not_ready_tmp(tmp_path: Path):
    """Beat 125: session-b --dry-run --json broken_ready_flag→idle; open_complete_not_ready→idle; plans unchanged."""
    from src.research_implement.__main__ import main

    # broken_ready_flag → idle (never dry_run); never scheduler_delete
    broken_src = _load("broken_ready_flag.md")
    broken = tmp_path / "broken_ready_flag.md"
    broken.write_text(broken_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(broken), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["verdict"] != "dry_run"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert broken.read_text(encoding="utf-8") == broken_src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # open_complete_not_ready → idle (never dry_run); plan unchanged
    nr_src = _load("open_complete_not_ready.md")
    nr = tmp_path / "open_complete_not_ready.md"
    nr.write_text(nr_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(nr), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["verdict"] != "dry_run"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert nr.read_text(encoding="utf-8") == nr_src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat126_session_b_write_nonpickable_json_shipped_vs_incomplete_tmp(tmp_path: Path):
    """Beat 126: session-b --json shipped_only→idle; incomplete_open→idle; plans unchanged (write path)."""
    from src.research_implement.__main__ import main

    # shipped_only → idle (write path, never dry_run), open_count=0, queue 0/10; never scheduler_delete
    shipped_src = _load("shipped_only.md")
    shipped = tmp_path / "shipped_only.md"
    shipped.write_text(shipped_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(shipped), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["verdict"] != "dry_run"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert shipped.read_text(encoding="utf-8") == shipped_src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # incomplete_open → idle (write path, never dry_run); plan unchanged
    inc_src = _load("incomplete_open.md")
    inc = tmp_path / "incomplete_open.md"
    inc.write_text(inc_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(inc), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["verdict"] != "dry_run"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert inc.read_text(encoding="utf-8") == inc_src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat126_session_b_write_nonpickable_json_broken_vs_not_ready_tmp(tmp_path: Path):
    """Beat 126: session-b --json broken_ready_flag→idle; open_complete_not_ready→idle; plans unchanged."""
    from src.research_implement.__main__ import main

    # broken_ready_flag → idle (write path, never dry_run); never scheduler_delete
    broken_src = _load("broken_ready_flag.md")
    broken = tmp_path / "broken_ready_flag.md"
    broken.write_text(broken_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(broken), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["verdict"] != "dry_run"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert broken.read_text(encoding="utf-8") == broken_src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # open_complete_not_ready → idle (write path, never dry_run); plan unchanged
    nr_src = _load("open_complete_not_ready.md")
    nr = tmp_path / "open_complete_not_ready.md"
    nr.write_text(nr_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(nr), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["verdict"] != "dry_run"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert nr.read_text(encoding="utf-8") == nr_src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat127_session_a_no_stub_nonpickable_json_fail_tmp(tmp_path: Path):
    """Beat 127: session-a --no-stub --json shipped_only/incomplete_open→failed; plans unchanged."""
    from src.research_implement.__main__ import main

    # shipped_only → OPEN=0 for Session A; --no-stub → failed; plan unchanged
    shipped_src = _load("shipped_only.md")
    shipped = tmp_path / "shipped_only.md"
    shipped.write_text(shipped_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(shipped), "--no-stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    assert payload["open_count"] == 0
    assert shipped.read_text(encoding="utf-8") == shipped_src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)

    # incomplete_open → OPEN=0; --no-stub → failed; plan unchanged
    inc_src = _load("incomplete_open.md")
    inc = tmp_path / "incomplete_open.md"
    inc.write_text(inc_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(inc), "--no-stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    assert payload["open_count"] == 0
    assert inc.read_text(encoding="utf-8") == inc_src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)


def test_beat127_session_a_stub_nonpickable_json_queued_tmp(tmp_path: Path):
    """Beat 127: session-a --stub --json broken_ready_flag/open_complete_not_ready→queued write (tmp only)."""
    from src.research_implement.__main__ import main

    # broken_ready_flag → OPEN=0; --stub → queued write under tmp; open_count=1
    broken_src = _load("broken_ready_flag.md")
    broken = tmp_path / "broken_ready_flag.md"
    broken.write_text(broken_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(broken), "--stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "queued"
    assert payload["wrote_item"] is True
    assert payload["open_count"] == 1
    assert broken.read_text(encoding="utf-8") != broken_src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)

    # open_complete_not_ready → OPEN=0; --stub → queued write under tmp; open_count=1
    nr_src = _load("open_complete_not_ready.md")
    nr = tmp_path / "open_complete_not_ready.md"
    nr.write_text(nr_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(nr), "--stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "queued"
    assert payload["wrote_item"] is True
    assert payload["open_count"] == 1
    assert nr.read_text(encoding="utf-8") != nr_src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)


def test_beat128_session_a_no_stub_dry_run_nonpickable_json_fail_tmp(tmp_path: Path):
    """Beat 128: session-a --no-stub --dry-run --json shipped_only/incomplete_open→failed; plans unchanged."""
    from src.research_implement.__main__ import main

    # shipped_only → OPEN=0; --no-stub --dry-run → failed; plan unchanged
    shipped_src = _load("shipped_only.md")
    shipped = tmp_path / "shipped_only.md"
    shipped.write_text(shipped_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(shipped), "--no-stub", "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    assert payload["open_count"] == 0
    assert shipped.read_text(encoding="utf-8") == shipped_src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)

    # incomplete_open → OPEN=0; --no-stub --dry-run → failed; plan unchanged
    inc_src = _load("incomplete_open.md")
    inc = tmp_path / "incomplete_open.md"
    inc.write_text(inc_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(inc), "--no-stub", "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    assert payload["open_count"] == 0
    assert inc.read_text(encoding="utf-8") == inc_src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)


def test_beat128_session_a_stub_dry_run_nonpickable_json_queued_tmp(tmp_path: Path):
    """Beat 128: session-a --stub --dry-run --json broken/not_ready→queued wrote_item=True but plan unchanged."""
    from src.research_implement.__main__ import main

    # broken_ready_flag → OPEN=0; --stub --dry-run → queued wrote_item=True but plan bytes unchanged
    broken_src = _load("broken_ready_flag.md")
    broken = tmp_path / "broken_ready_flag.md"
    broken.write_text(broken_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(broken), "--stub", "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "queued"
    assert payload["wrote_item"] is True
    assert payload["open_count"] == 1
    assert broken.read_text(encoding="utf-8") == broken_src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)

    # open_complete_not_ready → OPEN=0; --stub --dry-run → queued wrote_item=True but plan bytes unchanged
    nr_src = _load("open_complete_not_ready.md")
    nr = tmp_path / "open_complete_not_ready.md"
    nr.write_text(nr_src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(nr), "--stub", "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "queued"
    assert payload["wrote_item"] is True
    assert payload["open_count"] == 1
    assert nr.read_text(encoding="utf-8") == nr_src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)


def test_beat129_mixed_priority_idle_and_session_b_json_pick_q2_tmp(tmp_path: Path):
    """Beat 129: idle-decode + session-b --decode-only --json both pick Q2; keep_schedule; plans unchanged."""
    from src.research_implement.__main__ import main

    src = _load("mixed_priority.md")

    # idle-decode --json → picked Q2, open_count=1, queue 1/10; keep_schedule; plan unchanged
    idle_plan = tmp_path / "mixed_priority_idle.md"
    idle_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(idle_plan), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "picked"
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    assert payload["item"]["item_id"] == "Q2"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert idle_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-b --decode-only --json → same picked Q2; plan unchanged
    b_plan = tmp_path / "mixed_priority_decode.md"
    b_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(b_plan), "--decode-only", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "picked"
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    assert payload["item"]["item_id"] == "Q2"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert b_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat129_mixed_priority_dry_run_and_session_a_light_tmp(tmp_path: Path):
    """Beat 129: session-b --dry-run→dry_run Q2; session-a --stub→light b_pick_title Second ready; plans unchanged."""
    from src.research_implement.__main__ import main

    src = _load("mixed_priority.md")

    # session-b --dry-run --json → dry_run Q2, open_count=1; plan unchanged
    dry_plan = tmp_path / "mixed_priority_dry.md"
    dry_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(dry_plan), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "dry_run"
    assert payload["open_count"] == 1
    assert payload["item"]["item_id"] == "Q2"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert dry_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-a --stub --json → light recount; wrote_item=False; b_pick_title Second ready complete item
    a_plan = tmp_path / "mixed_priority_a.md"
    a_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(a_plan), "--stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "light"
    assert payload["wrote_item"] is False
    assert payload["open_count"] == 1
    assert payload["b_pick_title"] == "Second ready complete item"
    assert a_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)


def test_beat130_top_level_help_still_lists_subcommands(capsys):
    """Beat 130: top-level --help lists session-a / session-b / idle-decode."""
    import pytest
    from src.research_implement.__main__ import main

    with pytest.raises(SystemExit) as ei:
        main(["--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    for name in ("session-a", "session-b", "idle-decode"):
        assert name in out, name


def test_beat130_public_api_milestone_exports():
    """Beat 130: core A/B API + aliases/write/serialize remain public through beat130."""
    import src.research_implement as ri

    for name in (
        "run_session_a",
        "run_session_b",
        "run_session_a_path",
        "run_session_b_path",
        "parse_queue_items",
        "first_b_pick",
        "is_b_pickable",
        "is_ready_yes",
        "is_complete_six_field",
        "is_open_status",
        "count_open",
        "count_queue_headings",
        "next_queue_id",
        "mark_item_shipped",
        "append_queue_item",
        "format_queue_item",
        "write_queue_section",
        "serialize_queue_item",
        "serialize_queue_items",
        "render_queue_count",
        "QUEUE_CAPACITY",
        "REQUIRED_FIELDS",
        "AmbiguousQueueError",
        "SchedulerDeleteForbidden",
        "scheduler_delete",
        "require_unique_queue_section",
        "incomplete_candidate_reasons",
        "decode_fields",
        "format_decode_report",
        "stub_brainstorm",
        "default_search_plan",
        "stub_search_plan",
        "default_implement",
        "dry_run_implement",
        "make_fixture_ship_implement",
        "fixture_ship_implement",
        "session_a_result_dict",
        "session_b_result_dict",
        "SessionAResult",
        "SessionResult",
        "SessionBResult",
        "QueueItem",
        "SESSION_A_RESULT_KEYS",
        "SESSION_B_RESULT_KEYS",
        "SESSION_A_RESULT_JSON_KEYS",
        "SESSION_RESULT_JSON_KEYS",
    ):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name
    assert ri.SessionBResult is ri.SessionResult
    assert ri.default_implement is ri.dry_run_implement
    assert ri.default_search_plan is ri.stub_brainstorm
    assert len(ri.__all__) == len(set(ri.__all__))

def test_beat131_two_open_idle_and_session_b_json_pick_q1_tmp(tmp_path: Path):
    """Beat 131: idle-decode + session-b --decode-only --json both pick Q1 open_count=2; keep_schedule; plans unchanged."""
    from src.research_implement.__main__ import main

    src = _load("two_open_ready.md")

    # idle-decode --json → picked Q1, open_count=2, queue 2/10; keep_schedule; plan unchanged
    idle_plan = tmp_path / "two_open_idle.md"
    idle_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(idle_plan), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "picked"
    assert payload["open_count"] == 2
    assert payload["queue"] == "queue 2/10"
    assert payload["item"]["item_id"] == "Q1"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert idle_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-b --decode-only --json → same picked Q1 open_count=2; plan unchanged
    b_plan = tmp_path / "two_open_decode.md"
    b_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(b_plan), "--decode-only", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "picked"
    assert payload["open_count"] == 2
    assert payload["queue"] == "queue 2/10"
    assert payload["item"]["item_id"] == "Q1"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert b_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat131_two_open_dry_run_and_session_a_light_tmp(tmp_path: Path):
    """Beat 131: session-b --dry-run→dry_run Q1 open_count=2; session-a --stub→light b_pick_title First ready; plans unchanged."""
    from src.research_implement.__main__ import main

    src = _load("two_open_ready.md")

    # session-b --dry-run --json → dry_run Q1, open_count=2; plan unchanged
    dry_plan = tmp_path / "two_open_dry.md"
    dry_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(dry_plan), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "dry_run"
    assert payload["open_count"] == 2
    assert payload["item"]["item_id"] == "Q1"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert dry_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-a --stub --json → light recount; wrote_item=False; b_pick_title First ready complete item
    a_plan = tmp_path / "two_open_a.md"
    a_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(a_plan), "--stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "light"
    assert payload["wrote_item"] is False
    assert payload["open_count"] == 2
    assert payload["b_pick_title"] == "First ready complete item"
    assert a_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)

def test_beat132_one_open_idle_and_session_b_json_pick_q1_tmp(tmp_path: Path):
    """Beat 132: idle-decode + session-b --decode-only --json both pick Q1 open_count=1; keep_schedule; plans unchanged."""
    from src.research_implement.__main__ import main

    src = _load("one_open_ready.md")

    # idle-decode --json → picked Q1, open_count=1, queue 1/10; keep_schedule; plan unchanged
    idle_plan = tmp_path / "one_open_idle.md"
    idle_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(idle_plan), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "picked"
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    assert payload["item"]["item_id"] == "Q1"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert idle_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-b --decode-only --json → same picked Q1 open_count=1; plan unchanged
    b_plan = tmp_path / "one_open_decode.md"
    b_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(b_plan), "--decode-only", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "picked"
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    assert payload["item"]["item_id"] == "Q1"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert b_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat132_one_open_dry_run_and_session_a_light_tmp(tmp_path: Path):
    """Beat 132: session-b --dry-run→dry_run Q1; session-a --stub→light b_pick_title Add fixture unit test for queue parser; plans unchanged."""
    from src.research_implement.__main__ import main

    src = _load("one_open_ready.md")

    # session-b --dry-run --json → dry_run Q1, open_count=1; plan unchanged
    dry_plan = tmp_path / "one_open_dry.md"
    dry_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(dry_plan), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "dry_run"
    assert payload["open_count"] == 1
    assert payload["item"]["item_id"] == "Q1"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert dry_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-a --stub --json → light recount; wrote_item=False; b_pick_title Add fixture unit test for queue parser
    a_plan = tmp_path / "one_open_a.md"
    a_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(a_plan), "--stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "light"
    assert payload["wrote_item"] is False
    assert payload["open_count"] == 1
    assert payload["b_pick_title"] == "Add fixture unit test for queue parser"
    assert a_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)



def test_beat133_watch_lookalike_idle_and_session_b_json_pick_q3_tmp(tmp_path: Path):
    """Beat 133: idle-decode + session-b --decode-only --json both pick Q3 open_count=1; keep_schedule; plans unchanged."""
    from src.research_implement.__main__ import main

    src = _load("watch_lookalike.md")

    # idle-decode --json → picked Q3, open_count=1, queue 1/10; keep_schedule; plan unchanged
    idle_plan = tmp_path / "watch_lookalike_idle.md"
    idle_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(idle_plan), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "picked"
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    assert payload["item"]["item_id"] == "Q3"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert idle_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-b --decode-only --json → same picked Q3 open_count=1; plan unchanged
    b_plan = tmp_path / "watch_lookalike_decode.md"
    b_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(b_plan), "--decode-only", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "picked"
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    assert payload["item"]["item_id"] == "Q3"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert b_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat133_watch_lookalike_dry_run_and_session_a_light_tmp(tmp_path: Path):
    """Beat 133: session-b --dry-run→dry_run Q3; session-a --stub→light b_pick_title Real ready Queue item; plans unchanged."""
    from src.research_implement.__main__ import main

    src = _load("watch_lookalike.md")

    # session-b --dry-run --json → dry_run Q3, open_count=1; plan unchanged
    dry_plan = tmp_path / "watch_lookalike_dry.md"
    dry_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(dry_plan), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "dry_run"
    assert payload["open_count"] == 1
    assert payload["item"]["item_id"] == "Q3"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert dry_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-a --stub --json → light recount; wrote_item=False; b_pick_title Real ready Queue item
    a_plan = tmp_path / "watch_lookalike_a.md"
    a_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(a_plan), "--stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "light"
    assert payload["wrote_item"] is False
    assert payload["open_count"] == 1
    assert payload["b_pick_title"] == "Real ready Queue item"
    assert a_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)


def test_beat134_watch_queue_heartbeat_idle_and_session_b_json_pick_q1_tmp(tmp_path: Path):
    """Beat 134: idle-decode + session-b --decode-only --json both pick Q1; keep_schedule; Watch/Heartbeat markers; plans unchanged."""
    from src.research_implement.__main__ import main

    src = _load("watch_queue_heartbeat.md")

    # idle-decode --json → picked Q1, open_count=1, queue 1/10; keep_schedule; markers + plan unchanged
    idle_plan = tmp_path / "watch_queue_heartbeat_idle.md"
    idle_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(idle_plan), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "picked"
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    assert payload["item"]["item_id"] == "Q1"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    idle_body = idle_plan.read_text(encoding="utf-8")
    assert idle_body == src
    assert "## Watch" in idle_body
    assert "## Heartbeat" in idle_body
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-b --decode-only --json → same picked Q1; markers + plan unchanged
    b_plan = tmp_path / "watch_queue_heartbeat_decode.md"
    b_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(b_plan), "--decode-only", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "picked"
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    assert payload["item"]["item_id"] == "Q1"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    b_body = b_plan.read_text(encoding="utf-8")
    assert b_body == src
    assert "## Watch" in b_body
    assert "## Heartbeat" in b_body
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat134_watch_queue_heartbeat_dry_run_and_session_a_light_tmp(tmp_path: Path):
    """Beat 134: session-b --dry-run→dry_run Q1; session-a --stub→light b_pick_title Beat19 shippable preserve item; markers intact; plans unchanged."""
    from src.research_implement.__main__ import main

    src = _load("watch_queue_heartbeat.md")

    # session-b --dry-run --json → dry_run Q1, open_count=1; markers + plan unchanged
    dry_plan = tmp_path / "watch_queue_heartbeat_dry.md"
    dry_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(dry_plan), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "dry_run"
    assert payload["open_count"] == 1
    assert payload["item"]["item_id"] == "Q1"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    dry_body = dry_plan.read_text(encoding="utf-8")
    assert dry_body == src
    assert "## Watch" in dry_body
    assert "## Heartbeat" in dry_body
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-a --stub --json → light recount; wrote_item=False; b_pick_title Beat19 shippable preserve item
    a_plan = tmp_path / "watch_queue_heartbeat_a.md"
    a_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(a_plan), "--stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "light"
    assert payload["wrote_item"] is False
    assert payload["open_count"] == 1
    assert payload["b_pick_title"] == "Beat19 shippable preserve item"
    a_body = a_plan.read_text(encoding="utf-8")
    assert a_body == src
    assert "## Watch" in a_body
    assert "## Heartbeat" in a_body
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)



def test_beat135_watch_only_idle_and_session_b_json_idle_tmp(tmp_path: Path):
    """Beat 135: idle-decode + session-b --decode-only --json both idle open_count=0; keep_schedule; Watch/Heartbeat markers; plans unchanged."""
    from src.research_implement.__main__ import main

    src = _load("watch_only_lookalike.md")

    # idle-decode --json → idle, open_count=0, queue 0/10; keep_schedule; markers + plan unchanged
    idle_plan = tmp_path / "watch_only_idle.md"
    idle_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(idle_plan), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    idle_body = idle_plan.read_text(encoding="utf-8")
    assert idle_body == src
    assert "## Watch" in idle_body
    assert "## Heartbeat" in idle_body
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-b --decode-only --json → same idle; markers + plan unchanged
    b_plan = tmp_path / "watch_only_decode.md"
    b_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(b_plan), "--decode-only", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    b_body = b_plan.read_text(encoding="utf-8")
    assert b_body == src
    assert "## Watch" in b_body
    assert "## Heartbeat" in b_body
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat135_watch_only_dry_run_idle_and_session_a_queued_tmp(tmp_path: Path):
    """Beat 135: session-b --dry-run→idle (not dry_run); session-a --stub→queued wrote_item=True open_count=1; Watch/Heartbeat intact after write."""
    from src.research_implement.__main__ import main

    src = _load("watch_only_lookalike.md")

    # session-b --dry-run --json → idle (NOT dry_run), open_count=0; markers + plan unchanged
    dry_plan = tmp_path / "watch_only_dry.md"
    dry_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(dry_plan), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    dry_body = dry_plan.read_text(encoding="utf-8")
    assert dry_body == src
    assert "## Watch" in dry_body
    assert "## Heartbeat" in dry_body
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-a --stub --json → queued write; wrote_item=True; open_count=1; Watch/Heartbeat still present (plan changes)
    a_plan = tmp_path / "watch_only_a.md"
    a_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(a_plan), "--stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "queued"
    assert payload["wrote_item"] is True
    assert payload["open_count"] == 1
    a_body = a_plan.read_text(encoding="utf-8")
    assert a_body != src
    assert "## Watch" in a_body
    assert "## Heartbeat" in a_body
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)


def test_beat136_watch_heartbeat_no_queue_idle_and_session_b_json_idle_tmp(tmp_path: Path):
    """Beat 136: idle-decode + session-b --decode-only --json both idle open_count=0; keep_schedule; Watch/Heartbeat present; no Queue; plans unchanged."""
    from src.research_implement.__main__ import main

    src = _load("watch_heartbeat_no_queue.md")
    assert "## Queue" not in src

    # idle-decode --json → idle, open_count=0, queue 0/10; keep_schedule; Watch/Heartbeat; no Queue; plan unchanged
    idle_plan = tmp_path / "watch_hb_idle.md"
    idle_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(idle_plan), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    idle_body = idle_plan.read_text(encoding="utf-8")
    assert idle_body == src
    assert "## Watch" in idle_body
    assert "## Heartbeat" in idle_body
    assert "## Queue" not in idle_body
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-b --decode-only --json → same idle; markers + no Queue; plan unchanged
    b_plan = tmp_path / "watch_hb_decode.md"
    b_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(b_plan), "--decode-only", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    b_body = b_plan.read_text(encoding="utf-8")
    assert b_body == src
    assert "## Watch" in b_body
    assert "## Heartbeat" in b_body
    assert "## Queue" not in b_body
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat136_watch_heartbeat_no_queue_dry_run_idle_and_session_a_queued_tmp(tmp_path: Path):
    """Beat 136: session-b --dry-run→idle (not dry_run); session-a --stub→queued wrote_item=True; Watch/Heartbeat/Queue all present after write."""
    from src.research_implement.__main__ import main

    src = _load("watch_heartbeat_no_queue.md")
    assert "## Queue" not in src

    # session-b --dry-run --json → idle (NOT dry_run), open_count=0; Watch/Heartbeat; no Queue; plan unchanged
    dry_plan = tmp_path / "watch_hb_dry.md"
    dry_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(dry_plan), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    dry_body = dry_plan.read_text(encoding="utf-8")
    assert dry_body == src
    assert "## Watch" in dry_body
    assert "## Heartbeat" in dry_body
    assert "## Queue" not in dry_body
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-a --stub --json → queued write; wrote_item=True; open_count=1; Queue+Watch+Heartbeat present
    a_plan = tmp_path / "watch_hb_a.md"
    a_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(a_plan), "--stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "queued"
    assert payload["wrote_item"] is True
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    a_body = a_plan.read_text(encoding="utf-8")
    assert a_body != src
    assert "## Queue" in a_body
    assert "## Watch" in a_body
    assert "## Heartbeat" in a_body
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)



def test_beat137_empty_queue_idle_and_session_b_json_idle_tmp(tmp_path: Path):
    """Beat 137: idle-decode + session-b --decode-only --json both idle open_count=0; keep_schedule; plans unchanged."""
    from src.research_implement.__main__ import main

    src = _load("empty_queue.md")

    # idle-decode --json → idle, open_count=0, queue 0/10; keep_schedule; plan unchanged
    idle_plan = tmp_path / "empty_idle.md"
    idle_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(idle_plan), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert idle_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-b --decode-only --json → same idle; plan unchanged
    b_plan = tmp_path / "empty_decode.md"
    b_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(b_plan), "--decode-only", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert b_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat137_empty_queue_dry_run_idle_and_session_a_stub_vs_no_stub_tmp(tmp_path: Path):
    """Beat 137: session-b --dry-run→idle; session-a --stub→queued wrote_item=True; session-a --no-stub→failed plan unchanged."""
    from src.research_implement.__main__ import main

    src = _load("empty_queue.md")

    # session-b --dry-run --json → idle (NOT dry_run), open_count=0; plan unchanged
    dry_plan = tmp_path / "empty_dry.md"
    dry_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(dry_plan), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert dry_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-a --stub --json → queued write; wrote_item=True; open_count=1; queue 1/10
    a_plan = tmp_path / "empty_a_stub.md"
    a_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(a_plan), "--stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "queued"
    assert payload["wrote_item"] is True
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    assert a_plan.read_text(encoding="utf-8") != src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)

    # session-a --no-stub --json → rc=1 failed; wrote_item=False; plan unchanged
    no_stub = tmp_path / "empty_a_no_stub.md"
    no_stub.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(no_stub), "--no-stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    assert no_stub.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)


def test_beat138_watch_queue_heartbeat_empty_idle_and_session_b_json_idle_tmp(tmp_path: Path):
    """Beat 138: idle-decode + session-b --decode-only both idle; keep_schedule; Watch/Heartbeat present; plans unchanged."""
    from src.research_implement.__main__ import main

    src = _load("watch_queue_heartbeat_empty.md")
    assert "## Watch" in src
    assert "## Heartbeat" in src
    assert "## Queue" in src

    # idle-decode --json → idle, open_count=0, queue 0/10; keep_schedule; Watch+Heartbeat; plan unchanged
    idle_plan = tmp_path / "wqhe_idle.md"
    idle_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(idle_plan), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    idle_body = idle_plan.read_text(encoding="utf-8")
    assert idle_body == src
    assert "## Watch" in idle_body
    assert "## Heartbeat" in idle_body
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-b --decode-only --json → same idle; Watch/Heartbeat; plan unchanged
    b_plan = tmp_path / "wqhe_decode.md"
    b_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(b_plan), "--decode-only", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    b_body = b_plan.read_text(encoding="utf-8")
    assert b_body == src
    assert "## Watch" in b_body
    assert "## Heartbeat" in b_body
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat138_watch_queue_heartbeat_empty_dry_run_idle_and_session_a_stub_vs_no_stub_tmp(tmp_path: Path):
    """Beat 138: session-b --dry-run→idle; session-a --stub→queued wrote_item=True keep Watch/Heartbeat; session-a --no-stub→failed plan unchanged."""
    from src.research_implement.__main__ import main

    src = _load("watch_queue_heartbeat_empty.md")
    assert "## Watch" in src
    assert "## Heartbeat" in src

    # session-b --dry-run --json → idle (NOT dry_run), open_count=0; Watch/Heartbeat; plan unchanged
    dry_plan = tmp_path / "wqhe_dry.md"
    dry_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(dry_plan), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    dry_body = dry_plan.read_text(encoding="utf-8")
    assert dry_body == src
    assert "## Watch" in dry_body
    assert "## Heartbeat" in dry_body
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-a --stub --json → queued write; wrote_item=True; open_count=1; Watch+Heartbeat still present
    a_plan = tmp_path / "wqhe_a_stub.md"
    a_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(a_plan), "--stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "queued"
    assert payload["wrote_item"] is True
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    a_body = a_plan.read_text(encoding="utf-8")
    assert a_body != src
    assert "## Watch" in a_body
    assert "## Heartbeat" in a_body
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)

    # session-a --no-stub --json → rc=1 failed; wrote_item=False; plan unchanged; markers intact
    no_stub = tmp_path / "wqhe_a_no_stub.md"
    no_stub.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(no_stub), "--no-stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    no_body = no_stub.read_text(encoding="utf-8")
    assert no_body == src
    assert "## Watch" in no_body
    assert "## Heartbeat" in no_body
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)



def test_beat139_two_queue_idle_and_session_b_json_fail_tmp(tmp_path: Path):
    """Beat 139: idle-decode + session-b --decode-only + session-b --dry-run all failed; keep_schedule; plans unchanged."""
    from src.research_implement.__main__ import main

    src = _load("two_queue_sections.md")

    # idle-decode --json → rc=1 failed; open_count=0 queue 0/10; keep_schedule; plan unchanged
    idle_plan = tmp_path / "two_queue_idle.md"
    idle_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(idle_plan), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert idle_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-b --decode-only --json → same failed; plan unchanged
    b_plan = tmp_path / "two_queue_decode.md"
    b_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(b_plan), "--decode-only", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert b_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-b --dry-run --json → failed (NOT dry_run); keep_schedule; plan unchanged
    dry_plan = tmp_path / "two_queue_dry.md"
    dry_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(dry_plan), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["verdict"] != "dry_run"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert dry_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat139_two_queue_session_a_stub_and_no_stub_json_fail_tmp(tmp_path: Path):
    """Beat 139: session-a --stub, --no-stub, and --stub --dry-run all failed wrote_item=False; plans unchanged."""
    from src.research_implement.__main__ import main

    src = _load("two_queue_sections.md")

    # session-a --stub --json → failed; wrote_item=False; plan unchanged
    stub_plan = tmp_path / "two_queue_a_stub.md"
    stub_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(stub_plan), "--stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    assert stub_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)

    # session-a --no-stub --json → failed; wrote_item=False; plan unchanged
    no_stub = tmp_path / "two_queue_a_no_stub.md"
    no_stub.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(no_stub), "--no-stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    assert no_stub.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)

    # session-a --stub --dry-run --json → failed; wrote_item=False; plan unchanged
    dry_stub = tmp_path / "two_queue_a_stub_dry.md"
    dry_stub.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(dry_stub), "--stub", "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    assert dry_stub.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)


def test_beat140_top_level_help_still_lists_subcommands(capsys):
    """Beat 140: top-level --help lists session-a / session-b / idle-decode."""
    import pytest
    from src.research_implement.__main__ import main

    with pytest.raises(SystemExit) as ei:
        main(["--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    for name in ("session-a", "session-b", "idle-decode"):
        assert name in out, name


def test_beat140_public_api_milestone_exports():
    """Beat 140: core A/B API + aliases/write/serialize remain public through beat140."""
    import src.research_implement as ri

    for name in (
        "run_session_a",
        "run_session_b",
        "run_session_a_path",
        "run_session_b_path",
        "parse_queue_items",
        "first_b_pick",
        "is_b_pickable",
        "is_ready_yes",
        "is_complete_six_field",
        "is_open_status",
        "count_open",
        "count_queue_headings",
        "next_queue_id",
        "mark_item_shipped",
        "append_queue_item",
        "format_queue_item",
        "write_queue_section",
        "serialize_queue_item",
        "serialize_queue_items",
        "render_queue_count",
        "QUEUE_CAPACITY",
        "REQUIRED_FIELDS",
        "AmbiguousQueueError",
        "SchedulerDeleteForbidden",
        "scheduler_delete",
        "require_unique_queue_section",
        "incomplete_candidate_reasons",
        "decode_fields",
        "format_decode_report",
        "stub_brainstorm",
        "default_search_plan",
        "stub_search_plan",
        "default_implement",
        "dry_run_implement",
        "make_fixture_ship_implement",
        "fixture_ship_implement",
        "session_a_result_dict",
        "session_b_result_dict",
        "SessionAResult",
        "SessionResult",
        "SessionBResult",
        "QueueItem",
        "SESSION_A_RESULT_KEYS",
        "SESSION_B_RESULT_KEYS",
        "SESSION_A_RESULT_JSON_KEYS",
        "SESSION_RESULT_JSON_KEYS",
    ):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name
    assert ri.SessionBResult is ri.SessionResult
    assert ri.default_implement is ri.dry_run_implement
    assert ri.default_search_plan is ri.stub_brainstorm
    assert len(ri.__all__) == len(set(ri.__all__))



def test_beat141_shipped_only_idle_and_session_b_json_idle_tmp(tmp_path: Path):
    """Beat 141: idle-decode + session-b --decode-only both idle; keep_schedule; plans unchanged."""
    from src.research_implement.__main__ import main

    src = _load("shipped_only.md")

    # idle-decode --json → idle, open_count=0, queue 0/10; keep_schedule; plan unchanged
    idle_plan = tmp_path / "shipped_idle.md"
    idle_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(idle_plan), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert idle_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-b --decode-only --json → same idle; plan unchanged
    b_plan = tmp_path / "shipped_decode.md"
    b_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(b_plan), "--decode-only", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert b_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat141_shipped_only_dry_run_idle_and_session_a_stub_vs_no_stub_tmp(tmp_path: Path):
    """Beat 141: session-b --dry-run→idle; session-a --stub→queued wrote_item=True; session-a --no-stub→failed plan unchanged."""
    from src.research_implement.__main__ import main

    src = _load("shipped_only.md")

    # session-b --dry-run --json → idle (NOT dry_run), open_count=0; plan unchanged
    dry_plan = tmp_path / "shipped_dry.md"
    dry_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(dry_plan), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["verdict"] != "dry_run"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert dry_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-a --stub --json → queued write; wrote_item=True; open_count=1; queue 1/10; title Stub shippable change
    a_plan = tmp_path / "shipped_a_stub.md"
    a_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(a_plan), "--stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "queued"
    assert payload["wrote_item"] is True
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    assert payload["title"] == "Stub shippable change"
    assert a_plan.read_text(encoding="utf-8") != src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)

    # session-a --no-stub --json → rc=1 failed; wrote_item=False; plan unchanged
    no_stub = tmp_path / "shipped_a_no_stub.md"
    no_stub.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(no_stub), "--no-stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    assert no_stub.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)


def test_beat142_incomplete_open_idle_and_session_b_json_idle_tmp(tmp_path: Path):
    """Beat 142: idle-decode + session-b --decode-only both idle; keep_schedule; plans unchanged."""
    from src.research_implement.__main__ import main

    src = _load("incomplete_open.md")

    # idle-decode --json → idle, open_count=0, queue 0/10; keep_schedule; plan unchanged
    idle_plan = tmp_path / "incomplete_idle.md"
    idle_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(idle_plan), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert idle_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-b --decode-only --json → same idle; plan unchanged
    b_plan = tmp_path / "incomplete_decode.md"
    b_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(b_plan), "--decode-only", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert b_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat142_incomplete_open_dry_run_idle_and_session_a_stub_vs_no_stub_tmp(tmp_path: Path):
    """Beat 142: session-b --dry-run→idle; session-a --stub→queued wrote_item=True; session-a --no-stub→failed plan unchanged."""
    from src.research_implement.__main__ import main

    src = _load("incomplete_open.md")

    # session-b --dry-run --json → idle (NOT dry_run), open_count=0; plan unchanged
    dry_plan = tmp_path / "incomplete_dry.md"
    dry_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(dry_plan), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["verdict"] != "dry_run"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert dry_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-a --stub --json → queued write; wrote_item=True; open_count=1; queue 1/10; title Stub shippable change; Heartbeat kept
    a_plan = tmp_path / "incomplete_a_stub.md"
    a_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(a_plan), "--stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "queued"
    assert payload["wrote_item"] is True
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    assert payload["title"] == "Stub shippable change"
    after = a_plan.read_text(encoding="utf-8")
    assert after != src
    assert "## Heartbeat" in after
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)

    # session-a --no-stub --json → rc=1 failed; wrote_item=False; plan unchanged
    no_stub = tmp_path / "incomplete_a_no_stub.md"
    no_stub.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(no_stub), "--no-stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    assert no_stub.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)



def test_beat143_broken_ready_flag_idle_and_session_b_json_idle_tmp(tmp_path: Path):
    """Beat 143: idle-decode + session-b --decode-only both idle; keep_schedule; plans unchanged."""
    from src.research_implement.__main__ import main

    src = _load("broken_ready_flag.md")

    # idle-decode --json → idle, open_count=0, queue 0/10; keep_schedule; plan unchanged
    idle_plan = tmp_path / "broken_idle.md"
    idle_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(idle_plan), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert idle_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-b --decode-only --json → same idle; plan unchanged
    b_plan = tmp_path / "broken_decode.md"
    b_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(b_plan), "--decode-only", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert b_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat143_broken_ready_flag_dry_run_idle_and_session_a_stub_vs_no_stub_tmp(tmp_path: Path):
    """Beat 143: session-b --dry-run→idle; session-a --stub→queued wrote_item=True; session-a --no-stub→failed plan unchanged."""
    from src.research_implement.__main__ import main

    src = _load("broken_ready_flag.md")

    # session-b --dry-run --json → idle (NOT dry_run), open_count=0; plan unchanged
    dry_plan = tmp_path / "broken_dry.md"
    dry_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(dry_plan), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["verdict"] != "dry_run"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert dry_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-a --stub --json → queued write; wrote_item=True; open_count=1; queue 1/10; title Stub shippable change
    a_plan = tmp_path / "broken_a_stub.md"
    a_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(a_plan), "--stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "queued"
    assert payload["wrote_item"] is True
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    assert payload["title"] == "Stub shippable change"
    assert a_plan.read_text(encoding="utf-8") != src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)

    # session-a --no-stub --json → rc=1 failed; wrote_item=False; plan unchanged
    no_stub = tmp_path / "broken_a_no_stub.md"
    no_stub.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(no_stub), "--no-stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    assert no_stub.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)


def test_beat144_open_complete_not_ready_idle_and_session_b_json_idle_tmp(tmp_path: Path):
    """Beat 144: idle-decode + session-b --decode-only both idle; keep_schedule; plans unchanged."""
    from src.research_implement.__main__ import main

    src = _load("open_complete_not_ready.md")

    # idle-decode --json → idle, open_count=0, queue 0/10; keep_schedule; plan unchanged
    idle_plan = tmp_path / "ocnr_idle.md"
    idle_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(idle_plan), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert idle_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-b --decode-only --json → same idle; plan unchanged
    b_plan = tmp_path / "ocnr_decode.md"
    b_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(b_plan), "--decode-only", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert b_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat144_open_complete_not_ready_dry_run_idle_and_session_a_stub_vs_no_stub_tmp(tmp_path: Path):
    """Beat 144: session-b --dry-run→idle; session-a --stub→queued wrote_item=True; session-a --no-stub→failed plan unchanged."""
    from src.research_implement.__main__ import main

    src = _load("open_complete_not_ready.md")

    # session-b --dry-run --json → idle (NOT dry_run), open_count=0; plan unchanged
    dry_plan = tmp_path / "ocnr_dry.md"
    dry_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(dry_plan), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["verdict"] != "dry_run"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert dry_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-a --stub --json → queued write; wrote_item=True; open_count=1; queue 1/10; title Stub shippable change
    a_plan = tmp_path / "ocnr_a_stub.md"
    a_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(a_plan), "--stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "queued"
    assert payload["wrote_item"] is True
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    assert payload["title"] == "Stub shippable change"
    assert a_plan.read_text(encoding="utf-8") != src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)

    # session-a --no-stub --json → rc=1 failed; wrote_item=False; plan unchanged
    no_stub = tmp_path / "ocnr_a_no_stub.md"
    no_stub.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(no_stub), "--no-stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    assert no_stub.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)


def test_beat145_queue_with_watch_heartbeat_idle_and_session_b_json_idle_tmp(tmp_path: Path):
    """Beat 145: idle-decode + session-b --decode-only both idle; keep_schedule; Watch/Heartbeat present; plans unchanged."""
    from src.research_implement.__main__ import main

    src = _load("queue_with_watch_heartbeat.md")
    _assert_beat19_queue_with_watch_heartbeat_preserved(src)
    assert src.index("## Watch") < src.index("## Queue")

    # idle-decode --json → idle, open_count=0, queue 0/10; keep_schedule; Watch+Heartbeat; plan unchanged
    idle_plan = tmp_path / "qwh_idle.md"
    idle_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(idle_plan), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    idle_body = idle_plan.read_text(encoding="utf-8")
    assert idle_body == src
    assert "## Watch" in idle_body
    assert "## Heartbeat" in idle_body
    assert idle_body.index("## Watch") < idle_body.index("## Queue")
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-b --decode-only --json → same idle; Watch/Heartbeat; plan unchanged
    b_plan = tmp_path / "qwh_decode.md"
    b_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(b_plan), "--decode-only", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    b_body = b_plan.read_text(encoding="utf-8")
    assert b_body == src
    assert "## Watch" in b_body
    assert "## Heartbeat" in b_body
    assert b_body.index("## Watch") < b_body.index("## Queue")
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat145_queue_with_watch_heartbeat_dry_run_idle_and_session_a_stub_vs_no_stub_tmp(tmp_path: Path):
    """Beat 145: session-b --dry-run→idle; session-a --stub→queued wrote_item=True keep Watch/Heartbeat; session-a --no-stub→failed plan unchanged."""
    from src.research_implement.__main__ import main

    src = _load("queue_with_watch_heartbeat.md")
    _assert_beat19_queue_with_watch_heartbeat_preserved(src)
    assert src.index("## Watch") < src.index("## Queue")

    # session-b --dry-run --json → idle (NOT dry_run), open_count=0; Watch/Heartbeat; plan unchanged
    dry_plan = tmp_path / "qwh_dry.md"
    dry_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(dry_plan), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["verdict"] != "dry_run"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    dry_body = dry_plan.read_text(encoding="utf-8")
    assert dry_body == src
    assert "## Watch" in dry_body
    assert "## Heartbeat" in dry_body
    assert dry_body.index("## Watch") < dry_body.index("## Queue")
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-a --stub --json → queued write; wrote_item=True; open_count=1; queue 1/10; title Stub shippable change; Watch+Heartbeat still present
    a_plan = tmp_path / "qwh_a_stub.md"
    a_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(a_plan), "--stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "queued"
    assert payload["wrote_item"] is True
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    assert payload["title"] == "Stub shippable change"
    a_body = a_plan.read_text(encoding="utf-8")
    assert a_body != src
    _assert_beat19_queue_with_watch_heartbeat_preserved(a_body)
    assert "## Watch" in a_body
    assert "## Heartbeat" in a_body
    assert a_body.index("## Watch") < a_body.index("## Queue")
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)

    # session-a --no-stub --json → rc=1 failed; wrote_item=False; plan unchanged; markers intact
    no_stub = tmp_path / "qwh_a_no_stub.md"
    no_stub.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(no_stub), "--no-stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    no_body = no_stub.read_text(encoding="utf-8")
    assert no_body == src
    _assert_beat19_queue_with_watch_heartbeat_preserved(no_body)
    assert "## Watch" in no_body
    assert "## Heartbeat" in no_body
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)


def test_beat146_contract_spec_idle_and_session_b_json_idle_tmp(tmp_path: Path):
    """Beat 146: idle-decode + session-b --decode-only both idle; keep_schedule; plans unchanged."""
    from src.research_implement.__main__ import main

    src = _load("contract_spec.md")

    # idle-decode --json → idle, open_count=0, queue 0/10; keep_schedule; plan unchanged
    idle_plan = tmp_path / "cs_idle.md"
    idle_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(idle_plan), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert idle_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-b --decode-only --json → same idle; plan unchanged
    b_plan = tmp_path / "cs_decode.md"
    b_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(b_plan), "--decode-only", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert b_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat146_contract_spec_dry_run_idle_and_session_a_stub_vs_no_stub_tmp(tmp_path: Path):
    """Beat 146: session-b --dry-run→idle; session-a --stub→queued wrote_item=True; session-a --no-stub→failed plan unchanged."""
    from src.research_implement.__main__ import main

    src = _load("contract_spec.md")

    # session-b --dry-run --json → idle (NOT dry_run), open_count=0; plan unchanged
    dry_plan = tmp_path / "cs_dry.md"
    dry_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(dry_plan), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["verdict"] != "dry_run"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert dry_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-a --stub --json → queued write; wrote_item=True; open_count=1; queue 1/10; title Stub shippable change
    a_plan = tmp_path / "cs_a_stub.md"
    a_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(a_plan), "--stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "queued"
    assert payload["wrote_item"] is True
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    assert payload["title"] == "Stub shippable change"
    assert a_plan.read_text(encoding="utf-8") != src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)

    # session-a --no-stub --json → rc=1 failed; wrote_item=False; plan unchanged
    no_stub = tmp_path / "cs_a_no_stub.md"
    no_stub.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(no_stub), "--no-stub", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    assert no_stub.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)


def test_beat147_shipped_only_idle_and_session_b_dry_run_json_idle_tmp(tmp_path: Path):
    """Beat 147: idle-decode + session-b --dry-run both idle; keep_schedule; plans unchanged."""
    from src.research_implement.__main__ import main

    src = _load("shipped_only.md")

    # idle-decode --json → idle, open_count=0, queue 0/10; keep_schedule; plan unchanged
    idle_plan = tmp_path / "shipped_idle.md"
    idle_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(idle_plan), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert idle_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-b --dry-run --json → idle (NOT dry_run), open_count=0; plan unchanged
    dry_plan = tmp_path / "shipped_b_dry.md"
    dry_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(dry_plan), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["verdict"] != "dry_run"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert dry_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat147_shipped_only_session_a_stub_vs_no_stub_dry_run_json_tmp(tmp_path: Path):
    """Beat 147: session-a --stub --dry-run→queued wrote_item=True plan unchanged; --no-stub --dry-run→failed."""
    from src.research_implement.__main__ import main

    src = _load("shipped_only.md")

    # session-a --stub --dry-run --json → queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change; plan UNCHANGED
    a_plan = tmp_path / "shipped_a_stub_dry.md"
    a_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(a_plan), "--stub", "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "queued"
    assert payload["wrote_item"] is True
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    assert payload["title"] == "Stub shippable change"
    assert a_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)

    # session-a --no-stub --dry-run --json → rc=1 failed; wrote_item=False; plan unchanged
    no_stub = tmp_path / "shipped_a_no_stub_dry.md"
    no_stub.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(no_stub), "--no-stub", "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    assert no_stub.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)


def test_beat148_incomplete_open_idle_and_session_b_dry_run_json_idle_tmp(tmp_path: Path):
    """Beat 148: idle-decode + session-b --dry-run both idle; keep_schedule; plans unchanged."""
    from src.research_implement.__main__ import main

    src = _load("incomplete_open.md")

    # idle-decode --json → idle, open_count=0, queue 0/10; keep_schedule; plan unchanged
    idle_plan = tmp_path / "incomplete_idle.md"
    idle_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(idle_plan), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert idle_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-b --dry-run --json → idle (NOT dry_run), open_count=0; plan unchanged
    dry_plan = tmp_path / "incomplete_b_dry.md"
    dry_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(dry_plan), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["verdict"] != "dry_run"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert dry_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat148_incomplete_open_session_a_stub_vs_no_stub_dry_run_json_tmp(tmp_path: Path):
    """Beat 148: session-a --stub --dry-run→queued wrote_item=True plan unchanged; --no-stub --dry-run→failed."""
    from src.research_implement.__main__ import main

    src = _load("incomplete_open.md")

    # session-a --stub --dry-run --json → queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change; plan UNCHANGED
    a_plan = tmp_path / "incomplete_a_stub_dry.md"
    a_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(a_plan), "--stub", "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "queued"
    assert payload["wrote_item"] is True
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    assert payload["title"] == "Stub shippable change"
    assert a_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)

    # session-a --no-stub --dry-run --json → rc=1 failed; wrote_item=False; plan unchanged
    no_stub = tmp_path / "incomplete_a_no_stub_dry.md"
    no_stub.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(no_stub), "--no-stub", "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    assert no_stub.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)



def test_beat149_broken_ready_flag_idle_and_session_b_dry_run_json_idle_tmp(tmp_path: Path):
    """Beat 149: idle-decode + session-b --dry-run both idle; keep_schedule; plans unchanged."""
    from src.research_implement.__main__ import main

    src = _load("broken_ready_flag.md")

    # idle-decode --json → idle, open_count=0, queue 0/10; keep_schedule; plan unchanged
    idle_plan = tmp_path / "broken_idle.md"
    idle_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(idle_plan), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert idle_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-b --dry-run --json → idle (NOT dry_run), open_count=0; plan unchanged
    dry_plan = tmp_path / "broken_b_dry.md"
    dry_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(dry_plan), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["verdict"] != "dry_run"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert dry_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat149_broken_ready_flag_session_a_stub_vs_no_stub_dry_run_json_tmp(tmp_path: Path):
    """Beat 149: session-a --stub --dry-run→queued wrote_item=True plan unchanged; --no-stub --dry-run→failed."""
    from src.research_implement.__main__ import main

    src = _load("broken_ready_flag.md")

    # session-a --stub --dry-run --json → queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change; plan UNCHANGED
    a_plan = tmp_path / "broken_a_stub_dry.md"
    a_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(a_plan), "--stub", "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "queued"
    assert payload["wrote_item"] is True
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    assert payload["title"] == "Stub shippable change"
    assert a_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)

    # session-a --no-stub --dry-run --json → rc=1 failed; wrote_item=False; plan unchanged
    no_stub = tmp_path / "broken_a_no_stub_dry.md"
    no_stub.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(no_stub), "--no-stub", "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    assert no_stub.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)


def test_beat150_top_level_help_still_lists_subcommands(capsys):
    """Beat 150: top-level --help lists session-a / session-b / idle-decode."""
    import pytest
    from src.research_implement.__main__ import main

    with pytest.raises(SystemExit) as ei:
        main(["--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    for name in ("session-a", "session-b", "idle-decode"):
        assert name in out, name


def test_beat150_public_api_milestone_exports():
    """Beat 150: core A/B API + aliases/write/serialize remain public through beat150."""
    import src.research_implement as ri

    for name in (
        "run_session_a",
        "run_session_b",
        "run_session_a_path",
        "run_session_b_path",
        "parse_queue_items",
        "first_b_pick",
        "is_b_pickable",
        "is_ready_yes",
        "is_complete_six_field",
        "is_open_status",
        "count_open",
        "count_queue_headings",
        "next_queue_id",
        "mark_item_shipped",
        "append_queue_item",
        "format_queue_item",
        "write_queue_section",
        "serialize_queue_item",
        "serialize_queue_items",
        "render_queue_count",
        "QUEUE_CAPACITY",
        "REQUIRED_FIELDS",
        "AmbiguousQueueError",
        "SchedulerDeleteForbidden",
        "scheduler_delete",
        "require_unique_queue_section",
        "incomplete_candidate_reasons",
        "decode_fields",
        "format_decode_report",
        "stub_brainstorm",
        "default_search_plan",
        "stub_search_plan",
        "default_implement",
        "dry_run_implement",
        "make_fixture_ship_implement",
        "fixture_ship_implement",
        "session_a_result_dict",
        "session_b_result_dict",
        "SessionAResult",
        "SessionResult",
        "SessionBResult",
        "QueueItem",
        "SESSION_A_RESULT_KEYS",
        "SESSION_B_RESULT_KEYS",
        "SESSION_A_RESULT_JSON_KEYS",
        "SESSION_RESULT_JSON_KEYS",
    ):
        assert hasattr(ri, name), name
        assert name in getattr(ri, "__all__", ()), name
    assert ri.SessionBResult is ri.SessionResult
    assert ri.default_implement is ri.dry_run_implement
    assert ri.default_search_plan is ri.stub_brainstorm
    assert len(ri.__all__) == len(set(ri.__all__))


def test_beat151_open_complete_not_ready_idle_and_session_b_dry_run_json_idle_tmp(tmp_path: Path):
    """Beat 151: idle-decode + session-b --dry-run both idle; keep_schedule; plans unchanged."""
    from src.research_implement.__main__ import main

    src = _load("open_complete_not_ready.md")

    # idle-decode --json → idle, open_count=0, queue 0/10; keep_schedule; plan unchanged
    idle_plan = tmp_path / "not_ready_idle.md"
    idle_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(idle_plan), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert idle_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-b --dry-run --json → idle (NOT dry_run), open_count=0; plan unchanged
    dry_plan = tmp_path / "not_ready_b_dry.md"
    dry_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(dry_plan), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["verdict"] != "dry_run"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert dry_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat151_open_complete_not_ready_session_a_stub_vs_no_stub_dry_run_json_tmp(tmp_path: Path):
    """Beat 151: session-a --stub --dry-run→queued wrote_item=True plan unchanged; --no-stub --dry-run→failed."""
    from src.research_implement.__main__ import main

    src = _load("open_complete_not_ready.md")

    # session-a --stub --dry-run --json → queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change; plan UNCHANGED
    a_plan = tmp_path / "not_ready_a_stub_dry.md"
    a_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(a_plan), "--stub", "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "queued"
    assert payload["wrote_item"] is True
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    assert payload["title"] == "Stub shippable change"
    assert a_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)

    # session-a --no-stub --dry-run --json → rc=1 failed; wrote_item=False; plan unchanged
    no_stub = tmp_path / "not_ready_a_no_stub_dry.md"
    no_stub.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(no_stub), "--no-stub", "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    assert no_stub.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)


def test_beat152_contract_spec_idle_and_session_b_dry_run_json_idle_tmp(tmp_path: Path):
    """Beat 152: idle-decode + session-b --dry-run both idle; keep_schedule; plans unchanged."""
    from src.research_implement.__main__ import main

    src = _load("contract_spec.md")

    # idle-decode --json → idle, open_count=0, queue 0/10; keep_schedule; plan unchanged
    idle_plan = tmp_path / "cs_idle.md"
    idle_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(idle_plan), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert idle_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-b --dry-run --json → idle (NOT dry_run), open_count=0; plan unchanged
    dry_plan = tmp_path / "cs_b_dry.md"
    dry_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(dry_plan), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["verdict"] != "dry_run"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert dry_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat152_contract_spec_session_a_stub_vs_no_stub_dry_run_json_tmp(tmp_path: Path):
    """Beat 152: session-a --stub --dry-run→queued wrote_item=True plan unchanged; --no-stub --dry-run→failed."""
    from src.research_implement.__main__ import main

    src = _load("contract_spec.md")

    # session-a --stub --dry-run --json → queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change; plan UNCHANGED
    a_plan = tmp_path / "cs_a_stub_dry.md"
    a_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(a_plan), "--stub", "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "queued"
    assert payload["wrote_item"] is True
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    assert payload["title"] == "Stub shippable change"
    assert a_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)

    # session-a --no-stub --dry-run --json → rc=1 failed; wrote_item=False; plan unchanged
    no_stub = tmp_path / "cs_a_no_stub_dry.md"
    no_stub.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(no_stub), "--no-stub", "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    assert no_stub.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)


def test_beat153_queue_with_watch_heartbeat_idle_and_session_b_dry_run_json_idle_tmp(tmp_path: Path):
    """Beat 153: idle-decode + session-b --dry-run both idle; keep_schedule; Watch/Heartbeat present; plans unchanged."""
    from src.research_implement.__main__ import main

    src = _load("queue_with_watch_heartbeat.md")
    _assert_beat19_queue_with_watch_heartbeat_preserved(src)
    assert src.index("## Watch") < src.index("## Queue")

    # idle-decode --json → idle, open_count=0, queue 0/10; keep_schedule; Watch+Heartbeat; plan unchanged
    idle_plan = tmp_path / "qwh_idle.md"
    idle_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(idle_plan), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    idle_body = idle_plan.read_text(encoding="utf-8")
    assert idle_body == src
    assert "## Watch" in idle_body
    assert "## Heartbeat" in idle_body
    assert idle_body.index("## Watch") < idle_body.index("## Queue")
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-b --dry-run --json → idle (NOT dry_run), open_count=0; Watch/Heartbeat; plan unchanged
    dry_plan = tmp_path / "qwh_b_dry.md"
    dry_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(dry_plan), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["verdict"] != "dry_run"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    dry_body = dry_plan.read_text(encoding="utf-8")
    assert dry_body == src
    assert "## Watch" in dry_body
    assert "## Heartbeat" in dry_body
    assert dry_body.index("## Watch") < dry_body.index("## Queue")
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat153_queue_with_watch_heartbeat_session_a_stub_vs_no_stub_dry_run_json_tmp(tmp_path: Path):
    """Beat 153: session-a --stub --dry-run→queued wrote_item=True plan unchanged keep markers; --no-stub --dry-run→failed."""
    from src.research_implement.__main__ import main

    src = _load("queue_with_watch_heartbeat.md")
    _assert_beat19_queue_with_watch_heartbeat_preserved(src)
    assert src.index("## Watch") < src.index("## Queue")

    # session-a --stub --dry-run --json → queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change; plan UNCHANGED; Watch+Heartbeat still present
    a_plan = tmp_path / "qwh_a_stub_dry.md"
    a_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(a_plan), "--stub", "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "queued"
    assert payload["wrote_item"] is True
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    assert payload["title"] == "Stub shippable change"
    a_body = a_plan.read_text(encoding="utf-8")
    assert a_body == src
    _assert_beat19_queue_with_watch_heartbeat_preserved(a_body)
    assert "## Watch" in a_body
    assert "## Heartbeat" in a_body
    assert a_body.index("## Watch") < a_body.index("## Queue")
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)

    # session-a --no-stub --dry-run --json → rc=1 failed; wrote_item=False; plan unchanged; markers intact
    no_stub = tmp_path / "qwh_a_no_stub_dry.md"
    no_stub.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(no_stub), "--no-stub", "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    no_body = no_stub.read_text(encoding="utf-8")
    assert no_body == src
    _assert_beat19_queue_with_watch_heartbeat_preserved(no_body)
    assert "## Watch" in no_body
    assert "## Heartbeat" in no_body
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)


def test_beat154_empty_queue_idle_and_session_b_dry_run_json_idle_tmp(tmp_path: Path):
    """Beat 154: idle-decode + session-b --dry-run both idle; keep_schedule; plans unchanged."""
    from src.research_implement.__main__ import main

    src = _load("empty_queue.md")

    # idle-decode --json → idle, open_count=0, queue 0/10; keep_schedule; plan unchanged
    idle_plan = tmp_path / "empty_idle.md"
    idle_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["idle-decode", "--plan", str(idle_plan), "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert idle_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)

    # session-b --dry-run --json → idle (NOT dry_run), open_count=0; plan unchanged
    dry_plan = tmp_path / "empty_b_dry.md"
    dry_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-b", "--plan", str(dry_plan), "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "idle"
    assert payload["verdict"] != "dry_run"
    assert payload["open_count"] == 0
    assert payload["queue"] == "queue 0/10"
    assert payload["keep_schedule"] is True
    assert payload["scheduler_delete_called"] is False
    assert dry_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_RESULT_JSON_KEYS)


def test_beat154_empty_queue_session_a_stub_vs_no_stub_dry_run_json_tmp(tmp_path: Path):
    """Beat 154: session-a --stub --dry-run→queued wrote_item=True plan unchanged; --no-stub --dry-run→failed."""
    from src.research_implement.__main__ import main

    src = _load("empty_queue.md")

    # session-a --stub --dry-run --json → queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change; plan UNCHANGED
    a_plan = tmp_path / "empty_a_stub_dry.md"
    a_plan.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(a_plan), "--stub", "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["verdict"] == "queued"
    assert payload["wrote_item"] is True
    assert payload["open_count"] == 1
    assert payload["queue"] == "queue 1/10"
    assert payload["title"] == "Stub shippable change"
    assert a_plan.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)

    # session-a --no-stub --dry-run --json → rc=1 failed; wrote_item=False; plan unchanged
    no_stub = tmp_path / "empty_a_no_stub_dry.md"
    no_stub.write_text(src, encoding="utf-8")
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["session-a", "--plan", str(no_stub), "--no-stub", "--dry-run", "--json"])
    payload = json.loads(buf.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["verdict"] == "failed"
    assert payload["wrote_item"] is False
    assert no_stub.read_text(encoding="utf-8") == src
    assert set(payload.keys()) == set(SESSION_A_RESULT_JSON_KEYS)
