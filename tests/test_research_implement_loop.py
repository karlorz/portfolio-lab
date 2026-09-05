"""Fixture-based unit tests for Session A/B research-implement loop.

These pass WITHOUT a host ``logs/research-implement.md``. The skippable
contract tests in ``test_research_implement_loop_contract.py`` remain unchanged.
"""

from __future__ import annotations

import json
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
    is_open_status,
    parse_queue_items,
    render_queue_count,
)
from src.research_implement.session_a import (
    SESSION_A_RESULT_JSON_KEYS,
    SESSION_A_RESULT_KEYS,
    SessionAResult,
    default_search_plan,
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
