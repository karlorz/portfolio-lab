"""Session A/B research-implement loop (Queue producer/consumer).

Distinct from ``src.research.agent`` / ``make research`` (regime crystallizer).
Session A accepts a pluggable search/plan callback (default: ``stub_brainstorm``).
Session B accepts a pluggable implement callback (default: ``dry_run_implement``).
Empty Queue is an idle fire (``queue 0/10``); never ``scheduler_delete``.

Dry-run contract: ``dry_run_implement`` is non-mutating — it never marks
SHIPPED, never deletes Queue rows, and never calls ``scheduler_delete``.
A second Session B fire therefore picks the same OPEN again (not idle).

Optional ship path (test double only): pass ``fixture_ship_implement`` /
``make_fixture_ship_implement`` explicitly — never the default. Live prod
implement stays unwired.

CLI: ``python -m src.research_implement {session-a|session-b|idle-decode}``.
E2E dry-run (tmp plan): see module ``__main__`` examples / ``make research-implement-e2e-dry-run``.

JSON: ``SessionResult.to_dict()`` (aliases ``to_json_dict`` / ``session_b_result_dict``)
is the shared Session B ``--json`` / test contract for idle / decode_only / dry_run /
shipped. Session A: ``SessionAResult.to_dict()`` (aliases ``to_json_dict`` /
``session_a_result_dict``) covers append/queued vs recount-only/light for
``session-a --json``.
"""

from __future__ import annotations

from src.research_implement.queue import (
    QUEUE_CAPACITY,
    REQUIRED_FIELDS,
    QueueItem,
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
    session_a_result_dict,
    stub_brainstorm,
    stub_search_plan,
)
from src.research_implement.session_b import (
    SESSION_B_RESULT_KEYS,
    SESSION_RESULT_JSON_KEYS,
    SessionBResult,
    SessionResult,
    decode_fields,
    default_implement,
    dry_run_implement,
    fixture_ship_implement,
    format_decode_report,
    make_fixture_ship_implement,
    run_session_b,
    scheduler_delete,
    session_b_result_dict,
)

__all__ = [
    "QUEUE_CAPACITY",
    "REQUIRED_FIELDS",
    "QueueItem",
    "SessionAResult",
    "SessionBResult",
    "SessionResult",
    "SESSION_A_RESULT_JSON_KEYS",
    "SESSION_A_RESULT_KEYS",
    "SESSION_RESULT_JSON_KEYS",
    "SESSION_B_RESULT_KEYS",
    "session_a_result_dict",
    "session_b_result_dict",
    "append_queue_item",
    "count_open",
    "first_b_pick",
    "format_queue_item",
    "is_b_pickable",
    "is_complete_six_field",
    "is_open_status",
    "parse_queue_items",
    "render_queue_count",
    "default_search_plan",
    "run_session_a",
    "run_session_b",
    "stub_brainstorm",
    "stub_search_plan",
    "decode_fields",
    "format_decode_report",
    "scheduler_delete",
    "dry_run_implement",
    "default_implement",
    "fixture_ship_implement",
    "make_fixture_ship_implement",
]
