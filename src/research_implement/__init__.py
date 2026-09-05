"""Session A/B research-implement loop (Queue producer/consumer).

Distinct from ``src.research.agent`` / ``make research`` (regime crystallizer).
Session A accepts a pluggable search/plan callback (default: ``stub_brainstorm``).
Session B accepts a pluggable implement callback (default: ``dry_run_implement``).
Empty Queue is an idle fire (``queue 0/10``); never ``scheduler_delete``.

Dry-run contract: ``dry_run_implement`` is non-mutating — it never marks
SHIPPED, never deletes Queue rows, and never calls ``scheduler_delete``.
A second Session B fire therefore picks the same OPEN again (not idle).

CLI: ``python -m src.research_implement {session-a|session-b|idle-decode}``.
E2E dry-run (tmp plan): see module ``__main__`` examples / ``make research-implement-e2e-dry-run``.
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
    SessionAResult,
    default_search_plan,
    run_session_a,
    stub_brainstorm,
    stub_search_plan,
)
from src.research_implement.session_b import (
    SessionBResult,
    decode_fields,
    default_implement,
    dry_run_implement,
    format_decode_report,
    run_session_b,
    scheduler_delete,
)

__all__ = [
    "QUEUE_CAPACITY",
    "REQUIRED_FIELDS",
    "QueueItem",
    "SessionAResult",
    "SessionBResult",
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
]
