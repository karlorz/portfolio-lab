"""Session A/B research-implement loop (Queue producer/consumer).

Distinct from ``src.research.agent`` / ``make research`` (regime crystallizer).
Empty Queue is an idle fire (``queue 0/10``); never ``scheduler_delete``.
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
from src.research_implement.session_a import SessionAResult, run_session_a
from src.research_implement.session_b import (
    SessionBResult,
    decode_fields,
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
    "run_session_a",
    "run_session_b",
    "decode_fields",
    "format_decode_report",
    "scheduler_delete",
]
