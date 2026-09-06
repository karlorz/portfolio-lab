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
Full pipeline (Beat 11): A stub append → B dry_run (JSON) → B fixture_ship → then
A light recount (while OPEN>=1, mid-pipeline) and/or B idle — on tmp_path only;
``make research-implement-e2e-pipeline`` / pytest ``-k beat11``.

Beat 12: CLI ``--help`` smoke (session-a / session-b / idle-decode mention
JSON / dry-run / idle fire as appropriate) and OPEN>=1 brainstorm/search_plan
spy (recount-only; custom callback never called). Proof: pytest ``-k beat12``.

Beat 13: Queue markdown round-trip (parse → serialize/write → parse preserves
six fields + ready-for-implement for OPEN items) and Session A stub append id
stability (empty→Q1; after ship/clear→new id no collide; two_open first-OPEN
unchanged). Proof: pytest ``-k beat13``.

Beat 14: Session A fail-closed on incomplete brainstorm/search_plan candidate
(missing required six fields or ready-for-implement not yes) → verdict failed,
wrote_item=False, plan unchanged, no partial OPEN append. Complete stub on
empty Queue still queues. Proof: pytest ``-k beat14``.

Beat 15: CLI ``session-a --candidate-json`` loads dict/list candidate for
brainstorm when OPEN=0; incomplete fail-closes; complete queues one OPEN;
OPEN>=1 recount-only ignores candidate-json. Proof: pytest ``-k beat15``.

Beat 16: CLI ``--candidate-json`` error paths (missing / invalid JSON / wrong
type) → clear non-zero failure, no plan mutation; sequential double-OPEN ship
on tmp_path (ship → ship → idle) never ``scheduler_delete``. Proof: pytest
``-k beat16``.

Beat 17: Session B decode_only default / CLI session-b never calls implement;
``decode_only=False`` dry_run path invokes ``dry_run_implement``; fixture ship
only when opted in (``decode_only=False`` + ``fixture_ship_implement``).
Proof: pytest ``-k beat17``.

Beat 18: Public API export smoke (``import src.research_implement`` + key
``__all__`` / getattr names: SessionAResult, SessionResult, queue helpers,
dry_run_implement, fixture_ship, incomplete_candidate_reasons, …) and Makefile
help/echo listing ``test-research-implement``, ``e2e-dry-run``, ``e2e-pipeline``.
Proof: pytest ``-k beat18``.

Beat 19: When rewriting ``## Queue`` (append / ship / serialize via
``write_queue_section``), preserve other markdown sections (Watch / Project
Work / Heartbeat). Fixture ``watch_queue_heartbeat*.md`` + pytest ``-k beat19``:
A append or B ship updates Queue while non-Queue markers stay present.

Beat 20: If markdown has no ``## Queue`` section, Session A append /
``write_queue_section`` creates one without destroying Watch / Heartbeat /
YAML front matter. Fixture ``watch_heartbeat_no_queue.md`` + empty file:
A stub append → ``## Queue`` with one OPEN; prior beat19 stays green.
Proof: pytest ``-k beat20``.

Beat 21: More than one ``## Queue`` heading is fail-closed
(``AmbiguousQueueError``): parse / write / append raise; Session A/B return
failed results without mutating the plan (no silent merge). Fixture
``two_queue_sections.md`` + pytest ``-k beat21``. Prior beat20 stays green.

JSON: ``SessionResult.to_dict()`` (aliases ``to_json_dict`` / ``session_b_result_dict``)
is the shared Session B ``--json`` / test contract for idle / decode_only / dry_run /
shipped. Session A: ``SessionAResult.to_dict()`` (aliases ``to_json_dict`` /
``session_a_result_dict``) covers append/queued vs recount-only/light for
``session-a --json``.


Beat 22: see ``docs/research-implement-ab-side-dev.md`` for side-dev CLI/make notes.

Beat 23: CLI ``session-a`` / ``session-b`` / ``idle-decode`` with missing
``--plan`` / ``--log`` path → clear non-zero ``SystemExit`` (``file not found``);
does not create the plan file. Existing plan path still works. Proof: pytest
``-k beat23``. Prior beat22 stays green.

Beat 40: CLI help contracts — top-level lists session-a/session-b/idle-decode; idle-decode has no --dry-run; session-b does. Proof: pytest ``-k beat40``.

Beat 41–42: side-dev help wording / no CLI ``--implement``; Makefile lists ``test-research-implement`` + e2e targets; B/idle help never ``scheduler_delete``. Proof: pytest ``-k beat41`` / ``beat42``.

Beat 43: side-dev guide pins make/ship contracts; package docs mention Beat 40 help. Proof: pytest ``-k beat43``.

Beat 44: Session B dry-run message includes decode-pick six-field report; ``session-b --json`` help matches ``SessionResult.to_dict``. Proof: pytest ``-k beat44``.

Beat 50: broken ready-for-implement flags idle via CLI; public API exports remain stable through A/B side-dev. Proof: pytest ``-k beat50``.

Beat 51: ready-for-implement aliases (YES/Y/TRUE/1) remain B-pickable via CLI; non-aliases still idle. Proof: pytest ``-k beat51``.

Beat 52: ambiguous dual-Queue plans fail closed via CLI with plan unchanged. Proof: pytest ``-k beat52``.

Beat 53: empty Queue + Watch/Project/Heartbeat CLI idle/append preserves non-Queue markers. Proof: pytest ``-k beat53``.
"""

from __future__ import annotations

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
    next_queue_id,
    parse_queue_items,
    render_queue_count,
    require_unique_queue_section,
    serialize_queue_item,
    serialize_queue_items,
    write_queue_section,
)
from src.research_implement.session_a import (
    SESSION_A_RESULT_JSON_KEYS,
    SESSION_A_RESULT_KEYS,
    SessionAResult,
    default_search_plan,
    incomplete_candidate_reasons,
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
    "AmbiguousQueueError",
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
    "count_queue_headings",
    "first_b_pick",
    "format_queue_item",
    "next_queue_id",
    "serialize_queue_item",
    "serialize_queue_items",
    "write_queue_section",
    "is_b_pickable",
    "is_complete_six_field",
    "is_open_status",
    "parse_queue_items",
    "require_unique_queue_section",
    "render_queue_count",
    "default_search_plan",
    "run_session_a",
    "run_session_b",
    "incomplete_candidate_reasons",
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
