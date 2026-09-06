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

Beat 54: OPEN Watch/Heartbeat plan CLI pick/dry-run preserves non-Queue markers. Proof: pytest ``-k beat54``.

Beat 55: no-Queue Watch/Heartbeat plans: B idle CLI; A stub CLI creates Queue without dropping markers. Proof: pytest ``-k beat55``.

Beat 56: light/dry_run/failed CLI JSON key contracts stay locked; Watch-before-Queue A→B CLI preserve. Proof: pytest ``-k beat56``.

Beat 57: Session A failed JSON key contract + AmbiguousQueueError export; dry-run A no write. Proof: pytest ``-k beat57``.

Beat 58: empty-queue ``session-b --dry-run`` stays idle; queue helper exports remain public. Proof: pytest ``-k beat58``.

Beat 59: two-OPEN dry-run picks first only; plan stays dual-OPEN. Proof: pytest ``-k beat59``.

Beat 60: Watch lookalike dry-run/pick stays Queue-only. Proof: pytest ``-k beat60``.

Beat 61: non-pickable plans stay idle under ``session-b --dry-run``. Proof: pytest ``-k beat61``.

Beat 62: not-ready / broken-ready dry-run idle; next_queue_id export. Proof: pytest ``-k beat62``.

Beat 63: mixed_priority dry-run/pick first ready; count_queue_headings export. Proof: pytest ``-k beat63``.

Beat 64: one_open dry-run/pick; format_queue_item + require_unique_queue exports. Proof: pytest ``-k beat64``.

Beat 65: dual-Queue dry-run still fail-closed; write_queue_section export. Proof: pytest ``-k beat65``.

Beat 66: --no-stub light/failed CLI paths; implement helper exports. Proof: pytest ``-k beat66``.

Beat 67: --no-stub --dry-run light/failed; decode helper exports. Proof: pytest ``-k beat67``.

Beat 68: first_b_pick / incomplete_candidate_reasons contracts; idle-decode decode-pick message. Proof: pytest ``-k beat68``.

Beat 69: session_*_result_dict helpers align with CLI --json. Proof: pytest ``-k beat69``.

Beat 70: path runners exported; light/failed Session A JSON keys locked. Proof: pytest ``-k beat70``.

Beat 71: Session B dry_run/idle/failed JSON keys; to_dict aliases. Proof: pytest ``-k beat71``.

Beat 72: Session A to_dict aliases align across queued/light/failed. Proof: pytest ``-k beat72``.

Beat 73: result/queue dataclasses remain public API. Proof: pytest ``-k beat73``.

Beat 74: QueueItem pickability + format/parse round-trip. Proof: pytest ``-k beat74``.

Beat 75: incomplete/shipped fixtures never B-pickable; count_open=0. Proof: pytest ``-k beat75``.

Beat 76: not-ready / broken-ready fixtures never pickable. Proof: pytest ``-k beat76``.

Beat 77: mixed_priority first_b_pick skips incomplete. Proof: pytest ``-k beat77``.

Beat 78: two_open_ready first_b_pick is first of two. Proof: pytest ``-k beat78``.

Beat 79: one_open vs empty_queue pick/count contract. Proof: pytest ``-k beat79``.

Beat 80: milestone — CLI help + public API export smoke through beat80. Proof: pytest ``-k beat80``.
Beat 81: watch_lookalike / watch_only first_b_pick + mark_item_shipped export. Proof: pytest ``-k beat81``.
Beat 82: subcommand help plan/json + render_queue_count/QUEUE_CAPACITY export. Proof: pytest ``-k beat82``.
Beat 83: non-pickable fixtures first_b_pick=None; serialize_queue_item(s) export. Proof: pytest ``-k beat83``.
Beat 84: pickable fixtures first_b_pick ids; is_complete_six_field/is_open_status export. Proof: pytest ``-k beat84``.
Beat 85: producer/decode help flags + REQUIRED_FIELDS export. Proof: pytest ``-k beat85``.
Beat 86: idle ``queue 0/10`` + ``scheduler_delete`` guard; ``SchedulerDeleteForbidden`` export;
decode report starts with ``decode pick``. Proof: pytest ``-k beat86``.
Beat 87: two_queue AmbiguousQueueError + heading count; decode helpers export. Proof: pytest ``-k beat87``.
Beat 88: Watch/Heartbeat fixture idle/pick; SESSION_A/B_RESULT_KEYS export. Proof: pytest ``-k beat88``.
Beat 89: incomplete_candidate_reasons contract; SESSION_*_JSON_KEYS export. Proof: pytest ``-k beat89``.
Beat 90: milestone — CLI help + public API export smoke through beat90. Proof: pytest ``-k beat90``.
Beat 91: empty append next Q1 pickable; append/format/write exports. Proof: pytest ``-k beat91``.
Beat 92: mark_item_shipped clears pick; stub_brainstorm + implement helpers export. Proof: pytest ``-k beat92``.
Beat 93: is_ready_yes aliases; search_plan aliases; session_*_result_dict export. Proof: pytest ``-k beat93``.
Beat 94: result_dict/to_dict JSON key align; path runners export. Proof: pytest ``-k beat94``.
Beat 95: serialize round-trip pickable; count_queue_headings fixtures. Proof: pytest ``-k beat95``.
Beat 96: decode_fields + format_decode_report line contract. Proof: pytest ``-k beat96``.
Beat 97: next_queue_id progression; require_unique fail-closed. Proof: pytest ``-k beat97``.
Beat 98: is_b_pickable fixture matrix; complete/open/ready exports. Proof: pytest ``-k beat98``.
Beat 99: count_open fixture matrix + render_queue_count align. Proof: pytest ``-k beat99``.
Beat 100: milestone — CLI help + public API export smoke through beat100. Proof: pytest ``-k beat100``.
Beat 101: format_queue_item round-trip; dry_run/fixture-ship exports. Proof: pytest ``-k beat101``.
Beat 102: write_queue_section transplant; default_implement alias dry-run. Proof: pytest ``-k beat102``.
Beat 103: serialize_queue_items two_open round-trip. Proof: pytest ``-k beat103``.
Beat 104: QueueItem.field_map covers REQUIRED_FIELDS. Proof: pytest ``-k beat104``.
Beat 105: to_json_dict aliases to_dict; SessionBResult is SessionResult. Proof: pytest ``-k beat105``.
Beat 106: keep_schedule True; never scheduler_delete_called. Proof: pytest ``-k beat106``.
Beat 107: Session A queued/light/failed to_dict matrix. Proof: pytest ``-k beat107``.
Beat 108: Session B idle/picked/failed to_dict matrix. Proof: pytest ``-k beat108``.
Beat 109: first_b_pick/count_open fixture consistency; __all__ unique. Proof: pytest ``-k beat109``.
Beat 110: milestone — CLI help + public API export smoke through beat110. Proof: pytest ``-k beat110``.
Beat 111: idle fixtures queue 0/10 + decode_report None; picked decode_report == format_decode_report. Proof: pytest ``-k beat111``.
Beat 112: path runners A write / B idle no-touch. Proof: pytest ``-k beat112``.
Beat 113: idle-decode CLI --json idle/picked/failed matrix. Proof: pytest ``-k beat113``.
Beat 114: session-b --decode-only --json idle/picked/failed matrix. Proof: pytest ``-k beat114``.
Beat 115: session-a --stub|--no-stub --json CLI matrix (queued/light/failed) via tmp_path. Proof: pytest ``-k beat115``.
Beat 116: session-b --dry-run --json CLI matrix (idle/dry_run/failed) via tmp_path; plan untouched. Proof: pytest ``-k beat116``.
Beat 117: session-a --stub|--no-stub --dry-run --json CLI matrix via tmp_path (queued wrote_item=True but plan unchanged / light / failed). Contrast Beat 115. Proof: pytest ``-k beat117``.
Beat 118: idle-decode --json Watch-fixture CLI matrix via tmp_path (watch_only→idle; watch_lookalike/heartbeat→picked; two_queue→failed). Contrast Beat 113. Proof: pytest ``-k beat118``.
Beat 119: session-b --dry-run --json Watch-fixture CLI matrix via tmp_path (watch_only→idle; lookalike/heartbeat→dry_run; two_queue→failed). Contrast Beat 116 + Beat 118. Proof: pytest ``-k beat119``.
Beat 120: milestone — CLI help + public API export smoke through beat120. Proof: pytest ``-k beat120``.
Beat 121: session-a --stub|--no-stub --json Watch-fixture CLI matrix via tmp_path (queued keeps markers / light / failed). Contrast Beats 115/118/119. Proof: pytest ``-k beat121``.
Beat 122: session-a --stub|--no-stub --dry-run --json Watch-fixture CLI matrix via tmp_path (queued wrote_item=True but plan unchanged / light / failed). Contrast Beat 117 + Beat 121. Proof: pytest ``-k beat122``.
Beat 123: session-b --decode-only --json Watch-fixture CLI matrix via tmp_path (watch_only→idle; lookalike/heartbeat→picked; two_queue→failed). Completes Watch trilogy with Beat 118 + Beat 119. Proof: pytest ``-k beat123``.
Beat 124: idle-decode --json non-pickable OPEN CLI matrix via tmp_path (shipped_only/incomplete_open/broken_ready_flag/open_complete_not_ready→idle); plan untouched; keep_schedule / no scheduler_delete. Distinct from Beat 113 + Beat 118. Proof: pytest ``-k beat124``.
Beat 125: session-b --dry-run --json non-pickable OPEN CLI matrix via tmp_path (shipped_only/incomplete_open/broken_ready_flag/open_complete_not_ready→idle, never dry_run); plan untouched; keep_schedule / no scheduler_delete. Pairs with Beat 124. Proof: pytest ``-k beat125``.
Beat 126: session-b --json non-pickable OPEN CLI matrix via tmp_path (shipped/incomplete/broken/not_ready→idle, never dry_run). Completes non-pickable trilogy with Beat 124 + Beat 125. Proof: pytest ``-k beat126``.
Beat 127: session-a --no-stub|--stub --json non-pickable OPEN CLI matrix via tmp_path (non-pickable=OPEN=0: --no-stub→failed; --stub→queued under tmp). Contrast Beats 115 + 124–126. Proof: pytest ``-k beat127``.
Beat 128: session-a --no-stub|--stub --dry-run --json non-pickable OPEN CLI matrix via tmp_path (--no-stub dry-run→failed; --stub dry-run→queued wrote_item=True but plan unchanged). Contrast Beat 127. Proof: pytest ``-k beat128``.
Beat 129: mixed_priority cross-CLI --json matrix via tmp_path (first ready OPEN=Q2): idle-decode + session-b --decode-only pick Q2; session-b --dry-run→dry_run Q2; session-a --stub→light b_pick_title Second ready complete item; plans unchanged. Proof: pytest ``-k beat129``.
Beat 130: milestone — CLI help + public API export smoke through beat130. Proof: pytest ``-k beat130``.
Beat 131: two_open_ready cross-CLI --json matrix via tmp_path (first ready OPEN=Q1; open_count=2): idle-decode + session-b --decode-only pick Q1; session-b --dry-run→dry_run Q1; session-a --stub→light b_pick_title First ready complete item; plans unchanged. Parallel to Beat 129. Proof: pytest ``-k beat131``.
Beat 132: one_open_ready cross-CLI --json matrix via tmp_path (single ready OPEN=Q1; open_count=1): idle-decode + session-b --decode-only pick Q1; session-b --dry-run→dry_run Q1; session-a --stub→light b_pick_title Add fixture unit test for queue parser; plans unchanged. Completes cross-CLI pick trilogy with Beat 129 + Beat 131. Proof: pytest ``-k beat132``.
Beat 133: watch_lookalike cross-CLI --json matrix via tmp_path (real Queue OPEN=Q3; never Watch lookalike; open_count=1): idle-decode + session-b --decode-only pick Q3; session-b --dry-run→dry_run Q3; session-a --stub→light b_pick_title Real ready Queue item; plans unchanged. Distinct from Beat 118 + Beats 129/131/132. Proof: pytest ``-k beat133``.
Beat 134: watch_queue_heartbeat cross-CLI --json matrix via tmp_path (Queue OPEN=Q1; preserve Watch/Heartbeat markers; open_count=1): idle-decode + session-b --decode-only pick Q1; session-b --dry-run→dry_run Q1; session-a --stub→light b_pick_title Beat19 shippable preserve item; plans unchanged. Distinct from Beat 133 + Beats 118/119/123. Proof: pytest ``-k beat134``.
Beat 135: watch_only_lookalike cross-CLI --json matrix via tmp_path (empty Queue; Watch lookalike never B-picked): B paths idle; session-a --stub queues while keeping Watch/Heartbeat. Idle counterpart to Beat 133 + Beat 134. Proof: pytest ``-k beat135``.
Beat 136: watch_heartbeat_no_queue cross-CLI --json matrix via tmp_path (no Queue initially; B idle; session-a --stub creates Queue keeping Watch/Heartbeat). Completes Watch idle trilogy with Beat 135. Proof: pytest ``-k beat136``.
Beat 137: empty_queue cross-CLI --json matrix via tmp_path (foundational baseline after Watch idle trilogy 135/136): B/idle paths idle keep_schedule; session-a --stub→queued write; --no-stub→failed plan unchanged. Proof: pytest ``-k beat137``.
Beat 138: watch_queue_heartbeat_empty cross-CLI --json matrix via tmp_path (empty Queue WITH Watch/Heartbeat markers; after plain empty_queue Beat 137): B/idle paths idle keep_schedule Watch/Heartbeat kept; session-a --stub→queued write keep markers; --no-stub→failed plan unchanged markers intact. Proof: pytest ``-k beat138``.
Beat 139: two_queue_sections fail-closed cross-CLI --json matrix via tmp_path (ambiguous dual Queue sections; every path fails; plan unchanged): idle-decode + session-b --decode-only + session-b --dry-run all failed keep_schedule; session-a --stub/--no-stub/--stub --dry-run all failed wrote_item=False; plans unchanged. Proof: pytest ``-k beat139``.
Beat 140: milestone — CLI help + public API export smoke through beat140. Proof: pytest ``-k beat140``.
Beat 141: shipped_only cross-CLI --json matrix via tmp_path (foundational post-milestone baseline; only SHIPPED; open_count=0): B/idle paths idle keep_schedule; session-a --stub→queued write title Stub shippable change; --no-stub→failed plan unchanged. Proof: pytest ``-k beat141``.
Beat 142: incomplete_open cross-CLI --json matrix via tmp_path (foundational non-pickable OPEN after shipped_only Beat 141; incomplete OPEN missing fields; open_count=0): B/idle paths idle keep_schedule; session-a --stub→queued write title Stub shippable change Heartbeat kept; --no-stub→failed plan unchanged. Proof: pytest ``-k beat142``.
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
    is_ready_yes,
    is_complete_six_field,
    is_open_status,
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
    stub_search_plan,
)
from src.research_implement.session_b import (
    SESSION_B_RESULT_KEYS,
    SESSION_RESULT_JSON_KEYS,
    SchedulerDeleteForbidden,
    SessionBResult,
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

__all__ = [
    "QUEUE_CAPACITY",
    "REQUIRED_FIELDS",
    "AmbiguousQueueError",
    "SchedulerDeleteForbidden",
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
    "mark_item_shipped",
    "is_b_pickable",
    "is_ready_yes",
    "is_complete_six_field",
    "is_open_status",
    "parse_queue_items",
    "require_unique_queue_section",
    "render_queue_count",
    "default_search_plan",
    "run_session_a",
    "run_session_a_path",
    "run_session_b",
    "run_session_b_path",
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
