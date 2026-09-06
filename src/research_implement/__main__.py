"""CLI: python -m src.research_implement {session-a|session-b|idle-decode}

Side-dev entrypoints only. Does not start Tasker, does not load Tasker yaml on
prod ports (8000/8001/18000), and does not call external LLMs / Hermes /
``src.research.agent``.

Examples (fixture or --log path)::

    python -m src.research_implement session-a --plan tests/fixtures/research_implement/empty_queue.md --stub --dry-run --json
    python -m src.research_implement session-b --log logs/research-implement.md --json
    python -m src.research_implement session-b --plan tests/fixtures/research_implement/one_open_ready.md --dry-run --json
    python -m src.research_implement idle-decode --plan tests/fixtures/research_implement/empty_queue.md

E2E dry-run on a temp plan (A stub → B dry-run → B again; OPEN stays OPEN)::

    cp tests/fixtures/research_implement/empty_queue.md /tmp/ri-plan.md
    python -m src.research_implement session-a --plan /tmp/ri-plan.md --stub
    python -m src.research_implement session-b --plan /tmp/ri-plan.md --dry-run --json
    python -m src.research_implement session-b --plan /tmp/ri-plan.md --dry-run --json

Dry-run contract: non-mutating — never marks SHIPPED, never deletes Queue
rows, never ``scheduler_delete``. Second B therefore picks the same OPEN
again (not idle ``queue 0/10``). Or: ``make research-implement-e2e-dry-run``.

Full pipeline (Beat 11, tmp_path / test double only)::

    A stub append → B dry_run --json → B fixture_ship → A light (OPEN>=1) and/or B idle

Light recount runs mid-pipeline while OPEN>=1 (after dry_run, before ship). Dry-run
never ships; ``fixture_ship_implement`` ships on tmp_path only (never CLI default).
Proof: ``make research-implement-e2e-pipeline`` / pytest ``-k beat11``.

Beat 12: CLI ``--help`` smoke (session-a / session-b / idle-decode) + OPEN>=1
brainstorm/search_plan spy (recount-only; callback never called). Proof:
pytest ``-k beat12``.

Beat 13: Queue markdown round-trip (parse → serialize/write → parse) preserves
six fields + ready-for-implement for OPEN; Session A stub append id stability
(empty→Q1; after ship/clear→new id no collide; two_open first-OPEN unchanged).
Proof: pytest ``-k beat13``.

Beat 14: Session A fail-closed on incomplete brainstorm/search_plan candidate
(missing six fields or ready flag) → failed, wrote_item=False, plan unchanged,
no partial OPEN append; empty + complete stub still queues. Proof: pytest
``-k beat14``.

Beat 15: CLI ``session-a --candidate-json <path>`` loads a candidate dict (or
first dict in a JSON list) as the brainstorm callback when OPEN=0 (instead of
stub). Incomplete JSON still fail-closes (Beat 14); complete candidate queues
one OPEN; OPEN>=1 remains recount-only and ignores candidate-json. Proof:
pytest ``-k beat15``.

Beat 16: CLI ``--candidate-json`` error paths (missing file, invalid JSON, wrong
type not object/list) → non-zero exit / clear failure; plan unchanged. Sequential
double-OPEN ship on tmp_path: two_open_ready → ship first → ship second → idle;
never ``scheduler_delete``; JSON shapes ok. Proof: pytest ``-k beat16``.

Beat 17: Session B default/CLI decode-only never invokes implement callback
(spy); ``--dry-run`` may call ``dry_run_implement``; ``fixture_ship_implement``
only when ``decode_only=False`` + ``implement=`` passed (never CLI default).
Proof: pytest ``-k beat17``.

Beat 19: Queue rewrite (append/ship/serialize) preserves Watch / Project
Work / Heartbeat. Proof: pytest ``-k beat19``.

Beat 20: Missing ``## Queue`` → Session A append / ``write_queue_section``
creates the section without destroying Watch / Heartbeat / front matter;
empty file creates Queue. Proof: pytest ``-k beat20``.

Beat 21: More than one ``## Queue`` → fail-closed (``AmbiguousQueueError`` /
Session A/B failed); plan unchanged. Proof: pytest ``-k beat21``.

Beat 23: Missing ``--plan`` / ``--log`` path (or omitted flag) → clear
non-zero failure (``SystemExit``); CLI does not create random plan files.
Existing plan still works. Proof: pytest ``-k beat23``.

Beat 24: Passing both ``--plan`` and ``--log`` → argparse mutually exclusive exit 2; ``--log`` alone aliases ``--plan``. Proof: pytest ``-k beat24``.

Beat 25: Passing both ``--stub`` and ``--no-stub`` → clear ``SystemExit``; plan unchanged. ``--candidate-json`` still overrides stub when OPEN=0. Proof: pytest ``-k beat25``.

Beat 26: ``idle-decode`` rejects ``--dry-run`` (decode-only alias; use ``session-b --dry-run``). Session A ``--dry-run`` leaves the plan file unchanged. Proof: pytest ``-k beat26``.

Beat 27: Producer-only flags (``--stub`` / ``--no-stub`` / ``--candidate-json``) are rejected on ``session-b`` / ``idle-decode``; plan unchanged. Proof: pytest ``-k beat27``.

Beat 28: ``session-a --candidate-json`` with an empty JSON list ``[]`` → failed fire (no stub fallback), plan unchanged; unknown CLI subcommand → non-zero exit. Proof: pytest ``-k beat28``.

Beat 29: ``--candidate-json`` list with no dict elements → failed fire (no stub), plan unchanged; CLI ``--json`` emits exactly ``SESSION_A_RESULT_JSON_KEYS`` / ``SESSION_RESULT_JSON_KEYS``. Proof: pytest ``-k beat29``.

Beat 30: ``--candidate-json`` list skips leading non-dicts and uses the first dict; relative ``--plan`` path resolves when the file exists. Proof: pytest ``-k beat30``.

Beat 31: ``--plan`` / ``--log`` path that exists but is not a file (e.g. directory) → clear non-zero exit; relative ``--log`` alias works like relative ``--plan``. Proof: pytest ``-k beat31``.

Beat 32: ``--candidate-json`` path that exists but is not a file (e.g. directory) → clear non-zero exit; top-level JSON ``null`` also fails closed. Proof: pytest ``-k beat32``.

Beat 33: empty / whitespace-only ``--plan`` / ``--log`` → clear non-zero exit; idle-decode ``--json`` keeps ``keep_schedule=true`` and ``scheduler_delete_called=false``. Proof: pytest ``-k beat33``.

Beat 34: empty / whitespace ``--log`` alias fails like ``--plan``; ``session-b --dry-run --json`` keeps ``keep_schedule=true`` and ``scheduler_delete_called=false``. Proof: pytest ``-k beat34``.

Beat 35: ``session-b`` decode-only ``--json`` (default) keeps ``keep_schedule=true`` / ``scheduler_delete_called=false`` and leaves the plan unchanged; Session A recount-only (OPEN>=1) ``--json`` has ``wrote_item=false``. Proof: pytest ``-k beat35``.

Beat 36: ``idle-decode --json`` on a ready OPEN matches session-b decode-only (picked + keep_schedule); Session A stub on empty Queue ``--json`` has ``wrote_item=true`` / ``verdict=queued``. Proof: pytest ``-k beat36``.

Beat 37: empty / whitespace ``--candidate-json`` → clear non-zero exit (no Path(".") coerce); Session A ``--stub --dry-run --json`` reports ``wrote_item=true`` / ``verdict=queued`` but leaves the plan unchanged. Proof: pytest ``-k beat37``.

Beat 38: relative ``--candidate-json`` resolves when the file exists in cwd; top-level JSON boolean ``true``/``false`` fails closed (wrong type). Proof: pytest ``-k beat38``.

Beat 39: ``--candidate-json`` empty object ``{}`` fail-closes via Session A incomplete candidate (rc=1, plan unchanged); ``session-a --help`` mentions ``--stub`` / ``--no-stub`` / ``--candidate-json``. Proof: pytest ``-k beat39``.

Beat 40: top-level ``--help`` lists ``session-a`` / ``session-b`` / ``idle-decode``; ``idle-decode --help`` usage has no ``--dry-run``; ``session-b --help`` usage includes ``--dry-run``. Proof: pytest ``-k beat40``.

Beat 41: top-level ``--help`` mentions side-dev / no Tasker / no live LLM; CLI rejects ``--implement`` / stub-ship style flags (SHIPPED remains callback-only). Proof: pytest ``-k beat41``.

Beat 42: Makefile lists ``test-research-implement`` / ``research-implement-e2e-dry-run`` / ``research-implement-e2e-pipeline``; ``session-b`` / ``idle-decode --help`` mention never ``scheduler_delete``. Proof: pytest ``-k beat42``.

Beat 43: side-dev guide documents make targets + no CLI ship flag + never ``scheduler_delete``; ``__init__`` mentions Beats 40–42. Proof: pytest ``-k beat43``.

Beat 44: ``session-b --json`` help matches shared ``SessionResult.to_dict`` (idle | picked | dry_run); dry-run message includes ``decode pick`` six-field report like decode-only. Proof: pytest ``-k beat44``.

Beat 45: ``idle-decode --json`` / ``session-a --json`` help mention shared ``SessionResult.to_dict`` / ``SessionAResult.to_dict``; decode-only ``session-b`` message includes ``decode pick``. Proof: pytest ``-k beat45``.

Beat 46: OPEN complete but not ready-for-implement → idle fire (``queue 0/10``, keep_schedule) via ``idle-decode`` / ``session-b`` CLI; plan unchanged. Proof: pytest ``-k beat46``.

Beat 47: incomplete OPEN / SHIPPED-only → idle CLI JSON; ``mixed_priority`` picks first ready OPEN via ``session-b --json`` (plan unchanged on decode-only). Proof: pytest ``-k beat47``.

Beat 48: Watch lookalike rows are never B-picked; ``watch_lookalike`` CLI picks real Queue OPEN; ``watch_only_lookalike`` / no-Queue watch plans idle. Proof: pytest ``-k beat48``.

Beat 49: ``two_open_ready`` CLI picks first ready OPEN (second stays); Session A on OPEN>=1 is recount-only; Queue+Watch+Heartbeat plans keep non-Queue sections on stub append. Proof: pytest ``-k beat49``.

Beat 50: ``broken_ready_flag`` → idle CLI JSON (never pick garbage ready); public ``research_implement`` exports still include Session A/B result helpers; Makefile echo mentions beat50. Proof: pytest ``-k beat50``.

Beat 51: ``ready-for-implement`` accepts case-insensitive aliases (``yes``/``y``/``true``/``1``) for B-pick CLI; rejects non-alias garbage. Proof: pytest ``-k beat51``.

Beat 52: ``two_queue_sections`` CLI (session-a / session-b / idle-decode) → non-zero exit, ``verdict=failed``, plan untouched (ambiguous Queue fail-closed). Proof: pytest ``-k beat52``.

Beat 53: ``watch_queue_heartbeat_empty`` idle-decode/session-b → idle with Watch/Project/Heartbeat markers intact; session-a stub append keeps those sections. Proof: pytest ``-k beat53``.

Beat 54: ``watch_queue_heartbeat`` (OPEN ready) idle-decode pick + ``session-b --dry-run`` leave Watch/Project/Heartbeat markers intact. Proof: pytest ``-k beat54``.

Beat 55: ``watch_heartbeat_no_queue`` idle-decode stays idle with markers intact; ``session-a --stub`` creates ``## Queue`` while keeping Watch/Heartbeat markers. Proof: pytest ``-k beat55``.

Beat 56: CLI ``--json`` for light / dry_run / failed (dual-Queue) still matches ``SESSION_A_RESULT_JSON_KEYS`` / ``SESSION_RESULT_JSON_KEYS``; ``queue_with_watch_heartbeat`` A-stub then idle-decode pick keeps Watch markers. Proof: pytest ``-k beat56``.

Beat 57: ``session-a`` failed (dual-Queue) ``--json`` keys match ``SESSION_A_RESULT_JSON_KEYS``; ``AmbiguousQueueError`` stays public; ``session-a --dry-run`` on empty Queue leaves plan unchanged. Proof: pytest ``-k beat57``.

Beat 58: ``session-b --dry-run`` on empty Queue is idle (not dry_run); public exports still include ``is_b_pickable`` / ``is_ready_yes`` / ``count_open``. Proof: pytest ``-k beat58``.

Beat 59: ``two_open_ready`` ``session-b --dry-run`` picks first OPEN (``open_count`` stays 2); plan unchanged with both still OPEN. Proof: pytest ``-k beat59``.

Beat 60: ``watch_lookalike`` ``session-b --dry-run`` / idle-decode pick Queue ``Q3`` only (never Watch lookalike); plan unchanged. Proof: pytest ``-k beat60``.

Beat 61: ``session-b --dry-run`` on ``shipped_only`` / ``incomplete_open`` / ``watch_only_lookalike`` stays idle (never dry_run); plan unchanged. Proof: pytest ``-k beat61``.

Beat 62: ``session-b --dry-run`` on ``open_complete_not_ready`` / ``broken_ready_flag`` stays idle; ``next_queue_id`` remains public. Proof: pytest ``-k beat62``.

Beat 63: ``mixed_priority`` ``session-b --dry-run`` / idle-decode pick first ready OPEN; plan unchanged; ``count_queue_headings`` stays public. Proof: pytest ``-k beat63``.

Beat 64: ``one_open_ready`` ``session-b --dry-run`` / idle-decode pick Q1; plan unchanged; ``format_queue_item`` / ``require_unique_queue_section`` stay public. Proof: pytest ``-k beat64``.

Beat 65: ``two_queue_sections`` ``session-b --dry-run`` / ``session-a --dry-run`` still fail-closed (``verdict=failed``, non-zero); ``write_queue_section`` stays public. Proof: pytest ``-k beat65``.

Beat 66: ``session-a --no-stub`` on OPEN>=1 is light recount; on empty Queue fails closed (no stub fallback); ``stub_brainstorm`` / ``dry_run_implement`` / ``make_fixture_ship_implement`` stay public. Proof: pytest ``-k beat66``.

Beat 67: ``session-a --no-stub --dry-run`` keeps the same light/failed paths without mutating the plan; ``decode_fields`` / ``format_decode_report`` / ``first_b_pick`` stay public. Proof: pytest ``-k beat67``.

Beat 68: ``first_b_pick`` returns the first ready OPEN (None on empty/shipped-only); idle-decode ``--json`` message still includes ``decode pick``; ``incomplete_candidate_reasons({{}})`` lists all six fields + ready. Proof: pytest ``-k beat68``.

Beat 69: CLI ``--json`` matches ``session_a_result_dict`` / ``session_b_result_dict`` for queued/picked; those helpers stay public with ``SESSION_*_RESULT_JSON_KEYS``. Proof: pytest ``-k beat69``.

Beat 70: ``session-a --json`` light/failed keys match ``SESSION_A_RESULT_JSON_KEYS``; ``run_session_a_path`` / ``run_session_b_path`` are public. Proof: pytest ``-k beat70``.

Beat 71: ``session-b --json`` dry_run/idle/failed keys match ``SESSION_RESULT_JSON_KEYS``; ``to_dict`` / ``to_json_dict`` / ``session_b_result_dict`` stay aligned. Proof: pytest ``-k beat71``.

Beat 72: Session A ``to_dict`` / ``to_json_dict`` / ``session_a_result_dict`` align for queued/light/failed; CLI ``--json`` light already keys-locked. Proof: pytest ``-k beat72``.

Beat 73: ``SessionAResult`` / ``SessionResult`` / ``SessionBResult`` / ``QueueItem`` stay public; result classes still expose ``to_dict``. Proof: pytest ``-k beat73``.

Beat 74: ``QueueItem`` from ``one_open_ready`` is B-pickable; ``format_queue_item`` round-trips via ``parse_queue_items``; ``is_b_pickable`` stays public. Proof: pytest ``-k beat74``.

Beat 75: ``incomplete_open`` / ``shipped_only`` items are not B-pickable; ``count_open`` stays public and returns 0 for those fixtures. Proof: pytest ``-k beat75``.

Beat 76: ``open_complete_not_ready`` is not B-pickable (``count_open`` 0); ``broken_ready_flag`` likewise; ``is_ready_yes`` rejects READY/maybe. Proof: pytest ``-k beat76``.

Beat 77: ``mixed_priority`` ``first_b_pick`` skips incomplete rows and returns the first ready OPEN; ``count_open`` counts only pickable items. Proof: pytest ``-k beat77``.

Beat 78: ``two_open_ready`` ``first_b_pick`` returns the first OPEN; ``count_open`` is 2; second stays OPEN. Proof: pytest ``-k beat78``.

Beat 79: ``one_open_ready`` ``first_b_pick`` / ``count_open`` are 1; ``empty_queue`` stays 0 / None. Proof: pytest ``-k beat79``.

Beat 80: milestone — top-level ``--help`` still lists session-a/session-b/idle-decode; public API still exports Session A/B runners + queue helpers; Makefile echo reaches beat80. Proof: pytest ``-k beat80``.
Beat 81: watch_lookalike ``first_b_pick`` is Q3 (not Watch); watch_only idle; ``mark_item_shipped`` stays public. Proof: pytest ``-k beat81``.
Beat 82: subcommand ``--help`` still lists ``--plan``/``--json``; ``render_queue_count`` + ``QUEUE_CAPACITY`` stay public. Proof: pytest ``-k beat82``.
Beat 83: incomplete/shipped/not-ready/broken/empty ``first_b_pick`` is None; ``serialize_queue_item(s)`` stay public. Proof: pytest ``-k beat83``.
Beat 84: pickable fixtures ``first_b_pick`` ids (Q1/Q1/Q3/Q2); ``is_complete_six_field`` + ``is_open_status`` stay public. Proof: pytest ``-k beat84``.
Beat 85: session-a producer help flags; session-b ``--decode-only``; idle-decode omits ``--dry-run``; ``REQUIRED_FIELDS`` public. Proof: pytest ``-k beat85``.
Beat 86: idle ``queue 0/10`` + ``scheduler_delete`` guard; ``SchedulerDeleteForbidden`` public; decode report ``decode pick``. Proof: pytest ``-k beat86``.
Beat 87: two_queue ``AmbiguousQueueError`` + heading count=2; ``format_decode_report``/``decode_fields`` stay public. Proof: pytest ``-k beat87``.
Beat 88: Watch/Heartbeat fixtures idle or pick Q1; ``SESSION_A_RESULT_KEYS`` / ``SESSION_B_RESULT_KEYS`` stay public. Proof: pytest ``-k beat88``.
Beat 89: ``incomplete_candidate_reasons`` contract; ``SESSION_*_JSON_KEYS`` stay public. Proof: pytest ``-k beat89``.
Beat 90: milestone — top-level ``--help`` + public API through SchedulerDeleteForbidden/REQUIRED_FIELDS/JSON keys; Makefile echo reaches beat90. Proof: pytest ``-k beat90``.
Beat 91: empty Queue append ``next_queue_id`` Q1 becomes pickable; append/format/write helpers stay public. Proof: pytest ``-k beat91``.
Beat 92: ``mark_item_shipped`` clears pick; ``stub_brainstorm`` ready yes; stub/dry-run helpers stay public. Proof: pytest ``-k beat92``.
Beat 93: ``is_ready_yes`` aliases; ``default_search_plan``/``stub_search_plan`` alias stub; result-dict helpers public. Proof: pytest ``-k beat93``.
Beat 94: session_*_result_dict / to_dict match JSON keys; path runners stay public. Proof: pytest ``-k beat94``.
Beat 95: serialize_queue_item round-trip stays pickable; count_queue_headings fixture contract. Proof: pytest ``-k beat95``.
Beat 96: ``decode_fields`` covers six fields + status/ready; decode report lines 1–6. Proof: pytest ``-k beat96``.
Beat 97: ``next_queue_id`` progression (empty→Q1, one→Q2, two→Q3, lookalike→Q4); require_unique fail-closed. Proof: pytest ``-k beat97``.
Beat 98: ``is_b_pickable`` fixture matrix; complete/open/ready helpers stay public. Proof: pytest ``-k beat98``.

Session A: when OPEN is 0, uses ``--stub`` (deterministic six-field fill) or
``--candidate-json``; recount-only when OPEN >= 1. Appends at most one OPEN.
Session B / idle-decode: decode-only pick or idle fire (queue 0/10); never
``scheduler_delete``. Session B ``--dry-run`` exercises the default dry-run
implement callback (records file_touch / acceptance; no repo write).

SHIPPED is callback-only: there is no CLI ``--implement stub-ship`` (or similar)
flag. Prefer pytest with ``make_fixture_ship_implement`` / ``fixture_ship_implement``
on ``tmp_path`` plans. Live prod implement stays unwired.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.research_implement.session_a import run_session_a_path, stub_brainstorm
from src.research_implement.session_b import dry_run_implement, run_session_b_path


def _load_candidate(path: Path | None) -> dict | None:
    """Load a Session A brainstorm candidate from JSON (dict or list).

    Beat 15: ``session-a --candidate-json`` accepts either a single candidate
    object or a JSON list; a list uses the first dict element. Incomplete
    candidates still fail-closed in Session A (Beat 14). Returns None when
    path is omitted or the list has no dict element.

    Beat 16: missing file, invalid JSON, or wrong top-level type (not object/list)
    raise ``SystemExit`` with a clear ``--candidate-json ...`` message (non-zero
    CLI failure). Callers must not mutate the plan on these paths.
    """
    if path is None:
        return None
    # Beat 32: existing non-file (directory) → clear not-a-file exit.
    if path.exists() and not path.is_file():
        raise SystemExit(f"--candidate-json is not a file: {path}")
    if not path.is_file():
        raise SystemExit(f"--candidate-json file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise SystemExit(f"--candidate-json invalid JSON: {err}") from err
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                return item
        return None
    raise SystemExit(
        f"--candidate-json must be a JSON object or list of objects, got {type(raw).__name__}"
    )


def _resolve_plan(args: argparse.Namespace) -> Path:
    plan = getattr(args, "plan", None)
    log = getattr(args, "log", None)
    if plan is not None and log is not None:
        raise SystemExit("pass only one of --plan / --log")
    # Beat 33: do not use ``plan or log`` — empty string is falsy but still "passed".
    if plan is not None:
        path = plan
    elif log is not None:
        path = log
    else:
        raise SystemExit("--plan or --log is required")
    # Empty / whitespace-only path fails closed (do not coerce to Path(".")).
    text = str(path).strip()
    if not text:
        raise SystemExit("--plan/--log path is empty")
    resolved = Path(text)
    # Beat 23: missing plan/log path fail-closed (do not create).
    # Beat 31: existing non-file (directory/symlink-to-dir) → clear not-a-file exit.
    if resolved.exists() and not resolved.is_file():
        raise SystemExit(f"--plan/--log is not a file: {resolved}")
    if not resolved.is_file():
        raise SystemExit(f"--plan/--log file not found: {resolved}")
    return resolved

def _add_plan_log(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group(required=True)
    # Beat 33: keep as str so empty / whitespace is not coerced to Path(".").
    g.add_argument(
        "--plan",
        type=str,
        help="Living plan.md path (fixture or working copy)",
    )
    g.add_argument(
        "--log",
        type=str,
        help="Alias for --plan (e.g. logs/research-implement.md host contract)",
    )


def _run_session_b(
    plan: Path,
    *,
    as_json: bool,
    dry_run: bool = False,
) -> int:
    """Session B CLI: decode-only by default; ``dry_run`` uses dry_run_implement.

    Empty Queue still idle-fires (queue 0/10). Never ``scheduler_delete``.
    Dry-run records intended file_touch / acceptance and never writes the repo.

    Beat 17: default path never invokes implement (spy-proven); only ``dry_run``
    passes ``dry_run_implement``. fixture_ship is never CLI-wired.
    """
    if dry_run:
        result = run_session_b_path(
            plan,
            implement=dry_run_implement,
            decode_only=False,
            write=False,
        )
    else:
        # Beat 17: decode-only — never pass/call implement; never delete schedule.
        result = run_session_b_path(plan, decode_only=True, write=False)
    assert result.keep_schedule and not result.scheduler_delete_called
    if as_json:
        # Single shared SessionResult.to_dict shape (idle / decode_only / dry_run / shipped).
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(result.message)
        if result.decode_report and result.verdict in {"picked", "dry_run"}:
            if result.item is not None and f"decode pick {result.item.item_id}" not in result.message:
                print(result.decode_report)
    return 0 if result.ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.research_implement",
        description=(
            "Session A/B research-implement loop (Queue producer/consumer). "
            "Side-dev only — no Tasker yaml on prod ports, no live LLM."
        ),
        epilog=(
            "Subcommands: session-a (producer), session-b (decode / optional --dry-run implement), "
            "idle-decode (alias of session-b decode; idle fire on empty Queue)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser(
        "session-a",
        help="Producer: stub/candidate brainstorm when OPEN is 0; recount when OPEN>=1",
    )
    _add_plan_log(a)
    a.add_argument(
        "--candidate-json",
        type=str,
        default=None,
        help=(
            "Candidate JSON (object or list of objects) loaded when OPEN is 0 "
            "(overrides --stub); incomplete fail-closes; ignored on OPEN>=1 recount-only"
        ),
    )
    a.add_argument(
        "--stub",
        action="store_true",
        help="Use deterministic stub_brainstorm when OPEN is 0 (no LLM)",
    )
    a.add_argument(
        "--no-stub",
        action="store_true",
        help="Do not use stub; without --candidate-json empty OPEN fails fire",
    )
    a.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write the plan file",
    )
    a.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit SessionAResult.to_dict JSON (append/queued vs recount-only/light)",
    )

    b = sub.add_parser(
        "session-b",
        help="Consumer: decode first ready OPEN Queue item (decode-only; idle on empty)",
    )
    _add_plan_log(b)
    b.add_argument(
        "--decode-only",
        action="store_true",
        default=True,
        help="Decode/pick only (default); never implements code / no prod wire-up",
    )
    b.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run pluggable dry-run implement (record file_touch / acceptance; "
            "no repo write). Still idle-fires on empty Queue; never scheduler_delete."
        ),
    )
    b.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help=(
            "Emit shared SessionResult.to_dict JSON "
            "(idle | picked decode_only | dry_run; never scheduler_delete)"
        ),
    )

    idle = sub.add_parser(
        "idle-decode",
        help=(
            "Decode-only Session B alias: idle fire when empty/not-ready; "
            "picked decode_only when OPEN ready; never scheduler_delete"
        ),
        description=(
            "Decode-only Session B alias. Idle fire when Queue empty/not-ready; "
            "picked decode_only when OPEN ready. Never scheduler_delete."
        ),
    )
    _add_plan_log(idle)
    idle.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help=(
            "Emit shared SessionResult.to_dict JSON "
            "(idle | picked decode_only; same keys as session-b --json)"
        ),
    )

    args = parser.parse_args(argv)

    if args.cmd == "session-a":
        plan = _resolve_plan(args)
        # Beat 25: --stub and --no-stub are mutually exclusive.
        if getattr(args, "stub", False) and getattr(args, "no_stub", False):
            raise SystemExit("pass only one of --stub / --no-stub")
        # Beat 15: --candidate-json supplies brainstorm when OPEN==0. Load is
        # deferred inside the callback so OPEN>=1 recount-only never reads or
        # appends the candidate. Empty/non-dict JSON → None → failed fire (no stub).
        if args.candidate_json is not None:
            # Beat 37: empty / whitespace --candidate-json fails closed (no Path(".")).
            cand_text = str(args.candidate_json).strip()
            if not cand_text:
                raise SystemExit("--candidate-json path is empty")
            cand_path = Path(cand_text)

            def _brainstorm(_items, _path=cand_path):
                return _load_candidate(_path)

            brainstorm = _brainstorm
        elif args.no_stub:
            brainstorm = None
        else:
            # Default side-dev (--stub or omitted): deterministic stub, no LLM.
            brainstorm = stub_brainstorm

        result = run_session_a_path(
            plan,
            brainstorm=brainstorm,
            write=not args.dry_run,
        )
        if getattr(args, "as_json", False):
            # Single shared SessionAResult.to_dict shape (queued / light / failed).
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        else:
            print(result.message)
        return 0 if result.ok else 1

    if args.cmd in {"session-b", "idle-decode"}:
        plan = _resolve_plan(args)
        as_json = bool(getattr(args, "as_json", False))
        dry_run = bool(getattr(args, "dry_run", False))
        # idle-decode stays decode-only; session-b may opt into --dry-run implement.
        return _run_session_b(plan, as_json=as_json, dry_run=dry_run)

    parser.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
