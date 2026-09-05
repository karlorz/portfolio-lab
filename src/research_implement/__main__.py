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
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_plan(args: argparse.Namespace) -> Path:
    plan = getattr(args, "plan", None)
    log = getattr(args, "log", None)
    if plan and log:
        raise SystemExit("pass only one of --plan / --log")
    path = plan or log
    if path is None:
        raise SystemExit("--plan or --log is required")
    return Path(path)


def _add_plan_log(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--plan",
        type=Path,
        help="Living plan.md path (fixture or working copy)",
    )
    g.add_argument(
        "--log",
        type=Path,
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
    """
    if dry_run:
        result = run_session_b_path(
            plan,
            implement=dry_run_implement,
            decode_only=False,
            write=False,
        )
    else:
        # Decode-only: pick Q id / idle; never implement; never delete schedule.
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
        type=Path,
        default=None,
        help="Six-field candidate JSON used when OPEN is 0 (overrides --stub)",
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
        help="Emit decode fields as JSON (Q id + six fields) when picked",
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
        candidate = _load_candidate(args.candidate_json)
        if candidate is not None:
            def _brainstorm(_items):
                return candidate

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
