"""CLI: python -m src.research_implement {session-a|session-b} --plan PATH

Side-dev entrypoints only. Does not start Tasker or touch production ports.
Session A requires --candidate-json when OPEN is 0 (no live LLM brainstorm).
Session B defaults to decode-only (no repo writes, no live implement to prod).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.research_implement.session_a import run_session_a_path
from src.research_implement.session_b import run_session_b_path


def _load_candidate(path: Path | None) -> dict | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.research_implement",
        description="Session A/B research-implement loop (Queue producer/consumer).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("session-a", help="Producer: brainstorm/append when OPEN is 0")
    a.add_argument("--plan", type=Path, required=True, help="Living plan.md path")
    a.add_argument(
        "--candidate-json",
        type=Path,
        default=None,
        help="Six-field candidate JSON used when OPEN is 0",
    )
    a.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write the plan file",
    )

    b = sub.add_parser(
        "session-b",
        help="Consumer: decode first ready OPEN Queue item (decode-only by default)",
    )
    b.add_argument("--plan", type=Path, required=True, help="Living plan.md path")
    b.add_argument(
        "--decode-only",
        action="store_true",
        default=True,
        help="Decode/pick only (default); never implements code / no prod wire-up",
    )
    b.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit decode fields as JSON (Q id + six fields) when picked",
    )

    args = parser.parse_args(argv)

    if args.cmd == "session-a":
        candidate = _load_candidate(args.candidate_json)

        def _brainstorm(_items):
            return candidate

        result = run_session_a_path(
            args.plan,
            brainstorm=_brainstorm if candidate is not None else None,
            write=not args.dry_run,
        )
        print(result.message)
        return 0 if result.ok else 1

    if args.cmd == "session-b":
        # Decode-only path: pick Q id, report fields; implement callback stays optional
        # and is intentionally not wired here (no live implement to prod).
        result = run_session_b_path(args.plan, decode_only=True, write=False)
        # Idle and picked are both success; never delete schedule.
        assert result.keep_schedule and not result.scheduler_delete_called
        if args.as_json:
            payload = {
                "ok": result.ok,
                "verdict": result.verdict,
                "open_count": result.open_count,
                "queue": result.queue_label,
                "keep_schedule": result.keep_schedule,
                "scheduler_delete_called": result.scheduler_delete_called,
                "item": None,
            }
            if result.item is not None:
                from src.research_implement.session_b import decode_fields

                payload["item"] = decode_fields(result.item)
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(result.message)
            if result.decode_report and result.verdict == "picked":
                # message already embeds report; ensure Q id is visible on stdout
                if f"decode pick {result.item.item_id}" not in result.message:
                    print(result.decode_report)
        return 0 if result.ok else 1

    parser.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
