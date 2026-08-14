#!/usr/bin/env python3
"""Explicit operator incident resolution (Task 2A path) via the incident manager.

Resolves an OPEN incident by id through ``IncidentStore.resolve_operator``:
journal append (resolved event) + kill-switch clear + summary write + kill/open
surface fan-out — the same lifecycle the alert path uses, without the
PASS-only manual-review hold. This is the operator's own action; PASS alerts
never invoke it.

Idempotent: resolving an already-resolved incident is a no-op (exit 0).
Refuses unknown incident ids (exit 1).

Usage:
    python scripts/resolve_incident.py --incident-id <id> --message "<notes>"
    python scripts/resolve_incident.py --incident-id <id> --message "<notes>" --dry-run
    # tmp-store (tests / rehearsal):
    python scripts/resolve_incident.py --incident-id <id> --message x \
        --log-path /tmp/incidents.jsonl --summary-path /tmp/incidents.json \
        --kill-switch-path /tmp/kill_switch.json
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.monitor.incident_manager import (  # noqa: E402  # deliberate placement (bootstrap/sys.path ordering)
    DEFAULT_INCIDENT_LOG_PATH,
    DEFAULT_INCIDENT_SUMMARY_PATH,
    DEFAULT_KILL_SWITCH_PATH,
    IncidentManager,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explicit operator incident resolution (Task 2A path)."
    )
    parser.add_argument("--incident-id", required=True, help="Incident id to resolve (e.g. 8115a9c1-...).")
    parser.add_argument("--message", required=True, help="Resolution notes recorded on the resolved event.")
    parser.add_argument("--dry-run", action="store_true", help="Print the action without writing anything.")
    parser.add_argument("--log-path", type=Path, default=DEFAULT_INCIDENT_LOG_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_INCIDENT_SUMMARY_PATH)
    parser.add_argument("--kill-switch-path", type=Path, default=DEFAULT_KILL_SWITCH_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = IncidentManager(
        log_path=args.log_path,
        summary_path=args.summary_path,
        kill_switch_path=args.kill_switch_path,
    )
    state = store.incident_state(args.incident_id)
    if state is None:
        print(
            f"unknown incident id: {args.incident_id} "
            f"(no events in {args.log_path})",
            file=sys.stderr,
        )
        return 1
    if state == "resolved":
        print(f"already resolved: {args.incident_id} (no-op)")
        return 0
    if args.dry_run:
        print(
            f"[dry-run] would resolve {args.incident_id} (state={state}) "
            f"with message: {args.message}"
        )
        return 0
    incident = store.resolve_operator(args.incident_id, args.message)
    if incident is None:
        print(f"resolve failed: {args.incident_id}", file=sys.stderr)
        return 1
    print(f"resolved: {incident.incident_id} (channel={incident.channel})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
