#!/usr/bin/env python3
"""Emit a dry-run artifact retention report for Labs and dashboard files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from src.research.artifact_retention import build_archive_dry_run_plan, build_retention_report


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for non-destructive retention reporting."""
    parser = argparse.ArgumentParser(description="Report keep/archive/prune recommendations for project artifacts")
    parser.add_argument("--data-dir", type=Path, default=None, help="Data directory to scan")
    parser.add_argument("--public-data-dir", type=Path, default=None, help="Public data directory to scan")
    parser.add_argument("--project-root", type=Path, default=None, help="Project root used for relative report paths")
    parser.add_argument("--archive-root", type=Path, default=None, help="Archive root to display in the report")
    parser.add_argument("--archive-plan", action="store_true", help="Emit a report-only archive move plan")
    parser.add_argument(
        "--execute-move",
        action="store_true",
        help="Reserved explicit opt-in for future move behavior; currently refused",
    )
    parser.add_argument(
        "--reference-root",
        action="append",
        type=Path,
        default=None,
        help="Markdown/text tree to scan for artifact references; may be repeated",
    )
    args = parser.parse_args(argv)
    if args.execute_move:
        sys.stderr.write("Move execution is not implemented by this report-only command.\n")
        return 2

    kwargs = {}
    if args.data_dir is not None:
        kwargs["data_dir"] = args.data_dir
    if args.public_data_dir is not None:
        kwargs["public_data_dir"] = args.public_data_dir
    if args.project_root is not None:
        kwargs["project_root"] = args.project_root
    if args.archive_root is not None:
        kwargs["archive_root"] = args.archive_root
    if args.reference_root is not None:
        kwargs["reference_roots"] = args.reference_root

    payload = build_archive_dry_run_plan(**kwargs) if args.archive_plan else build_retention_report(**kwargs)
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
