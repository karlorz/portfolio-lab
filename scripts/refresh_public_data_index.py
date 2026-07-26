#!/usr/bin/env python3
"""Rebuild public/data/index.json (Batch BY / market-data freshness).

fetch-data writes source_manifest after prices; dashboard gen rebuilds index,
but if dashboard lags, fails, or ops patches market files, SLO flags
``stale_index`` (index.generated_at < source_manifest.generated_at).

Usage::
  uv run python scripts/refresh_public_data_index.py
  uv run python scripts/refresh_public_data_index.py --reason source_manifest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--public-dir",
        type=Path,
        default=None,
        help="Public data dir (default: PUBLIC_DATA_DIR)",
    )
    parser.add_argument(
        "--reason",
        default="cli_refresh",
        help="content_patch_source suffix (default: cli_refresh)",
    )
    args = parser.parse_args(argv)

    from src.dashboard.public_data_index import refresh_public_data_index_after_partial_write
    from src.paths import PUBLIC_DATA_DIR

    root = Path(args.public_dir) if args.public_dir else Path(PUBLIC_DATA_DIR)
    out = refresh_public_data_index_after_partial_write(
        public_dir=root,
        reason=str(args.reason),
    )
    if out is None:
        print(json.dumps({"ok": False, "public_dir": str(root), "error": "refresh_failed"}))
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "public_dir": str(root),
                "generated_at": out.get("generated_at"),
                "generator_git_sha": out.get("generator_git_sha"),
                "content_patch_source": out.get("content_patch_source"),
                "files": len(out.get("files") or []),
                "live_authoritative": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
