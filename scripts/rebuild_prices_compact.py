#!/usr/bin/env python3
"""Rebuild prices_compact.json as last-N compact-v1 (Batch BK/BV honesty).

Repo ``public/data/prices_compact.json`` can drift to a full-history mirror after
partial writes. This rewrites compact from prices.json (or market.db fallback)
using the same last-N contract as fetch-data.ts.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DEFAULT_N = 504


def _last_n_series(series: List[Any], n: int) -> List[Any]:
    if not isinstance(series, list):
        return []
    if n <= 0 or len(series) <= n:
        return series
    return series[-n:]


def compact_from_prices(prices: Dict[str, Any], n_bars: int = DEFAULT_N) -> Dict[str, Any]:
    symbols: Dict[str, List[Any]] = {}
    for sym, bars in prices.items():
        if str(sym).startswith("_") or sym in {"meta", "schema"}:
            continue
        if isinstance(bars, list):
            symbols[str(sym)] = _last_n_series(bars, n_bars)
        elif isinstance(bars, dict) and "bars" in bars and isinstance(bars["bars"], list):
            symbols[str(sym)] = _last_n_series(bars["bars"], n_bars)
    sha = None
    try:
        from src.dashboard.generator import _generator_git_sha_short
        sha = _generator_git_sha_short()
    except Exception:
        pass
    return {
        "meta": {
            "schema": "prices/compact-v1",
            "n_bars": int(n_bars),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator_git_sha": sha,
            "generator_git_sha_status": "full" if sha else "unavailable",
            "note": f"prices_compact is last-{n_bars} bars/symbol (not a full-history mirror)",
            "n_symbols": len(symbols),
            "live_authoritative": False,
        },
        "symbols": symbols,
    }


def main() -> int:
    from src.paths import DATA_DIR, PUBLIC_DATA_DIR

    n = int(os.environ.get("PRICES_COMPACT_N_BARS", str(DEFAULT_N)) or DEFAULT_N)
    # Prefer full prices from PUBLIC then private
    candidates = [
        Path(PUBLIC_DATA_DIR) / "prices.json",
        Path(DATA_DIR) / "prices.json",
        _PROJECT_ROOT / "public" / "data" / "prices.json",
    ]
    prices_path = next((p for p in candidates if p.exists()), None)
    if prices_path is None:
        print("No prices.json found", file=sys.stderr)
        return 1
    prices = json.loads(prices_path.read_text(encoding="utf-8"))
    if not isinstance(prices, dict):
        print("prices.json must be object", file=sys.stderr)
        return 1
    # unwrap nested symbols if compact-like already
    if "symbols" in prices and isinstance(prices["symbols"], dict):
        prices = prices["symbols"]
    compact = compact_from_prices(prices, n_bars=n)

    targets = [
        Path(PUBLIC_DATA_DIR) / "prices_compact.json",
        Path(DATA_DIR) / "prices_compact.json" if Path(DATA_DIR) != Path(PUBLIC_DATA_DIR) else None,
        _PROJECT_ROOT / "public" / "data" / "prices_compact.json",
    ]
    written = []
    for t in targets:
        if t is None:
            continue
        try:
            t.parent.mkdir(parents=True, exist_ok=True)
            tmp = t.with_suffix(t.suffix + ".tmp")
            tmp.write_text(json.dumps(compact), encoding="utf-8")
            tmp.replace(t)
            written.append(str(t))
        except OSError as exc:
            print(f"skip {t}: {exc}", file=sys.stderr)
    print(json.dumps({"ok": True, "n_bars": n, "n_symbols": compact["meta"]["n_symbols"], "written": written}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
