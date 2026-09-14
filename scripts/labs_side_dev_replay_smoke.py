#!/usr/bin/env python3
"""Deterministic Labs replay smoke target for side-dev publish path.

Emits fixed metrics on stdout so ``publish_labs_replays`` can produce a real
``labs_replays.json`` via ``data/labs_replay_targets.json`` without market data
fetch or unsafe shell. Not a fake-empty report — metrics match the paired
artifact under ``data/side_dev_replay_artifact.json``.
"""

from __future__ import annotations

import json

METRICS = {
    "sharpe": 1.05,
    "cagr_pct": 9.8,
    "max_drawdown_pct": -14.2,
}


def main() -> int:
    print(json.dumps({"metrics": METRICS}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
