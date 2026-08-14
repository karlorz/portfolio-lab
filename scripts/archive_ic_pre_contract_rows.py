#!/usr/bin/env python3
"""Archive IC observation rows that can never align under the corrected contracts.

MAIN-ITEM-1 s4 one-shot maintenance: loads the live IC monitor state
(data/ic_monitor_state.json), archives rows that can never pass the dynamic
alignment check — (a) None rows (missing metadata entirely) or (b) rows whose
stamped prediction_field differs from the current contract's intended field —
in the alignment-participating cohorts (declared "misaligned"/"provisional"),
and saves the state. Archived rows land in data/ic_rebaseline_archives/ in the
ic-rebaseline-archive/v1 format ("nothing is lost" precedent).

Idempotent: a re-run finds no matching rows and archives 0.

Usage:
    python scripts/archive_ic_pre_contract_rows.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.monitor.ic_decay_monitor import ICMonitor  # noqa: E402  # deliberate placement (bootstrap/sys.path ordering)


def main() -> int:
    monitor = ICMonitor()
    monitor.load_state()
    archived = monitor.archive_pre_contract_rows()
    monitor.save_state()
    print(f"archived: {archived}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
