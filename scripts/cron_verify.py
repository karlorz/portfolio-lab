#!/usr/bin/env python3
"""Verify cron_status.json integrity — all expected jobs present, no extras."""
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.cron_compat import CRON_TARGETS

status_file = os.environ.get(
    "CRON_STATUS_FILE",
    os.path.join(str(PROJECT_ROOT), "data", "cron_status.json"),
)

if not os.path.exists(status_file):
    print(f"MISSING: {status_file} — run 'make cron-reset' first")
    sys.exit(1)

with open(status_file) as f:
    data = json.load(f)

names = [j["name"] for j in data["jobs"]]
missing = set(CRON_TARGETS) - set(names)
extra = set(names) - set(CRON_TARGETS)

if missing:
    print(f"FAIL: Missing jobs in cron_status.json: {missing}")
    sys.exit(1)
if extra:
    print(f"WARN: Extra jobs in cron_status.json: {extra}")

print(f"OK: {len(names)} jobs tracked, all {len(CRON_TARGETS)} expected targets present")
