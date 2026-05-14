#!/usr/bin/env python3
"""Verify cron_status.json integrity — all expected jobs present, no extras."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from cron_compat import CRON_TARGETS

status_file = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "cron_status.json"
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
