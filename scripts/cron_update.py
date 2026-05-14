#!/usr/bin/env python3
"""Update cron_status.json after a job run. Called from Makefile targets."""
import json
import os
import sys
from datetime import datetime

def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <job_name> <status> <duration_seconds> [backend]", file=sys.stderr)
        sys.exit(1)

    job_name = sys.argv[1]
    status = sys.argv[2]
    duration = float(sys.argv[3])
    backend = sys.argv[4] if len(sys.argv) > 4 else os.environ.get("CRON_BACKEND", "manual")

    status_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "cron_status.json"
    )

    os.makedirs(os.path.dirname(status_file), exist_ok=True)

    if os.path.exists(status_file):
        with open(status_file) as f:
            data = json.load(f)
    else:
        data = {"jobs": []}

    now = datetime.now().isoformat()
    found = False
    for job in data["jobs"]:
        if job["name"] == job_name:
            job["status"] = status
            job["last_run"] = now
            job["duration_seconds"] = duration
            job["backend"] = backend
            found = True
            break

    if not found:
        data["jobs"].append({
            "name": job_name,
            "status": status,
            "last_run": now,
            "duration_seconds": duration,
            "backend": backend,
        })

    with open(status_file, "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    main()
