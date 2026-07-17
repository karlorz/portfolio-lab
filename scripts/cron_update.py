#!/usr/bin/env python3
"""Update cron_status.json after a job run. Called from Makefile targets."""
import json
import os
import sys
import re
from datetime import datetime
from pathlib import Path

# Use cron_compat for backend discovery
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
try:
    from src.cron_compat import active_backend
    _default_backend = active_backend()
except ImportError:
    _default_backend = os.environ.get("CRON_BACKEND", "manual")

_METADATA_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_metadata(args):
    metadata = {}
    for item in args:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        if not _METADATA_KEY_RE.match(key):
            continue
        metadata[key] = value
    return metadata

def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <job_name> <status> <duration_seconds> [backend]", file=sys.stderr)
        sys.exit(1)

    job_name = sys.argv[1]
    status = sys.argv[2]
    duration = float(sys.argv[3])
    backend = sys.argv[4] if len(sys.argv) > 4 else _default_backend
    metadata = _parse_metadata(sys.argv[5:])

    status_file = os.path.join(
        str(PROJECT_ROOT),
        "data", "cron_status.json"
    )

    os.makedirs(os.path.dirname(status_file), exist_ok=True)

    if os.path.exists(status_file):
        with open(status_file) as f:
            data = json.load(f)
    else:
        data = {"jobs": []}

    # Prefer the file-level SSOT backend (tasker) when an ad-hoc CLI invocation
    # omits an explicit backend and would otherwise stamp "manual"/hermes and
    # scramble dual-ownership bookkeeping.
    file_backend = str(data.get("backend") or "").strip().lower()
    backend_explicit = len(sys.argv) > 4 and "=" not in sys.argv[4]
    if not backend_explicit and file_backend in {"tasker", "crontab", "hermes"}:
        backend = file_backend

    now = datetime.now().isoformat()
    found = False
    for job in data["jobs"]:
        if job["name"] == job_name:
            job["status"] = status
            job["last_run"] = now
            job["duration_seconds"] = duration
            # Keep the job's established backend unless the caller explicitly
            # passed a fourth positional backend arg.
            if backend_explicit or not job.get("backend"):
                job["backend"] = backend
            job.update(metadata)
            found = True
            break

    if not found:
        row = {
            "name": job_name,
            "status": status,
            "last_run": now,
            "duration_seconds": duration,
            "backend": backend,
        }
        row.update(metadata)
        data["jobs"].append(row)

    data.setdefault("backend", backend if backend_explicit else (file_backend or backend))
    data["timestamp"] = now

    with open(status_file, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

if __name__ == "__main__":
    main()
