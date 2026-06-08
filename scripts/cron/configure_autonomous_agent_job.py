#!/usr/bin/env python3
"""Configure the portfolio-lab autonomous Hermes cron job as guarded no-agent."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
JOB_NAME = "portfolio-lab-autonomous-agent"
SCRIPT_NAME = "portfolio-lab-autonomous-agent.sh"


def _load_jobs(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_jobs(path: Path, data: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)
        handle.write("\n")
    tmp_path.replace(path)


def _copy_script(hermes_home: Path, project_dir: Path, dry_run: bool) -> bool:
    source = project_dir / "scripts" / "cron" / SCRIPT_NAME
    target_dir = hermes_home / "scripts"
    target = target_dir / SCRIPT_NAME

    source_bytes = source.read_bytes()
    if target.exists() and target.read_bytes() == source_bytes:
        return False

    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        target.chmod(target.stat().st_mode | stat.S_IXUSR)
    return True


def configure_autonomous_agent_job(
    *,
    hermes_home: str | Path | None = None,
    project_dir: str | Path = PROJECT_ROOT,
    dry_run: bool = False,
) -> bool:
    """Rewrite the live Hermes job to run the guarded no-agent wrapper."""
    resolved_home = Path(hermes_home or os.environ.get("HERMES_HOME", "/root/.hermes")).expanduser()
    resolved_project = Path(project_dir).expanduser().resolve()
    jobs_path = resolved_home / "cron" / "jobs.json"
    data = _load_jobs(jobs_path)

    changed = _copy_script(resolved_home, resolved_project, dry_run)
    matched = False
    for job in data.get("jobs", []):
        if job.get("name") != JOB_NAME:
            continue
        matched = True
        updates = {
            "script": SCRIPT_NAME,
            "no_agent": True,
            "enabled_toolsets": None,
            "workdir": str(resolved_project),
        }
        for key, value in updates.items():
            if job.get(key) != value:
                job[key] = value
                changed = True

    if not matched:
        raise RuntimeError(f"Hermes cron job {JOB_NAME!r} not found in {jobs_path}")

    if changed:
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        if not dry_run:
            _write_jobs(jobs_path, data)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes-home", default=os.environ.get("HERMES_HOME", "/root/.hermes"))
    parser.add_argument("--project-dir", default=str(PROJECT_ROOT))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    changed = configure_autonomous_agent_job(
        hermes_home=args.hermes_home,
        project_dir=args.project_dir,
        dry_run=args.dry_run,
    )
    print("changed" if changed else "already configured")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
