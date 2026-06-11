#!/usr/bin/env python3
"""Audit SkillWiki and Hermes routing invariants for this host."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


PORTFOLIO_WIKI = "/root/wiki"
FINANCE_WIKI = "/root/wiki-fin"
FINANCE_JOB = "finance-digest"
FINANCE_PROFILE = "finance"
PORTFOLIO_JOB_PREFIX = "portfolio-lab-"


def parse_dotenv(path: Path) -> dict[str, str]:
    """Parse simple KEY=VALUE dotenv files without expanding secrets."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_jobs(path: Path) -> list[dict[str, Any]]:
    """Load Hermes cron jobs, ignoring non-object entries defensively."""
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    jobs = data.get("jobs", []) if isinstance(data, dict) else []
    return [job for job in jobs if isinstance(job, dict)]


def enabled_portfolio_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        job
        for job in jobs
        if str(job.get("name", "")).startswith(PORTFOLIO_JOB_PREFIX)
        and job.get("enabled") is True
        and job.get("state") != "paused"
    ]


def find_job(jobs: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for job in jobs:
        if job.get("name") == name:
            return job
    return None


def check(condition: bool, message: str, errors: list[str]) -> None:
    if condition:
        print(f"OK: {message}")
    else:
        print(f"ERROR: {message}")
        errors.append(message)


def audit(args: argparse.Namespace) -> int:
    errors: list[str] = []

    home = Path(args.home).expanduser()
    hermes_home = Path(args.hermes_home).expanduser()
    skillwiki_env = Path(args.skillwiki_env).expanduser()
    finance_profile_home = hermes_home / "profiles" / FINANCE_PROFILE
    coder_profile_home = hermes_home / "profiles" / "coder"

    skillwiki = parse_dotenv(skillwiki_env)
    default_hermes_env = parse_dotenv(hermes_home / ".env")
    finance_hermes_env = parse_dotenv(finance_profile_home / ".env")

    print("=== SkillWiki routing ===")
    check(skillwiki.get("WIKI_PATH") == PORTFOLIO_WIKI, "global WIKI_PATH points to /root/wiki", errors)
    check(skillwiki.get("WIKI_DEFAULT") == "portfolio", "default SkillWiki profile is portfolio", errors)
    check(skillwiki.get("WIKI_PORTFOLIO_PATH") == PORTFOLIO_WIKI, "portfolio profile points to /root/wiki", errors)
    check(skillwiki.get("WIKI_FINANCE_PATH") == FINANCE_WIKI, "finance profile points to /root/wiki-fin", errors)

    print("\n=== Hermes env routing ===")
    check(default_hermes_env.get("WIKI_PATH") == PORTFOLIO_WIKI, "default Hermes env points to /root/wiki", errors)
    check(finance_hermes_env.get("WIKI_PATH") == FINANCE_WIKI, "finance Hermes env points to /root/wiki-fin", errors)

    print("\n=== Hermes cron routing ===")
    default_jobs = load_jobs(hermes_home / "cron" / "jobs.json")
    finance_job = find_job(default_jobs, FINANCE_JOB)
    check(finance_job is not None, "finance-digest exists in default Hermes cron store", errors)
    if finance_job is not None:
        check(finance_job.get("enabled") is True, "finance-digest remains active", errors)
        check(finance_job.get("profile") == FINANCE_PROFILE, "finance-digest runs under Hermes profile finance", errors)
        check(finance_job.get("workdir") == FINANCE_WIKI, "finance-digest workdir is /root/wiki-fin", errors)
        check(
            finance_job.get("script") == "finance-digest-wrapper.sh",
            "finance-digest uses the finance wrapper script",
            errors,
        )

    default_enabled = enabled_portfolio_jobs(default_jobs)
    check(
        not default_enabled,
        "default Hermes cron has no enabled portfolio-lab jobs",
        errors,
    )

    coder_jobs = load_jobs(coder_profile_home / "cron" / "jobs.json")
    coder_enabled = enabled_portfolio_jobs(coder_jobs)
    check(
        not coder_enabled,
        "coder Hermes profile has no enabled portfolio-lab jobs",
        errors,
    )

    print("\n=== Finance profile scripts ===")
    wrapper = finance_profile_home / "scripts" / "finance-digest-wrapper.sh"
    collector = finance_profile_home / "scripts" / "finance-news-collector.py"
    check(wrapper.exists(), "finance wrapper exists in finance profile scripts", errors)
    check(collector.exists(), "finance collector exists in finance profile scripts", errors)
    check(os.access(wrapper, os.X_OK), "finance wrapper is executable", errors)
    check(os.access(collector, os.X_OK), "finance collector is executable", errors)

    if errors:
        print(f"\nFAIL: {len(errors)} routing contract violation(s)")
        return 1
    print("\nOK: routing contract holds")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", default="/root", help="home directory for default paths")
    parser.add_argument("--hermes-home", default="/root/.hermes", help="Hermes home directory")
    parser.add_argument(
        "--skillwiki-env",
        default="/root/.skillwiki/.env",
        help="SkillWiki dotenv file",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return audit(parse_args(argv or sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
