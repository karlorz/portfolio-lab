"""Tests for SkillWiki/Hermes routing contract audit."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "audit_routing_contract.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_routing_contract", AUDIT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_fixture(root: Path, *, coder_enabled: bool = False, finance_profile: str = "finance") -> tuple[Path, Path]:
    skillwiki_env = root / ".skillwiki" / ".env"
    skillwiki_env.parent.mkdir(parents=True)
    skillwiki_env.write_text(
        "\n".join(
            [
                "WIKI_PATH=/root/wiki",
                "WIKI_LANG=en",
                "WIKI_DEFAULT=portfolio",
                "WIKI_PORTFOLIO_PATH=/root/wiki",
                "WIKI_FINANCE_PATH=/root/wiki-fin",
                "",
            ]
        ),
        encoding="utf-8",
    )

    hermes_home = root / ".hermes"
    (hermes_home / "cron").mkdir(parents=True)
    (hermes_home / ".env").write_text("WIKI_PATH=/root/wiki\n", encoding="utf-8")
    (hermes_home / "cron" / "jobs.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "finance-1",
                        "name": "finance-digest",
                        "enabled": True,
                        "state": "scheduled",
                        "profile": finance_profile,
                        "workdir": "/root/wiki-fin",
                        "script": "finance-digest-wrapper.sh",
                    },
                    {
                        "id": "portfolio-default-paused",
                        "name": "portfolio-lab-dashboard",
                        "enabled": False,
                        "state": "paused",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    finance_home = hermes_home / "profiles" / "finance"
    (finance_home / "scripts").mkdir(parents=True)
    (finance_home / ".env").write_text("WIKI_PATH=/root/wiki-fin\n", encoding="utf-8")
    for name in ("finance-digest-wrapper.sh", "finance-news-collector.py"):
        script = finance_home / "scripts" / name
        script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        script.chmod(script.stat().st_mode | stat.S_IXUSR)

    coder_home = hermes_home / "profiles" / "coder" / "cron"
    coder_home.mkdir(parents=True)
    coder_home.joinpath("jobs.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "coder-portfolio",
                        "name": "portfolio-lab-dashboard",
                        "enabled": coder_enabled,
                        "state": "scheduled" if coder_enabled else "paused",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return skillwiki_env, hermes_home


def test_audit_passes_when_routing_contract_holds(tmp_path, capsys) -> None:
    module = _load_module()
    skillwiki_env, hermes_home = _write_fixture(tmp_path)

    assert (
        module.main(["--home", str(tmp_path), "--hermes-home", str(hermes_home), "--skillwiki-env", str(skillwiki_env)])
        == 0
    )
    output = capsys.readouterr().out
    assert "OK: routing contract holds" in output


def test_audit_rejects_enabled_coder_portfolio_jobs(tmp_path, capsys) -> None:
    module = _load_module()
    skillwiki_env, hermes_home = _write_fixture(tmp_path, coder_enabled=True)

    assert (
        module.main(["--home", str(tmp_path), "--hermes-home", str(hermes_home), "--skillwiki-env", str(skillwiki_env)])
        == 1
    )
    output = capsys.readouterr().out
    assert "coder Hermes profile has no enabled portfolio-lab jobs" in output


def test_audit_rejects_finance_digest_without_finance_profile(tmp_path, capsys) -> None:
    module = _load_module()
    skillwiki_env, hermes_home = _write_fixture(tmp_path, finance_profile="")

    assert (
        module.main(["--home", str(tmp_path), "--hermes-home", str(hermes_home), "--skillwiki-env", str(skillwiki_env)])
        == 1
    )
    output = capsys.readouterr().out
    assert "finance-digest runs under Hermes profile finance" in output
