"""Ban ENABLE/ALLOW-style raw os.environ comparisons to literal 1 outside env_flags."""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
ENV_FLAGS = SRC_ROOT / "env_flags.py"

# Flag-like names: ENABLE/ALLOW/REQUIRE/DISABLE/FORCE or known gates.
FLAG_NAME = re.compile(
    r"(?:ENABLE|ALLOW|REQUIRE|DISABLE|FORCE|_ENABLED|_LOCKED)",
    re.IGNORECASE,
)
RAW_COMPARE = re.compile(
    r"""os\.(?:environ\.get|getenv)\(\s*['"]([A-Z0-9_]+)['"][^)]*\)\s*(==|!=)\s*['"]1['"]"""
)


def _iter_src_py() -> list[Path]:
    return sorted(
        path
        for path in SRC_ROOT.rglob("*.py")
        if path.resolve() != ENV_FLAGS.resolve()
        and "__pycache__" not in path.parts
    )


def test_no_raw_enable_style_literal_one_compares_outside_env_flags():
    offenders: list[str] = []
    for path in _iter_src_py():
        text = path.read_text(encoding="utf-8")
        for match in RAW_COMPARE.finditer(text):
            name, _op = match.group(1), match.group(2)
            if not FLAG_NAME.search(name):
                continue
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            offenders.append(f"{rel}: {match.group(0)}")
    assert offenders == [], "use env_literal_one(...); raw compares:\n" + "\n".join(offenders)


def test_env_flags_module_is_the_single_literal_one_helper():
    body = ENV_FLAGS.read_text(encoding="utf-8")
    assert "def env_literal_one(" in body
    assert '== "1"' in body


RAW_ALPACA_PAPER = re.compile(
    r"""os\.(?:environ\.get|getenv)\(\s*['\"]ALPACA_PAPER['\"]"""
)


def test_no_raw_alpaca_paper_reads_outside_env_flags():
    """ALPACA_PAPER must go through env_paper_unless_false (fail-open to paper)."""
    offenders: list[str] = []
    for path in _iter_src_py():
        text = path.read_text(encoding="utf-8")
        for match in RAW_ALPACA_PAPER.finditer(text):
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            offenders.append(f"{rel}: {match.group(0)}")
    assert offenders == [], (
        "use env_paper_unless_false(); raw ALPACA_PAPER reads:\n" + "\n".join(offenders)
    )


def test_env_flags_module_is_the_single_paper_helper():
    body = ENV_FLAGS.read_text(encoding="utf-8")
    assert "def env_paper_unless_false(" in body
    assert 'not in ("false", "0", "no")' in body


# Migrated fail-closed gates: any direct getenv must go through env_literal_one.
KNOWN_LITERAL_ONE_GATES = frozenset(
    {
        "PORTFOLIO_LAB_ENABLE_ML",
        "PORTFOLIO_LAB_ENABLE_BROKER_SNAPSHOT",
        "PORTFOLIO_LAB_ALLOW_REPO_PUBLIC_DATA",
        "PORTFOLIO_LAB_ALLOW_LIVE_PUBLIC",
        "PORTFOLIO_LAB_ALLOW_LIVE_INCIDENTS",
        "PORTFOLIO_LAB_ALLOW_PROD_SIDECAR",
        "PORTFOLIO_LAB_FORCE_PUBLIC_PROJECTION",
        "TASKER_DISABLE_SCHEDULER",
        "TASKER_INCLUDE_HERMES_AUDIT",
        "JSON_LOGS",
        "PROMOTION_REQUIRE_WFE",
        "DECISION_REGISTRY_RECORD_BACKTEST",
        "PORTFOLIO_LAB_ALT_DATA_AUTO_PROJECT",
        "INCIDENT_KILL_SWITCH_ESCALATION_ENABLED",
        "INFERENCE_TIME_PLANNING",
        "ALPACA_ALLOW_EXTENDED_HOURS",
        "BROKER_ALLOW_EXTENDED_HOURS",
    }
)
RAW_KNOWN_GATE = re.compile(
    r"""os\.(?:environ\.get|getenv)\(\s*['\"]("""
    + "|".join(sorted(KNOWN_LITERAL_ONE_GATES))
    + r""")['\"]"""
)


def test_no_raw_known_literal_one_gate_reads_outside_env_flags():
    """Known env_literal_one gates must not be read via raw getenv/environ.get."""
    offenders: list[str] = []
    for path in _iter_src_py():
        body = path.read_text(encoding="utf-8")
        for match in RAW_KNOWN_GATE.finditer(body):
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            offenders.append(f"{rel}: {match.group(0)}")
    assert offenders == [], (
        "use env_literal_one(...); raw known-gate reads:\n" + "\n".join(offenders)
    )
