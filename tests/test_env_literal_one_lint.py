"""Ban ENABLE/ALLOW-style raw os.environ comparisons to literal 1 outside env_flags."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

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
