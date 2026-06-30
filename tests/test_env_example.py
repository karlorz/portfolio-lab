"""Regression tests for environment variable documentation."""

from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENV_GET_PATTERN = re.compile(r"os\.(?:environ\.get|getenv)\(\s*['\"]([A-Z0-9_]+)['\"]")
ENV_EXAMPLE_PATTERN = re.compile(r"^([A-Z0-9_]+)=")

DYNAMIC_ENV_VARS = {
    "ALPACA_MARKET_SESSION_STATE",
    "BROKER_MARKET_SESSION_STATE",
    "LABS_SCORECARD_POLICY_FILE",
}


def _env_vars_used_by_src() -> set[str]:
    used: set[str] = set()
    for path in (PROJECT_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        used.update(ENV_GET_PATTERN.findall(text))
    return used | DYNAMIC_ENV_VARS


def _env_vars_documented_in_example() -> set[str]:
    documented: set[str] = set()
    for line in (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        match = ENV_EXAMPLE_PATTERN.match(line)
        if match:
            documented.add(match.group(1))
    return documented


def test_env_example_documents_src_env_vars():
    missing = _env_vars_used_by_src() - _env_vars_documented_in_example()
    assert missing == set(), f"Missing .env.example entries: {sorted(missing)}"
