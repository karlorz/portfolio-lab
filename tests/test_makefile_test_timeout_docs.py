import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = PROJECT_ROOT / "Makefile"
EXPECTED_SAFE_TEST_TIMEOUT_SECONDS = "1200"


def _make_test_target_body() -> str:
    text = MAKEFILE.read_text(encoding="utf-8")
    match = re.search(
        r"^test:\n(?P<body>.*?)(?=^\S|\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None
    return match.group("body")


def test_make_test_timeout_docs_match_command() -> None:
    """The make test timeout preamble, command, and failure text should agree."""
    body = _make_test_target_body()

    findings: list[str] = []
    if f"Timeout: {EXPECTED_SAFE_TEST_TIMEOUT_SECONDS}s" not in body:
        findings.append("make test preamble does not disclose the 1200s timeout")
    if f"timeout {EXPECTED_SAFE_TEST_TIMEOUT_SECONDS} uv run pytest" not in body:
        findings.append("make test command does not use timeout 1200")
    if f"exceeded {EXPECTED_SAFE_TEST_TIMEOUT_SECONDS}s limit" not in body:
        findings.append("make test timeout failure text does not disclose 1200s")
    if "exceeded 600s limit" in body:
        findings.append("make test timeout failure text still discloses stale 600s")

    assert findings == []
