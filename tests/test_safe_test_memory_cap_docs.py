import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_SAFE_TEST_SURFACES = (
    PROJECT_ROOT / "Makefile",
    PROJECT_ROOT / "scripts" / "run-tests-safe",
    PROJECT_ROOT / "tests" / "conftest.py",
)
SAFE_TEST_CAP_KB = 3_145_728
STALE_ONE_GB_LABEL = re.compile(r"\b1\s*GB\b", re.IGNORECASE)
THREE_GB_LABEL = re.compile(r"\b3\s*GB\b|\b3,?072\s*MB\b", re.IGNORECASE)


def test_safe_test_memory_cap_docs_match_ulimit_value() -> None:
    """Active safe-test docs should label 3145728 KB as 3GB, not 1GB."""
    assert SAFE_TEST_CAP_KB // 1024 == 3072
    assert SAFE_TEST_CAP_KB / 1024 / 1024 == 3.0

    findings: list[str] = []
    for path in ACTIVE_SAFE_TEST_SURFACES:
        text = path.read_text(encoding="utf-8")
        rel_path = path.relative_to(PROJECT_ROOT)

        if STALE_ONE_GB_LABEL.search(text):
            findings.append(f"{rel_path}: stale 1GB memory-cap label")
        if not THREE_GB_LABEL.search(text):
            findings.append(f"{rel_path}: missing 3GB/3072 MB memory-cap label")

    assert findings == []
