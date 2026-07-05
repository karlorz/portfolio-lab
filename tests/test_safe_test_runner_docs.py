from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_SAFE_TEST_DOCS = (
    PROJECT_ROOT / "scripts" / "run-tests-safe",
    PROJECT_ROOT / "tests" / "conftest.py",
)
STALE_SAFE_COUNT_LABELS = (
    "~3700",
    "3700 tests",
    "3700 non-ML tests",
    "3900-test suite",
)


def test_safe_test_docs_do_not_hardcode_stale_count_labels() -> None:
    """Safe-test docs should defer exact counts to current pytest output."""
    findings: list[str] = []

    for path in ACTIVE_SAFE_TEST_DOCS:
        text = path.read_text(encoding="utf-8")
        for label in STALE_SAFE_COUNT_LABELS:
            if label in text:
                findings.append(f"{path.relative_to(PROJECT_ROOT)}: {label}")

    assert findings == []
