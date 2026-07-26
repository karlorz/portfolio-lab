import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_SAFE_TEST_SURFACES = (
    PROJECT_ROOT / "Makefile",
    PROJECT_ROOT / "scripts" / "run-tests-safe",
    PROJECT_ROOT / "tests" / "conftest.py",
)
# make test / run-tests-safe virtual address space cap (KB)
SAFE_TEST_CAP_KB = 6_291_456
STALE_ONE_GB_LABEL = re.compile(r"\b1\s*GB\b", re.IGNORECASE)
STALE_THREE_GB_CAP = re.compile(r"ulimit -v 3145728")
SIX_GB_LABEL = re.compile(r"\b6\s*GB\b|\b6,?144\s*MB\b", re.IGNORECASE)


def test_safe_test_memory_cap_docs_match_ulimit_value() -> None:
    """Active safe-test docs should label 6291456 KB as 6GB (raised after MemoryError cascade)."""
    assert SAFE_TEST_CAP_KB // 1024 == 6144
    assert SAFE_TEST_CAP_KB / 1024 / 1024 == 6.0

    findings: list[str] = []
    for path in ACTIVE_SAFE_TEST_SURFACES:
        text = path.read_text(encoding="utf-8")
        rel_path = path.relative_to(PROJECT_ROOT)

        if STALE_ONE_GB_LABEL.search(text):
            findings.append(f"{rel_path}: stale 1GB memory-cap label")
        # make test target must not still use 3GB ulimit (cron jobs may)
        if path.name == "Makefile":
            # Extract test: target body only
            import re as _re
            m = _re.search(r"^test:\n(?P<body>.*?)(?=^\S|\Z)", text, _re.M | _re.S)
            body = m.group("body") if m else text
            if "ulimit -v 3145728" in body:
                findings.append(f"{rel_path}: make test still uses 3GB ulimit")
            if "ulimit -v 6291456" not in body:
                findings.append(f"{rel_path}: make test missing 6GB ulimit")
            if not SIX_GB_LABEL.search(body):
                findings.append(f"{rel_path}: make test preamble missing 6GB label")
        else:
            if STALE_THREE_GB_CAP.search(text) and path.name == "run-tests-safe":
                findings.append(f"{rel_path}: still has ulimit -v 3145728")
            if not SIX_GB_LABEL.search(text):
                findings.append(f"{rel_path}: missing 6GB memory-cap label")

    assert findings == []
