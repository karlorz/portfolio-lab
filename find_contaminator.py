"""Binary search for test contaminator.

The contaminator is a test file whose MODULE-LEVEL code (executed during
pytest collection) patches numpy, sys.modules, or global state, causing
StatePersistence tests to fail when the full test directory is collected.

Strategy: Run collection on half the test files, then run the failing tests.
If they fail, the contaminator is in that half. If they pass, it's in the other half.
"""

import sys
import os
import subprocess
import glob

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.join(PROJECT_DIR, "tests")
ALL_FILES = sorted(glob.glob(os.path.join(TESTS_DIR, "test_*.py")))

# These are the known-good files (StatePersistence files that don't cause failures)
STATEPERSISTENCE_BASENAMES = set([
    "test_adaptive_ensemble_weights.py",
    "test_adaptive_sizing.py",
    "test_basis_pursuit_selector.py",
    "test_cross_asset_regime_arb.py",
    "test_cross_asset_relative_value.py",
    "test_duration_overlay.py",
    "test_hedge_efficiency.py",
    "test_macro_regime_synthesis.py",
    "test_regret_weighted_selector.py",
    "test_risk_budget_optimizer.py",
    "test_skew_engineering.py",
    "test_turnover_validator.py",
    "test_vixy_hedge_sizing.py",
])

# The 2 failing tests (relative to PROJECT_DIR so pytest can find them)
FAILING_TESTS = [
    "tests/test_regime_classifier.py::TestRegimeClassifierStatePersistence::test_regime_history_accumulates",
    "tests/test_transient_factors.py::TestStatePersistence::test_save_load_state",
]

FAILING_BASENAMES = {"test_regime_classifier.py", "test_transient_factors.py"}

def get_test_files_to_collect(file_subset):
    """Return list of test file paths for a given subset (excluding failing test files)."""
    result = []
    for f in file_subset:
        basename = os.path.basename(f)
        if basename not in FAILING_BASENAMES:
            result.append(f)
    return result

def relpath(f):
    """Convert absolute path to relative path from PROJECT_DIR."""
    return os.path.relpath(f, PROJECT_DIR)

def run_with_collection(file_subset):
    """Run pytest: collect from file_subset, then run failing tests."""
    files = get_test_files_to_collect(file_subset)
    if not files:
        return True  # no files to collect, must pass
    
    # Build relative paths
    file_args = [relpath(f) for f in files]
    
    cmd = [
        sys.executable, "-m", "pytest",
    ] + file_args + FAILING_TESTS + ["-v"]
    
    first = os.path.basename(files[0])
    last = os.path.basename(files[-1])
    print(f"Testing {len(files)} files: {first} ... {last}")
    sys.stdout.flush()
    
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=PROJECT_DIR,
        timeout=120
    )
    
    # Check if either failing test failed
    failed = False
    for line in result.stdout.split('\n'):
        if 'FAILED' in line:
            for ft in FAILING_TESTS:
                if ft.split("::")[0] in line:
                    failed = True
                    break
    
    # Also check stderr
    for line in result.stderr.split('\n'):
        if 'FAILED' in line:
            for ft in FAILING_TESTS:
                if ft.split("::")[0] in line:
                    failed = True
                    break
    
    if failed:
        print(f"  *** CONTAMINATION DETECTED in this batch!")
        for line in result.stdout.split('\n'):
            if 'FAILED' in line and ('test_regime_classifier' in line or 'test_transient_factors' in line):
                print(f"    {line}")
        return False
    else:
        print(f"  Clean ({len(files)} files)")
        return True

def binary_search(files, left=0, right=None, depth=0):
    if right is None:
        right = len(files)
    
    if left >= right:
        return None
    
    if right - left == 1:
        return files[left]
    
    mid = (left + right) // 2
    indent = "  " * depth
    
    # Test left half
    print(f"{indent}[{left}:{mid}] ({mid-left} files)")
    left_clean = run_with_collection(files[left:mid])
    
    if not left_clean:
        return binary_search(files, left, mid, depth + 1)
    else:
        # Test right half
        print(f"{indent}[{mid}:{right}] ({right-mid} files)")
        right_clean = run_with_collection(files[mid:right])
        if not right_clean:
            return binary_search(files, mid, right, depth + 1)
        else:
            return None  # No contaminator found in this range

# Separate files: exclude StatePersistence files and the 2 failing test files
search_files = []
for f in ALL_FILES:
    basename = os.path.basename(f)
    if basename not in STATEPERSISTENCE_BASENAMES and basename not in FAILING_BASENAMES:
        search_files.append(f)

print(f"Total test files: {len(ALL_FILES)}")
print(f"StatePersistence (non-failing) files: {len(STATEPERSISTENCE_BASENAMES)}")
print(f"Searching among: {len(search_files)} files")
print()

# First verify: does running with ALL test files cause contamination?
print("=== Verifying: running with ALL test files ===")
all_except_failing = [f for f in ALL_FILES if os.path.basename(f) not in FAILING_BASENAMES]
clean = run_with_collection(all_except_failing)
print()

if not clean:
    print("=== Confirmed! Contaminator is among the other test files. Starting binary search... ===")
    culprit = binary_search(search_files)
    if culprit:
        print(f"\n=== CONTAMINATOR FOUND: {culprit} ===")
        # Show which file it is
        print(f"File: {relpath(culprit)}")
    else:
        print("\n=== No single contaminator found (maybe combination effect) ===")
else:
    print("No contamination detected with ALL other files. Rethink approach...")
    print("Trying alternative: maybe the StatePersistence files themselves cause it in specific order?")
