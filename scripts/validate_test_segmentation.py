#!/usr/bin/env python3
"""
Test Segmentation Validation Script

Monitors test execution times and identifies slow tests for optimization.
Provides metrics for test-fast target validation and test suite health monitoring.

Usage:
    python scripts/validate_test_segmentation.py
    python scripts/validate_test_segmentation.py --verbose
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(
    os.environ.get("PORTFOLIO_LAB_PROJECT_DIR", Path(__file__).resolve().parents[1])
)


def run_pytest_with_timing(test_path: str = "tests/", timeout: int = 600) -> Dict:
    """Run pytest and capture timing information."""
    print(f"Running pytest on {test_path} with timeout {timeout}s...")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [
                "uv", "run", "pytest", test_path,
                "--tb=short",
                "-q",
                "--no-header",
                "-p", "no:cacheprovider",
                "--durations=50",  # Show 50 slowest tests
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "PORTFOLIO_LAB_ENABLE_ML": "0"}
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Parse output for timing information
        output = result.stdout + result.stderr
        lines = output.split("\n")
        
        # Extract slowest tests
        slow_tests = []
        for line in lines:
            if "::" in line and ("PASSED" in line or "FAILED" in line or "ERROR" in line):
                # Parse test timing from pytest output
                parts = line.strip().split()
                if len(parts) >= 2:
                    test_name = parts[0]
                    # Look for timing in format [xx.xxs]
                    for part in parts:
                        if part.startswith("[") and part.endswith("s]"):
                            try:
                                time_str = part[1:-2]
                                test_time = float(time_str)
                                slow_tests.append((test_name, test_time))
                            except ValueError:
                                pass
        
        # Sort by time descending
        slow_tests.sort(key=lambda x: x[1], reverse=True)
        
        # Parse test counts
        passed = 0
        failed = 0
        skipped = 0
        errors = 0
        
        for line in lines:
            if "passed" in line:
                try:
                    passed = int(line.split("passed")[0].strip().split()[-1])
                except (ValueError, IndexError):
                    pass
            if "failed" in line:
                try:
                    failed = int(line.split("failed")[0].strip().split()[-1])
                except (ValueError, IndexError):
                    pass
            if "skipped" in line:
                try:
                    skipped = int(line.split("skipped")[0].strip().split()[-1])
                except (ValueError, IndexError):
                    pass
            if "error" in line:
                try:
                    errors = int(line.split("error")[0].strip().split()[-1])
                except (ValueError, IndexError):
                    pass
        
        return {
            "success": result.returncode == 0,
            "duration": duration,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "errors": errors,
            "slow_tests": slow_tests[:10],  # Top 10 slowest
            "output": output,
            "return_code": result.returncode
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "duration": timeout,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 1,
            "slow_tests": [],
            "output": f"Test suite timed out after {timeout} seconds",
            "return_code": -1
        }
    except Exception as e:
        return {
            "success": False,
            "duration": time.time() - start_time,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 1,
            "slow_tests": [],
            "output": f"Error running tests: {str(e)}",
            "return_code": -1
        }


def validate_test_fast_target() -> Dict:
    """Validate that make test-fast completes within target time."""
    print("Validating make test-fast target...")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ["make", "test-fast"],
            capture_output=True,
            text=True,
            timeout=180,  # 3 minutes max for test-fast
            cwd=str(PROJECT_ROOT)
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "success": result.returncode == 0,
            "duration": duration,
            "target_met": duration < 120,  # 2 minutes target
            "output": result.stdout + result.stderr,
            "return_code": result.returncode
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "duration": 180,
            "target_met": False,
            "output": "test-fast timed out after 180 seconds",
            "return_code": -1
        }
    except Exception as e:
        return {
            "success": False,
            "duration": time.time() - start_time,
            "target_met": False,
            "output": f"Error running test-fast: {str(e)}",
            "return_code": -1
        }


def analyze_test_segmentation() -> Dict:
    """Analyze test suite for segmentation opportunities."""
    print("Analyzing test suite for segmentation...")
    
    # Find test files and estimate sizes
    test_files = []
    tests_dir = PROJECT_ROOT / "tests"
    
    if tests_dir.exists():
        for py_file in tests_dir.rglob("test_*.py"):
            try:
                # Count test functions/methods
                with open(py_file, "r") as f:
                    content = f.read()
                
                test_count = content.count("def test_") + content.count("async def test_")
                file_size = py_file.stat().st_size
                
                test_files.append({
                    "path": str(py_file.relative_to(tests_dir)),
                    "test_count": test_count,
                    "file_size_kb": file_size / 1024,
                    "estimated_time": test_count * 0.1  # Rough estimate: 0.1s per test
                })
            except Exception:
                pass
    
    # Sort by estimated time
    test_files.sort(key=lambda x: x["estimated_time"], reverse=True)
    
    # Identify fast tests (< 2s estimated)
    fast_tests = [f for f in test_files if f["estimated_time"] < 2.0]
    
    # Identify slow tests (> 10s estimated)
    slow_tests = [f for f in test_files if f["estimated_time"] > 10.0]
    
    return {
        "total_test_files": len(test_files),
        "total_estimated_tests": sum(f["test_count"] for f in test_files),
        "fast_test_files": len(fast_tests),
        "slow_test_files": len(slow_tests),
        "fast_tests": fast_tests[:10],  # Top 10 fastest
        "slow_tests": slow_tests[:10],  # Top 10 slowest
        "segmentation_recommendation": {
            "fast_target": [f["path"] for f in fast_tests[:20]],
            "medium_target": [f["path"] for f in test_files[20:50]],
            "full_suite": [f["path"] for f in test_files]
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Validate test segmentation and performance")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--full-suite", action="store_true", help="Run full test suite (slow)")
    parser.add_argument("--save-results", action="store_true", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Test Segmentation Validation")
    print("=" * 60)
    
    results = {}
    
    # 1. Analyze test files
    print("\n1. Analyzing test files...")
    analysis = analyze_test_segmentation()
    results["analysis"] = analysis
    
    print(f"   Total test files: {analysis['total_test_files']}")
    print(f"   Estimated total tests: {analysis['total_estimated_tests']}")
    print(f"   Fast test files (<2s): {analysis['fast_test_files']}")
    print(f"   Slow test files (>10s): {analysis['slow_test_files']}")
    
    if args.verbose:
        print("\n   Top 10 slowest test files:")
        for test in analysis["slow_tests"][:10]:
            print(f"     {test['path']}: ~{test['estimated_time']:.1f}s ({test['test_count']} tests)")
    
    # 2. Validate test-fast target
    print("\n2. Validating make test-fast target...")
    fast_result = validate_test_fast_target()
    results["test_fast"] = fast_result
    
    status = "✅ PASS" if fast_result["target_met"] else "❌ FAIL"
    print(f"   {status}: test-fast completed in {fast_result['duration']:.1f}s (target: <120s)")
    
    if fast_result["success"]:
        print(f"   Tests passed: {fast_result.get('passed', 'N/A')}")
    else:
        print(f"   Error: {fast_result['output'][:200]}...")
    
    # 3. Run full suite if requested (slow)
    if args.full_suite:
        print("\n3. Running full test suite...")
        full_result = run_pytest_with_timing(timeout=600)
        results["full_suite"] = full_result
        
        print(f"   Duration: {full_result['duration']:.1f}s")
        print(f"   Passed: {full_result['passed']}")
        print(f"   Failed: {full_result['failed']}")
        print(f"   Skipped: {full_result['skipped']}")
        
        if full_result["slow_tests"]:
            print("\n   Top 5 slowest tests:")
            for test_name, test_time in full_result["slow_tests"][:5]:
                print(f"     {test_name}: {test_time:.2f}s")
    
    # 4. Generate recommendations
    print("\n4. Recommendations:")
    
    if fast_result["target_met"]:
        print("   ✅ test-fast target is met (<120s)")
        print("   → Use 'make test-fast' for quick development feedback")
    else:
        print("   ❌ test-fast target not met")
        print("   → Consider optimizing slow tests or adjusting target")
    
    if analysis["slow_test_files"] > 5:
        print(f"   ⚠️  {analysis['slow_test_files']} slow test files detected")
        print("   → Consider test segmentation or optimization")
    
    if analysis["fast_test_files"] > 20:
        print(f"   ✅ {analysis['fast_test_files']} fast test files available for test-fast target")
        print("   → Good candidates for quick feedback suite")
    
    # Save results if requested
    if args.save_results:
        output_file = PROJECT_ROOT / "data" / "test_segmentation_results.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    print("\n" + "=" * 60)
    print("Validation complete")
    print("=" * 60)
    
    return 0 if fast_result["target_met"] else 1


if __name__ == "__main__":
    sys.exit(main())
