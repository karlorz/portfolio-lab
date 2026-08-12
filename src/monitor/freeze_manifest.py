"""
Freeze manifest for config drift detection.

Snapshots config, git state, feature flags, and file hashes per session.
Diff against a clean baseline to detect config drift.

Usage:
    from src.monitor.freeze_manifest import create_manifest, diff_manifests

    manifest = create_manifest()
    # Store as "clean baseline" on first deployment
    # On subsequent runs, diff against baseline to detect drift

Environment variables
---------------------
FREEZE_MANIFEST_PATH : str
    Output path for freeze manifest JSON (default: DATA_DIR/freeze_manifest.json)
"""

import hashlib
import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any

from src.paths import DATA_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)

__all__ = ["create_manifest", "diff_manifests", "load_manifest", "save_manifest"]

FREEZE_PATH = Path(os.environ.get("FREEZE_MANIFEST_PATH", str(DATA_DIR / "freeze_manifest.json")))

# Files/directories to include in hash computation
_HASH_GLOBS = [
    "pyproject.toml",
    "tsconfig.json",
    "Makefile",
    "crontab",
    "src/**/*.py",
    "src/**/*.ts",
    "src/**/*.tsx",
    "data/ensemble_weights.json",
]

# Directories to skip
_SKIP_DIRS = {
    "__pycache__", ".git", "node_modules", ".venv", "dist", ".next",
    ".claude", "wiki", "research",
}


def _git_state(project_root: Path) -> Dict[str, Any]:
    """Capture current git state."""
    result: Dict[str, Any] = {
        "commit": None,
        "branch": None,
        "dirty": None,
        "tag": None,
    }

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=project_root, timeout=10,
        )
        if commit.returncode == 0:
            result["commit"] = commit.stdout.strip()[:12]

        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=project_root, timeout=10,
        )
        if branch.returncode == 0:
            result["branch"] = branch.stdout.strip()

        dirty = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True, text=True, cwd=project_root, timeout=10,
        )
        result["dirty"] = dirty.returncode != 0

        tag = subprocess.run(
            ["git", "describe", "--tags", "--always"],
            capture_output=True, text=True, cwd=project_root, timeout=10,
        )
        if tag.returncode == 0:
            result["tag"] = tag.stdout.strip()

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.warning("Git state capture failed: %s", e)

    return result


def _config_state() -> Dict[str, Any]:
    """Capture current config from env vars that affect behavior."""
    config_keys = [
        "ALPHALAB_MODE",
        "CRON_BACKEND",
        "JSON_LOGS",
        "LOG_LEVEL",
        "PAPER_INITIAL_CAPITAL",
        "PAPER_MAX_POSITION_PCT",
        "PAPER_MAX_DRAWDOWN_PCT",
        "PAPER_VOLATILITY_TARGET",
        "PRICE_CACHE_TTL_SECONDS",
        "COMPUTATION_CACHE_TTL_SECONDS",
        "SIGNAL_STALENESS_TTL_HOURS",
        "KILL_WARNING_DRAWDOWN_PCT",
        "KILL_RESTRICT_DRAWDOWN_PCT",
        "KILL_HALT_DRAWDOWN_PCT",
        "KILL_LIQUIDATE_DRAWDOWN_PCT",
        "GRADUATION_MIN_SHARPE",
        "GRADUATION_MAX_DRAWDOWN",
        "ENSEMBLE_WEIGHTS_FILE",
        "BROKER_CIRCUIT_FAIL_MAX",
        "BROKER_CIRCUIT_RESET_TIMEOUT",
    ]
    return {k: os.environ.get(k, "") for k in config_keys}


def _file_hashes(project_root: Path) -> Dict[str, str]:
    """Compute SHA256 hashes of key source files."""
    hashes: Dict[str, str] = {}

    for glob_pattern in _HASH_GLOBS:
        for path in project_root.glob(glob_pattern):
            # Skip files in excluded directories
            if any(skip in path.parts for skip in _SKIP_DIRS):
                continue
            try:
                content = path.read_bytes()
                sha = hashlib.sha256(content).hexdigest()[:16]
                rel = str(path.relative_to(project_root))
                hashes[rel] = sha
            except OSError:
                continue

    return hashes


def create_manifest(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """Create a freeze manifest snapshotting current system state.

    Args:
        project_root: Root directory (default: auto-detect from src/paths)

    Returns:
        Dict with git_state, config_state, file_hashes, timestamp.
    """
    if project_root is None:
        project_root = PROJECT_ROOT

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": _git_state(project_root),
        "config": _config_state(),
        "file_hashes": _file_hashes(project_root),
        "file_count": 0,  # filled below
    }
    manifest["file_count"] = len(manifest["file_hashes"])

    return manifest


def save_manifest(manifest: Dict[str, Any], path: Optional[Path] = None) -> Path:
    """Save manifest to disk."""
    if path is None:
        path = FREEZE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    logger.info("Freeze manifest saved: %s (%d files)", path, manifest.get("file_count", 0))
    return path


def load_manifest(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load a previously saved manifest."""
    if path is None:
        path = FREEZE_PATH
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load manifest: %s", e)
        return None


def diff_manifests(
    baseline: Dict[str, Any],
    current: Dict[str, Any],
) -> Dict[str, Any]:
    """Diff two manifests to detect config drift.

    Returns a dict with:
    - drifted: bool (True if any drift detected)
    - git_changed: bool
    - config_drift: dict of changed env vars
    - file_changes: dict of added/removed/modified files
    """
    result: Dict[str, Any] = {
        "drifted": False,
        "git_changed": False,
        "config_drift": {},
        "file_changes": {
            "added": [],
            "removed": [],
            "modified": [],
        },
    }

    # Git state drift
    baseline_git = baseline.get("git", {})
    current_git = current.get("git", {})
    if baseline_git.get("commit") != current_git.get("commit"):
        result["git_changed"] = True
        result["drifted"] = True
    if current_git.get("dirty") and not baseline_git.get("dirty"):
        result["git_changed"] = True
        result["drifted"] = True

    # Config drift
    baseline_config = baseline.get("config", {})
    current_config = current.get("config", {})
    for key in set(list(baseline_config.keys()) + list(current_config.keys())):
        old_val = baseline_config.get(key, "")
        new_val = current_config.get(key, "")
        if old_val != new_val:
            result["config_drift"][key] = {"from": old_val, "to": new_val}
            result["drifted"] = True

    # File hash drift
    baseline_hashes = baseline.get("file_hashes", {})
    current_hashes = current.get("file_hashes", {})
    baseline_files = set(baseline_hashes.keys())
    current_files = set(current_hashes.keys())

    added = sorted(current_files - baseline_files)
    removed = sorted(baseline_files - current_files)
    modified = sorted(
        f for f in baseline_files & current_files
        if baseline_hashes[f] != current_hashes[f]
    )

    if added or removed or modified:
        result["file_changes"]["added"] = added
        result["file_changes"]["removed"] = removed
        result["file_changes"]["modified"] = modified
        result["drifted"] = True

    return result
