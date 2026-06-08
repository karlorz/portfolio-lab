"""Shared Labs artifact fixture helpers."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

LABS_FIXTURE_DIR = Path(__file__).resolve().parent

LABS_FIXTURE_NAMES: tuple[str, ...] = (
    "valid_registry",
    "valid_provenance",
    "valid_scorecard",
    "valid_replay_pass",
    "valid_replay_fail",
    "validation_report",
    "invalid_missing_metrics",
    "invalid_mixed_units",
    "stale_schema",
    "dirty_provenance",
)

_BUILD_VARIANTS: dict[tuple[str, str], str] = {
    ("registry", "valid"): "valid_registry",
    ("registry", "missing_metrics"): "invalid_missing_metrics",
    ("registry", "stale_schema"): "stale_schema",
    ("provenance", "valid"): "valid_provenance",
    ("provenance", "dirty"): "dirty_provenance",
    ("scorecard", "valid"): "valid_scorecard",
    ("scorecard", "mixed_units"): "invalid_mixed_units",
    ("replay", "pass"): "valid_replay_pass",
    ("replay", "drift_fail"): "valid_replay_fail",
}


def labs_fixture_path(name: str) -> Path:
    """Return the JSON path for a named Labs fixture."""
    if name not in LABS_FIXTURE_NAMES:
        raise KeyError(f"Unknown Labs fixture: {name}")
    return LABS_FIXTURE_DIR / f"{name}.json"


def load_labs_fixture(name: str) -> dict[str, Any]:
    """Load a named Labs fixture as a mutable dictionary."""
    with open(labs_fixture_path(name)) as f:
        return json.load(f)


def iter_labs_fixture_paths() -> list[Path]:
    """Return all fixture paths in the canonical fixture order."""
    return [labs_fixture_path(name) for name in LABS_FIXTURE_NAMES]


def build_labs_fixture(kind: str, variant: str = "valid") -> dict[str, Any]:
    """Build a named Labs fixture variant without exposing file names to tests."""
    fixture_name = _BUILD_VARIANTS.get((kind, variant))
    if fixture_name is None:
        raise KeyError(f"Unknown Labs fixture variant: {kind}/{variant}")
    return copy.deepcopy(load_labs_fixture(fixture_name))
