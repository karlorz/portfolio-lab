"""Offline Labs experiment registry generation.

The registry scanner reads existing experiment artifacts only. It never fetches
market data or reruns backtests, which keeps dashboard generation safe for cron.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.paths import BACKTEST_RESULTS_DIR, DATA_DIR, PROJECT_ROOT, PUBLIC_DATA_DIR
from src.research.experiment_artifact_validator import (
    LABS_REGISTRY_SCHEMA_VERSION,
    validate_artifact,
)
from src.research.experiment_manifest import EXPERIMENT_MANIFEST_SCHEMA_VERSION, manifest_sidecar_path
from src.research.promotion_policy import governance_disclosure_fields

logger = logging.getLogger(__name__)

LABS_REGISTRY_FILENAME = "labs_registry.json"
OPTIMIZER_LABS_SCHEMA_VERSION = "optimizer-labs-output/v1"

_ROOT_PATTERNS: tuple[str, ...] = (
    "optimized_weights.json",
    "optimizer_labs_output.json",
    "*_results.json",
    "*_sweep*.json",
)
_NESTED_DIRS: tuple[str, ...] = ("backtest_results",)
_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "sharpe": ("sharpe", "sharpe_ratio", "static_sharpe", "mean_oos_sharpe"),
    "cagr_pct": ("cagr_pct", "cagr", "static_cagr"),
    "volatility_pct": ("volatility_pct", "volatility", "vol", "static_vol"),
    "max_drawdown_pct": ("max_drawdown_pct", "max_drawdown", "max_dd", "static_max_dd"),
    "wfe": ("wfe",),
    "dsr": ("dsr",),
    "positive_oos_ratio": ("positive_oos_ratio",),
    "regime_coverage": ("regime_coverage",),
}
_BASELINE_ALIASES: dict[str, tuple[str, ...]] = {
    "sharpe": ("baseline_sharpe",),
    "cagr_pct": ("baseline_cagr",),
    "volatility_pct": ("baseline_vol", "baseline_volatility"),
    "max_drawdown_pct": ("baseline_max_dd", "baseline_max_drawdown"),
}
_DELTA_ALIASES: dict[str, tuple[str, ...]] = {
    "sharpe": ("sharpe_improvement", "combined_sharpe_delta"),
    "cagr_pct": ("cagr_improvement",),
    "max_drawdown_pct": ("max_drawdown_improvement",),
}

__all__ = [
    "LABS_REGISTRY_FILENAME",
    "build_labs_registry",
    "discover_registry_candidates",
    "save_labs_registry",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _is_candidate_path(path: Path) -> bool:
    name = path.name
    if name in {LABS_REGISTRY_FILENAME, "index.json"}:
        return False
    if name.endswith(".manifest.json"):
        return False
    return path.suffix == ".json"


def discover_registry_candidates(
    data_dirs: Sequence[str | Path] | None = None,
) -> list[Path]:
    """Discover known offline experiment result artifacts."""
    if data_dirs is None:
        roots = [DATA_DIR, BACKTEST_RESULTS_DIR]
    else:
        roots = [Path(root) for root in data_dirs]

    candidates: set[Path] = set()
    for root in roots:
        if root.is_file():
            if _is_candidate_path(root):
                candidates.add(root)
            continue
        if not root.exists():
            continue

        for pattern in _ROOT_PATTERNS:
            candidates.update(path for path in root.glob(pattern) if _is_candidate_path(path))

        if root.name in _NESTED_DIRS:
            candidates.update(path for path in root.glob("*.json") if _is_candidate_path(path))
        else:
            for dirname in _NESTED_DIRS:
                nested = root / dirname
                if nested.exists():
                    candidates.update(path for path in nested.glob("*.json") if _is_candidate_path(path))

    return sorted(candidates)


def _load_json(path: Path) -> tuple[Mapping[str, Any] | None, str | None]:
    try:
        with open(path) as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg}"
    except OSError as exc:
        return None, f"could not read file: {exc}"
    if not isinstance(payload, Mapping):
        return None, "expected object"
    return payload, None


def _numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def _first_numeric(payload: Mapping[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        value = _numeric(payload.get(key))
        if value is not None:
            return value
    return None


def _metric_source(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    best = payload.get("best_sharpe_row")
    if isinstance(best, Mapping):
        return best
    return payload


def _extract_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    source = _metric_source(payload)
    metrics: dict[str, float] = {}
    for target_key, source_keys in _METRIC_ALIASES.items():
        value = _first_numeric(source, source_keys)
        if value is not None:
            metrics[target_key] = value
    return metrics


def _extract_baseline_deltas(payload: Mapping[str, Any], metrics: Mapping[str, float]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    source = _metric_source(payload)
    for metric_key, delta_keys in _DELTA_ALIASES.items():
        value = _first_numeric(source, delta_keys)
        if value is not None:
            deltas[metric_key] = value

    for metric_key, baseline_keys in _BASELINE_ALIASES.items():
        if metric_key in deltas or metric_key not in metrics:
            continue
        baseline = _first_numeric(source, baseline_keys)
        if baseline is not None:
            deltas[metric_key] = round(float(metrics[metric_key]) - baseline, 10)
    return deltas


def _valid_provenance_status(provenance: Any) -> str | None:
    if not isinstance(provenance, Mapping):
        return None
    result = validate_artifact(provenance)
    if result.valid and result.schema_version == EXPERIMENT_MANIFEST_SCHEMA_VERSION:
        return "embedded"
    return "malformed"


def _sidecar_provenance(path: Path) -> tuple[Mapping[str, Any] | None, str | None]:
    sidecar = manifest_sidecar_path(path)
    if not sidecar.exists():
        return None, None
    payload, error = _load_json(sidecar)
    if error is not None:
        return None, "malformed"
    result = validate_artifact(payload)
    if result.valid and result.schema_version == EXPERIMENT_MANIFEST_SCHEMA_VERSION:
        return payload, "sidecar"
    return payload, "malformed"


def _experiment_id(path: Path, payload: Mapping[str, Any], provenance: Mapping[str, Any] | None) -> str:
    if provenance is not None and isinstance(provenance.get("experiment_id"), str):
        return str(provenance["experiment_id"])
    if isinstance(payload.get("experiment_id"), str) and str(payload["experiment_id"]).strip():
        return str(payload["experiment_id"])
    return f"artifact:{path.stem}"


def _warning_row(path: Path, project_root: Path, error: str) -> dict[str, Any]:
    return {
        "experiment_id": f"artifact:{path.stem}",
        "artifact_path": _relative_path(path, project_root),
        "status": "warning",
        "provenance_status": "malformed",
        "metrics": {},
        "baseline_deltas": {},
        "_warning": error,
    }


def _strip_internal_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _validate_row(row: Mapping[str, Any], generated_at: str) -> bool:
    registry = {
        "schema_version": LABS_REGISTRY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "experiments": [_strip_internal_fields(row)],
    }
    return validate_artifact(registry).valid


def _rows_from_registry_payload(
    payload: Mapping[str, Any],
    *,
    generated_at: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    result = validate_artifact(payload)
    if not result.valid:
        return [], result.error_messages()
    rows = [_strip_internal_fields(row) for row in payload.get("experiments", []) if isinstance(row, Mapping)]
    return rows, []


def _rows_from_plain_payload(
    path: Path,
    payload: Mapping[str, Any],
    *,
    project_root: Path,
    generated_at: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    provenance = payload.get("_provenance")
    provenance_status = _valid_provenance_status(provenance)
    provenance_payload = provenance if isinstance(provenance, Mapping) else None

    if provenance_status is None:
        sidecar_payload, sidecar_status = _sidecar_provenance(path)
        provenance_payload = sidecar_payload
        provenance_status = sidecar_status or "missing"

    metrics = _extract_metrics(payload)
    row = {
        "experiment_id": _experiment_id(path, payload, provenance_payload),
        "artifact_path": _relative_path(path, project_root),
        "status": "candidate" if metrics else "warning",
        "provenance_status": provenance_status,
        "metrics": metrics,
        "baseline_deltas": _extract_baseline_deltas(payload, metrics),
    }
    row.update(
        governance_disclosure_fields(
            row,
            metric_gate_status="candidate" if metrics else "rejected",
            metric_gate_pass=bool(metrics),
            metric_failures=[] if metrics else ["missing_metrics"],
        )
    )
    if not _validate_row(row, generated_at):
        return [], [f"{row['artifact_path']}: generated registry row failed validation"]
    return [row], []


def _rows_for_candidate(
    path: Path,
    *,
    project_root: Path,
    generated_at: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    payload, error = _load_json(path)
    if error is not None:
        row = _warning_row(path, project_root, error)
        return ([_strip_internal_fields(row)] if _validate_row(row, generated_at) else []), [
            f"{_relative_path(path, project_root)}: {error}"
        ]

    if payload.get("schema_version") == LABS_REGISTRY_SCHEMA_VERSION:
        rows, errors = _rows_from_registry_payload(payload, generated_at=generated_at)
    elif payload.get("schema_version") == OPTIMIZER_LABS_SCHEMA_VERSION and isinstance(payload.get("registry"), Mapping):
        rows, errors = _rows_from_registry_payload(payload["registry"], generated_at=generated_at)
    else:
        rows, errors = _rows_from_plain_payload(path, payload, project_root=project_root, generated_at=generated_at)

    return rows, [f"{_relative_path(path, project_root)}: {error}" for error in errors]


def build_labs_registry(
    *,
    data_dirs: Sequence[str | Path] | None = None,
    project_root: str | Path = PROJECT_ROOT,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a Labs registry from existing local result artifacts."""
    project_root = Path(project_root)
    generated_at = generated_at or _now_iso()
    candidates = discover_registry_candidates(data_dirs)

    rows: list[dict[str, Any]] = []
    warnings: list[dict[str, str]] = []
    sources: list[str] = []
    seen_ids: set[str] = set()

    for path in candidates:
        candidate_rows, candidate_warnings = _rows_for_candidate(
            path,
            project_root=project_root,
            generated_at=generated_at,
        )
        if candidate_rows or candidate_warnings:
            sources.append(_relative_path(path, project_root))
        for warning in candidate_warnings:
            artifact_path, _, error = warning.partition(": ")
            warnings.append({"artifact_path": artifact_path, "error": error})
        for row in candidate_rows:
            experiment_id = str(row["experiment_id"])
            if experiment_id in seen_ids:
                warnings.append(
                    {
                        "artifact_path": _relative_path(path, project_root),
                        "error": f"duplicate experiment_id skipped: {experiment_id}",
                    }
                )
                continue
            seen_ids.add(experiment_id)
            rows.append(row)

    registry = {
        "schema_version": LABS_REGISTRY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "experiments": rows,
        "sources": sources,
        "warnings": warnings,
    }
    validation = validate_artifact(registry)
    if not validation.valid:
        raise ValueError(f"generated Labs registry failed validation: {validation.error_messages()}")
    return registry


def save_labs_registry(
    *,
    data_dirs: Sequence[str | Path] | None = None,
    public_dir: str | Path = PUBLIC_DATA_DIR,
    project_root: str | Path = PROJECT_ROOT,
    generated_at: str | None = None,
    write_empty: bool = False,
) -> Path | None:
    """Write `labs_registry.json` to the public data directory when rows exist."""
    registry = build_labs_registry(data_dirs=data_dirs, project_root=project_root, generated_at=generated_at)
    if not registry["experiments"] and not write_empty:
        return None

    output_path = Path(public_dir) / LABS_REGISTRY_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(registry, f, indent=2, sort_keys=True)
    logger.info("Labs registry written: %s (%d rows)", output_path, len(registry["experiments"]))
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the offline Labs experiment registry")
    parser.add_argument("--data-dir", action="append", dest="data_dirs", help="Directory or file to scan")
    parser.add_argument("--public-dir", default=str(PUBLIC_DATA_DIR), help="Public data output directory")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="Project root for relative artifact paths")
    parser.add_argument("--write-empty", action="store_true", help="Write an empty registry when no rows are found")
    args = parser.parse_args(argv)

    output_path = save_labs_registry(
        data_dirs=args.data_dirs,
        public_dir=args.public_dir,
        project_root=args.project_root,
        write_empty=args.write_empty,
    )
    sys.stdout.write(json.dumps({"output_path": str(output_path) if output_path else None}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
