"""Experiment artifact provenance helpers for Labs result files."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.monitor import freeze_manifest
from src.paths import BACKTEST_RESULTS_DIR, DATA_DIR, PROJECT_ROOT

EXPERIMENT_MANIFEST_SCHEMA_VERSION = "experiment-manifest/v1"

DEFAULT_ENV_KEYS: tuple[str, ...] = (
    "PORTFOLIO_LAB_ENABLE_ML",
    "LOG_LEVEL",
    "JSON_LOGS",
    "PRICE_CACHE_TTL_SECONDS",
    "COMPUTATION_CACHE_TTL_SECONDS",
    "GRADUATION_MIN_SHARPE",
    "GRADUATION_MAX_DRAWDOWN",
)

DEFAULT_EXPERIMENT_ARTIFACTS: tuple[Path, ...] = (
    DATA_DIR / "combined_regime_alloc_vol_target_results.json",
    DATA_DIR / "gold_allocation_sweep.json",
    DATA_DIR / "gold_allocation_sweep_2026.json",
    DATA_DIR / "walk_forward_report.json",
    DATA_DIR / "regime_alloc_backtest_results.json",
)

__all__ = [
    "DEFAULT_EXPERIMENT_ARTIFACTS",
    "EXPERIMENT_MANIFEST_SCHEMA_VERSION",
    "backfill_experiment_manifests",
    "build_experiment_manifest",
    "file_sha256",
    "manifest_sidecar_path",
    "save_experiment_result_json",
]


def file_sha256(path: str | Path) -> str:
    """Return a deterministic SHA-256 hex digest for a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sidecar_path(artifact_path: str | Path) -> Path:
    """Return the sidecar manifest path for an artifact."""
    path = Path(artifact_path)
    return path.with_suffix(path.suffix + ".manifest.json")


def _string_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _input_hashes(input_paths: Sequence[str | Path]) -> dict[str, str | None]:
    hashes: dict[str, str | None] = {}
    for path in input_paths:
        path_obj = Path(path)
        key = _string_path(path_obj)
        hashes[key] = file_sha256(path_obj) if path_obj.exists() else None
    return hashes


def _environment_snapshot(
    env_keys: Iterable[str],
    freeze_config: Mapping[str, Any],
) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for key in env_keys:
        value = os.environ.get(key)
        if value is None:
            value = str(freeze_config.get(key, ""))
        snapshot[key] = value
    return snapshot


def build_experiment_manifest(
    *,
    experiment_id: str,
    source_artifact_path: str | Path,
    command: str | None = None,
    module: str | None = None,
    config_snapshot: Mapping[str, Any] | None = None,
    env_keys: Iterable[str] = DEFAULT_ENV_KEYS,
    input_paths: Sequence[str | Path] = (),
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Build a canonical provenance manifest for a Labs experiment artifact."""
    if project_root is None:
        project_root = PROJECT_ROOT

    freeze = freeze_manifest.create_manifest(project_root=project_root)
    return {
        "schema_version": EXPERIMENT_MANIFEST_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_artifact_path": _string_path(source_artifact_path),
        "command": command,
        "module": module,
        "git": freeze.get("git", {}),
        "config_snapshot": dict(config_snapshot or {}),
        "environment": _environment_snapshot(env_keys, freeze.get("config", {})),
        "input_file_hashes": _input_hashes(input_paths),
        "freeze_manifest": {
            "timestamp": freeze.get("timestamp"),
            "config": freeze.get("config", {}),
            "file_hashes": freeze.get("file_hashes", {}),
            "file_count": freeze.get("file_count", 0),
        },
    }


def _json_serializer(obj: Any) -> Any:
    """Reuse the backtest JSON serializer without importing it at module load."""
    from src.backtest.metrics import _json_serializer as backtest_json_serializer

    return backtest_json_serializer(obj)


def save_experiment_result_json(
    data: Mapping[str, Any],
    output_path: str | Path,
    *,
    experiment_id: str,
    manifest_mode: str = "embedded",
    command: str | None = None,
    module: str | None = None,
    config_snapshot: Mapping[str, Any] | None = None,
    env_keys: Iterable[str] = DEFAULT_ENV_KEYS,
    input_paths: Sequence[str | Path] = (),
    project_root: Path | None = None,
) -> Path:
    """Save an experiment result with embedded or sidecar provenance.

    Returns the artifact path for embedded mode and the sidecar manifest path
    for sidecar mode.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_experiment_manifest(
        experiment_id=experiment_id,
        source_artifact_path=path,
        command=command,
        module=module,
        config_snapshot=config_snapshot,
        env_keys=env_keys,
        input_paths=input_paths,
        project_root=project_root,
    )

    if manifest_mode == "embedded":
        payload = dict(data)
        payload["_provenance"] = manifest
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=_json_serializer)
        return path

    if manifest_mode == "sidecar":
        with open(path, "w") as f:
            json.dump(dict(data), f, indent=2, default=_json_serializer)
        sidecar_path = manifest_sidecar_path(path)
        with open(sidecar_path, "w") as f:
            json.dump(manifest, f, indent=2, default=_json_serializer)
        return sidecar_path

    raise ValueError("manifest_mode must be 'embedded' or 'sidecar'")


def backfill_experiment_manifests(
    artifact_paths: Sequence[str | Path] | None = None,
    *,
    experiment_id_prefix: str = "portfolio-lab",
    command: str | None = None,
) -> list[Path]:
    """Write sidecar manifests for existing experiment artifacts."""
    if artifact_paths is None:
        paths = list(DEFAULT_EXPERIMENT_ARTIFACTS)
        paths.extend(sorted(BACKTEST_RESULTS_DIR.glob("*.json")))
    else:
        paths = [Path(path) for path in artifact_paths]

    written: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        experiment_id = f"{experiment_id_prefix}:{path.stem}"
        manifest = build_experiment_manifest(
            experiment_id=experiment_id,
            source_artifact_path=path,
            command=command,
            module=__name__,
            input_paths=[path],
        )
        sidecar_path = manifest_sidecar_path(path)
        with open(sidecar_path, "w") as f:
            json.dump(manifest, f, indent=2, default=_json_serializer)
        written.append(sidecar_path)
    return written


def main() -> int:
    """CLI entry point for sidecar provenance backfill."""
    if len(sys.argv) > 1 and sys.argv[1] != "backfill":
        print("Usage: python -m src.research.experiment_manifest [backfill]")
        return 2
    written = backfill_experiment_manifests(command="python -m src.research.experiment_manifest backfill")
    print(json.dumps({"written": [str(path) for path in written]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
