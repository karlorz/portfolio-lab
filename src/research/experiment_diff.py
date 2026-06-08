"""Compare Labs experiment artifacts or registry rows."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

EXPERIMENT_DIFF_SCHEMA_VERSION = "experiment-diff/v1"

TRACKED_METRICS: tuple[str, ...] = (
    "sharpe",
    "cagr_pct",
    "volatility_pct",
    "volatility",
    "max_drawdown_pct",
    "max_drawdown",
    "wfe",
    "dsr",
    "positive_oos_ratio",
    "regime_coverage",
)

CONFIG_FIELDS: tuple[str, ...] = (
    "config_snapshot",
    "config",
    "parameters",
    "params",
)

__all__ = [
    "EXPERIMENT_DIFF_SCHEMA_VERSION",
    "diff_experiment_artifacts",
    "diff_experiment_files",
]


def _load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _unwrap_artifact(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("experiment artifact must be a JSON object")
    experiments = payload.get("experiments")
    if isinstance(experiments, list) and len(experiments) == 1 and isinstance(experiments[0], Mapping):
        return experiments[0]
    return payload


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _extract_metrics(artifact: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for field in ("metrics", "summary", "results"):
        value = artifact.get(field)
        if isinstance(value, Mapping):
            for key, metric_value in value.items():
                if _is_number(metric_value):
                    metrics[key] = float(metric_value)

    for key in TRACKED_METRICS:
        value = artifact.get(key)
        if _is_number(value):
            metrics[key] = float(value)
    return metrics


def _metric_order(metric_names: set[str]) -> list[str]:
    ordered = [metric for metric in TRACKED_METRICS if metric in metric_names]
    ordered.extend(sorted(metric_names.difference(TRACKED_METRICS)))
    return ordered


def _extract_config(artifact: Mapping[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for field in CONFIG_FIELDS:
        value = artifact.get(field)
        if isinstance(value, Mapping):
            config.update(value)
    return config


def _artifact_type(artifact: Mapping[str, Any]) -> str:
    schema_version = artifact.get("schema_version")
    if schema_version == "experiment-manifest/v1":
        return "provenance_manifest"
    if schema_version == "labs-registry/v1":
        return "registry"
    if schema_version == "labs-scorecard/v1":
        return "scorecard"
    if schema_version == "labs-replay/v1":
        return "replay"
    if artifact.get("artifact_path") and isinstance(artifact.get("metrics"), Mapping):
        return "registry_row"
    return "experiment_artifact"


def _provenance_status(artifact: Mapping[str, Any]) -> str:
    status = artifact.get("provenance_status")
    if isinstance(status, str) and status:
        return status
    if isinstance(artifact.get("_provenance"), Mapping):
        return "embedded"
    if artifact.get("schema_version") == "experiment-manifest/v1":
        return "manifest"
    return "unknown"


def _artifact_summary(artifact: Mapping[str, Any], label: str) -> dict[str, Any]:
    experiment_id = artifact.get("experiment_id")
    artifact_path = artifact.get("artifact_path") or artifact.get("source_artifact_path")
    return {
        "label": label,
        "experiment_id": experiment_id if isinstance(experiment_id, str) else None,
        "artifact_path": artifact_path if isinstance(artifact_path, str) else None,
        "artifact_type": _artifact_type(artifact),
    }


def _metric_deltas(left_metrics: Mapping[str, float], right_metrics: Mapping[str, float]) -> dict[str, dict[str, float]]:
    deltas: dict[str, dict[str, float]] = {}
    for metric in _metric_order(set(left_metrics).intersection(right_metrics)):
        left_value = left_metrics[metric]
        right_value = right_metrics[metric]
        deltas[metric] = {
            "left": left_value,
            "right": right_value,
            "delta": right_value - left_value,
        }
    return deltas


def _missing_metrics(left_metrics: Mapping[str, float], right_metrics: Mapping[str, float]) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for metric in _metric_order(set(left_metrics).symmetric_difference(right_metrics)):
        missing_from: list[str] = []
        if metric not in left_metrics:
            missing_from.append("left")
        if metric not in right_metrics:
            missing_from.append("right")
        missing.append({"metric": metric, "missing_from": missing_from})
    return missing


def _config_diffs(left_config: Mapping[str, Any], right_config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    diffs: dict[str, dict[str, Any]] = {}
    for key in sorted(set(left_config).union(right_config)):
        left_value = left_config.get(key)
        right_value = right_config.get(key)
        if left_value != right_value:
            diffs[key] = {"left": left_value, "right": right_value}
    return diffs


def diff_experiment_artifacts(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    left_label: str = "left",
    right_label: str = "right",
) -> dict[str, Any]:
    """Return a JSON-serializable diff between two experiment payloads."""
    left = _unwrap_artifact(left)
    right = _unwrap_artifact(right)
    left_metrics = _extract_metrics(left)
    right_metrics = _extract_metrics(right)
    left_provenance = _provenance_status(left)
    right_provenance = _provenance_status(right)

    return {
        "schema_version": EXPERIMENT_DIFF_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "left": _artifact_summary(left, left_label),
        "right": _artifact_summary(right, right_label),
        "metric_deltas": _metric_deltas(left_metrics, right_metrics),
        "missing_metrics": _missing_metrics(left_metrics, right_metrics),
        "config_diffs": _config_diffs(_extract_config(left), _extract_config(right)),
        "provenance": {
            "left": left_provenance,
            "right": right_provenance,
            "changed": left_provenance != right_provenance,
        },
    }


def diff_experiment_files(left_path: str | Path, right_path: str | Path) -> dict[str, Any]:
    """Load and compare two JSON artifact files."""
    left = _unwrap_artifact(_load_json(left_path))
    right = _unwrap_artifact(_load_json(right_path))
    return diff_experiment_artifacts(
        left,
        right,
        left_label=Path(left_path).name,
        right_label=Path(right_path).name,
    )


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def format_experiment_diff(diff: Mapping[str, Any]) -> str:
    """Render a compact human-readable diff."""
    left_id = diff.get("left", {}).get("experiment_id") or diff.get("left", {}).get("label")
    right_id = diff.get("right", {}).get("experiment_id") or diff.get("right", {}).get("label")
    lines = [f"Experiment Diff: {left_id} -> {right_id}"]

    metric_deltas = diff.get("metric_deltas", {})
    if isinstance(metric_deltas, Mapping) and metric_deltas:
        lines.append("Metric deltas:")
        for metric, payload in metric_deltas.items():
            if not isinstance(payload, Mapping):
                continue
            lines.append(
                "  "
                f"{metric}: {_format_value(payload.get('left'))} -> {_format_value(payload.get('right'))} "
                f"(delta {_format_value(payload.get('delta'))})"
            )

    missing_metrics = diff.get("missing_metrics", [])
    if isinstance(missing_metrics, list) and missing_metrics:
        lines.append("Missing metrics:")
        for item in missing_metrics:
            if isinstance(item, Mapping):
                lines.append(f"  {item.get('metric')}: missing from {', '.join(item.get('missing_from', []))}")

    config_diffs = diff.get("config_diffs", {})
    if isinstance(config_diffs, Mapping) and config_diffs:
        lines.append("Config changes:")
        for key, payload in config_diffs.items():
            if isinstance(payload, Mapping):
                lines.append(f"  {key}: {_format_value(payload.get('left'))} -> {_format_value(payload.get('right'))}")

    provenance = diff.get("provenance", {})
    if isinstance(provenance, Mapping):
        lines.append(
            "Provenance: "
            f"{_format_value(provenance.get('left'))} -> {_format_value(provenance.get('right'))}"
        )

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for comparing two saved experiment artifacts."""
    parser = argparse.ArgumentParser(description="Compare two Labs experiment JSON artifacts")
    parser.add_argument("left", type=Path, help="Baseline artifact or registry-row JSON")
    parser.add_argument("right", type=Path, help="Candidate artifact or registry-row JSON")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format")
    args = parser.parse_args(argv)

    diff = diff_experiment_files(args.left, args.right)
    if args.format == "json":
        sys.stdout.write(json.dumps(diff, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(format_experiment_diff(diff) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
