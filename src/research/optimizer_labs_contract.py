"""Labs-compatible serialization for optimizer output artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

OPTIMIZER_LABS_SCHEMA_VERSION = "optimizer-labs-output/v1"

OPTIMIZER_METHOD_ORDER: tuple[str, ...] = (
    "max_sharpe",
    "min_volatility",
    "efficient_risk",
    "hrp",
    "champion",
)

__all__ = [
    "OPTIMIZER_LABS_SCHEMA_VERSION",
    "build_optimizer_labs_output",
    "save_optimizer_labs_output",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _experiment_id(method: str) -> str:
    if method == "champion":
        return "optimizer:champion_reference"
    return f"optimizer:{method}"


def _status(method: str, result: Mapping[str, Any]) -> str:
    if result.get("error"):
        return "failed"
    if method == "champion":
        return "reference"
    return "succeeded"


def _registry_status(row_status: str) -> str:
    if row_status == "failed":
        return "warning"
    if row_status == "reference":
        return "validated"
    return "candidate"


def _numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _metrics(result: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    metric_fields = {
        "sharpe": "sharpe",
        "cagr": "cagr_pct",
        "cagr_pct": "cagr_pct",
        "volatility": "volatility_pct",
        "volatility_pct": "volatility_pct",
        "max_drawdown": "max_drawdown_pct",
        "max_drawdown_pct": "max_drawdown_pct",
        "wfe": "wfe",
        "dsr": "dsr",
        "positive_oos_ratio": "positive_oos_ratio",
        "regime_coverage": "regime_coverage",
    }
    for source_key, target_key in metric_fields.items():
        numeric_value = _numeric(result.get(source_key))
        if numeric_value is not None:
            metrics[target_key] = numeric_value
    return metrics


def _weights(result: Mapping[str, Any]) -> dict[str, float]:
    raw_weights = result.get("weights")
    if not isinstance(raw_weights, Mapping):
        return {}
    weights: dict[str, float] = {}
    for symbol, value in raw_weights.items():
        numeric_value = _numeric(value)
        if isinstance(symbol, str) and numeric_value is not None:
            weights[symbol] = numeric_value
    return weights


def _baseline_deltas(metrics: Mapping[str, float], champion_metrics: Mapping[str, float]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for metric, value in metrics.items():
        champion_value = champion_metrics.get(metric)
        if champion_value is not None:
            deltas[metric] = value - champion_value
    return deltas


def _method_items(results: Mapping[str, Mapping[str, Any]]) -> list[tuple[str, Mapping[str, Any]]]:
    ordered = [(method, results[method]) for method in OPTIMIZER_METHOD_ORDER if method in results]
    ordered.extend((method, result) for method, result in sorted(results.items()) if method not in OPTIMIZER_METHOD_ORDER)
    return ordered


def _optimizer_row(
    *,
    method: str,
    result: Mapping[str, Any],
    symbols: Sequence[str],
    default_target_vol: float,
    generated_at: str,
    artifact_path: str,
    champion_metrics: Mapping[str, float],
) -> dict[str, Any]:
    status = _status(method, result)
    metrics = {} if status == "failed" else _metrics(result)
    row_target_vol = result.get("target_vol") if method == "efficient_risk" else None
    if row_target_vol is None and method == "efficient_risk":
        row_target_vol = default_target_vol

    return {
        "experiment_id": _experiment_id(method),
        "method": method,
        "status": status,
        "generated_at": generated_at,
        "artifact_path": f"{artifact_path}#{method}",
        "symbols": list(symbols),
        "target_vol": row_target_vol,
        "weights": _weights(result),
        "metrics": metrics,
        "baseline_deltas": _baseline_deltas(metrics, champion_metrics),
        "provenance_status": "present",
        "error": str(result["error"]) if result.get("error") else None,
        "note": result.get("note") if isinstance(result.get("note"), str) else None,
    }


def _registry_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": row["experiment_id"],
        "artifact_path": row["artifact_path"],
        "status": _registry_status(str(row["status"])),
        "provenance_status": row["provenance_status"],
        "metrics": dict(row["metrics"]),
        "baseline_deltas": dict(row["baseline_deltas"]),
    }


def build_optimizer_labs_output(
    results: Mapping[str, Mapping[str, Any]],
    *,
    symbols: Sequence[str],
    target_vol: float,
    generated_at: str | None = None,
    artifact_path: str = "data/optimized_weights.json",
) -> dict[str, Any]:
    """Build a stable Labs-compatible optimizer output payload."""
    generated_at = generated_at or _now_iso()
    champion_metrics = _metrics(results.get("champion", {}))
    rows = [
        _optimizer_row(
            method=method,
            result=result,
            symbols=symbols,
            default_target_vol=target_vol,
            generated_at=generated_at,
            artifact_path=artifact_path,
            champion_metrics=champion_metrics,
        )
        for method, result in _method_items(results)
    ]

    return {
        "schema_version": OPTIMIZER_LABS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "symbols": list(symbols),
        "target_vol": target_vol,
        "artifact_path": artifact_path,
        "optimizer_results": rows,
        "registry": {
            "schema_version": "labs-registry/v1",
            "generated_at": generated_at,
            "experiments": [_registry_row(row) for row in rows],
        },
    }


def save_optimizer_labs_output(
    results: Mapping[str, Mapping[str, Any]],
    *,
    output_path: str | Path,
    symbols: Sequence[str],
    target_vol: float,
    generated_at: str | None = None,
) -> Path:
    """Write optimizer Labs output JSON and return the saved path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_optimizer_labs_output(
        results,
        symbols=symbols,
        target_vol=target_vol,
        generated_at=generated_at,
        artifact_path=output_path.as_posix(),
    )
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return output_path
