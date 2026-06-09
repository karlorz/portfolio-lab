"""Offline Labs experiment scorecard generation.

The scorecard generator reads already-published Labs registry, replay, and
validation artifacts only. It never reruns experiments or fetches market data.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.paths import PUBLIC_DATA_DIR
from src.research.experiment_artifact_validator import (
    LABS_SCORECARD_SCHEMA_VERSION,
    validate_artifact,
)
from src.research.experiment_registry import LABS_REGISTRY_FILENAME
from src.research.experiment_replay_batch import LABS_REPLAYS_FILENAME
from src.research.labs_validation_report import LABS_VALIDATION_FILENAME

logger = logging.getLogger(__name__)

LABS_SCORECARDS_FILENAME = "labs_scorecards.json"
LABS_SCORECARD_POLICY_ENV_VAR = "LABS_SCORECARD_POLICY_FILE"

_CLEAN_PROVENANCE_STATUSES = {"present", "embedded", "sidecar"}
_BAD_PROVENANCE_STATUSES = {"malformed", "stale"}
_REJECT_REGISTRY_STATUSES = {"warning", "rejected", "archived"}
DEFAULT_SCORECARD_POLICY_VERSION = "default-v1"
DEFAULT_SCORECARD_POLICY_THRESHOLDS: dict[str, float] = {
    "min_promote_sharpe": 0.9,
    "min_promote_sharpe_delta": 0.04,
    "min_promote_dsr": 0.95,
    "min_promote_wfe": 1.0,
}
_SCORECARD_POLICY_THRESHOLD_KEYS = tuple(DEFAULT_SCORECARD_POLICY_THRESHOLDS)

__all__ = [
    "LABS_SCORECARDS_FILENAME",
    "LABS_SCORECARD_POLICY_ENV_VAR",
    "DEFAULT_SCORECARD_POLICY_THRESHOLDS",
    "build_labs_scorecards",
    "save_labs_scorecards",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: str | Path | None) -> Any:
    if path is None:
        return None
    with open(Path(path)) as f:
        return json.load(f)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def _metric_value(metrics: Mapping[str, Any], key: str) -> float | None:
    return _finite_number(metrics.get(key))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    if isinstance(value, Mapping):
        return [value]
    return []


def _default_scorecard_policy() -> dict[str, Any]:
    return {
        "version": DEFAULT_SCORECARD_POLICY_VERSION,
        "thresholds": dict(DEFAULT_SCORECARD_POLICY_THRESHOLDS),
    }


def _policy_threshold(policy: Mapping[str, Any], key: str) -> float:
    thresholds = _mapping(policy.get("thresholds"))
    value = _finite_number(thresholds.get(key))
    return value if value is not None else DEFAULT_SCORECARD_POLICY_THRESHOLDS[key]


def _policy_copy(policy: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "version": str(policy.get("version", DEFAULT_SCORECARD_POLICY_VERSION)),
        "thresholds": {
            key: _policy_threshold(policy, key)
            for key in _SCORECARD_POLICY_THRESHOLD_KEYS
        },
    }


def _parse_scorecard_policy(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    version = payload.get("version", DEFAULT_SCORECARD_POLICY_VERSION)
    if not isinstance(version, str) or not version.strip():
        return None
    thresholds_payload = payload.get("thresholds")
    if not isinstance(thresholds_payload, Mapping):
        return None

    thresholds = dict(DEFAULT_SCORECARD_POLICY_THRESHOLDS)
    for key in _SCORECARD_POLICY_THRESHOLD_KEYS:
        if key not in thresholds_payload:
            continue
        value = _finite_number(thresholds_payload[key])
        if value is None or value < 0:
            return None
        thresholds[key] = value

    return {
        "version": version,
        "thresholds": thresholds,
    }


def _load_scorecard_policy(
    *,
    policy: Mapping[str, Any] | None,
    policy_path: str | Path | None,
) -> dict[str, Any]:
    if policy is not None:
        parsed = _parse_scorecard_policy(policy)
        if parsed is not None:
            return parsed
        logger.warning("Invalid inline Labs scorecard policy; using default policy")
        return _default_scorecard_policy()

    resolved_policy_path = policy_path or os.environ.get(LABS_SCORECARD_POLICY_ENV_VAR)
    if not resolved_policy_path:
        return _default_scorecard_policy()

    try:
        payload = _load_json(resolved_policy_path)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        logger.warning("Unable to load Labs scorecard policy %s: %s; using default policy", resolved_policy_path, exc)
        return _default_scorecard_policy()

    parsed = _parse_scorecard_policy(payload)
    if parsed is not None:
        return parsed
    logger.warning("Invalid Labs scorecard policy %s; using default policy", resolved_policy_path)
    return _default_scorecard_policy()


def _registry_payload(
    *,
    registry: Mapping[str, Any] | None,
    registry_path: str | Path | None,
) -> Mapping[str, Any]:
    payload = registry if registry is not None else _load_json(registry_path)
    if not isinstance(payload, Mapping):
        raise ValueError("Labs scorecard generation requires a registry object or registry_path")
    result = validate_artifact(payload)
    if not result.valid:
        raise ValueError(f"invalid Labs registry for scorecards: {result.error_messages()}")
    return payload


def _experiment_rows(registry: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = registry.get("experiments")
    return [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []


def _invalid_validation_keys(validation_report: Mapping[str, Any] | None) -> tuple[set[str], set[str]]:
    if validation_report is None:
        return set(), set()
    results = validation_report.get("results")
    if not isinstance(results, list):
        return set(), set()

    experiment_ids: set[str] = set()
    artifact_paths: set[str] = set()
    for row in results:
        if not isinstance(row, Mapping) or row.get("valid") is not False:
            continue
        experiment_id = row.get("experiment_id")
        if isinstance(experiment_id, str) and experiment_id:
            experiment_ids.add(experiment_id)
        artifact_path = row.get("artifact_path") or row.get("path")
        if isinstance(artifact_path, str) and artifact_path:
            artifact_paths.add(artifact_path)
    return experiment_ids, artifact_paths


def _failed_replay_ids(replays: Iterable[Mapping[str, Any]] | None) -> set[str]:
    failed: set[str] = set()
    for row in replays or []:
        experiment_id = row.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id:
            continue
        if row.get("status") == "failed" or row.get("passed") is False:
            failed.add(experiment_id)
    return failed


def _load_validation_report(
    validation_report: Mapping[str, Any] | None,
    validation_report_path: str | Path | None,
) -> Mapping[str, Any] | None:
    if validation_report is not None:
        return validation_report
    payload = _load_json(validation_report_path)
    return payload if isinstance(payload, Mapping) else None


def _load_replays(
    replays: Sequence[Mapping[str, Any]] | None,
    replay_path: str | Path | None,
) -> list[Mapping[str, Any]]:
    if replays is not None:
        return list(replays)
    return _mapping_list(_load_json(replay_path))


def _classification(
    row: Mapping[str, Any],
    *,
    invalid_experiment_ids: set[str],
    invalid_artifact_paths: set[str],
    failed_replay_ids: set[str],
    policy: Mapping[str, Any],
) -> str:
    experiment_id = str(row.get("experiment_id", ""))
    artifact_path = str(row.get("artifact_path", ""))
    row_status = str(row.get("status", ""))
    provenance_status = str(row.get("provenance_status", "unknown"))
    metrics = _mapping(row.get("metrics"))
    baseline_deltas = _mapping(row.get("baseline_deltas"))

    if (
        row_status in _REJECT_REGISTRY_STATUSES
        or provenance_status in _BAD_PROVENANCE_STATUSES
        or experiment_id in invalid_experiment_ids
        or artifact_path in invalid_artifact_paths
        or experiment_id in failed_replay_ids
    ):
        return "reject"

    sharpe = _metric_value(metrics, "sharpe")
    sharpe_delta = _metric_value(baseline_deltas, "sharpe")
    max_drawdown_delta = _metric_value(baseline_deltas, "max_drawdown_pct")
    dsr = _metric_value(metrics, "dsr")
    wfe = _metric_value(metrics, "wfe")

    if sharpe_delta is not None and sharpe_delta < 0:
        return "reject"
    if max_drawdown_delta is not None and max_drawdown_delta < 0:
        return "reject"

    promote_ready = (
        provenance_status in _CLEAN_PROVENANCE_STATUSES
        and sharpe is not None
        and sharpe >= _policy_threshold(policy, "min_promote_sharpe")
        and sharpe_delta is not None
        and sharpe_delta >= _policy_threshold(policy, "min_promote_sharpe_delta")
        and (dsr is None or dsr >= _policy_threshold(policy, "min_promote_dsr"))
        and (wfe is None or wfe >= _policy_threshold(policy, "min_promote_wfe"))
    )
    return "promote" if promote_ready else "watch"


def _scorecard_row(
    row: Mapping[str, Any],
    *,
    generated_at: str,
    invalid_experiment_ids: set[str],
    invalid_artifact_paths: set[str],
    failed_replay_ids: set[str],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    scorecard = {
        "schema_version": LABS_SCORECARD_SCHEMA_VERSION,
        "experiment_id": str(row.get("experiment_id", "")),
        "generated_at": generated_at,
        "status": _classification(
            row,
            invalid_experiment_ids=invalid_experiment_ids,
            invalid_artifact_paths=invalid_artifact_paths,
            failed_replay_ids=failed_replay_ids,
            policy=policy,
        ),
        "provenance_status": str(row.get("provenance_status", "unknown")),
        "metrics": dict(_mapping(row.get("metrics"))),
        "baseline_deltas": dict(_mapping(row.get("baseline_deltas"))),
        "policy": _policy_copy(policy),
    }
    validation = validate_artifact(scorecard)
    if not validation.valid:
        raise ValueError(f"generated Labs scorecard failed validation: {validation.error_messages()}")
    return scorecard


def build_labs_scorecards(
    *,
    registry: Mapping[str, Any] | None = None,
    registry_path: str | Path | None = None,
    validation_report: Mapping[str, Any] | None = None,
    validation_report_path: str | Path | None = None,
    replays: Sequence[Mapping[str, Any]] | None = None,
    replay_path: str | Path | None = None,
    policy: Mapping[str, Any] | None = None,
    policy_path: str | Path | None = None,
    generated_at: str | None = None,
) -> list[dict[str, Any]]:
    """Build scorecards from existing Labs artifacts."""
    generated_at = generated_at or _now_iso()
    scorecard_policy = _load_scorecard_policy(policy=policy, policy_path=policy_path)
    registry_payload = _registry_payload(registry=registry, registry_path=registry_path)
    validation_payload = _load_validation_report(validation_report, validation_report_path)
    invalid_experiment_ids, invalid_artifact_paths = _invalid_validation_keys(validation_payload)
    failed_replay_ids = _failed_replay_ids(_load_replays(replays, replay_path))

    return [
        _scorecard_row(
            row,
            generated_at=generated_at,
            invalid_experiment_ids=invalid_experiment_ids,
            invalid_artifact_paths=invalid_artifact_paths,
            failed_replay_ids=failed_replay_ids,
            policy=scorecard_policy,
        )
        for row in _experiment_rows(registry_payload)
    ]


def save_labs_scorecards(
    *,
    registry: Mapping[str, Any] | None = None,
    registry_path: str | Path | None = None,
    validation_report: Mapping[str, Any] | None = None,
    validation_report_path: str | Path | None = None,
    replays: Sequence[Mapping[str, Any]] | None = None,
    replay_path: str | Path | None = None,
    policy: Mapping[str, Any] | None = None,
    policy_path: str | Path | None = None,
    public_dir: str | Path = PUBLIC_DATA_DIR,
    output_path: str | Path | None = None,
    generated_at: str | None = None,
    write_empty: bool = False,
) -> Path | None:
    """Write `labs_scorecards.json` when registry rows exist."""
    scorecards = build_labs_scorecards(
        registry=registry,
        registry_path=registry_path,
        validation_report=validation_report,
        validation_report_path=validation_report_path,
        replays=replays,
        replay_path=replay_path,
        policy=policy,
        policy_path=policy_path,
        generated_at=generated_at,
    )
    if not scorecards and not write_empty:
        return None

    public_dir_path = Path(public_dir)
    target_path = Path(output_path) if output_path is not None else public_dir_path / LABS_SCORECARDS_FILENAME
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w") as f:
        json.dump(scorecards, f, indent=2, sort_keys=True)
        f.write("\n")
    logger.info("Labs scorecards written: %s (%d rows)", target_path, len(scorecards))
    return target_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate offline Labs scorecards")
    parser.add_argument("--registry-path", default=str(PUBLIC_DATA_DIR / LABS_REGISTRY_FILENAME))
    parser.add_argument("--validation-report-path", default=str(PUBLIC_DATA_DIR / LABS_VALIDATION_FILENAME))
    parser.add_argument("--replay-path", default=str(PUBLIC_DATA_DIR / LABS_REPLAYS_FILENAME))
    parser.add_argument("--policy-path", default=None)
    parser.add_argument("--public-dir", default=str(PUBLIC_DATA_DIR))
    parser.add_argument("--write-empty", action="store_true")
    args = parser.parse_args(argv)

    output_path = save_labs_scorecards(
        registry_path=args.registry_path,
        validation_report_path=args.validation_report_path if Path(args.validation_report_path).exists() else None,
        replay_path=args.replay_path if Path(args.replay_path).exists() else None,
        policy_path=args.policy_path,
        public_dir=args.public_dir,
        write_empty=args.write_empty,
    )
    sys.stdout.write(json.dumps({"output_path": str(output_path) if output_path else None}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
