"""Decision replay ledger and experiment registry (SQLite + dashboard JSON).

Complements offline ``src/research/experiment_registry.py`` (labs artifact scanner)
with append-only runtime decisions from dashboard / ensemble / rebalance flows.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.paths import DATA_DIR, PROJECT_ROOT, PUBLIC_DATA_DIR, sqlite_connect
from src.research.promotion_policy import (
    classify_offline_promotion_governance,
    governance_failures,
)

logger = logging.getLogger(__name__)

DECISION_REGISTRY_SCHEMA_VERSION = "decision-registry/v1"
DECISION_REGISTRY_DB = Path(
    os.environ.get("DECISION_REGISTRY_DB", str(DATA_DIR / "decision_registry.db"))
)
DECISION_REGISTRY_JSON = "decision_registry.json"

PromotionStatus = Literal["candidate", "shadow", "promoted", "rejected", "archived"]
DecisionRole = Literal["live_executed", "shadow"]
DecisionSource = Literal["dashboard_cycle", "evaluator_cycle", "manual"]
DecisionPromotionReviewStatus = Literal[
    "not_applicable",
    "pending_review",
    "eligible_for_promotion",
    "promoted",
    "rejected",
    "archived",
]
BenchmarkWindowValue = str | int | float | None

__all__ = [
    "DECISION_REGISTRY_DB",
    "DECISION_REGISTRY_JSON",
    "DECISION_REGISTRY_SCHEMA_VERSION",
    "DecisionRecord",
    "DecisionRegistry",
    "ExperimentRecord",
    "build_decision_registry_snapshot",
    "evaluate_promotion_candidate",
    "publish_decision_registry_json",
    "record_backtest_experiment",
    "record_dashboard_cycle_decision",
    "record_evaluator_cycle_decision",
    "sync_labs_registry_experiments",
]


class DecisionRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    decision_id: str
    timestamp_utc: str
    git_sha: str | None = None
    run_id: str
    strategy_version: str = "champion"
    portfolio_value: float | None = None
    current_weights: dict[str, float] = Field(default_factory=dict)
    target_weights: dict[str, float] = Field(default_factory=dict)
    action: str
    reason: str = ""
    regime: str | None = None
    regime_confidence: float | None = None
    signal_votes: dict[str, float] = Field(default_factory=dict)
    signal_weights: dict[str, float] = Field(default_factory=dict)
    risk_metrics: dict[str, float] = Field(default_factory=dict)
    gates_triggered: list[str] = Field(default_factory=list)
    data_snapshot_hash: str | None = None
    freeze_manifest_hash: str | None = None
    decision_role: DecisionRole = "live_executed"
    decision_source: DecisionSource | None = None
    controller_id: str | None = None
    live_decision_id: str | None = None
    baseline_controller_id: str | None = None
    benchmark_window: dict[str, BenchmarkWindowValue] = Field(default_factory=dict)
    divergence_metrics: dict[str, float] = Field(default_factory=dict)
    promotion_review_status: DecisionPromotionReviewStatus = "not_applicable"
    extras: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_shadow_evidence_contract(self) -> "DecisionRecord":
        if self.decision_role != "shadow":
            return self
        if not self.controller_id:
            raise ValueError("shadow decisions require controller_id")
        if not self.live_decision_id:
            raise ValueError("shadow decisions require live_decision_id linkage")
        if self.promotion_review_status == "not_applicable":
            self.promotion_review_status = "pending_review"
        return self


class ExperimentRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    experiment_id: str
    timestamp_utc: str
    name: str
    hypothesis: str = ""
    git_sha: str | None = None
    data_start: str | None = None
    data_end: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    benchmark_metrics: dict[str, float] = Field(default_factory=dict)
    promotion_status: PromotionStatus = "candidate"
    rejection_reason: str | None = None
    artifacts: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_sha_short(project_root: Path | None = None) -> str | None:
    root = project_root or PROJECT_ROOT
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=root,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _freeze_manifest_hash() -> str | None:
    try:
        from src.monitor.freeze_manifest import load_manifest

        manifest = load_manifest()
        if not manifest:
            return None
        payload = json.dumps(manifest, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
    except Exception:
        return None


def _is_temp_path(value: Any) -> bool:
    text = str(value or "")
    return "/tmp/pytest-of-root/" in text or text.startswith("/tmp/pytest-")


def _is_fixture_or_temp_experiment(experiment_id: str, artifacts: Mapping[str, Any]) -> bool:
    """Identify rows that are test fixtures rather than production evidence."""
    return (
        experiment_id == "bad-registry-row"
        or _is_temp_path(artifacts.get("artifact_path"))
        or _is_temp_path(artifacts.get("output_path"))
    )


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS decisions (
            decision_id TEXT PRIMARY KEY,
            timestamp_utc TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(timestamp_utc);

        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id TEXT PRIMARY KEY,
            timestamp_utc TEXT NOT NULL,
            promotion_status TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(promotion_status);
        """
    )


class DecisionRegistry:
    """SQLite-backed decision and experiment ledger."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or DECISION_REGISTRY_DB)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite_connect(self.db_path)) as conn:
            _init_schema(conn)
            conn.commit()

    def record_decision(self, record: DecisionRecord) -> str:
        payload = record.model_dump(mode="json")
        with closing(sqlite_connect(self.db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO decisions (decision_id, timestamp_utc, payload_json) VALUES (?, ?, ?)",
                (record.decision_id, record.timestamp_utc, _json_dumps(payload)),
            )
            conn.commit()
        return record.decision_id

    def record_experiment(self, record: ExperimentRecord) -> str:
        payload = record.model_dump(mode="json")
        with closing(sqlite_connect(self.db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO experiments (experiment_id, timestamp_utc, promotion_status, payload_json) VALUES (?, ?, ?, ?)",
                (
                    record.experiment_id,
                    record.timestamp_utc,
                    record.promotion_status,
                    _json_dumps(payload),
                ),
            )
            conn.commit()
        return record.experiment_id

    def get_decision(self, decision_id: str) -> DecisionRecord | None:
        with closing(sqlite_connect(self.db_path)) as conn:
            row = conn.execute(
                "SELECT payload_json FROM decisions WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
        if not row:
            return None
        return DecisionRecord.model_validate(json.loads(row[0]))

    def list_recent_decisions(self, limit: int = 50) -> list[DecisionRecord]:
        with closing(sqlite_connect(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT payload_json FROM decisions ORDER BY timestamp_utc DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [DecisionRecord.model_validate(json.loads(r[0])) for r in rows]

    def decision_head(self) -> dict[str, Any]:
        """Return the authoritative ledger head and total decision count."""
        with closing(sqlite_connect(self.db_path)) as conn:
            count = int(conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0])
            row = conn.execute(
                "SELECT decision_id, timestamp_utc FROM decisions "
                "ORDER BY timestamp_utc DESC LIMIT 1",
            ).fetchone()
        head = None
        if row:
            head = {"decision_id": str(row[0]), "timestamp_utc": str(row[1])}
        return {"head": head, "count": count}

    def list_experiments(self, status: str | None = None) -> list[ExperimentRecord]:
        with closing(sqlite_connect(self.db_path)) as conn:
            if status:
                rows = conn.execute(
                    "SELECT payload_json FROM experiments WHERE promotion_status = ? ORDER BY timestamp_utc DESC",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT payload_json FROM experiments ORDER BY timestamp_utc DESC",
                ).fetchall()
        return [ExperimentRecord.model_validate(json.loads(r[0])) for r in rows]

    def replay_decision(self, decision_id: str) -> dict[str, Any]:
        record = self.get_decision(decision_id)
        if record is None:
            return {"decision_id": decision_id, "found": False, "replay": None}

        top_signals = sorted(
            record.signal_weights.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:8]
        weight_delta: dict[str, float] = {}
        keys = set(record.current_weights) | set(record.target_weights)
        for key in keys:
            cur = float(record.current_weights.get(key, 0.0))
            tgt = float(record.target_weights.get(key, 0.0))
            if abs(tgt - cur) >= 1e-6:
                weight_delta[key] = round(tgt - cur, 6)

        return {
            "decision_id": decision_id,
            "found": True,
            "replay": {
                "summary": (
                    f"Action {record.action!r} under regime {record.regime!r} "
                    f"(confidence={record.regime_confidence})."
                ),
                "action": record.action,
                "reason": record.reason,
                "regime": record.regime,
                "regime_confidence": record.regime_confidence,
                "gates_triggered": list(record.gates_triggered),
                "top_signal_weights": [{"signal": k, "weight": v} for k, v in top_signals],
                "weight_delta": weight_delta,
                "risk_metrics": dict(record.risk_metrics),
                "git_sha": record.git_sha,
                "run_id": record.run_id,
                "strategy_version": record.strategy_version,
                "decision_role": record.decision_role,
                "decision_source": record.decision_source,
                "controller_id": record.controller_id,
                "live_decision_id": record.live_decision_id,
                "baseline_controller_id": record.baseline_controller_id,
                "benchmark_window": dict(record.benchmark_window),
                "divergence_metrics": dict(record.divergence_metrics),
                "promotion_review_status": record.promotion_review_status,
                "data_snapshot_hash": record.data_snapshot_hash,
                "freeze_manifest_hash": record.freeze_manifest_hash,
            },
        }


def evaluate_promotion_candidate(
    experiment: ExperimentRecord | Mapping[str, Any],
    *,
    benchmark_metrics: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Rule-based promotion helper with conservative env-configurable thresholds."""
    if isinstance(experiment, ExperimentRecord):
        metrics = dict(experiment.metrics)
        bench = dict(experiment.benchmark_metrics)
        exp_id = experiment.experiment_id
        artifacts = dict(experiment.artifacts or {})
        governance_row: dict[str, Any] = {
            "promotion_status": experiment.promotion_status,
            "registry_status": artifacts.get("registry_status", experiment.promotion_status),
            "provenance_status": artifacts.get("provenance_status"),
            "artifacts": artifacts,
        }
        if _is_fixture_or_temp_experiment(exp_id, artifacts):
            governance_row["promotion_status"] = "rejected"
            governance_row["registry_status"] = "rejected"
            governance_row["provenance_status"] = "invalid"
    else:
        raw_artifacts = experiment.get("artifacts", {})
        artifacts = dict(raw_artifacts) if isinstance(raw_artifacts, Mapping) else {}
        metrics = {
            k: float(v)
            for k, v in dict(experiment.get("metrics", {})).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        bench = {
            k: float(v)
            for k, v in dict(experiment.get("benchmark_metrics", {})).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        exp_id = str(experiment.get("experiment_id", "unknown"))
        governance_row = {
            "promotion_status": experiment.get("promotion_status", experiment.get("status")),
            "registry_status": artifacts.get(
                "registry_status",
                experiment.get("registry_status", experiment.get("status")),
            ),
            "provenance_status": experiment.get(
                "provenance_status",
                artifacts.get("provenance_status"),
            ),
            "artifacts": artifacts,
        }
        if _is_fixture_or_temp_experiment(exp_id, artifacts):
            governance_row["promotion_status"] = "rejected"
            governance_row["registry_status"] = "rejected"
            governance_row["provenance_status"] = "invalid"

    if benchmark_metrics:
        bench.update({k: float(v) for k, v in benchmark_metrics.items()})

    min_sharpe_delta = float(os.environ.get("PROMOTION_MIN_SHARPE_DELTA", "0.02"))
    max_dd_worse = float(os.environ.get("PROMOTION_MAX_DD_WORSE_PCT", "2.0"))
    max_cvar_worse = float(os.environ.get("PROMOTION_MAX_CVAR_WORSE", "0.05"))
    max_turnover = float(os.environ.get("PROMOTION_MAX_TURNOVER", "1.5"))
    require_wfe = os.environ.get("PROMOTION_REQUIRE_WFE", "0").lower() in {"1", "true", "yes"}

    failures: list[str] = []
    if not metrics:
        failures.append("missing_metrics")
    if not bench:
        failures.append("missing_benchmark_metrics")

    sharpe = metrics.get("sharpe")
    bench_sharpe = bench.get("sharpe")
    if sharpe is not None and bench_sharpe is not None:
        if sharpe - bench_sharpe < min_sharpe_delta:
            failures.append(f"sharpe_delta<{min_sharpe_delta}")

    dd = metrics.get("max_drawdown_pct", metrics.get("max_drawdown"))
    bench_dd = bench.get("max_drawdown_pct", bench.get("max_drawdown"))
    if dd is not None and bench_dd is not None:
        if float(dd) < float(bench_dd) - max_dd_worse:
            failures.append("max_drawdown_worse_than_benchmark")

    cvar = metrics.get("cvar")
    bench_cvar = bench.get("cvar")
    if cvar is not None and bench_cvar is not None:
        if float(cvar) > float(bench_cvar) + max_cvar_worse:
            failures.append("cvar_worse_than_benchmark")

    turnover = metrics.get("turnover")
    if turnover is not None and float(turnover) > max_turnover:
        failures.append("turnover_over_budget")

    if require_wfe and metrics.get("wfe") is None:
        failures.append("missing_walk_forward_wfe")

    metric_gate_status = "promoted" if not failures else "candidate"
    policy = classify_offline_promotion_governance(
        governance_row,
        metric_gate_status=metric_gate_status,
        metric_gate_pass=not failures,
        metric_failures=failures,
    )
    return {
        "experiment_id": exp_id,
        "recommended_status": policy["recommended_status"],
        "metric_gate_status": policy["metric_gate_status"],
        "metric_gate_pass": policy["metric_gate_pass"],
        "pass": policy["pass"],
        "failures": policy["failures"],
        "thresholds": {
            "min_sharpe_delta": min_sharpe_delta,
            "max_dd_worse_pct": max_dd_worse,
            "max_cvar_worse": max_cvar_worse,
            "max_turnover": max_turnover,
            "require_wfe": require_wfe,
        },
    }


def _promotion_semantic_disclosure(
    experiment: ExperimentRecord,
    promotion_row: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Disclose when a metric-only promotion is blocked by governance policy."""
    recommended_status = str(promotion_row.get("recommended_status") or "")
    metric_gate_status = str(promotion_row.get("metric_gate_status") or "")
    artifacts = dict(experiment.artifacts or {})
    governance_status = str(artifacts.get("registry_status") or experiment.promotion_status or "candidate")
    provenance_raw = artifacts.get("provenance_status")
    provenance_status = str(provenance_raw) if provenance_raw is not None else "unknown"

    if metric_gate_status != "promoted" or recommended_status == "promoted":
        return None

    reasons = governance_failures(
        {
            "registry_status": governance_status,
            "provenance_status": provenance_status,
        }
    )
    if not reasons:
        return None

    return {
        "state": "governance_blocked",
        "recommendation_type": "metric_gate",
        "governance_status": governance_status,
        "provenance_status": provenance_status,
        "metric_gate_status": metric_gate_status,
        "reasons": reasons,
    }


def _build_shadow_evidence_summary(decisions: list[DecisionRecord]) -> dict[str, Any]:
    shadow_decisions = [decision for decision in decisions if decision.decision_role == "shadow"]
    status_counts: dict[str, int] = {}
    for decision in shadow_decisions:
        status = decision.promotion_review_status
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "shadow_decision_count": len(shadow_decisions),
        "linked_shadow_decision_count": sum(1 for d in shadow_decisions if d.live_decision_id),
        "controllers": sorted({d.controller_id for d in shadow_decisions if d.controller_id}),
        "promotion_review_status_counts": dict(sorted(status_counts.items())),
        "latest_shadow_decisions": [
            {
                "decision_id": decision.decision_id,
                "controller_id": decision.controller_id,
                "live_decision_id": decision.live_decision_id,
                "baseline_controller_id": decision.baseline_controller_id,
                "promotion_review_status": decision.promotion_review_status,
                "divergence_metrics": dict(decision.divergence_metrics),
                "benchmark_window": dict(decision.benchmark_window),
            }
            for decision in shadow_decisions[:10]
        ],
    }


def _parse_iso_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).timestamp()
    except ValueError:
        return None


def _build_projection_freshness(
    registry: DecisionRegistry,
    decisions: list[DecisionRecord],
) -> dict[str, Any]:
    """Describe whether the public replay projection is current to ledger head."""
    ledger = registry.decision_head()
    ledger_head = ledger["head"]
    projection_head = None
    if decisions:
        projection_head = {
            "decision_id": decisions[0].decision_id,
            "timestamp_utc": decisions[0].timestamp_utc,
        }

    current = ledger_head == projection_head
    lag_seconds: float | None = 0.0
    if ledger_head and projection_head and not current:
        ledger_ts = _parse_iso_seconds(ledger_head.get("timestamp_utc"))
        projection_ts = _parse_iso_seconds(projection_head.get("timestamp_utc"))
        if ledger_ts is not None and projection_ts is not None:
            lag_seconds = max(ledger_ts - projection_ts, 0.0)
    elif ledger_head and projection_head is None:
        lag_seconds = None

    return {
        "status": "current" if current else "projection_lagged",
        "ledger_head": ledger_head,
        "projection_head": projection_head,
        "lag_decision_count": 0 if current else max(int(ledger["count"]) - len(decisions), 1),
        "lag_seconds": lag_seconds,
    }


def _public_experiment_payload(experiment: ExperimentRecord) -> dict[str, Any]:
    """Serialize experiment rows without exposing test fixtures as production evidence."""
    payload = experiment.model_dump(mode="json")
    artifacts = dict(payload.get("artifacts") or {})
    if not _is_fixture_or_temp_experiment(experiment.experiment_id, artifacts):
        return payload

    for key in ("artifact_path", "output_path"):
        if _is_temp_path(artifacts.get(key)):
            artifacts[key] = None
    artifacts["registry_status"] = "rejected"
    artifacts["provenance_status"] = "invalid"
    artifacts["temp_path_redacted"] = True
    artifacts["synthetic_or_test_only"] = True

    tags = list(payload.get("tags") or [])
    if "synthetic_or_test_only" not in tags:
        tags.append("synthetic_or_test_only")

    payload["hypothesis"] = ""
    payload["promotion_status"] = "rejected"
    payload["rejection_reason"] = "synthetic_or_test_only"
    payload["artifacts"] = artifacts
    payload["tags"] = tags
    return payload


def build_decision_registry_snapshot(
    registry: DecisionRegistry | None = None,
    *,
    decision_limit: int = 25,
    experiment_limit: int = 25,
    generated_at: str | None = None,
) -> dict[str, Any]:
    reg = registry or DecisionRegistry()
    decisions = reg.list_recent_decisions(limit=decision_limit)
    experiments = reg.list_experiments()[:experiment_limit]

    replay_summaries = [reg.replay_decision(dec.decision_id) for dec in decisions[:5]]
    promotion_rows = []
    evaluated_experiment_ids: set[str] = set()
    for exp in experiments:
        row = evaluate_promotion_candidate(exp)
        disclosure = _promotion_semantic_disclosure(exp, row)
        if disclosure is not None:
            row["semantic_disclosure"] = disclosure
        promotion_rows.append(row)
        evaluated_experiment_ids.add(exp.experiment_id)

    recent_experiments = [_public_experiment_payload(e) for e in experiments]
    unmatched_experiment_ids = [
        str(row.get("experiment_id"))
        for row in recent_experiments
        if str(row.get("experiment_id")) not in evaluated_experiment_ids
    ]

    ledger = reg.decision_head()
    ledger_total = int(ledger.get("count") or 0)
    # Experiment ledger total (window vs full)
    try:
        with closing(sqlite_connect(reg.db_path)) as conn:
            experiment_total = int(conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0])
    except Exception:  # noqa: BLE001
        experiment_total = len(experiments)

    snapshot = {
        "schema_version": DECISION_REGISTRY_SCHEMA_VERSION,
        "generated_at": generated_at or _now_iso(),
        "projection_freshness": _build_projection_freshness(reg, decisions),
        "recent_decisions": [d.model_dump(mode="json") for d in decisions],
        "recent_experiments": recent_experiments,
        "replay_summaries": replay_summaries,
        "promotion_evaluations": promotion_rows,
        "promotion_coverage": {
            "recent_experiment_count": len(recent_experiments),
            "evaluated_count": len(promotion_rows),
            "unmatched_count": len(unmatched_experiment_ids),
            "unmatched_experiment_ids": unmatched_experiment_ids,
            "disclosure": (
                "partial_promotion_evaluation_coverage"
                if unmatched_experiment_ids
                else "complete_promotion_evaluation_coverage"
            ),
        },
        "shadow_evidence": _build_shadow_evidence_summary(decisions),
        "counts": {
            # Ledger totals (fleet cardinality) — not projection window length
            "decisions": ledger_total,
            "experiments": experiment_total,
            "decisions_window": len(decisions),
            "experiments_window": len(experiments),
            "decision_limit": int(decision_limit),
            "experiment_limit": int(experiment_limit),
            "counts_scope": "ledger_total",
        },
    }
    sha = _git_sha_short()
    if sha:
        snapshot["generator_git_sha"] = sha
        snapshot["generator_git_sha_status"] = "full"
    else:
        snapshot["generator_git_sha"] = None
        snapshot["generator_git_sha_status"] = "unavailable"
    return snapshot


def publish_decision_registry_json(
    public_dir: str | Path = PUBLIC_DATA_DIR,
    *,
    registry: DecisionRegistry | None = None,
    private_dir: str | Path | None = None,
) -> Path:
    """Write decision_registry.json to public (and private DATA_DIR when distinct).

    Batch CK: private ``DATA_DIR/decision_registry.json`` was missing while
    public WWW carried a copy, so ``make mirror-repo-public-data-lag`` reported
    source_present=false / bytes lag forever. Dual-write both trees.
    """
    snapshot = build_decision_registry_snapshot(registry=registry)
    out = Path(public_dir) / DECISION_REGISTRY_JSON
    out.parent.mkdir(parents=True, exist_ok=True)
    priv_root = Path(private_dir) if private_dir is not None else Path(DATA_DIR)
    private_out = priv_root / DECISION_REGISTRY_JSON
    # Keep isolated pytest fixtures byte-identical, while real public roots
    # receive the shared logical-reference projection and additive provenance.
    from src.monitor.signal_authority import (
        _public_projection_enabled,
        serialize_json_payload,
    )

    public_body = serialize_json_payload(
        snapshot,
        output_path=out,
        public=_public_projection_enabled(out),
    )
    private_body = serialize_json_payload(
        snapshot,
        output_path=private_out,
        public=False,
    )
    with open(out, "w", encoding="utf-8") as f:
        f.write(public_body)
    logger.info(
        "Decision registry JSON written: %s (%d decisions)",
        out,
        snapshot["counts"]["decisions"],
    )
    # Private SSOT for mirror lag / repo-local consumers
    try:
        if private_out.resolve() != out.resolve():
            private_out.parent.mkdir(parents=True, exist_ok=True)
            private_out.write_text(private_body, encoding="utf-8")
            logger.info("Decision registry JSON dual-wrote private: %s", private_out)
    except OSError as exc:
        logger.warning("Decision registry private dual-write failed: %s", exc)
    return out


def record_dashboard_cycle_decision(
    signals_output: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
    registry: DecisionRegistry | None = None,
    run_id: str | None = None,
) -> str | None:
    """Record one dashboard-cycle decision from ensemble + smart rebalance sections."""
    reg = registry or DecisionRegistry()
    ctx = dict(context or {})

    ensemble = signals_output.get("ensemble_voting") or {}
    smart = signals_output.get("smart_rebalance") or {}
    if not isinstance(ensemble, dict):
        ensemble = {}
    if not isinstance(smart, dict):
        smart = {}

    action = str(smart.get("decision") or ensemble.get("action") or "observe")
    reason = str(smart.get("reason") or ensemble.get("action") or "")

    target_alloc = ctx.get("target_alloc") or signals_output.get("target_allocation") or {}
    positions = ctx.get("positions") or []
    current_weights: dict[str, float] = {}
    total_value = float(ctx.get("total_value") or 0.0)
    if total_value > 0 and isinstance(positions, list):
        for pos in positions:
            if isinstance(pos, dict) and pos.get("symbol"):
                current_weights[str(pos["symbol"])] = float(pos.get("value", 0)) / total_value

    target_weights: dict[str, float] = {}
    if isinstance(target_alloc, dict):
        target_weights = {str(k): float(v) for k, v in target_alloc.items()}

    signal_weights: dict[str, float] = {}
    signal_votes: dict[str, float] = {}
    for row in ensemble.get("source_breakdown") or []:
        if not isinstance(row, dict):
            continue
        src = str(row.get("source", "unknown"))
        signal_weights[src] = float(row.get("weight", 0.0))
        signal_votes[src] = float(row.get("strength", 0.0))

    gates: list[str] = []
    if smart.get("should_execute") is False and action not in {"no_positions", "observe"}:
        gates.append("smart_rebalance_hold")
    staleness = signals_output.get("staleness") or {}
    if isinstance(staleness, dict):
        stale = staleness.get("stale_signals") or staleness.get("stale") or []
        if isinstance(stale, list) and stale:
            gates.append("signal_staleness")

    risk_metrics: dict[str, float] = {}
    if smart.get("max_drift") is not None:
        risk_metrics["max_drift"] = float(smart["max_drift"])
    if smart.get("estimated_cost_bps") is not None:
        risk_metrics["estimated_cost_bps"] = float(smart["estimated_cost_bps"])

    ts = _now_iso()
    decision_id = str(uuid.uuid4())
    record = DecisionRecord(
        decision_id=decision_id,
        timestamp_utc=ts,
        git_sha=_git_sha_short(),
        run_id=run_id or f"dashboard-{ts[:19]}",
        decision_role="live_executed",
        decision_source="dashboard_cycle",
        controller_id="dashboard_generator",
        portfolio_value=total_value if total_value > 0 else None,
        current_weights=current_weights,
        target_weights=target_weights,
        action=action,
        reason=reason,
        regime=str(ensemble.get("regime") or ctx.get("current_regime") or "") or None,
        regime_confidence=(
            float(ensemble["regime_confidence"])
            if ensemble.get("regime_confidence") is not None
            else None
        ),
        signal_votes=signal_votes,
        signal_weights=signal_weights,
        risk_metrics=risk_metrics,
        gates_triggered=gates,
        freeze_manifest_hash=_freeze_manifest_hash(),
        extras={
            "ensemble_confidence": ensemble.get("confidence"),
            "weighted_consensus": ensemble.get("weighted_consensus"),
        },
    )
    try:
        return reg.record_decision(record)
    except (sqlite3.Error, OSError, ValueError) as exc:
        logger.warning("Decision registry record skipped: %s", exc)
        return None


def record_evaluator_cycle_decision(
    *,
    mode: str,
    regime: str,
    target_alloc: Mapping[str, float],
    prices: Mapping[str, float],
    portfolio_value: float,
    current_weights: Mapping[str, float],
    orders: list[Mapping[str, Any]] | None,
    kill_reason: str | None = None,
    registry: DecisionRegistry | None = None,
    public_dir: str | Path | None = None,
) -> str | None:
    """Record one strategy-evaluator rebalance / hold / kill-switch decision."""
    owns_default_registry = registry is None
    reg = registry or DecisionRegistry()
    ts = _now_iso()
    target_weights = {str(k): float(v) for k, v in target_alloc.items()}
    cur = {str(k): float(v) for k, v in current_weights.items()}

    gates: list[str] = []
    if kill_reason:
        gates.append("kill_switch")
        action = "halt"
        reason = kill_reason
    elif orders:
        action = "rebalance"
        reason = f"executed_{len(orders)}_orders"
    else:
        action = "hold"
        reason = "no_rebalancing_needed"

    risk_metrics: dict[str, float] = {}
    if portfolio_value > 0:
        risk_metrics["portfolio_value"] = float(portfolio_value)
        max_drift = 0.0
        for sym, tgt in target_weights.items():
            drift = abs(float(cur.get(sym, 0.0)) - tgt)
            max_drift = max(max_drift, drift)
        risk_metrics["max_drift"] = round(max_drift, 6)

    record = DecisionRecord(
        decision_id=str(uuid.uuid4()),
        timestamp_utc=ts,
        git_sha=_git_sha_short(),
        run_id=f"evaluator-{mode}-{ts[:19]}",
        decision_role="live_executed",
        decision_source="evaluator_cycle",
        controller_id="strategy_evaluator",
        strategy_version="champion",
        portfolio_value=float(portfolio_value) if portfolio_value > 0 else None,
        current_weights=cur,
        target_weights=target_weights,
        action=action,
        reason=reason,
        regime=str(regime) if regime else None,
        risk_metrics=risk_metrics,
        gates_triggered=gates,
        freeze_manifest_hash=_freeze_manifest_hash(),
        extras={"source": "evaluator", "mode": mode, "order_count": len(orders or [])},
    )
    try:
        decision_id = reg.record_decision(record)
    except (sqlite3.Error, OSError, ValueError) as exc:
        logger.warning("Evaluator decision registry record skipped: %s", exc)
        return None
    if owns_default_registry or public_dir is not None:
        try:
            publish_decision_registry_json(
                public_dir=public_dir or PUBLIC_DATA_DIR,
                registry=reg,
            )
        except (sqlite3.Error, OSError, ValueError, TypeError) as exc:
            logger.warning("Evaluator decision registry projection refresh skipped: %s", exc)
    return decision_id


def _metrics_from_result_payload(data: Mapping[str, Any]) -> dict[str, float]:
    """Normalize sharpe / drawdown keys from backtest or labs JSON artifacts."""
    out: dict[str, float] = {}
    for key, val in data.items():
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        out[str(key)] = float(val)

    nested = data.get("metrics")
    if isinstance(nested, Mapping):
        for key, val in nested.items():
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                continue
            out[str(key)] = float(val)

    if "sharpe" not in out and "sharpe_ratio" in out:
        out["sharpe"] = out["sharpe_ratio"]
    if "max_drawdown_pct" not in out and "max_drawdown" in out:
        out["max_drawdown_pct"] = out["max_drawdown"]
    return out


def record_backtest_experiment(
    data: Mapping[str, Any],
    *,
    experiment_id: str,
    output_path: str | Path | None = None,
    name: str | None = None,
    hypothesis: str = "",
    benchmark_metrics: Mapping[str, float] | None = None,
    tags: list[str] | None = None,
    registry: DecisionRegistry | None = None,
) -> str | None:
    """Register a backtest or labs result artifact in the SQLite experiment ledger."""
    if os.environ.get("DECISION_REGISTRY_RECORD_BACKTEST", "1").lower() in {
        "0",
        "false",
        "no",
    }:
        return None

    reg = registry or DecisionRegistry()
    metrics = _metrics_from_result_payload(data)
    bench = dict(benchmark_metrics or {})
    for alias, keys in (
        ("sharpe", ("baseline_sharpe", "benchmark_sharpe")),
        ("max_drawdown_pct", ("baseline_max_dd", "baseline_max_drawdown")),
        ("cagr_pct", ("baseline_cagr",)),
    ):
        if alias not in bench:
            for key in keys:
                raw = data.get(key)
                if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                    bench[alias] = float(raw)
                    break

    ts = _now_iso()
    record = ExperimentRecord(
        experiment_id=experiment_id,
        timestamp_utc=ts,
        name=name or experiment_id,
        hypothesis=hypothesis,
        git_sha=_git_sha_short(),
        data_start=str(data.get("start_date") or data.get("data_start") or "") or None,
        data_end=str(data.get("end_date") or data.get("data_end") or "") or None,
        parameters={
            k: data[k]
            for k in ("base_weights", "config", "weights")
            if k in data and isinstance(data[k], (dict, list, str, int, float))
        },
        metrics=metrics,
        benchmark_metrics=bench,
        promotion_status="candidate",
        artifacts={
            "output_path": str(output_path) if output_path else None,
            "has_provenance": "_provenance" in data,
            "has_data_snapshot": "_data_snapshot" in data,
        },
        tags=list(tags or []) + ["backtest_save"],
    )
    promo = evaluate_promotion_candidate(record)
    extras_rejection = ",".join(promo["failures"][:5]) if promo["failures"] else None
    if extras_rejection:
        record = record.model_copy(update={"rejection_reason": extras_rejection})

    try:
        return reg.record_experiment(record)
    except (sqlite3.Error, OSError, ValueError) as exc:
        logger.warning("Backtest experiment registry record skipped: %s", exc)
        return None


def sync_labs_registry_experiments(
    labs_registry_path: str | Path,
    *,
    registry: DecisionRegistry | None = None,
    limit: int = 50,
) -> int:
    """Import rows from offline labs_registry.json into SQLite experiments table."""
    path = Path(labs_registry_path)
    if not path.exists():
        return 0
    with open(path) as f:
        payload = json.load(f)
    rows = payload.get("experiments") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return 0

    reg = registry or DecisionRegistry()
    written = 0
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        exp_id = str(row.get("experiment_id") or "")
        if not exp_id:
            continue
        metrics_raw = row.get("metrics") or {}
        metrics = {
            k: float(v)
            for k, v in metrics_raw.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        registry_status = row.get("status") or "candidate"
        status = registry_status
        if status not in {"candidate", "shadow", "promoted", "rejected", "archived"}:
            status = "candidate"
        artifact_path = row.get("artifact_path")
        is_fixture = _is_fixture_or_temp_experiment(
            exp_id,
            {"artifact_path": artifact_path},
        )
        artifacts = {
            "artifact_path": None if is_fixture else artifact_path,
            "registry_status": "rejected" if is_fixture else registry_status,
            "provenance_status": "invalid" if is_fixture else row.get("provenance_status"),
        }
        tags = ["labs_registry"]
        rejection_reason = None
        if is_fixture:
            status = "rejected"
            artifacts["temp_path_redacted"] = True
            artifacts["synthetic_or_test_only"] = True
            tags.append("synthetic_or_test_only")
            rejection_reason = "synthetic_or_test_only"
        record = ExperimentRecord(
            experiment_id=exp_id,
            timestamp_utc=str(payload.get("generated_at") or _now_iso()),
            name=exp_id,
            hypothesis="" if is_fixture else str(artifact_path or ""),
            metrics=metrics,
            promotion_status=status,  # type: ignore[arg-type]
            rejection_reason=rejection_reason,
            artifacts=artifacts,
            tags=tags,
        )
        reg.record_experiment(record)
        written += 1
    return written
