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

from pydantic import BaseModel, ConfigDict, Field

from src.paths import DATA_DIR, PROJECT_ROOT, PUBLIC_DATA_DIR, sqlite_connect

logger = logging.getLogger(__name__)

DECISION_REGISTRY_SCHEMA_VERSION = "decision-registry/v1"
DECISION_REGISTRY_DB = Path(
    os.environ.get("DECISION_REGISTRY_DB", str(DATA_DIR / "decision_registry.db"))
)
DECISION_REGISTRY_JSON = "decision_registry.json"

PromotionStatus = Literal["candidate", "shadow", "promoted", "rejected", "archived"]

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
    extras: dict[str, Any] = Field(default_factory=dict)


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
        rejected = experiment.promotion_status == "rejected"
    else:
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
        rejected = experiment.get("promotion_status") == "rejected"

    if benchmark_metrics:
        bench.update({k: float(v) for k, v in benchmark_metrics.items()})

    min_sharpe_delta = float(os.environ.get("PROMOTION_MIN_SHARPE_DELTA", "0.02"))
    max_dd_worse = float(os.environ.get("PROMOTION_MAX_DD_WORSE_PCT", "2.0"))
    max_cvar_worse = float(os.environ.get("PROMOTION_MAX_CVAR_WORSE", "0.05"))
    max_turnover = float(os.environ.get("PROMOTION_MAX_TURNOVER", "1.5"))
    require_wfe = os.environ.get("PROMOTION_REQUIRE_WFE", "0").lower() in {"1", "true", "yes"}

    failures: list[str] = []
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

    recommended: PromotionStatus = "promoted" if not failures else "candidate"
    if failures and rejected:
        recommended = "rejected"

    return {
        "experiment_id": exp_id,
        "recommended_status": recommended,
        "pass": not failures,
        "failures": failures,
        "thresholds": {
            "min_sharpe_delta": min_sharpe_delta,
            "max_dd_worse_pct": max_dd_worse,
            "max_cvar_worse": max_cvar_worse,
            "max_turnover": max_turnover,
            "require_wfe": require_wfe,
        },
    }


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
    promotion_rows = [evaluate_promotion_candidate(exp) for exp in experiments[:10]]

    return {
        "schema_version": DECISION_REGISTRY_SCHEMA_VERSION,
        "generated_at": generated_at or _now_iso(),
        "recent_decisions": [d.model_dump(mode="json") for d in decisions],
        "recent_experiments": [e.model_dump(mode="json") for e in experiments],
        "replay_summaries": replay_summaries,
        "promotion_evaluations": promotion_rows,
        "counts": {
            "decisions": len(decisions),
            "experiments": len(experiments),
        },
    }


def publish_decision_registry_json(
    public_dir: str | Path = PUBLIC_DATA_DIR,
    *,
    registry: DecisionRegistry | None = None,
) -> Path:
    snapshot = build_decision_registry_snapshot(registry=registry)
    out = Path(public_dir) / DECISION_REGISTRY_JSON
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)
    logger.info(
        "Decision registry JSON written: %s (%d decisions)",
        out,
        snapshot["counts"]["decisions"],
    )
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
) -> str | None:
    """Record one strategy-evaluator rebalance / hold / kill-switch decision."""
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
        return reg.record_decision(record)
    except (sqlite3.Error, OSError, ValueError) as exc:
        logger.warning("Evaluator decision registry record skipped: %s", exc)
        return None


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
        status = row.get("status") or "candidate"
        if status not in {"candidate", "shadow", "promoted", "rejected", "archived"}:
            status = "candidate"
        record = ExperimentRecord(
            experiment_id=exp_id,
            timestamp_utc=str(payload.get("generated_at") or _now_iso()),
            name=exp_id,
            hypothesis=str(row.get("artifact_path") or ""),
            metrics=metrics,
            promotion_status=status,  # type: ignore[arg-type]
            artifacts={
                "artifact_path": row.get("artifact_path"),
                "provenance_status": row.get("provenance_status"),
            },
            tags=["labs_registry"],
        )
        reg.record_experiment(record)
        written += 1
    return written