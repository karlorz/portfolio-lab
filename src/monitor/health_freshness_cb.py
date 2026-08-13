"""Health-check freshness / circuit-breaker / graduation-CB cluster
(extracted from src/monitor/health_check.py, Item 5 s1).
"""

import logging
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from src.paths import DATA_DIR, PUBLIC_DATA_DIR
from src.monitor.hermes_cron import (
    combine_scheduler_backends,
    load_hermes_portfolio_cron_jobs,
    load_local_cron_jobs,
    resolve_hermes_cron_jobs_path,
    rollup_failed_cron_jobs,
)

logger = logging.getLogger(__name__)

def _resolve_freshness_public_root() -> Path:
    """Operator public SoT for freshness probes (Batch HX).

    Under pytest, honor the hermetic ``PUBLIC_DATA_DIR`` isolation tree.
    Outside pytest, never probe an ephemeral plab-pytest path — fall back to
    the live WWW tree so private ``data/health.json`` cannot stamp
    ``signals: missing`` while live authority JSON is present.
    """
    root = Path(PUBLIC_DATA_DIR)
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return root
    try:
        from src.monitor.signal_authority import is_ephemeral_write_path
        from src.paths import DEFAULT_LIVE_PUBLIC_DATA_DIR

        if is_ephemeral_write_path(root):
            live = Path(DEFAULT_LIVE_PUBLIC_DATA_DIR)
            if live.is_dir():
                logger.warning(
                    "freshness public root ephemeral %s → live %s", root, live
                )
                return live
    except Exception:  # noqa: BLE001 — keep PUBLIC_DATA_DIR on import/path failure
        pass
    return root


def _freshness_artifact_check(
    *,
    basenames: tuple[str, ...],
    roots: list[Path],
    stale_hours: float,
) -> dict[str, Any]:
    """Return freshness status for the first existing basename under roots.

    Batch HX: prefer public SoT, then private DATA_DIR twin (signals multi-dest
    writes both; prices may lag private). Never report missing when a non-
    ephemeral twin holds the live authority artifact.
    """
    now = datetime.now(timezone.utc)
    for idx, root in enumerate(roots):
        if root is None:
            continue
        try:
            from src.monitor.signal_authority import is_ephemeral_write_path

            # Outside pytest, skip *secondary* ephemeral roots (e.g. leftover
            # plab isolation). Always probe the first (resolved public SoT)
            # root even if the test harness placed it under /tmp — production
            # live WWW is non-ephemeral; first root is already healed by
            # _resolve_freshness_public_root.
            if (
                idx > 0
                and not os.environ.get("PYTEST_CURRENT_TEST")
                and is_ephemeral_write_path(root)
            ):
                continue
        except Exception:  # noqa: BLE001
            pass
        for name in basenames:
            path = Path(root) / name
            if not path.is_file():
                continue
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            age_hours = (now - mtime).total_seconds() / 3600
            return {
                "status": "ok" if age_hours < float(stale_hours) else "stale",
                "age_hours": round(age_hours, 1),
                "last_updated": mtime.isoformat(),
                "path": str(path),
            }
    return {"status": "missing", "age_hours": None, "last_updated": None}


def _check_data_freshness() -> dict:
    """Check how fresh the price data and signal data are."""
    from src.monitor.health_check import _DEFAULT_DATA_DIR, _backend_summary_excluding_health_self, _should_include_hermes_audit, check_scheduler_drift
    checks = {}
    public_root = _resolve_freshness_public_root()
    private_root = Path(DATA_DIR)

    # Price data freshness (public SoT first; private twin optional fallback)
    checks["prices"] = _freshness_artifact_check(
        basenames=("prices.json",),
        roots=[public_root, private_root],
        stale_hours=24.0,
    )

    # Signal data freshness — live authority artifact. Prefer public twin, then
    # private DATA_DIR/signals.json (multi-dest SSOT). Avoid false missing when
    # only the private twin is visible to the health process.
    checks["signals"] = _freshness_artifact_check(
        basenames=("signals.json",),
        roots=[public_root, private_root],
        stale_hours=4.0,
    )

    # Cron status
    cron_path = DATA_DIR / "cron_status.json"
    local_jobs, local_backend = load_local_cron_jobs(cron_path)
    jobs = list(local_jobs)

    hermes_path = None
    if _should_include_hermes_audit(local_backend):
        hermes_path = resolve_hermes_cron_jobs_path(
            current_data_dir=DATA_DIR,
            default_data_dir=_DEFAULT_DATA_DIR,
        )
    hermes_backend: dict[str, Any] | None = None
    if hermes_path is not None:
        hermes_jobs, hermes_backend = load_hermes_portfolio_cron_jobs(hermes_path)
        jobs.extend(hermes_jobs)

    # Rebuild backend summaries with self-job failures excluded from rollup.
    scheduler_backends: dict[str, dict[str, Any]] = {
        str(local_backend.get("backend", "local")): _backend_summary_excluding_health_self(
            local_backend, jobs
        ),
    }
    if hermes_backend is not None:
        scheduler_backends["hermes"] = _backend_summary_excluding_health_self(
            hermes_backend, jobs
        )

    scheduler_status = combine_scheduler_backends(scheduler_backends)
    scheduler_drift = check_scheduler_drift(scheduler_status["backends"])
    failed = rollup_failed_cron_jobs(jobs)
    backend_error = any(
        backend.get("status") == "error" for backend in scheduler_backends.values()
    )
    adjusted_local = scheduler_backends.get(
        str(local_backend.get("backend", "local")), local_backend
    )
    if scheduler_drift["status"] == "critical":
        cron_status = "error"
    elif backend_error:
        cron_status = "error"
    elif adjusted_local.get("status") == "unavailable" and len(scheduler_backends) == 1:
        cron_status = "missing"
    elif scheduler_status["status"] in {"unavailable", "warning"}:
        cron_status = "warning"
    else:
        cron_status = scheduler_status["status"]
    checks["cron"] = {
        "status": cron_status,
        "total_jobs": len(jobs),
        "failed_jobs": len(failed),
        "backends": scheduler_status["backends"],
        "jobs": jobs,
        "scheduler_drift": scheduler_drift,
    }

    return checks


def _check_circuit_breaker() -> dict:
    """Check broker circuit breaker state."""
    try:
        from src.broker.circuit_breaker import get_circuit_state
        state = get_circuit_state()
        return {
            "status": "ok" if state["state"] == "closed" else "degraded",
            "state": state["state"],
            "fail_count": state["fail_count"],
            "reset_timeout": state["reset_timeout"],
        }
    except ImportError:
        return {"status": "unavailable", "state": None, "fail_count": None, "reset_timeout": None}


def update_graduation_circuit_breaker_state(
    *,
    system_status: str,
    broker_circuit: dict | None = None,
    data_dir: Path | None = None,
    signal_health: dict | None = None,
) -> dict[str, Any]:
    """Persist ``DATA_DIR/.circuit_breaker.json`` consecutive_ok for graduation.

    Graduation's circuit_breaker_confidence gate needs a producer that increments
    ``consecutive_ok`` when ops are green and the broker CB is closed. Without
    this file the gate stays failed forever even when health looks healthy.

    - Healthy + broker closed → consecutive_ok += 1 (capped at 30)
    - Batch CB / EL: when ``signal_health`` is provided and contribution is
      **degraded or critical** (e.g. SH healthy==0 of N tracked), **hold** the
      streak (do not climb) so CB cannot greenwash a 0/N signal fleet while
      ops-only status is ok. Soft ``warning`` contribution (partial healthy
      sleeves) does **not** freeze climb — it only demotes dashboard
      system_status. Reset-to-zero still applies for broker open / ops red.
    - Otherwise (ops/broker fail) → consecutive_ok reset to 0
    """
    root = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    path = root / ".circuit_breaker.json"
    broker = broker_circuit if isinstance(broker_circuit, dict) else {}
    broker_state = str(broker.get("state") or "").lower()
    broker_fail = int(broker.get("fail_count") or 0)
    status_l = str(system_status or "").lower()
    ops_ok = status_l in {"ok", "healthy", "green"}
    broker_ok = broker_state in {"closed", "ok", "green", ""} and broker_fail == 0
    # Empty broker_state (unavailable import) still allows ops-only streak so
    # labs without pybreaker can graduate paper CB; open/half-open block.
    if broker_state in {"open", "half-open", "half_open"}:
        broker_ok = False

    # Batch CB: optional SH quality gate (max-severity readiness)
    sh_blocked = False
    sh_contrib: str | None = None
    if signal_health is not None:
        try:
            from src.dashboard.health_report import signal_health_status_contribution

            sh_contrib = signal_health_status_contribution(
                signal_health if isinstance(signal_health, dict) else None
            )
            # Batch EL: freeze climb only on hard quality outages
            # (0/N healthy → degraded/critical). Soft ``warning`` (e.g. 1/9
            # healthy + majority degraded, or majority unhealthy with ≥1
            # healthy) demotes dashboard system_status but must not freeze
            # graduation consecutive_ok forever — that starves the
            # circuit_breaker_confidence gate while ops stays green.
            if sh_contrib in {"degraded", "critical"}:
                sh_blocked = True
        except Exception:  # noqa: BLE001 — never fail CB on SH import
            sh_contrib = None

    prev: dict[str, Any] = {}
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                prev = raw
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Failed to read graduation CB state %s: %s", path, exc)

    prev_ok = int(prev.get("consecutive_ok") or 0)
    broker_open = broker_state in {"open", "half-open", "half_open"}
    if broker_open or (not broker_ok and not ops_ok):
        # Hard fail: open broker, or both ops and broker bad
        consecutive_ok = 0
        status = "red" if broker_open else "yellow"
        trips = int(prev.get("trips") or 0)
        if broker_open:
            trips = trips + 1
    elif sh_blocked:
        # Batch CB: SH quality outage freezes climb even if ops is warning
        # (lab FRED/cron dims must not wipe streak while SH 0/N is the gate).
        consecutive_ok = prev_ok
        status = "yellow"
        trips = int(prev.get("trips") or 0)
    elif ops_ok and broker_ok:
        consecutive_ok = min(prev_ok + 1, 30)
        status = "green"
        trips = 0
    else:
        # Ops warning/degraded without SH block → reset streak
        consecutive_ok = 0
        status = "red" if broker_open else "yellow"
        trips = int(prev.get("trips") or 0)
        if broker_open:
            trips = trips + 1

    payload: dict[str, Any] = {
        "schema_version": "graduation-circuit-breaker/v1",
        "status": status,
        "consecutive_ok": consecutive_ok,
        "trips": trips,
        "broker_state": broker_state or None,
        "broker_fail_count": broker_fail,
        "system_status": status_l or None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "producer": "src.monitor.health_check.update_graduation_circuit_breaker_state",
    }
    if sh_blocked:
        payload["signal_health_blocked"] = True
        if sh_contrib:
            payload["signal_health_contribution"] = sh_contrib
    try:
        root.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        logger.info(
            "Graduation CB state: consecutive_ok=%s status=%s → %s",
            consecutive_ok,
            status,
            path,
        )
    except OSError as exc:
        logger.warning("Failed to write graduation CB state %s: %s", path, exc)
    # Batch EM: dual-project SSOT onto monitor health.json so out-of-band CB
    # updates (or EL-style producer-only climbs) cannot leave sticky
    # graduation_circuit_breaker at consecutive_ok=0 / yellow while SSOT is green.
    try:
        reconcile_graduation_cb_projection(data_dir=root)
    except Exception as proj_exc:  # noqa: BLE001 — never fail CB write on project
        logger.warning("Graduation CB health re-projection skipped: %s", proj_exc)
    return payload


def load_graduation_cb_ssot(
    data_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Load private graduation CB SSOT from ``DATA_DIR/.circuit_breaker.json``."""
    root = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    path = root / ".circuit_breaker.json"
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError, UnicodeDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def project_graduation_cb_onto_report(
    report: dict[str, Any] | None,
    *,
    data_dir: Path | None = None,
    ssot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project ``.circuit_breaker.json`` SSOT onto ``graduation_circuit_breaker``.

    Batch EM (deep-research dual-surface consistency): every materialize of
    monitor health must re-read private SSOT rather than trust sticky
    ``health.json`` fields written by an older job. Mirrors Batch EI disk
    load of rebalance_health for timeline SLIs.
    """
    if not isinstance(report, dict):
        report = {}
    blob = ssot if isinstance(ssot, dict) else load_graduation_cb_ssot(data_dir)
    if blob is None:
        existing = report.get("graduation_circuit_breaker")
        if isinstance(existing, dict):
            existing = dict(existing)
            existing.setdefault("graduation_cb_source", "missing")
            report["graduation_circuit_breaker"] = existing
        else:
            report["graduation_circuit_breaker"] = {
                "graduation_cb_source": "missing",
            }
        return report

    report["graduation_circuit_breaker"] = {
        "consecutive_ok": blob.get("consecutive_ok"),
        "status": blob.get("status"),
        "updated_at": blob.get("updated_at"),
        "signal_health_blocked": bool(blob.get("signal_health_blocked")),
        "signal_health_contribution": blob.get("signal_health_contribution"),
        "system_status": blob.get("system_status"),
        "graduation_cb_source": "disk_ssot",
        "schema_version": blob.get("schema_version"),
        "producer": blob.get("producer"),
    }
    return report


def project_graduation_cb_onto_compact_health(
    health: dict[str, Any] | None,
    *,
    data_dir: Path | None = None,
    ssot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project graduation CB SSOT onto compact ``signals.json#health`` keys."""
    if not isinstance(health, dict):
        health = {}
    blob = ssot if isinstance(ssot, dict) else load_graduation_cb_ssot(data_dir)
    if blob is None:
        health.setdefault("graduation_cb_source", "missing")
        return health
    health["graduation_circuit_breaker_status"] = blob.get("status")
    health["graduation_circuit_breaker_consecutive_ok"] = blob.get("consecutive_ok")
    health["graduation_circuit_breaker_updated_at"] = blob.get("updated_at")
    health["graduation_circuit_breaker_signal_health_blocked"] = bool(
        blob.get("signal_health_blocked")
    )
    health["graduation_circuit_breaker_signal_health_contribution"] = blob.get(
        "signal_health_contribution"
    )
    health["graduation_cb_source"] = "disk_ssot"
    return health


def reconcile_graduation_cb_projection(
    *,
    data_dir: Path | None = None,
    health_path: Path | None = None,
) -> bool:
    """Rewrite monitor ``health.json`` graduation_circuit_breaker from SSOT.

    Returns True when a write occurred (fields differed from disk SSOT).
    """
    from src.monitor.health_kill_surfaces import _atomic_write_json_path
    root = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    path = Path(health_path) if health_path is not None else (root / "health.json")
    if not path.is_file():
        return False
    ssot = load_graduation_cb_ssot(root)
    if ssot is None:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return False
    if not isinstance(payload, dict):
        return False

    prev = payload.get("graduation_circuit_breaker")
    prev = prev if isinstance(prev, dict) else {}
    projected = project_graduation_cb_onto_report({}, data_dir=root, ssot=ssot)
    new_block = projected.get("graduation_circuit_breaker") or {}
    # Compare operator-visible fields only
    keys = (
        "consecutive_ok",
        "status",
        "updated_at",
        "signal_health_blocked",
        "signal_health_contribution",
    )
    same = all(prev.get(k) == new_block.get(k) for k in keys)
    if same and prev.get("graduation_cb_source") == "disk_ssot":
        return False

    payload["graduation_circuit_breaker"] = new_block
    payload["graduation_cb_reconciled_at"] = datetime.now(timezone.utc).isoformat()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json_path(path, payload)
    except OSError as exc:
        logger.warning(
            "Failed to reconcile graduation CB projection at %s: %s", path, exc
        )
        return False
    logger.info(
        "Reconciled graduation CB projection at %s (consecutive_ok=%s status=%s)",
        path,
        new_block.get("consecutive_ok"),
        new_block.get("status"),
    )
    return True
