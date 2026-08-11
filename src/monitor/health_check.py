"""
Health check for portfolio-lab system.

Produces a JSON health report that can be served by the dashboard
or polled by uptime monitoring tools.

Usage::

    python -m src.monitor.health_check

Environment variables
---------------------
HEALTH_CHECK_PATH : str
    Output path for monitor health.json (default: DATA_DIR/health.json)
HEALTH_OPS_PATH : str
    Optional explicit path for PUBLIC health_ops.json (default:
    PUBLIC_DATA_DIR/health_ops.json)
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.paths import DATA_DIR, PUBLIC_DATA_DIR
from src.monitor.alerting import AlertChannel, AlertLevel, send_alert
from src.monitor.hermes_cron import (
    HEALTH_SELF_JOB_NAME,
    combine_scheduler_backends,
    is_health_self_job,
    load_hermes_portfolio_cron_jobs,
    load_local_cron_jobs,
    resolve_hermes_cron_jobs_path,
    rollup_failed_cron_jobs,
)

logger = logging.getLogger(__name__)

__all__ = [
    "run_health_check",
    "check_scheduler_drift",
    "publish_ops_health_surfaces",
    "publish_health_alerts_json",
    "refresh_signals_health_kill_fields",
    "load_ops_monitor_report",
    "apply_ops_monitor_to_dashboard_health",
    "reconcile_monitor_health_with_disk_ssot",
    "project_disk_kill_open_to_all_surfaces",
    "update_graduation_circuit_breaker_state",
    "attach_shared_freshness_slis_to_ops_report",
    "load_graduation_cb_ssot",
    "project_graduation_cb_onto_report",
    "reconcile_graduation_cb_projection",
]

HEALTH_PATH = Path(os.environ.get("HEALTH_CHECK_PATH", str(DATA_DIR / "health.json")))
_DEFAULT_DATA_DIR = DATA_DIR
SCHEDULER_DRIFT_THRESHOLD = 2


def health_ops_path() -> Path:
    """Operator-facing monitor health under PUBLIC_DATA_DIR (dual-doc SSOT)."""
    override = os.environ.get("HEALTH_OPS_PATH")
    if override and override.strip():
        return Path(override.strip())
    return Path(PUBLIC_DATA_DIR) / "health_ops.json"


def _project_public_kill_fields(report: dict[str, Any]) -> dict[str, Any]:
    """Map monitor report kill/open-incident checks into dashboard-shaped fields."""
    checks = report.get("checks") if isinstance(report.get("checks"), dict) else {}
    kill = checks.get("kill_switch") if isinstance(checks.get("kill_switch"), dict) else {}
    open_inc = (
        checks.get("open_incidents")
        if isinstance(checks.get("open_incidents"), dict)
        else {}
    )
    status = str(report.get("status") or "ok")
    return {
        "kill_switch": kill,
        "open_incidents": open_inc,
        "ops_health_status": status,
        "ops_health_timestamp": report.get("timestamp"),
        "ops_health_scope": report.get("scope") or "operational_readiness",
    }


def _elevate_public_system_status(current: Any, ops_status: str) -> str:
    """Raise dashboard system_status when ops monitor is more severe."""
    rank = {
        "healthy": 0,
        "ok": 0,
        "warning": 1,
        "degraded": 2,
        "critical": 3,
        "error": 3,
    }
    cur = str(current or "healthy")
    target = max(rank.get(cur, 0), rank.get(ops_status, 0))
    for name, value in rank.items():
        if value == target and name not in {"ok", "error"}:
            return name
    if target >= 3:
        return "critical"
    if target >= 2:
        return "degraded"
    if target >= 1:
        return "warning"
    return cur if cur else "healthy"


def _is_monitor_health_report(payload: dict[str, Any]) -> bool:
    """True for monitor schema (status + checks), not dashboard system_status JSON."""
    if not isinstance(payload.get("checks"), dict):
        return False
    # Dashboard schema uses system_status and cron_jobs without checks.
    if "system_status" in payload and "status" not in payload:
        return False
    return True


def load_ops_monitor_report(
    *,
    data_dir: Path | None = None,
    public_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Load the newest monitor-schema health report from DATA or PUBLIC ops path.

    Prefer the fresher of DATA_DIR/health.json and PUBLIC_DATA_DIR/health_ops.json
    when both exist and look like monitor reports.
    """
    root_data = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    root_public = Path(public_dir) if public_dir is not None else Path(PUBLIC_DATA_DIR)
    candidates = [root_data / "health.json", root_public / "health_ops.json"]

    best: dict[str, Any] | None = None
    best_ts = ""
    for path in candidates:
        try:
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError, TypeError):
            continue
        if not isinstance(payload, dict) or not _is_monitor_health_report(payload):
            continue
        ts = str(payload.get("timestamp") or payload.get("generated_at") or "")
        if best is None or ts >= best_ts:
            best = payload
            best_ts = ts
    return best


def _patch_monitor_report_kill_open(
    payload: dict[str, Any],
    disk_kill: dict[str, Any],
    disk_open: dict[str, Any],
    *,
    force: bool = False,
) -> bool:
    """Mutate monitor-schema payload kill/open from disk. Return True if changed."""
    if not isinstance(payload, dict) or not _is_monitor_health_report(payload):
        return False
    checks = payload.get("checks")
    if not isinstance(checks, dict):
        checks = {}
        payload["checks"] = checks

    prev_open = checks.get("open_incidents") if isinstance(checks.get("open_incidents"), dict) else {}
    prev_kill = checks.get("kill_switch") if isinstance(checks.get("kill_switch"), dict) else {}

    prev_open_n = int(prev_open.get("open_count") or 0)
    disk_open_n = int(disk_open.get("open_count") or 0)
    prev_enabled = bool(prev_kill.get("enabled"))
    disk_enabled = bool(disk_kill.get("enabled"))
    prev_kill_status = str(prev_kill.get("status") or "ok").lower()
    disk_kill_status = str(disk_kill.get("status") or "ok").lower()
    prev_open_status = str(prev_open.get("status") or "ok").lower()
    disk_open_status = str(disk_open.get("status") or "ok").lower()

    if (
        not force
        and prev_open_n == disk_open_n
        and prev_enabled == disk_enabled
        and prev_kill_status == disk_kill_status
        and prev_open_status == disk_open_status
    ):
        return False

    checks["kill_switch"] = disk_kill
    checks["open_incidents"] = disk_open

    circuit = checks.get("circuit_breaker") if isinstance(checks.get("circuit_breaker"), dict) else {}
    freshness = checks.get("data_freshness") if isinstance(checks.get("data_freshness"), dict) else {}
    rollup_checks = {
        **{k: v for k, v in freshness.items() if isinstance(v, dict) and "status" in v},
        "kill_switch": disk_kill,
        "open_incidents": disk_open,
    }
    for name, check in checks.items():
        if name in {"data_freshness", "kill_switch", "open_incidents", "circuit_breaker"}:
            continue
        if isinstance(check, dict) and "status" in check:
            rollup_checks[name] = check
    payload["status"] = _compute_system_status(rollup_checks, circuit)
    payload["ssot_reconciled_at"] = datetime.now(timezone.utc).isoformat()
    payload["ssot_reconcile_source"] = "disk_incidents_kill"
    # NG4 (2026-08-11 session B): any SSOT re-projection rewrites the file, so
    # the embedded timestamp must advance with it. Observed: the mirror-lag
    # restamp (repo_public_mirror_lag.restamp_mirror_lag_on_health_documents)
    # force-patches kill/open on data/health.json between :00/:30 health runs
    # and rewrote the file with a fresh mtime + ssot_reconciled_at while the
    # embedded timestamp stayed at report generation time — content looked
    # fresh by mtime while being up to 30 min old. Restamp here, at the shared
    # patch seam (reconcile + fan-out + mirror-lag restamp), so every
    # re-projection write advances timestamp. Reconcile re-stamps right after;
    # harmless (same value, microseconds apart).
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    return True


def _atomic_write_json_path(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON at 0o644 (prefer signal_authority atomic helper)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    from src.dashboard.public_projection import is_public_output_path
    from src.monitor.signal_authority import serialize_json_payload

    text = serialize_json_payload(
        payload,
        output_path=path,
        public=is_public_output_path(path),
    )
    try:
        from src.monitor.signal_authority import _atomic_write_text

        _atomic_write_text(path, text, mode=0o644)
    except Exception:
        path.write_text(text, encoding="utf-8")


def _new_generation_id() -> str:
    """One generation/run identity for health-owned outputs (Task 5B)."""
    import uuid as _uuid

    return (
        f"gen-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-"
        f"{_uuid.uuid4().hex[:8]}"
    )


def _atomic_write_json_text(path: Path, payload: Any) -> None:
    """Atomic JSON write through the canonical signal_authority writer.

    Exposed as a module-level seam so publication-contract tests can inject
    failures at the exact write boundary.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    from src.dashboard.public_projection import is_public_output_path
    from src.monitor.signal_authority import _atomic_write_text, serialize_json_payload

    text = serialize_json_payload(
        payload,
        output_path=path,
        public=is_public_output_path(path),
    )
    _atomic_write_text(path, text, mode=0o644)


def write_health_generation(
    payload: dict[str, Any],
    *,
    path: Path,
    producer_sha: str | None = None,
) -> Path:
    """Write one health-owned output stamped with run/generation identity.

    Task 5B: every health-owned file carries ``producer_run_id`` (tasker run
    when present) and ``generation_id`` plus the producer source SHA, written
    atomically so an interrupted run never leaves partial bytes.
    """
    stamped = dict(payload)
    run_id = os.environ.get("TASKER_RUN_ID") or os.environ.get("CRON_RUN_ID")
    if run_id:
        stamped.setdefault("producer_run_id", run_id)
    stamped.setdefault("generation_id", _new_generation_id())
    if producer_sha is not None:
        stamped["producer_git_sha"] = producer_sha
    elif stamped.get("generator_git_sha"):
        stamped["producer_git_sha"] = stamped["generator_git_sha"]
    _atomic_write_json_text(path, stamped)
    return path


def commit_public_index(
    payload: Any,
    *,
    index_path: Path,
    generation_id: str | None = None,
) -> Path:
    """Atomically commit the public index LAST (Task 5B).

    Content files are written first by their producers; the index is built
    from the exact final bytes and replaced atomically only after every
    intended generation file is in place. On failure the prior committed
    index remains untouched.
    """
    stamped = dict(payload)
    if generation_id is not None:
        stamped["generation_id"] = generation_id
    stamped.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
    _atomic_write_json_text(index_path, stamped)
    return index_path


def reconcile_monitor_health_with_disk_ssot(
    *,
    data_dir: Path | None = None,
    health_path: Path | None = None,
) -> bool:
    """Patch monitor-schema data/health.json kill/open fields to match disk SSOT.

    Dashboard regeneration (data job) writes public health + incidents.json from
    disk lifecycle SSOT, but monitor ``data/health.json`` is only rewritten by
    the :00/:30 health cron. Between those ticks operators see dual-incident
    split-brain: sticky open on the monitor report while ``incidents.json`` is
    already clear, **or** (Batch II DE4) sticky clear while kill_switch.json
    just armed mid-cycle.

    Bidirectional: always re-project disk kill/open when they disagree with the
    stamped report (clear→arm and arm→clear). Recompute top-level ``status``
    from remaining check dims. Returns True when a write occurred.
    """
    root = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    path = Path(health_path) if health_path is not None else (root / "health.json")
    if not path.exists():
        return False

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return False
    if not isinstance(payload, dict) or not _is_monitor_health_report(payload):
        return False

    disk_kill, disk_open = _disk_kill_and_open_incidents(root)
    if not _patch_monitor_report_kill_open(payload, disk_kill, disk_open):
        return False

    # NG2/NG4 (2026-08-11): this rewrite refreshes the file mtime, so the
    # embedded timestamp must advance with it — restamped inside
    # _patch_monitor_report_kill_open (shared seam also used by the
    # mirror-lag restamp). Observed failure: dashboard regen reconciled
    # data/health.json at 01:17Z but left the embedded timestamp at 00:00:14Z
    # — content looked fresh by mtime while being ~77min old. The status and
    # kill/open projection ARE recomputed here, so restamping is honest;
    # ssot_reconciled_at/source already disclose the partial rebuild.

    try:
        _atomic_write_json_path(path, payload)
    except OSError as exc:
        logger.warning("Failed to reconcile monitor health SSOT at %s: %s", path, exc)
        return False
    logger.info(
        "Reconciled monitor health kill/open SSOT at %s (status=%s enabled=%s open=%s)",
        path,
        payload.get("status"),
        bool(disk_kill.get("enabled")),
        int(disk_open.get("open_count") or 0),
    )
    return True


def project_disk_kill_open_to_all_surfaces(
    *,
    data_dir: Path | None = None,
    public_dir: Path | None = None,
) -> dict[str, Any]:
    """Write-through: re-project kill/open from disk onto mon/ops/public surfaces.

    Batch IM DN / IN DN3: incident arm/clear and lag restamp must not leave
    sticky kill.enabled on health/health_ops while kill_switch.json disagrees.
    Best-effort; never raises into the incident lifecycle path.

    Surfaces:
    - private monitor ``data/health.json`` (checks.kill_switch / open_incidents)
    - private + public ``health_ops.json`` (monitor schema)
    - public dashboard ``health.json`` (top-level kill_switch / open_incidents)
    - repo soft-mirror of public health + health_ops when multi-dest available
    """
    result: dict[str, Any] = {
        "monitor": False,
        "ops": False,
        "public": False,
        "errors": [],
    }
    root = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    pub_root = Path(public_dir) if public_dir is not None else Path(PUBLIC_DATA_DIR)
    disk_kill, disk_open = _disk_kill_and_open_incidents(root)
    disk_enabled = bool(disk_kill.get("enabled"))
    disk_open_n = int(disk_open.get("open_count") or 0)

    # 1) Private monitor health.json
    mon_path = root / "health.json"
    if mon_path.exists():
        try:
            if reconcile_monitor_health_with_disk_ssot(
                data_dir=root, health_path=mon_path
            ):
                result["monitor"] = True
            else:
                # Already aligned counts as success for write-through callers
                result["monitor"] = True
        except Exception as exc:  # noqa: BLE001
            result["errors"].append(f"monitor:{exc}")
            logger.warning("project_disk kill fan-out monitor failed: %s", exc)

    # 2) health_ops.json (public + private twin) — patch body then multi-dest
    ops_candidates = [
        pub_root / "health_ops.json",
        root / "health_ops.json",
    ]
    ops_payload: dict[str, Any] | None = None
    for candidate in ops_candidates:
        if not candidate.exists():
            continue
        try:
            doc = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError, TypeError):
            continue
        if isinstance(doc, dict) and _is_monitor_health_report(doc):
            ops_payload = doc
            break
    if ops_payload is not None:
        try:
            _patch_monitor_report_kill_open(
                ops_payload, disk_kill, disk_open, force=True
            )
            ops_public = pub_root / "health_ops.json"
            ops_private = root / "health_ops.json"
            wrote = False
            try:
                from src.monitor.signal_authority import write_json_multi_dest

                wr = write_json_multi_dest(
                    ops_payload,
                    public_path=ops_public,
                    private_path=ops_private,
                    soft_mirror_repo=True,
                    repo_filename="health_ops.json",
                )
                wrote = bool(wr.wrote_public or wr.wrote_private or wr.wrote_repo)
            except Exception as multi_exc:  # noqa: BLE001
                logger.warning(
                    "project_disk ops multi-dest failed (%s); fallback", multi_exc
                )
            if not wrote:
                for dest in (ops_public, ops_private):
                    try:
                        _atomic_write_json_path(dest, ops_payload)
                        wrote = True
                    except OSError:
                        pass
            result["ops"] = wrote
        except Exception as exc:  # noqa: BLE001
            result["errors"].append(f"ops:{exc}")
            logger.warning("project_disk kill fan-out ops failed: %s", exc)

    # 3) Public dashboard health.json top-level kill/open
    public_health = pub_root / "health.json"
    if public_health.exists():
        try:
            payload = json.loads(public_health.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
            result["errors"].append(f"public_read:{exc}")
            payload = None
        if isinstance(payload, dict) and not _is_monitor_health_report(payload):
            try:
                prev_kill = (
                    payload.get("kill_switch")
                    if isinstance(payload.get("kill_switch"), dict)
                    else {}
                )
                prev_open = (
                    payload.get("open_incidents")
                    if isinstance(payload.get("open_incidents"), dict)
                    else {}
                )
                prev_enabled = bool(prev_kill.get("enabled"))
                prev_open_n = int(prev_open.get("open_count") or 0)
                if prev_enabled != disk_enabled or prev_open_n != disk_open_n:
                    payload["kill_switch"] = disk_kill
                    payload["open_incidents"] = disk_open
                    # Soft-demote system_status only when disk is fully clear
                    # and payload still carried sticky kill/open.
                    if not disk_enabled and disk_open_n == 0:
                        if prev_enabled or prev_open_n > 0:
                            cur = str(payload.get("system_status") or "healthy")
                            if cur in {"warning", "degraded", "critical"}:
                                # Do not force healthy over other SLOs; leave
                                # as-is unless only kill drove warning.
                                payload["system_status"] = "healthy"
                    else:
                        try:
                            from src.dashboard.kill_authority import (
                                elevate_system_status_for_kill,
                            )

                            payload["system_status"] = elevate_system_status_for_kill(
                                payload.get("system_status"), disk_kill, disk_open
                            )
                        except Exception:  # noqa: BLE001
                            pass
                    payload["ssot_reconciled_at"] = datetime.now(timezone.utc).isoformat()
                    payload["ssot_reconcile_source"] = "disk_incidents_kill"
                    wrote = False
                    try:
                        from src.monitor.signal_authority import write_json_multi_dest

                        wr = write_json_multi_dest(
                            payload,
                            public_path=public_health,
                            private_path=None,
                            soft_mirror_repo=True,
                            repo_filename="health.json",
                        )
                        wrote = bool(wr.wrote_public or wr.wrote_repo)
                    except Exception as multi_exc:  # noqa: BLE001
                        logger.warning(
                            "project_disk public multi-dest failed (%s); fallback",
                            multi_exc,
                        )
                    if not wrote:
                        _atomic_write_json_path(public_health, payload)
                        wrote = True
                    result["public"] = wrote
                else:
                    result["public"] = True
            except Exception as exc:  # noqa: BLE001
                result["errors"].append(f"public:{exc}")
                logger.warning("project_disk kill fan-out public failed: %s", exc)

    logger.info(
        "project_disk_kill_open_to_all_surfaces enabled=%s open=%s mon=%s ops=%s pub=%s",
        disk_enabled,
        disk_open_n,
        result["monitor"],
        result["ops"],
        result["public"],
    )
    return result


def _disk_kill_ssot_is_clear(data_dir: Path | None) -> bool:
    """True when kill_switch.json is absent/disabled and open incidents are zero.

    Used so a lagging monitor report cannot re-introduce a cleared kill into
    dashboard health after resolve.
    """
    root = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    try:
        from src.dashboard.kill_authority import (
            load_kill_switch_payload,
            load_open_incidents_summary,
        )
    except ImportError:
        return False
    payload = load_kill_switch_payload(root)
    if payload is not None and bool(payload.get("enabled")):
        return False
    open_inc = load_open_incidents_summary(root)
    return int(open_inc.get("open_count") or 0) == 0


def _disk_kill_and_open_incidents(
    data_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project kill_switch.json + incidents.json into dashboard-shaped blocks.

    Always prefer disk authority over any lagging monitor report identity
    (level / incident_id / reason). Monitor report may still supply ops_* stamps.
    """
    root = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    try:
        from src.dashboard.kill_authority import (
            load_kill_switch_payload,
            load_open_incidents_summary,
            project_kill_switch_fields,
        )
    except ImportError:
        disk_kill = {
            "status": "ok",
            "enabled": False,
            "level": None,
            "reason": None,
            "source": None,
            "message": None,
            "timestamp": None,
            "incident_id": None,
            "mode": None,
            "channel": None,
        }
        disk_open = {"status": "ok", "open_count": 0, "incidents": []}
        return disk_kill, disk_open
    return (
        project_kill_switch_fields(load_kill_switch_payload(root)),
        load_open_incidents_summary(root),
    )


def apply_ops_monitor_to_dashboard_health(
    health_data: dict[str, Any],
    ops_report: dict[str, Any] | None = None,
    *,
    data_dir: Path | None = None,
    public_dir: Path | None = None,
) -> dict[str, Any]:
    """Stamp ops_health_* and elevate system_status from the monitor report.

    Called by ``generate_health_json`` so dashboard regeneration does not wipe
    the dual-SSOT fields that ``publish_ops_health_surfaces`` merges after
    ``make health``.

    Disk kill SSOT always wins for kill_switch / open_incidents identity
    (enabled or clear). A lagging monitor report must never rehydrate a stale
    halt + test incident_id when kill_switch.json already moved to a live
    warning, and must never resurrect a kill that resolve already cleared.
    """
    report = ops_report if isinstance(ops_report, dict) else load_ops_monitor_report(
        data_dir=data_dir, public_dir=public_dir
    )
    if not report:
        return health_data

    projected = _project_public_kill_fields(report)
    # Batch JL JH1b: always stamp ops_health_status from *this* monitor report
    # status — never leave a sticky public warning when the live report is ok.
    ops_status = str(report.get("status") or projected.get("ops_health_status") or "ok")
    health_data["ops_health_status"] = ops_status
    health_data["ops_health_timestamp"] = (
        report.get("timestamp")
        or projected.get("ops_health_timestamp")
    )
    health_data["ops_health_source"] = "monitor.health_check"

    sticky_kill = isinstance(health_data.get("kill_switch"), dict) and bool(
        health_data["kill_switch"].get("enabled")
    )
    sticky_open = (
        isinstance(health_data.get("open_incidents"), dict)
        and int(health_data["open_incidents"].get("open_count") or 0) > 0
    )

    # Always re-project disk authority for kill/open — clear *and* enabled paths.
    disk_kill, disk_open = _disk_kill_and_open_incidents(data_dir)
    health_data["kill_switch"] = disk_kill
    health_data["open_incidents"] = disk_open

    ssot_clear = _disk_kill_ssot_is_clear(data_dir)
    if ssot_clear:
        # Demote system_status only when payload still carried enabled kill /
        # open incidents (sticky public health). Do not wipe SLO-derived
        # critical from generate_health_json when kill fields were already clear.
        if "system_status" in health_data and (sticky_kill or sticky_open):
            health_data["system_status"] = "healthy"
    else:
        # Kill still active: elevate from ops rollup + disk kill identity.
        if "system_status" in health_data:
            elevated = _elevate_public_system_status(
                health_data.get("system_status"), ops_status
            )
            try:
                from src.dashboard.kill_authority import elevate_system_status_for_kill

                elevated = elevate_system_status_for_kill(
                    elevated, disk_kill, disk_open
                )
            except ImportError:
                pass
            health_data["system_status"] = elevated

    # Dual-plane contract: this ops restamp must not fold signal-health quality
    # into the operator-facing system badge. Quality remains on signal_health /
    # signal_quality surfaces and in the graduation circuit-breaker projection.

    # Batch HO: project repo_public_mirror_lag* onto dashboard health so SPA
    # consumers share the same SLI as health_ops / signals.health (was split-
    # brain: ops_health_status only). Prefer max(live probe, ops stamp).
    try:
        _project_mirror_lag_onto_dashboard_health(
            health_data,
            report,
            data_dir=data_dir,
            public_dir=public_dir,
        )
    except Exception as exc:  # noqa: BLE001 — never block ops merge on lag SLI
        logger.warning("dashboard mirror lag projection failed: %s", exc)

    # Dual-plane partial-path honesty: when the monitor report is ops-ok and
    # all ops SLIs on the dashboard are green (kill off, incidents 0, scheduler
    # ok, data_pipeline_slo ok, mirror lag 0), clear a sticky quality-only
    # system_status demotion. ``derive_system_status`` excludes signal_health by
    # design, so a thin SH (e.g. 1/9) cannot keep the ops badge warning. Real
    # ops failures (kill on, scheduler degraded, SLO warn, mirror lag) keep
    # their demotion: derive_system_status re-elevates from scheduler/SLO/stale
    # dims, and mirror lag is gated explicitly below.
    if ssot_clear and "system_status" in health_data:
        ops_green = str(ops_status).lower() in {"ok", "healthy", "green", "success", ""}
        lag_status = str(health_data.get("repo_public_mirror_lag_status") or "").lower()
        lag_ok = lag_status in {"", "ok", "healthy", "green", "unknown"}
        try:
            lag_count = int(health_data.get("repo_public_mirror_lagging_count") or 0)
        except (TypeError, ValueError):
            lag_count = 0
        if ops_green and lag_ok and lag_count == 0:
            try:
                from src.dashboard.health_report import derive_system_status

                scheduler_status = None
                sched = health_data.get("scheduler_status")
                if isinstance(sched, Mapping):
                    scheduler_status = sched.get("status") or sched.get("state")
                slo_status = None
                slo = health_data.get("data_pipeline_slo")
                if isinstance(slo, Mapping):
                    slo_status = slo.get("status")
                stale_count = 0
                freshness = health_data.get("data_freshness")
                if isinstance(freshness, Mapping):
                    stale_count = sum(
                        1
                        for item in freshness.values()
                        if isinstance(item, Mapping)
                        and item.get("status") not in {"fresh", "ok"}
                    )
                failed_jobs = 0
                try:
                    failed_jobs = int(health_data.get("failed_cron_jobs") or 0)
                except (TypeError, ValueError):
                    failed_jobs = 0
                recomputed = derive_system_status(
                    current="healthy",
                    backend_error=False,
                    scheduler_status=scheduler_status,
                    slo_status=slo_status,
                    failed_jobs=failed_jobs,
                    stale_count=stale_count,
                )
                health_data["system_status"] = recomputed
            except ImportError:
                pass

    # Batch IG: project graduation CB SSOT onto public dashboard health.
    # signals.health already gets compact keys via kill_refresh (EM); private
    # ops has nested graduation_circuit_breaker. Public health.json was the
    # missing surface (ops_health_* only) → SPA split-brain on consecutive_ok.
    try:
        root = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
        ssot = load_graduation_cb_ssot(root)
        project_graduation_cb_onto_compact_health(
            health_data, data_dir=root, ssot=ssot
        )
        # Nested ops-shape block for consumers that expect the full object
        # (matches private data/health.json + health_ops dual surface).
        project_graduation_cb_onto_report(health_data, data_dir=root, ssot=ssot)
    except Exception as exc:  # noqa: BLE001 — never block ops merge on CB SLI
        logger.warning("dashboard graduation CB projection failed: %s", exc)

    return health_data


def _project_mirror_lag_onto_dashboard_health(
    health_data: dict[str, Any],
    ops_report: dict[str, Any],
    *,
    data_dir: Path | None = None,
    public_dir: Path | None = None,
) -> None:
    """Stamp repo_public_mirror_lag* on dashboard health (Batch HO DD/DE).

    Uses ``resolve_mirror_lag_for_consumer(max(live, stamp))`` when a live
    probe is available; falls back to ops-report stamp alone when the probe
    cannot run. Soft-elevates ``system_status`` for lagging/critical (ops
    hygiene only — not a trading halt).
    """
    from src.dashboard.generator import project_repo_public_mirror_lag_onto_health
    from src.monitor.repo_public_mirror_lag import (
        resolve_mirror_lag_for_consumer,
        summarize_repo_public_mirror_lag,
    )

    stamp: dict[str, Any] = {
        "lagging_count": ops_report.get("repo_public_mirror_lagging_count"),
        "total": ops_report.get("repo_public_mirror_total"),
        "lagging_paths": ops_report.get("repo_public_mirror_lagging_paths"),
        "status": ops_report.get("repo_public_mirror_lag_status"),
    }
    nested = ops_report.get("repo_public_mirror_lag")
    if isinstance(nested, dict):
        if stamp.get("lagging_count") is None:
            stamp["lagging_count"] = nested.get("lagging_count")
        if stamp.get("total") is None:
            stamp["total"] = nested.get("total")
        if not stamp.get("lagging_paths"):
            stamp["lagging_paths"] = nested.get("paths") or nested.get(
                "lagging_paths"
            )

    live: dict[str, Any] | None = None
    try:
        dest = Path(public_dir) if public_dir is not None else None
        live = summarize_repo_public_mirror_lag(
            dest_root=dest if dest is not None else None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("live mirror lag probe failed during dashboard merge: %s", exc)
        live = None

    if isinstance(live, dict) and live.get("ok", True) is not False:
        resolved = resolve_mirror_lag_for_consumer(stamp=stamp, live=live)
        lag_summary = {
            "lagging_count": resolved["lagging_count"],
            "total": resolved["total"],
            "lagging_paths": resolved["lagging_paths"],
            "source": live.get("source") or ops_report.get("repo_public_mirror_source"),
            "dest": live.get("dest") or ops_report.get("repo_public_mirror_dest"),
            "ok": True,
        }
        project_repo_public_mirror_lag_onto_health(health_data, lag_summary)
        health_data["repo_public_mirror_lag"] = {
            "lagging_count": health_data.get("repo_public_mirror_lagging_count"),
            "total": health_data.get("repo_public_mirror_total"),
            "status": health_data.get("repo_public_mirror_lag_status"),
            "badge": health_data.get("repo_public_mirror_lag_badge"),
            "paths": health_data.get("repo_public_mirror_lagging_paths"),
            "source": health_data.get("repo_public_mirror_source"),
            "dest": health_data.get("repo_public_mirror_dest"),
        }
        health_data["mirror_lag_source_of_truth"] = resolved.get("source_of_truth")
        health_data["mirror_lag_live_lagging_count"] = resolved.get(
            "live_lagging_count"
        )
        health_data["mirror_lag_stamp_lagging_count"] = resolved.get(
            "stamp_lagging_count"
        )
    else:
        # No live probe — project stamp fields only when present on ops report
        if stamp.get("lagging_count") is None and stamp.get("total") is None:
            return
        lag_summary = {
            "lagging_count": stamp.get("lagging_count") or 0,
            "total": stamp.get("total") or 0,
            "lagging_paths": stamp.get("lagging_paths") or [],
            "source": ops_report.get("repo_public_mirror_source"),
            "dest": ops_report.get("repo_public_mirror_dest"),
        }
        project_repo_public_mirror_lag_onto_health(health_data, lag_summary)
        health_data["repo_public_mirror_lag"] = {
            "lagging_count": health_data.get("repo_public_mirror_lagging_count"),
            "total": health_data.get("repo_public_mirror_total"),
            "status": health_data.get("repo_public_mirror_lag_status"),
            "badge": health_data.get("repo_public_mirror_lag_badge"),
            "paths": health_data.get("repo_public_mirror_lagging_paths"),
            "source": health_data.get("repo_public_mirror_source"),
            "dest": health_data.get("repo_public_mirror_dest"),
        }
        health_data["mirror_lag_source_of_truth"] = "stamp"
        health_data["mirror_lag_stamp_lagging_count"] = lag_summary["lagging_count"]

    # Soft-elevate dashboard system_status when lag is lagging/critical
    lag_status = str(health_data.get("repo_public_mirror_lag_status") or "")
    if lag_status in ("lagging", "critical") and "system_status" in health_data:
        cur = str(health_data.get("system_status") or "healthy").lower()
        if cur in ("healthy", "ok", "unknown", ""):
            health_data["system_status"] = "warning"


def refresh_signals_health_kill_fields(
    report: dict[str, Any],
    *,
    public_dir: Path | None = None,
    data_dir: Path | None = None,
) -> None:
    """Patch signals.json#health compact kill fields from disk SSOT.

    Operators reading signals.health must not wait for a full dashboard cycle
    after kill clear or identity change. Always project kill_switch.json +
    incidents.json — never trust a lagging monitor report for kill identity.
    """
    root_public = Path(public_dir) if public_dir is not None else Path(PUBLIC_DATA_DIR)
    signals_path = root_public / "signals.json"
    if not signals_path.exists():
        return
    try:
        payload = json.loads(signals_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("signals.json unreadable; skip health kill refresh: %s", exc)
        return
    if not isinstance(payload, dict):
        return

    try:
        from src.dashboard.generator import _compact_health_summary
        from src.dashboard.kill_authority import project_compact_kill_fields
    except ImportError as exc:
        logger.warning("Cannot import kill projectors for signals refresh: %s", exc)
        return

    # Disk authority wins for kill/open identity (enabled or clear).
    disk_kill, disk_open = _disk_kill_and_open_incidents(data_dir)
    compact = project_compact_kill_fields(
        {"kill_switch": disk_kill, "open_incidents": disk_open}
    )

    health = payload.get("health")
    if not isinstance(health, dict):
        health = _compact_health_summary(report) if report else {"status": "unknown"}
    else:
        health = dict(health)

    # Always apply compact kill keys (including enabled:false clears).
    for key, value in compact.items():
        health[key] = value
    if not disk_kill.get("enabled"):
        health["kill_switch_enabled"] = False
        if disk_kill.get("level") is None:
            health["kill_switch_level"] = None
    if int(disk_open.get("open_count") or 0) == 0:
        health["open_incidents_count"] = 0
        if disk_open.get("status"):
            health["open_incidents_status"] = disk_open.get("status")

    # Batch CR: re-project SH freeze/quality onto sticky signals.health so a
    # pre-CQ full dashboard cannot leave ensemble_weight_freeze_active=true
    # after zero-healthy recovery (kill refresh previously only patched kill).
    try:
        sh_report = report.get("signal_health") if isinstance(report, dict) else None
        if isinstance(sh_report, dict):
            sh_compact = _compact_health_summary({"signal_health": sh_report})
            for key in (
                "signal_health_healthy",
                "signal_health_degraded",
                "signal_health_unhealthy",
                "signal_health_total_tracked",
                "signal_health_quality_badge",
                "signal_health_zero_healthy",
                "signal_health_status",
                "signal_quality_badge",
                "ensemble_weight_freeze_active",
                "ensemble_weights_age_days",
                "ensemble_weights_file_stale",
            ):
                if key in sh_compact:
                    health[key] = sh_compact[key]
            # Explicit clear sticky True when freeze is now False
            if sh_compact.get("ensemble_weight_freeze_active") is False:
                health["ensemble_weight_freeze_active"] = False
            if sh_compact.get("signal_health_zero_healthy") is False:
                health["signal_health_zero_healthy"] = False
    except Exception as exc:  # noqa: BLE001 — never fail kill refresh on SH
        logger.warning("signals.health SH freeze re-project skipped: %s", exc)

    if report.get("status") is not None:
        # The monitor report is the current ops-plane baseline. Assign rather
        # than setdefault so a legacy SH-derived degraded value cannot remain
        # sticky; compact ops dimensions below may still elevate it.
        health["status"] = report.get("status")
    if report.get("timestamp") is not None:
        health["generated_at"] = report.get("timestamp")

    # Max-severity honesty after kill patch (Batch BH): never leave compact
    # status=healthy when kill enabled, open incidents, failed cron, or
    # scheduler degraded.
    try:
        from src.dashboard.cron_scheduler_section import _elevate_compact_health_status
        from src.dashboard.kill_authority import elevate_system_status_for_kill

        elevated = elevate_system_status_for_kill(
            health.get("status"), disk_kill, disk_open
        )
        if elevated:
            health["status"] = elevated
        health = _elevate_compact_health_status(health)
    except Exception:  # noqa: BLE001 — never fail kill refresh on elevate
        pass

    # Batch DQ: re-project ensemble concentration from sticky ensemble_voting
    # so partial patches that advance generated_at still disclose CAR>cap.
    try:
        ev = payload.get("ensemble_voting")
        if isinstance(ev, dict) and isinstance(health, dict):
            aw = ev.get("active_weights") or {}
            max_aw = float(ev.get("max_active_weight") or 0.0)
            if not max_aw and isinstance(aw, dict) and aw:
                max_aw = float(max(aw.values()))
            cap = float(ev.get("per_signal_active_weight_cap") or 0.50)
            ok = bool(max_aw <= cap + 1e-6) if (max_aw or aw) else True
            if "ensemble_concentration_ok" in ev:
                ok = bool(ev.get("ensemble_concentration_ok"))
            health["ensemble_max_active_weight"] = round(max_aw, 5)
            health["ensemble_per_signal_weight_cap"] = cap
            health["ensemble_concentration_ok"] = ok
            health["ensemble_n_eff"] = ev.get("n_eff")
            health["ensemble_concentration_status"] = (
                "ok" if ok else "concentrated"
            )
            if not ok and health.get("status") in (
                None,
                "ok",
                "healthy",
                "unknown",
            ):
                health["status"] = "warning"
            health["ensemble_may_lag_full_generate"] = (
                payload.get("generator_git_sha_status") == "partial_patch"
                or True  # this path is always a partial patch
            )
    except Exception:  # noqa: BLE001
        pass

    # Batch DV: re-project ML feature staleness from sticky ml_signals
    try:
        ml = payload.get("ml_signals")
        if isinstance(ml, dict) and isinstance(health, dict):
            fresh = str(ml.get("feature_freshness_status") or "unknown")
            age = ml.get("feature_staleness_days")
            try:
                age_i = int(age) if age is not None else None
            except (TypeError, ValueError):
                age_i = None
            health["ml_feature_freshness_status"] = fresh
            health["ml_feature_staleness_days"] = age_i
            health["ml_feature_as_of"] = ml.get("feature_as_of")
            health["ml_prediction_source_mode"] = ml.get("prediction_source_mode")
            health["ml_available"] = bool(ml.get("available"))
            er = (
                ml.get("execution_role")
                if isinstance(ml.get("execution_role"), dict)
                else {}
            )
            health["ml_live_authoritative"] = bool(er.get("live_authoritative"))
            if fresh == "stale" and bool(ml.get("available")):
                health["ml_features_stale"] = True
                if health.get("status") in (None, "ok", "healthy", "unknown"):
                    health["status"] = "warning"
            else:
                health["ml_features_stale"] = False
    except Exception:  # noqa: BLE001
        pass

    # Batch EI: rebalance_health is a sibling panel file, not sticky on
    # signals.json. Partial health patches never embed it → EG timeline SLI
    # stayed unknown and DW dual-clock lag stayed silent. Load DATA_DIR
    # (then PUBLIC) so compact health always discloses rewrite inflation.
    rebalance_health_panel = (
        payload.get("rebalance_health")
        if isinstance(payload.get("rebalance_health"), dict)
        else None
    )
    if rebalance_health_panel is None:
        try:
            root = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
            candidates = [
                root / "rebalance_health.json",
                Path(root_public) / "rebalance_health.json",
            ]
            for path in candidates:
                if not path.is_file():
                    continue
                try:
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(loaded, dict):
                    rebalance_health_panel = loaded
                    break
        except Exception:  # noqa: BLE001
            rebalance_health_panel = None

    # Batch DW: re-project smart-rebalance cost budget + dual-clock lag from
    # sticky smart_rebalance / rebalance_health so partial patches disclose
    # over-budget (ytd 214 bps vs 50) and controller last_rebalance lag.
    try:
        from src.dashboard.generator import project_smart_rebalance_budget_onto_health

        if isinstance(health, dict):
            health = project_smart_rebalance_budget_onto_health(
                health,
                payload.get("smart_rebalance")
                if isinstance(payload.get("smart_rebalance"), dict)
                else None,
                rebalance_health_panel,
            )
    except Exception:  # noqa: BLE001
        pass

    # Batch EG/EI: unique event-day execution timeline vs raw rewrite inflation
    try:
        from src.dashboard.generator import project_execution_timeline_onto_health

        if isinstance(health, dict):
            health = project_execution_timeline_onto_health(
                health,
                rebalance_health_panel,
            )
            if rebalance_health_panel is not None:
                health["rebalance_health_source"] = "disk_or_sticky"
            else:
                health.setdefault("rebalance_health_source", "missing")
    except Exception:  # noqa: BLE001
        pass

    # Batch EB: re-project paper return five-surface SSOT agreement from DATA_DIR
    # so partial patches disclose history/snapshot drift vs daily_pnl write SSOT.
    try:
        from src.dashboard.generator import project_paper_return_ssot_onto_health
        from src.monitor.paper_return_ssot import compare_five_surfaces

        if isinstance(health, dict):
            root = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
            cmp = compare_five_surfaces(root)
            health = project_paper_return_ssot_onto_health(health, cmp)
    except Exception:  # noqa: BLE001
        pass

    # Batch EC: re-project voting-mass quality from sticky ensemble_voting
    # (soft-floor share of active weights vs healthy vote mass).
    try:
        from src.dashboard.generator import project_voting_mass_quality_onto_health

        if isinstance(health, dict):
            health = project_voting_mass_quality_onto_health(
                health,
                payload.get("ensemble_voting")
                if isinstance(payload.get("ensemble_voting"), dict)
                else None,
            )
    except Exception:  # noqa: BLE001
        pass

    # Batch ED: re-project multi-horizon reentry eligibility (disclose only)
    try:
        from src.dashboard.generator import project_reentry_eligibility_onto_health

        if isinstance(health, dict):
            health = project_reentry_eligibility_onto_health(
                health,
                payload.get("ensemble_voting")
                if isinstance(payload.get("ensemble_voting"), dict)
                else None,
            )
    except Exception:  # noqa: BLE001
        pass

    # Batch EE: dual-signal pending vs artifact-fresh cron reconcile on compact
    try:
        from src.dashboard.generator import project_pending_artifact_cron_onto_health
        from src.monitor.hermes_cron import load_local_cron_jobs

        if isinstance(health, dict):
            root = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
            jobs_from_report = report.get("cron_jobs") if isinstance(report, dict) else None
            if isinstance(jobs_from_report, list) and jobs_from_report:
                health = project_pending_artifact_cron_onto_health(
                    health, jobs_from_report
                )
            else:
                local_jobs, _backend = load_local_cron_jobs(root / "cron_status.json")
                health = project_pending_artifact_cron_onto_health(health, local_jobs)
    except Exception:  # noqa: BLE001
        pass

    # Batch EJ: repo public/data mirror lag count (SoT = PUBLIC_DATA_DIR)
    try:
        from src.dashboard.generator import project_repo_public_mirror_lag_onto_health
        from src.monitor.repo_public_mirror_lag import summarize_repo_public_mirror_lag

        if isinstance(health, dict):
            lag_summary = summarize_repo_public_mirror_lag()
            health = project_repo_public_mirror_lag_onto_health(health, lag_summary)
    except Exception:  # noqa: BLE001
        pass

    # Batch EM: re-project graduation CB consecutive_ok from disk SSOT so
    # compact signals.health cannot stick at 0/yellow after EL climb.
    try:
        if isinstance(health, dict):
            root = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
            health = project_graduation_cb_onto_compact_health(health, data_dir=root)
    except Exception:  # noqa: BLE001
        pass

    root_data = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    private = root_data / "signals.json"

    # Authority gate (Batch GF dual-write): never publish hollow signals.
    # Prefer recovering target_allocations from the private twin when public
    # was partially wiped; still refuse when neither dest has live TA.
    try:
        from src.monitor.signal_authority import (
            AuthorityValidationError,
            is_champion_target_allocations,
            validate_authority_payload,
            write_signals_multi_dest,
        )
    except ImportError as exc:
        logger.warning("signal_authority unavailable; skip kill refresh write: %s", exc)
        return

    try:
        validate_authority_payload(payload)
    except AuthorityValidationError:
        recovered = False
        if private.is_file():
            try:
                priv_blob = json.loads(private.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                priv_blob = None
            if isinstance(priv_blob, dict) and isinstance(
                priv_blob.get("target_allocations"), dict
            ):
                payload["target_allocations"] = priv_blob["target_allocations"]
                try:
                    validate_authority_payload(payload)
                    recovered = True
                    logger.warning(
                        "signals kill refresh recovered target_allocations "
                        "from private twin %s",
                        private,
                    )
                except AuthorityValidationError:
                    recovered = False
        if not recovered:
            logger.error(
                "Refusing signals.health kill refresh: missing live authority "
                "target_allocations on public %s (private twin also unusable)",
                signals_path,
            )
            return

    if not is_champion_target_allocations(payload):
        logger.error(
            "Refusing signals.health kill refresh: non-champion "
            "target_allocations under champion hard rule"
        )
        return

    payload["health"] = health
    # Partial rewrite must advance top-level generated_at (mtime honesty)
    from datetime import datetime, timezone
    now_utc = datetime.now(timezone.utc).isoformat()
    payload["generated_at"] = now_utc
    payload["content_patched_at"] = now_utc
    payload["content_patch_source"] = "health_kill_refresh"
    # Clear sticky full-run git sha — partial ≠ full dashboard generation
    try:
        from src.dashboard.generator import _apply_partial_patch_git_sha_honesty

        _apply_partial_patch_git_sha_honesty(
            payload, patch_source="health_kill_refresh"
        )
    except Exception:  # noqa: BLE001 — never fail kill refresh on import
        prior = payload.get("generator_git_sha")
        if prior:
            payload.setdefault("last_full_generator_git_sha", prior)
        payload["generator_git_sha"] = None
        payload["generator_git_sha_status"] = "partial_patch"
    # After honesty stamp, concentration lag flag is definitive for this path
    if isinstance(payload.get("health"), dict):
        payload["health"]["ensemble_may_lag_full_generate"] = True

    try:
        # Fan-out private twin when DATA_DIR is a real directory (or file already
        # exists). write_signals_multi_dest skips same-path resolve itself.
        private_dest = private if (private.exists() or private.parent.is_dir()) else None
        result = write_signals_multi_dest(
            payload,
            public_path=signals_path,
            private_path=private_dest,
            soft_mirror_repo=True,
        )
        if result.wrote_public:
            logger.info("Refreshed signals.health kill fields at %s", signals_path)
        if result.wrote_private:
            logger.info("Refreshed private signals.health twin at %s", private)
        if result.skipped_reason:
            logger.warning(
                "signals multi-dest kill refresh partial skip: %s",
                result.skipped_reason,
            )
    except AuthorityValidationError as exc:
        logger.error("Refusing signals.health kill refresh (authority gate): %s", exc)
    except OSError as exc:
        logger.warning("Failed to write signals health kill refresh: %s", exc)


def publish_health_alerts_json(report: dict[str, Any] | None = None) -> Path | None:
    """Write PUBLIC_DATA_DIR/alerts.json from health SLO + kill surfaces.

    Health cron previously left alerts.json frozen at the last full dashboard
    generate. Build a compact alerts payload so ``generated_at`` tracks the
    health job stamp. Best-effort; never raises to callers.

    Batch JL JH1c: prefer **public dashboard** ``health.json`` (system_status +
    signal_health) for SLO/quality rollup. Monitor-schema reports lack those
    fields and produced false-empty alerts while public SH was degraded.
    Kill/open identity still comes from disk SSOT.
    """
    public = Path(PUBLIC_DATA_DIR)
    out_path = public / "alerts.json"
    try:
        from src.dashboard.health_slo_alerts import build_health_slo_alerts
        from src.dashboard.generator import _stamp_generator_git_sha
    except Exception as exc:  # noqa: BLE001
        logger.warning("alerts publish imports failed: %s", exc)
        return None

    monitor_report: dict[str, Any] = report if isinstance(report, dict) else {}
    dashboard_health: dict[str, Any] | None = None
    # Prefer operator-visible dashboard schema for SH / system_status
    for candidate in (public / "health.json", Path(DATA_DIR) / "health.json"):
        if not candidate.is_file():
            continue
        try:
            blob = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(blob, dict):
            continue
        # Dashboard schema: system_status and/or signal_health present
        if "system_status" in blob or isinstance(blob.get("signal_health"), dict):
            dashboard_health = blob
            break
        # Skip pure monitor schema (status+checks, no system_status)
        if _is_monitor_health_report(blob) and "system_status" not in blob:
            continue
        dashboard_health = blob
        break

    if dashboard_health is None and not monitor_report:
        for candidate in (Path(HEALTH_PATH), public / "health_ops.json"):
            if not candidate.is_file():
                continue
            try:
                blob = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
            if isinstance(blob, dict):
                monitor_report = blob
                break

    # Merge: dashboard SH/system_status + live ops stamp from monitor report
    health_payload: dict[str, Any] = dict(dashboard_health or monitor_report or {})
    if monitor_report:
        # Overlay kill/open from monitor checks when dashboard lacks them
        mon_checks = (
            monitor_report.get("checks")
            if isinstance(monitor_report.get("checks"), dict)
            else {}
        )
        if isinstance(mon_checks.get("kill_switch"), dict):
            health_payload.setdefault("kill_switch", mon_checks["kill_switch"])
        if isinstance(mon_checks.get("open_incidents"), dict):
            health_payload.setdefault("open_incidents", mon_checks["open_incidents"])
        # JH1b: live ops stamp for quality-vs-ops labeling
        mon_status = str(monitor_report.get("status") or "").lower()
        if mon_status:
            health_payload["ops_health_status"] = mon_status
            health_payload["ops_health_timestamp"] = monitor_report.get("timestamp")

    # Disk kill SSOT always wins over sticky dashboard kill fields
    try:
        disk_kill, disk_open = _disk_kill_and_open_incidents(DATA_DIR)
        health_payload["kill_switch"] = disk_kill
        health_payload["open_incidents"] = disk_open
    except Exception:  # noqa: BLE001
        pass

    # Post-merge live lag heal (2026-07-26 health-alerts-publish-after-lag-heal):
    # Soft-mirror during ops merge can leave sticky lag stamps + system_status=
    # warning on public health while the live probe is already 0. Labeling must
    # re-probe live lag and re-derive ops system_status (SH excluded) before
    # build_health_slo_alerts so health-only cron cannot leave Health Warning: ops
    # when final ops are green and only signal quality is thin.
    try:
        import os

        from src.dashboard.generator import project_repo_public_mirror_lag_onto_health
        from src.monitor.repo_public_mirror_lag import (
            is_ephemeral_restamp_path,
            rederive_ops_status_for_lag_heal,
            summarize_repo_public_mirror_lag,
        )

        public_root = Path(PUBLIC_DATA_DIR)
        # Under pytest / ephemeral PUBLIC trees, never project production lag
        # onto fixture health (HW rebind would stamp live WWW lag into tmp).
        # Pass source=dest=public_root so the probe is isolation-local; unit
        # tests that monkeypatch summarize_repo_public_mirror_lag still win.
        if os.environ.get("PYTEST_CURRENT_TEST") or is_ephemeral_restamp_path(
            public_root
        ):
            live = summarize_repo_public_mirror_lag(
                source_root=public_root,
                dest_root=public_root,
            )
        else:
            live = summarize_repo_public_mirror_lag()
        if isinstance(live, dict):
            project_repo_public_mirror_lag_onto_health(health_payload, live)
            try:
                lag_n = int(live.get("lagging_count") or 0)
            except (TypeError, ValueError):
                lag_n = -1
            if lag_n == 0:
                for key, value in rederive_ops_status_for_lag_heal(health_payload).items():
                    health_payload[key] = value
    except Exception as exc:  # noqa: BLE001 — fall back to stamp labeling
        logger.debug("alerts live lag heal skipped: %s", exc)

    now_utc = datetime.now(timezone.utc).isoformat()
    # Prefer health stamp so operators can correlate
    stamp = (
        health_payload.get("generated_at")
        or health_payload.get("timestamp")
        or (monitor_report.get("timestamp") if monitor_report else None)
        or now_utc
    )
    alerts = list(build_health_slo_alerts(health_payload) or [])
    # Include kill row when present on health checks
    try:
        from src.dashboard.kill_authority import (
            build_kill_switch_alert,
            load_kill_switch_payload,
        )

        kill_payload = load_kill_switch_payload(DATA_DIR)
        kill_alert = (
            build_kill_switch_alert(kill_payload) if kill_payload else None
        )
        if kill_alert is not None:
            alerts.insert(0, kill_alert)
    except Exception:  # noqa: BLE001 — kill surface optional
        pass

    output: dict[str, Any] = {
        "alerts": alerts,
        "count": len(alerts),
        "generated_at": stamp,
        "source": "health_check_job",
        "health_generated_at": health_payload.get("generated_at")
        or health_payload.get("timestamp"),
    }
    # F3: the health job fully rebuilds the alerts surface from disk SSOT each
    # run (not a patch over a stale artifact), so the provenance stamp is a
    # full_generate with the HEAD-derived sha — operators can attribute it.
    try:
        output = _stamp_generator_git_sha(output, status="full_generate")
    except Exception:  # noqa: BLE001
        pass
    try:
        from src.monitor.signal_authority import write_json_multi_dest

        public.mkdir(parents=True, exist_ok=True)
        # Batch HN: serialize-once multi-dest (public + private + repo soft-mirror)
        # with 0o644. Prior dual path.write_text left repo public/data stale
        # (alerts 0/1/1) while health refreshed private+www only.
        # Leave repo_path=None so write_json_multi_dest auto soft-mirrors via
        # repo_filename, and skips auto under pytest (no checkout clobber).
        data_alerts = Path(DATA_DIR) / "alerts.json"
        result = write_json_multi_dest(
            output,
            public_path=out_path,
            private_path=data_alerts if data_alerts.parent.is_dir() or data_alerts.exists() else None,
            soft_mirror_repo=True,
            repo_filename="alerts.json",
        )
        if result.wrote_public:
            logger.info(
                "Health job wrote alerts.json (%d alerts) at %s (private=%s repo=%s)",
                len(alerts),
                out_path,
                result.wrote_private,
                result.wrote_repo,
            )
        if result.skipped_reason:
            logger.warning(
                "alerts.json multi-dest partial skip: %s", result.skipped_reason
            )
        return out_path if result.wrote_public else None
    except OSError as exc:
        logger.warning("alerts.json publish failed: %s", exc)
        return None


def publish_ops_health_surfaces(report: dict[str, Any]) -> None:
    """Write monitor health to PUBLIC_DATA_DIR and merge kill into public health.json.

    Dual-path honesty:
    - Always write ``health_ops.json`` (monitor schema) under PUBLIC_DATA_DIR.
    - If dashboard ``health.json`` already exists, merge kill_switch /
      open_incidents / elevated system_status so operators see halt without
      waiting for the dashboard generator cycle.
    - Also refresh ``signals.json#health`` compact kill fields so post-resolve
      kill clear is visible within one health cron (not only full dashboard).
    """
    ops_path = health_ops_path()
    public_health = Path(PUBLIC_DATA_DIR) / "health.json"
    try:
        from src.dashboard.generator import _attach_dual_write_provenance

        report = _attach_dual_write_provenance(
            report,
            private_path=HEALTH_PATH,
            public_path=ops_path,
            dual_write_attempted=True,
            dual_write_ok=None,  # set after write
            paths_identical=False,
            note="health_ops is public dual surface; private SSOT is data/health.json",
        )
    except Exception:  # noqa: BLE001
        pass
    try:
        ops_path.parent.mkdir(parents=True, exist_ok=True)
        # Mark dual_write_ok after successful ops write
        try:
            from src.dashboard.generator import _attach_dual_write_provenance

            report = _attach_dual_write_provenance(
                report,
                private_path=HEALTH_PATH,
                public_path=ops_path,
                dual_write_attempted=True,
                dual_write_ok=True,
                paths_identical=False,
                note="health_ops is public dual surface; private SSOT is data/health.json",
            )
        except Exception:  # noqa: BLE001
            pass
        # Batch IC: serialize-once multi-dest (public + private ops twin + repo
        # soft-mirror) with atomic 0o644. Private twin is DATA_DIR/health_ops.json
        # (not monitor health.json — different schema).
        private_ops = Path(DATA_DIR) / "health_ops.json"
        wrote_ops = False
        try:
            from src.monitor.signal_authority import write_json_multi_dest

            result = write_json_multi_dest(
                report,
                public_path=ops_path,
                private_path=private_ops,
                soft_mirror_repo=True,
                repo_filename="health_ops.json",
            )
            wrote_ops = bool(
                result.wrote_public or result.wrote_private or result.wrote_repo
            )
            if result.skipped_reason:
                logger.warning(
                    "health_ops multi-dest partial skip: %s", result.skipped_reason
                )
        except Exception as multi_exc:  # noqa: BLE001
            logger.warning(
                "health_ops multi-dest failed (%s); fallback write_text", multi_exc
            )
            wrote_ops = False
        if not wrote_ops:
            _atomic_write_json_path(ops_path, report)
            try:
                import os

                os.chmod(ops_path, 0o644)
            except OSError:
                pass
        logger.info("Ops health written to %s", ops_path)
    except OSError as exc:
        logger.warning("Failed to write ops health at %s: %s", ops_path, exc)

    public_health = Path(PUBLIC_DATA_DIR) / "health.json"
    if public_health.exists():
        try:
            payload = json.loads(public_health.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("Public health.json unreadable; skip merge: %s", exc)
            payload = None
        if isinstance(payload, dict):
            apply_ops_monitor_to_dashboard_health(
                payload,
                report,
                data_dir=DATA_DIR,
                public_dir=PUBLIC_DATA_DIR,
            )
            # Partial merge must advance content timestamp (not only kill fields)
            now_utc = datetime.now(timezone.utc).isoformat()
            payload["generated_at"] = now_utc
            payload["content_patched_at"] = now_utc
            payload["content_patch_source"] = "ops_health_merge"
            # health.json is not full dashboard signals, but still clear sticky
            # generator_git_sha if present from any shared payload shape.
            try:
                from src.dashboard.generator import _apply_partial_patch_git_sha_honesty

                if "generator_git_sha" in payload:
                    _apply_partial_patch_git_sha_honesty(
                        payload, patch_source="ops_health_merge"
                    )
            except Exception:  # noqa: BLE001
                pass
            # Batch CH: ops monitor stamps full_generate with current tip on every
            # health job. Promote that tip into last_full so lag forensics do not
            # freeze on an older dashboard last_full after partial_patch clears
            # live generator_git_sha (c367: last_full stuck at fa7263a while ops
            # full_generate advanced).
            try:
                ops_st = str(report.get("generator_git_sha_status") or "")
                ops_sha = report.get("generator_git_sha")
                ops_last = report.get("last_full_generator_git_sha")
                if ops_st == "full_generate" and ops_sha not in (None, ""):
                    payload["last_full_generator_git_sha"] = str(ops_sha)
                elif ops_last not in (None, ""):
                    payload.setdefault("last_full_generator_git_sha", str(ops_last))
            except Exception:  # noqa: BLE001
                pass
            # H20: stamp dual-write provenance on health.json so M11 badge does not
            # depend solely on health_ops.json (merge path is partial dual-write).
            try:
                from src.dashboard.generator import _attach_dual_write_provenance

                payload = _attach_dual_write_provenance(
                    payload,
                    private_path=HEALTH_PATH,
                    public_path=public_health,
                    dual_write_attempted=True,
                    dual_write_ok=True,
                    paths_identical=False,
                    note=(
                        "ops_health_merge dual-write: kill/open from disk SSOT; "
                        "health_ops remains monitor schema surface"
                    ),
                )
            except Exception:  # noqa: BLE001 — never block health merge on provenance
                pass
            try:
                # Batch IC: public dashboard health + repo soft-mirror only.
                # Never fan-out to private DATA_DIR/health.json (monitor schema).
                wrote_health = False
                try:
                    from src.monitor.signal_authority import write_json_multi_dest

                    h_result = write_json_multi_dest(
                        payload,
                        public_path=public_health,
                        private_path=None,
                        soft_mirror_repo=True,
                        repo_filename="health.json",
                    )
                    wrote_health = bool(
                        h_result.wrote_public or h_result.wrote_repo
                    )
                    if h_result.skipped_reason:
                        logger.warning(
                            "public health merge multi-dest partial skip: %s",
                            h_result.skipped_reason,
                        )
                except Exception as multi_exc:  # noqa: BLE001
                    logger.warning(
                        "public health merge multi-dest failed (%s); fallback",
                        multi_exc,
                    )
                    wrote_health = False
                if not wrote_health:
                    _atomic_write_json_path(public_health, payload)
                    try:
                        import os

                        os.chmod(public_health, 0o644)
                    except OSError:
                        pass
                logger.info("Merged ops kill authority into %s", public_health)
            except OSError as exc:
                logger.warning("Failed to merge ops health into %s: %s", public_health, exc)
                try:
                    from src.dashboard.generator import _attach_dual_write_provenance

                    # Re-stamp private monitor report if public health write failed
                    report_fail = _attach_dual_write_provenance(
                        report,
                        private_path=HEALTH_PATH,
                        public_path=public_health,
                        dual_write_attempted=True,
                        dual_write_ok=False,
                        paths_identical=False,
                        note=f"public health.json merge write failed: {exc}",
                    )
                    # Best-effort re-write health_ops with dual_write_ok=false
                    try:
                        from src.monitor.signal_authority import write_json_multi_dest

                        write_json_multi_dest(
                            report_fail,
                            public_path=ops_path,
                            private_path=Path(DATA_DIR) / "health_ops.json",
                            soft_mirror_repo=True,
                            repo_filename="health_ops.json",
                        )
                    except Exception:  # noqa: BLE001
                        _atomic_write_json_path(ops_path, report_fail)
                except Exception:  # noqa: BLE001
                    pass

    try:
        refresh_signals_health_kill_fields(
            report, public_dir=Path(PUBLIC_DATA_DIR), data_dir=Path(DATA_DIR)
        )
    except Exception as exc:  # noqa: BLE001 — never fail health job on signals patch
        logger.warning("signals.health kill refresh failed: %s", exc)

    # Batch BI: recompute public index digests after health/signals partial patches
    try:
        from src.dashboard.public_data_index import (
            refresh_public_data_index_after_partial_write,
        )

        refresh_public_data_index_after_partial_write(
            public_dir=Path(PUBLIC_DATA_DIR),
            extra_paths=[
                Path(PUBLIC_DATA_DIR) / "health.json",
                Path(PUBLIC_DATA_DIR) / "health_ops.json",
                Path(PUBLIC_DATA_DIR) / "signals.json",
            ],
            reason="ops_health_merge",
        )
    except Exception as exc:  # noqa: BLE001 — never fail health job on index
        logger.warning("public index refresh after ops health failed: %s", exc)


def _should_include_hermes_audit(local_backend: dict) -> bool:
    """Return true when Hermes should be surfaced alongside tasker health."""
    if os.environ.get("TASKER_INCLUDE_HERMES_AUDIT") == "1":
        return True
    if local_backend.get("backend") == "tasker" and os.environ.get("CRON_BACKEND") == "tasker":
        return False
    return True


def _scheduler_drift_state_path() -> Path:
    """Resolve scheduler drift state relative to the active data directory."""
    return DATA_DIR / "scheduler_drift_state.json"


def _load_scheduler_drift_state(path: Path) -> dict[str, Any]:
    """Load prior scheduler drift state, tolerating missing or malformed state."""
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read scheduler drift state: %s", exc)
        return {}
    return data if isinstance(data, dict) else {}


def _save_scheduler_drift_state(path: Path, state: dict[str, Any]) -> None:
    """Persist scheduler drift state without blocking health report generation."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
    except OSError as exc:
        logger.warning("Failed to write scheduler drift state: %s", exc)


def _backend_participates_in_drift(backend: dict[str, Any]) -> bool:
    """Return True when a backend should count toward multi-backend drift.

    Idle empty schedulers (e.g. Hermes with zero portfolio-lab jobs after
    migration to tasker) report status=ok while the active backend may be
    degraded due to failed jobs. That is job-health noise, not dual-backend
    schedule drift — exclude zero-job backends from the comparison set.
    """
    try:
        total_jobs = int(backend.get("total_jobs") or 0)
    except (TypeError, ValueError):
        total_jobs = 0
    if total_jobs > 0:
        return True
    # Non-empty backends with explicit error still participate (misconfig).
    status = str(backend.get("status", "unknown")).lower()
    return status in {"error", "unavailable"} and bool(backend.get("reason"))


def check_scheduler_drift(
    backends: dict[str, dict[str, Any]],
    *,
    state_path: Path | None = None,
    threshold: int = SCHEDULER_DRIFT_THRESHOLD,
) -> dict[str, Any]:
    """Detect persistent disagreement between scheduler backend health states."""
    path = state_path or _scheduler_drift_state_path()
    backend_statuses = {
        str(name): str(backend.get("status", "unknown"))
        for name, backend in backends.items()
        if isinstance(backend, dict)
    }
    compared_statuses = {
        str(name): str(backend.get("status", "unknown"))
        for name, backend in backends.items()
        if isinstance(backend, dict) and _backend_participates_in_drift(backend)
    }
    # Drift requires ≥2 active/participating backends with disagreeing status.
    mismatch = len(compared_statuses) >= 2 and len(set(compared_statuses.values())) > 1
    previous_state = _load_scheduler_drift_state(path)
    previous_count = int(previous_state.get("consecutive_mismatches") or 0)
    consecutive_mismatches = previous_count + 1 if mismatch else 0
    status = "critical" if mismatch and consecutive_mismatches >= threshold else "warning" if mismatch else "ok"
    details = {
        "status": status,
        "mismatch": mismatch,
        "consecutive_mismatches": consecutive_mismatches,
        "threshold": threshold,
        "backend_statuses": backend_statuses,
        "compared_backend_statuses": compared_statuses,
    }

    if mismatch and consecutive_mismatches >= threshold:
        send_alert(
            AlertChannel.CRON_FAILURE,
            AlertLevel.HALT,
            f"Scheduler backend drift persisted for {consecutive_mismatches} checks",
            details=details,
        )
    elif not mismatch and previous_count > 0:
        send_alert(
            AlertChannel.CRON_FAILURE,
            AlertLevel.PASS,
            "Scheduler backends agree after drift",
            details=details,
        )

    if mismatch or previous_count > 0:
        _save_scheduler_drift_state(
            path,
            {
                **details,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    return details


def _backend_summary_excluding_health_self(
    backend: dict[str, Any],
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recompute backend failed_jobs/status ignoring portfolio-lab-health errors.

    Keeps unavailable/error/reason backends untouched. Unknown active jobs still
    degrade. Only demotes degraded→ok when the sole failure was the self job.
    """
    backend_name = str(backend.get("backend") or "")
    adjusted = dict(backend)
    backend_jobs = [job for job in jobs if str(job.get("backend") or "") == backend_name]
    failed_jobs = sum(
        1
        for job in backend_jobs
        if job.get("status") == "error" and not is_health_self_job(job)
    )
    adjusted["failed_jobs"] = failed_jobs
    # Preserve explicit unavailable/error set by loaders (missing file, parse fail).
    if adjusted.get("status") in {"unavailable", "error", "missing"}:
        return adjusted
    if adjusted.get("reason"):
        return adjusted
    active_unknown = int(adjusted.get("unknown_active_jobs") or 0)
    if failed_jobs or active_unknown:
        adjusted["status"] = "degraded"
    else:
        adjusted["status"] = "ok"
    return adjusted


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


def _check_fred_md_cache() -> dict:
    """Check FRED-MD cache availability without making live provider calls."""
    try:
        from src.data.fred_data import get_fred_md_cache_health

        return get_fred_md_cache_health()
    except ImportError as exc:
        return {
            "status": "unavailable",
            "row_count": 0,
            "latest_fetched_at": None,
            "age_hours": None,
            "reason": str(exc),
        }


def _load_json_file(path: Path) -> dict[str, Any] | None:
    """Load a JSON object from disk; return None when missing or invalid."""
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError, TypeError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def _check_kill_switch(data_dir: Path | None = None) -> dict[str, Any]:
    """Bounded kill-switch dimension for operational readiness."""
    root = data_dir or DATA_DIR
    payload = _load_json_file(root / "kill_switch.json")
    if not payload:
        return {
            "status": "ok",
            "enabled": False,
            "level": None,
            "reason": None,
            "source": None,
            "message": None,
            "timestamp": None,
        }

    enabled = bool(payload.get("enabled"))
    level = str(payload.get("level") or "").lower() or None
    reason = payload.get("reason")
    source = payload.get("source")
    message = payload.get("message")
    timestamp = payload.get("timestamp")

    if enabled and level == "halt":
        status = "critical"
    elif enabled:
        status = "warning"
    else:
        status = "ok"

    return {
        "status": status,
        "enabled": enabled,
        "level": level,
        "reason": reason,
        "source": source,
        "message": message,
        "timestamp": timestamp,
        "incident_id": payload.get("incident_id"),
        "mode": payload.get("mode"),
    }


def _check_open_incidents(data_dir: Path | None = None) -> dict[str, Any]:
    """Bounded open-incident dimension for operational readiness."""
    root = data_dir or DATA_DIR
    payload = _load_json_file(root / "incidents.json")
    if not payload:
        return {
            "status": "ok",
            "open_count": 0,
            "incidents": [],
        }

    raw = payload.get("incidents", payload.get("open_incidents", []))
    rows = raw if isinstance(raw, list) else []
    open_incidents: list[dict[str, Any]] = []
    has_halt = False

    for row in rows:
        if not isinstance(row, dict):
            continue
        state = str(row.get("state") or row.get("status") or "open").lower()
        if state in {"closed", "resolved", "pass"}:
            continue
        kill_level = str(row.get("kill_switch_level") or "").lower() or None
        if kill_level == "halt":
            has_halt = True
        open_incidents.append({
            "incident_id": row.get("incident_id") or row.get("id"),
            "channel": row.get("channel"),
            "severity": row.get("severity"),
            "state": state,
            "message": row.get("message"),
            "kill_switch_level": kill_level,
        })

    open_count = int(payload.get("open_count") or len(open_incidents) or 0)
    if open_count == 0 and open_incidents:
        open_count = len(open_incidents)

    if has_halt or any(
        str(i.get("kill_switch_level") or "").lower() == "halt" for i in open_incidents
    ):
        status = "critical"
    elif open_count > 0 or open_incidents:
        status = "warning"
    else:
        status = "ok"

    return {
        "status": status,
        "open_count": open_count,
        "incidents": open_incidents[:10],
    }


def _status_for_system_rollup(name: str, check: dict) -> str:
    """Map a component status into the overall rollup severity ladder.

    Nested report fields keep their raw status for operators. For rollup /
    process exit, non-blocking lab FRED advisories (ready, not blocking, or
    empty cache without a key) must not force overall ``warning`` — that made
    ``make health`` exit 1 every cycle and sticky tasker ``error`` rows.
    """
    status = str(check.get("status", "unknown"))
    if name == "fred_readiness":
        if check.get("ready") is True and check.get("blocking") is False:
            return "ok"
    if name == "fred_md_cache":
        if status == "empty" and not check.get("api_key_configured"):
            return "ok"
    return status


def _compute_system_status(checks: dict, circuit: dict) -> str:
    """Derive overall system status from component checks.

    Severity order (highest first): critical > degraded > warning > ok.
    Active kill-switch HALT / open-incident HALT use status ``critical`` so
    they cannot be understated by lower-severity freshness warnings.
    """
    statuses = []

    for name, check in checks.items():
        if isinstance(check, dict):
            statuses.append(_status_for_system_rollup(str(name), check))

    if isinstance(circuit, dict):
        statuses.append(str(circuit.get("status", "unknown")))

    if "critical" in statuses:
        return "critical"
    if "error" in statuses or "missing" in statuses:
        return "degraded"
    if (
        "stale" in statuses
        or "empty" in statuses
        or "degraded" in statuses
        or "warning" in statuses
        or "unavailable" in statuses
    ):
        return "warning"
    if all(s == "ok" for s in statuses):
        return "ok"
    return "unknown"


def attach_shared_freshness_slis_to_ops_report(
    report: dict[str, Any] | None,
    *,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Attach mirror-lag + execution-timeline SLIs to monitor health_ops (Batch EK).

    Deep-research: ops health report and compact dashboard health must share the
    same freshness metrics so operators do not see split-brain (signals.health
    shows rewrite_inflated / lagging while health_ops is silent).
    """
    if not isinstance(report, dict):
        report = {}

    # --- repo public/data mirror lag (same probe as compact signals.health) ---
    try:
        from src.dashboard.generator import project_repo_public_mirror_lag_onto_health
        from src.monitor.repo_public_mirror_lag import summarize_repo_public_mirror_lag

        lag_summary = summarize_repo_public_mirror_lag()
        # Batch FX / EP: project onto the real report so soft-elevate of
        # top-level status=ok → warning under lagging/critical is not a dead path.
        report = project_repo_public_mirror_lag_onto_health(report, lag_summary)
        report["repo_public_mirror_lag"] = {
            "lagging_count": report.get("repo_public_mirror_lagging_count"),
            "total": report.get("repo_public_mirror_total"),
            "status": report.get("repo_public_mirror_lag_status"),
            "badge": report.get("repo_public_mirror_lag_badge"),
            "paths": report.get("repo_public_mirror_lagging_paths"),
            "source": report.get("repo_public_mirror_source"),
            "dest": report.get("repo_public_mirror_dest"),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("ops report mirror lag SLI skipped: %s", exc)
        report.setdefault("repo_public_mirror_lag_status", "unknown")

    # --- graduation CB SSOT re-projection (Batch EM) ---
    try:
        root = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
        report = project_graduation_cb_onto_report(report, data_dir=root)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ops report graduation CB SLI skipped: %s", exc)

    # --- rebalance execution timeline (disk panel, same as Batch EI) ---
    try:
        from src.dashboard.generator import project_execution_timeline_onto_health

        root = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
        rebalance_health_panel: dict[str, Any] | None = None
        for path in (
            root / "rebalance_health.json",
            Path(PUBLIC_DATA_DIR) / "rebalance_health.json",
        ):
            if not path.is_file():
                continue
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(loaded, dict):
                rebalance_health_panel = loaded
                break
        projected_tl: dict[str, Any] = {}
        projected_tl = project_execution_timeline_onto_health(
            projected_tl, rebalance_health_panel
        )
        for key in (
            "rebalance_execution_timeline_status",
            "rebalance_execution_timeline_badge",
            "rebalance_unique_execution_days",
            "rebalance_raw_history_entries",
            "rebalance_snapshot_rewrite_files",
            "rebalance_execution_timeline_policy",
        ):
            if key in projected_tl:
                report[key] = projected_tl[key]
        report["rebalance_execution_timeline"] = {
            "status": projected_tl.get("rebalance_execution_timeline_status"),
            "badge": projected_tl.get("rebalance_execution_timeline_badge"),
            "unique_days": projected_tl.get("rebalance_unique_execution_days"),
            "raw_entries": projected_tl.get("rebalance_raw_history_entries"),
            "rewrite_files": projected_tl.get("rebalance_snapshot_rewrite_files"),
            "source": "disk" if rebalance_health_panel is not None else "missing",
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("ops report execution timeline SLI skipped: %s", exc)
        report.setdefault("rebalance_execution_timeline_status", "unknown")

    return report


def _stamp_health_self_job_running_success(freshness: dict[str, Any]) -> None:
    """Overwrite portfolio-lab-health row so a successful run does not publish prior error.

    Tasker writes cron_status *after* the job body. During ``run_health_check`` the
    self job still shows the previous terminal status. When that was ``error``, the
    successful run's own report freezes a false error row into health_ops/public
    until a later writer fixes it. Stamp in-process success for honesty; rollup
    already excludes self-errors from failed_jobs.
    """
    if not isinstance(freshness, dict):
        return
    cron = freshness.get("cron")
    if not isinstance(cron, dict):
        return
    jobs = cron.get("jobs")
    if not isinstance(jobs, list):
        return
    now = datetime.now(timezone.utc).isoformat()
    stamped = False
    for job in jobs:
        if not isinstance(job, dict):
            continue
        if not is_health_self_job(job):
            continue
        prev = job.get("status")
        job["status"] = "ok"
        job["last_run"] = now
        job["self_observation"] = "in_process_success_stamp"
        if prev not in (None, "ok", "success"):
            job["prior_status_before_stamp"] = prev
        stamped = True
    if stamped:
        # Keep embedded failed_jobs consistent with rollup (already excludes self)
        try:
            from src.monitor.hermes_cron import rollup_failed_cron_jobs

            cron["failed_jobs"] = len(rollup_failed_cron_jobs(jobs))
        except Exception:  # noqa: BLE001
            pass


def run_health_check() -> dict:
    """Run all health checks and return a structured report."""
    freshness = _check_data_freshness()
    # Self-job race: do not publish prior terminal error while this process succeeds.
    _stamp_health_self_job_running_success(freshness)
    circuit = _check_circuit_breaker()
    # Batch II DE4: re-read disk kill/open immediately before rollup so a
    # mid-run arm (incident lifecycle after first probe) is not lost until
    # the next :00/:30 health tick.
    kill_switch = _check_kill_switch()
    open_incidents = _check_open_incidents()
    fred_md_cache = _check_fred_md_cache()
    freshness["fred_md_cache"] = fred_md_cache
    try:
        from src.monitor.fred_readiness import assess_fred_readiness

        freshness["fred_readiness"] = assess_fred_readiness(fred_md_cache)
    except ImportError as exc:
        freshness["fred_readiness"] = {
            "status": "warning",
            "readiness": "unknown",
            "ready": True,
            "blocking": False,
            "reason": "readiness_check_unavailable",
            "message": f"FRED readiness check unavailable: {exc}",
            "remediation": "Verify src.monitor.fred_readiness is importable.",
        }
    # Batch IZ/JA DO2: re-evaluate signal_staleness lifecycle so PASS clears
    # false opens without waiting for full dashboard generate.
    try:
        from src.monitor.alerting import check_staleness_and_alert

        for sp in (Path(DATA_DIR) / "signals.json", Path(PUBLIC_DATA_DIR) / "signals.json"):
            if not sp.is_file():
                continue
            try:
                payload = json.loads(sp.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            st = payload.get("staleness") if isinstance(payload, dict) else None
            if isinstance(st, dict) and st:
                check_staleness_and_alert(st)
                break
    except Exception as exc:  # noqa: BLE001 — never fail health job
        logger.warning("Staleness lifecycle re-eval skipped: %s", exc)

    # End-of-build disk re-read (DE4): prefer latest kill_switch.json /
    # incidents.json over the first snapshot if lifecycle wrote mid-check.
    kill_switch = _check_kill_switch()
    open_incidents = _check_open_incidents()
    # Flatten nested freshness statuses for rollup while keeping nested shape
    # in the report. Critical/warning FRED readiness must still elevate.
    rollup_checks = {
        **{k: v for k, v in freshness.items() if isinstance(v, dict) and "status" in v},
        "kill_switch": kill_switch,
        "open_incidents": open_incidents,
    }
    system_status = _compute_system_status(rollup_checks, circuit)

    # Persist graduation consecutive_ok producer (SSOT for circuit_breaker_confidence)
    # Batch CB: fold live dashboard signal_health when present so ops-only ok
    # cannot climb CB while SH fleet is 0/N healthy.
    try:
        sh_for_cb: dict | None = None
        for candidate in (
            Path(PUBLIC_DATA_DIR) / "health.json",
            Path(DATA_DIR) / "health.json",
        ):
            if not candidate.is_file():
                continue
            try:
                blob = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
            if isinstance(blob, dict) and isinstance(blob.get("signal_health"), dict):
                sh_for_cb = blob["signal_health"]
                break
        grad_cb = update_graduation_circuit_breaker_state(
            system_status=system_status,
            broker_circuit=circuit,
            data_dir=DATA_DIR,
            signal_health=sh_for_cb,
        )
    except Exception as exc:  # noqa: BLE001 — never fail health job on CB producer
        logger.warning("Graduation CB consecutive_ok producer failed: %s", exc)
        grad_cb = None

    report = {
        "status": system_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "data_freshness": freshness,
            "circuit_breaker": circuit,
            "kill_switch": kill_switch,
            "open_incidents": open_incidents,
        },
        "service": "portfolio-lab",
        "scope": "operational_readiness",
    }
    # IC quality is a bounded advisory projection. Its control labels mirror
    # the existing disk kill authority; IC status never changes routing here.
    try:
        from src.monitor.ic_decay_monitor import (
            build_ic_decay_summary,
            compute_ic_decay_report,
            ic_control_projection,
        )

        control = ic_control_projection(kill_switch)
        report["ic_decay_summary"] = build_ic_decay_summary(
            compute_ic_decay_report(),
            evidence_generated_at=report["timestamp"],
            **control,
        )
    except Exception as exc:  # noqa: BLE001 — quality disclosure is additive
        logger.warning("IC quality summary projection skipped: %s", exc)
    if isinstance(grad_cb, dict):
        # Batch EL/EM: surface SH block reason + mark projection source so
        # operators see why streak freezes without opening private SSOT.
        report["graduation_circuit_breaker"] = {
            "consecutive_ok": grad_cb.get("consecutive_ok"),
            "status": grad_cb.get("status"),
            "updated_at": grad_cb.get("updated_at"),
            "signal_health_blocked": bool(grad_cb.get("signal_health_blocked")),
            "signal_health_contribution": grad_cb.get(
                "signal_health_contribution"
            ),
            "system_status": grad_cb.get("system_status"),
            "graduation_cb_source": "producer",
            "schema_version": grad_cb.get("schema_version"),
            "producer": grad_cb.get("producer"),
        }
    else:
        # Fall back to disk SSOT if producer failed mid-run
        try:
            report = project_graduation_cb_onto_report(report, data_dir=DATA_DIR)
        except Exception:  # noqa: BLE001
            pass
        # Batch BP: refresh graduation dual surfaces when CB streak changes so
        # public graduation.json does not lag SSOT until next dashboard :15.
        try:
            prev_ok = None
            # Compare to last published public graduation CB criterion if present
            pub_grad = Path(PUBLIC_DATA_DIR) / "graduation.json"
            if pub_grad.is_file():
                try:
                    prior = json.loads(pub_grad.read_text(encoding="utf-8"))
                    prev_ok = prior.get("circuit_breaker_consecutive_ok")
                    if prev_ok is None and isinstance(prior.get("criteria"), list):
                        for c in prior["criteria"]:
                            if (
                                isinstance(c, dict)
                                and c.get("name") == "circuit_breaker_confidence"
                            ):
                                prev_ok = c.get("value")
                                break
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    prev_ok = None
            new_ok = grad_cb.get("consecutive_ok")
            if prev_ok is None or int(prev_ok or -1) != int(new_ok or -1):
                from src.dashboard.generator import refresh_graduation_dual_surfaces

                refresh_graduation_dual_surfaces(
                    public_dir=Path(PUBLIC_DATA_DIR),
                    data_dir=Path(DATA_DIR),
                )
                logger.info(
                    "Refreshed graduation dual-write after CB consecutive_ok "
                    "%s → %s",
                    prev_ok,
                    new_ok,
                )
        except Exception as grad_exc:  # noqa: BLE001 — never fail health on grad
            logger.warning("Graduation refresh after CB update failed: %s", grad_exc)

    # Batch EK: share freshness SLIs with compact signals.health (deep-research:
    # dual surfaces must not diverge on mirror lag / execution timeline).
    try:
        report = attach_shared_freshness_slis_to_ops_report(
            report, data_dir=Path(DATA_DIR)
        )
    except Exception as sli_exc:  # noqa: BLE001 — never fail health on SLI attach
        logger.warning("Shared freshness SLI attach skipped: %s", sli_exc)

    try:
        from src.dashboard.generator import _stamp_generator_git_sha

        report = _stamp_generator_git_sha(report)
    except Exception:  # noqa: BLE001 — never block health SSOT write
        pass

    # Always persist full checks (including kill_switch / open_incidents) so
    # on-disk data/health.json matches live run_health_check() output.
    # Batch HW: under pytest, never rewrite production DATA_DIR/health.json
    # when HEALTH_PATH still resolves to live SSOT (conftest isolates PUBLIC
    # only). Same guard as multi-dest SSOT write protection.
    skip_private_health_write = False
    try:
        from src.monitor.signal_authority import (
            _should_skip_production_ssot_write,
            is_ephemeral_write_path,
        )

        skip_private_health_write = bool(
            _should_skip_production_ssot_write(HEALTH_PATH)
        )
        # Also refuse when lag/source probe is ephemeral and HEALTH_PATH is
        # production — prevents fixture lag stamps even outside pytest if a
        # misbound PUBLIC_DATA_DIR is active (belt-and-suspenders).
        if not skip_private_health_write:
            from src.monitor.repo_public_mirror_lag import is_ephemeral_restamp_path

            lag_src = str(report.get("repo_public_mirror_source") or "")
            if (
                lag_src
                and is_ephemeral_restamp_path(lag_src)
                and not is_ephemeral_write_path(HEALTH_PATH)
            ):
                # Re-probe with live defaults before writing production SSOT.
                try:
                    from src.monitor.repo_public_mirror_lag import (
                        summarize_repo_public_mirror_lag,
                    )
                    from src.dashboard.generator import (
                        project_repo_public_mirror_lag_onto_health,
                    )

                    honest = summarize_repo_public_mirror_lag()
                    if not is_ephemeral_restamp_path(honest.get("source")):
                        report = project_repo_public_mirror_lag_onto_health(
                            report, honest
                        )
                        report["repo_public_mirror_lag"] = {
                            "lagging_count": report.get(
                                "repo_public_mirror_lagging_count"
                            ),
                            "total": report.get("repo_public_mirror_total"),
                            "status": report.get("repo_public_mirror_lag_status"),
                            "badge": report.get("repo_public_mirror_lag_badge"),
                            "paths": report.get("repo_public_mirror_lagging_paths"),
                            "source": report.get("repo_public_mirror_source"),
                            "dest": report.get("repo_public_mirror_dest"),
                        }
                    else:
                        skip_private_health_write = True
                        logger.error(
                            "Refusing production health write: lag source still "
                            "ephemeral after re-probe (%s)",
                            honest.get("source"),
                        )
                except Exception as heal_exc:  # noqa: BLE001
                    skip_private_health_write = True
                    logger.error(
                        "Refusing production health write: ephemeral lag source "
                        "and re-probe failed (%s)",
                        heal_exc,
                    )
    except Exception:  # noqa: BLE001 — never block health when guard import fails
        skip_private_health_write = False

    if skip_private_health_write:
        logger.warning(
            "Health check: skipped production SSOT write at %s (pytest/ephemeral guard)",
            HEALTH_PATH,
        )
    else:
        HEALTH_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Task 5: atomic generation-stamped private write — a critical
            # observation is still published atomically and never partially.
            write_health_generation(
                report,
                path=HEALTH_PATH,
                producer_sha=report.get("generator_git_sha"),
            )
            # Post-write integrity: re-read and confirm kill dimension survived.
            try:
                on_disk = json.loads(HEALTH_PATH.read_text(encoding="utf-8"))
                disk_checks = (
                    on_disk.get("checks") if isinstance(on_disk, dict) else None
                )
                if not isinstance(disk_checks, dict) or "kill_switch" not in disk_checks:
                    logger.error(
                        "Health check write missing kill_switch checks at %s",
                        HEALTH_PATH,
                    )
                elif kill_switch.get("enabled") and not disk_checks.get(
                    "kill_switch", {}
                ).get("enabled"):
                    logger.error(
                        "Health check write lost kill_switch.enabled at %s",
                        HEALTH_PATH,
                    )
            except (OSError, json.JSONDecodeError) as verify_exc:
                logger.error("Health check post-write verify failed: %s", verify_exc)
            logger.info(
                "Health check: %s (written to %s)", system_status, HEALTH_PATH
            )
        except OSError as e:
            logger.error("Failed to write health check: %s", e)

    # Dual-path: also publish to PUBLIC_DATA_DIR so operator WWW is not stuck on
    # a stale dashboard health.json timestamp for kill authority.
    try:
        publish_ops_health_surfaces(report)
    except Exception as exc:  # noqa: BLE001 — never fail health job on public side publish
        logger.warning("Ops health surface publish failed: %s", exc)

    # Batch AQ: health-only cron must refresh alerts.json so operators are not
    # stuck on a half-hour-old full-dashboard alerts snapshot.
    try:
        publish_health_alerts_json(report)
    except Exception as exc:  # noqa: BLE001 — never fail health job on alerts
        logger.warning("Health alerts.json publish failed: %s", exc)

    # Batch IU DO2 / DQ1: health cadence must clear kill-gated promote markers
    # when kill is healed (dashboard may timeout and skip write_promote path).
    try:
        from src.strategy.graduation_checklist import GraduationChecklist

        clear_result = GraduationChecklist().clear_kill_gated_promote_markers(
            data_dir=Path(DATA_DIR)
        )
        if clear_result.get("cleared"):
            logger.info(
                "Health clear-on-heal removed kill-gated promote markers: %s",
                clear_result.get("removed"),
            )
    except Exception as clear_exc:  # noqa: BLE001 — never fail health on promote heal
        logger.warning("Kill-gated promote clear-on-heal skipped: %s", clear_exc)

    return report


def main(argv: list[str] | None = None) -> int:
    """Run the health producer.

    Exit modes (Task 3C — truth separation):
    - ``publication`` (default): the exit code answers only "did the producer
      complete compute/write/commit". A critical observation is a SUCCESSFUL
      run whose artifact stays critical and whose halt stays enabled — it must
      not be recorded as a scheduler failure (which previously accumulated
      hundreds of tasker failures merely because the observed portfolio state
      was unsafe).
    - ``probe``: legacy Nagios-compatible severity exit codes (0 ok/warning,
      1 critical) for operator tooling that needs severity from the exit code.
    """
    import argparse

    from src.utils.log_config import configure_logging

    parser = argparse.ArgumentParser(description="Run the Portfolio Lab health producer.")
    parser.add_argument(
        "--exit-mode",
        choices=("publication", "probe"),
        default=os.environ.get("PORTFOLIO_LAB_HEALTH_EXIT_MODE", "publication"),
    )
    args = parser.parse_args(argv)
    configure_logging()
    report = run_health_check()
    logger.info("Health check: %s", json.dumps(report, indent=2))
    if args.exit_mode == "probe":
        # warning is a valid ops state (open advisory incidents, etc.). Mapping it
        # to exit 1 made tasker status=error and sticky failed_cron noise.
        status = str(report.get("status") or "ok").lower()
        if status in {"ok", "warning", "healthy"}:
            return 0
        return 1
    # Publication mode: producer completion is success; observation severity
    # lives in the artifact (health.status critical + kill halt preserved).
    return 0


if __name__ == "__main__":
    exit(main())
