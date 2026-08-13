"""Health-check kill-surface projection + disk SSOT (extracted from
src/monitor/health_check.py, Item 5 s1 HEALTH-CHECK-SPLIT).
"""

import logging
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from src.paths import DATA_DIR, PUBLIC_DATA_DIR

logger = logging.getLogger(__name__)

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


def enforce_worst_wins_system_status(payload: dict[str, Any]) -> str:
    """G7 (2026-08-11 session B): one rollup assertion for dashboard system_status.

    Public ``system_status`` must equal ``worst(ops_health_status,
    kill_switch, open_incidents)`` on every write. Elevates only (never
    demotes — the worst dimension must surface even when a writer set a
    milder value), then stamps provenance so operators can see the seam
    applied. Mirrors ``elevate_system_status_for_kill`` semantics so a halt
    can never be masked by a milder rollup (observed 09:05:41Z: public
    payload served system_status=healthy while kill halt + open incident
    critical).
    """
    if not isinstance(payload, dict):
        return "healthy"
    current = str(payload.get("system_status") or "healthy")
    ops_status = str(payload.get("ops_health_status") or "ok")
    elevated = _elevate_public_system_status(current, ops_status)
    kill = (
        payload.get("kill_switch")
        if isinstance(payload.get("kill_switch"), dict)
        else {}
    )
    open_inc = (
        payload.get("open_incidents")
        if isinstance(payload.get("open_incidents"), dict)
        else {}
    )
    try:
        from src.dashboard.kill_authority import elevate_system_status_for_kill

        elevated = elevate_system_status_for_kill(elevated, kill, open_inc)
    except ImportError:
        pass
    if elevated != current:
        payload["system_status"] = elevated
        payload["system_status_rollup"] = (
            "worst_wins:ops_health_status,kill_switch,open_incidents"
        )
    return elevated


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
    from src.monitor.health_rollup import _compute_system_status
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
