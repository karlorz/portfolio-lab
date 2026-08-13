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
from typing import Any

from src.paths import DATA_DIR, PUBLIC_DATA_DIR
from src.monitor.alerting import AlertChannel, AlertLevel, send_alert, webhook_config_state
from src.monitor.hermes_cron import (
    is_health_self_job,
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

    webhook_configured, webhook_source = webhook_config_state()
    output: dict[str, Any] = {
        "alerts": alerts,
        "count": len(alerts),
        "generated_at": stamp,
        "source": "health_check_job",
        "health_generated_at": health_payload.get("generated_at")
        or health_payload.get("timestamp"),
        "alerting": {
            "webhook_configured": webhook_configured,
            "webhook_source": webhook_source,
        },
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

from src.monitor.health_kill_surfaces import (  # noqa: E402, F401  # re-export hub
    health_ops_path,
    _project_public_kill_fields,
    _elevate_public_system_status,
    enforce_worst_wins_system_status,
    _is_monitor_health_report,
    load_ops_monitor_report,
    _patch_monitor_report_kill_open,
    _atomic_write_json_path,
    _new_generation_id,
    _atomic_write_json_text,
    write_health_generation,
    commit_public_index,
    reconcile_monitor_health_with_disk_ssot,
    project_disk_kill_open_to_all_surfaces,
    _disk_kill_ssot_is_clear,
    _disk_kill_and_open_incidents,
)
from src.monitor.health_dashboard_apply import (  # noqa: E402, F401  # re-export hub
    apply_ops_monitor_to_dashboard_health,
    _project_mirror_lag_onto_dashboard_health,
    refresh_signals_health_kill_fields,
)
from src.monitor.health_freshness_cb import (  # noqa: E402, F401  # re-export hub
    _resolve_freshness_public_root,
    _freshness_artifact_check,
    _check_data_freshness,
    _check_circuit_breaker,
    update_graduation_circuit_breaker_state,
    load_graduation_cb_ssot,
    project_graduation_cb_onto_report,
    project_graduation_cb_onto_compact_health,
    reconcile_graduation_cb_projection,
)
from src.monitor.health_rollup import (  # noqa: E402, F401  # re-export hub
    _check_fred_md_cache,
    _load_json_file,
    _check_kill_switch,
    _check_open_incidents,
    _status_for_system_rollup,
    _compute_system_status,
    attach_shared_freshness_slis_to_ops_report,
    _stamp_health_self_job_running_success,
)

if __name__ == "__main__":
    exit(main())
