"""Health-check dashboard application / ops-monitor reconciliation
(extracted from src/monitor/health_check.py, Item 5 s1).
"""

import logging
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from src.paths import DATA_DIR, PUBLIC_DATA_DIR

logger = logging.getLogger(__name__)

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
    from src.monitor.health_kill_surfaces import _disk_kill_and_open_incidents, _disk_kill_ssot_is_clear, _elevate_public_system_status, _project_public_kill_fields, enforce_worst_wins_system_status, load_ops_monitor_report
    from src.monitor.health_freshness_cb import load_graduation_cb_ssot, project_graduation_cb_onto_compact_health, project_graduation_cb_onto_report
    report = ops_report if isinstance(ops_report, dict) else load_ops_monitor_report(
        data_dir=data_dir, public_dir=public_dir
    )
    if not report:
        # G7: even without a monitor report, never serve a payload whose
        # system_status understates its own kill/open fields.
        enforce_worst_wins_system_status(health_data)
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

    # G7 (2026-08-11 session B): worst-wins rollup assertion — the final word
    # on system_status after every demotion/elevation branch above, so the
    # public badge can never understate ops / kill / open-incident severity
    # (observed: 09:05:41Z public system_status=healthy while halt active).
    enforce_worst_wins_system_status(health_data)

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
    from src.monitor.health_kill_surfaces import _disk_kill_and_open_incidents
    from src.monitor.health_freshness_cb import project_graduation_cb_onto_compact_health
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
