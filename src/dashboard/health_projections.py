"""Health-projection helpers extracted from ``src.dashboard.generator``.

Module-level health projections (Batch EG/DW cluster: cost-budget + dual-clock
lag, execution-timeline honesty, mirror-lag badge, pending-artifact cron,
re-entry eligibility, voting-mass quality, paper-return SSOT, kill-field
projection, canonical health loading) moved here by Item 8 (2026-08-12).

``generator.py`` re-exports every name below to preserve its public attribute
surface (``signal_section_builder.py`` and ``health_check.py`` resolve these
via the generator module).
"""

import json
from datetime import datetime, timezone
from typing import Any

from src.paths import PUBLIC_DATA_DIR

# generator.py defines PUBLIC_DIR = PUBLIC_DATA_DIR; alias here so the moved
# bodies stay byte-identical.
PUBLIC_DIR = PUBLIC_DATA_DIR

def _parse_rebalance_clock(value: Any) -> datetime | None:
    """Parse controller or order-event timestamps for dual-clock lag."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        # Support trailing Z and bare dates
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        if "T" in text:
            dt = datetime.fromisoformat(text)
        else:
            dt = datetime.strptime(text[:10], "%Y-%m-%d")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (TypeError, ValueError):
        return None


def project_smart_rebalance_budget_onto_health(
    health: dict[str, Any] | None,
    smart_rebalance: dict[str, Any] | None,
    rebalance_health: dict[str, Any] | None = None,
    *,
    clock_lag_warn_days: float = 7.0,
) -> dict[str, Any]:
    """Project cost-budget + dual-clock lag onto compact ``signals.health``.

    Nested ``smart_rebalance`` already carries ``is_over_budget`` /
    ``ytd_cost_bps`` and controller ``last_rebalance``, while
    ``rebalance_health.next_rebalance`` carries order-event
    ``last_execution_at``. Operators reading only compact health missed
    4× annual cost overruns and multi-week controller lag (Batch DW).

    Soft warning only — does not change routing authority
    (``target_allocations`` remains live SSOT).
    """
    if not isinstance(health, dict):
        health = {}
    if not isinstance(smart_rebalance, dict):
        # Explicit clear when panel absent so sticky True cannot persist
        health.setdefault("rebalance_budget_status", "unknown")
        return health

    status_block = (
        smart_rebalance.get("status")
        if isinstance(smart_rebalance.get("status"), dict)
        else {}
    )

    def _float(key: str, *sources: Any) -> float | None:
        for src in sources:
            if not isinstance(src, dict):
                continue
            if key not in src or src.get(key) is None:
                continue
            try:
                return float(src.get(key))
            except (TypeError, ValueError):
                continue
        return None

    ytd_bps = _float("ytd_cost_bps", smart_rebalance, status_block)
    remaining_pct = _float("remaining_budget_pct", smart_rebalance, status_block)
    annual_limit_pct = _float("annual_cost_limit_pct", smart_rebalance)
    if annual_limit_pct is None:
        # Controller status.config uses display string "0.5%"
        cfg = status_block.get("config") if isinstance(status_block.get("config"), dict) else {}
        raw_lim = cfg.get("annual_cost_limit") if isinstance(cfg, dict) else None
        if isinstance(raw_lim, str) and raw_lim.endswith("%"):
            try:
                annual_limit_pct = float(raw_lim[:-1].strip())
            except ValueError:
                annual_limit_pct = None
        elif raw_lim is not None:
            try:
                # Fraction 0.005 → display 0.5
                v = float(raw_lim)
                annual_limit_pct = v * 100.0 if v < 0.1 else v
            except (TypeError, ValueError):
                annual_limit_pct = None
    if annual_limit_pct is None:
        annual_limit_pct = 0.5  # default matches SmartRebalancingController

    is_over = status_block.get("is_over_budget")
    if is_over is None and ytd_bps is not None and annual_limit_pct is not None:
        # limit is percent-of-portfolio (0.5 = 0.5%); ytd_bps/100 = pct points
        is_over = (ytd_bps / 100.0) >= float(annual_limit_pct) - 1e-9
    is_over = bool(is_over) if is_over is not None else False

    is_warn = status_block.get("is_warning")
    if is_warn is None and ytd_bps is not None and annual_limit_pct is not None:
        # warning threshold ~80% of annual limit (matches CostBudgetTracker default)
        is_warn = (ytd_bps / 100.0) >= float(annual_limit_pct) * 0.8 - 1e-9
    is_warn = bool(is_warn) if is_warn is not None else False

    controller_last = status_block.get("last_rebalance") or smart_rebalance.get(
        "last_rebalance"
    )
    next_reb: dict[str, Any] = {}
    if isinstance(rebalance_health, dict):
        nr = rebalance_health.get("next_rebalance")
        if isinstance(nr, dict):
            next_reb = nr
    last_exec_at = next_reb.get("last_execution_at")
    last_exec_clock = next_reb.get("last_execution_clock")

    ctrl_dt = _parse_rebalance_clock(controller_last)
    exec_dt = _parse_rebalance_clock(last_exec_at)
    lag_days: float | None = None
    lagging = False
    if ctrl_dt is not None and exec_dt is not None:
        lag_days = round((exec_dt - ctrl_dt).total_seconds() / 86400.0, 2)
        # Only flag when controller lags event clock (positive lag)
        lagging = lag_days >= float(clock_lag_warn_days)

    if is_over:
        budget_status = "over_budget"
    elif is_warn:
        budget_status = "warning"
    elif ytd_bps is not None:
        budget_status = "ok"
    else:
        budget_status = "unknown"

    health["rebalance_ytd_cost_bps"] = (
        round(ytd_bps, 3) if ytd_bps is not None else None
    )
    health["rebalance_remaining_budget_pct"] = (
        round(remaining_pct, 4) if remaining_pct is not None else None
    )
    health["rebalance_annual_cost_limit_pct"] = round(float(annual_limit_pct), 4)
    health["rebalance_is_over_budget"] = is_over
    health["rebalance_is_warning"] = is_warn or is_over
    health["rebalance_budget_status"] = budget_status
    health["rebalance_controller_last_rebalance"] = (
        str(controller_last) if controller_last else None
    )
    health["rebalance_last_execution_at"] = (
        str(last_exec_at) if last_exec_at else None
    )
    health["rebalance_last_execution_clock"] = (
        str(last_exec_clock) if last_exec_clock else None
    )
    health["rebalance_controller_clock_lag_days"] = lag_days
    health["rebalance_controller_clock_lagging"] = lagging

    # Soft elevate: over-budget or multi-day controller lag → warning
    if (is_over or lagging) and health.get("status") in (
        None,
        "ok",
        "healthy",
        "unknown",
    ):
        health["status"] = "warning"

    return health


def project_execution_timeline_onto_health(
    health: dict[str, Any] | None,
    rebalance_health: dict[str, Any] | None,
    *,
    rewrite_inflate_ratio: float = 2.0,
    rewrite_inflate_min_raw: int = 5,
) -> dict[str, Any]:
    """Project event-day execution timeline honesty onto compact health (Batch EG).

    Daily ``order-history-YYYY-MM-DD.json`` rewrites re-emit the same fills with
    a new write day. Raw parse counts (``raw_history_entries`` / legacy inflated
    ``total_executions``) mislead operators when UI total ≫ unique event days.
    Unique count SLI uses event-day canonical history; rewrite ratio is forensic.
    """
    if not isinstance(health, dict):
        health = {}
    if not isinstance(rebalance_health, dict):
        health.setdefault("rebalance_execution_timeline_status", "unknown")
        return health

    def _int(key: str) -> int | None:
        raw = rebalance_health.get(key)
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    unique_days = _int("canonical_execution_days")
    if unique_days is None:
        unique_days = _int("total_executions")
    raw_entries = _int("raw_history_entries")
    if raw_entries is None:
        # Pre-EG payloads: total_executions was raw; prefer explicit raw when set
        raw_entries = _int("total_executions")
    rewrite_files = _int("snapshot_rewrite_files") or 0

    health["rebalance_unique_execution_days"] = unique_days
    health["rebalance_raw_history_entries"] = raw_entries
    health["rebalance_snapshot_rewrite_files"] = rewrite_files
    health["rebalance_execution_timeline_policy"] = rebalance_health.get(
        "execution_timeline_policy"
    ) or rebalance_health.get("snapshot_rewrite_policy")

    inflated = False
    # G6 (2026-08-11 session B): re-policy — raw snapshot rewrites are
    # forensic-only and bounded by the producer retention cap (14 days of
    # daily order-history-*.json; see rebalance_health.py). The old ratio
    # test (raw >= unique × 2 / rewrites > unique) flagged the intended
    # retention forever (live: raw=116 rewrites=73 vs 5 canonical days).
    # Flag only when rewrite files exceed twice the retention window — i.e.
    # the prune failed and raw history is growing unboundedly again.
    REWRITE_RETENTION_MAX = 2 * 14  # 2 × rebalance_health.DAILY_SNAPSHOT_RETENTION_DAYS
    if rewrite_files > REWRITE_RETENTION_MAX:
        inflated = True

    if inflated:
        status = "rewrite_inflated"
        badge = (
            f"unique={unique_days} raw={raw_entries} rewrites={rewrite_files}"
        )
    elif unique_days is not None:
        status = "ok"
        badge = f"unique={unique_days}"
        if raw_entries is not None and raw_entries != unique_days:
            badge = f"unique={unique_days} raw={raw_entries}"
    else:
        status = "unknown"
        badge = "no_execution_history"

    health["rebalance_execution_timeline_status"] = status
    health["rebalance_execution_timeline_badge"] = badge
    return health


def project_repo_public_mirror_lag_onto_health(
    health: dict[str, Any] | None,
    lag_summary: dict[str, Any] | None,
    *,
    warn_threshold: int = 1,
    critical_threshold: int = 10,
) -> dict[str, Any]:
    """Project repo ``public/data`` mirror lag onto compact health (Batch EJ).

    Operator ``PUBLIC_DATA_DIR`` is SoT; repo ``public/data`` is a derived
    static mirror (``make mirror-repo-public-data``). Deep-research: expose
    ``mirror_lagging_count`` as a freshness gauge so lag cannot hide behind
    green cron while the checkout mirror drifts (historical 28–32/32 lag).

    ``lag_summary`` shape (from ``summarize_repo_public_mirror_lag``)::

        {
          "lagging_count": int,
          "total": int,
          "lagging_paths": list[str],  # optional, capped
          "source": str,
          "dest": str,
        }
    """
    if not isinstance(health, dict):
        health = {}
    if not isinstance(lag_summary, dict):
        health.setdefault("repo_public_mirror_lag_status", "unknown")
        return health

    try:
        lagging = int(lag_summary.get("lagging_count") or 0)
    except (TypeError, ValueError):
        lagging = 0
    try:
        total = int(lag_summary.get("total") or 0)
    except (TypeError, ValueError):
        total = 0

    paths = lag_summary.get("lagging_paths")
    if not isinstance(paths, list):
        paths = []
    paths = [str(p) for p in paths[:12]]

    health["repo_public_mirror_lagging_count"] = lagging
    health["repo_public_mirror_total"] = total
    health["repo_public_mirror_lagging_paths"] = paths
    # Batch HW: never stamp pytest isolation paths onto health SLI source/dest
    # (private data/health.json pollution → false-green lag under make test).
    raw_source = lag_summary.get("source")
    raw_dest = lag_summary.get("dest")
    try:
        from src.monitor.repo_public_mirror_lag import is_ephemeral_restamp_path
    except Exception:  # noqa: BLE001
        is_ephemeral_restamp_path = None  # type: ignore[assignment]
    if raw_source:
        src_s = str(raw_source)
        if is_ephemeral_restamp_path is None or not is_ephemeral_restamp_path(src_s):
            health["repo_public_mirror_source"] = src_s
        else:
            # Keep prior honest source if present; else omit ephemeral stamp.
            if is_ephemeral_restamp_path(
                health.get("repo_public_mirror_source")
            ):
                health.pop("repo_public_mirror_source", None)
    if raw_dest:
        dst_s = str(raw_dest)
        if is_ephemeral_restamp_path is None or not is_ephemeral_restamp_path(dst_s):
            health["repo_public_mirror_dest"] = dst_s
        else:
            if is_ephemeral_restamp_path(health.get("repo_public_mirror_dest")):
                health.pop("repo_public_mirror_dest", None)

    if lagging >= int(critical_threshold):
        status = "critical"
        badge = f"lagging={lagging}/{total}"
    elif lagging >= int(warn_threshold):
        status = "lagging"
        badge = f"lagging={lagging}/{total}"
    elif total > 0:
        status = "ok"
        badge = f"lagging=0/{total}"
    else:
        status = "unknown"
        badge = "no_catalog"

    health["repo_public_mirror_lag_status"] = status
    health["repo_public_mirror_lag_badge"] = badge
    health["repo_public_mirror_lag_policy"] = (
        "PUBLIC_DATA_DIR is SoT; repo public/data is derived static mirror "
        "(make mirror-repo-public-data). Count is bytes-unequal or dest-missing."
    )
    # Soft elevate only — mirror lag is ops hygiene, not trading halt
    if status in ("lagging", "critical") and health.get("status") in (
        None,
        "ok",
        "healthy",
        "unknown",
    ):
        health["status"] = "warning"
    return health


def project_pending_artifact_cron_onto_health(
    health: dict[str, Any] | None,
    cron_jobs: list | None,
) -> dict[str, Any]:
    """Project dual-signal pending/artifact reconcile onto compact health.

    Raw ``cron_status.json`` may still show ``status=pending`` for weekly
    tasker jobs (e.g. portfolio-lab-fetch-trends) while Batch DT already
    soft-oks via fresh producer artifact (google_trends.json). Compact health
    previously only had job counts — operators saw false "never run" noise.

    Does not elevate status for true pending_never_run alone (weekly schedule
    is expected). Disclosure only.
    """
    if not isinstance(health, dict):
        health = {}
    if not isinstance(cron_jobs, list):
        health.setdefault("cron_pending_artifact_status", "unknown")
        return health

    reconciled: list[str] = []
    true_pending: list[str] = []
    samples: list[str] = []

    for job in cron_jobs:
        if not isinstance(job, dict):
            continue
        if not job.get("enabled", True) or job.get("manual_only"):
            continue
        if job.get("state") in {"manual_only", "paused"}:
            continue
        name = str(job.get("name") or job.get("id") or "")
        if job.get("pending_artifact_reconciled"):
            reconciled.append(name or "unknown")
            ev = job.get("pending_artifact_evidence")
            if isinstance(ev, dict) and ev.get("artifact"):
                samples.append(f"{name}:{ev.get('artifact')}")
            elif job.get("heartbeat_disclosure"):
                samples.append(str(job.get("heartbeat_disclosure"))[:120])
        elif (
            str(job.get("status") or "").lower() == "pending"
            and not job.get("last_run")
        ):
            true_pending.append(name or "unknown")

    n_rec = len(reconciled)
    n_pend = len(true_pending)
    if n_rec == 0 and n_pend == 0:
        status = "none"
    elif n_rec > 0 and n_pend == 0:
        status = "reconciled"
    elif n_rec == 0 and n_pend > 0:
        status = "pending_never_run"
    else:
        status = "mixed"

    health["cron_pending_artifact_reconciled_jobs"] = n_rec
    health["cron_pending_never_run_jobs"] = n_pend
    health["cron_pending_artifact_reconciled_names"] = (
        ",".join(reconciled[:8]) if reconciled else None
    )
    health["cron_pending_never_run_names"] = (
        ",".join(true_pending[:8]) if true_pending else None
    )
    health["cron_pending_artifact_sample"] = samples[0] if samples else None
    health["cron_pending_artifact_status"] = status
    health["cron_pending_artifact_badge"] = (
        f"artifact_ok={n_rec} pending_never_run={n_pend}"
    )
    return health


def project_reentry_eligibility_onto_health(
    health: dict[str, Any] | None,
    ensemble_voting: dict[str, Any] | None,
) -> dict[str, Any]:
    """Project multi-horizon reentry eligibility onto compact health (Batch ED).

    Nested ``health_metrics.reentry`` already carries multi-horizon hysteresis
    (eligible / blocked_reason / no_force_wake). Operators reading only compact
    health missed eligible sleepers (MSM/INTL/VIXTS) vs blocked (ALT/CARA).

    Disclosure only — does **not** force-wake or change routing authority.
    """
    if not isinstance(health, dict):
        health = {}
    if not isinstance(ensemble_voting, dict):
        health.setdefault("ensemble_reentry_status", "unknown")
        return health

    eligible: list[str] = []
    blocked: list[str] = []
    blocked_reasons: list[str] = []
    policy = "multi_horizon_hysteresis_no_force_wake"
    tracked = 0

    for row in ensemble_voting.get("configured_source_status") or []:
        if not isinstance(row, dict):
            continue
        src = str(row.get("source") or "")
        if not src:
            continue
        hm = row.get("health_metrics") if isinstance(row.get("health_metrics"), dict) else {}
        re = hm.get("reentry") if isinstance(hm.get("reentry"), dict) else {}
        # Prefer nested reentry block; fall back to flat flags
        if "reentry_eligible" in re:
            elig = bool(re.get("reentry_eligible"))
            tracked += 1
            if re.get("policy"):
                policy = str(re.get("policy"))
            if elig:
                eligible.append(src)
            else:
                blocked.append(src)
                br = re.get("reentry_blocked_reason")
                if br:
                    blocked_reasons.append(f"{src}:{br}")
        elif hm.get("reentry_eligible") is not None:
            tracked += 1
            if bool(hm.get("reentry_eligible")):
                eligible.append(src)
            else:
                blocked.append(src)
        elif row.get("reentry_eligible") is not None:
            tracked += 1
            if bool(row.get("reentry_eligible")):
                eligible.append(src)
            else:
                blocked.append(src)

    slept = ensemble_voting.get("health_gate_slept")
    slept_n = len(slept) if isinstance(slept, dict) else 0

    if tracked == 0:
        status = "unknown"
    elif eligible:
        status = "eligible_pending"
    else:
        status = "none_eligible"

    health["ensemble_reentry_eligible_count"] = len(eligible)
    health["ensemble_reentry_blocked_count"] = len(blocked)
    health["ensemble_reentry_tracked_count"] = tracked
    health["ensemble_reentry_eligible_sources"] = (
        ",".join(sorted(eligible)) if eligible else None
    )
    health["ensemble_reentry_blocked_sources"] = (
        ",".join(sorted(blocked)) if blocked else None
    )
    health["ensemble_reentry_blocked_sample"] = (
        blocked_reasons[0] if blocked_reasons else None
    )
    health["ensemble_reentry_status"] = status
    health["ensemble_reentry_policy"] = policy
    health["ensemble_reentry_slept_count"] = slept_n
    health["ensemble_reentry_badge"] = (
        f"reentry_eligible={len(eligible)}/{tracked} "
        f"blocked={len(blocked)} policy=no_force_wake"
    )
    # Never elevate status solely for eligible-pending — wake is human/natural
    return health


def project_voting_mass_quality_onto_health(
    health: dict[str, Any] | None,
    ensemble_voting: dict[str, Any] | None,
    *,
    soft_floor_mass_warn: float = 0.50,
) -> dict[str, Any]:
    """Project voting-mass quality (soft-floor vs healthy) onto compact health.

    Source-count badges (e.g. 1/9 healthy) can greenwash when the only healthy
    source is zero_baseline non-voting and 100% of ``active_weights`` sit on
    soft_floor (Batch EC live shape). Portfolio SLI = soft-floor mass share
    of contributing vote weight.
    """
    if not isinstance(health, dict):
        health = {}
    if not isinstance(ensemble_voting, dict):
        health.setdefault("ensemble_voting_quality_status", "unknown")
        return health

    aw = ensemble_voting.get("active_weights")
    if not isinstance(aw, dict) or not aw:
        # Fall back to configured_source_status active_weight
        aw = {}
        for row in ensemble_voting.get("configured_source_status") or []:
            if not isinstance(row, dict) or not row.get("contributing"):
                continue
            try:
                w = float(row.get("active_weight") or 0)
            except (TypeError, ValueError):
                w = 0.0
            if w > 0:
                aw[str(row.get("source"))] = w

    soft_map = ensemble_voting.get("health_gate_soft_floor")
    if not isinstance(soft_map, dict):
        soft_map = {}
    soft_keys = {str(k) for k in soft_map.keys()}

    # Also treat status active_soft_floor as soft-floor mass
    status_by_source: dict[str, str] = {}
    health_status_by_source: dict[str, str] = {}
    for row in ensemble_voting.get("configured_source_status") or []:
        if not isinstance(row, dict):
            continue
        src = str(row.get("source") or "")
        if not src:
            continue
        status_by_source[src] = str(row.get("status") or "")
        hm = row.get("health_metrics")
        if isinstance(hm, dict) and hm.get("status"):
            health_status_by_source[src] = str(hm.get("status")).lower()
        if status_by_source[src] == "active_soft_floor":
            soft_keys.add(src)

    total = 0.0
    soft_mass = 0.0
    healthy_mass = 0.0
    soft_count = 0
    healthy_contrib = 0
    for src, w_raw in aw.items():
        try:
            w = float(w_raw)
        except (TypeError, ValueError):
            continue
        if w <= 0:
            continue
        total += w
        src_s = str(src)
        if src_s in soft_keys or status_by_source.get(src_s) == "active_soft_floor":
            soft_mass += w
            soft_count += 1
        elif health_status_by_source.get(src_s) == "healthy":
            healthy_mass += w
            healthy_contrib += 1
        elif status_by_source.get(src_s) in ("active", "active_ok", ""):
            # No health metrics: treat non-soft contributing as non-soft mass
            if src_s not in soft_keys:
                healthy_mass += w
                healthy_contrib += 1

    soft_share = round(soft_mass / total, 5) if total > 0 else 0.0
    healthy_share = round(healthy_mass / total, 5) if total > 0 else 0.0

    if total <= 0:
        quality = "no_vote_mass"
    elif soft_share >= 0.999:
        quality = "soft_floor_dominant"
    elif soft_share >= float(soft_floor_mass_warn):
        quality = "soft_floor_heavy"
    elif healthy_share >= 0.5:
        quality = "ok"
    else:
        quality = "mixed"

    slept = ensemble_voting.get("health_gate_slept")
    slept_n = len(slept) if isinstance(slept, dict) else 0
    contrib_n = ensemble_voting.get("contributing_source_count")
    try:
        contrib_n_i = int(contrib_n) if contrib_n is not None else len(aw)
    except (TypeError, ValueError):
        contrib_n_i = len(aw)

    health["ensemble_voting_soft_floor_mass"] = soft_share
    health["ensemble_voting_soft_floor_count"] = soft_count
    health["ensemble_voting_healthy_mass"] = healthy_share
    health["ensemble_voting_healthy_contributors"] = healthy_contrib
    health["ensemble_voting_contributing_count"] = contrib_n_i
    health["ensemble_voting_slept_count"] = slept_n
    health["ensemble_voting_quality_status"] = quality
    health["ensemble_voting_quality_badge"] = (
        f"soft_floor={soft_share:.0%}/vote healthy_contrib={healthy_contrib}"
    )

    if quality in ("soft_floor_dominant", "soft_floor_heavy") and health.get(
        "status"
    ) in (None, "ok", "healthy", "unknown"):
        health["status"] = "warning"

    return health


def project_paper_return_ssot_onto_health(
    health: dict[str, Any] | None,
    comparison: dict[str, Any] | None,
) -> dict[str, Any]:
    """Project five-surface paper return SSOT agreement onto compact health.

    Write authority is ``daily_pnl.jsonl`` / ``daily_pnl_latest.json``. Other
    surfaces (portfolio history, unified dashboard, stats, paper-trading-
    performance) must match session NAV/return within epsilon (Batch EB / c358).
    Soft warning on disagreement — does not change routing authority.
    """
    if not isinstance(health, dict):
        health = {}
    if not isinstance(comparison, dict):
        health.setdefault("paper_return_ssot_status", "unknown")
        health.setdefault("paper_return_ssot_agree", None)
        return health

    agree = comparison.get("agree")
    ssot = comparison.get("ssot") if isinstance(comparison.get("ssot"), dict) else {}
    disagreements = comparison.get("disagreements")
    if not isinstance(disagreements, list):
        disagreements = []
    surfaces = comparison.get("surfaces")
    surface_names: list[str] = []
    if isinstance(surfaces, list):
        for s in surfaces:
            if isinstance(s, dict) and s.get("surface"):
                surface_names.append(str(s.get("surface")))
    elif isinstance(surfaces, dict):
        surface_names = [str(k) for k in surfaces.keys()]

    disagree_surfaces = []
    for d in disagreements:
        if isinstance(d, dict) and d.get("surface"):
            disagree_surfaces.append(str(d.get("surface")))

    health["paper_return_ssot_agree"] = bool(agree) if agree is not None else None
    if agree is True:
        health["paper_return_ssot_status"] = "ok"
    elif agree is False:
        health["paper_return_ssot_status"] = "disagree"
    else:
        health["paper_return_ssot_status"] = "unknown"
    health["paper_return_ssot_date"] = ssot.get("date")
    health["paper_return_ssot_nav"] = ssot.get("total_value")
    health["paper_return_ssot_daily_return"] = ssot.get("daily_return")
    health["paper_return_ssot_source"] = ssot.get("return_source")
    health["paper_return_ssot_disagreement_count"] = len(disagreements)
    health["paper_return_ssot_surfaces"] = ",".join(disagree_surfaces[:8]) or None
    if disagree_surfaces:
        health["paper_return_ssot_why"] = (
            str(disagreements[0].get("why_not"))
            if disagreements and isinstance(disagreements[0], dict)
            else "disagree"
        )
    else:
        health["paper_return_ssot_why"] = None

    if agree is False and health.get("status") in (
        None,
        "ok",
        "healthy",
        "unknown",
    ):
        health["status"] = "warning"

    return health


def _apply_kill_to_smart_rebalance(
    smart: dict[str, Any] | None,
    kill_payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Annotate smart_rebalance with kill halt and force non-execute under kill.

    Keeps drift/VPIN diagnostics visible but never implies actionable execute
    when authority kill_switch.json is enabled (same gate as order_router).
    """
    if not isinstance(smart, dict):
        return smart
    try:
        from src.dashboard.kill_authority import is_kill_execution_blocked
    except ImportError:
        def is_kill_execution_blocked(p):
            return bool(isinstance(p, dict) and p.get("enabled"))

    if not is_kill_execution_blocked(kill_payload):
        # Explicit clear fields when kill is off (stable schema for consumers)
        smart.setdefault("execution_blocked", False)
        smart.setdefault("kill_switch_enabled", False)
        return smart

    level = None
    reason = None
    incident_id = None
    message = None
    if isinstance(kill_payload, dict):
        level = kill_payload.get("level")
        reason = kill_payload.get("reason")
        incident_id = kill_payload.get("incident_id")
        message = kill_payload.get("message")

    smart["execution_blocked"] = True
    smart["kill_switch_enabled"] = True
    if level is not None:
        smart["kill_switch_level"] = level
    if reason is not None:
        smart["kill_switch_reason"] = reason
    if incident_id is not None:
        smart["kill_switch_incident_id"] = incident_id
    if message is not None:
        smart["kill_switch_message"] = message

    # Force non-actionable decision; preserve original decision for operators
    prior_decision = smart.get("decision")
    smart["should_execute"] = False
    smart["decision"] = "blocked_kill_switch"
    human = message if isinstance(message, str) and message.strip() else reason
    smart["reason"] = (
        f"blocked_by_kill_switch:{level or 'enabled'}"
        + (f" ({human})" if human else "")
        + (f"; prior={prior_decision}" if prior_decision else "")
    )
    return smart


def _remaining_budget_ratio(metadata: dict[str, Any], status: dict[str, Any]) -> float:
    """Return remaining rebalance budget as a fraction of portfolio value."""
    value = metadata.get("remaining_budget_ratio")
    if value is None:
        value = metadata.get("remaining_budget_pct")
    if value is None:
        value = status.get("remaining_budget_ratio")
    if value is None and status.get("remaining_budget_pct") is not None:
        value = status.get("remaining_budget_pct") / 100
    if value is None:
        return 1.0
    return round(float(value), 6)


def _remaining_budget_display_pct(ratio: float, status: dict[str, Any]) -> float:
    """Return remaining rebalance budget in display percent units."""
    status_pct = status.get("remaining_budget_pct")
    if status_pct is not None:
        return float(status_pct)
    return round(ratio * 100, 3)


def _load_canonical_health_report() -> dict[str, Any] | None:
    """Load canonical health.json when already published."""
    # Defer to the generator module's PUBLIC_DIR at call time: tests and the
    # runtime mirror-lag seam patch ``src.dashboard.generator.PUBLIC_DIR``;
    # import-time aliasing here would silently break those patch targets.
    from src.dashboard import generator as _generator

    health_path = _generator.PUBLIC_DIR / "health.json"
    try:
        with health_path.open(encoding="utf-8") as handle:
            report = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return report if isinstance(report, dict) else None


