#!/usr/bin/env python3
"""
Portfolio-Lab Alpha: Dashboard Generator
Creates static dashboard from SQLite data for Vite/React app consumption.
"""

import json
import sqlite3
import logging
import os
import shutil
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from src.paths import BASE_ALLOCATION, YIELDS_JSON, DATA_DIR, PUBLIC_DATA_DIR, MARKET_DB, REGIME_OVERRIDES, sqlite_connect
from src.strategy.regime_allocation import (
    get_regime_allocation_with_override,
    normalize_allocation_regime,
)
from src.utils import safe_get, classify_vix_regime
from src.backtest.metrics import save_results_json
from src.dashboard.public_data_index import build_public_data_index
from src.monitor.hermes_cron import resolve_hermes_cron_jobs_path
from src.monitor.signal_schemas import validate_all_signals, validate_signal
from src.dashboard.cron_scheduler_section import build_cron_scheduler_section
from src.dashboard.data_freshness_section import build_data_freshness_section
from src.dashboard.health_report import (
    derive_system_status,
    summarize_stale_symbol_count,
)
from src.dashboard.data_pipeline_slo_section import build_data_pipeline_slo_section
from src.dashboard.health_slo_alerts import build_health_slo_alerts
from src.dashboard.kill_authority import (
    allocation_roles_under_kill,
    build_kill_switch_alert,
    elevate_system_status_for_kill,
    load_kill_switch_payload,
    load_open_incidents_summary,
    project_compact_kill_fields,
    project_kill_switch_fields,
)
from src.dashboard.signal_health_section import (
    build_fred_readiness_section,
    build_signal_health_section,
)

__all__ = [
    "DashboardGenerator",
    "PUBLIC_DIR",
    "DB_PATH",
    "project_alternative_data_signal",
    "project_smart_rebalance_budget_onto_health",
    "project_paper_return_ssot_onto_health",
    "project_voting_mass_quality_onto_health",
    "project_reentry_eligibility_onto_health",
    "project_pending_artifact_cron_onto_health",
    "project_execution_timeline_onto_health",
    "project_repo_public_mirror_lag_onto_health",
]

# Legacy flat keys (pre seven-component producer) → panel component names
_ALT_DATA_LEGACY_COMPONENT_KEYS = (
    ("earnings", "earnings_sentiment", "earnings_confidence"),
    ("news", "news_sentiment", "news_confidence"),
    ("jobs", "jobs_signal", "jobs_confidence"),
    ("social", "social_sentiment", "social_confidence"),
)


def project_alternative_data_signal(alt_data_raw: Dict[str, Any]) -> Dict[str, Any]:
    """Project producer alternative_data_latest.json into signals.json shape.

    Primary path: ``raw_data.components`` / ``component_confidences`` / ``weights``
    (current seven-component producer). Fallback: legacy flat
    earnings/news/jobs/social keys when the components map is absent.
    """
    raw = alt_data_raw.get("raw_data") if isinstance(alt_data_raw, dict) else None
    if not isinstance(raw, dict):
        raw = {}

    components_src = raw.get("components")
    confidences_src = raw.get("component_confidences") if isinstance(raw.get("component_confidences"), dict) else {}
    weights_src = raw.get("weights") if isinstance(raw.get("weights"), dict) else {}
    components: Dict[str, Dict[str, Any]] = {}

    if isinstance(components_src, dict) and components_src:
        for key, score in components_src.items():
            components[str(key)] = {
                "score": score,
                "confidence": confidences_src.get(key),
                "weight": weights_src.get(key),
            }
    else:
        # Legacy fallback only when producer lacks the components map
        for name, score_key, conf_key in _ALT_DATA_LEGACY_COMPONENT_KEYS:
            components[name] = {
                "score": raw.get(score_key),
                "confidence": raw.get(conf_key),
                "weight": weights_src.get(name),
            }

    alt_regime = alt_data_raw.get("regime")
    return {
        # Keep ``regime`` for backward compat but mark as advisory shadow so it
        # cannot be read as peer-level live regime_authority (VIX classify).
        "regime": alt_regime,
        "alt_regime": alt_regime,
        "role": "advisory_shadow",
        "live_authoritative": False,
        "canonical_controller": "signals.json.regime / regime_authority",
        "probability": alt_data_raw.get("probability"),
        "confidence": alt_data_raw.get("confidence"),
        "timestamp": alt_data_raw.get("timestamp"),
        "components": components,
        "composite_score": raw.get("composite_score"),
        "z_score": raw.get("z_score"),
        "sources_count": raw.get("sources_count"),
        "data_freshness_hours": raw.get("data_freshness_hours"),
        "producer_path": "data/signals/alternative_data_latest.json",
    }


def load_alternative_data_producer_timestamp(
    data_dir: Path | None = None,
) -> str | None:
    """Return producer alternative_data_latest.json timestamp when readable."""
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    path = root / "signals" / "alternative_data_latest.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    ts = payload.get("timestamp") or payload.get("generated_at")
    return ts if isinstance(ts, str) and ts.strip() else None


def _apply_partial_patch_git_sha_honesty(
    payload: Dict[str, Any],
    *,
    patch_source: str,
) -> None:
    """Clear sticky full-generation git sha on partial section rewrites.

    Partial writers advance ``generated_at`` / ``content_patched_at`` but leave
    ``generator_git_sha`` from the last full dashboard run. Operators then
    attribute a partial patch to a wrong code tip. Keep the prior full-run sha
    under ``last_full_generator_git_sha`` for lag forensics; null the live stamp
    and disclose ``generator_git_sha_status=partial_patch``.
    """
    prior = payload.get("generator_git_sha")
    if prior is not None and prior != "":
        payload.setdefault("last_full_generator_git_sha", prior)
    payload["generator_git_sha"] = None
    payload["generator_git_sha_status"] = "partial_patch"
    payload["generator_git_sha_reason"] = (
        f"cleared by partial rewrite ({patch_source}); "
        "not a full dashboard generation"
    )


def _enrich_duration_allocation_provenance(
    payload: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Ensure duration_allocation never publishes bare weights without role/unit.

    Live partial patches and legacy consumers have left only ``{tlt,ief,shy,bil}``
    which looks like a live sleeve without advisory disclosure.
    """
    if not isinstance(payload, dict) or not payload:
        return payload
    out = dict(payload)
    # Collect weight symbols if nested under weights or flat
    weights = out.get("weights")
    if not isinstance(weights, dict):
        weights = {
            k: out[k]
            for k in ("tlt", "ief", "shy", "bil")
            if isinstance(out.get(k), (int, float))
        }
        if weights:
            out["weights"] = weights
    if weights:
        try:
            out["sum"] = round(sum(float(v) for v in weights.values()), 4)
        except (TypeError, ValueError):
            pass
    out.setdefault("unit", "portfolio_weight_fraction")
    out.setdefault("live_authoritative", False)
    out.setdefault("role", "advisory_sleeve")
    out.setdefault(
        "description",
        "Bond duration sleeve from 2s10s regime table; "
        "not target_allocations / order-routing authority",
    )
    out.setdefault("source", "yield_curve_regime_table")
    return out


def refresh_public_alternative_data_projection(
    *,
    data_dir: Path | None = None,
    public_dir: Path | None = None,
) -> bool:
    """Bounded refresh: rewrite alternative_data on multi-dest signals.json.

    Called after producer save so operators do not wait for the next full
    dashboard cron when only alt-data was updated. Fan-out is single-payload
    same-bytes to PUBLIC + private DATA_DIR + repo soft-mirror (authority gate
    requires target_allocations). Returns True when the public artifact was
    updated.
    """
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    public = Path(public_dir) if public_dir is not None else PUBLIC_DIR
    producer = root / "signals" / "alternative_data_latest.json"
    signals_path = public / "signals.json"
    private_path = root / "signals.json"
    if not producer.exists() or not signals_path.exists():
        return False
    try:
        alt_raw = json.loads(producer.read_text(encoding="utf-8"))
        signals = json.loads(signals_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("Alt-data projection refresh skipped (read failed): %s", exc)
        return False
    if not isinstance(signals, dict) or not isinstance(alt_raw, dict):
        return False

    # Recover authority from private twin if public was hollow mid-window.
    try:
        from src.monitor.signal_authority import (
            AuthorityValidationError,
            validate_authority_payload,
            write_signals_multi_dest,
        )

        try:
            validate_authority_payload(signals)
        except AuthorityValidationError:
            if private_path.is_file():
                try:
                    priv = json.loads(private_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                    priv = None
                if isinstance(priv, dict) and isinstance(
                    priv.get("target_allocations"), dict
                ):
                    signals["target_allocations"] = priv["target_allocations"]
            try:
                validate_authority_payload(signals)
            except AuthorityValidationError as exc:
                logger.error(
                    "Alt-data projection refused (no live authority TA): %s",
                    exc,
                )
                return False
    except ImportError as exc:
        logger.warning("signal_authority unavailable for alt-data refresh: %s", exc)
        return False

    projected = project_alternative_data_signal(alt_raw)
    signals["alternative_data"] = projected
    # Recompute only alt-related staleness hints on the embedded block is hard
    # without full generator; stamp lag metadata for operators.
    now_utc = datetime.now(timezone.utc).isoformat()
    signals["alternative_data_projection"] = {
        "refreshed_at": now_utc,
        "producer_timestamp": projected.get("timestamp"),
        "source": "bounded_alt_data_refresh",
    }
    # Partial rewrite must advance top-level generated_at (mtime honesty)
    # and clear sticky full-generation git sha (partial ≠ full dashboard run).
    signals["generated_at"] = now_utc
    signals["content_patched_at"] = now_utc
    signals["content_patch_source"] = "bounded_alt_data_refresh"
    _apply_partial_patch_git_sha_honesty(signals, patch_source="bounded_alt_data_refresh")
    try:
        result = write_signals_multi_dest(
            signals,
            public_path=signals_path,
            private_path=private_path,
            soft_mirror_repo=True,
        )
    except AuthorityValidationError as exc:
        logger.error("Alt-data projection authority gate refused write: %s", exc)
        return False
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Alt-data projection refresh write failed: %s", exc)
        return False
    if not result.wrote_public:
        logger.warning("Alt-data projection refresh wrote no public dest")
        return False
    logger.info(
        "Refreshed public alternative_data projection at %s (producer_ts=%s; "
        "private=%s repo=%s)",
        signals_path,
        projected.get("timestamp"),
        result.wrote_private,
        result.wrote_repo,
    )
    return True

# Map ensemble signal source names to staleness check keys
_ENSEMBLE_STALENESS_MAP = {
    "multi_speed_momentum": "ensemble_voting",
    "cross_asset_rv": "ensemble_voting",
    "international_momentum": "ensemble_voting",
    "alternative_data": "alternative_data",
    "cross_asset_regime_arb": "ensemble_voting",
    "unified_overlay": "ensemble_voting",
}

logger = logging.getLogger(__name__)

PUBLIC_DATA_DIST_MIRROR_FILES = ("source_manifest.json", "index.json", "health.json")


def _is_predictive_fred_macro(fred: Any) -> bool:
    """Return true when FRED macro has observed inputs suitable for IC staging."""
    if not isinstance(fred, dict) or fred.get("confidence") is None:
        return False
    indicators = fred.get("indicators")
    if not isinstance(indicators, dict) or not indicators:
        return False
    if fred.get("indicators_observed") is False:
        return False
    unavailable_values = {"unavailable", "missing", "empty", "failed", "degraded", "fallback"}
    fallback_modes = {"unavailable", "synthetic", "last_good", "fallback"}
    source_mode = str(fred.get("source_mode", "")).lower()
    status = str(fred.get("status", "")).lower()
    cache_status = str(fred.get("cache_status", "")).lower()
    if source_mode in fallback_modes:
        return False
    if status in unavailable_values:
        return False
    return cache_status not in unavailable_values


def _source_manifest_row_for(public_dir: Path, artifact_name: str) -> dict[str, Any] | None:
    """Return the compact source-manifest row for a public data artifact."""
    manifest_path = public_dir / "source_manifest.json"
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return None

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        return None

    for row in artifacts:
        if not isinstance(row, dict):
            continue
        candidates = {
            row.get("artifact"),
            row.get("filename"),
            row.get("path"),
        }
        if artifact_name in candidates:
            return row
    return None


def _yield_source_provenance(public_dir: Path) -> dict[str, Any]:
    """Map yields source-manifest metadata into the yield curve payload."""
    row = _source_manifest_row_for(public_dir, "yields.json")
    if row is None:
        return {}
    return {
        "source_mode": row.get("source_mode"),
        "source_status": row.get("status"),
        "source_reason": row.get("failure_reason") or row.get("reason"),
        "source_provider": row.get("provider"),
        "source_generated_at": row.get("generated_at"),
        "source_latest_observation": row.get("latest_observation"),
    }


def _first_known_value(*values: Any, default: str = "unknown") -> Any:
    """Return the first non-empty metadata value, treating 'unknown' as absent."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.lower() in {"", "unknown"}:
            continue
        return value
    return default

# Common exception types caught when signal generators fail.
# ValueError/TypeError indicate likely bugs — callers should log these at
# error level. ImportError/AttributeError indicate missing deps — warning.
SIGNAL_EXCEPTIONS = (
    ImportError, AttributeError, KeyError,
    ValueError, TypeError, RuntimeError, OSError,
)

# Lighter exception tuple for monitoring/utility modules that don't
# touch external data structures (no AttributeError/KeyError risk).
MONITOR_EXCEPTIONS = (ImportError, ValueError, OSError, RuntimeError)

# Exceptions that indicate likely bugs rather than missing dependencies.
_BUG_EXCEPTIONS = (ValueError, TypeError)


def _attach_signal_metadata(output: Dict, *, generated_at: str | None = None) -> Dict:
    """Attach dashboard-level generation timestamps to a signals payload."""
    timestamp = generated_at or datetime.now(timezone.utc).isoformat()
    enriched = dict(output)
    enriched["generated_at"] = timestamp
    enriched.setdefault("timestamp", timestamp)
    return enriched


def _generator_git_sha_short() -> str | None:
    """Short HEAD for operator lag detection (code vs projected artifact)."""
    try:
        from src.monitor.decision_registry import _git_sha_short

        return _git_sha_short()
    except Exception:
        return None


def _stamp_generator_git_sha(
    payload: Dict[str, Any],
    *,
    status: str = "full_generate",
) -> Dict[str, Any]:
    """Attach generator_git_sha when available (stats/analytics/graduation/overlay).

    Batch BJ residual honesty (SLSA-style prior identity retention):
    when the new tip differs from the previous full stamp, archive the prior
    under ``last_full_generator_git_sha`` for lag forensics. Never clear an
    existing last_full trail on full_generate.
    """
    out = dict(payload)
    sha = _generator_git_sha_short()
    if sha:
        prior = out.get("generator_git_sha")
        prior_s = str(prior).strip() if prior not in (None, "") else ""
        if prior_s and prior_s != sha:
            out["last_full_generator_git_sha"] = prior_s
        # Never drop an existing last_full trail when re-stamping same tip
        # or when prior was already cleared by a partial_patch path.
        out["generator_git_sha"] = sha
        out["generator_git_sha_status"] = status
        # Batch CB: full_generate with empty last_full gets self-trail so lag
        # forensics always has a non-null full stamp after a complete generate.
        if status == "full_generate":
            existing_last = out.get("last_full_generator_git_sha")
            if existing_last in (None, ""):
                out["last_full_generator_git_sha"] = sha
    return out


def _canonical_file_content_hash(path: Path) -> str | None:
    """SHA-256 of file bytes after stripping a single trailing newline.

    Dual-write trees often differ only by final-newline policy; content hash
    treats those as identical so sticky lag is not reported when payloads match.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    # Normalize trailing newlines only (do not alter interior whitespace)
    while raw.endswith(b"\n"):
        raw = raw[:-1]
    import hashlib

    return hashlib.sha256(raw).hexdigest()


def _attach_dual_write_provenance(
    payload: Dict[str, Any],
    *,
    private_path: str | Path | None = None,
    public_path: str | Path | None = None,
    dual_write_attempted: bool = False,
    dual_write_ok: bool | None = None,
    paths_identical: bool | None = None,
    note: str | None = None,
    lag_threshold_seconds: float = 120.0,
) -> Dict[str, Any]:
    """Attach dual-write completeness block for operator lag / split-brain forensics.

    Does not alter live authority. Complements generator_git_sha stamps (Batch AR).

    When both private and public paths exist and differ, sets
    ``dual_write_lag_seconds`` = public_mtime - private_mtime (negative means
    public is older than private — typical split-brain lag). Advisory only.

    Batch dual-write: if path resolves differ but **canonical content hashes**
    match (trailing-newline-normalized), clear sticky lag_stale and set lag
    seconds to 0. ``paths_identical`` remains path-resolve identity (caller
    flag or resolve equality); content match is ``content_hash_identical``.
    """
    out = dict(payload)
    sha = out.get("generator_git_sha")
    priv = Path(private_path) if private_path is not None else None
    pub = Path(public_path) if public_path is not None else None

    lag_seconds: float | None = None
    private_mtime: float | None = None
    public_mtime: float | None = None
    lag_stale = False
    content_hash_identical: bool | None = None
    private_content_hash: str | None = None
    public_content_hash: str | None = None
    # Path identity: caller flag, else compare resolves when both exist
    if paths_identical is None and priv is not None and pub is not None:
        try:
            paths_identical = priv.resolve() == pub.resolve()
        except OSError:
            paths_identical = False
    path_identical = bool(paths_identical)
    try:
        if priv is not None and priv.is_file():
            private_mtime = float(priv.stat().st_mtime)
            private_content_hash = _canonical_file_content_hash(priv)
        if pub is not None and pub.is_file():
            public_mtime = float(pub.stat().st_mtime)
            public_content_hash = _canonical_file_content_hash(pub)
        if private_content_hash is not None and public_content_hash is not None:
            content_hash_identical = private_content_hash == public_content_hash
        # Lag is irrelevant when paths are the same OR content hashes match
        lag_cleared = path_identical or bool(content_hash_identical)
        if (
            private_mtime is not None
            and public_mtime is not None
            and not lag_cleared
        ):
            # public - private: negative => public behind private (lag)
            lag_seconds = round(public_mtime - private_mtime, 3)
            # Stale if public is older than private by more than threshold
            if lag_seconds < -abs(float(lag_threshold_seconds)):
                lag_stale = True
        elif lag_cleared and private_mtime is not None and public_mtime is not None:
            lag_seconds = 0.0
            lag_stale = False
    except OSError:
        pass

    block: Dict[str, Any] = {
        "generator_git_sha_present": bool(sha),
        "dual_write_attempted": bool(dual_write_attempted),
        "dual_write_ok": dual_write_ok,
        "private_path": str(priv) if priv is not None else None,
        "public_path": str(pub) if pub is not None else None,
        "paths_identical": path_identical,
        "content_hash_identical": content_hash_identical,
        "private_content_hash": private_content_hash,
        "public_content_hash": public_content_hash,
        "dual_write_lag_seconds": lag_seconds,
        "dual_write_lag_unit": "seconds_public_mtime_minus_private",
        "dual_write_lag_stale": lag_stale,
        "dual_write_lag_threshold_seconds": float(lag_threshold_seconds),
        "private_mtime": private_mtime,
        "public_mtime": public_mtime,
        "disclosure": (
            "Dual-write provenance is advisory for split-brain detection; "
            "private DATA_DIR remains the producer SSOT when paths differ. "
            "Lag uses filesystem mtimes (public - private); negative means "
            "public is older than private. Content-hash equality "
            "(trailing-newline-normalized) clears sticky lag when payloads match."
        ),
    }
    if note:
        block["note"] = note
    out["provenance_completeness"] = block
    return out


def finalize_dual_write_provenance_after_sync(
    payload: Dict[str, Any],
    *,
    private_path: str | Path,
    public_path: str | Path,
    dual_write_ok: bool = True,
    note: str | None = None,
    lag_threshold_seconds: float = 120.0,
    write_json: bool = True,
) -> Dict[str, Any]:
    """Recompute dual-write lag/hash **after** both trees exist on disk (Batch CJ).

    Producers often stamp provenance *before* the public write, freezing the
    previous public mtime into ``dual_write_lag_stale=true`` forever even when
    the subsequent dual-write succeeds and content hashes match. Call this
    after both files are written (or after public replace) so lag/hash reflect
    post-sync reality. Optionally rewrites private + public with the honest
    block so operator canaries clear.

    Deep-research: content-hash / sync_verified events beat sticky pre-write
    lag gauges.
    """
    priv = Path(private_path)
    pub = Path(public_path)
    paths_identical = False
    try:
        paths_identical = priv.resolve() == pub.resolve()
    except OSError:
        paths_identical = False

    stamped = _attach_dual_write_provenance(
        payload,
        private_path=priv,
        public_path=pub,
        dual_write_attempted=not paths_identical,
        dual_write_ok=dual_write_ok if not paths_identical else True,
        paths_identical=paths_identical,
        note=note
        or (
            "post_sync dual-write provenance (Batch CJ): lag/hash after both "
            "trees exist"
        ),
        lag_threshold_seconds=lag_threshold_seconds,
    )
    if not write_json:
        return stamped

    body = json.dumps(stamped, indent=2, default=str) + "\n"
    try:
        priv.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = priv.with_suffix(priv.suffix + ".postsync.tmp")
        tmp_p.write_text(body, encoding="utf-8")
        tmp_p.replace(priv)
        if not paths_identical:
            pub.parent.mkdir(parents=True, exist_ok=True)
            tmp_u = pub.with_suffix(pub.suffix + ".postsync.tmp")
            tmp_u.write_text(body, encoding="utf-8")
            tmp_u.replace(pub)
        # Second pass: mtimes now both post-sync; refresh lag/hash once more
        stamped = _attach_dual_write_provenance(
            stamped,
            private_path=priv,
            public_path=pub,
            dual_write_attempted=not paths_identical,
            dual_write_ok=True if dual_write_ok or paths_identical else dual_write_ok,
            paths_identical=paths_identical,
            note=note
            or (
                "post_sync dual-write provenance (Batch CJ): lag/hash after both "
                "trees exist"
            ),
            lag_threshold_seconds=lag_threshold_seconds,
        )
        body2 = json.dumps(stamped, indent=2, default=str) + "\n"
        tmp_p = priv.with_suffix(priv.suffix + ".postsync.tmp")
        tmp_p.write_text(body2, encoding="utf-8")
        tmp_p.replace(priv)
        if not paths_identical:
            tmp_u = pub.with_suffix(pub.suffix + ".postsync.tmp")
            tmp_u.write_text(body2, encoding="utf-8")
            tmp_u.replace(pub)
    except OSError:
        # Best-effort; return last stamped payload even if rewrite fails
        pass
    return stamped


def _finalize_signal_metadata(output: Dict, *, finalized_at: str | None = None) -> Dict:
    """Stamp final artifact metadata after all signal sections are assembled."""
    timestamp = finalized_at or datetime.now(timezone.utc).isoformat()
    finalized = dict(output)
    finalized["generated_at"] = timestamp
    finalized["timestamp"] = timestamp
    # Batch BJ: full stamp with last_full retention + status (same contract as
    # health/overlay paths via _stamp_generator_git_sha).
    return _stamp_generator_git_sha(finalized, status="full_generate")


def _dist_data_dir_for_public_dir(public_dir: Path) -> Path:
    """Return the app dist/data directory that mirrors public/data."""
    if public_dir.name == "data" and public_dir.parent.name == "public":
        app_root = public_dir.parent.parent
    else:
        app_root = public_dir.parent
    return app_root / "dist" / "data"


def _mirror_public_data_contract_files_to_dist(public_dir: Path) -> None:
    """Mirror deploy-checked public data files after final generation."""
    dist_data = _dist_data_dir_for_public_dir(public_dir)
    dist_data.mkdir(parents=True, exist_ok=True)
    for filename in PUBLIC_DATA_DIST_MIRROR_FILES:
        source = public_dir / filename
        if source.exists():
            shutil.copyfile(source, dist_data / filename)


def _compact_health_summary(report: Dict) -> Dict:
    """Build a bounded health summary suitable for embedding in signals.json."""
    if not isinstance(report, dict):
        return {"status": "error", "error": "invalid health report"}

    status = report.get("system_status") or report.get("status") or "unknown"
    summary = {"status": status}

    if report.get("generated_at"):
        summary["generated_at"] = report.get("generated_at")
    if report.get("error"):
        summary["error"] = str(report.get("error"))

    cron_jobs = report.get("cron_jobs")
    if isinstance(cron_jobs, list):
        from src.monitor.hermes_cron import rollup_failed_cron_jobs

        summary["cron_job_count"] = len(cron_jobs)
        summary["failed_cron_jobs"] = len(rollup_failed_cron_jobs(cron_jobs))
        # Batch EE: dual-signal pending vs artifact-fresh reconcile on compact
        summary = project_pending_artifact_cron_onto_health(summary, cron_jobs)

    data_freshness = report.get("data_freshness")
    if isinstance(data_freshness, dict):
        summary["stale_data_count"] = sum(
            1
            for item in data_freshness.values()
            if isinstance(item, dict) and item.get("status") != "fresh"
        )

    scheduler_status = report.get("scheduler_status")
    if isinstance(scheduler_status, dict) and scheduler_status.get("status"):
        summary["scheduler_status"] = scheduler_status.get("status")

    data_pipeline_slo = report.get("data_pipeline_slo")
    if isinstance(data_pipeline_slo, dict):
        if data_pipeline_slo.get("status"):
            summary["data_pipeline_slo_status"] = data_pipeline_slo.get("status")
        if data_pipeline_slo.get("top_dimension"):
            summary["top_slo_dimension"] = data_pipeline_slo.get("top_dimension")
        runbook = data_pipeline_slo.get("runbook")
        if isinstance(runbook, dict):
            top_cause = runbook.get("top_cause")
            if isinstance(top_cause, dict) and top_cause.get("code"):
                summary["top_slo_cause_code"] = top_cause.get("code")

    # Project kill/incident halt into compact summary so signals.health
    # matches signals.broker.kill_switch without requiring broker-only reads.
    summary.update(project_compact_kill_fields(report))

    # Batch BN: surface signal_health rollup so compact view discloses 0/N healthy
    signal_health = report.get("signal_health")
    if isinstance(signal_health, dict):
        sh_summary = signal_health.get("summary")
        if isinstance(sh_summary, dict):
            summary["signal_health_healthy"] = sh_summary.get("healthy")
            summary["signal_health_degraded"] = sh_summary.get("degraded")
            summary["signal_health_unhealthy"] = sh_summary.get("unhealthy")
            summary["signal_health_total_tracked"] = sh_summary.get(
                "total_tracked"
            ) or sh_summary.get("total")
            # Batch CM: quality badge + freeze flag for compact consumers
            if sh_summary.get("quality_badge"):
                summary["signal_health_quality_badge"] = sh_summary.get("quality_badge")
            if sh_summary.get("zero_healthy_sources"):
                summary["signal_health_zero_healthy"] = True
            else:
                # Batch CR: clear sticky zero-healthy when SH recovered
                summary["signal_health_zero_healthy"] = False
            # Batch CQ/CR: always project freeze True/False (never leave sticky True)
            freeze_active = bool(sh_summary.get("ensemble_weight_freeze_active"))
            summary["ensemble_weight_freeze_active"] = freeze_active
            if sh_summary.get("ensemble_weights_age_days") is not None:
                summary["ensemble_weights_age_days"] = sh_summary.get(
                    "ensemble_weights_age_days"
                )
            if sh_summary.get("ensemble_weights_file_stale"):
                summary["ensemble_weights_file_stale"] = True
            elif not freeze_active:
                summary["ensemble_weights_file_stale"] = bool(
                    sh_summary.get("ensemble_weights_file_stale")
                )
        overall = signal_health.get("overall_health") or signal_health.get("status")
        if overall:
            summary["signal_health_status"] = overall
        qd = signal_health.get("quality_disclosure")
        if isinstance(qd, dict) and qd.get("badge"):
            summary["signal_quality_badge"] = qd.get("badge")
            # Prefer quality_disclosure freeze fields when summary omitted them
            freeze_block = qd.get("ensemble_weight_freeze")
            if isinstance(freeze_block, dict):
                if "weight_freeze_active" in freeze_block:
                    summary["ensemble_weight_freeze_active"] = bool(
                        freeze_block.get("weight_freeze_active")
                    )
                if freeze_block.get("ensemble_weights_age_days") is not None:
                    summary["ensemble_weights_age_days"] = freeze_block.get(
                        "ensemble_weights_age_days"
                    )
                if freeze_block.get("weight_file_stale") is not None:
                    summary["ensemble_weights_file_stale"] = bool(
                        freeze_block.get("weight_file_stale")
                    )

    return summary


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
    if (
        unique_days is not None
        and raw_entries is not None
        and unique_days > 0
        and raw_entries >= int(rewrite_inflate_min_raw)
        and raw_entries >= unique_days * float(rewrite_inflate_ratio)
    ):
        inflated = True
    elif rewrite_files >= int(rewrite_inflate_min_raw) and (
        unique_days is not None and rewrite_files > unique_days
    ):
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
        is_kill_execution_blocked = lambda p: bool(isinstance(p, dict) and p.get("enabled"))

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
    health_path = PUBLIC_DIR / "health.json"
    try:
        with health_path.open(encoding="utf-8") as handle:
            report = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return report if isinstance(report, dict) else None


def _log_signal_error(signal_name: str, exc: Exception) -> None:
    """Log a signal exception at the appropriate level.

    ValueError/TypeError are likely code bugs → logger.error.
    ImportError/AttributeError/KeyError are missing deps → logger.warning.
    RuntimeError/OSError are environmental → logger.warning.
    """
    if isinstance(exc, _BUG_EXCEPTIONS):
        logger.error("Signal %s failed with likely bug: %s", signal_name, exc)
    else:
        logger.warning("Signal %s not available: %s", signal_name, exc)


PUBLIC_DIR = PUBLIC_DATA_DIR
DB_PATH = MARKET_DB
_DEFAULT_DATA_DIR = DATA_DIR


def _resolve_hermes_cron_jobs_path() -> Optional[Path]:
    """Return the Hermes cron jobs file to probe, if this run should probe it."""
    return resolve_hermes_cron_jobs_path(
        current_data_dir=DATA_DIR,
        default_data_dir=_DEFAULT_DATA_DIR,
    )


class DashboardGenerator:
    # SPC monitor instance (class-level to persist across runs)
    _spc_monitor = None

    def __init__(self):
        PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite_connect(DB_PATH)
        self.conn.row_factory = sqlite3.Row

    @staticmethod
    def _deduplicate_performance_entries_by_date(
        entries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Keep the last performance entry for each calendar date."""
        daily_map: Dict[str, Dict[str, Any]] = {}
        for idx, entry in enumerate(entries):
            ts = entry.get("timestamp", "")
            date_key = ts[:10] if len(ts) >= 10 else ""
            if not date_key:
                # Preserve legacy behavior: timestamp-less rows are distinct.
                date_key = f"__no_ts_{idx}__"
            daily_map[date_key] = entry
        return [daily_map[d] for d in sorted(daily_map)]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        """Close the SQLite connection if open."""
        if self.conn is not None:
            try:
                self.conn.close()
            except (OSError, sqlite3.Error) as e:
                logger.warning("Error closing SQLite connection: %s", e)
            self.conn = None
    
    def generate_performance_json(self) -> Path:
        """Generate performance history for dashboard charts."""
        cursor = self.conn.cursor()
        
        # Get portfolio history
        cursor.execute("""
            SELECT symbol, date, close FROM prices 
            WHERE symbol IN ('SPY', 'GLD', 'TLT', 'QQQ')
            AND date >= date('now', '-365 days')
            ORDER BY date
        """)
        
        prices = {}
        for row in cursor.fetchall():
            sym = row[0]
            if sym not in prices:
                prices[sym] = []
            prices[sym].append({"d": row[1], "p": row[2]})
        
        # Get regime history
        cursor.execute("""
            SELECT date, regime, vix_level FROM regime_log
            WHERE date >= date('now', '-90 days')
            ORDER BY detected_at
        """)
        
        regimes = [{"d": row[0], "r": row[1], "v": row[2]} for row in cursor.fetchall()]
        
        # Get paper portfolio performance (from JSONL log — tail read only)
        perf_log = DATA_DIR / "performance.jsonl"
        paper_perf = []
        if perf_log.exists():
            raw_entries = []
            with open(perf_log) as f:
                for line in deque(f, maxlen=500):
                    try:
                        raw_entries.append(json.loads(line))
                    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                        logger.warning("Failed to parse performance log entry: %s", e)
            paper_perf = [
                {
                    "t": entry.get("timestamp", "")[:10],
                    "v": entry.get("total_value", 0),
                    "r": entry.get("daily_return", 0)
                }
                for entry in self._deduplicate_performance_entries_by_date(raw_entries)
            ]

        output = _stamp_generator_git_sha({
            "prices": prices,
            "regimes": regimes,
            "paper_portfolio": paper_perf,
            "generated_at": datetime.now(timezone.utc).isoformat()
        })

        out_path = PUBLIC_DIR / "dashboard.json"
        save_results_json(output, output_path=str(out_path))

        return out_path

    @staticmethod
    def _unavailable_zero_dte_payload() -> Dict[str, Any]:
        """Schema-compatible zero_dte panel when no producer is wired.

        LiveDashboard expects positions/config fields; never publish silent {}.
        Not live order-routing authority.
        """
        now_ts = datetime.now().isoformat()
        return {
            "positions": [],
            "config": None,
            "weekly_trades_used": 0,
            "total_premium_collected_mtd": 0.0,
            "status": "unavailable",
            "runtime_status": "unavailable_no_producer",
            "active": False,
            "live_authoritative": False,
            "reason": "zero_dte producer not wired into overlay merge",
            "generated_at": now_ts,
            "timestamp": now_ts,
        }

    @staticmethod
    def _unavailable_closing_auction_payload() -> Dict[str, Any]:
        """Schema-compatible closing_auction panel when no producer is wired."""
        now_ts = datetime.now().isoformat()
        return {
            "signals": [],
            "last_update": None,
            "market_open": False,
            "status": "unavailable",
            "runtime_status": "unavailable_no_producer",
            "active": False,
            "live_authoritative": False,
            "reason": "closing_auction producer not wired into overlay merge",
            "generated_at": now_ts,
            "timestamp": now_ts,
        }

    @staticmethod
    def _is_populated_overlay_section(value: Any) -> bool:
        """True when overlay merge produced a non-empty, non-placeholder section."""
        if not isinstance(value, dict) or not value:
            return False
        status = str(value.get("status") or value.get("runtime_status") or "").lower()
        if status in {"unavailable", "unavailable_no_producer"}:
            return False
        # Real producers set active/positions/signals or domain fields
        if value.get("positions") or value.get("signals"):
            return True
        if value.get("active") is True:
            return True
        # Any domain payload beyond honesty metadata counts as populated
        meta_keys = {
            "status",
            "runtime_status",
            "active",
            "live_authoritative",
            "reason",
            "generated_at",
            "timestamp",
            "last_update",
            "market_open",
            "config",
            "weekly_trades_used",
            "total_premium_collected_mtd",
            "positions",
            "signals",
        }
        return any(k not in meta_keys for k in value.keys())

    def _get_overlay_data(self) -> Dict:
        """Merge overlay dashboard data (collar, crypto, calendar, kurtosis, etc.)

        Pulls data from the OverlayDashboardGenerator and maps keys to the
        format expected by LiveDashboard.tsx panels.
        """
        result: Dict = {}

        # Primary overlay data
        try:
            from src.dashboard.overlay_dashboard import OverlayDashboardGenerator
            gen = OverlayDashboardGenerator()
            dashboard = gen.generate()
            data = dashboard.to_dict()

            result.update({
                "collar": data.get("collar", {}),
                "crypto": data.get("crypto", {}),
                "calendar": data.get("calendar", {}),
                "kurtosis": data.get("kurtosis", {}),
                "bond_momentum": data.get("bond_duration", {}),
                "unified": data.get("unified", {}),
            })
            # Pass through producers if overlay ever ships them
            if isinstance(data.get("zero_dte"), dict):
                result["zero_dte"] = data["zero_dte"]
            if isinstance(data.get("closing_auction"), dict):
                result["closing_auction"] = data["closing_auction"]
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("overlay_dashboard", e)

        # VIX term structure
        try:
            from src.signals.vix_term_structure import VIXTermStructureSignalGenerator
            vix_gen = VIXTermStructureSignalGenerator()
            signal = vix_gen.generate_signal()
            result["vix_term_structure"] = signal.to_dict()
            # VIX overlay state
            state_file = DATA_DIR / "vix_overlay_state.json"
            if state_file.exists():
                with open(state_file) as f:
                    result["vix_overlay"] = json.load(f)
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("vix_term_structure", e)

        # Dead schema surfaces: never publish silent {} without honesty fields
        if not self._is_populated_overlay_section(result.get("zero_dte")):
            result["zero_dte"] = self._unavailable_zero_dte_payload()
        if not self._is_populated_overlay_section(result.get("closing_auction")):
            result["closing_auction"] = self._unavailable_closing_auction_payload()

        return result

    def _record_ic_data(self, output: Dict) -> None:
        """Record signal predictions for IC decay monitoring.

        Two-phase lifecycle:
        1. Resolve: pair previously staged predictions with the forward
           return that materialized since they were staged.
        2. Stage: store current predictions for resolution next run.

        Uses SPY return as the forward return for all signals (SPY is the
        primary portfolio driver). Saves monitor state to disk so IC data
        survives across cron runs.
        """
        try:
            from src.monitor.ic_decay_monitor import ICMonitor

            monitor = ICMonitor()
            monitor.load_state()
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT date, close FROM prices WHERE symbol = 'SPY' "
                "ORDER BY date DESC LIMIT 1",
            )
            latest_spy_row = cursor.fetchone()
            latest_spy_date = latest_spy_row[0] if latest_spy_row else None

            # Phase 1: Resolve previously staged predictions
            if monitor.has_staged_predictions():
                staged_date = monitor.get_staged_date()
                if staged_date:
                    # Compute SPY forward return from staged date to latest
                    cursor.execute(
                        "SELECT date, close FROM prices WHERE symbol = 'SPY' "
                        "AND date >= ? ORDER BY date ASC LIMIT 1",
                        (staged_date,),
                    )
                    start_row = cursor.fetchone()
                    end_row = latest_spy_row
                    if start_row and end_row and start_row[0] != end_row[0]:
                        start_price = float(start_row[1])
                        end_price = float(end_row[1])
                        if start_price > 0:
                            forward_return = (end_price / start_price) - 1.0
                            n_resolved = monitor.resolve_staged(forward_return)
                            logger.info(
                                "IC decay: resolved %d staged predictions "
                                "(%s → %s, forward return=%.4f%%)",
                                n_resolved, staged_date, end_row[0],
                                forward_return * 100,
                            )

            # Phase 2: Stage current predictions for next run
            predictions: Dict[str, float] = {}

            # Ensemble voter biases
            ensemble = output.get("ensemble_voting")
            if isinstance(ensemble, dict):
                if "equity_bias" in ensemble and ensemble["equity_bias"] is not None:
                    predictions["ensemble_equity"] = float(ensemble["equity_bias"])
                if "gold_bias" in ensemble and ensemble["gold_bias"] is not None:
                    predictions["ensemble_gold"] = float(ensemble["gold_bias"])
                if "duration_bias" in ensemble and ensemble["duration_bias"] is not None:
                    predictions["ensemble_duration"] = float(ensemble["duration_bias"])
                if "weighted_consensus" in ensemble and ensemble["weighted_consensus"] is not None:
                    predictions["ensemble_consensus"] = float(ensemble["weighted_consensus"])

            # Alternative data composite score
            alt = output.get("alternative_data")
            if isinstance(alt, dict) and alt.get("composite_score") is not None:
                predictions["alternative_data"] = float(alt["composite_score"])

            # Behavioral sentiment composite
            beh = output.get("behavioral_sentiment")
            if isinstance(beh, dict) and beh.get("composite_score") is not None:
                predictions["behavioral_sentiment"] = float(beh["composite_score"])

            # Factor rotation signal strength
            fr = output.get("factor_rotation")
            if isinstance(fr, dict) and fr.get("signal_strength") is not None:
                predictions["factor_rotation"] = float(fr["signal_strength"])

            # FRED-MD macro confidence
            fred = output.get("fred_macro")
            if _is_predictive_fred_macro(fred):
                predictions["fred_macro"] = float(fred["confidence"])

            if predictions and latest_spy_date and not monitor.has_staged_predictions():
                monitor.stage_predictions(
                    predictions,
                    str(latest_spy_date),
                )

            monitor.save_state()
        except MONITOR_EXCEPTIONS as e:
            _log_signal_error("ic_decay_record", e)

    def _generate_two_stage_regime(self) -> Optional[Dict]:
        """Generate two-stage k-means macro regime signal.

        Uses Oliveira et al. 2025 two-layer k-means (L2 crisis detection +
        cosine directional clustering) on FRED-MD data. Falls back gracefully
        if FRED-MD data or fredapi is not available.

        Returns:
            Dict with regime, confidence, probabilities, or None if unavailable.
        """
        try:
            from src.regime.fred_md_two_stage_kmeans import TwoStageKMeansRegime
            from src.data.fred_data import FredMdFetcher
        except ImportError:
            return None

        # Try to load FRED-MD data via the existing pipeline
        try:
            fetcher = FredMdFetcher()
            df = fetcher.get_all_series(cache_ok=True)
        except (ValueError, ImportError) as e:
            logger.info("FRED-MD data not available for two-stage k-means: %s", e)
            return None

        if df is None or df.empty or len(df.columns) < 10:
            logger.info("Insufficient FRED-MD series for two-stage k-means")
            return None

        # Build data matrix: rows=dates, cols=series
        # Forward-fill missing values, drop rows with too many NaNs
        df_filled = df.ffill().dropna(thresh=max(10, len(df.columns) // 2))
        if len(df_filled) < 24:
            logger.info("Too few FRED-MD observations for two-stage k-means")
            return None

        # Standardize: z-score each series
        X_raw = df_filled.values.astype(np.float64)
        X_std = (X_raw - np.nanmean(X_raw, axis=0)) / np.nanstd(X_raw, axis=0)
        X = np.nan_to_num(X_std, nan=0.0)

        # Fit two-stage k-means
        classifier = TwoStageKMeansRegime(random_state=42, max_pca_components=15)
        classifier.fit(X)

        signal = classifier.get_signal(latest_index=-1)

        return {
            "regime": signal["regime"],
            "confidence": signal["confidence"],
            "crisis_probability": signal.get("crisis_probability", 0.0),
            "probabilities": signal.get("probabilities", {}),
            "n_pca_components": signal.get("n_pca_components", 0),
            "variance_retained": signal.get("variance_retained", 0.0),
            "n_observations": len(df_filled),
            "n_series": len(df_filled.columns),
            "method": "oliveira_2025_two_stage_kmeans",
            "timestamp": datetime.now().isoformat(),
        }

    def _generate_bocd_regime(self) -> Optional[Dict]:
        """Generate BOCD (Bayesian Online Changepoint Detection) regime signal.

        Uses Adams & MacKay (2007) for real-time structural break detection
        in daily return series without fixed observation windows.

        Returns:
            Dict with regime, regime_change_prob, changepoint_count, etc.
            None if insufficient data.
        """
        try:
            from src.regime.bocd_detector import BOCDDetector
        except ImportError:
            return None

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT date, close FROM prices
            WHERE symbol = 'SPY'
            ORDER BY date ASC
        """)
        rows = cursor.fetchall()
        if len(rows) < 2:
            return None

        prices = np.array([row[1] for row in rows], dtype=float)
        returns = np.diff(np.log(prices))

        # threshold=0.5 keeps legacy dashboard wiring; max_lookback caps O(n)
        # collapse on multi-year SPY history (see BOCDDetector.DEFAULT_MAX_LOOKBACK).
        detector = BOCDDetector(
            hazard_rate=1.0 / 252,
            threshold=0.5,
            min_run_length=5,
            max_lookback=BOCDDetector.DEFAULT_MAX_LOOKBACK,
        )
        detector.fit(returns)

        signal = detector.get_signal()
        bocd_data = signal["bocd_detector"]
        bocd_data["timestamp"] = datetime.now().isoformat()

        return bocd_data

    @staticmethod
    def _coerce_vix_level(value: Any) -> Optional[float]:
        """Parse a positive finite VIX level, else None."""
        if value is None:
            return None
        try:
            level = float(value)
        except (TypeError, ValueError):
            return None
        if level != level or level <= 0:  # NaN or non-positive
            return None
        return level

    @classmethod
    def _enrich_regime_vix(
        cls,
        regime_data: Dict[str, Any],
        *,
        vix_term_structure: Any = None,
        behavioral_sentiment: Any = None,
    ) -> Dict[str, Any]:
        """Fill regime.vix from best available surface; disclose vix_source.

        Preference: existing market.db level on regime_data → vix_term_structure
        → behavioral_sentiment. Does not change live target_allocations authority.
        """
        out = dict(regime_data or {})
        existing = cls._coerce_vix_level(out.get("vix"))
        if existing is not None:
            out["vix"] = existing
            out.setdefault("vix_source", "market.db")
            return out

        # vix_term_structure payload (dict from VIXTermStructureSignal.to_dict)
        if isinstance(vix_term_structure, dict):
            for key in ("vix_spot", "vix", "spot"):
                level = cls._coerce_vix_level(vix_term_structure.get(key))
                if level is not None:
                    out["vix"] = level
                    out["vix_source"] = "vix_term_structure"
                    return out

        # behavioral_sentiment may nest vix under top-level or snapshot
        if isinstance(behavioral_sentiment, dict):
            candidates = [
                behavioral_sentiment.get("vix"),
                behavioral_sentiment.get("vix_level"),
            ]
            opts = behavioral_sentiment.get("options")
            if isinstance(opts, dict):
                candidates.append(opts.get("vix"))
            for cand in candidates:
                level = cls._coerce_vix_level(cand)
                if level is not None:
                    out["vix"] = level
                    out["vix_source"] = "behavioral_sentiment"
                    return out

        out["vix"] = None
        out["vix_source"] = "unavailable"
        return out

    def _load_signal_generation_context(self) -> Dict[str, Any]:
        """Load DB, portfolio, and regime context for signals.json."""
        cursor = self.conn.cursor()

        # Get latest VIX level directly from prices table
        cursor.execute("""
            SELECT close FROM prices 
            WHERE symbol = '^VIX' 
            ORDER BY date DESC LIMIT 1
        """)
        vix_row = cursor.fetchone()
        vix_level = self._coerce_vix_level(vix_row[0] if vix_row else None)
        
        # Try to get trend signal from regime_log
        cursor.execute("""
            SELECT regime, detected_at FROM regime_log
            ORDER BY detected_at DESC LIMIT 1
        """)
        trend_row = cursor.fetchone()
        trend_regime = trend_row[0] if trend_row else "normal"
        trend_detected = trend_row[1] if trend_row else None

        # VIX-based regime detection (shared logic with evaluator.py)
        current_regime = classify_vix_regime(vix_level, trend_regime)

        regime_data = {
            "regime": current_regime,
            "vix": vix_level,
            "vix_source": "market.db" if vix_level is not None else "unavailable",
            "detected": trend_detected
        }
        
        # Latest prices
        cursor.execute("""
            SELECT symbol, close FROM prices 
            WHERE (symbol, date) IN (
                SELECT symbol, MAX(date) FROM prices GROUP BY symbol
            )
        """)
        latest = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Current paper portfolio state
        portfolio_state = DATA_DIR / "portfolio_paper.json"
        positions = []
        if portfolio_state.exists():
            with open(portfolio_state) as f:
                state = json.load(f)
                for sym, pos in state.get("positions", {}).items():
                    positions.append({
                        "symbol": sym,
                        "shares": pos.get("shares", 0),
                        "value": pos.get("value", 0),
                        "weight": pos.get("weight", 0),
                        "unrealized": pos.get("unrealized_pnl", 0)
                    })
                total_value = state.get("cash", 0) + sum(p["value"] for p in positions)
                cash = state.get("cash", 0)
        else:
            total_value = 100000  # Initial
            cash = 100000
        
        # Target allocation based on regime
        if os.environ.get("REGIME_ALLOC_ENABLED", "0") == "1":
            target_alloc = get_regime_allocation_with_override(current_regime)
        else:
            target_alloc = REGIME_OVERRIDES.get(current_regime) or BASE_ALLOCATION
        
        # Pending orders (tail read only)
        orders = []
        orders_log = DATA_DIR / "orders.jsonl"
        if orders_log.exists():
            with open(orders_log) as f:
                for line in deque(f, maxlen=5):
                    try:
                        order = json.loads(line)
                        orders.append({
                            "sym": order.get("symbol"),
                            "side": order.get("side"),
                            "shares": round(order.get("shares", 0), 2),
                            "value": round(order.get("fill_value", 0), 2)
                        })
                    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                        logger.warning("Failed to parse order log entry: %s", e)

        return {
            "cursor": cursor,
            "vix_level": vix_level,
            "trend_regime": trend_regime,
            "trend_detected": trend_detected,
            "current_regime": current_regime,
            "regime_data": regime_data,
            "latest": latest,
            "positions": positions,
            "cash": cash,
            "total_value": total_value,
            "target_alloc": target_alloc,
            "orders": orders,
        }

    @staticmethod
    def _build_ensemble_source_breakdown(source_votes: List[Any]) -> List[Dict[str, Any]]:
        """Serialize ensemble source readings for downstream postprocessors."""
        source_breakdown = []
        for src in source_votes:
            value = float(src.value)
            is_active = bool(getattr(src, "is_active", True))
            entry = {
                "source": src.source.value if hasattr(src.source, 'value') else str(src.source),
                "value": round(value, 4),
                "direction": "bullish" if value > 0 else ("bearish" if value < 0 else "neutral"),
                "strength": round(abs(value), 3),
                "confidence": round(src.confidence, 3),
                "weight": round(src.weight, 3),
                # Batch CY: surface snapshot activity for inactive_signal disclosure
                "is_active": is_active,
            }
            expl = getattr(src, "explanation", None) or ""
            if expl and not is_active:
                entry["inactive_explanation"] = str(expl)[:200]
            source_breakdown.append(entry)
        return source_breakdown

    @staticmethod
    def _build_ensemble_source_count_metadata(
        regime: Any,
        source_breakdown: List[Dict[str, Any]],
        configured_source_status: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Describe configured, collected, and positive-weight ensemble sources.

        When ``configured_source_status`` is provided, ``inactive_*`` rolls up
        rows with status in {missing, stale, inactive, zero_weight, unavailable}
        so headline counters match the detail table (not only zero-weight
        collected rows).
        """
        configured_sources = []
        try:
            from src.strategy.ensemble_voter import REGIME_WEIGHTS, Regime, SignalSource

            regime_key = regime if isinstance(regime, Regime) else Regime(str(regime).lower())
            configured_sources = [
                source.value if hasattr(source, "value") else str(source)
                for source in REGIME_WEIGHTS.get(regime_key, {})
            ] or [source.value for source in SignalSource]
        except (ImportError, AttributeError, KeyError, TypeError, ValueError):
            configured_sources = []

        inactive_sources = []
        contributing_count = 0
        for source in source_breakdown:
            try:
                weight = float(source.get("weight", 0.0))
            except (TypeError, ValueError):
                weight = 0.0
            source_name = str(source.get("source", "unknown"))
            if np.isfinite(weight) and weight > 0:
                contributing_count += 1
            else:
                inactive_sources.append(source_name)

        # Prefer configured-status rollup when present (includes missing/stale)
        inactive_statuses = {
            "missing",
            "stale",
            "inactive",
            "zero_weight",
            "zero_baseline",  # Batch CU: intentional weight-0 roster arms
            "health_sleep",  # Batch CW: CN unhealthy / degraded+neg-IC sleep
            "regime_gate",  # Batch CX: intentional OFF for current regime
            "inactive_signal",  # Batch CY: snapshot is_active=False
            "unavailable",
        }
        if configured_source_status:
            rolled: List[str] = []
            for row in configured_source_status:
                if not isinstance(row, dict):
                    continue
                status = str(row.get("status") or "").lower()
                contributing = bool(row.get("contributing"))
                name = str(row.get("source") or "")
                if not name:
                    continue
                if status in inactive_statuses or (
                    status != "active" and not contributing
                ):
                    rolled.append(name)
            inactive_sources = rolled

        collected_count = len(source_breakdown)
        configured_count = len(set(configured_sources)) if configured_sources else collected_count
        if configured_source_status:
            configured_count = max(configured_count, len(configured_source_status))
        return {
            "num_sources": collected_count,
            "configured_source_count": configured_count,
            "collected_source_count": collected_count,
            "contributing_source_count": contributing_count,
            "inactive_source_count": len(inactive_sources),
            "inactive_sources": inactive_sources,
        }

    @staticmethod
    def _get_configured_ensemble_source_weights(regime: Any) -> Dict[str, float]:
        """Return configured ensemble source weights for the active regime."""
        try:
            from src.strategy.ensemble_voter import REGIME_WEIGHTS, Regime

            regime_key = regime if isinstance(regime, Regime) else Regime(str(regime).lower())
            weights_file = Path(os.environ.get("ENSEMBLE_WEIGHTS_FILE", str(DATA_DIR / "ensemble_weights.json")))
            if weights_file.exists():
                try:
                    with open(weights_file) as f:
                        configured = json.load(f)
                    regime_weights = configured.get(regime_key.value)
                    if isinstance(regime_weights, dict):
                        return {
                            str(source): float(weight)
                            for source, weight in regime_weights.items()
                        }
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    pass
            return {
                source.value if hasattr(source, "value") else str(source): float(weight)
                for source, weight in REGIME_WEIGHTS.get(regime_key, {}).items()
            }
        except (ImportError, AttributeError, KeyError, TypeError, ValueError):
            return {}

    @staticmethod
    def _format_ensemble_source_label(source: str) -> str:
        """Format a source identifier for operator-facing source disclosure."""
        return source.replace("_", " ").title()

    @staticmethod
    def _google_trends_inactive_disclosure() -> tuple[str, str]:
        """Inspect Google Trends directly when it is configured but not collected."""
        try:
            from src.signals.google_trends_signal import GoogleTrendsSignal

            snapshot = GoogleTrendsSignal().get_signal_snapshot()
            if snapshot.is_active:
                return "missing", "Configured Google Trends did not appear in ensemble source rows."

            reason = snapshot.metadata.get("inactive_reason") if isinstance(snapshot.metadata, dict) else None
            if not reason:
                reason = snapshot.explanation.replace("Google Trends:", "", 1).strip()
            category = snapshot.metadata.get("inactive_category") if isinstance(snapshot.metadata, dict) else None
            status = str(category or "inactive")
            return status, str(reason or "Google Trends source is inactive.")
        except (ImportError, AttributeError, KeyError, ValueError, TypeError, OSError, RuntimeError) as e:
            return "unavailable", f"Google Trends status unavailable: {e}"

    # Batch DA: multi-horizon IC reentry hysteresis (disclosure; never force-wake)
    # Sleep stays fail-closed at IC < 0 (voter). Reentry needs IC > REENTRY_IC_EPS
    # on *all* horizons (30/60/90) so a single short-window bounce cannot re-arm.
    IC_REENTRY_EPS: float = 0.02
    IC_REENTRY_HORIZONS: tuple[int, ...] = (30, 60, 90)
    # Batch DE: short-horizon IC for half-life / recent collapse disclosure
    IC_SHORT_HORIZON_DAYS: int = 14

    @staticmethod
    def _evaluate_ic_reentry(
        *,
        ic_30d: float | None,
        ic_60d: float | None,
        ic_90d: float | None,
        reentry_eps: float | None = None,
    ) -> Dict[str, Any]:
        """Hysteresis reentry checklist from multi-horizon IC (Batch DA).

        Policy (sleeping-experts + control hysteresis):
        - Do not force-wake if any horizon IC is missing or < 0.
        - Eligible only when every horizon IC > reentry_eps (default +0.02),
          i.e. a positive gap above the sleep threshold (0) to prevent chatter.
        - Disclosure only: this does not change voter weights.
        """
        eps = (
            float(DashboardGenerator.IC_REENTRY_EPS)
            if reentry_eps is None
            else float(reentry_eps)
        )
        horizons = {
            "ic_30d": ic_30d,
            "ic_60d": ic_60d,
            "ic_90d": ic_90d,
        }
        missing = [k for k, v in horizons.items() if v is None]
        negative = [k for k, v in horizons.items() if v is not None and v < 0.0]
        below_eps = [
            k for k, v in horizons.items() if v is not None and v <= eps
        ]
        eligible = not missing and not negative and not below_eps
        if missing:
            blocked = f"insufficient_ic_horizons({','.join(missing)})"
        elif negative:
            blocked = f"negative_ic_horizon({','.join(negative)})"
        elif below_eps:
            blocked = f"below_reentry_eps({eps:g};{','.join(below_eps)})"
        else:
            blocked = None
        return {
            "reentry_eligible": bool(eligible),
            "reentry_eps": eps,
            "sleep_threshold": 0.0,
            "horizons": {
                k: (None if v is None else round(float(v), 4))
                for k, v in horizons.items()
            },
            "horizons_all_positive": bool(
                not missing and not negative and all(
                    v is not None and v > 0.0 for v in horizons.values()
                )
            ),
            "horizons_all_above_eps": bool(eligible),
            "reentry_blocked_reason": blocked,
            "policy": "multi_horizon_hysteresis_no_force_wake",
        }

    @staticmethod
    def _signal_health_metrics_map() -> Dict[str, Dict[str, Any]]:
        """Batch CZ/DA: SH metrics + multi-horizon IC reentry for sleep disclosure."""
        try:
            from src.signals.health_tracker import SignalHealthTracker

            tracker = SignalHealthTracker()
            scores = tracker.calculate_all_health_scores()
        except Exception:  # noqa: BLE001 — never block signals gen on SH metrics
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        if not isinstance(scores, dict):
            return out
        for name, health in scores.items():
            if health is None:
                continue
            try:
                ic_raw = getattr(health, "ic", None)
                try:
                    ic_val = float(ic_raw) if ic_raw is not None else None
                except (TypeError, ValueError):
                    ic_val = None
                acc30 = getattr(health, "accuracy_30d", None)
                acc60 = getattr(health, "accuracy_60d", None)
                try:
                    acc30_f = float(acc30) if acc30 is not None else None
                except (TypeError, ValueError):
                    acc30_f = None
                try:
                    acc60_f = float(acc60) if acc60 is not None else None
                except (TypeError, ValueError):
                    acc60_f = None
                hs = getattr(health, "health_score", None)
                try:
                    hs_f = float(hs) if hs is not None else None
                except (TypeError, ValueError):
                    hs_f = None
                hl = getattr(health, "ic_half_life_days", None)
                try:
                    hl_f = float(hl) if hl is not None else None
                except (TypeError, ValueError):
                    hl_f = None
                status = str(getattr(health, "status", "") or "")
                collapse = bool(getattr(health, "window_collapse_90_60", False))

                # Batch DA: multi-horizon IC (primary HealthScore.ic is ~90d)
                def _safe_ic(days: int) -> float | None:
                    try:
                        raw = tracker.compute_ic(str(name), lookback_days=days)
                        return float(raw) if raw is not None else None
                    except Exception:  # noqa: BLE001
                        return None

                ic_14 = _safe_ic(int(DashboardGenerator.IC_SHORT_HORIZON_DAYS))
                ic_30 = _safe_ic(30)
                ic_60 = _safe_ic(60)
                ic_90 = ic_val if ic_val is not None else _safe_ic(90)
                reentry = DashboardGenerator._evaluate_ic_reentry(
                    ic_30d=ic_30,
                    ic_60d=ic_60,
                    ic_90d=ic_90,
                )
                hint = DashboardGenerator._health_recovery_hint(
                    status=status,
                    ic=ic_val,
                    acc30=acc30_f,
                    acc60=acc60_f,
                    health_score=hs_f,
                    half_life=hl_f,
                    reentry=reentry,
                    ic_14d=ic_14,
                )
                row: Dict[str, Any] = {
                    "status": status,
                    "health_score": None if hs_f is None else round(hs_f, 4),
                    "ic": None if ic_val is None else round(ic_val, 4),
                    "ic_14d": None if ic_14 is None else round(ic_14, 4),
                    "ic_30d": None if ic_30 is None else round(ic_30, 4),
                    "ic_60d": None if ic_60 is None else round(ic_60, 4),
                    "ic_90d": None if ic_90 is None else round(ic_90, 4),
                    "accuracy_30d": None if acc30_f is None else round(acc30_f, 4),
                    "accuracy_60d": None if acc60_f is None else round(acc60_f, 4),
                    "ic_half_life_days": hl_f,
                    "window_collapse_90_60": collapse,
                    "reentry": reentry,
                    "reentry_eligible": reentry["reentry_eligible"],
                    "recovery_hint": hint,
                }
                # Batch DE: alt_data component long-bias / saturation diagnostic
                if str(name) == "alternative_data":
                    comp = DashboardGenerator._alt_data_component_bias_diagnostic()
                    if comp:
                        row["component_bias"] = comp
                        if comp.get("bias_issue") and not reentry.get("reentry_eligible"):
                            row["recovery_hint"] = (
                                f"{hint} | components: {comp['bias_issue']}"
                            )
                out[str(name)] = row
            except Exception:  # noqa: BLE001
                continue
        return out

    @staticmethod
    def _alt_data_component_bias_diagnostic() -> Dict[str, Any] | None:
        """Batch DE: live component saturation / long-bias for alternative_data."""
        try:
            from src.paths import DATA_DIR
            import json

            path = DATA_DIR / "signals" / "alternative_data_latest.json"
            if not path.exists():
                path = DATA_DIR / "alternative_data_state.json"
            if not path.exists():
                return None
            data = json.loads(path.read_text())
            raw = data.get("raw_data") if isinstance(data.get("raw_data"), dict) else data
            components = raw.get("components") if isinstance(raw, dict) else None
            if not isinstance(components, dict) or not components:
                return None
            vals = {}
            saturated = []
            n_pos = 0
            n = 0
            for k, v in components.items():
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                vals[str(k)] = round(fv, 4)
                n += 1
                if fv > 0:
                    n_pos += 1
                if abs(fv) >= 0.95:
                    saturated.append(str(k))
            pos_rate = (n_pos / n) if n else None
            try:
                composite = float(raw.get("composite_score"))
            except (TypeError, ValueError):
                composite = None
            issue = None
            if saturated and pos_rate is not None and pos_rate >= 0.6:
                issue = (
                    f"component_saturation({','.join(saturated)}) with "
                    f"{pos_rate:.0%} components positive — composite long-bias risk; "
                    "Batch DE soft-scales broad_momentum; keep slept until multi-horizon IC>eps."
                )
            elif pos_rate is not None and pos_rate >= 0.85:
                issue = (
                    f"component_long_bias ({pos_rate:.0%} positive) — "
                    "macro composite rarely bears; do not force-wake on IC30 alone."
                )
            return {
                "composite_score": composite,
                "components": vals,
                "component_positive_rate": None if pos_rate is None else round(pos_rate, 4),
                "saturated_components": saturated,
                "bias_issue": issue,
            }
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _health_recovery_hint(
        *,
        status: str,
        ic: float | None,
        acc30: float | None,
        acc60: float | None,
        health_score: float | None,
        half_life: float | None,
        reentry: Dict[str, Any] | None = None,
        ic_14d: float | None = None,
    ) -> str:
        """Operator-facing recovery guidance for slept/degraded arms (Batch CZ/DA/DE)."""
        # Batch DE: very-short IC collapse overrides optimistic mid-window bounce
        if (
            ic_14d is not None
            and ic_14d < -0.1
            and isinstance(reentry, dict)
            and not reentry.get("reentry_eligible")
        ):
            return (
                f"Recent IC14d={ic_14d:.3f} collapse — wait for multi-horizon recovery "
                "(14d then 30/60/90); do not force-wake on older positive windows."
            )
        # Batch DA: prefer multi-horizon reentry state when present
        if isinstance(reentry, dict):
            if reentry.get("reentry_eligible"):
                return (
                    "Multi-horizon IC reentry eligible (all horizons > "
                    f"{reentry.get('reentry_eps', DashboardGenerator.IC_REENTRY_EPS)}); "
                    "shadow-monitor then allow natural health gate wake — do not force."
                )
            blocked = reentry.get("reentry_blocked_reason") or ""
            horizons = reentry.get("horizons") or {}
            if blocked.startswith("negative_ic_horizon"):
                short_pos = (
                    horizons.get("ic_30d") is not None
                    and float(horizons["ic_30d"]) > 0
                    and any(
                        horizons.get(k) is not None and float(horizons[k]) < 0
                        for k in ("ic_60d", "ic_90d")
                    )
                )
                if short_pos:
                    return (
                        "Short-horizon IC bounce only — multi-horizon hysteresis "
                        "blocks reentry until 60d/90d IC also clear; do not force-wake."
                    )
                if ic is not None and ic < -0.15:
                    return (
                        "Deeply negative multi-horizon IC — investigate label/feature "
                        "alignment; keep slept until all horizons > reentry eps."
                    )
                return (
                    f"Reentry blocked ({blocked}) — sleep until all IC horizons "
                    f"> {reentry.get('reentry_eps', DashboardGenerator.IC_REENTRY_EPS)}; "
                    "shadow-monitor only."
                )
            if blocked.startswith("below_reentry_eps"):
                return (
                    "Horizons non-negative but below reentry hysteresis eps — "
                    "wait for confirmed multi-horizon IC > eps; do not force-wake."
                )
            if blocked.startswith("insufficient_ic"):
                return (
                    "Insufficient multi-horizon IC sample — keep slept; "
                    "do not force-wake without horizon evidence."
                )

        st = (status or "").lower()
        if ic is not None and ic < -0.15:
            return (
                "Deeply negative IC — investigate label/feature alignment; "
                "keep slept until rolling IC > 0 with multi-horizon confirmation."
            )
        if ic is not None and ic < 0:
            if acc30 is not None and acc60 is not None and acc30 + 0.05 < acc60:
                return (
                    "Negative IC with recent accuracy decay vs 60d — wait for "
                    "label resolve + IC reentry (IC>0); do not force-wake."
                )
            return (
                "Negative IC (toxic drag gate) — sleep until IC recovers > 0; "
                "shadow-monitor predictions while slept."
            )
        if st == "unhealthy":
            return (
                "Quality unhealthy — soft-floor if IC≥0 (Batch CY); improve "
                "accuracy/health_score before expecting full weight."
            )
        if half_life is not None and half_life < 20:
            return (
                f"Short IC half-life (~{half_life:.0f}d) — edge decays fast; "
                "prefer recent windows and re-check before promotion."
            )
        if health_score is not None and health_score < 0.55:
            return "Borderline health_score — monitor 30d accuracy before promoting weight."
        return "Monitor multi-horizon IC and accuracy; reenter only after confirmed recovery."

    # Batch DB: international RS activation thresholds (fractional outperformance)
    # Match InternationalMomentumGenerator.EFA_THRESHOLD / EEM_THRESHOLD.
    INTL_EFA_THRESHOLD_PP: float = 5.0
    INTL_EEM_THRESHOLD_PP: float = 8.0

    # Batch DD: intentional zero-baseline soft-delete rationale (not fetch failure).
    # Re-enable is human/ADR only — never auto-restore weight from health alone.
    ZERO_BASELINE_SOFT_DELETE: Dict[str, str] = {
        "multi_speed_momentum": (
            "net_negative_sharpe_backtest(-0.012); weight redistributed to "
            "ALT_DATA / INTL_MOM — soft-delete, not missing."
        ),
    }
    SHADOW_REENABLE_MIN_HEALTH: float = 0.55

    @staticmethod
    def _international_activation_disclosure(
        explanation: str | None = None,
        value: float | None = None,
        confidence: float | None = None,
    ) -> Dict[str, Any]:
        """Structured inactive gaps for international_momentum (Batch DB).

        Neutral band is intentional when EFA/SPY and EEM/SPY are inside
        activation thresholds (5pp / 8pp). Ops need gap-to-threshold, not
        only free-text explanation.
        """
        import re

        expl = str(explanation or "")
        efa_pp: float | None = None
        eem_pp: float | None = None
        m_efa = re.search(r"EFA/SPY\s*=\s*([+-]?\d+(?:\.\d+)?)\s*pp", expl, re.I)
        m_eem = re.search(r"EEM/SPY\s*=\s*([+-]?\d+(?:\.\d+)?)\s*pp", expl, re.I)
        if m_efa:
            try:
                efa_pp = float(m_efa.group(1))
            except ValueError:
                efa_pp = None
        if m_eem:
            try:
                eem_pp = float(m_eem.group(1))
            except ValueError:
                eem_pp = None

        efa_thr = float(DashboardGenerator.INTL_EFA_THRESHOLD_PP)
        eem_thr = float(DashboardGenerator.INTL_EEM_THRESHOLD_PP)
        gaps: list[str] = []
        conf_f: float | None
        try:
            conf_f = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            conf_f = None
        try:
            val_f = float(value) if value is not None else None
        except (TypeError, ValueError):
            val_f = None

        if "neutral" in expl.lower() or (val_f is not None and abs(val_f) < 1e-12):
            gaps.append("signal_type_neutral")
        if conf_f is not None and conf_f < 0.5:
            gaps.append("confidence_below_0.5")
        if efa_pp is not None and efa_pp <= efa_thr:
            gaps.append(
                f"efa_rs_below_threshold({efa_pp:+.2f}pp need >+{efa_thr:.0f}pp)"
            )
        if eem_pp is not None and eem_pp <= eem_thr:
            gaps.append(
                f"eem_rs_below_threshold({eem_pp:+.2f}pp need >+{eem_thr:.0f}pp)"
            )
        if "vix_filter=true" in expl.lower():
            gaps.append("vix_filter_active")
        if not gaps and "inactive" not in expl.lower():
            gaps.append("inactive_unspecified")

        efa_gap = None if efa_pp is None else round(efa_thr - efa_pp, 2)
        eem_gap = None if eem_pp is None else round(eem_thr - eem_pp, 2)
        policy = (
            "neutral_band_hold — RS inside activation thresholds; "
            "not a fetch failure (ensemble weight stays 0 until lead)."
        )
        return {
            "policy": policy,
            "efa_vs_spy_pp": efa_pp,
            "eem_vs_spy_pp": eem_pp,
            "efa_threshold_pp": efa_thr,
            "eem_threshold_pp": eem_thr,
            "efa_gap_to_threshold_pp": efa_gap,
            "eem_gap_to_threshold_pp": eem_gap,
            "activation_gaps": gaps,
            "activation_hint": (
                "International RS neutral band: wait for EFA>+5pp or EEM>+8pp "
                "vs SPY (6m relative) with conf≥0.5 and risk controls passed; "
                "do not lower thresholds without backtest (whipsaw risk)."
            ),
        }

    @staticmethod
    def _label_alignment_diagnostic(source: str) -> Dict[str, Any] | None:
        """Batch DB/DC: deadband honesty + polarity bias (no auto-invert)."""
        try:
            from src.signals.health_tracker import SignalHealthTracker
            from src.paths import MARKET_DB
            import sqlite3

            deadband = float(SignalHealthTracker.DIRECTION_DEADBAND)
            with sqlite3.connect(str(MARKET_DB)) as conn:
                row = conn.execute(
                    """
                    SELECT
                      COUNT(*) AS n,
                      SUM(CASE WHEN predicted_direction = 0 THEN 1 ELSE 0 END) AS pred0,
                      SUM(CASE WHEN ABS(signal_value) >= ? AND predicted_direction = 0
                               THEN 1 ELSE 0 END) AS mislabeled_neutral,
                      SUM(CASE WHEN ABS(signal_value) >= ? THEN 1 ELSE 0 END) AS abs_ge_db,
                      AVG(ABS(signal_value)) AS mean_abs,
                      SUM(CASE WHEN signal_value > 0 THEN 1 ELSE 0 END) AS n_pos,
                      SUM(CASE WHEN signal_value < 0 THEN 1 ELSE 0 END) AS n_neg
                    FROM signal_predictions
                    WHERE source = ?
                      AND signal_value IS NOT NULL
                      AND date(timestamp) >= date('now', '-90 day')
                    """,
                    (deadband, deadband, source),
                ).fetchone()
                # polarity: raw vs sign-flipped Spearman IC (Batch DC)
                pairs = conn.execute(
                    """
                    SELECT signal_value, actual_direction
                    FROM signal_predictions
                    WHERE source = ?
                      AND signal_value IS NOT NULL
                      AND actual_direction IS NOT NULL
                      AND date(timestamp) >= date('now', '-90 day')
                    """,
                    (source,),
                ).fetchall()
            if not row or not row[0]:
                return None
            n, pred0, mislab, abs_ge, mean_abs, n_pos, n_neg = row
            n = int(n or 0)
            pred0 = int(pred0 or 0)
            mislab = int(mislab or 0)
            abs_ge = int(abs_ge or 0)
            n_pos = int(n_pos or 0)
            n_neg = int(n_neg or 0)
            rate_pred0 = (pred0 / n) if n else None
            pos_rate = (n_pos / n) if n else None

            ic_raw = None
            ic_flipped = None
            if len(pairs) >= 10:
                try:
                    import numpy as np
                    from scipy.stats import spearmanr

                    s = np.asarray([p[0] for p in pairs], dtype=float)
                    a = np.asarray([p[1] for p in pairs], dtype=float)
                    ic_raw = float(spearmanr(s, a).statistic)
                    ic_flipped = float(spearmanr(-s, a).statistic)
                    if ic_raw != ic_raw:  # NaN
                        ic_raw = None
                        ic_flipped = None
                except Exception:  # noqa: BLE001
                    ic_raw = None
                    ic_flipped = None

            issue = None
            if n and rate_pred0 is not None and rate_pred0 > 0.9 and abs_ge > 0:
                issue = (
                    "direction_deadband_collapse — almost all predicted_direction=0 "
                    f"while |signal| often ≥ {deadband:g}; accuracy health is uninformative; "
                    "prefer multi-horizon IC; repair via repair_neutral_predicted_directions."
                )
            elif (
                pos_rate is not None
                and pos_rate > 0.85
                and ic_raw is not None
                and ic_raw < -0.05
                and ic_flipped is not None
                and ic_flipped > 0
            ):
                issue = (
                    "sign_bias_long_with_negative_ic — predictions overwhelmingly "
                    f"positive ({pos_rate:.0%}) while IC={ic_raw:.3f}; flipped IC≈"
                    f"{ic_flipped:.3f}. Do NOT auto-invert; fix classifier polarity "
                    "(Batch DC EQUITY_ROTATION / SPY map) and shadow-monitor."
                )
            elif (
                ic_raw is not None
                and ic_raw < -0.1
                and ic_flipped is not None
                and ic_flipped > abs(ic_raw) * 0.5
            ):
                issue = (
                    f"polarity_flip_hypothesis IC={ic_raw:.3f} vs flipped={ic_flipped:.3f} "
                    "— keep slept; no auto-invert (production health-gate policy)."
                )

            out: Dict[str, Any] = {
                "source": source,
                "window_days": 90,
                "n_rows": n,
                "predicted_zero_rate": None if rate_pred0 is None else round(rate_pred0, 4),
                "mislabeled_neutral_rows": mislab,
                "abs_signal_ge_deadband": abs_ge,
                "mean_abs_signal": None if mean_abs is None else round(float(mean_abs), 4),
                "direction_deadband": deadband,
                "signal_positive_rate": None if pos_rate is None else round(pos_rate, 4),
                "ic_raw": None if ic_raw is None else round(ic_raw, 4),
                "ic_sign_flipped": None if ic_flipped is None else round(ic_flipped, 4),
                "auto_invert_policy": "disabled",
                "alignment_issue": issue,
            }
            # Batch DF/DG: post-fix provenance + min-sample cohort readiness
            try:
                from src.signals.health_tracker import SignalHealthTracker

                prov = SignalHealthTracker().count_provenance_rows(source)
                out["provenance"] = prov
                readiness = prov.get("cohort_readiness") or {}
                out["cohort_readiness"] = readiness
                if prov.get("ic_polarity_cohort") is not None:
                    out["ic_post_polarity_fix"] = prov.get("ic_polarity_cohort")
                if source == "cross_asset_regime_arb" and not readiness.get("ready"):
                    base = issue or ""
                    hint = readiness.get("readiness_hint") or (
                        "post_fix_cohort_thin — shadow IC until min labeled sample"
                    )
                    out["alignment_issue"] = (
                        (base + " | ") if base else ""
                    ) + str(hint)
            except Exception:  # noqa: BLE001
                pass
            return out
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _inactive_signal_shadow_checklist(
        source: str,
        metrics: Dict[str, Any] | None = None,
        activation: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Batch DJ: health/IC shadow for inactive_signal (e.g. intl RS neutral).

        Neutral-band / conf gates keep the arm non-actionable even when multi-horizon
        IC is reentry-eligible. Disclosure only — do not lower RS thresholds or force
        activate without backtest (whipsaw risk).
        """
        m = metrics if isinstance(metrics, dict) else {}
        act = activation if isinstance(activation, dict) else {}
        reentry = m.get("reentry") if isinstance(m.get("reentry"), dict) else None
        if reentry is None:
            reentry = DashboardGenerator._evaluate_ic_reentry(
                ic_30d=m.get("ic_30d"),
                ic_60d=m.get("ic_60d"),
                ic_90d=m.get("ic_90d") if m.get("ic_90d") is not None else m.get("ic"),
            )
        status = str(m.get("status") or "").lower()
        try:
            hs = float(m["health_score"]) if m.get("health_score") is not None else None
        except (TypeError, ValueError):
            hs = None
        multi_ok = bool(reentry.get("reentry_eligible"))
        # Batch DJ: inactive shadow uses IC reentry + non-toxic status (not the
        # 0.55 soft-delete floor). Health-sleep is IC-toxic; degraded+IC-ok is fine.
        health_ok = multi_ok and status in {"healthy", "degraded", ""}
        health_gates_pass = bool(health_ok)
        gaps = list(act.get("activation_gaps") or [])
        activation_cleared = len(gaps) == 0
        if health_gates_pass and not activation_cleared:
            hint = (
                "Health/IC shadow gates pass but signal inactive (RS neutral band / "
                "conf/risk filters) — keep weight 0; do not lower activation thresholds "
                "without backtest; wait for EFA/EEM lead or conf≥0.5."
            )
        elif not multi_ok:
            blocked = (reentry or {}).get("reentry_blocked_reason") or "ic_pending"
            hint = (
                f"Inactive and IC reentry blocked ({blocked}) — dual hold "
                "(activation + health); shadow-monitor only."
            )
        elif not health_ok:
            hint = (
                "Inactive with weak/toxic health — improve accuracy/status before expecting "
                "activation to matter."
            )
        else:
            hint = "Inactive signal; shadow-monitor health and activation gaps."
        return {
            "source": source,
            "policy": "inactive_signal_shadow_no_force_activate",
            "health_gates_pass": health_gates_pass,
            "activation_cleared": bool(activation_cleared),
            "force_activate": False,
            "gates": {
                "multi_horizon_ic_reentry": multi_ok,
                "health_status_ok": health_ok,
                "activation_gaps_empty": bool(activation_cleared),
            },
            "activation_gaps": gaps,
            "reentry": reentry,
            "reentry_eligible": multi_ok,
            "status": status or None,
            "health_score": hs,
            "ic": m.get("ic"),
            "ic_30d": m.get("ic_30d"),
            "ic_60d": m.get("ic_60d"),
            "ic_90d": m.get("ic_90d"),
            "shadow_hint": hint,
        }

    @staticmethod
    def _zero_baseline_shadow_checklist(
        source: str,
        metrics: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Batch DD: shadow re-enable gates for intentional zero-weight arms.

        Soft-delete keeps the arm on the roster at weight 0. Health/IC may
        recover while economic soft-delete (e.g. net-negative Sharpe) still
        requires ADR/backtest before live weight. Never auto-reenable.
        """
        m = metrics if isinstance(metrics, dict) else {}
        soft = DashboardGenerator.ZERO_BASELINE_SOFT_DELETE.get(
            source,
            "configured baseline weight 0 (soft-delete / intentional skip).",
        )
        reentry = m.get("reentry") if isinstance(m.get("reentry"), dict) else None
        if reentry is None:
            reentry = DashboardGenerator._evaluate_ic_reentry(
                ic_30d=m.get("ic_30d"),
                ic_60d=m.get("ic_60d"),
                ic_90d=m.get("ic_90d") if m.get("ic_90d") is not None else m.get("ic"),
            )
        status = str(m.get("status") or "").lower()
        try:
            hs = float(m["health_score"]) if m.get("health_score") is not None else None
        except (TypeError, ValueError):
            hs = None
        min_hs = float(DashboardGenerator.SHADOW_REENABLE_MIN_HEALTH)
        health_ok = status in {"healthy", "degraded"} and (
            hs is None or hs >= min_hs
        )
        # Prefer healthy for promotion review; degraded+IC ok for shadow only
        health_preferred = status == "healthy" and (hs is None or hs >= min_hs)
        multi_ok = bool(reentry.get("reentry_eligible"))
        # Batch DH: portfolio gate from walk-forward ADR evidence (never auto-weight)
        adr: Dict[str, Any] | None = None
        portfolio_ok = False
        if source == "multi_speed_momentum":
            try:
                from src.backtest.multi_speed_momentum_backtest import (
                    evaluate_msm_soft_delete_adr,
                )

                adr = evaluate_msm_soft_delete_adr()
                portfolio_ok = bool(adr.get("portfolio_gates_pass"))
            except Exception:  # noqa: BLE001
                adr = {
                    "adr_status": "evaluation_error",
                    "portfolio_gates_pass": False,
                    "auto_reenable": False,
                    "hint": "ADR evaluation failed — keep soft-delete.",
                }
                portfolio_ok = False
        gates = {
            "multi_horizon_ic_reentry": multi_ok,
            "health_status_ok": health_ok,
            "health_preferred_healthy": health_preferred,
            "min_health_score": hs is None or hs >= min_hs,
            "soft_delete_adr_cleared": portfolio_ok,
        }
        health_gates_pass = bool(
            multi_ok and health_ok and (hs is None or hs >= min_hs)
        )
        # shadow_reenable_ready still False always — human REGIME_WEIGHTS ADR only
        if health_gates_pass and portfolio_ok:
            hint = (
                "Health/IC + walk-forward ADR evidence pass — still requires "
                "human REGIME_WEIGHTS promote; do not auto-reenable weight."
            )
        elif health_gates_pass and not portfolio_ok:
            adr_hint = (adr or {}).get("hint") if isinstance(adr, dict) else None
            hint = (
                adr_hint
                or (
                    "Health/IC shadow gates pass — still soft-deleted until "
                    "walk-forward net Sharpe ADR clears; do not auto-reenable weight."
                )
            )
        elif multi_ok and not health_ok:
            hint = (
                "Multi-horizon IC clear but health status/score weak — "
                "keep zero_baseline; improve accuracy before promotion review."
            )
        elif not multi_ok:
            blocked = (reentry or {}).get("reentry_blocked_reason") or "ic_pending"
            hint = (
                f"Shadow re-enable blocked on IC ({blocked}); "
                "keep weight 0 and shadow-monitor only."
            )
        else:
            hint = "Shadow-monitor zero_baseline arm; re-enable only after ADR."

        out: Dict[str, Any] = {
            "source": source,
            "policy": "soft_delete_shadow_no_auto_reenable",
            "soft_delete_reason": soft,
            "health_gates_pass": health_gates_pass,
            "portfolio_gates_pass": portfolio_ok,
            "shadow_reenable_ready": False,  # hard: human promote only
            "gates": gates,
            "reentry": reentry,
            "reentry_eligible": bool(reentry.get("reentry_eligible")),
            "status": status or None,
            "health_score": hs,
            "ic": m.get("ic"),
            "ic_30d": m.get("ic_30d"),
            "ic_60d": m.get("ic_60d"),
            "ic_90d": m.get("ic_90d"),
            "shadow_hint": hint,
        }
        if isinstance(adr, dict):
            out["adr"] = adr
        return out

    @staticmethod
    def _build_configured_source_status(
        regime: Any,
        source_breakdown: List[Dict[str, Any]],
        health_gate_slept: Dict[str, str] | None = None,
        regime_gated: Dict[str, str] | None = None,
        health_metrics: Dict[str, Dict[str, Any]] | None = None,
        health_gate_soft_floor: Dict[str, str] | None = None,
    ) -> List[Dict[str, Any]]:
        """Explain configured source state, including missing stale configured sources."""
        configured_weights = DashboardGenerator._get_configured_ensemble_source_weights(regime)
        if not configured_weights:
            return []

        slept_map = {
            str(k): str(v)
            for k, v in (health_gate_slept or {}).items()
            if k is not None
        }
        regime_map = {
            str(k): str(v)
            for k, v in (regime_gated or {}).items()
            if k is not None
        }
        soft_floor_map = {
            str(k): str(v)
            for k, v in (health_gate_soft_floor or {}).items()
            if k is not None
        }
        metrics = health_metrics if health_metrics is not None else {}
        zero_baseline_sources = {
            str(s)
            for s, w in configured_weights.items()
            if float(w or 0.0) <= 0.0
        }
        # Batch DJ: inactive intl etc. also need SH metrics for shadow checklist
        if not metrics and (
            slept_map
            or zero_baseline_sources
            or any(
                isinstance(r, dict) and r.get("is_active") is False
                for r in (source_breakdown or [])
            )
        ):
            # Batch CZ/DD/DJ: recovery + zero_baseline + inactive_signal shadow
            metrics = DashboardGenerator._signal_health_metrics_map()

        rows_by_source = {
            str(row.get("source", "")): row
            for row in source_breakdown
            if isinstance(row, dict) and row.get("source")
        }
        statuses: List[Dict[str, Any]] = []
        dropped_weight_mass = 0.0
        contributing_mass = 0.0

        for source, configured_weight in configured_weights.items():
            cfg_w = float(configured_weight or 0.0)
            row = rows_by_source.get(source)
            collected = row is not None
            effective_weight = 0.0
            sleep_reason = slept_map.get(source)
            regime_reason = regime_map.get(source)
            # Batch DM: soft-delete (configured baseline 0) never contributes vote
            # mass in disclosure — even if source_breakdown leaked positive weight
            # from a pre-pin bandit path. Collect/provenance still allowed (DJ).
            soft_delete = cfg_w <= 0.0

            if row is not None:
                try:
                    row_weight = float(row.get("weight", 0.0))
                except (TypeError, ValueError):
                    row_weight = 0.0
                if soft_delete:
                    # Pin: ignore leaked vote weight for soft-delete arms
                    contributing = False
                    effective_weight = 0.0
                    status = "zero_baseline"
                    soft = DashboardGenerator.ZERO_BASELINE_SOFT_DELETE.get(source)
                    reason = (
                        "Configured baseline weight is 0 (soft-delete); "
                        "collected for provenance/shadow only — not contributing "
                        "to the ensemble vote (Batch DM disclosure pin)."
                    )
                    if soft:
                        reason = f"{reason} Soft-delete: {soft}"
                    if abs(row_weight) > 1e-12:
                        reason = (
                            f"{reason} Note: raw vote weight {row_weight:.5f} "
                            "ignored (vote-mass pin / sleeping-expert policy)."
                        )
                else:
                    contributing = bool(np.isfinite(row_weight) and row_weight > 0)
                    effective_weight = row_weight if contributing else 0.0
                    if contributing:
                        # Batch DU: soft-floor unhealthy/degraded still vote — disclose
                        if source in soft_floor_map:
                            status = "active_soft_floor"
                            reason = (
                                "Contributing under health soft-floor (not hard-slept): "
                                f"{soft_floor_map[source]}"
                            )
                        else:
                            status = "active"
                            reason = "Collected and contributing to the ensemble vote."
                    elif sleep_reason:
                        # Batch CW: CN health-gate sleep is not a generic zero_weight
                        status = "health_sleep"
                        reason = f"Health-gated sleep: {sleep_reason}"
                    elif regime_reason:
                        # Batch CX: intentional regime OFF (e.g. unified_overlay in NORMAL)
                        status = "regime_gate"
                        reason = f"Regime-gated off: {regime_reason}"
                    elif row is not None and row.get("is_active") is False:
                        # Batch CY: snapshot inactive (neutral/low conf) ≠ pipeline zero
                        status = "inactive_signal"
                        expl = str(
                            row.get("inactive_explanation")
                            or row.get("explanation")
                            or ""
                        )
                        reason = (
                            f"Signal inactive (not actionable): {expl}"
                            if expl
                            else "Signal inactive (not actionable this cycle)."
                        )
                        # Batch DB: structured RS activation gaps for international
                        if source == "international_momentum":
                            try:
                                conf_raw = row.get("confidence")
                            except Exception:  # noqa: BLE001
                                conf_raw = None
                            try:
                                val_raw = row.get("value")
                            except Exception:  # noqa: BLE001
                                val_raw = None
                            act = DashboardGenerator._international_activation_disclosure(
                                explanation=expl,
                                value=val_raw,
                                confidence=conf_raw,
                            )
                            # stash on row via reason append after entry built — use local
                            row["_activation_disclosure"] = act
                            gaps = act.get("activation_gaps") or []
                            if gaps:
                                reason = (
                                    f"{reason} | activation: {', '.join(gaps[:3])}"
                                )
                    else:
                        status = "zero_weight"
                        reason = "Collected but assigned zero effective weight."
            else:
                contributing = False
                effective_weight = 0.0
                # Batch CU: intentional zero-baseline (e.g. multi_speed_momentum
                # weight 0.0 all regimes) is skipped by collector — disclose as
                # zero_baseline, not "missing" (SRE: zero-weight arm ≠ failure).
                if soft_delete:
                    status = "zero_baseline"
                    soft = DashboardGenerator.ZERO_BASELINE_SOFT_DELETE.get(source)
                    reason = (
                        "Configured baseline weight is 0 for this regime; "
                        "collector intentionally skips (not a fetch failure)."
                    )
                    if soft:
                        reason = f"{reason} Soft-delete: {soft}"
                else:
                    status = "missing"
                    reason = (
                        "Configured source did not produce an active ensemble reading."
                    )
                    if source == "google_trends":
                        status, reason = (
                            DashboardGenerator._google_trends_inactive_disclosure()
                        )

            if contributing:
                contributing_mass += effective_weight
            else:
                # Stale/missing/zero: configured mass does not participate in vote
                # Zero baseline drops 0 mass but still discloses status (Batch CU)
                dropped_weight_mass += cfg_w

            entry: Dict[str, Any] = {
                "source": source,
                "label": DashboardGenerator._format_ensemble_source_label(source),
                "configured": True,
                "configured_weight": round(cfg_w, 5),
                "effective_weight": round(effective_weight, 5),
                "collected": collected,
                "active": collected and contributing,
                "contributing": contributing,
                "status": status,
                "reason": reason,
            }
            if sleep_reason:
                entry["health_sleep_reason"] = sleep_reason
            if source in soft_floor_map:
                entry["health_soft_floor_reason"] = soft_floor_map[source]
            if regime_reason:
                entry["regime_gate_reason"] = regime_reason
            # Batch DB: international activation checklist on inactive rows
            if (
                status == "inactive_signal"
                and source == "international_momentum"
                and isinstance(row, dict)
                and isinstance(row.get("_activation_disclosure"), dict)
            ):
                entry["activation"] = row["_activation_disclosure"]
                if entry["activation"].get("activation_hint"):
                    entry["activation_hint"] = entry["activation"]["activation_hint"]
            # Batch CZ: attach SH recovery metrics for slept / degraded inactive arms
            m = metrics.get(source) if isinstance(metrics, dict) else None
            if isinstance(m, dict) and (
                status in {"health_sleep", "inactive_signal", "zero_baseline"}
                or (status == "active" and (m.get("health_score") or 1) < 0.55)
            ):
                entry["health_metrics"] = m
                if status == "health_sleep" and m.get("recovery_hint"):
                    entry["recovery_hint"] = m["recovery_hint"]
                    # Append concise recovery cue to reason for compact UIs
                    entry["reason"] = f"{reason} | recovery: {m['recovery_hint']}"
                # Batch DA: surface reentry hysteresis at row level
                if status == "health_sleep" and "reentry" in m:
                    entry["reentry"] = m["reentry"]
                    entry["reentry_eligible"] = bool(m.get("reentry_eligible"))
                # Batch DB/DC/DG: label/direction/polarity + post-fix cohort readiness
                if status == "health_sleep":
                    diag = DashboardGenerator._label_alignment_diagnostic(source)
                    if diag:
                        entry["label_alignment"] = diag
                        readiness = diag.get("cohort_readiness") or {}
                        if readiness:
                            entry["cohort_readiness"] = readiness
                            entry["post_fix_cohort_ready"] = bool(
                                readiness.get("ready")
                            )
                        if diag.get("alignment_issue"):
                            entry["reason"] = (
                                f"{entry['reason']} | label: {diag['alignment_issue']}"
                            )
                            # Prefer polarity guidance over generic deep-neg when present
                            if "auto_invert" in (diag.get("alignment_issue") or "").lower() or (
                                "sign_bias" in (diag.get("alignment_issue") or "")
                                or "polarity" in (diag.get("alignment_issue") or "")
                                or "label_lag" in (diag.get("alignment_issue") or "")
                                or "cohort" in (diag.get("alignment_issue") or "")
                            ):
                                entry["recovery_hint"] = (
                                    readiness.get("readiness_hint")
                                    if readiness and not readiness.get("ready")
                                    else (
                                        "Polarity/sign-bias detected — do not auto-invert; "
                                        "Batch DC maps EQUITY_ROTATION to equity regime sign; "
                                        "shadow-monitor IC after classifier fix before reentry."
                                    )
                                )
                                entry["reason"] = (
                                    f"{entry['reason']} | recovery: {entry['recovery_hint']}"
                                )
            # Batch DD: zero_baseline shadow re-enable checklist (never auto-weight)
            if status == "zero_baseline":
                shadow = DashboardGenerator._zero_baseline_shadow_checklist(
                    source, m if isinstance(m, dict) else {}
                )
                entry["shadow"] = shadow
                entry["shadow_hint"] = shadow.get("shadow_hint")
                entry["health_gates_pass"] = shadow.get("health_gates_pass")
                entry["shadow_reenable_ready"] = False
                entry["reason"] = f"{entry['reason']} | shadow: {shadow.get('shadow_hint')}"
            # Batch DJ: inactive_signal health/IC shadow (intl RS neutral etc.)
            if status == "inactive_signal":
                act = entry.get("activation") if isinstance(entry.get("activation"), dict) else None
                ishadow = DashboardGenerator._inactive_signal_shadow_checklist(
                    source,
                    m if isinstance(m, dict) else {},
                    act,
                )
                entry["shadow"] = ishadow
                entry["shadow_hint"] = ishadow.get("shadow_hint")
                entry["health_gates_pass"] = ishadow.get("health_gates_pass")
                entry["force_activate"] = False
                entry["reason"] = f"{entry['reason']} | shadow: {ishadow.get('shadow_hint')}"
            statuses.append(entry)

        # Renormalize over contributors so sum(active_weight) ≈ 1 when any active
        for row in statuses:
            if contributing_mass > 0 and row.get("contributing"):
                row["active_weight"] = round(
                    float(row["effective_weight"]) / contributing_mass, 5
                )
            else:
                row["active_weight"] = 0.0

        return statuses

    # Batch DP: match EnsembleVoter.DEFAULT_PER_SIGNAL_WEIGHT_CAP for rollup safety
    PER_SIGNAL_ACTIVE_WEIGHT_CAP = 0.50

    @staticmethod
    def _ensemble_active_weights_rollup(
        configured_source_status: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Rollup renormed active weights + dropped configured mass after stale drop.

        Batch DP: after renorm over contributing arms, clip to 50% per-signal
        and water-fill so dashboard active_weights never re-concentrates past
        the voter cap when inactive/slept mass was dropped upstream.
        """
        active_weights: Dict[str, float] = {}
        dropped = 0.0
        active_mass = 0.0
        for row in configured_source_status or []:
            if not isinstance(row, dict):
                continue
            name = str(row.get("source") or "")
            if row.get("contributing"):
                aw = float(row.get("active_weight") or 0.0)
                active_weights[name] = aw
                active_mass += float(row.get("effective_weight") or 0.0)
            else:
                dropped += float(row.get("configured_weight") or 0.0)

        cap = DashboardGenerator.PER_SIGNAL_ACTIVE_WEIGHT_CAP
        capped = False
        if active_weights:
            # Renorm to simplex first
            total0 = sum(max(0.0, float(v)) for v in active_weights.values())
            if total0 > 0:
                active_weights = {
                    k: max(0.0, float(v)) / total0 for k, v in active_weights.items()
                }
            # Single contributing arm cannot diversify — leave at 1.0
            positive = [k for k, v in active_weights.items() if v > 1e-12]
            if len(positive) >= 2:
                for _ in range(16):
                    over = [k for k, v in active_weights.items() if v > cap + 1e-12]
                    if not over:
                        break
                    capped = True
                    excess = 0.0
                    for k in over:
                        excess += active_weights[k] - cap
                        active_weights[k] = cap
                    under = [
                        k for k, v in active_weights.items() if v < cap - 1e-12
                    ]
                    if not under:
                        break
                    under_sum = sum(active_weights[k] for k in under)
                    if under_sum <= 0:
                        share = excess / len(under)
                        for k in under:
                            active_weights[k] = min(
                                cap, active_weights[k] + share
                            )
                    else:
                        scale = (under_sum + excess) / under_sum
                        for k in under:
                            active_weights[k] = min(
                                cap, active_weights[k] * scale
                            )
                # Final renorm if clip left mass short
                total1 = sum(active_weights.values())
                if total1 > 0 and abs(total1 - 1.0) > 1e-9 and not any(
                    v > cap + 1e-12 for v in active_weights.values()
                ):
                    active_weights = {
                        k: v / total1 for k, v in active_weights.items()
                    }
            active_weights = {
                k: round(float(v), 5) for k, v in active_weights.items()
            }

        disclosure = (
            "active_weights renormalized over contributing sources; "
            "stale/missing configured mass in dropped_weight_mass"
        )
        if capped:
            disclosure += (
                f"; Batch DP per-signal cap {cap:.0%} applied after renorm"
            )
        return {
            "active_weights": active_weights,
            "dropped_weight_mass": round(dropped, 5),
            "active_weight_mass": round(active_mass, 5),
            "active_weights_sum": round(sum(active_weights.values()), 5),
            "weight_disclosure": disclosure,
            "per_signal_active_weight_cap": cap,
            "per_signal_active_weight_cap_applied": capped,
        }

    @staticmethod
    def _build_ensemble_adaptive_learning_disclosure(ensemble_result: Any) -> Dict[str, Any]:
        """Serialize adaptive-learning branch status from an ensemble vote."""
        disclosure = getattr(ensemble_result, "adaptive_learning", {})
        return disclosure if isinstance(disclosure, dict) else {}

    @staticmethod
    def _build_allocation_surface_roles(data_dir: Path | None = None) -> Dict[str, Any]:
        """Describe the current live-routing role of allocation-like signals surfaces.

        When the kill switch is enabled, target_allocations remains the routing
        surface but is disclosed as execution-blocked (not live_authoritative).
        """
        advisory_description = (
            "Published for advisory diagnostics; current order routing uses "
            "target_allocations."
        )
        roles: Dict[str, Any] = {
            "schema_version": "allocation-surface-roles/v1",
            "routed_surface": "target_allocations",
            "routed_by": "src.broker.order_router",
            "surfaces": {
                "target_allocations": {
                    "label": "Target Allocation",
                    "role": "execution_routed",
                    "routed": True,
                    "routed_by": "src.broker.order_router",
                    "live_authoritative": True,
                    "canonical_controller": "signals.json.target_allocations",
                    "description": (
                        "Current order-routing input consumed by src.broker.order_router."
                    ),
                },
                "ensemble_voting": {
                    "label": "Ensemble Voting",
                    "role": "advisory_non_routed",
                    "routed": False,
                    "routed_by": None,
                    "live_authoritative": False,
                    "canonical_controller": "signals.json.target_allocations",
                    "description": advisory_description,
                },
                "adaptive_sizing": {
                    "label": "Adaptive Sizing",
                    "role": "advisory_non_routed",
                    "routed": False,
                    "routed_by": None,
                    "live_authoritative": False,
                    "canonical_controller": "signals.json.target_allocations",
                    "description": advisory_description,
                },
                "black_litterman": {
                    "label": "Black-Litterman",
                    "role": "advisory_non_routed",
                    "routed": False,
                    "routed_by": None,
                    "live_authoritative": False,
                    "canonical_controller": "signals.json.target_allocations",
                    "description": advisory_description,
                },
                "calendar_seasonality": {
                    "label": "Calendar Seasonality",
                    "role": "advisory_non_routed",
                    "routed": False,
                    "routed_by": None,
                    "live_authoritative": False,
                    "applies_to_target_allocations": False,
                    "canonical_controller": "signals.json.target_allocations",
                    "description": (
                        "Urgency/execution timing advisory only. "
                        "modifier does not scale target_allocations "
                        "(paper book stays champion weights)."
                    ),
                },
                "factor_rotation": {
                    "label": "Factor Rotation",
                    "role": "advisory_non_routed",
                    "routed": False,
                    "routed_by": None,
                    "live_authoritative": False,
                    "allocation_field": "allocation",
                    "canonical_controller": "signals.json.target_allocations",
                    "description": (
                        "Advisory factor sleeve weights (e.g. VLUE/VBR). "
                        "Not order-routed; live routing uses target_allocations."
                    ),
                },
            },
        }
        root = Path(data_dir) if data_dir is not None else DATA_DIR
        kill = project_kill_switch_fields(load_kill_switch_payload(root))
        if kill.get("enabled"):
            roles = allocation_roles_under_kill(
                roles,
                kill_enabled=True,
                kill_level=kill.get("level"),
            )
        return roles

    @staticmethod
    def _build_advisory_allocation_artifact_role(
        surface: str,
        allocation_field: str,
    ) -> Dict[str, Any]:
        """Describe a standalone allocation artifact as advisory/non-routed."""
        return {
            "schema_version": "allocation-artifact-role/v1",
            "surface": surface,
            "allocation_field": allocation_field,
            "runtime_role": "advisory_non_routed",
            "live_authoritative": False,
            "routed": False,
            "routed_by": None,
            "canonical_controller": "signals.json.target_allocations",
            "routed_surface": "target_allocations",
            "routed_surface_path": "public/data/signals.json#target_allocations",
            "description": (
                f"{surface} is published for advisory diagnostics; live order routing "
                "continues to consume signals.json.target_allocations."
            ),
        }

    @staticmethod
    def _flatten_advisory_authority(authority: Dict[str, Any]) -> Dict[str, Any]:
        """Top-level authority fields for operator greps (vixy_hedge pattern).

        Nested ``authority`` remains the schema contract for AdaptiveSizingPanel;
        top-level mirrors make ``runtime_role`` / ``routed`` visible without
        digging into the nested block.
        """
        return {
            "runtime_role": authority.get("runtime_role"),
            "live_authoritative": authority.get("live_authoritative"),
            "routed": authority.get("routed"),
            "routed_by": authority.get("routed_by"),
            "canonical_controller": authority.get("canonical_controller"),
            "routed_surface": authority.get("routed_surface"),
        }

    @staticmethod
    def _canonicalize_public_weights(
        weights: Dict[str, Any],
        canonical_assets: tuple[str, ...] = ("SPY", "GLD", "TLT"),
    ) -> Dict[str, Any]:
        """Uppercase public weight keys and preserve zero-weight diagnostics."""
        normalized: Dict[str, float] = {}
        excluded_assets: list[str] = []
        for symbol, raw_weight in (weights or {}).items():
            canonical = str(symbol).upper()
            try:
                normalized[canonical] = float(raw_weight)
            except (TypeError, ValueError):
                excluded_assets.append(canonical)

        public_weights = {
            symbol: normalized.get(symbol, 0.0)
            for symbol in canonical_assets
        }
        zero_weight_assets = [
            symbol for symbol, weight in public_weights.items()
            if abs(weight) < 1e-12
        ]

        return {
            "weights": public_weights,
            "excluded_assets": sorted(set(excluded_assets)),
            "zero_weight_assets": zero_weight_assets,
        }

    @staticmethod
    def _build_regime_authority(
        current_regime: str,
        target_alloc: Dict[str, float],
    ) -> Dict[str, Any]:
        """Document the live regime controller and advisory role of advanced regimes."""
        return {
            "schema_version": "regime-authority/v1",
            "live_controller": "classify_vix_regime",
            "live_controller_module": "src.utils.classify_vix_regime",
            "live_regime": current_regime,
            "allocation_regime": normalize_allocation_regime(current_regime) or "normal",
            "routed_surface": "target_allocations",
            "target_allocations": target_alloc,
            "advanced_regime_signals": {
                "two_stage_regime": {
                    "role": "advisory_shadow",
                    "routed": False,
                    "availability": "unknown",
                    "published": False,
                    "description": "Availability pending staleness check; not live order-routing authority.",
                },
                "bocd_regime": {
                    "role": "advisory_shadow",
                    "routed": False,
                    "availability": "unknown",
                    "published": False,
                    "description": "Availability pending staleness check; not live order-routing authority.",
                },
                "regime_transition": {
                    "role": "advisory_shadow",
                    "routed": False,
                    "availability": "unknown",
                    "published": False,
                    "description": "Availability pending staleness check; not live order-routing authority.",
                },
            },
        }

    @staticmethod
    def _dedupe_preserve_order(values: list[str]) -> list[str]:
        """Return unique string identifiers while preserving first occurrence order."""
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            key = str(value)
            if key in seen:
                continue
            seen.add(key)
            result.append(key)
        return result

    @classmethod
    def _update_regime_authority_availability(cls, output: Dict[str, Any]) -> None:
        """Update advanced regime authority entries with observed snapshot state."""
        authority = output.get("regime_authority")
        if not isinstance(authority, dict):
            return

        advanced = authority.get("advanced_regime_signals")
        if not isinstance(advanced, dict):
            return

        staleness = output.get("staleness") if isinstance(output.get("staleness"), dict) else {}
        unavailable = set(staleness.get("unavailable_signals") or [])
        stale = set(staleness.get("stale_signals") or [])

        for signal_name, entry in advanced.items():
            if not isinstance(entry, dict):
                continue

            signal_block = output.get(signal_name)
            if signal_name in unavailable or signal_block is None:
                availability = "unavailable"
                published = False
                description = "Unavailable in this snapshot; not live order-routing authority."
            elif signal_name in stale:
                availability = "stale"
                published = False
                description = "Stale in this snapshot; not live order-routing authority."
            elif cls._is_unavailable_signal_block(signal_block):
                availability = "error"
                published = False
                description = "Error or degraded placeholder in this snapshot; not live order-routing authority."
            else:
                availability = "present"
                published = True
                description = "Published for advisory diagnostics; not live order-routing authority."

            entry.update({
                "availability": availability,
                "published": published,
                "description": description,
            })

    def _build_base_signal_sections(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build the core signal sections before dashboard-level metadata."""
        vix_level = context["vix_level"]
        trend_regime = context["trend_regime"]
        current_regime = context["current_regime"]
        regime_data = context["regime_data"]
        latest = context["latest"]
        positions = context["positions"]
        cash = context["cash"]
        total_value = context["total_value"]
        target_alloc = context["target_alloc"]
        orders = context["orders"]

        # Add factor rotation signals if engine available
        factor_rotation_signal = None
        factor_rotation_result = None
        try:
            from src.strategy.factor_rotation import FactorMomentumEngine
            engine = FactorMomentumEngine()
            factor_rotation_result = engine.evaluate()
            if factor_rotation_result and "error" not in factor_rotation_result:
                now_ts = datetime.now(timezone.utc).isoformat()
                strength = float(factor_rotation_result.get("signal_strength", 0.0) or 0.0)
                allocations = factor_rotation_result.get("allocation", {})
                # Single canonical payload (no dual top-level weight fork)
                factor_rotation_signal = {
                    "selected_factors": factor_rotation_result.get("selected_factors", []),
                    "allocation": allocations,
                    "factor_allocations": allocations,
                    "signal_strength": strength,
                    "recommendation": factor_rotation_result.get("recommendation", {}),
                    "active": True,
                    "live_authoritative": False,
                    "role": "advisory_non_routed",
                    "canonical_controller": "signals.json.target_allocations",
                    "research_caveats": [
                        {
                            "kind": "research_caveat",
                            "role": "non_actionable",
                            "summary": (
                                "Factor rotation reduces MaxDD by 5.8pp (2021-2026) in "
                                "backtests; advisory sleeve only."
                            ),
                        }
                    ],
                    # Staleness TTL requires generated_at; missing field → optional unavailable.
                    "generated_at": now_ts,
                    "timestamp": now_ts,
                }
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("factor_rotation", e)

        # Add yield curve data from yields.json
        yield_curve_data = self._get_yield_curve_data()
        
        # Add volatility parity / convexity harvest signals
        convexity_signal = None
        vol_parity_signal = None
        try:
            from src.strategy.convexity_harvest import ConvexityHarvestStrategy
            from src.strategy.vol_parity_allocator import VolatilityParityAllocator
            
            # Get convexity harvest signal
            convexity_engine = ConvexityHarvestStrategy()
            convexity_signal = convexity_engine.get_current_signal()
            if isinstance(convexity_signal, dict):
                # Ensure TTL fields always present for staleness classifier
                now_ts = datetime.now(timezone.utc).isoformat()
                convexity_signal.setdefault("generated_at", now_ts)
                convexity_signal.setdefault("timestamp", now_ts)

            # Get volatility parity allocation (full to_dict provenance —
            # weight_unit / role / live_authoritative — not bare pct fields only)
            vol_allocator = VolatilityParityAllocator(vix_strategy=convexity_engine)
            vol_parity_data = vol_allocator.get_current_allocation()
            if vol_parity_data:
                alloc = vol_parity_data.get("allocation")
                if isinstance(alloc, dict):
                    vol_parity_signal = dict(alloc)
                    # Ensure advisory provenance even if older to_dict path
                    now_ts = datetime.now(timezone.utc).isoformat()
                    vol_parity_signal.setdefault(
                        "weight_unit", "percent_of_portfolio_0_100"
                    )
                    vol_parity_signal.setdefault("live_authoritative", False)
                    vol_parity_signal.setdefault("role", "advisory_research_sleeve")
                    vol_parity_signal.setdefault("generated_at", now_ts)
                    vol_parity_signal.setdefault("timestamp", now_ts)
                else:
                    vol_parity_signal = alloc
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("convexity_harvest", e)

        # Add LLM sentiment signals (v2.30 Phase 5)
        sentiment_signal = None
        try:
            from src.strategy.regime_sentiment import RegimeSentimentPipeline
            
            sentiment_pipeline = RegimeSentimentPipeline()
            # Get current technical regime for combination
            tech_regime = trend_regime if trend_regime else "neutral"
            tech_confidence = 0.6  # Default confidence
            
            # Get combined sentiment signal (mock mode if no API keys)
            sentiment_signal = sentiment_pipeline.get_combined_signal(
                technical_regime=tech_regime,
                technical_confidence=tech_confidence,
                news_texts=[],  # Empty for mock mode
                earnings_texts=[],
                macro_texts=[],
            )
            sentiment_signal = sentiment_signal.to_dict()
            # Honesty: empty news texts → mock/unavailable sentiment, not live NLP
            empty_inputs = True  # this call path always passes empty lists today
            sentiment_signal["source_mode"] = "mock_empty_inputs" if empty_inputs else "live"
            sentiment_signal["live_authoritative"] = False
            sentiment_signal["role"] = "advisory_shadow"
            if empty_inputs or float(sentiment_signal.get("sentiment_confidence") or 0) == 0.0:
                sentiment_signal["sentiment_status"] = "unavailable_no_news_inputs"
                sentiment_signal["sentiment_status_reason"] = (
                    "news/earnings/macro texts empty — sentiment_confidence=0 is "
                    "mock placeholder, not measured market neutral"
                )
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("llm_sentiment", e)

        # Add ensemble voting signals (v2.20 Phase 3)
        ensemble_signal = None
        try:
            from src.strategy.ensemble_voter import EnsembleVoter

            ensemble_engine = EnsembleVoter()
            # Daily reward train (advisory bandit): prefer daily contribution
            # credit (Batch BR) then windowed attribution (Batch BQ); fall back
            # to scalar only for single-arm / multi-arm skip (BO).
            # Failures never block vote. Bandit remains non-authoritative.
            try:
                daily_ret = EnsembleVoter.load_latest_daily_return_from_performance()
                if daily_ret is not None:
                    src_rewards, reward_mode = (
                        EnsembleVoter.load_preferred_source_rewards()
                    )
                    ensemble_engine.apply_daily_bandit_rewards(
                        daily_ret,
                        persist=True,
                        source_rewards=src_rewards,
                        reward_mode=reward_mode if src_rewards else None,
                    )
            except SIGNAL_EXCEPTIONS as bandit_exc:
                logger.debug("ensemble bandit daily reward skipped: %s", bandit_exc)

            ensemble_result = ensemble_engine.compute_vote()
            if ensemble_result:
                source_breakdown = self._build_ensemble_source_breakdown(
                    ensemble_result.source_votes
                )
                sleep_map = getattr(ensemble_result, "health_gate_slept", None) or {}
                if not isinstance(sleep_map, dict):
                    sleep_map = {}
                regime_map = getattr(ensemble_result, "regime_gated", None) or {}
                if not isinstance(regime_map, dict):
                    regime_map = {}
                soft_floor_map = getattr(
                    ensemble_result, "health_gate_soft_floor", None
                ) or {}
                if not isinstance(soft_floor_map, dict):
                    soft_floor_map = {}
                sh_metrics = DashboardGenerator._signal_health_metrics_map()
                configured_source_status = self._build_configured_source_status(
                    ensemble_result.regime,
                    source_breakdown,
                    health_gate_slept=sleep_map,
                    regime_gated=regime_map,
                    health_metrics=sh_metrics,
                    health_gate_soft_floor=soft_floor_map,
                )
                source_counts = self._build_ensemble_source_count_metadata(
                    ensemble_result.regime,
                    source_breakdown,
                    configured_source_status=configured_source_status,
                )
                weight_rollup = self._ensemble_active_weights_rollup(
                    configured_source_status
                )
                zero_baseline_shadow = [
                    row["shadow"]
                    for row in configured_source_status
                    if isinstance(row, dict)
                    and row.get("status") == "zero_baseline"
                    and isinstance(row.get("shadow"), dict)
                ]
                # Batch DQ: surface concentration SLI on ensemble payload
                aw_map = weight_rollup.get("active_weights") or {}
                max_aw = max(aw_map.values()) if aw_map else 0.0
                ensemble_signal = {
                    "regime": ensemble_result.regime.value,
                    "regime_confidence": ensemble_result.regime_confidence,
                    "weighted_consensus": ensemble_result.weighted_consensus,
                    "agreement_ratio": ensemble_result.agreement_ratio,
                    "action": ensemble_result.action,
                    "confidence": ensemble_result.confidence,
                    "equity_bias": round(ensemble_result.equity_bias, 3),
                    "duration_bias": round(ensemble_result.duration_bias, 3),
                    "gold_bias": round(ensemble_result.gold_bias, 3),
                    **source_counts,
                    **weight_rollup,
                    "configured_source_status": configured_source_status,
                    "n_eff": round(getattr(ensemble_result, 'n_eff', 0), 2),
                    "weight_entropy": round(getattr(ensemble_result, 'weight_entropy', 0), 4),
                    "max_active_weight": round(float(max_aw), 5),
                    "ensemble_concentration_ok": bool(
                        float(max_aw) <= float(
                            weight_rollup.get("per_signal_active_weight_cap")
                            or DashboardGenerator.PER_SIGNAL_ACTIVE_WEIGHT_CAP
                        )
                        + 1e-6
                    ),
                    "adaptive_learning": self._build_ensemble_adaptive_learning_disclosure(
                        ensemble_result
                    ),
                    "source_breakdown": source_breakdown,
                    # Batch CW: top-level sleep disclosure for ops panels
                    "health_gate_slept": sleep_map,
                    "health_gate_freeze": bool(
                        getattr(ensemble_result, "health_gate_freeze", False)
                    ),
                    "health_gate_slept_count": len(sleep_map),
                    # Batch DU: soft-floor (unhealthy still voting with IC≥min)
                    "health_gate_soft_floor": soft_floor_map,
                    "health_gate_soft_floor_count": len(soft_floor_map),
                    # Batch CX: regime-gate OFF disclosure
                    "regime_gated": regime_map,
                    "regime_gated_count": len(regime_map),
                    # Batch CZ/DA/DB: recovery checklist + label alignment
                    "health_gate_recovery": [
                        {
                            "source": name,
                            "sleep_reason": sleep_map.get(name),
                            **(sh_metrics.get(name) or {}),
                            **(
                                {
                                    "label_alignment": la,
                                }
                                if (
                                    la := DashboardGenerator._label_alignment_diagnostic(
                                        name
                                    )
                                )
                                else {}
                            ),
                        }
                        for name in sorted(sleep_map.keys())
                    ],
                    # Batch DD: zero_baseline soft-delete shadow re-enable (no auto-weight)
                    "zero_baseline_shadow": zero_baseline_shadow,
                    "zero_baseline_shadow_count": len(zero_baseline_shadow),
                    # Batch DJ: inactive_signal shadow (RS neutral but health/IC ok)
                    "inactive_signal_shadow": [
                        row["shadow"]
                        for row in configured_source_status
                        if isinstance(row, dict)
                        and row.get("status") == "inactive_signal"
                        and isinstance(row.get("shadow"), dict)
                    ],
                    # Batch DG: post-fix polarity cohort readiness (regime_arb etc.)
                    "post_fix_cohorts": [
                        {
                            "source": row.get("source"),
                            **(row.get("cohort_readiness") or {}),
                        }
                        for row in configured_source_status
                        if isinstance(row, dict)
                        and isinstance(row.get("cohort_readiness"), dict)
                    ],
                }
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("ensemble_voting", e)

        # Sector rotation is generated later (after overlay merge) so VIX can
        # fall back to term-structure spot when market.db lacks ^VIX.
        sector_momentum_signal = None

        # Add smart rebalancing status (v2.90)
        smart_rebalance_data = None
        try:
            import importlib
            rebalancing_pkg = importlib.import_module('src.rebalancing')
            SmartRebalanceGate = rebalancing_pkg.integration.SmartRebalanceGate

            gate = SmartRebalanceGate()
            # Build current holdings from positions
            holdings = {p['symbol']: p['value'] for p in positions} if positions else {}
            if holdings and total_value > 0:
                gate_result = gate.evaluate(
                    current_holdings=holdings,
                    target_allocations=target_alloc,
                    total_value=total_value,
                )
                gate_status = gate.get_status()
                remaining_budget_ratio = _remaining_budget_ratio(
                    gate_result.metadata,
                    gate_status,
                )
                smart_rebalance_data = {
                    'should_execute': gate_result.should_execute,
                    'decision': gate_result.decision,
                    'urgency': gate_result.urgency,
                    'max_drift': gate_result.max_drift,
                    'estimated_cost_bps': gate_result.estimated_cost_bps,
                    'reason': gate_result.reason,
                    'drift_details': gate_result.metadata.get('drift_details', {}),
                    'vpin': gate_result.metadata.get('vpin', 0.30),
                    'in_optimal_window': gate_result.metadata.get('in_optimal_window', False),
                    'ytd_cost_bps': gate_result.metadata.get('ytd_cost_bps', 0),
                    'remaining_budget_pct': _remaining_budget_display_pct(
                        remaining_budget_ratio,
                        gate_status,
                    ),
                    'remaining_budget_ratio': remaining_budget_ratio,
                    # Unit honesty: pct is percent-of-portfolio (0.5 = 0.5%),
                    # ratio is portfolio fraction (0.005 = 0.5%).
                    'remaining_budget_pct_unit': 'percent_of_portfolio',
                    'remaining_budget_ratio_unit': 'portfolio_fraction',
                    'annual_cost_limit_pct': 0.5,
                    'status': gate_status,
                }
            else:
                # No positions — use gate status only
                gate_status = gate.get_status()
                remaining_budget_ratio = _remaining_budget_ratio({}, gate_status)
                smart_rebalance_data = {
                    'should_execute': False,
                    'decision': 'no_positions',
                    'urgency': 'low',
                    'max_drift': 0,
                    'estimated_cost_bps': 0,
                    'reason': 'no_positions',
                    'drift_details': {},
                    'vpin': 0.30,
                    'in_optimal_window': False,
                    'ytd_cost_bps': 0,
                    'remaining_budget_pct': _remaining_budget_display_pct(
                        remaining_budget_ratio,
                        gate_status,
                    ),
                    'remaining_budget_ratio': remaining_budget_ratio,
                    'remaining_budget_pct_unit': 'percent_of_portfolio',
                    'remaining_budget_ratio_unit': 'portfolio_fraction',
                    'annual_cost_limit_pct': 0.5,
                    'status': gate_status,
                }
            # Kill authority blocks actionable execute (order_router SSOT)
            smart_rebalance_data = _apply_kill_to_smart_rebalance(
                smart_rebalance_data,
                load_kill_switch_payload(DATA_DIR),
            )
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError, ImportError, RuntimeError) as e:
            logger.warning("Dashboard generation error: %s", e)

        # Add alternative data signals (v2.60 Phase 3)
        alternative_data_signal = None
        try:
            alt_data_file = DATA_DIR / "signals" / "alternative_data_latest.json"
            if alt_data_file.exists():
                with open(alt_data_file) as f:
                    alt_data_raw = json.load(f)
                alternative_data_signal = project_alternative_data_signal(alt_data_raw)
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logger.warning("Alternative data signal not available: %s", e)

        # Load broker data (Phase 4: live trading prep)
        broker_data = self._load_broker_data()

        # Portfolio drift alerting
        try:
            from src.monitor.alerting import check_drift_and_alert
            drift_info = broker_data.get("drift", {})
            max_drift = drift_info.get("max_drift_pct", 0.0) if isinstance(drift_info, dict) else 0.0
            if max_drift:
                check_drift_and_alert(max_drift)
        except (ImportError, ValueError, OSError, RuntimeError) as e:
            _log_signal_error("drift_alerting", e)

        # Add GARCH-CVaR metrics (v3.21)
        garch_cvar_data = self._load_garch_cvar_data()

        # Add entropy diversification metrics (v3.22)
        entropy_data = self._load_entropy_data()

        # Behavioral sentiment data (v2.70)
        behavioral_sentiment_data = None
        try:
            from src.signals.behavioral_sentiment import BehavioralSentimentSignal
            from src.data.behavioral_sentiment_fetcher import BehavioralSentimentFetcher

            sig_gen = BehavioralSentimentSignal(cache_db=DB_PATH)
            fetcher = BehavioralSentimentFetcher(cache_db=DB_PATH)
            snapshot = fetcher.fetch_snapshot()
            signal = sig_gen.get_signal(snapshot)
            status = sig_gen.get_status()

            now_ts = datetime.now(timezone.utc).isoformat()
            # Align active with RegimeGate + producer regime_suppressed (no dual SSOT)
            beh_active = (
                not bool(signal.regime_suppressed)
                and float(signal.confidence) >= 0.3
            )
            behavioral_sentiment_data = {
                "active": beh_active,
                "composite_score": signal.composite_score,
                "signal_type": signal.signal_type,
                "confidence": signal.confidence,
                "equity_shift_pct": signal.equity_shift_pct,
                "z_score": signal.z_score,
                "vix": signal.vix,
                "regime_suppressed": signal.regime_suppressed,
                "signal_count_5d": status.get("signal_count_5d", 0),
                "options": {
                    "skew_index": round(snapshot.options.skew_index, 1),
                    "vix": round(snapshot.options.vix, 1),
                    "vix9d": round(snapshot.options.vix9d, 1),
                    "vix9d_ratio": round(snapshot.options.vix9d_ratio, 2),
                    "put_call_ratio": round(snapshot.options.put_call_ratio, 2),
                    "fear_greed_score": round(snapshot.options.fear_greed_score, 1),
                },
                "retail": {
                    "retail_call_put_ratio": round(snapshot.retail.retail_call_put_ratio, 2),
                    "retail_buy_sell_imbalance": round(snapshot.retail.retail_buy_sell_imbalance, 2),
                },
                "social": {
                    "mention_velocity_7d": round(snapshot.social.mention_velocity_7d, 2),
                    "sentiment_divergence": round(snapshot.social.sentiment_divergence, 3),
                },
                # Research caveat is non-actionable metadata — not a live alpha narrative
                "research_caveats": [
                    {
                        "kind": "research_caveat",
                        "role": "non_actionable",
                        "summary": (
                            "VIX-proxy contrarian signals degraded Sharpe by -0.216 "
                            "(2021-2026) in backtests; live SKEW/PCR path required."
                        ),
                    }
                ],
                "timestamp": now_ts,
                "generated_at": now_ts,
            }
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("behavioral_sentiment", e)

        # Stacking ensemble dashboard data (v3.10)
        stacking_ensemble_dashboard = None
        try:
            from src.signals.stacking_integrator import StackingIntegrator

            integrator = StackingIntegrator()
            if integrator.model is None:
                stacking_ensemble_dashboard = self._build_stacking_no_model_dashboard(integrator)
            else:
                prediction = integrator.predict({})
                stacking_ensemble_dashboard = self._build_stacking_model_dashboard(
                    integrator,
                    prediction,
                )
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("stacking_ensemble", e)

        # Backward-compat alias: nest under factor_rotation (no dual weight SSOT)
        factor_rotation_dashboard = None
        if isinstance(factor_rotation_signal, dict):
            factor_rotation_dashboard = {
                "alias_of": "factor_rotation",
                "live_authoritative": False,
                "role": "advisory_non_routed",
                "active": factor_rotation_signal.get("active", True),
                "selected_factors": factor_rotation_signal.get("selected_factors"),
                # Same strength as canonical (no silent 2-decimal fork)
                "signal_strength": factor_rotation_signal.get("signal_strength"),
                "factor_allocations": factor_rotation_signal.get("allocation"),
                "research_caveats": factor_rotation_signal.get("research_caveats"),
                "generated_at": factor_rotation_signal.get("generated_at"),
                "timestamp": factor_rotation_signal.get("timestamp"),
            }

        # Merge overlay dashboard data (collar, crypto, calendar, kurtosis, etc.)
        overlay_data = self._get_overlay_data()

        # Sector rotation — after overlay so missing market.db ^VIX can use
        # vix_term_structure.vix_spot (same SSOT as regime/collar/hedge).
        try:
            sector_vix = self._resolve_hedge_vix_level(
                vix_level,
                overlay_data.get("vix_term_structure"),
            )
            sector_momentum_signal = self._generate_sector_momentum_signals(
                vix_level=sector_vix
            )
            # Disclose which VIX SSOT fed the high-vol gate when term structure
            # rescued a missing market.db row.
            if isinstance(sector_momentum_signal, dict):
                # Batch JG DS3: ensure preferred staleness field is present even
                # if an older producer only emitted timestamp.
                ts = sector_momentum_signal.get("timestamp") or sector_momentum_signal.get(
                    "generated_at"
                )
                if ts:
                    sector_momentum_signal.setdefault("generated_at", ts)
                    sector_momentum_signal.setdefault("timestamp", ts)
                if sector_vix is not None and vix_level is None:
                    sector_momentum_signal["vix"] = sector_vix
                    sector_momentum_signal["vix_source"] = "vix_term_structure"
                elif sector_momentum_signal.get("vix_source") is None:
                    sector_momentum_signal["vix_source"] = (
                        "market.db" if vix_level is not None else "unavailable"
                    )
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("sector_momentum", e)
            sector_momentum_signal = None

        # Hedge selector recommendation
        hedge_selector_signal = None
        try:
            hedge_selector_signal = self._get_hedge_selector_signal(
                vix_level,
                current_regime,
                overlay_data.get("vix_term_structure", {}),
            )
        except MONITOR_EXCEPTIONS as e:
            _log_signal_error("hedge_selector", e)

        # Operator regime card: never leave vix null when another surface has a level
        regime_data = self._enrich_regime_vix(
            regime_data,
            vix_term_structure=overlay_data.get("vix_term_structure"),
            behavioral_sentiment=behavioral_sentiment_data,
        )

        return {
            "regime": validate_signal("regime", regime_data),
            "target_allocations": target_alloc,
            "allocation_surface_roles": self._build_allocation_surface_roles(),
            "regime_authority": self._build_regime_authority(current_regime, target_alloc),
            "current_positions": positions,
            "cash": round(cash, 2),
            "total_value": round(total_value, 2),
            "latest_prices": latest,
            "recent_orders": list(reversed(orders)),
            "ml_signals": self._generate_ml_signals(),
            "marl_status": validate_signal("marl_status", self._generate_marl_status()),
            "factor_rotation": factor_rotation_signal,
            "yield_curve": validate_signal("yield_curve", yield_curve_data.get("yield_curve")),
            "duration_allocation": _enrich_duration_allocation_provenance(
                yield_curve_data.get("duration_allocation")
            ),
            "convexity_harvest": convexity_signal,
            "volatility_parity": vol_parity_signal,
            "llm_sentiment": sentiment_signal,
            "ensemble_voting": validate_signal("ensemble_voting", ensemble_signal),
            "sector_rotation": sector_momentum_signal,
            "alternative_data": alternative_data_signal,
            "behavioral_sentiment": behavioral_sentiment_data,
            "collar": overlay_data.get("collar", {}),
            "crypto_allocation": overlay_data.get("crypto", {}),
            "calendar_seasonality": overlay_data.get("calendar", {}),
            "kurtosis_regime": overlay_data.get("kurtosis", {}),
            "vix_term_structure": overlay_data.get("vix_term_structure", {}),
            "zero_dte": (
                overlay_data.get("zero_dte")
                if self._is_populated_overlay_section(overlay_data.get("zero_dte"))
                else self._unavailable_zero_dte_payload()
            ),
            "closing_auction": (
                overlay_data.get("closing_auction")
                if self._is_populated_overlay_section(overlay_data.get("closing_auction"))
                else self._unavailable_closing_auction_payload()
            ),
            "stacking_ensemble": stacking_ensemble_dashboard,
            "factor_rotation_dashboard": factor_rotation_dashboard,
            "smart_rebalance": validate_signal("smart_rebalance", smart_rebalance_data),
            "broker": broker_data,
            "garch_cvar": validate_signal("garch_cvar", garch_cvar_data),
            "entropy": entropy_data,
            "bond_momentum": overlay_data.get("bond_momentum", {}),
            "hedge_selector": validate_signal("hedge_selector", hedge_selector_signal),
            # Sidecar JSON is generated later in the same cycle; embed prior or
            # freshly computed snapshot so optional staleness sees a section.
            "risk_decomposition": self._load_risk_decomposition_signal_section(),
        }

    @staticmethod
    def _build_stacking_feature_count_metadata(integrator: Any) -> Dict[str, Any]:
        """Expose stacking feature count only when loaded model metadata backs it."""
        metadata = getattr(integrator, "metadata", None)
        feature_count = getattr(metadata, "feature_count", None)
        model_loaded = getattr(integrator, "model", None) is not None

        if model_loaded and feature_count is not None:
            return {
                "feature_count": int(feature_count),
                "feature_count_metadata_available": True,
                "feature_count_source": "model_metadata",
                "source_roster": list(getattr(metadata, "source_roster", [])),
                "source_roster_version": getattr(
                    metadata,
                    "source_roster_version",
                    "unavailable_missing_metadata",
                ),
                "fallback_semantics": getattr(
                    metadata,
                    "fallback_semantics",
                    "unavailable_missing_metadata",
                ),
            }

        return {
            "feature_count": None,
            "feature_count_metadata_available": False,
            "feature_count_source": (
                "unavailable_missing_metadata" if model_loaded else "unavailable_no_model"
            ),
            "source_roster": [],
            "source_roster_version": (
                "unavailable_missing_metadata" if model_loaded else "unavailable_no_model"
            ),
            "fallback_semantics": (
                "unavailable_missing_metadata" if model_loaded else "no_model_feature_count_unavailable"
            ),
        }

    @staticmethod
    def _build_stacking_no_model_dashboard(integrator: Any) -> Dict[str, Any]:
        """Build the explicit dormant stacking artifact when no model is loaded."""
        now_ts = datetime.now(timezone.utc).isoformat()
        return {
            "active": False,
            "stacking_available": False,
            "runtime_role": "research_dormant",
            "runtime_status": "unavailable_no_model",
            "live_authoritative": False,
            "routed": False,
            "routed_by": None,
            "prediction_available": False,
            "prediction_direction": "unavailable",
            "confidence": 0.0,
            "probability_bullish": 0.0,
            "probability_bearish": 0.0,
            "probability_neutral": 0.0,
            "fallback_used": False,
            "model_version": "unavailable_no_model",
            "voting_accuracy": None,
            "stacking_accuracy": None,
            "accuracy_metrics_available": False,
            **DashboardGenerator._build_stacking_feature_count_metadata(integrator),
            "latency_ms": 0.0,
            "status_reason": (
                "No stacking model artifact is loaded and no runtime base-signal "
                "input path is available."
            ),
            "operator_message": (
                "Stacking ensemble is research/dormant, not live-authoritative, "
                "and not order-routed."
            ),
            "timestamp": now_ts,
            "generated_at": now_ts,
        }

    @staticmethod
    def _build_stacking_model_dashboard(integrator: Any, prediction: Any) -> Dict[str, Any]:
        """Build the model-backed stacking dashboard artifact."""
        now_ts = datetime.now(timezone.utc).isoformat()
        return {
            "active": True,
            "stacking_available": True,
            "runtime_role": "model_backed_advisory",
            "runtime_status": "model_loaded",
            "live_authoritative": False,
            "routed": False,
            "routed_by": None,
            "prediction_available": prediction is not None,
            "prediction_direction": prediction.direction if prediction else "unavailable",
            "confidence": prediction.confidence if prediction else 0.0,
            "probability_bullish": prediction.probability_bullish if prediction else 0.0,
            "probability_bearish": prediction.probability_bearish if prediction else 0.0,
            "probability_neutral": prediction.probability_neutral if prediction else 0.0,
            "fallback_used": prediction.fallback_used if prediction else False,
            "model_version": prediction.model_version if prediction else "unknown",
            "voting_accuracy": 0.65,
            "stacking_accuracy": 0.76,
            "accuracy_metrics_available": True,
            **DashboardGenerator._build_stacking_feature_count_metadata(integrator),
            "latency_ms": prediction.latency_ms if prediction else 0.0,
            "status_reason": "Stacking model artifact is loaded for advisory inference.",
            "operator_message": (
                "Stacking ensemble is advisory and not order-routed; live routing "
                "still consumes target_allocations."
            ),
            "backtest_finding": (
                "+11% accuracy produces negligible Sharpe gain (2021-2026). "
                "Signal frequency and shift magnitude are binding constraints."
            ),
            "timestamp": now_ts,
            "generated_at": now_ts,
        }

    def _build_optional_signal_sections(
        self,
        output: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Append optional operational sections that precede staleness checks."""
        # Rebalance health data
        try:
            from src.monitor.rebalance_health import generate as gen_rebalance_health
            output["rebalance_health"] = gen_rebalance_health()
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("rebalance_health", e)
            output["rebalance_health"] = {"generated": None, "error": str(e)}

        # Circuit breaker state (broker API resilience)
        try:
            from src.broker.circuit_breaker import get_circuit_state
            output["broker_circuit_breaker"] = get_circuit_state()
        except ImportError:
            pass  # circuit_breaker module not available

        return output

    def _apply_signal_postprocessors(
        self,
        output: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply staleness, monitoring, alerting, and final signal appenders."""
        cursor = context["cursor"]
        current_regime = context["current_regime"]

        # FRED-MD macro regime signal
        try:
            from src.data import fred_data
            fred_signal = fred_data.get_fred_signal()
            readiness_getter = getattr(fred_data, "get_fred_md_cache_health", None)
            fred_readiness = readiness_getter() if callable(readiness_getter) else {}
            indicators = getattr(fred_signal, "indicators", {}) or {}
            indicators_observed = bool(
                getattr(fred_signal, "indicators_observed", bool(indicators))
            )
            source_mode = _first_known_value(
                getattr(fred_signal, "source_mode", None)
                if indicators_observed else None,
                fred_readiness.get("source_mode"),
                getattr(fred_signal, "source_mode", None),
                default="unknown",
            )
            cache_status = _first_known_value(
                fred_readiness.get("status")
                if fred_readiness else None,
                getattr(fred_signal, "cache_status", None),
                default="unknown",
            )
            status = (
                "ok"
                if _is_predictive_fred_macro({
                    "confidence": fred_signal.confidence,
                    "indicators": indicators,
                    "indicators_observed": indicators_observed,
                    "source_mode": source_mode,
                    "cache_status": cache_status,
                })
                else "unavailable"
            )
            output["fred_macro"] = validate_signal("fred_macro", {
                "regime": fred_signal.regime,
                "confidence": fred_signal.confidence,
                "recession_probability": fred_signal.recession_probability,
                "inflation_pressure": fred_signal.inflation_pressure,
                "monetary_stance": fred_signal.monetary_stance,
                "manufacturing_health": fred_signal.manufacturing_health,
                "credit_conditions": fred_signal.credit_conditions,
                "indicators": indicators,
                "timestamp": fred_signal.timestamp,
                "status": status,
                "source_mode": source_mode,
                "cache_status": cache_status,
                "api_key_configured": fred_readiness.get(
                    "api_key_configured",
                    getattr(fred_signal, "api_key_configured", False),
                ),
                "reason": getattr(fred_signal, "reason", None) or fred_readiness.get("reason"),
                "latest_fetched_at": fred_readiness.get(
                    "latest_fetched_at",
                    getattr(fred_signal, "latest_fetched_at", None),
                ),
                "row_count": fred_readiness.get("row_count"),
                "age_hours": fred_readiness.get("age_hours"),
                "ttl_hours": fred_readiness.get("ttl_hours"),
                "indicators_observed": indicators_observed,
            })
        except MONITOR_EXCEPTIONS as e:
            _log_signal_error("fred_macro", e)
            output["fred_macro"] = {
                "regime": "UNKNOWN",
                "confidence": 0.0,
                "status": "unavailable",
                "source_mode": "unavailable",
                "cache_status": "unavailable",
                "indicators": {},
                "indicators_observed": False,
                "error": str(e),
            }

        # Two-stage k-means macro regime classifier (Oliveira et al. 2025)
        try:
            two_stage_signal = self._generate_two_stage_regime()
            if two_stage_signal:
                output["two_stage_regime"] = validate_signal(
                    "two_stage_regime", two_stage_signal,
                )
            else:
                # Honesty: do not omit section when generator returns None
                # (missing FRED-MD / import / insufficient data).
                # Null metric slots — do not publish 0.0 confidence/crisis as live zeros.
                output["two_stage_regime"] = {
                    "regime": None,
                    "confidence": None,
                    "crisis_probability": None,
                    "probabilities": None,
                    "n_pca_components": None,
                    "variance_retained": None,
                    "n_observations": None,
                    "n_series": None,
                    "status": "unavailable",
                    "runtime_status": "unavailable",
                    "reason": "generator_returned_none",
                    "method": "oliveira_2025_two_stage_kmeans",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
        except MONITOR_EXCEPTIONS as e:
            _log_signal_error("two_stage_regime", e)
            output["two_stage_regime"] = {
                "regime": None,
                "confidence": None,
                "crisis_probability": None,
                "probabilities": None,
                "n_pca_components": None,
                "variance_retained": None,
                "n_observations": None,
                "n_series": None,
                "status": "unavailable",
                "runtime_status": "unavailable",
                "error": str(e),
                "method": "oliveira_2025_two_stage_kmeans",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Bayesian Online Changepoint Detection (BOCD) regime signal
        try:
            bocd_signal = self._generate_bocd_regime()
            if bocd_signal:
                output["bocd_regime"] = validate_signal(
                    "bocd_regime", bocd_signal,
                )
        except MONITOR_EXCEPTIONS as e:
            _log_signal_error("bocd_regime", e)
            output["bocd_regime"] = {
                "regime": 0,
                "regime_change_prob": 0.0,
                "error": str(e),
            }

        # Regime transition forecast (Oliveira et al. 2025 step 2)
        try:
            from src.regime.regime_history import load_daily_regime_history
            from src.regime.regime_transition_forecaster import RegimeTransitionForecaster

            forecaster = RegimeTransitionForecaster()
            daily_history = load_daily_regime_history(DATA_DIR / "regime_log.json")
            history = daily_history.labels
            history_metadata = daily_history.metadata
            # Forecast on one basis end-to-end: the daily VIX/controller series.
            current = history[-1] if history else str(current_regime).upper()
            if len(history) >= 2:
                forecaster.fit(history)
                forecast = forecaster.forecast(current, horizon_days=5)
                output["regime_transition"] = {
                    "current_regime": current,
                    "horizon_days": 5,
                    "forecast_probs": {k: round(v, 4) for k, v in forecast.probabilities.items()},
                    "most_likely": forecast.most_likely,
                    "persistence_params": {k: round(v, 1) for k, v in forecast.persistence_params.items()},
                    "status": "ok",
                    "runtime_status": "ok",
                    "role": "advisory_shadow",
                    "routed": False,
                    **history_metadata,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            else:
                output["regime_transition"] = {
                    "current_regime": current,
                    "horizon_days": 5,
                    "status": "unavailable",
                    "runtime_status": "unavailable",
                    "reason": "insufficient_regime_history",
                    "role": "advisory_shadow",
                    "routed": False,
                    **history_metadata,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
        except MONITOR_EXCEPTIONS as e:
            _log_signal_error("regime_transition", e)
            output["regime_transition"] = {
                "status": "unavailable",
                "runtime_status": "unavailable",
                "role": "advisory_shadow",
                "routed": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Signal staleness must be computed after optional regime sections are appended.
        output["staleness"] = self._check_signal_staleness(output)
        self._update_regime_authority_availability(output)

        # Apply staleness-weighted decay to ensemble weights
        output = self._apply_staleness_decay(output)

        # Health check report
        try:
            from src.monitor.health_check import run_health_check
            health_report = _load_canonical_health_report() or run_health_check()
            # Batch CT: canonical WWW health.json may embed pre-CQ/CR/CS
            # signal_health quality_disclosure (sticky freeze@46d age). Rebuild
            # SH section live so compact freeze/stale matches current thresholds
            # and ensemble_weights mtime — do not trust lagging published SH.
            if isinstance(health_report, dict):
                try:
                    health_report = dict(health_report)
                    health_report["signal_health"] = build_signal_health_section(
                        resolve_labels=False
                    )
                except Exception as sh_exc:  # noqa: BLE001
                    logger.warning(
                        "live signal_health rebuild for compact skipped: %s",
                        sh_exc,
                    )
            output["health"] = _compact_health_summary(health_report)
        except Exception as e:
            output["health"] = _compact_health_summary({"status": "error", "error": str(e)})

        # Batch DQ: project ensemble concentration SLI onto compact health so
        # partial writers (health kill refresh, alt patch) that advance
        # generated_at cannot hide a pre-cap sticky CAR>50% snapshot without
        # operators noticing via health.ensemble_concentration_ok.
        try:
            health = output.get("health")
            if not isinstance(health, dict):
                health = {}
                output["health"] = health
            ev = output.get("ensemble_voting")
            if isinstance(ev, dict):
                aw = ev.get("active_weights") or {}
                max_aw = float(ev.get("max_active_weight") or 0.0)
                if not max_aw and isinstance(aw, dict) and aw:
                    max_aw = float(max(aw.values()))
                cap = float(
                    ev.get("per_signal_active_weight_cap")
                    or DashboardGenerator.PER_SIGNAL_ACTIVE_WEIGHT_CAP
                )
                ok = bool(max_aw <= cap + 1e-6) if max_aw or aw else True
                if "ensemble_concentration_ok" in ev:
                    ok = bool(ev.get("ensemble_concentration_ok"))
                health["ensemble_max_active_weight"] = round(max_aw, 5)
                health["ensemble_per_signal_weight_cap"] = cap
                health["ensemble_concentration_ok"] = ok
                health["ensemble_n_eff"] = ev.get("n_eff")
                if not ok:
                    health["ensemble_concentration_status"] = "concentrated"
                    # Degrade compact health status when concentration breaches
                    if health.get("status") in (None, "ok", "healthy", "unknown"):
                        health["status"] = "warning"
                else:
                    health["ensemble_concentration_status"] = "ok"
                # Stale partial-patch forensic: if git status is partial_patch,
                # flag that ensemble may lag full generate (operators check sha).
                if output.get("generator_git_sha_status") == "partial_patch":
                    health["ensemble_may_lag_full_generate"] = True
                else:
                    health["ensemble_may_lag_full_generate"] = False
        except Exception as conc_exc:  # noqa: BLE001
            logger.warning("ensemble concentration health project skipped: %s", conc_exc)

        # Batch DV: project ML feature freshness onto compact health so operators
        # see advisory-stale features (features.jsonl ~75d) without opening the
        # ml_signals panel. Does not change routing authority (still advisory).
        try:
            health = output.get("health")
            if not isinstance(health, dict):
                health = {}
                output["health"] = health
            ml = output.get("ml_signals")
            if isinstance(ml, dict):
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
                er = ml.get("execution_role") if isinstance(ml.get("execution_role"), dict) else {}
                health["ml_live_authoritative"] = bool(er.get("live_authoritative"))
                # Soft warning when features are stale but still published as available
                if fresh == "stale" and bool(ml.get("available")):
                    health["ml_features_stale"] = True
                    if health.get("status") in (None, "ok", "healthy", "unknown"):
                        health["status"] = "warning"
                else:
                    health["ml_features_stale"] = False
        except Exception as ml_exc:  # noqa: BLE001
            logger.warning("ml feature freshness health project skipped: %s", ml_exc)

        # Batch DW: project smart-rebalance cost budget + dual-clock lag onto
        # compact health so 4× annual overrun / controller lag are not nested-only.
        try:
            health = output.get("health")
            if not isinstance(health, dict):
                health = {}
                output["health"] = health
            output["health"] = project_smart_rebalance_budget_onto_health(
                health,
                output.get("smart_rebalance")
                if isinstance(output.get("smart_rebalance"), dict)
                else None,
                output.get("rebalance_health")
                if isinstance(output.get("rebalance_health"), dict)
                else None,
            )
        except Exception as budget_exc:  # noqa: BLE001
            logger.warning(
                "smart rebalance budget health project skipped: %s", budget_exc
            )

        # Batch EG: unique event-day timeline vs raw snapshot-rewrite inflation
        try:
            health = output.get("health")
            if not isinstance(health, dict):
                health = {}
                output["health"] = health
            output["health"] = project_execution_timeline_onto_health(
                health,
                output.get("rebalance_health")
                if isinstance(output.get("rebalance_health"), dict)
                else None,
            )
        except Exception as tl_exc:  # noqa: BLE001
            logger.warning(
                "execution timeline health project skipped: %s", tl_exc
            )

        # Batch EJ: repo public/data mirror lag vs operator PUBLIC_DATA_DIR SoT
        try:
            health = output.get("health")
            if not isinstance(health, dict):
                health = {}
                output["health"] = health
            from src.monitor.repo_public_mirror_lag import (
                summarize_repo_public_mirror_lag,
            )

            lag_summary = summarize_repo_public_mirror_lag()
            output["health"] = project_repo_public_mirror_lag_onto_health(
                health, lag_summary
            )
        except Exception as mir_exc:  # noqa: BLE001
            logger.warning(
                "repo public mirror lag health project skipped: %s", mir_exc
            )

        # Batch EB: project five-surface paper return SSOT agreement onto compact
        # health so portfolio_history / snapshot drift cannot hide from ops.
        try:
            health = output.get("health")
            if not isinstance(health, dict):
                health = {}
                output["health"] = health
            from src.monitor.paper_return_ssot import compare_five_surfaces

            cmp = compare_five_surfaces(Path(DATA_DIR))
            output["health"] = project_paper_return_ssot_onto_health(health, cmp)
        except Exception as ssot_exc:  # noqa: BLE001
            logger.warning("paper return SSOT health project skipped: %s", ssot_exc)

        # Batch EC: voting-mass quality (soft-floor share) — source-count badges
        # alone miss 100% soft-floor vote mass when healthy sources are zero-baseline.
        try:
            health = output.get("health")
            if not isinstance(health, dict):
                health = {}
                output["health"] = health
            output["health"] = project_voting_mass_quality_onto_health(
                health,
                output.get("ensemble_voting")
                if isinstance(output.get("ensemble_voting"), dict)
                else None,
            )
        except Exception as vm_exc:  # noqa: BLE001
            logger.warning("voting mass quality health project skipped: %s", vm_exc)

        # Batch ED: multi-horizon reentry eligibility (disclose only, no force-wake)
        try:
            health = output.get("health")
            if not isinstance(health, dict):
                health = {}
                output["health"] = health
            output["health"] = project_reentry_eligibility_onto_health(
                health,
                output.get("ensemble_voting")
                if isinstance(output.get("ensemble_voting"), dict)
                else None,
            )
        except Exception as re_exc:  # noqa: BLE001
            logger.warning("reentry eligibility health project skipped: %s", re_exc)

        # Fire external alerts on staleness state transitions (+ recovery ownership)
        try:
            from src.monitor.alerting import check_staleness_and_alert
            from src.monitor.signal_ownership import (
                annotate_unavailable_signals,
                recovery_summary,
            )

            staleness = output.get("staleness")
            if isinstance(staleness, dict):
                ml_on = os.environ.get("PORTFOLIO_LAB_ENABLE_ML", "0") == "1"
                ownership = annotate_unavailable_signals(
                    staleness.get("unavailable_signals") or [],
                    ml_enabled=ml_on,
                )
                if ownership:
                    staleness = dict(staleness)
                    staleness["unavailable_ownership"] = ownership
                    staleness["recovery"] = recovery_summary(ownership)
                    output["staleness"] = staleness
            check_staleness_and_alert(output["staleness"])
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("alerting", e)

        # SPC signal quality monitoring
        output["spc"] = self._run_spc_monitor(output)

        # IC decay data recording — resolve staged predictions + stage new ones
        self._record_ic_data(output)

        # IC decay monitoring (signal predictive quality tracking)
        try:
            from src.monitor.ic_decay_monitor import compute_ic_decay_report
            output["ic_decay"] = compute_ic_decay_report()
        except MONITOR_EXCEPTIONS as e:
            _log_signal_error("ic_decay_monitor", e)
            output["ic_decay"] = {"error": str(e)}

        # IC decay alerting — fire alerts for signals with degrading IC
        try:
            from src.monitor.alerting import check_ic_decay_and_alert
            check_ic_decay_and_alert(output.get("ic_decay", {}))
        except MONITOR_EXCEPTIONS as e:
            _log_signal_error("ic_decay_alerting", e)

        # Per-signal walk-forward validation
        try:
            from src.monitor.signal_walk_forward import compute_signal_wfe_report
            output["signal_wfe"] = compute_signal_wfe_report()
        except MONITOR_EXCEPTIONS as e:
            _log_signal_error("signal_wfe", e)
            output["signal_wfe"] = {"error": str(e)}

        # Gold-TLT correlation regime monitor
        try:
            from src.research.gold_tlt_correlation import run_analysis
            analysis = run_analysis(window=252, save=False)
            output["gold_tlt_correlation"] = {
                "current_correlation": analysis.current_correlation,
                "current_regime": analysis.current_regime,
                "correlation_trend": analysis.correlation_trend,
                "mean_correlation": analysis.mean_correlation,
                "min_correlation": analysis.min_correlation,
                "max_correlation": analysis.max_correlation,
                "structural_breaks_count": len(analysis.structural_breaks),
                "regimes_count": len(analysis.regimes),
                "implications": analysis.implications,
            }
        except MONITOR_EXCEPTIONS as e:
            _log_signal_error("gold_tlt_correlation", e)
            output["gold_tlt_correlation"] = {"error": str(e)}

        # Paper→Live ramp status
        try:
            from src.broker.alpaca import LiveTransitionManager
            ramp_mgr = LiveTransitionManager()
            output["ramp"] = ramp_mgr.get_status()
        except MONITOR_EXCEPTIONS as e:
            _log_signal_error("ramp_status", e)
            output["ramp"] = {"error": str(e)}

        return output

    def generate_signals_json(self) -> Path:
        """Generate current signals and allocations."""
        context = self._load_signal_generation_context()
        output = _attach_signal_metadata(self._build_base_signal_sections(context))
        output = self._build_optional_signal_sections(output, context)
        output = self._apply_signal_postprocessors(output, context)
        output = _finalize_signal_metadata(output)

        try:
            from src.monitor.decision_registry import record_dashboard_cycle_decision

            record_dashboard_cycle_decision(output, context=context)
        except (ImportError, ValueError, OSError, TypeError) as e:
            logger.warning("Decision registry record skipped: %s", e)

        out_path = PUBLIC_DIR / "signals.json"
        private_path = Path(DATA_DIR) / "signals.json"

        # Validate once (same callback as legacy save_results_json).
        try:
            output = validate_all_signals(output)
        except Exception as e:  # noqa: BLE001 — match save_results_json soft-fail
            logger.warning("Validation callback failed: %s", e)

        # Batch HK: serialize-once multi-dest (public + private + repo soft-mirror)
        # with authority gate + 0o644 mode. Stops Dual-wrote dual open('w') path
        # that re-sticky 0600 and content-drifts under concurrent partials.
        try:
            from src.monitor.signal_authority import (
                AuthorityValidationError,
                default_repo_signals_path,
                write_signals_multi_dest,
            )

            # Explicit repo_path so soft-mirror is not skipped under pytest
            # (auto soft-mirror is gated off when PYTEST_CURRENT_TEST is set).
            result = write_signals_multi_dest(
                output,
                public_path=out_path,
                private_path=private_path,
                repo_path=default_repo_signals_path(),
                soft_mirror_repo=True,
            )
            if result.wrote_public:
                logger.info("Generated signals.json → %s", out_path)
            if result.wrote_private:
                logger.info("Multi-dest private signals.json → %s", private_path)
            if result.wrote_repo:
                logger.info("Multi-dest repo soft-mirror signals.json → %s", result.repo_path)
            if result.skipped_reason:
                logger.warning(
                    "signals full-generate multi-dest partial skip: %s",
                    result.skipped_reason,
                )
        except AuthorityValidationError as exc:
            logger.error(
                "Refusing full generate signals write (authority gate): %s",
                exc,
            )
        except (OSError, TypeError, ValueError, ImportError) as exc:
            logger.warning(
                "signals multi-dest full generate failed (%s); falling back to "
                "legacy public save_results_json only",
                exc,
            )
            save_results_json(output, output_path=str(out_path))

        return out_path

    def _load_broker_data(self) -> Dict:
        """Load broker position sync and order data for dashboard."""
        from src.dashboard.broker_data_loader import BrokerDataLoader

        return BrokerDataLoader(data_dir=DATA_DIR).load()

    def _load_garch_cvar_data(self) -> Dict:
        """Load GARCH-filtered CVaR metrics for dashboard (v3.21).

        Also computes a conformal CVaR cross-check (distribution-free)
        as a model-risk validation against the parametric GARCH estimate.
        """
        garch_cvar = {
            "cvar_95": -0.0179,
            "cvar_95_garch": -0.0215,
            "var_95": -0.0127,
            "var_95_garch": -0.0142,
            "cvar_ratio": 1.51,
            "garch_active": True,
            "current_volatility": 0.012,
            "forecast_volatility": 0.015,
            "volatility_clustering": "elevated",
            # Conformal cross-check defaults
            "conformal_cvar_95": None,
            "conformal_var_95": None,
            "conformal_cvar_ratio": None,
            "coverage_diagnostics": None,
        }

        # Compute conformal CVaR cross-check from SPY returns
        try:
            from src.monitor.conformal_risk import (
                conformal_coverage_diagnostics,
                conformal_cvar,
                conformal_var,
            )
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT close FROM prices WHERE symbol = 'SPY' ORDER BY date ASC"
            )
            rows = cursor.fetchall()
            if len(rows) >= 22:  # Need at least 22 days for meaningful split
                prices = np.array([r[0] for r in rows], dtype=float)
                returns = np.diff(np.log(prices))
                garch_cvar["conformal_cvar_95"] = round(
                    float(conformal_cvar(returns, alpha=0.05)), 6,
                )
                garch_cvar["conformal_var_95"] = round(
                    float(conformal_var(returns, alpha=0.05)), 6,
                )
                var_thresholds = np.full_like(
                    returns,
                    garch_cvar["conformal_var_95"],
                    dtype=float,
                )
                garch_cvar["coverage_diagnostics"] = conformal_coverage_diagnostics(
                    returns,
                    var_thresholds,
                    alpha=0.05,
                    rolling_window=252,
                )
                if garch_cvar["conformal_var_95"] != 0:
                    garch_cvar["conformal_cvar_ratio"] = round(
                        garch_cvar["conformal_cvar_95"]
                        / garch_cvar["conformal_var_95"], 3,
                    )
        except (ImportError, ValueError, TypeError, IndexError) as e:
            logger.info("Conformal CVaR cross-check unavailable: %s", e)

        try:
            # Load from GARCH-CVaR health report (flat format from compute_garch_risk.py)
            health_file = DATA_DIR / ".health_report.json"
            if health_file.exists():
                with open(health_file) as f:
                    data = json.load(f)

                # Support both flat GARCHCVaRMetrics format and nested checks format
                if data.get("garch_filtered") is not None:
                    # Flat format (from compute_garch_risk.py / evaluator._write_garch_health_report)
                    garch_cvar["cvar_95"] = data.get("cvar_95", garch_cvar["cvar_95"]) / 100.0 if abs(data.get("cvar_95", 0)) > 1 else data.get("cvar_95", garch_cvar["cvar_95"])
                    garch_cvar["var_95"] = data.get("var_95", garch_cvar["var_95"]) / 100.0 if abs(data.get("var_95", 0)) > 1 else data.get("var_95", garch_cvar["var_95"])
                    garch_cvar["cvar_ratio"] = data.get("cvar_ratio", garch_cvar["cvar_ratio"])
                    garch_cvar["garch_active"] = data.get("filter_active", False)
                    if data.get("conditional_volatility_current") is not None:
                        garch_cvar["current_volatility"] = data["conditional_volatility_current"] / 100.0
                    if data.get("garch_persistence") is not None:
                        garch_cvar["volatility_clustering"] = "high" if data["garch_persistence"] > 0.95 else "elevated" if data["garch_persistence"] > 0.85 else "normal"
                elif safe_get(data, "checks", "cvar_metrics", "garch_filtered"):
                    # Legacy nested format
                    cvar_check = data["checks"]["cvar_metrics"]
                    garch_cvar["cvar_95"] = cvar_check.get("cvar_95", -0.0179)
                    garch_cvar["var_95"] = cvar_check.get("var_95", -0.0127)
                    garch_cvar["cvar_ratio"] = cvar_check.get("cvar_ratio", 1.51)
                    garch_cvar["garch_active"] = cvar_check.get("garch_active", True)
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logger.warning("Using default values: %s", e)

        # Coverage fail → demote primary GARCH only on over-exceedance hard fail.
        # Under-exceedance is efficiency warning (over-conservative), not demotion.
        cov = garch_cvar.get("coverage_diagnostics")
        if isinstance(cov, dict):
            coverage_pass = cov.get("coverage_pass")
            direction = cov.get("coverage_direction") or cov.get("exceedance_bias")
            hard_fail = cov.get("coverage_hard_fail")
            if hard_fail is None:
                # Directed hard fail (over) OR legacy undirected coverage_pass=false
                # without direction metadata (treat as hard fail for safety).
                if direction in (None, "", "ok") and coverage_pass is False:
                    hard_fail = True
                    direction = direction or "over"  # assume over for legacy
                else:
                    hard_fail = bool(
                        coverage_pass is False and direction == "over"
                    )
            direction = direction or "ok"
            # Surface direction on risk metrics for operators
            garch_cvar["coverage_direction"] = direction
            garch_cvar["exceedance_bias"] = direction
            if hard_fail and garch_cvar.get("garch_active"):
                garch_cvar["garch_active"] = False
                garch_cvar["runtime_role"] = "advisory_degraded"
                garch_cvar["garch_active_reason"] = (
                    f"coverage_hard_fail (direction={direction}, "
                    f"coverage_pass={coverage_pass}); "
                    "over-exceedance — GARCH not primary risk authority"
                )
            elif cov.get("coverage_efficiency_warning") and garch_cvar.get(
                "garch_active"
            ):
                garch_cvar.setdefault("runtime_role", "primary")
                garch_cvar["garch_active_reason"] = (
                    f"coverage_efficiency_warning (direction={direction}); "
                    "under-exceedance — advisory capital inefficiency, "
                    "GARCH remains primary"
                )
            elif coverage_pass is True:
                garch_cvar.setdefault("runtime_role", "primary")
        return garch_cvar

    def _load_entropy_data(self) -> Dict:
        """Load entropy-based diversification metrics for dashboard (v3.22).

        Correlation-axis metrics are **not** hard-coded (prior 0.95 / 2.5 defaults
        looked like live diversification quality). Publish null + status until a
        real covariance path computes them.
        """
        entropy: Dict[str, Any] = {
            "shannon_entropy": None,
            "effective_n": None,
            "max_possible": None,
            "normalized_score": None,
            "concentration_risk": "unknown",
            "hhi_index": None,
            "correlation_entropy": None,
            "participation_ratio": None,
            "correlation_metrics_status": "unavailable",
            "status": "partial",
        }
        
        try:
            # Try to load from health report which now includes entropy metrics
            health_file = DATA_DIR / ".health_report.json"
            if health_file.exists():
                with open(health_file) as f:
                    health = json.load(f)
                    entropy_check = safe_get(health, "checks", "portfolio_entropy", default={})
                    metrics = entropy_check.get("metrics", {})
                    if metrics:
                        if metrics.get("shannon_entropy") is not None:
                            entropy["shannon_entropy"] = metrics.get("shannon_entropy")
                        if metrics.get("effective_n") is not None:
                            entropy["effective_n"] = metrics.get("effective_n")
                        if metrics.get("normalized_score") is not None:
                            entropy["normalized_score"] = metrics.get("normalized_score")
                        if metrics.get("hhi_index") is not None:
                            entropy["hhi_index"] = metrics.get("hhi_index")
                        if metrics.get("max_possible") is not None:
                            entropy["max_possible"] = metrics.get("max_possible")
                        # Derive H_max = ln(n) when shannon present but max missing
                        if (
                            entropy.get("max_possible") is None
                            and entropy.get("shannon_entropy") is not None
                        ):
                            try:
                                import math

                                from src.paths import BASE_ALLOCATION

                                n = len(BASE_ALLOCATION)
                                if n > 1:
                                    entropy["max_possible"] = round(math.log(n), 4)
                            except Exception:  # noqa: BLE001 — leave null
                                pass
                        # Only surface correlation metrics when actually computed
                        if metrics.get("correlation_entropy") is not None:
                            entropy["correlation_entropy"] = metrics.get("correlation_entropy")
                            entropy["correlation_metrics_status"] = "ok"
                        if metrics.get("participation_ratio") is not None:
                            entropy["participation_ratio"] = metrics.get("participation_ratio")
                            entropy["correlation_metrics_status"] = "ok"
                        
                        # Determine concentration risk from normalized score
                        score = entropy.get("normalized_score")
                        if isinstance(score, (int, float)):
                            if score > 90:
                                entropy["concentration_risk"] = "good"
                            elif score > 70:
                                entropy["concentration_risk"] = "low"
                            elif score > 50:
                                entropy["concentration_risk"] = "medium"
                            elif score > 30:
                                entropy["concentration_risk"] = "high"
                            else:
                                entropy["concentration_risk"] = "critical"
                            entropy["status"] = "ok"
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logger.warning("Entropy metrics unavailable: %s", e)

        return entropy

    def _generate_ml_signals(self) -> Dict:
        """Generate ML-based signals from features data."""
        def parse_timestamp(value: Any) -> Optional[datetime]:
            if not isinstance(value, str) or not value:
                return None
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
            except ValueError:
                return None

        def staleness_days(value: Any, now: datetime) -> Optional[int]:
            parsed = parse_timestamp(value)
            if parsed is None:
                return None
            return max(0, (now - parsed).days)

        def feature_freshness(value: Any, now: datetime) -> tuple[str, Optional[int]]:
            age_days = staleness_days(value, now)
            if age_days is None:
                return "unknown", None
            return ("stale" if age_days > 2 else "fresh"), age_days

        generated_at_dt = datetime.now(timezone.utc)
        generated_at = generated_at_dt.isoformat()
        signals = {
            "available": False,
            "timestamp": None,
            "generated_at": None,
            "feature_source_artifact": None,
            "feature_as_of": None,
            "feature_freshness_status": "unknown",
            "feature_staleness_days": None,
            "prediction_source_mode": "unavailable",
            "execution_role": {
                "role": "advisory_non_routed",
                "routed": False,
                "routed_by": None,
                "live_authoritative": False,
            },
            "predictions": {},
            "features": {},
            "grid_search": {},
        }
        
        # Check for features file
        features_file = DATA_DIR / "features.jsonl"
        if features_file.exists():
            try:
                # Get latest features for each symbol (tail read — last 500 lines)
                latest_features = {}
                with open(features_file, 'r') as f:
                    for line in deque(f, maxlen=500):
                        try:
                            feat = json.loads(line)
                            sym = feat.get("symbol")
                            ts = feat.get("timestamp", "")
                            if sym and (sym not in latest_features or ts > latest_features[sym].get("timestamp", "")):
                                latest_features[sym] = feat
                        except json.JSONDecodeError:
                            logger.exception("Failed to parse feature line in features.jsonl")
                            continue

                if latest_features:
                    feature_as_of = max(
                        (feat.get("timestamp") for feat in latest_features.values() if feat.get("timestamp")),
                        default=None,
                    )
                    freshness_status, age_days = feature_freshness(feature_as_of, generated_at_dt)
                    signals["available"] = True
                    signals["timestamp"] = generated_at
                    signals["generated_at"] = generated_at
                    signals["feature_source_artifact"] = "features.jsonl"
                    signals["feature_as_of"] = feature_as_of
                    signals["feature_freshness_status"] = freshness_status
                    signals["feature_staleness_days"] = age_days
                    signals["prediction_source_mode"] = (
                        "stale_features" if freshness_status == "stale" else "features"
                    )
                    signals["features"] = {
                        sym: {
                            "vix_level": feat.get("vix_level"),
                            "trend_direction": feat.get("trend_direction"),
                            "price_vs_sma20": feat.get("price_vs_sma20"),
                            "return_5d": feat.get("return_5d"),
                            "spy_correlation": feat.get("spy_correlation_20d"),
                            "feature_timestamp": feat.get("timestamp"),
                        }
                        for sym, feat in latest_features.items()
                    }
                    
                    # Generate simple heuristic predictions
                    for sym, feat in latest_features.items():
                        vix = feat.get("vix_level", 20)
                        trend = feat.get("trend_direction", 0)
                        price_vs_sma = feat.get("price_vs_sma20", 0)
                        
                        # Simple regime probability
                        if vix > 25:
                            p_bear, p_neutral, p_bull = 0.5, 0.3, 0.2
                        elif vix > 20:
                            p_bear, p_neutral, p_bull = 0.3, 0.5, 0.2
                        elif trend > 0 and price_vs_sma > 0:
                            p_bear, p_neutral, p_bull = 0.1, 0.3, 0.6
                        elif trend < 0:
                            p_bear, p_neutral, p_bull = 0.4, 0.4, 0.2
                        else:
                            p_bear, p_neutral, p_bull = 0.2, 0.6, 0.2
                        
                        # Map to regime names
                        probs = {"bear": p_bear, "neutral": p_neutral, "bull": p_bull}
                        predicted = max(probs, key=probs.get)
                        confidence = probs[predicted]
                        
                        signals["predictions"][sym] = {
                            "predicted_regime": predicted,
                            "confidence": round(confidence, 3),
                            "probabilities": {k: round(v, 3) for k, v in probs.items()},
                            "heuristic": True,  # Not ML-based yet
                            "feature_timestamp": feat.get("timestamp"),
                            "feature_freshness_status": freshness_status,
                            "source_artifact": "features.jsonl",
                        }
            except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError, ImportError, RuntimeError) as e:
                signals["error"] = str(e)

        # Check for grid search results (tail read only)
        grid_file = DATA_DIR / "grid_search_results.jsonl"
        if grid_file.exists():
            try:
                with open(grid_file, 'r') as f:
                    tail = deque(f, maxlen=1)
                if tail:
                    latest = json.loads(tail[0])
                    benchmark_timestamp = latest.get("timestamp")
                    signals["grid_search"] = {
                        "available": True,
                        "timestamp": benchmark_timestamp,
                        "top_allocation": latest.get("allocations"),
                        "sharpe": latest.get("sharpe"),
                        "volatility": latest.get("volatility"),
                        "source_artifact": "grid_search_results.jsonl",
                        "benchmark_timestamp": benchmark_timestamp,
                        "observation_semantics": "frozen_benchmark_not_live_snapshot",
                        "freshness_status": "frozen_benchmark",
                        "staleness_days": staleness_days(benchmark_timestamp, generated_at_dt),
                        "live_authoritative": False,
                    }
            except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logger.warning("Failed to load grid search results: %s", e)

        return signals

    @staticmethod
    def _marl_execution_role() -> Dict[str, Any]:
        """Describe MARL's current operator-visible but non-routed role."""
        return {
            "role": "research_shadow_non_routed",
            "routed": False,
            "routed_by": None,
            "live_authoritative": False,
            "description": (
                "MARL status is visible for research/shadow diagnostics; "
                "order routing still consumes target_allocations."
            ),
        }

    @staticmethod
    def _default_marl_runtime_status() -> Dict[str, Any]:
        """Return a stable runtime shape for unavailable MARL status."""
        return {
            "version": "unknown",
            "device": "unknown",
            "agents_loaded": [],
            "signal_integrator_connected": False,
            "checkpoint_loaded": False,
            "inference_count": 0,
            "current_allocation": {},
            "graph_metrics": {},
        }

    @staticmethod
    def _generate_marl_status() -> Dict[str, Any]:
        """Expose AIController runtime status without implying live routing authority.

        ``available`` means checkpoint-backed runtime readiness for shadow use,
        not merely that the MARL module imported. Controllers without a loaded
        checkpoint report available=false with reason, plus module_importable.
        """
        status = {
            "schema_version": "marl-runtime-status/v1",
            "available": False,
            "module_importable": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "runtime": DashboardGenerator._default_marl_runtime_status(),
            "execution_role": DashboardGenerator._marl_execution_role(),
        }

        try:
            from src.agents.ai_controller import AIController

            controller = AIController(use_signal_integrator=False)
            runtime_status = controller.get_status()
            if isinstance(runtime_status, dict):
                status["module_importable"] = True
                status["runtime"] = {
                    **DashboardGenerator._default_marl_runtime_status(),
                    **runtime_status,
                }
                checkpoint_loaded = bool(runtime_status.get("checkpoint_loaded"))
                # available = ready for shadow inference path (checkpoint present)
                status["available"] = checkpoint_loaded
                if not checkpoint_loaded:
                    status["reason"] = "checkpoint_not_loaded"
                else:
                    status["reason"] = "checkpoint_loaded"
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("marl_status", e)
            status["error"] = str(e)
            status["reason"] = "controller_import_or_init_failed"

        return status
    
    def _get_yield_curve_data(self) -> Dict:
        """Get yield curve data from yields.json and calculate duration allocation."""
        result = {
            "yield_curve": None,
            "duration_allocation": None
        }
        
        yields_file = YIELDS_JSON
        if not yields_file.exists():
            return result
        
        try:
            with open(yields_file, 'r') as f:
                yields = json.load(f)
            
            if not yields or len(yields) == 0:
                return result
            
            # Get latest yield entry
            latest = yields[-1]
            
            # Calculate regime based on 2s10s spread
            spread = latest.get("spread2s10s", 0)
            if spread > 100:
                regime = "steep"
            elif spread > 50:
                regime = "normal"
            elif spread > 0:
                regime = "flat"
            else:
                regime = "inverted"
            
            # Get last 30 days of spread history for sparkline
            spread_history = []
            for entry in yields[-30:]:
                if entry.get("spread2s10s") is not None:
                    spread_history.append(entry["spread2s10s"])
            
            asof = latest.get("date")
            # Wall-clock weekday lag vs today (UTC) for freeze detection
            lag_weekdays = 0
            status = "ok"
            reason = None
            if isinstance(asof, str) and len(asof) >= 10:
                try:
                    from datetime import date as _date
                    asof_d = _date.fromisoformat(asof[:10])
                    today = datetime.now(timezone.utc).date()
                    # Count Mon-Fri strictly after asof through today
                    cur = asof_d
                    from datetime import timedelta as _td
                    cur = cur + _td(days=1)
                    while cur <= today:
                        if cur.weekday() < 5:
                            lag_weekdays += 1
                        cur = cur + _td(days=1)
                except ValueError:
                    lag_weekdays = 0
            max_lag = int(os.environ.get("YIELD_CURVE_MAX_STALE_WEEKDAYS", "5"))
            if lag_weekdays > max_lag:
                status = "stale"
                reason = f"asof_lag_weekdays_{lag_weekdays}_gt_{max_lag}"

            result["yield_curve"] = {
                "spread2s10s": spread,
                "dgs2": latest.get("dgs2"),
                "dgs10": latest.get("dgs10"),
                "duration_regime": regime,
                "spread_history": spread_history,
                "asof": asof,
                "asof_lag_weekdays": lag_weekdays,
                "status": status,
                **({"reason": reason} if reason else {}),
                **({
                    "runtime_status": "stale",
                } if status == "stale" else {}),
                **{
                    key: value
                    for key, value in _yield_source_provenance(PUBLIC_DIR).items()
                    if value is not None
                },
            }
            
            # Calculate duration allocation based on regime (advisory sleeve —
            # never bare weights without provenance; not live order-routing authority)
            regime_allocations = {
                "steep": {"tlt": 0.70, "ief": 0.25, "shy": 0.05, "bil": 0.00},
                "normal": {"tlt": 0.50, "ief": 0.35, "shy": 0.15, "bil": 0.00},
                "flat": {"tlt": 0.30, "ief": 0.40, "shy": 0.25, "bil": 0.05},
                "inverted": {"tlt": 0.15, "ief": 0.25, "shy": 0.35, "bil": 0.25}
            }
            weights = regime_allocations.get(regime, regime_allocations["normal"])
            result["duration_allocation"] = _enrich_duration_allocation_provenance(
                {
                    "weights": weights,
                    # Flat keys retained for backward-compat consumers
                    **weights,
                    "duration_regime": regime,
                    "source": "yield_curve_regime_table",
                }
            )

        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logger.warning("Failed to load yield curve data: %s", e)
        
        return result
    
    def generate_stats_json(self) -> Path:
        """Generate performance statistics."""
        cursor = self.conn.cursor()

        # Single batched query for all symbols instead of N+1 per-symbol queries
        # Champion book (live authority / paper) vs context benchmarks
        champion_symbols = ['SPY', 'GLD', 'TLT']
        context_symbols = ['QQQ', 'VIX']
        symbols = champion_symbols + context_symbols
        placeholders = ','.join('?' for _ in symbols)
        cursor.execute(f"""
            SELECT symbol, close FROM prices
            WHERE symbol IN ({placeholders}) AND date >= date('now', '-30 days')
            ORDER BY symbol, date
        """, symbols)

        # Group rows by symbol
        symbol_prices: Dict[str, List[float]] = {}
        for sym, close in cursor.fetchall():
            symbol_prices.setdefault(sym, []).append(close)

        stats = {}
        held_stats: Dict[str, Any] = {}
        context_stats: Dict[str, Any] = {}
        for symbol in symbols:
            prices = symbol_prices.get(symbol, [])
            if len(prices) >= 2:
                returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                in_portfolio = symbol in champion_symbols
                entry = {
                    "30d_return": round((prices[-1] - prices[0]) / prices[0] * 100, 2),
                    "volatility": round(np.std(returns) * np.sqrt(252) * 100, 2) if returns else 0,
                    "current": prices[-1],
                    "in_portfolio": in_portfolio,
                    "not_in_portfolio": not in_portfolio,
                    "role": "held" if in_portfolio else "benchmark_or_context",
                }
                stats[symbol] = entry
                if in_portfolio:
                    held_stats[symbol] = entry
                else:
                    context_stats[symbol] = entry
        
        # Paper portfolio metrics with SPY comparison.
        # Prefer daily_pnl.jsonl session series (SSOT) over performance.jsonl
        # which historically interleaves evaluator micro-noise.
        paper_metrics = {}
        spy_comparison = None
        daily_returns: list[float] = []
        daily_values: list[float] = []
        return_source = "none"

        def _material_return(val: Any) -> bool:
            """Include real session returns; drop evaluator micro-noise (~1e-8).

            Exact 0.0 is a valid flat session (Sharpe 0). Tiny non-zero noise
            from intraday evaluator rows must not enter the metric series.
            """
            try:
                v = float(val)
            except (TypeError, ValueError):
                return False
            if v == 0.0:
                return True
            return abs(v) >= 1e-6

        pnl_log = DATA_DIR / "daily_pnl.jsonl"
        if pnl_log.exists():
            try:
                by_date: dict[str, dict] = {}
                with open(pnl_log) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(row, dict):
                            continue
                        d = str(row.get("date") or "")[:10]
                        if not d:
                            continue
                        by_date[d] = row
                ordered = [by_date[d] for d in sorted(by_date.keys())]
                # Keep only material session returns for Sharpe input
                for e in ordered:
                    if not _material_return(e.get("daily_return")):
                        continue
                    daily_returns.append(float(e["daily_return"]))
                    try:
                        daily_values.append(float(e.get("total_value") or 0))
                    except (TypeError, ValueError):
                        daily_values.append(0.0)
                if len(daily_returns) >= 3:
                    return_source = "daily_pnl.jsonl_session"
            except OSError:
                daily_returns, daily_values = [], []

        if return_source == "none":
            perf_log = DATA_DIR / "performance.jsonl"
            if perf_log.exists():
                with open(perf_log) as f:
                    tail_lines = deque(f, maxlen=500)
                    # Raw tail can be short after dedup; require only enough
                    # lines to form a 3-day session series after filters.
                    if len(tail_lines) >= 3:
                        raw_entries = []
                        for l in tail_lines:
                            try:
                                raw_entries.append(json.loads(l))
                            except json.JSONDecodeError:
                                continue
                        daily_entries = self._deduplicate_performance_entries_by_date(
                            raw_entries
                        )
                        for e in daily_entries:
                            dr = e.get("daily_return")
                            if dr is None or not _material_return(dr):
                                continue
                            daily_returns.append(float(dr))
                            try:
                                daily_values.append(float(e.get("total_value") or 0))
                            except (TypeError, ValueError):
                                daily_values.append(0.0)
                        if daily_returns:
                            return_source = "performance.jsonl_daily_dedup"

        if daily_returns and len(daily_returns) >= 3:
                        std_r = float(np.std(daily_returns))
                        raw_sharpe = (
                            float(np.mean(daily_returns) / std_r * np.sqrt(252))
                            if std_r > 0
                            else 0.0
                        )
                        # Honesty: same implausibility gate as graduation (raw > 3.0).
                        # Keep raw value; never coerce to 0.0 (looked like zero skill).
                        implausible = bool(raw_sharpe > 3.0)
                        n_pts = len(daily_returns)
                        paper_metrics = {
                            "sharpe": round(raw_sharpe, 2),
                            "sharpe_raw": round(raw_sharpe, 4),
                            "sharpe_implausible": implausible,
                            "sharpe_plausibility_status": (
                                "implausible_short_sample"
                                if implausible
                                else "ok"
                            ),
                            "sharpe_note": (
                                (
                                    f"implausible raw Sharpe {raw_sharpe:.2f} > 3.0 "
                                    f"over {n_pts} daily points "
                                    "(likely short-sample / low-vol artifact; "
                                    "not graduation-ready skill)"
                                )
                                if implausible
                                else None
                            ),
                            "total_return": (
                                round(
                                    (daily_values[-1] - daily_values[0])
                                    / daily_values[0]
                                    * 100,
                                    2,
                                )
                                if daily_values
                                and daily_values[0]
                                and daily_values[-1]
                                else None
                            ),
                            "max_value": round(max(daily_values), 2) if daily_values else None,
                            "min_value": round(min(daily_values), 2) if daily_values else None,
                            "days_tracked": n_pts,
                            "return_source": return_source,
                        }
                        
                        # Calculate SPY comparison if we have enough data
                        cursor.execute("""
                            SELECT date, close FROM prices 
                            WHERE symbol = 'SPY' 
                            AND date >= date('now', '-63 days')
                            ORDER BY date
                        """)
                        spy_rows = cursor.fetchall()
                        if len(spy_rows) >= 20 and len(daily_values) >= 20:
                            spy_prices = [r[1] for r in spy_rows[-len(daily_values):]]
                            spy_returns = [(spy_prices[i] - spy_prices[i-1]) / spy_prices[i-1]
                                          for i in range(1, len(spy_prices))]

                            # Calculate metrics
                            spy_total_return = (spy_prices[-1] - spy_prices[0]) / spy_prices[0]
                            portfolio_total_return = (daily_values[-1] - daily_values[0]) / daily_values[0]

                            # Correlation and Beta (30-day rolling)
                            min_len = min(len(daily_returns), len(spy_returns))
                            if min_len >= 20:
                                returns_arr = np.array(daily_returns[-20:])
                                spy_returns_arr = np.array(spy_returns[-20:])
                                
                                # Check for variance before calculating correlation
                                if np.std(returns_arr) > 0 and np.std(spy_returns_arr) > 0:
                                    corr = np.corrcoef(returns_arr, spy_returns_arr)[0,1]
                                    spy_vol = np.std(spy_returns_arr)
                                    if spy_vol > 0:
                                        beta = np.cov(returns_arr, spy_returns_arr)[0,1] / (spy_vol ** 2)
                                    else:
                                        beta = 1.0
                                else:
                                    corr = 0
                                    beta = 1.0
                            else:
                                corr = 0
                                beta = 1.0
                            
                            spy_comparison = {
                                "portfolio_value": round(daily_values[-1], 2),
                                "spy_value": round(daily_values[0] * (1 + spy_total_return), 2),
                                "relative_return": round((portfolio_total_return - spy_total_return) * 100, 2),
                                "correlation_30d": round(float(corr), 2),
                                "beta": round(float(beta), 2),
                                "outperformance": round((portfolio_total_return - spy_total_return) * 100, 2)
                            }
        
        output = _stamp_generator_git_sha({
            "asset_stats": stats,
            "held_asset_stats": held_stats,
            "context_asset_stats": context_stats,
            "champion_symbols": list(champion_symbols),
            "paper_portfolio": paper_metrics,
            "spy_comparison": spy_comparison,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })
        
        out_path = PUBLIC_DIR / "stats.json"
        save_results_json(output, output_path=str(out_path))
        
        return out_path
    
    @staticmethod
    def _load_json_file(path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None

    @staticmethod
    def _has_open_blocking_incident(data_dir: Path) -> bool:
        for filename in ("incidents.json", "incident_state.json"):
            payload = DashboardGenerator._load_json_file(data_dir / filename)
            if not payload:
                continue
            raw_incidents = payload.get("incidents", payload.get("open_incidents", []))
            incidents = raw_incidents if isinstance(raw_incidents, list) else []
            for incident in incidents:
                if not isinstance(incident, dict):
                    continue
                status = str(incident.get("status", "open")).lower()
                blocking = bool(incident.get("blocking") or incident.get("blocks_promotion"))
                if blocking and status not in {"closed", "resolved", "pass"}:
                    return True
        return False

    @staticmethod
    def _promotion_gate_status(data_dir: Path) -> tuple[bool, list[str]]:
        kill_switch = DashboardGenerator._load_json_file(data_dir / "kill_switch.json")
        blockers: list[str] = []
        if kill_switch and kill_switch.get("enabled"):
            blockers.append("kill_switch")
        if DashboardGenerator._has_open_blocking_incident(data_dir):
            blockers.append("blocking_incident")

        try:
            from src.strategy.graduation_checklist import GraduationChecklist

            checklist = GraduationChecklist()
            results = checklist.check()
            manual = results.get("manual_approval")
            if manual is None or not manual.passed:
                blockers.append("manual_approval")
            if not checklist.is_graduation_ready(results):
                blockers.append("graduation_checklist")
        except SIGNAL_EXCEPTIONS:
            blockers.append("graduation_checklist_unavailable")

        return not blockers, blockers

    @staticmethod
    def _is_active_promote_candidacy(data: Dict[str, Any]) -> bool:
        """True only for live promote candidacy, not tombstones.

        GraduationChecklist rewrites ``.promote_to_live`` with
        ``action: promote_blocked_*`` when kill or checklist blocks.
        Those are not candidates — alerts must ignore them.
        """
        action = data.get("action")
        if action is None:
            # Legacy markers omit action; treat as candidacy.
            return True
        if not isinstance(action, str):
            return False
        if action == "promote_to_live":
            return True
        if action.startswith("promote_blocked"):
            return False
        # Unknown action strings are not live candidacy claims.
        return False

    @staticmethod
    def _graduation_candidate_alert(data_dir: Path) -> Optional[Dict[str, Any]]:
        data = DashboardGenerator._load_json_file(data_dir / ".promote_to_live")
        if not data:
            return None
        if not DashboardGenerator._is_active_promote_candidacy(data):
            return None

        allowed, blockers = DashboardGenerator._promotion_gate_status(data_dir)
        if allowed:
            return {
                "level": "success",
                "type": "graduation_candidate",
                "title": "Paper Trading Graduation Ready",
                "message": f"Sharpe: {safe_get(data, 'metrics', 'sharpe')}, ready for live approval",
                "timestamp": data.get("timestamp"),
                "requires_action": True,
            }
        return {
            "level": "warning",
            "type": "graduation_candidate",
            "title": "Paper Trading Graduation Blocked",
            "message": "Promotion marker present but current gates block live approval: "
            + ", ".join(sorted(set(blockers))),
            "timestamp": data.get("timestamp"),
            "requires_action": True,
        }

    @staticmethod
    def _stale_data_alerts_from_quality_report(public_dir: Path) -> Optional[List[Dict[str, Any]]]:
        report = DashboardGenerator._load_json_file(public_dir / "data_quality.json")
        if report is None:
            return None

        issue_counts = report.get("issue_counts")
        stale_count = (
            issue_counts.get("stale_latest_dates", 0)
            if isinstance(issue_counts, dict)
            else 0
        )
        if not isinstance(stale_count, int) or isinstance(stale_count, bool) or stale_count <= 0:
            return []

        alerts: List[Dict[str, Any]] = []
        symbols = report.get("symbols")
        rows = symbols if isinstance(symbols, list) else []
        stale_rows = [
            row for row in rows
            if isinstance(row, dict)
            and (
                row.get("stale_latest_date")
                or str(row.get("status", "")).lower() in {"fail", "failed", "critical"}
                or (isinstance(row.get("issue_counts"), dict)
                    and row["issue_counts"].get("stale_latest_dates", 0) > 0)
            )
        ]

        for row in stale_rows[:stale_count]:
            stale_meta = row.get("stale_latest_date") if isinstance(row.get("stale_latest_date"), dict) else {}
            symbol = row.get("symbol", "unknown")
            latest_date = stale_meta.get("latest_date") or row.get("latest_date", "unknown")
            reference_date = stale_meta.get("reference_date") or report.get("reference_date", "unknown")
            lag_days = stale_meta.get("latest_lag_days")
            lag = f" ({lag_days} trading day lag)" if isinstance(lag_days, int) else ""
            alerts.append({
                "level": "warning",
                "type": "stale_data",
                "title": f"Stale Data: {symbol}",
                "message": f"{symbol} latest date {latest_date} lags reference {reference_date}{lag}",
                "timestamp": report.get("generated_at"),
                "requires_action": False,
            })

        while len(alerts) < stale_count:
            alerts.append({
                "level": "warning",
                "type": "stale_data",
                "title": "Stale Data",
                "message": f"data_quality.json reports {stale_count} stale latest-date issue(s)",
                "timestamp": report.get("generated_at"),
                "requires_action": False,
            })
        return alerts

    def generate_alerts_json(self, health: Optional[Dict[str, Any]] = None) -> Path:
        """Generate active alerts and notifications.

        Args:
            health: Optional health.json payload. When omitted, loads
                ``PUBLIC_DIR/health.json`` if present. Prefer passing the
                in-process payload from ``generate_health_json`` / ``run`` so
                projection does not depend on filesystem ordering.
        """
        alerts = []

        promote_alert = self._graduation_candidate_alert(DATA_DIR)
        if promote_alert is not None:
            alerts.append(promote_alert)

        # Check for kill switch (SSOT: data/kill_switch.json)
        kill_payload = load_kill_switch_payload(DATA_DIR)
        kill_alert = build_kill_switch_alert(kill_payload) if kill_payload else None
        if kill_alert is not None:
            alerts.append(kill_alert)
        
        # Check for regime trigger
        regime_file = DATA_DIR / ".regime_trigger"
        if regime_file.exists():
            with open(regime_file) as f:
                data = json.load(f)
                alerts.append({
                    "level": "warning",
                    "type": "regime_change",
                    "title": f"Regime Change: {data.get('regime', 'unknown')}",
                    "message": f"VIX: {data.get('vix', 'N/A')}",
                    "timestamp": data.get("timestamp"),
                    "requires_action": False
                })
        
        quality_alerts = self._stale_data_alerts_from_quality_report(PUBLIC_DIR)
        if quality_alerts is not None:
            alerts.extend(quality_alerts)
        else:
            # Fallback for test/local environments that do not publish data_quality.json.
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT symbol, MAX(date) as last_date, COUNT(*) as count
                FROM prices GROUP BY symbol
            """)
            for row in cursor.fetchall():
                last_date = datetime.strptime(row[1], "%Y-%m-%d") if row[1] else None
                if last_date and (datetime.now() - last_date).days > 2:
                    alerts.append({
                        "level": "warning",
                        "type": "stale_data",
                        "title": f"Stale Data: {row[0]}",
                        "message": f"Last update: {row[1]} ({(datetime.now() - last_date).days} days ago)",
                        "requires_action": False
                    })

        health_payload = health if isinstance(health, dict) else self._load_json_file(
            PUBLIC_DIR / "health.json"
        )
        alerts.extend(build_health_slo_alerts(health_payload))
        
        output = _stamp_generator_git_sha({
            "alerts": sorted(alerts, key=lambda x: x.get("timestamp", "") or "", reverse=True),
            "count": len(alerts),
            "generated_at": datetime.now(timezone.utc).isoformat()
        })
        
        out_path = PUBLIC_DIR / "alerts.json"
        private_alerts = Path(DATA_DIR) / "alerts.json"

        def _write_alerts_multi(payload: dict) -> None:
            """Serialize-once public + private + repo soft-mirror (Batch HN).

            repo_path left None so auto soft-mirror uses repo_filename and skips
            under pytest (avoids clobbering checkout public/data during tests).
            """
            try:
                from src.monitor.signal_authority import write_json_multi_dest

                write_json_multi_dest(
                    payload,
                    public_path=out_path,
                    private_path=private_alerts
                    if private_alerts.parent.is_dir() or private_alerts.exists()
                    else None,
                    soft_mirror_repo=True,
                    repo_filename="alerts.json",
                )
            except Exception as exc:  # noqa: BLE001 — fall back to legacy single dest
                logger.warning(
                    "alerts multi-dest write failed (%s); falling back to save_results_json",
                    exc,
                )
                save_results_json(payload, output_path=str(out_path))

        _write_alerts_multi(output)

        # Post-write integrity: on-disk kill row must match data/kill_switch.json identity.
        # Concurrent/stale writers previously left LIVE/position_limit rows without incident_id.
        if kill_payload and kill_payload.get("enabled") and kill_alert is not None:
            try:
                on_disk = json.loads(out_path.read_text(encoding="utf-8"))
                disk_alerts = on_disk.get("alerts") if isinstance(on_disk, dict) else None
                disk_kills = [
                    a for a in (disk_alerts or [])
                    if isinstance(a, dict) and a.get("type") == "kill_switch"
                ]
                expected_id = kill_payload.get("incident_id")
                expected_reason = kill_payload.get("reason")
                expected_level = kill_payload.get("level")
                ok = False
                if disk_kills:
                    row = disk_kills[0]
                    ok = (
                        (expected_id is None or row.get("incident_id") == expected_id)
                        and (expected_reason is None or row.get("reason") == expected_reason)
                        and (
                            expected_level is None
                            or str(row.get("kill_switch_level") or "").lower()
                            == str(expected_level).lower()
                        )
                    )
                if not ok:
                    logger.error(
                        "alerts.json kill identity drift after write; rewriting from SSOT "
                        "(expected incident_id=%r reason=%r level=%r)",
                        expected_id,
                        expected_reason,
                        expected_level,
                    )
                    # Force SSOT kill row as sole kill_switch entry and rewrite.
                    rebuilt = [
                        a for a in output["alerts"]
                        if not (isinstance(a, dict) and a.get("type") == "kill_switch")
                    ]
                    rebuilt.insert(0, kill_alert)
                    output = _stamp_generator_git_sha({
                        "alerts": rebuilt,
                        "count": len(rebuilt),
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    })
                    _write_alerts_multi(output)
            except (OSError, json.JSONDecodeError, TypeError) as verify_exc:
                logger.error("alerts.json post-write kill verify failed: %s", verify_exc)
        
        return out_path

    @staticmethod
    def _empty_incident_summary() -> Dict[str, Any]:
        return {
            "schema_version": "incident-lifecycle/v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "open_count": 0,
            "incidents": [],
            "metrics": {
                "incident_frequency": 0,
                "open_count": 0,
                "resolved_count": 0,
                "mean_mttr_seconds": None,
            },
        }

    def generate_incidents_json(self) -> Path:
        """Publish the incident lifecycle summary consumed by LiveDashboard."""
        try:
            payload = self._load_json_file(DATA_DIR / "incidents.json") or self._empty_incident_summary()
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("Incident lifecycle summary unavailable; publishing empty summary: %s", exc)
            payload = self._empty_incident_summary()

        payload.setdefault("schema_version", "incident-lifecycle/v1")
        payload.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
        payload.setdefault("open_count", 0)
        payload.setdefault("incidents", [])
        payload.setdefault("metrics", {
            "incident_frequency": 0,
            "open_count": int(payload.get("open_count", 0) or 0),
            "resolved_count": 0,
            "mean_mttr_seconds": None,
        })
        payload = _stamp_generator_git_sha(payload)

        out_path = PUBLIC_DIR / "incidents.json"
        save_results_json(payload, output_path=str(out_path))
        return out_path
    
    def generate_health_json(self) -> Path:
        """Generate system health status for dashboard."""
        
        health_data = {
            "cron_jobs": [],
            "data_freshness": {},
            "system_status": "healthy",
            "signal_health": {},
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Cron job status from project-local status file and Hermes, when available.
        cron_section = build_cron_scheduler_section(
            cron_status_file=DATA_DIR / "cron_status.json",
            resolve_hermes_path=_resolve_hermes_cron_jobs_path,
            log_error=_log_signal_error,
        )
        health_data["cron_jobs"] = cron_section["cron_jobs"]
        health_data["scheduler_status"] = cron_section["scheduler_status"]
        
        # Per-symbol data freshness from SQLite
        freshness_section = build_data_freshness_section(conn=self.conn)
        health_data["data_freshness"] = freshness_section["data_freshness"]

        health_data["signal_health"] = build_signal_health_section(log_error=_log_signal_error)
        health_data["fred_readiness"] = build_fred_readiness_section(log_error=_log_signal_error)

        health_data["data_pipeline_slo"] = build_data_pipeline_slo_section(
            health_data=health_data,
            public_dir=PUBLIC_DIR,
            data_dir=DATA_DIR,
            log_error=_log_signal_error,
        )

        # Kill authority + open incidents (same SSOT as data/health monitor)
        kill_fields = project_kill_switch_fields(load_kill_switch_payload(DATA_DIR))
        open_incidents = load_open_incidents_summary(DATA_DIR)
        health_data["kill_switch"] = kill_fields
        health_data["open_incidents"] = open_incidents

        # Overall system health
        # Exclude portfolio-lab-health self-errors so sticky tasker mirrors of
        # prior health exits cannot degrade dashboard system_status forever.
        from src.monitor.hermes_cron import rollup_failed_cron_jobs

        stale_count = summarize_stale_symbol_count(health_data["data_freshness"])
        failed_jobs = len(rollup_failed_cron_jobs(health_data["cron_jobs"]))
        scheduler_status = health_data.get("scheduler_status", {}).get("status")
        slo_status = health_data.get("data_pipeline_slo", {}).get("status")
        backend_error = any(
            backend.get("status") == "error"
            for backend in health_data.get("scheduler_status", {}).get("backends", {}).values()
        )

        health_data["system_status"] = derive_system_status(
            current=health_data.get("system_status", "healthy"),
            backend_error=backend_error,
            scheduler_status=scheduler_status,
            slo_status=slo_status,
            failed_jobs=failed_jobs,
            stale_count=stale_count,
        )
        health_data["system_status"] = elevate_system_status_for_kill(
            health_data["system_status"],
            kill_fields,
            open_incidents,
        )

        # Dual-SSOT: re-stamp ops_health_* from the monitor report so dashboard
        # regeneration does not wipe fields merged by make health /
        # publish_ops_health_surfaces (ops_health_status/source/timestamp).
        try:
            from src.monitor.health_check import apply_ops_monitor_to_dashboard_health

            apply_ops_monitor_to_dashboard_health(
                health_data,
                data_dir=DATA_DIR,
                public_dir=PUBLIC_DIR,
            )
        except Exception as exc:  # noqa: BLE001 — never block health.json write
            logger.warning("ops monitor merge into dashboard health failed: %s", exc)

        # When dashboard regenerates from cleared incidents.json / kill SSOT,
        # also patch lagging monitor data/health.json so operators do not see
        # open_count=1 on the monitor report while incidents.json is 0.
        try:
            from src.monitor.health_check import reconcile_monitor_health_with_disk_ssot

            reconcile_monitor_health_with_disk_ssot(data_dir=DATA_DIR)
        except Exception as exc:  # noqa: BLE001 — never block public health write
            logger.warning("monitor health SSOT reconcile failed: %s", exc)

        health_data = _stamp_generator_git_sha(health_data)
        out_path = PUBLIC_DIR / "health.json"
        # Batch IE: serialize-once multi-dest for dashboard health.json
        # (public + repo soft-mirror @ 0o644). Never fan-out to private
        # DATA_DIR/health.json — that file is the monitor schema SSOT
        # (operational_readiness) and must not be overwritten by the
        # dashboard payload. Mirrors Batch IC public health merge contract.
        wrote = False
        try:
            from src.monitor.signal_authority import write_json_multi_dest

            result = write_json_multi_dest(
                health_data,
                public_path=out_path,
                private_path=None,
                soft_mirror_repo=True,
                repo_filename="health.json",
            )
            wrote = bool(result.wrote_public or result.wrote_repo)
            if result.skipped_reason:
                logger.warning(
                    "dashboard health multi-dest partial skip: %s",
                    result.skipped_reason,
                )
        except Exception as multi_exc:  # noqa: BLE001 — fall back below
            logger.warning(
                "dashboard health multi-dest failed (%s); falling back to save_results_json",
                multi_exc,
            )
            wrote = False
        if not wrote:
            save_results_json(health_data, output_path=str(out_path))

        return out_path
    
    def _generate_sector_momentum_signals(self, vix_level: Optional[float] = None) -> Optional[Dict]:
        """Generate sector rotation momentum signals from historical data."""
        try:
            from src.strategy.sector_momentum_calc import generate_sector_signals

            historical_path = PUBLIC_DIR.parent / "data" / "historical.json"

            # Pass through None when VIX is unknown — do not coerce to 0.0
            # (zero VIX looks like ultra-calm and misleads regime/threshold gates).
            signals = generate_sector_signals(historical_path, vix=vix_level)
            return signals

        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("sector_momentum", e)
            return None
    
    def generate_analytics_json(self) -> Path:
        """Generate analytics data (drawdown, rolling metrics, benchmarks)."""
        # Import analytics calculator
        try:
            from src.analytics.calculator import AnalyticsCalculator
            calc = AnalyticsCalculator(data_dir=str(DATA_DIR))
            report = calc.generate_analytics_report()
            if isinstance(report, dict):
                report = _stamp_generator_git_sha(report)
            out_path = PUBLIC_DIR / "analytics.json"
            save_results_json(report, output_path=str(out_path))

            return out_path
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError, ImportError, RuntimeError) as e:
            # Fallback: empty analytics
            report = _stamp_generator_git_sha({
                "status": "error",
                "message": str(e),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })
            out_path = PUBLIC_DIR / "analytics.json"
            save_results_json(report, output_path=str(out_path))
            return out_path
    
    def generate_overlay_json(self) -> Optional[Path]:
        """Generate overlay dashboard data from all tactical overlays."""
        try:
            from src.dashboard.overlay_dashboard import OverlayDashboardGenerator
            gen = OverlayDashboardGenerator()
            dashboard = gen.generate()
            gen.save(dashboard)
            public_path = PUBLIC_DIR / gen.OUTPUT_PATH.name
            payload = _stamp_generator_git_sha(dashboard.to_dict())
            save_results_json(payload, output_path=str(public_path))
            return public_path
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("overlay_dashboard_generate", e)
            return None

    def generate_adaptive_sizing_json(self) -> Optional[Path]:
        """Generate adaptive sizing data for dashboard."""
        try:
            from src.strategy.adaptive_sizing import AdaptiveSizer

            sizer = AdaptiveSizer()
            decision = sizer.compute_allocation()

            authority = self._build_advisory_allocation_artifact_role(
                surface="adaptive_sizing",
                allocation_field="adjusted_allocation",
            )
            sizing_data = _stamp_generator_git_sha({
                "base_allocation": decision.base_allocation,
                "adjusted_allocation": decision.adjusted_allocation,
                "adjustments": decision.adjustments,
                "regime_adjustment": decision.regime_adjustment,
                "volatility_adjustment": decision.volatility_adjustment,
                "signal_adjustment": decision.signal_adjustment,
                "drawdown_adjustment": decision.drawdown_adjustment,
                "factors": asdict(decision.factors) if hasattr(decision.factors, '__dataclass_fields__') else {},
                "authority": authority,
                **self._flatten_advisory_authority(authority),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })

            out_path = PUBLIC_DIR / "adaptive_sizing.json"
            save_results_json(sizing_data, output_path=str(out_path))
            return out_path

        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("adaptive_sizing", e)
            return None

    def generate_vixy_hedge_json(self) -> Optional[Path]:
        """Generate VIXY hedge sizing data for dashboard."""
        try:
            from src.strategy.vixy_hedge_sizing import VIXYHedgeSizer

            sizer = VIXYHedgeSizer()
            status = sizer.status()

            hedge_data = _stamp_generator_git_sha({
                **status,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "canonical_controller": "hedge_selector",
                "runtime_role": "diagnostic_cost_evidence",
                "live_authoritative": False,
                "routed": False,
            })

            out_path = PUBLIC_DIR / "vixy_hedge.json"
            save_results_json(hedge_data, output_path=str(out_path))
            return out_path

        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("vixy_hedge", e)
            return None

    def generate_black_litterman_json(self) -> Optional[Path]:
        """Generate Black-Litterman mapper data for dashboard.

        Uses live ensemble voter signal data to generate real BL views
        with Idzorek confidence from signal health scores.
        """
        try:
            from src.strategy.ensemble_voter import EnsembleVoter
            from src.strategy.black_litterman_mapper import run_black_litterman

            # Get live BL views from ensemble voter
            voter = EnsembleVoter()
            bl_input = voter.get_bl_views()

            views = bl_input["views"]

            # Compute covariance matrix from price data in market.db
            symbols = list(views.symbols) if hasattr(views, 'symbols') else ['SPY', 'TLT', 'GLD']
            import pandas as pd
            price_data = {}
            for sym in symbols:
                try:
                    rows = self.conn.execute(
                        "SELECT date, close FROM prices WHERE symbol = ? ORDER BY date DESC LIMIT 252",
                        (sym,)
                    ).fetchall()
                    if rows:
                        price_data[sym] = pd.Series(
                            {r[0]: r[1] for r in reversed(rows)}
                        )
                except (sqlite3.OperationalError, KeyError):
                    continue

            # Try full BL optimization (requires PyPortfolioOpt)
            bl_weights = None
            posterior_returns = None
            result = None
            if len(price_data) >= 2:
                try:
                    prices_df = pd.DataFrame(price_data)
                    returns = prices_df.pct_change().dropna()
                    available = [s for s in symbols if s in returns.columns]
                    cov_matrix = returns[available].cov().values * 252  # Annualized
                    result = run_black_litterman(cov_matrix, views)
                    bl_weights = result.bl_weights
                    posterior_returns = result.posterior_returns
                except (ImportError, ValueError) as e:
                    logger.info("BL optimization unavailable (%s), using views-only output", e)

            # Fallback: use views without full BL optimization
            if bl_weights is None:
                bl_weights = dict(BASE_ALLOCATION)
            if posterior_returns is None:
                posterior_returns = {}

            # Use base allocation as prior
            prior = dict(BASE_ALLOCATION)
            canonical_assets = tuple(str(symbol).upper() for symbol in symbols)
            prior_public = self._canonicalize_public_weights(prior, canonical_assets=canonical_assets)
            posterior_public = self._canonicalize_public_weights(bl_weights, canonical_assets=canonical_assets)
            returns_public = self._canonicalize_public_weights(posterior_returns, canonical_assets=canonical_assets)

            # Build views list for panel consumption
            view_list = []
            if hasattr(views, 'absolute_views') and views.absolute_views:
                abs_views = views.absolute_views if isinstance(views.absolute_views, dict) else dict(zip(views.symbols, views.absolute_views))
                for i, sym in enumerate(views.symbols):
                    ret = abs_views.get(sym, 0.0)
                    conf = views.view_confidences[i] if i < len(views.view_confidences) else 0.5
                    view_list.append({
                        "signal_name": "ensemble_consensus",
                        "asset": str(sym).upper(),
                        "direction": "bullish" if ret > 0 else ("bearish" if ret < 0 else "neutral"),
                        "confidence": round(conf, 3),
                        "expected_return_delta": round(ret, 6),
                    })

            authority = self._build_advisory_allocation_artifact_role(
                surface="black_litterman",
                allocation_field="posterior_weights",
            )
            bl_data = {
                "prior_weights": prior_public["weights"],
                "posterior_weights": posterior_public["weights"],
                "posterior_returns": returns_public["weights"],
                "views": view_list,
                "tau": bl_input.get("tau", 0.15),
                "view_confidence_method": "idzorek",
                "optimization_available": result is not None,
                "excluded_assets": sorted(set(
                    prior_public["excluded_assets"]
                    + posterior_public["excluded_assets"]
                    + returns_public["excluded_assets"]
                )),
                "zero_weight_assets": sorted(set(
                    prior_public["zero_weight_assets"]
                    + posterior_public["zero_weight_assets"]
                )),
                "authority": authority,
                **self._flatten_advisory_authority(authority),
                "health_scores": bl_input.get("health_scores_used", {}),
                "biases": {
                    "equity": round(bl_input.get("equity_bias", 0.0), 3),
                    "duration": round(bl_input.get("duration_bias", 0.0), 3),
                    "gold": round(bl_input.get("gold_bias", 0.0), 3),
                },
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            bl_data = _stamp_generator_git_sha(bl_data)

            out_path = PUBLIC_DIR / "black_litterman.json"
            save_results_json(bl_data, output_path=str(out_path))
            return out_path

        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("black_litterman", e)
            return None

    def generate_turnover_validator_json(self) -> Optional[Path]:
        """Generate turnover validator data for dashboard."""
        try:
            from src.signals.signal_source import SignalSource
            from src.strategy.turnover_validator import TurnoverValidator

            validator = TurnoverValidator()
            diagnostics = validator.get_state_diagnostics()
            canonical_sources = {source.value for source in SignalSource}
            production_signals = {
                source: data
                for source, data in diagnostics.items()
                if source in canonical_sources
            }
            synthetic_baselines = {
                source: {
                    "metadata": {"source_type": "synthetic_or_fixture"},
                    "diagnostics": data,
                }
                for source, data in diagnostics.items()
                if source not in canonical_sources
            }

            turnover_data = _stamp_generator_git_sha({
                "schema_version": "turnover-validator/v1",
                "signals": production_signals,
                "synthetic_baselines": synthetic_baselines,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })

            out_path = PUBLIC_DIR / "turnover_validator.json"
            save_results_json(turnover_data, output_path=str(out_path))
            return out_path

        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("turnover_validator", e)
            return None

    @staticmethod
    def _normalize_gate_regime_name(regime: str | None) -> str:
        """Map live lowercase regimes to RegimeGate uppercase labels."""
        if not regime:
            return "NORMAL"
        name = str(regime).strip()
        if not name:
            return "NORMAL"
        # Gate rules use NORMAL/HIGH_VOL/…; live classify uses normal/vol_spike/…
        upper = name.upper()
        aliases = {
            "VOL_SPIKE": "HIGH_VOL",
            "VOLSPIKE": "HIGH_VOL",
            "HIGHVOL": "HIGH_VOL",
            "LOWVOL": "LOW_VOL",
            "LOW_VOL": "LOW_VOL",
            "HIGH_VOL": "HIGH_VOL",
            "CRISIS": "CRISIS",
            "RECOVERY": "RECOVERY",
            "NORMAL": "NORMAL",
        }
        return aliases.get(upper.replace("-", "_"), upper.replace("-", "_"))

    def _resolve_current_regime_for_gate(self) -> tuple[str, float, str]:
        """Resolve current regime + confidence for gate SSOT (not live order authority).

        Preference order:
        1. Live VIX/trend classifier via open DB connection (same as signals path)
        2. ensemble_voting on public/data signals.json
        3. regime_classifier_state.json (adaptive path; may be stale)
        4. Explicit default with disclosed source
        """
        # 1) Live VIX path when generator has a DB connection
        conn = getattr(self, "conn", None)
        if conn is not None:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT close FROM prices WHERE symbol = '^VIX' ORDER BY date DESC LIMIT 1"
                )
                vix_row = cursor.fetchone()
                vix_level = float(vix_row[0]) if vix_row and vix_row[0] is not None else None
                cursor.execute(
                    "SELECT regime FROM regime_log ORDER BY detected_at DESC LIMIT 1"
                )
                trend_row = cursor.fetchone()
                trend_regime = trend_row[0] if trend_row else "normal"
                live = classify_vix_regime(vix_level, trend_regime)
                conf = 0.7 if vix_level is not None else 0.55
                return self._normalize_gate_regime_name(live), conf, "classify_vix_regime"
            except Exception as exc:  # noqa: BLE001 — fall through to file SSOT
                logger.debug("regime_state: VIX path failed: %s", exc)

        # 2) Ensemble voting block on published signals
        for signals_path in (PUBLIC_DIR / "signals.json", DATA_DIR / "signals.json"):
            try:
                if not signals_path.exists():
                    continue
                with open(signals_path) as f:
                    signals = json.load(f)
                ensemble = signals.get("ensemble_voting") or {}
                if ensemble.get("regime") is not None:
                    conf_raw = ensemble.get("regime_confidence", 0.5)
                    try:
                        conf = float(conf_raw)
                    except (TypeError, ValueError):
                        conf = 0.5
                    return (
                        self._normalize_gate_regime_name(str(ensemble.get("regime"))),
                        conf,
                        "ensemble_voting",
                    )
                regime_block = signals.get("regime") or {}
                if isinstance(regime_block, dict) and regime_block.get("regime"):
                    return (
                        self._normalize_gate_regime_name(str(regime_block.get("regime"))),
                        0.6,
                        "signals.regime",
                    )
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.debug("regime_state: signals read failed (%s): %s", signals_path, exc)

        # 3) Adaptive classifier state (legacy parallel file)
        clf_path = DATA_DIR / "regime_classifier_state.json"
        try:
            if clf_path.exists():
                with open(clf_path) as f:
                    clf = json.load(f)
                regime = clf.get("current_regime") or clf.get("regime")
                if regime:
                    conf_raw = clf.get("confidence", 0.5)
                    if isinstance(clf.get("history"), list) and clf["history"]:
                        last = clf["history"][-1]
                        if isinstance(last, dict) and last.get("confidence") is not None:
                            conf_raw = last.get("confidence")
                    try:
                        conf = float(conf_raw)
                    except (TypeError, ValueError):
                        conf = 0.5
                    return (
                        self._normalize_gate_regime_name(str(regime)),
                        conf,
                        "regime_classifier_state",
                    )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.debug("regime_state: classifier read failed: %s", exc)

        return "NORMAL", 0.5, "default_missing_state"

    def _persist_regime_state(
        self,
        regime_name: str,
        confidence: float,
        source: str,
    ) -> Path:
        """Write DATA_DIR/regime_state.json SSOT for gate + graduation consumers."""
        regime_file = DATA_DIR / "regime_state.json"
        history: list = []
        previous = None
        if regime_file.exists():
            try:
                with open(regime_file) as f:
                    prior = json.load(f)
                previous = prior.get("regime")
                hist = prior.get("history")
                if isinstance(hist, list):
                    history = hist[-49:]  # keep last 50 after append
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                history = []

        now_iso = datetime.now().isoformat()
        history.append(
            {
                "timestamp": now_iso,
                "regime": regime_name,
                "confidence": confidence,
                "source": source,
            }
        )
        payload = {
            "regime": regime_name,
            "confidence": confidence,
            "source": source,
            "previous_regime": previous,
            "updated_at": now_iso,
            "schema_version": "regime-state/v1",
            "note": (
                "SSOT for dashboard regime_gate + graduation regime_coverage; "
                "not live order-routing authority (see regime_authority / target_allocations)."
            ),
            "history": history,
        }
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        save_results_json(payload, output_path=str(regime_file))

        # Append one regime_log line so graduation coverage can accumulate over cycles
        try:
            log_path = DATA_DIR / "regime_log.json"
            with open(log_path, "a", encoding="utf-8") as logf:
                logf.write(
                    json.dumps(
                        {
                            "regime": regime_name,
                            "confidence": confidence,
                            "source": source,
                            "detected_at": now_iso,
                        }
                    )
                    + "\n"
                )
        except OSError as exc:
            logger.debug("regime_state: regime_log append failed: %s", exc)

        return regime_file

    def generate_regime_gate_json(self) -> Optional[Path]:
        """Generate regime gate status data for dashboard."""
        try:
            from src.signals.regime_gate import RegimeGate

            gate = RegimeGate()
            summary = gate.get_gate_summary()

            # Resolve + persist SSOT so gate/graduation never depend on a dead path
            regime_name, regime_confidence, confidence_source = (
                self._resolve_current_regime_for_gate()
            )
            self._persist_regime_state(regime_name, regime_confidence, confidence_source)

            # Build gate rules with current regime active status
            all_signals = self._dedupe_preserve_order(
                list(summary.keys()) + ["alt_data", "cross_asset_rv", "unified_overlay"],
            )
            active_signals = self._dedupe_preserve_order(gate.get_active_signal_names(
                all_signals,
                regime_name,
            ))
            inactive_signals = [s for s in all_signals if s not in active_signals]

            gate_data = {
                "current_regime": regime_name,
                "regime_confidence": regime_confidence,
                "confidence_source": confidence_source,
                "gate_rules": [
                    {"signal_name": sig, "off_regimes": sorted(regimes), "is_active": sig in active_signals}
                    for sig, regimes in summary.items()
                ],
                "active_signals": active_signals,
                "inactive_signals": inactive_signals,
                "min_dwell_days": gate.min_dwell_days,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

            # Compute data-driven regime Sharpe matrix (read-only, for monitoring)
            try:
                from src.monitor.regime_sharpe_matrix import (
                    compute_regime_sharpe_matrix,
                    derive_gate_rules,
                    derive_regime_weight_multipliers,
                    extract_signal_regime_data,
                )

                prices = self._load_price_data()
                db_path = DATA_DIR / "ensemble_signals.db"
                if prices is not None and db_path.exists():
                    hist_df = extract_signal_regime_data(db_path, prices)
                    if not hist_df.empty:
                        matrix = compute_regime_sharpe_matrix(
                            hist_df, n_bootstrap=500, seed=42,
                        )
                        data_rules = derive_gate_rules(matrix)
                        data_multipliers = derive_regime_weight_multipliers(matrix)

                        gate_data["data_driven"] = {
                            "gate_rules": {
                                sig: sorted(regimes)
                                for sig, regimes in data_rules.items()
                            },
                            "weight_multipliers": data_multipliers,
                            "n_observations": len(hist_df),
                            "n_signals": hist_df["signal"].nunique(),
                        }

                        # Persist data-driven rules for EnsembleVoter to load
                        persist_path = DATA_DIR / "regime_gate_persisted.json"
                        persist_data = {
                            "gate_rules": {
                                sig: sorted(regimes)
                                for sig, regimes in data_rules.items()
                            },
                            "weight_multipliers": data_multipliers,
                            "computed_at": datetime.now().isoformat(),
                            "n_observations": len(hist_df),
                        }
                        save_results_json(persist_data, output_path=str(persist_path))
            except (ImportError, Exception) as e:
                logger.debug("Regime Sharpe matrix computation skipped: %s", e)

            gate_data = _stamp_generator_git_sha(gate_data)
            out_path = PUBLIC_DIR / "regime_gate.json"
            save_results_json(gate_data, output_path=str(out_path))
            return out_path

        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("regime_gate", e)
            return None

    # Cache last-known regime for _is_msm_gated resilience
    _last_regime: str = "normal"

    def _is_msm_gated(self) -> bool:
        """Check if MSM should be gated off based on current regime.

        MSM has zero ensemble weight in HIGH_VOL/CRISIS regimes (health 0.55,
        net-negative -0.012 Sharpe). Returns True when gated off.

        On transient query failures, uses the last-known regime instead of
        immediately gating MSM off — a single SQLite hiccup should not
        disable a strategy.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT regime FROM regime_log ORDER BY detected_at DESC LIMIT 1")
            row = cursor.fetchone()
            regime = row[0] if row else "normal"
            DashboardGenerator._last_regime = regime
            return regime.lower() in {"high_vol", "crisis"}
        except Exception as e:
            logger.warning("_is_msm_gated: regime query failed (%s) — using last-known regime '%s'",
                           e, DashboardGenerator._last_regime)
            return DashboardGenerator._last_regime.lower() in {"high_vol", "crisis"}

    @staticmethod
    def _extract_vix_term_structure_signal(vix_term_structure: Optional[Dict]) -> Optional[float]:
        """Extract the fractional VIX term-structure signal when available."""
        if not isinstance(vix_term_structure, dict):
            return None
        raw_signal = vix_term_structure.get("signal_value")
        if raw_signal is None:
            return None
        try:
            return float(raw_signal)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _resolve_hedge_vix_level(
        vix_level: Optional[float],
        vix_term_structure: Optional[Dict],
    ) -> Optional[float]:
        """Use the VIX term-structure spot value when market.db lacks ^VIX."""
        if vix_level is not None:
            return vix_level
        if not isinstance(vix_term_structure, dict):
            return None
        raw_vix = vix_term_structure.get("vix_spot")
        if raw_vix is None:
            return None
        try:
            return float(raw_vix)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _build_unavailable_hedge_selector_signal(
        regime: str,
        term_structure_signal: Optional[float],
    ) -> Dict[str, Any]:
        """Publish a typed canonical hedge-selector artifact when VIX is unavailable."""
        return {
            "available": False,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "regime": regime,
            "regime_confidence": 0.0,
            "primary_hedge": "none",
            "primary_size_pct": 0.0,
            "secondary_hedge": None,
            "secondary_size_pct": 0.0,
            "cost_benefit_gate": False,
            "net_benefit_bps": 0.0,
            "kelly_fraction": 0.0,
            "expected_cost_bps": 0.0,
            "expected_benefit_bps": 0.0,
            "min_hold_days": 0,
            "transition_cost_bps": 0.0,
            "canonical_controller": "hedge_selector",
            "vixy_role": "diagnostic_sizing_helper",
            "term_structure_role": "gate_discount_multiplier",
            "term_structure_gate": False,
            "term_structure_multiplier": 0.0,
            "term_structure_signal": term_structure_signal,
            "gate_reason": "vix_unavailable",
        }

    def _get_hedge_selector_signal(
        self,
        vix_level: Optional[float],
        regime: str,
        vix_term_structure: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Get hedge selector recommendation for dashboard."""
        term_structure_signal = self._extract_vix_term_structure_signal(vix_term_structure)
        hedge_vix_level = self._resolve_hedge_vix_level(vix_level, vix_term_structure)
        if hedge_vix_level is None:
            return self._build_unavailable_hedge_selector_signal(
                regime,
                term_structure_signal,
            )
        try:
            from src.strategy.hedge_selector import HedgeSelector
            selector = HedgeSelector()
            # Estimate confidence based on regime stability
            regime_confidence = 0.8 if regime in ["normal", "crisis"] else 0.6
            rec = selector.select(
                vix_level=hedge_vix_level,
                regime_confidence=regime_confidence,
                regime_label=regime,
                term_structure_signal=term_structure_signal,
            )
            return {
                "available": True,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "regime": rec.regime,
                "regime_confidence": rec.regime_confidence,
                "primary_hedge": rec.primary_hedge,
                "primary_size_pct": rec.primary_size_pct,
                "secondary_hedge": rec.secondary_hedge,
                "secondary_size_pct": rec.secondary_size_pct,
                "cost_benefit_gate": rec.cost_benefit_gate,
                "net_benefit_bps": rec.net_benefit_bps,
                "kelly_fraction": rec.kelly_fraction,
                "expected_cost_bps": rec.expected_cost_bps,
                "expected_benefit_bps": rec.expected_benefit_bps,
                "min_hold_days": rec.min_hold_days,
                "transition_cost_bps": rec.transition_cost_bps,
                "canonical_controller": rec.canonical_controller,
                "vixy_role": rec.vixy_role,
                "term_structure_role": rec.term_structure_role,
                "term_structure_gate": rec.term_structure_gate,
                "term_structure_multiplier": rec.term_structure_multiplier,
                "term_structure_signal": term_structure_signal,
                "gate_reason": rec.gate_reason,
            }
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("hedge_selector", e)
            return None

    # Signal staleness detection (production readiness)
    SIGNAL_STALENESS_TTL_HOURS = int(os.environ.get("SIGNAL_STALENESS_TTL_HOURS", "4"))
    STALENESS_DECAY_TAU_HOURS = float(os.environ.get("STALENESS_DECAY_TAU_HOURS", "2.0"))
    from src.monitor.signal_ownership import optional_advisory_signals

    OPTIONAL_SIGNAL_STALENESS_KEYS = optional_advisory_signals()
    OPTIONAL_DAILY_SIGNAL_STALENESS_KEYS = {
        "convexity_harvest",
        "volatility_parity",
    }

    @staticmethod
    def _normalized_signal_timestamp(
        signal_block: Any,
        preferred_field: str,
        *,
        allow_date: bool = False,
    ) -> str | None:
        """Return the first usable timestamp from a generated signal block."""
        if not isinstance(signal_block, dict):
            return None
        fields = [
            preferred_field,
            "generated_at",
            "timestamp",
            "generated",
            "detected",
            "last_update",
        ]
        for field in dict.fromkeys(fields):
            value = signal_block.get(field)
            if isinstance(value, str) and value:
                return value
        if allow_date:
            value = signal_block.get("date")
            if isinstance(value, str) and value:
                try:
                    parsed_date = datetime.strptime(value, "%Y-%m-%d").date()
                    # Daily sections remain fresh through their UTC calendar day.
                    return datetime.combine(
                        parsed_date,
                        datetime.max.time().replace(microsecond=0),
                        tzinfo=timezone.utc,
                    ).isoformat()
                except ValueError:
                    return value
        return None

    @staticmethod
    def _is_unavailable_signal_block(signal_block: Any) -> bool:
        """Return true for explicit unavailable/error placeholders."""
        if signal_block is None:
            return True
        if not isinstance(signal_block, dict):
            return False
        status = str(signal_block.get("status", "")).lower()
        if status in {"unavailable", "disabled", "missing"}:
            return True
        source_mode = str(signal_block.get("source_mode", "")).lower()
        if source_mode in {"unavailable", "synthetic", "last_good", "fallback"}:
            return True
        cache_status = str(signal_block.get("cache_status", "")).lower()
        if cache_status in {"unavailable", "empty", "missing", "failed", "degraded"}:
            return True
        return "error" in signal_block

    def _check_signal_staleness(self, signal_data: Dict) -> Dict:
        """Check staleness of each signal source in signals.json output.

        Compares each signal's `generated_at` / `timestamp` field against a TTL
        (default 4 hours). Stale signals should be removed from ensemble weight
        numerator/denominator (not zeroed — zeroing distorts relative weights).

        Also computes per-signal staleness decay factors for ensemble weight
        adjustment. Decay uses exponential: weight *= exp(-age_hours / tau)
        where tau defaults to 2h (STALENESS_DECAY_TAU_HOURS env var).

        Returns:
            Dict with keys:
            - stale_signals: list of signal names that are stale
            - signal_timestamps: dict of signal_name -> last_known_timestamp
            - signal_age_hours: dict of signal_name -> age in hours (None if missing)
            - staleness_decay: dict of signal_name -> decay factor (0.0-1.0)
            - healthy_count: number of fresh signals
            - total_count: total number of signals checked
        """
        import math as _math

        ttl_seconds = self.SIGNAL_STALENESS_TTL_HOURS * 3600
        tau_hours = self.STALENESS_DECAY_TAU_HOURS
        now = datetime.now(timezone.utc)
        stale_signals = []
        unavailable_signals = []
        signal_timestamps = {}
        signal_age_hours = {}
        staleness_decay = {}

        # Known signal keys in signals.json that have timestamps
        timestamped_signals = {
            "ensemble_voting": ("generated_at", None),
            "alternative_data": ("timestamp", None),
            "behavioral_sentiment": ("timestamp", None),
            "garch_cvar": ("timestamp", None),
            "smart_rebalance": ("generated_at", None),
            "calendar_seasonality": ("generated_at", None),
            "crypto_allocation": ("generated_at", None),
            "factor_rotation": ("generated_at", None),
            "stacking_ensemble": ("generated_at", None),
            "convexity_harvest": ("generated_at", None),
            "llm_sentiment": ("generated_at", None),
            "sector_rotation": ("generated_at", None),
            "kurtosis_regime": ("generated_at", None),
            "volatility_parity": ("generated_at", None),
            "collar": ("generated_at", None),
            "bond_momentum": ("generated_at", None),
            "risk_decomposition": ("generated_at", None),
            "rebalance_health": ("generated_at", None),
            "two_stage_regime": ("timestamp", None),
            "bocd_regime": ("timestamp", None),
            "regime_transition": ("timestamp", None),
            "hedge_selector": ("generated_at", None),
            "fred_macro": ("timestamp", None),
        }

        for signal_key, (ts_field, _) in timestamped_signals.items():
            signal_block = signal_data.get(signal_key)
            if signal_block is None:
                if signal_key in self.OPTIONAL_SIGNAL_STALENESS_KEYS:
                    unavailable_signals.append(signal_key)
                    signal_timestamps[signal_key] = None
                    signal_age_hours[signal_key] = None
                    staleness_decay[signal_key] = 0.0
                    continue
                stale_signals.append(signal_key)
                signal_timestamps[signal_key] = None
                signal_age_hours[signal_key] = None
                staleness_decay[signal_key] = 0.0
                continue

            is_optional = signal_key in self.OPTIONAL_SIGNAL_STALENESS_KEYS
            if self._is_unavailable_signal_block(signal_block):
                unavailable_signals.append(signal_key)
                signal_timestamps[signal_key] = None
                signal_age_hours[signal_key] = None
                staleness_decay[signal_key] = 0.0
                continue

            ts_str = self._normalized_signal_timestamp(
                signal_block,
                ts_field,
                allow_date=signal_key in self.OPTIONAL_DAILY_SIGNAL_STALENESS_KEYS,
            )
            if ts_str is None and not is_optional and isinstance(signal_block, dict):
                artifact_ts = signal_data.get("generated_at") or signal_data.get("timestamp")
                if isinstance(artifact_ts, str) and artifact_ts:
                    ts_str = artifact_ts
            signal_timestamps[signal_key] = ts_str

            if ts_str is None:
                if is_optional:
                    unavailable_signals.append(signal_key)
                    signal_age_hours[signal_key] = None
                    staleness_decay[signal_key] = 0.0
                    continue
                stale_signals.append(signal_key)
                signal_age_hours[signal_key] = None
                staleness_decay[signal_key] = 0.0
                continue

            try:
                # Parse ISO timestamp — handle both Z and +00:00 suffixes.
                # Batch CL: naive timestamps are host-local wall clock (lab CST,
                # etc.). Prefer astimezone(UTC) so local evening is not treated
                # as UTC (false age 0 / false future).
                ts_str_clean = ts_str.replace("Z", "+00:00")
                ts = datetime.fromisoformat(ts_str_clean)
                ts = ts.astimezone(timezone.utc)
                age_seconds = max((now - ts).total_seconds(), 0.0)
                age_hours = age_seconds / 3600.0
                signal_age_hours[signal_key] = round(age_hours, 2)

                # Exponential decay: fresh signals get 1.0, stale signals approach 0.0
                decay = _math.exp(-age_hours / tau_hours) if tau_hours > 0 else 1.0
                decay = min(max(decay, 0.0), 1.0)
                staleness_decay[signal_key] = round(decay, 4)

                if age_seconds > ttl_seconds:
                    stale_signals.append(signal_key)
            except (ValueError, TypeError):
                stale_signals.append(signal_key)
                signal_age_hours[signal_key] = None
                staleness_decay[signal_key] = 0.0

        # Producer-aware override for alternative_data: do not escalate kill on
        # projection lag when alternative_data_latest.json is still fresh.
        projection_lag_signals: list[str] = []
        producer_ts = load_alternative_data_producer_timestamp(DATA_DIR)
        if producer_ts and "alternative_data" in timestamped_signals:
            try:
                pts = datetime.fromisoformat(producer_ts.replace("Z", "+00:00"))
                pts = pts.astimezone(timezone.utc)  # Batch CL: naive = local
                producer_age_hours = max((now - pts).total_seconds(), 0.0) / 3600.0
                producer_fresh = producer_age_hours * 3600.0 <= ttl_seconds
                projected_ts = signal_timestamps.get("alternative_data")
                projected_stale = "alternative_data" in stale_signals
                producer_ahead = False
                if projected_ts:
                    try:
                        ets = datetime.fromisoformat(
                            str(projected_ts).replace("Z", "+00:00")
                        )
                        ets = ets.astimezone(timezone.utc)
                        producer_ahead = pts > ets
                    except (ValueError, TypeError):
                        producer_ahead = True
                else:
                    producer_ahead = True

                if producer_fresh and (projected_stale or producer_ahead):
                    if projected_stale and "alternative_data" in stale_signals:
                        stale_signals = [s for s in stale_signals if s != "alternative_data"]
                    if producer_ahead:
                        projection_lag_signals.append("alternative_data")
                    # Prefer producer timestamp / age for operator honesty
                    signal_timestamps["alternative_data"] = producer_ts
                    signal_age_hours["alternative_data"] = round(producer_age_hours, 2)
                    decay = (
                        _math.exp(-producer_age_hours / tau_hours) if tau_hours > 0 else 1.0
                    )
                    staleness_decay["alternative_data"] = round(
                        min(max(decay, 0.0), 1.0), 4
                    )
            except (ValueError, TypeError):
                pass

        healthy_count = len(timestamped_signals) - len(stale_signals) - len(unavailable_signals)
        return {
            "stale_signals": stale_signals,
            "unavailable_signals": unavailable_signals,
            "projection_lag_signals": projection_lag_signals,
            "signal_timestamps": signal_timestamps,
            "signal_age_hours": signal_age_hours,
            "staleness_decay": staleness_decay,
            "decay_tau_hours": tau_hours,
            "healthy_count": healthy_count,
            "total_count": len(timestamped_signals),
            "required_count": len(timestamped_signals) - len(self.OPTIONAL_SIGNAL_STALENESS_KEYS),
            "optional_count": len(self.OPTIONAL_SIGNAL_STALENESS_KEYS),
            "ttl_hours": self.SIGNAL_STALENESS_TTL_HOURS,
            "checked_at": now.isoformat(),
        }

    def _apply_staleness_decay(self, output: Dict) -> Dict:
        """Apply staleness-weighted decay to ensemble voting weights.

        When signals are stale, their ensemble weights degrade proportionally
        using exponential decay. This ensures the dashboard and downstream
        consumers reflect signal freshness in allocation decisions.

        Decay formula: adjusted_weight = raw_weight * exp(-age_hours / tau)
        where tau = STALENESS_DECAY_TAU_HOURS (default 2h).
        """
        staleness = output.get("staleness", {})
        decay_factors = staleness.get("staleness_decay", {})
        if not decay_factors:
            return output

        # Apply decay to ensemble_voting source_breakdown weights
        ensemble = output.get("ensemble_voting")
        if isinstance(ensemble, dict) and "source_breakdown" in ensemble:
            for src in ensemble["source_breakdown"]:
                source_name = src.get("source", "")
                # Map ensemble source names to staleness signal keys
                staleness_key = _ENSEMBLE_STALENESS_MAP.get(source_name)
                if staleness_key and staleness_key in decay_factors:
                    decay = decay_factors[staleness_key]
                    original_weight = src.get("weight", 0.0)
                    src["weight_original"] = original_weight
                    src["weight"] = round(original_weight * decay, 4)
                    src["staleness_decay"] = decay

            # Recompute weighted_consensus with decayed weights
            valid_sources = []
            for source in ensemble["source_breakdown"]:
                try:
                    value = float(source.get("value", 0.0))
                    weight = float(source.get("weight", 0.0))
                except (TypeError, ValueError):
                    continue
                if np.isnan(value) or np.isnan(weight):
                    continue
                valid_sources.append((value, weight))

            total_weight = sum(weight for _, weight in valid_sources)
            if total_weight > 0:
                weighted_consensus = sum(
                    value * weight for value, weight in valid_sources
                ) / total_weight
                agreement_weight = sum(
                    weight for value, weight in valid_sources
                    if np.sign(value) == np.sign(weighted_consensus) or abs(value) < 0.1
                )
                ensemble["weighted_consensus"] = round(weighted_consensus, 4)
                ensemble["agreement_ratio"] = round(agreement_weight / total_weight, 4)
                # Raw mass after decay (may be < 1 when sources missing/stale)
                ensemble["total_weight_after_decay"] = round(total_weight, 4)
                ensemble["active_weight_mass"] = round(total_weight, 4)
                # Renorm before entropy / n_eff so diversification is not understated
                w_pos = np.array(
                    [w for _, w in valid_sources if w > 0],
                    dtype=float,
                )
                if len(w_pos) > 0 and float(np.sum(w_pos)) > 0:
                    w_norm = w_pos / float(np.sum(w_pos))
                    weight_entropy = float(-np.sum(w_norm * np.log(w_norm)))
                    ensemble["weight_entropy"] = round(weight_entropy, 4)
                    ensemble["n_eff"] = round(float(np.exp(weight_entropy)), 2)

            # Batch CW/CX/CZ/DU: preserve gate maps + recovery metrics through staleness rebuild
            sleep_map = ensemble.get("health_gate_slept") or {}
            if not isinstance(sleep_map, dict):
                sleep_map = {}
            regime_map = ensemble.get("regime_gated") or {}
            if not isinstance(regime_map, dict):
                regime_map = {}
            soft_floor_map = ensemble.get("health_gate_soft_floor") or {}
            if not isinstance(soft_floor_map, dict):
                soft_floor_map = {}
            sh_metrics = DashboardGenerator._signal_health_metrics_map()
            ensemble["configured_source_status"] = self._build_configured_source_status(
                ensemble.get("regime", "normal"),
                ensemble["source_breakdown"],
                health_gate_slept=sleep_map,
                regime_gated=regime_map,
                health_metrics=sh_metrics,
                health_gate_soft_floor=soft_floor_map,
            )
            if sleep_map:
                ensemble["health_gate_recovery"] = [
                    {
                        "source": name,
                        "sleep_reason": sleep_map.get(name),
                        **(sh_metrics.get(name) or {}),
                        **(
                            {"label_alignment": la}
                            if (
                                la := DashboardGenerator._label_alignment_diagnostic(
                                    name
                                )
                            )
                            else {}
                        ),
                    }
                    for name in sorted(sleep_map.keys())
                ]
            zb_shadow = [
                row["shadow"]
                for row in (ensemble.get("configured_source_status") or [])
                if isinstance(row, dict)
                and row.get("status") == "zero_baseline"
                and isinstance(row.get("shadow"), dict)
            ]
            ensemble["zero_baseline_shadow"] = zb_shadow
            ensemble["zero_baseline_shadow_count"] = len(zb_shadow)
            ensemble["inactive_signal_shadow"] = [
                row["shadow"]
                for row in (ensemble.get("configured_source_status") or [])
                if isinstance(row, dict)
                and row.get("status") == "inactive_signal"
                and isinstance(row.get("shadow"), dict)
            ]
            ensemble["post_fix_cohorts"] = [
                {
                    "source": row.get("source"),
                    **(row.get("cohort_readiness") or {}),
                }
                for row in (ensemble.get("configured_source_status") or [])
                if isinstance(row, dict)
                and isinstance(row.get("cohort_readiness"), dict)
            ]
            ensemble.update(self._build_ensemble_source_count_metadata(
                ensemble.get("regime", "normal"),
                ensemble["source_breakdown"],
                configured_source_status=ensemble.get("configured_source_status"),
            ))
            ensemble.update(
                self._ensemble_active_weights_rollup(
                    ensemble.get("configured_source_status") or []
                )
            )

        return output

    def _run_spc_monitor(self, output: Dict) -> Dict:
        """Run SPC monitoring on signal values.

        Tracks rolling statistics of signal values and flags signals whose
        distribution has shifted (3-sigma breach for 3+ consecutive periods).
        """
        try:
            from src.monitor.spc_monitor import SPCMonitor
        except (ImportError, AttributeError) as e:
            logger.warning("SPC monitor not available: %s", e)
            return {"status": "unavailable", "error": str(e)}

        # Initialize class-level SPC monitor (persists across runs)
        if DashboardGenerator._spc_monitor is None:
            DashboardGenerator._spc_monitor = SPCMonitor()
            DashboardGenerator._spc_monitor.load_state()

        monitor = DashboardGenerator._spc_monitor

        # Record current signal values for SPC tracking
        ensemble = output.get("ensemble_voting")
        if isinstance(ensemble, dict) and "source_breakdown" in ensemble:
            for src in ensemble["source_breakdown"]:
                source_name = src.get("source", "")
                value = src.get("value")
                if source_name and value is not None:
                    try:
                        monitor.record(source_name, float(value))
                    except (ValueError, TypeError):
                        pass

        # Also track key aggregate metrics
        if isinstance(ensemble, dict):
            consensus = ensemble.get("weighted_consensus")
            if consensus is not None:
                try:
                    monitor.record("_ensemble_consensus", float(consensus))
                except (ValueError, TypeError):
                    pass

        # Get status
        flags = monitor.check_flags()
        all_status = monitor.get_all_status()

        # Persist state for next process invocation
        monitor.save_state()

        # status must reflect flags — never hardcode ok when flagged_signals
        # is non-empty (operators gate on spc.status without re-parsing flags).
        spc_status = "ok"
        if flags:
            max_breaches = max(
                (int(f.get("consecutive_breaches") or 0) for f in flags),
                default=0,
            )
            limit = int(monitor.consecutive_breach_limit or 3)
            # Severe when well past the consecutive threshold; else alert.
            if max_breaches >= max(limit * 3, limit + 10):
                spc_status = "breach"
            else:
                spc_status = "alert"
        else:
            # Any is_flagged in signal_status without list entry (defensive).
            if any(
                isinstance(s, dict) and s.get("is_flagged")
                for s in (all_status or {}).values()
            ):
                spc_status = "alert"

        return {
            "status": spc_status,
            "flagged_signals": flags,
            "signal_status": all_status,
            "window_size": monitor.window_size,
            "sigma_threshold": monitor.sigma_threshold,
            "consecutive_breach_limit": monitor.consecutive_breach_limit,
        }

    def generate_tsmom_json(self) -> Optional[Path]:
        """Generate TSMOM overlay data for dashboard."""
        try:
            from src.signals.tsmom_overlay import TSMOMOverlay

            overlay = TSMOMOverlay()
            tickers = ['SPY', 'GLD', 'TLT']
            signals = []
            for ticker in tickers:
                sig = overlay.compute_signal(ticker)
                if sig is not None:
                    signals.append(sig)

            # Build speed breakdown from multi-speed TSMOM
            speed_breakdown = []
            for signal in signals:
                speed_breakdown.append({
                    "label": f"{signal.ticker} TSMOM",
                    "weight": signal.base_weight,
                    "signal": signal.signal,
                    "asset_signals": {signal.ticker: signal.adjustment},
                    "realized_vol": signal.realized_vol,
                    "adjustment": signal.adjustment,
                })

            tsmom_data = _stamp_generator_git_sha({
                "composite_signal": float(np.mean([s.signal for s in signals])) if signals else 0.0,
                "speed_breakdown": speed_breakdown,
                "position_recommendation": "long" if np.mean([s.signal for s in signals]) > 0.1 else ("short" if np.mean([s.signal for s in signals]) < -0.1 else "neutral") if signals else "neutral",
                "confidence": min(1.0, abs(np.mean([s.vol_scaled_position for s in signals]))) if signals else 0.0,
                "standalone_sharpe": 0.96,
                "overlay_sharpe": 0.93,
                "health_score": 0.55,
                "is_gated_off": self._is_msm_gated(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })

            out_path = PUBLIC_DIR / "tsmom.json"
            save_results_json(tsmom_data, output_path=str(out_path))
            return out_path

        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("tsmom", e)
            return None

    def generate_cross_asset_rv_json(self) -> Optional[Path]:
        """Generate cross-asset relative value data for dashboard."""
        try:
            from src.signals.cross_asset_relative_value import CrossAssetRVScanner

            scanner = CrossAssetRVScanner()
            signal = scanner.scan_all()

            # Determine gating from current regime (v961: RV fails in HIGH_VOL/CRISIS)
            cursor = self.conn.cursor()
            cursor.execute("SELECT regime FROM regime_log ORDER BY detected_at DESC LIMIT 1")
            row = cursor.fetchone()
            regime = row[0] if row else "normal"
            gated_regimes = {"high_vol", "crisis"}
            is_gated = regime.lower() in gated_regimes

            rv_data = {
                "signal_value": signal.risk_on_score,
                "pairs": [p.to_dict() for p in signal.pairs.values()],
                "avg_z_score": signal.avg_z_score,
                "max_divergence": signal.max_divergence,
                "num_diverged": signal.num_diverged,
                "total_pairs": signal.total_pairs,
                "available_pair_count": signal.available_pair_count,
                "unavailable_pair_count": signal.unavailable_pair_count,
                "unavailable_pairs": signal.unavailable_pairs,
                "missing_symbols": signal.missing_symbols,
                "risk_on_score": signal.risk_on_score,
                "duration_score": signal.duration_score,
                "overall_conviction": signal.overall_conviction,
                "current_regime": regime,
                "is_gated_off": is_gated,
                "regime_note": "Mean-reversion fails in volatile regimes" if is_gated else "Active — mean-reversion favorable",
                "weight_in_ensemble": 0.0 if is_gated else 0.13,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            rv_data = _stamp_generator_git_sha(rv_data)

            out_path = PUBLIC_DIR / "cross_asset_rv.json"
            save_results_json(rv_data, output_path=str(out_path))
            return out_path

        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("cross_asset_rv", e)
            return None

    @staticmethod
    def _graduation_display_value(value: Any) -> str:
        """Format checklist numeric/bool values for the dashboard criterion table."""
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        if isinstance(value, float):
            if value == 0.0:
                return "0"
            if abs(value) >= 100:
                return f"{value:.1f}"
            if abs(value) >= 1:
                return f"{value:.2f}"
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return str(value)

    @staticmethod
    def _paper_trading_summary_for_dashboard(
        state: Dict[str, Any],
        *,
        days_elapsed: Any,
        days_required: Any,
    ) -> Dict[str, Any]:
        """Build frontend paper_trading block from checklist-loaded state.

        Dual-shape: dashboard GraduationDataSchema requires start_date /
        initial_capital / current_value / days_elapsed / days_required while
        the producer keeps trading_days / min_trading_days at the top level.
        """
        portfolio = state.get("portfolio") if isinstance(state.get("portfolio"), dict) else {}
        summary = (
            state.get("paper_trading_summary")
            if isinstance(state.get("paper_trading_summary"), dict)
            else {}
        )
        history = portfolio.get("history") if isinstance(portfolio.get("history"), list) else []

        start_date = ""
        if history and isinstance(history[0], dict):
            ts = history[0].get("timestamp")
            if isinstance(ts, str) and ts:
                start_date = ts[:10]
        if not start_date:
            date_hint = summary.get("date")
            if isinstance(date_hint, str) and date_hint:
                start_date = date_hint[:10]

        initial_capital = 100_000.0
        current_value: Optional[float] = None

        if history and isinstance(history[0], dict):
            start_val = history[0].get("total_value")
            if isinstance(start_val, (int, float)):
                initial_capital = float(start_val)
        if history and isinstance(history[-1], dict):
            end_val = history[-1].get("total_value")
            if isinstance(end_val, (int, float)):
                current_value = float(end_val)

        # Prefer authoritative paper-trading-performance metrics when present.
        start_value_files = sorted(DATA_DIR.glob("paper-trading-performance-*.json"))
        if start_value_files:
            try:
                with open(start_value_files[-1]) as f:
                    perf_raw = json.load(f)
                if isinstance(perf_raw, dict):
                    if not start_date and isinstance(perf_raw.get("date"), str):
                        start_date = perf_raw["date"][:10]
                    perf = perf_raw.get("performance")
                    if isinstance(perf, dict):
                        sv = perf.get("start_value")
                        if isinstance(sv, (int, float)):
                            initial_capital = float(sv)
                        cv = perf.get("current_value")
                        if isinstance(cv, (int, float)):
                            current_value = float(cv)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass

        if current_value is None:
            cash = portfolio.get("cash")
            positions = portfolio.get("positions")
            if isinstance(cash, (int, float)) and isinstance(positions, dict):
                pos_sum = 0.0
                for pos in positions.values():
                    if isinstance(pos, dict) and isinstance(pos.get("value"), (int, float)):
                        pos_sum += float(pos["value"])
                current_value = float(cash) + pos_sum
        if current_value is None:
            current_value = initial_capital

        try:
            days_elapsed_n = int(days_elapsed) if days_elapsed is not None else 0
        except (TypeError, ValueError):
            days_elapsed_n = 0
        try:
            days_required_n = int(days_required) if days_required is not None else 0
        except (TypeError, ValueError):
            days_required_n = 0

        if not start_date:
            start_date = datetime.now().date().isoformat()

        return {
            "start_date": start_date,
            "initial_capital": round(initial_capital, 2),
            "current_value": round(float(current_value), 2),
            "days_elapsed": days_elapsed_n,
            "days_required": days_required_n,
        }

    def generate_graduation_json(self) -> Optional[Path]:
        """Generate graduation readiness progress for dashboard (dual private+public)."""
        return refresh_graduation_dual_surfaces(
            public_dir=PUBLIC_DIR,
            data_dir=DATA_DIR,
            paper_trading_builder=self._paper_trading_summary_for_dashboard,
            display_value=self._graduation_display_value,
        )


    @staticmethod
    def _latest_stale_explainability_metadata(source_dir: Path) -> Dict[str, Any]:
        """Return metadata for the newest historical explainability file, if any."""
        dated_files = sorted(source_dir.glob("explainability_*.json"), reverse=True)
        if not dated_files:
            return {}

        latest = dated_files[0]
        metadata: Dict[str, Any] = {"stale_source_file": latest.name}
        try:
            payload = json.loads(latest.read_text())
            analysis_date = payload.get("analysis_date")
            if analysis_date:
                metadata["stale_analysis_date"] = str(analysis_date)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            metadata["stale_read_error"] = str(e)
        return metadata

    @staticmethod
    def _build_unavailable_explainability_payload(
        *,
        generated_at: str,
        reason: str,
        source_file: Optional[str] = None,
        analysis_date: Optional[str] = None,
        stale_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build an explicit no-current-explainability artifact."""
        return {
            "timestamp": generated_at,
            "analysis_date": analysis_date or generated_at[:10],
            "latest_decision": None,
            "recent_decisions": [],
            "signal_deep_dives": {},
            "top_sources_today": [],
            "decision_quality": {
                "status": "unavailable_current_signals",
                "reason": reason,
            },
            "freshness": {
                "status": "unavailable",
                "generated_at": generated_at,
                "source_file": source_file,
                "reason": reason,
                **(stale_metadata or {}),
            },
        }

    def generate_explainability_json(self) -> Optional[Path]:
        """Generate current or explicitly unavailable explainability data."""
        try:
            from src.dashboard.explainability import build_explainability_from_signals_data

            source_dir = DATA_DIR / "explainability"
            target_dir = PUBLIC_DIR / "explainability"
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / "explainability_latest.json"
            generated_at = datetime.now().isoformat()
            stale_metadata = self._latest_stale_explainability_metadata(source_dir)
            signals_path = PUBLIC_DIR / "signals.json"

            if signals_path.exists():
                signals_data = json.loads(signals_path.read_text())
                payload = build_explainability_from_signals_data(
                    signals_data,
                    timestamp=generated_at,
                )
                has_current_decision = payload.get("latest_decision") is not None
                payload["freshness"] = {
                    "status": "current" if has_current_decision else "unavailable",
                    "generated_at": generated_at,
                    "source_file": signals_path.name,
                    "analysis_date": payload.get("analysis_date"),
                    "latest_decision_timestamp": (
                        payload.get("latest_decision") or {}
                    ).get("timestamp"),
                    **stale_metadata,
                }
                if not has_current_decision:
                    payload = self._build_unavailable_explainability_payload(
                        generated_at=generated_at,
                        reason="Current signals.json did not contain ensemble explainability inputs.",
                        source_file=signals_path.name,
                        analysis_date=payload.get("analysis_date"),
                        stale_metadata=stale_metadata,
                    )
                save_results_json(payload, output_path=str(target))
                return target

            payload = self._build_unavailable_explainability_payload(
                generated_at=generated_at,
                reason="Current signals.json was not available for explainability generation.",
                stale_metadata=stale_metadata,
            )
            save_results_json(payload, output_path=str(target))
            return target

        except (OSError, json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning("Failed to generate explainability data: %s", e)
            return None

    def _load_risk_decomposition_signal_section(self) -> Optional[Dict[str, Any]]:
        """Embed risk decomposition into signals.json for optional staleness TTL.

        Prefer computing live; fall back to the public sidecar when present so
        the section is not left missing (None → optional unavailable forever).
        """
        now_ts = datetime.now(timezone.utc).isoformat()

        def _stamp(payload: Dict[str, Any]) -> Dict[str, Any]:
            payload.setdefault("generated_at", now_ts)
            payload.setdefault("timestamp", payload.get("generated_at") or now_ts)
            return payload

        try:
            from src.monitor.risk_decomposition import decompose_portfolio

            result = decompose_portfolio(weights=BASE_ALLOCATION)
            return _stamp(result.to_dict())
        except Exception as exc:  # noqa: BLE001 — optional section
            logger.debug("Live risk_decomposition embed skipped: %s", exc)

        path = PUBLIC_DIR / "risk_decomposition.json"
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return None
            # Explicit unavailable/error sidecars stay unavailable for operators.
            if data.get("status") in {"unavailable", "error"} or "error" in data:
                return _stamp(data)
            # Successful decompose payloads should not look unavailable.
            data.pop("status", None)
            return _stamp(data)
        except (OSError, json.JSONDecodeError, TypeError):
            return None

    def generate_risk_decomposition_json(self) -> Path:
        """Generate risk factor decomposition for dashboard."""
        output_path = PUBLIC_DIR / "risk_decomposition.json"

        try:
            from src.monitor.risk_decomposition import decompose_portfolio

            result = decompose_portfolio(weights=BASE_ALLOCATION)
            data = result.to_dict()
            data["generated_at"] = datetime.now(timezone.utc).isoformat()
            data = _stamp_generator_git_sha(data)
            save_results_json(data, output_path=str(output_path))
            return output_path

        except ImportError:
            logger.warning("scipy not available — skipping risk decomposition")
            fallback = _stamp_generator_git_sha({
                "status": "unavailable",
                "reason": "scipy not installed",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })
            save_results_json(fallback, output_path=str(output_path))
            return output_path

        except (OSError, ValueError, TypeError, RuntimeError) as e:
            logger.warning("Risk decomposition failed: %s", e)
            fallback = _stamp_generator_git_sha({
                "status": "error",
                "reason": str(e),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })
            save_results_json(fallback, output_path=str(output_path))
            return output_path

    def run(self):
        """Generate all dashboard files."""
        logger.info("Generating dashboard data...")

        # Freeze manifest for config drift detection
        try:
            from src.monitor.freeze_manifest import create_manifest, save_manifest, load_manifest, diff_manifests
            current = create_manifest()
            baseline = load_manifest()
            if baseline:
                drift = diff_manifests(baseline, current)
                if drift["drifted"]:
                    logger.warning("Config drift detected: git_changed=%s config_drift=%d files_added=%d files_modified=%d",
                                   drift["git_changed"],
                                   len(drift["config_drift"]),
                                   len(drift["file_changes"]["added"]),
                                   len(drift["file_changes"]["modified"]))
            save_manifest(current)
        except Exception as e:
            logger.warning("Freeze manifest failed: %s", e)

        try:
            performance_path = self.generate_performance_json()
            health_path = self.generate_health_json()
            health_payload = self._load_json_file(health_path)
            paths = [
                performance_path,
                health_path,
                self.generate_signals_json(),
                self.generate_stats_json(),
                self.generate_alerts_json(health=health_payload),
                self.generate_incidents_json(),
                self.generate_analytics_json(),
                self.generate_graduation_json(),
                self.generate_adaptive_sizing_json(),
                self.generate_vixy_hedge_json(),
                self.generate_black_litterman_json(),
                self.generate_turnover_validator_json(),
                self.generate_regime_gate_json(),
                self.generate_tsmom_json(),
                self.generate_cross_asset_rv_json(),
                self.generate_explainability_json(),
                self.generate_risk_decomposition_json(),
            ]

            # Overlay dashboard (separate path — may fail gracefully)
            overlay_path = self.generate_overlay_json()
            if overlay_path:
                paths.append(overlay_path)

            labs_registry_path = None
            try:
                from src.research.experiment_registry import save_labs_registry

                labs_registry_path = save_labs_registry(
                    data_dirs=(DATA_DIR,),
                    public_dir=PUBLIC_DIR,
                    project_root=DATA_DIR.parent if DATA_DIR.name == "data" else DATA_DIR,
                )
                if labs_registry_path:
                    paths.append(labs_registry_path)
            except (ImportError, ValueError, OSError, TypeError) as e:
                logger.warning("Labs registry generation skipped: %s", e)

            if labs_registry_path:
                try:
                    from src.research.experiment_scorecard import save_labs_scorecards

                    labs_scorecard_path = save_labs_scorecards(
                        registry_path=labs_registry_path,
                        public_dir=PUBLIC_DIR,
                    )
                    if labs_scorecard_path:
                        paths.append(labs_scorecard_path)
                except (ImportError, ValueError, OSError, TypeError) as e:
                    logger.warning("Labs scorecard generation skipped: %s", e)

            try:
                from src.research.experiment_replay_batch import publish_labs_replays

                labs_replay_path = publish_labs_replays(
                    data_dir=DATA_DIR,
                    public_dir=PUBLIC_DIR,
                    project_root=DATA_DIR.parent if DATA_DIR.name == "data" else DATA_DIR,
                )
                if labs_replay_path:
                    paths.append(labs_replay_path)
            except (ImportError, ValueError, OSError, TypeError) as e:
                logger.warning("Labs replay report generation skipped: %s", e)

            try:
                from src.research.labs_validation_report import save_labs_validation_report

                labs_validation_path = save_labs_validation_report(
                    data_dirs=(DATA_DIR,),
                    public_dir=PUBLIC_DIR,
                    project_root=DATA_DIR.parent if DATA_DIR.name == "data" else DATA_DIR,
                )
                if labs_validation_path:
                    paths.append(labs_validation_path)
            except (ImportError, ValueError, OSError, TypeError) as e:
                logger.warning("Labs validation report generation skipped: %s", e)

            try:
                from src.monitor.decision_registry import (
                    publish_decision_registry_json,
                    sync_labs_registry_experiments,
                )
                from src.research.experiment_registry import LABS_REGISTRY_FILENAME

                labs_json = PUBLIC_DIR / LABS_REGISTRY_FILENAME
                if labs_json.exists():
                    sync_labs_registry_experiments(labs_json)
                decision_registry_path = publish_decision_registry_json(public_dir=PUBLIC_DIR)
                paths.append(decision_registry_path)
            except (ImportError, ValueError, OSError, TypeError) as e:
                logger.warning("Decision registry JSON generation skipped: %s", e)

            for p in paths:
                if p:
                    logger.info("Generated: %s", p)

            # Create a versioned public-data manifest while keeping files[] for
            # existing dashboard consumers.
            index = build_public_data_index(paths, public_dir=PUBLIC_DIR)
            save_results_json(index, output_path=str(PUBLIC_DIR / "index.json"))
            _mirror_public_data_contract_files_to_dist(PUBLIC_DIR)
        finally:
            self.close()

        logger.info("Dashboard generation complete")

def refresh_graduation_dual_surfaces(
    *,
    public_dir: Path | None = None,
    data_dir: Path | None = None,
    paper_trading_builder=None,
    display_value=None,
) -> Optional[Path]:
    """Compute graduation once; write public graduation.json + private report.

    Batch BP: single compute path so private/public readiness and CB criterion
    cannot invent different consecutive_ok values. Safe to call from health
    when ``.circuit_breaker.json`` consecutive_ok changes (no DB required).
    """
    try:
        from src.strategy.graduation_checklist import GraduationChecklist
        from src.paths import DATA_DIR as _DEFAULT_DATA, PUBLIC_DATA_DIR as _DEFAULT_PUB

        pub = Path(public_dir) if public_dir is not None else Path(_DEFAULT_PUB)
        # DATA_DIR for checklist is module-level; callers monkeypatch as needed.
        _ = data_dir  # reserved for future path injection

        checklist = GraduationChecklist()
        state = checklist._load_state()
        results = checklist.check(state)
        score = checklist.readiness_score(results)
        is_ready = checklist.is_graduation_ready(results)

        _display = display_value or (
            lambda v: str(v) if v is not None else ""
        )
        criteria_progress = []
        for name, result in results.items():
            criteria_progress.append({
                "name": name,
                "passed": result.passed,
                "value": result.value,
                "required": result.required,
                "description": result.description,
                "id": name,
                "label": result.description or name,
                "threshold": _display(result.required),
            })

        trading_days_result = results.get("min_trading_days")
        n_days = trading_days_result.value if trading_days_result is not None else 0
        min_trading_days = (
            trading_days_result.required
            if trading_days_result is not None
            else checklist.criteria["min_trading_days"]["value"]
        )
        manual_approval = results.get("manual_approval")
        if paper_trading_builder is not None:
            paper_trading = paper_trading_builder(
                state,
                days_elapsed=n_days,
                days_required=min_trading_days,
            )
        else:
            paper_trading = {
                "days_elapsed": n_days,
                "days_required": min_trading_days,
            }

        cb_result = results.get("circuit_breaker_confidence")
        graduation_data = _stamp_generator_git_sha({
            "readiness_score": score,
            "is_graduation_ready": is_ready,
            "manual_approval_required": True,
            "manual_approval_pending": not bool(
                manual_approval and manual_approval.passed
            ),
            "trading_days": n_days,
            "min_trading_days": min_trading_days,
            "criteria_met": sum(
                1 for n, r in results.items() if n != "manual_approval" and r.passed
            ),
            "criteria_total": sum(1 for n in results if n != "manual_approval"),
            "criteria": criteria_progress,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "readiness_pct": score,
            "eligible": is_ready,
            "paper_trading": paper_trading,
            "circuit_breaker_ssot": ".circuit_breaker.json",
            "circuit_breaker_consecutive_ok": (
                cb_result.value if cb_result is not None else None
            ),
        })

        out_path = pub / "graduation.json"
        save_results_json(graduation_data, output_path=str(out_path))
        try:
            checklist.save_report(results)
        except Exception as priv_exc:  # noqa: BLE001
            logger.warning("Private graduation report dual-write failed: %s", priv_exc)
        return out_path
    except Exception as e:  # noqa: BLE001 — health path must not crash
        try:
            _log_signal_error("graduation", e)
        except Exception:  # noqa: BLE001
            logger.warning("graduation dual-surface refresh failed: %s", e)
        return None

if __name__ == "__main__":
    from src.utils.log_config import configure_logging
    configure_logging()
    with DashboardGenerator() as gen:
        gen.run()
