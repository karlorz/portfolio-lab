"""Alternative-data projections extracted from ``src.dashboard.generator``.

Alt-data legacy component keys, the alternative-data signal projection,
producer-timestamp loading, public projection refresh and the predictive
FRED-MD macro gate moved here by Item 10 (2026-08-12). ``generator.py``
re-exports every name below (``signal_section_builder`` resolves
``generator.project_alternative_data_signal`` / ``generator._is_predictive_fred_macro``).
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src.paths import DATA_DIR, PUBLIC_DATA_DIR

# generator.py defines PUBLIC_DIR = PUBLIC_DATA_DIR; alias here so the moved
# bodies stay byte-identical.
PUBLIC_DIR = PUBLIC_DATA_DIR

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
    # Task 2B: canonical SPY-facing value at the projection boundary — never
    # recompute a hidden second policy in the IC monitor. The composite score
    # is the producer's SPY-facing score; bound it to [-1, 1] for staging.
    raw_composite = raw.get("composite_score")
    try:
        spy_value = max(-1.0, min(1.0, float(raw_composite))) if raw_composite is not None else None
    except (TypeError, ValueError):
        spy_value = None
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
        "composite_score": raw_composite,
        "spy_value": spy_value,
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
    from src.dashboard import generator as _generator
    now_utc = _generator.datetime.now(_generator.timezone.utc).isoformat()
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
    # Resolve via generator (Item 9 moved the helper to provenance.py; keep
    # the generator-module patch seam consistent with the other deferrals).
    from src.dashboard import generator as _generator

    _generator._apply_partial_patch_git_sha_honesty(
        signals, patch_source="bounded_alt_data_refresh"
    )
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
