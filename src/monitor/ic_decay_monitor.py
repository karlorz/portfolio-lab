"""
Rank-correlation decay monitor for signal quality tracking.

Tracks per-signal time-series Spearman rank correlation between signal
predictions and actual forward returns over rolling windows. Detects when a signal's
predictive power is degrading, which is a leading indicator of alpha
decay that SPC monitoring alone cannot catch.

Complements:
- SPC monitor: catches distribution shifts in signal *values*
- Staleness detection: catches timing/data freshness issues
- This module: catches decay in signal *predictive quality*

Usage:
    monitor = ICMonitor()
    monitor.record("alternative_data", prediction=0.3, actual_return=0.005)
    monitor.record("alternative_data", prediction=-0.1, actual_return=-0.002)
    decay_report = monitor.compute_decay()
    # decay_report["alternative_data"]["ic_rolling"] = 0.82
    # decay_report["alternative_data"]["ic_trend"] = "stable"

Environment variables
---------------------
IC_MONITOR_WINDOW : int
    Rolling window size for IC computation (default: 60)
IC_DECAY_THRESHOLD : float
    IC below this triggers "decaying" status (default: 0.05)
IC_STABLE_MIN : float
    IC above this is considered "stable" (default: 0.10)
IC_TREND_WINDOW : int
    Window for IC trend computation (default: 20)
"""

import json
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

import logging

from src.paths import DATA_DIR

logger = logging.getLogger(__name__)

__all__ = [
    "ICMonitor",
    "build_ic_decay_summary",
    "compute_ic_decay_report",
    "ic_control_projection",
    "advisory_factor_half_life_table",
    "ADVISORY_FACTOR_HALF_LIFE_DAYS",
    "IC_EVALUATION_CONTRACTS",
    "IC_EVALUATION_CONTRACT_VERSION",
    "IC_OBSERVATION_METADATA_VERSION",
]

# Configurable via environment variables
IC_WINDOW_SIZE = int(os.environ.get("IC_MONITOR_WINDOW", "60"))
IC_DECAY_THRESHOLD = float(os.environ.get("IC_DECAY_THRESHOLD", "0.05"))
IC_STABLE_MIN = float(os.environ.get("IC_STABLE_MIN", "0.10"))
IC_TREND_WINDOW = int(os.environ.get("IC_TREND_WINDOW", "20"))
# Spearman IC with n≈5–10 is extremely noisy; do not escalate kill on thin history.
IC_MIN_OBS_FOR_STATUS = int(os.environ.get("IC_MIN_OBS_FOR_STATUS", "20"))
IC_STATE_PATH = DATA_DIR / "ic_monitor_state.json"

IC_EVALUATION_CONTRACT_VERSION = "ic-evaluation-contract/v2"
IC_OBSERVATION_METADATA_VERSION = "ic-observation-metadata/v2"
IC_STATE_SCHEMA_VERSION = "ic-monitor-state/v2"
IC_ACTUAL_METRIC_AXIS = "time_series_rank_correlation"
IC_ACTUAL_METRIC_KIND = "correlation"
IC_METRIC_AXES = {
    "time_series_rank_correlation",
    "cross_sectional_ic",
    "calibration_proper_score",
}
IC_METRIC_KINDS = {"correlation", "calibration_proper_score"}

# These contracts describe the intended evaluation, while the current resolver
# truthfully records that it uses one shared SPY forward return.  Keeping the
# two concepts separate prevents a mechanically computed coefficient from being
# mistaken for aligned cross-sectional IC or calibrated significance.
IC_EVALUATION_CONTRACTS: Dict[str, Dict[str, Any]] = {
    "ensemble_equity": {
        "contract_version": IC_EVALUATION_CONTRACT_VERSION,
        "intended_metric_axis": IC_ACTUAL_METRIC_AXIS,
        "intended_metric_kind": IC_ACTUAL_METRIC_KIND,
        "target_asset": "SPY",
        "target_basket": None,
        "intended_horizon_sessions": 1,
        "prediction_field": "ensemble_voting.equity_bias",
        "prediction_transform": "identity",
        "observed_prediction_field": "ensemble_voting.equity_bias",
        "observed_prediction_transform": "identity",
        "declared_alignment_status": "provisional",
        "alignment_reason": "legacy_rows_missing_alignment_metadata",
    },
    "ensemble_gold": {
        "contract_version": IC_EVALUATION_CONTRACT_VERSION,
        "intended_metric_axis": IC_ACTUAL_METRIC_AXIS,
        "intended_metric_kind": IC_ACTUAL_METRIC_KIND,
        "target_asset": "GLD",
        "target_basket": None,
        "intended_horizon_sessions": 1,
        "prediction_field": "ensemble_voting.gold_bias",
        "prediction_transform": "identity",
        "observed_prediction_field": "ensemble_voting.gold_bias",
        "observed_prediction_transform": "identity",
        "declared_alignment_status": "provisional",
        # Truthful reason: staging/resolution are per-asset (GLD) and rows are
        # metadata-complete; alignment is computed dynamically from rows below
        # (the stale "actual_target_spy_expected_gld" reason predates the
        # per-asset resolution era — see MAIN-ITEM-1 s1).
        "alignment_reason": "per_asset_resolution_active_awaiting_dynamic_alignment",
    },
    "ensemble_duration": {
        "contract_version": IC_EVALUATION_CONTRACT_VERSION,
        "intended_metric_axis": IC_ACTUAL_METRIC_AXIS,
        "intended_metric_kind": IC_ACTUAL_METRIC_KIND,
        "target_asset": "TLT",
        "target_basket": None,
        "intended_horizon_sessions": 1,
        "prediction_field": "ensemble_voting.duration_bias",
        "prediction_transform": "identity",
        "observed_prediction_field": "ensemble_voting.duration_bias",
        "observed_prediction_transform": "identity",
        "declared_alignment_status": "provisional",
        # Truthful reason: staging/resolution are per-asset (TLT) and rows are
        # metadata-complete; alignment is computed dynamically from rows below.
        "alignment_reason": "per_asset_resolution_active_awaiting_dynamic_alignment",
    },
    "ensemble_consensus": {
        "contract_version": IC_EVALUATION_CONTRACT_VERSION,
        "intended_metric_axis": "cross_sectional_ic",
        "intended_metric_kind": "correlation",
        "target_asset": None,
        "target_basket": "SPY/GLD/TLT",
        "intended_horizon_sessions": 1,
        "prediction_field": "ensemble_voting.per_sleeve_bias_vector",
        "prediction_transform": "identity",
        "observed_prediction_field": "ensemble_voting.weighted_consensus",
        "observed_prediction_transform": "identity",
        "declared_alignment_status": "ambiguous",
        "alignment_reason": "mixed_asset_consensus_resolved_against_single_spy_return",
    },
    "alternative_data": {
        "contract_version": IC_EVALUATION_CONTRACT_VERSION,
        "intended_metric_axis": IC_ACTUAL_METRIC_AXIS,
        "intended_metric_kind": IC_ACTUAL_METRIC_KIND,
        "target_asset": "SPY",
        "target_basket": None,
        "intended_horizon_sessions": 1,
        "prediction_field": "alternative_data.spy_value",
        # Aspirational canonical_spy_polarity transform is UNIMPLEMENTED
        # (grep = 0 impls); the pipeline stages raw spy_value — identity is the
        # truthful contract (MAIN-ITEM-1 s3).
        "prediction_transform": "identity",
        "observed_prediction_field": "alternative_data.spy_value",
        "observed_prediction_transform": "identity",
        "declared_alignment_status": "provisional",
        "alignment_reason": "legacy_rows_archived_reaccumulating_under_corrected_contract",
    },
    "behavioral_sentiment": {
        "contract_version": IC_EVALUATION_CONTRACT_VERSION,
        "intended_metric_axis": IC_ACTUAL_METRIC_AXIS,
        "intended_metric_kind": IC_ACTUAL_METRIC_KIND,
        "target_asset": "SPY",
        "target_basket": None,
        "intended_horizon_sessions": 5,
        "prediction_field": "behavioral_sentiment.spy_value",
        # Aspirational canonical_spy_projection transform is UNIMPLEMENTED
        # (grep = 0 impls); the pipeline stages the clamped equity_shift_pct/5
        # value directly — identity is the truthful contract (MAIN-ITEM-1 s3).
        "prediction_transform": "identity",
        "observed_prediction_field": "behavioral_sentiment.spy_value",
        "observed_prediction_transform": "identity",
        "declared_alignment_status": "provisional",
        "alignment_reason": "legacy_rows_archived_reaccumulating_under_corrected_contract",
    },
    "factor_rotation": {
        "contract_version": IC_EVALUATION_CONTRACT_VERSION,
        "intended_metric_axis": IC_ACTUAL_METRIC_AXIS,
        "intended_metric_kind": IC_ACTUAL_METRIC_KIND,
        "target_asset": None,
        "target_basket": "selected_factor_etfs",
        "intended_horizon_sessions": None,
        "prediction_field": "factor_rotation.signal_strength",
        "prediction_transform": "identity",
        "observed_prediction_field": "factor_rotation.signal_strength",
        "observed_prediction_transform": "identity",
        "declared_alignment_status": "ambiguous",
        "alignment_reason": "selected_factor_return_target_and_horizon_undeclared",
    },
    "fred_macro": {
        "contract_version": IC_EVALUATION_CONTRACT_VERSION,
        "intended_metric_axis": "calibration_proper_score",
        "intended_metric_kind": "calibration_proper_score",
        "target_asset": None,
        "target_basket": "macro_regime_outcomes",
        "intended_horizon_sessions": None,
        "prediction_field": "fred_macro.confidence",
        "prediction_transform": "probability_confidence",
        "observed_prediction_field": "fred_macro.confidence",
        "observed_prediction_transform": "identity",
        "declared_alignment_status": "metric_mismatch",
        "alignment_reason": "unsigned_confidence_requires_calibration_not_signed_return_correlation",
    },
}

_OBSERVATION_METADATA_FIELDS = {
    "prediction_date",
    "realized_start_date",
    "resolved_date",
    "target_asset",
    "intended_horizon_sessions",
    "realized_horizon_sessions",
    "prediction_field",
    "prediction_transform",
    "metric_axis",
    "metric_kind",
    "contract_version",
}

_OBSERVATION_CONTRACT_DEFAULTS = {
    "metric_axis": IC_ACTUAL_METRIC_AXIS,
    "metric_kind": IC_ACTUAL_METRIC_KIND,
    "contract_version": IC_OBSERVATION_METADATA_VERSION,
}


def _normalize_observation_metadata(
    metadata: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return a JSON-safe, bounded observation-metadata record."""
    if not isinstance(metadata, Mapping):
        return None
    normalized: Dict[str, Any] = {}
    for key in _OBSERVATION_METADATA_FIELDS:
        value = metadata.get(key)
        if value is None:
            continue
        if key in {"intended_horizon_sessions", "realized_horizon_sessions"}:
            try:
                value = max(0, int(value))
            except (TypeError, ValueError):
                continue
        elif key in {
            "prediction_date",
            "realized_start_date",
            "resolved_date",
            "target_asset",
            "prediction_field",
            "prediction_transform",
            "metric_axis",
            "metric_kind",
            "contract_version",
        }:
            value = str(value).strip()
            if not value:
                continue
            if key == "metric_axis" and value not in IC_METRIC_AXES:
                continue
            if key == "metric_kind" and value not in IC_METRIC_KINDS:
                continue
            if (
                key == "contract_version"
                and value != IC_OBSERVATION_METADATA_VERSION
            ):
                continue
        normalized[key] = value
    return normalized or None

# Advisory factor / sleeve half-lives (trading days) from multi-market IC decay
# literature (Flint/Vermaak-style summaries via Alpha Architect / Quantpedia).
# NOT live-authoritative — operators only; rebalance cadence is still governed by
# signals.json.target_allocations + smart-rebalance cost budget.
ADVISORY_FACTOR_HALF_LIFE_DAYS: Dict[str, Dict[str, Any]] = {
    "investment": {
        "half_life_days": 21,
        "suggested_rebalance_days": 21,
        "note": "fastest equity-factor decay; ~1 month optimal",
    },
    "momentum": {
        "half_life_days": 63,
        "suggested_rebalance_days": 63,
        "note": "~3 months typical equity momentum half-life band",
    },
    "value": {
        "half_life_days": 84,
        "suggested_rebalance_days": 84,
        "note": "longest persistence; ~3–4 months rebalance",
    },
    "quality": {
        "half_life_days": 105,
        "suggested_rebalance_days": 105,
        "note": "~4–5 months optimal in global studies",
    },
    "low_volatility": {
        "half_life_days": 126,
        "suggested_rebalance_days": 126,
        "note": "~5–6 months; slow decay sleeve",
    },
    "strategic_spy_gld_tlt": {
        "half_life_days": None,
        "suggested_rebalance_days": 252,
        "note": "champion book risk control; annual or ±5% band, cost-aware",
    },
}


def advisory_factor_half_life_table() -> Dict[str, Any]:
    """Public advisory payload for IC half-life → rebalance cadence mapping."""
    return {
        "role": "advisory",
        "live_authoritative": False,
        "unit": "trading_days",
        "source": "literature_defaults_not_fitted_to_lab_ic",
        "sleeves": dict(ADVISORY_FACTOR_HALF_LIFE_DAYS),
        "disclosure": (
            "Half-lives are literature defaults for operator cadence design; "
            "they do not override signals.json.target_allocations or order_router."
        ),
    }



def _average_midranks(values: np.ndarray) -> np.ndarray:
    """Return deterministic zero-based average ranks, including ties."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman_rank_correlation(
    x: List[float], y: List[float]
) -> Optional[float]:
    """Compute Spearman rank correlation between two arrays.

    Returns ``None`` if either array has zero variance or insufficient data.
    """
    if len(x) < 5 or len(y) < 5:
        return None

    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)

    # Remove NaN/inf
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]

    if len(x_arr) < 5:
        return None

    # Zero-variance check on original values — argsort assigns unique ranks
    # to identical values, masking the lack of real variation
    if np.ptp(x_arr) < 1e-10 or np.ptp(y_arr) < 1e-10:
        return None

    # Project policy: ties receive average midranks.  Nested argsort would give
    # identical values arbitrary distinct ranks based on their input order.
    x_rank = _average_midranks(x_arr)
    y_rank = _average_midranks(y_arr)

    # Pearson correlation of ranks
    x_mean = x_rank.mean()
    y_mean = y_rank.mean()

    x_dev = x_rank - x_mean
    y_dev = y_rank - y_mean

    num = (x_dev * y_dev).sum()
    den = np.sqrt((x_dev ** 2).sum() * (y_dev ** 2).sum())

    if den < 1e-10:
        return None

    return float(num / den)


class ICMonitor:
    """Track per-signal Information Coefficient over rolling windows.

    For each signal, stores a rolling window of (prediction, actual_return)
    pairs and computes IC as the Spearman rank correlation between them.
    Tracks IC trend to detect decay.

    Attributes:
        window_size: Number of recent observations to include in IC.
        decay_threshold: IC below this triggers "decaying" status.
        stable_min: IC above this is considered "stable".
    """

    def __init__(
        self,
        window_size: int = IC_WINDOW_SIZE,
        decay_threshold: float = IC_DECAY_THRESHOLD,
        stable_min: float = IC_STABLE_MIN,
        min_obs_for_status: int = IC_MIN_OBS_FOR_STATUS,
        trend_window: int = IC_TREND_WINDOW,
    ):
        self.window_size = window_size
        self.decay_threshold = decay_threshold
        self.stable_min = stable_min
        self.min_obs_for_status = max(5, int(min_obs_for_status))
        self.trend_window = trend_window

        # Per-signal data: deque of (prediction, actual_return)
        self._data: Dict[str, deque] = {}
        # Optional v2 metadata aligned one-for-one with _data rows.  Legacy
        # state loads as explicit None entries; missing facts are never guessed.
        self._observation_metadata: Dict[str, deque] = {}
        # Staged predictions waiting for forward-return resolution (Task 2B).
        # Per-signal lifecycle: keyed by stable observation identity so each
        # signal keeps its own prediction date / target / horizon and resolves
        # independently and idempotently.
        # entry = {signal, prediction_date, prediction, metadata, identity}
        self._staged: Dict[str, Dict[str, Any]] = {}

    def record(
        self,
        signal_name: str,
        prediction: float,
        actual_return: float,
        observation_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Record a signal prediction and the corresponding actual return.

        Args:
            signal_name: Name of the signal (e.g., "alternative_data").
            prediction: The signal's predicted direction/strength.
            actual_return: The actual forward return that materialized.
        """
        if signal_name not in self._data:
            self._data[signal_name] = deque(maxlen=self.window_size)
        if signal_name not in self._observation_metadata:
            self._observation_metadata[signal_name] = deque(
                [None] * len(self._data[signal_name]),
                maxlen=self.window_size,
            )
        self._data[signal_name].append((prediction, actual_return))
        self._observation_metadata[signal_name].append(
            _normalize_observation_metadata(observation_metadata)
        )

    def stage_predictions(
        self,
        predictions: Dict[str, float],
        date_str: str,
        prediction_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> None:
        """Stage predictions for resolution on the next cron run.

        Per-signal lifecycle (Task 2B): each signal is stored under a stable
        observation identity (signal + prediction date + contract version), so
        re-staging the same cohort is idempotent, different cohorts coexist,
        and each signal resolves independently at its own horizon.
        """
        for signal_name, prediction in predictions.items():
            supplied = (
                prediction_metadata.get(signal_name)
                if isinstance(prediction_metadata, Mapping)
                else None
            )
            contract = IC_EVALUATION_CONTRACTS.get(signal_name, {})
            derived: Dict[str, Any] = {
                **_OBSERVATION_CONTRACT_DEFAULTS,
                "prediction_date": date_str,
                "prediction_field": contract.get("observed_prediction_field"),
                "prediction_transform": contract.get("observed_prediction_transform"),
                "intended_horizon_sessions": contract.get(
                    "intended_horizon_sessions"
                ),
                "target_asset": contract.get("target_asset"),
                "contract_version": contract.get(
                    "contract_version", IC_OBSERVATION_METADATA_VERSION
                ),
            }
            if isinstance(supplied, Mapping):
                derived.update(supplied)
            normalized = _normalize_observation_metadata(derived)
            metadata = normalized if normalized else dict(derived)
            contract_version = str(
                metadata.get("contract_version") or IC_OBSERVATION_METADATA_VERSION
            )
            identity = f"{signal_name}|{date_str}|{contract_version}"
            self._staged[identity] = {
                "signal": signal_name,
                "prediction_date": date_str,
                "prediction": prediction,
                "metadata": metadata,
                "identity": identity,
            }

    def resolve_staged(
        self,
        forward_return: float,
        *,
        resolved_date: Optional[str] = None,
        realized_start_date: Optional[str] = None,
        target_asset: Optional[str] = None,
        realized_horizon_sessions: Optional[int] = None,
    ) -> int:
        """Resolve previously staged predictions with the actual forward return.

        Per-signal lifecycle (Task 2B): only entries whose declared target
        asset matches ``target_asset`` (or that declare no target) and whose
        intended horizon has elapsed (``realized_horizon_sessions >=
        intended_horizon_sessions``) are resolved; everything else stays
        staged for a later run. Resolution is idempotent per identity.
        """
        if not self._staged:
            return 0
        count = 0
        for identity, entry in list(self._staged.items()):
            prediction = entry.get("prediction")
            if prediction is None or not np.isfinite(prediction):
                # Non-finite predictions never resolve; drop them rather than
                # letting a poisoned cohort block the identity forever.
                del self._staged[identity]
                continue
            metadata = dict(entry.get("metadata") or {})
            entry_target = metadata.get("target_asset")
            if entry_target is not None and entry_target != target_asset:
                continue  # different sleeve; leave staged for its own return
            try:
                intended_horizon = (
                    int(metadata.get("intended_horizon_sessions"))
                    if metadata.get("intended_horizon_sessions") is not None
                    else None
                )
            except (TypeError, ValueError):
                intended_horizon = None
            if (
                intended_horizon is not None
                and realized_horizon_sessions is not None
                and realized_horizon_sessions < intended_horizon
            ):
                continue  # not enough sessions elapsed yet; stay staged

            resolved_metadata = dict(_OBSERVATION_CONTRACT_DEFAULTS)
            for key, value in {
                "resolved_date": resolved_date,
                "realized_start_date": realized_start_date,
                "target_asset": target_asset,
                "realized_horizon_sessions": realized_horizon_sessions,
            }.items():
                if value is not None:
                    resolved_metadata[key] = value
            metadata.update(resolved_metadata)
            self.record(
                entry.get("signal"),
                float(prediction),
                forward_return,
                observation_metadata=metadata,
            )
            del self._staged[identity]
            count += 1
        return count

    def has_staged_predictions(self) -> bool:
        """Check if there are unresolved staged predictions."""
        return len(self._staged) > 0

    def get_staged_date(self) -> Optional[str]:
        """Return the earliest staged prediction date, if any."""
        if not self._staged:
            return None
        dates = [
            str(entry.get("prediction_date"))
            for entry in self._staged.values()
            if entry.get("prediction_date")
        ]
        return min(dates) if dates else None

    def get_staged_prediction_count(self) -> int:
        """Return the number of currently unresolved staged predictions."""
        return len(self._staged)

    def get_staged_prediction_names(self) -> List[str]:
        """Return bounded names for the currently staged IC predictions."""
        return sorted(
            {
                str(entry.get("signal"))
                for entry in self._staged.values()
                if str(entry.get("signal") or "").strip()
            }
        )

    def staged_observation_counts(self) -> Dict[str, int]:
        """Per-signal count of currently staged v2 observations."""
        counts: Dict[str, int] = {}
        for entry in self._staged.values():
            signal = str(entry.get("signal") or "").strip()
            if signal:
                counts[signal] = counts.get(signal, 0) + 1
        return counts

    def rebaseline_trigger_state(self) -> Dict[str, Any]:
        """Re-review trigger for incident 8115a9c1 (operator-approved 2026-08-11).

        Due when any signal accumulates ``min_obs_for_status`` staged v2
        observations — the point at which a re-baseline (or a resolution
        decision) can be evidence-based rather than noise-driven. Purely
        informational: never raises alerts or touches the kill switch.
        """
        counts = self.staged_observation_counts()
        max_staged = max(counts.values(), default=0)
        return {
            "due": max_staged >= self.min_obs_for_status,
            "threshold": self.min_obs_for_status,
            "max_staged_observations": max_staged,
            "staged_observations_per_signal": dict(sorted(counts.items())),
            "criterion": "max_staged_observations >= min_obs_for_status",
        }

    def rebaseline(self, archive_path: Optional[Path] = None) -> Path:
        """Archive the current staging epoch and start a fresh accumulation epoch.

        Operator-approved 2026-08-11 (incident 8115a9c1, Option B): re-anchors
        IC measurement at the current v2-contract point so the next report
        measures only observations accumulated from here. The prior epoch —
        staged rows, resolved pairs, and observation metadata — is snapshotted
        to a dated archive file before the monitor state is cleared; nothing is
        lost. This method only re-arms measurement; it does not touch the kill
        switch, thresholds, or the incident (all operator-gated).
        """
        if archive_path is None:
            archive_dir = DATA_DIR / "ic_rebaseline_archives"
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_path = archive_dir / (
                "ic_epoch_archive_"
                + datetime.now(timezone.utc)
                .isoformat(timespec="seconds")
                .replace(":", "-")
                + ".json"
            )
        archive_path = Path(archive_path)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "schema": "ic-rebaseline-archive/v1",
            "archived_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "incident_id": "8115a9c1-a167-4da7-9832-673617dc7de3",
            "staged": list(self._staged.values()),
            "observations": {
                signal: list(rows) for signal, rows in self._data.items()
            },
            "observation_metadata": {
                signal: list(rows)
                for signal, rows in self._observation_metadata.items()
            },
        }
        with open(archive_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)
        self._staged.clear()
        self._data.clear()
        self._observation_metadata.clear()
        logger.info("IC monitor re-baselined: epoch archived to %s", archive_path)
        return archive_path

    def archive_pre_contract_rows(
        self, archive_path: Optional[Path] = None
    ) -> Path:
        """Archive rows that can never align under the corrected contracts.

        One-shot maintenance for MAIN-ITEM-1 s4: rows in alignment-participating
        cohorts (contract ``declared_alignment_status`` in
        {"misaligned", "provisional"} — the fixable class) that can NEVER pass
        the dynamic alignment check are snapshotted to a dated archive file in
        the ic-rebaseline-archive/v1 format and removed from the live state:
        (a) ``None`` rows (missing metadata entirely — never align), or
        (b) rows whose stamped ``prediction_field`` differs from the contract's
        intended ``prediction_field`` (stamped under an older observed field).

        Cohorts declared "ambiguous"/"metric_mismatch" (ensemble_consensus,
        factor_rotation, fred_macro) never participate in the dynamic
        alignment path, so their legacy rows are untouched. The ensemble trio
        (equity/gold/duration) stamps match their contracts — untouched.
        Idempotent: a re-run finds no matching rows and archives 0.

        Mirrors ``rebaseline()`` snapshot semantics ("nothing is lost"):
        archived pairs + metadata rows are preserved on disk; only the
        matching rows are dropped from the live monitor state.
        """
        if archive_path is None:
            archive_dir = DATA_DIR / "ic_rebaseline_archives"
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_path = archive_dir / (
                "ic_pre_contract_archive_"
                + datetime.now(timezone.utc)
                .isoformat(timespec="seconds")
                .replace(":", "-")
                + ".json"
            )
        archive_path = Path(archive_path)
        archive_path.parent.mkdir(parents=True, exist_ok=True)

        archived_observations: Dict[str, list] = {}
        archived_metadata: Dict[str, list] = {}
        for signal_name, contract in IC_EVALUATION_CONTRACTS.items():
            if contract.get("declared_alignment_status") not in {
                "misaligned",
                "provisional",
            }:
                continue
            intended_field = contract.get("prediction_field")
            metadata_rows = list(self._observation_metadata.get(signal_name, ()))
            if not metadata_rows:
                continue
            data_rows = list(self._data.get(signal_name, ()))
            keep_meta: list = []
            keep_data: list = []
            dropped_meta: list = []
            dropped_data: list = []
            for idx, row in enumerate(metadata_rows):
                cannot_align = row is None or (
                    isinstance(row, Mapping)
                    and row.get("prediction_field") != intended_field
                )
                if cannot_align:
                    dropped_meta.append(row)
                    if idx < len(data_rows):
                        dropped_data.append(data_rows[idx])
                else:
                    keep_meta.append(row)
                    if idx < len(data_rows):
                        keep_data.append(data_rows[idx])
            if dropped_meta:
                archived_metadata[signal_name] = dropped_meta
                archived_observations[signal_name] = dropped_data
                self._observation_metadata[signal_name] = deque(keep_meta)
                self._data[signal_name] = deque(keep_data)

        snapshot = {
            "schema": "ic-rebaseline-archive/v1",
            "archive_kind": "pre-contract-rows",
            "archived_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "incident_id": "8115a9c1-a167-4da7-9832-673617dc7de3",
            "staged": [],
            "observations": archived_observations,
            "observation_metadata": archived_metadata,
        }
        with open(archive_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)
        logger.info(
            "IC pre-contract rows archived to %s (%s)",
            archive_path,
            {
                signal: len(rows)
                for signal, rows in archived_metadata.items()
            },
        )
        return archive_path

    def compute_ic(self, signal_name: str) -> Optional[float]:
        """Compute rolling IC for a specific signal.

        Returns None if insufficient data points.
        """
        if signal_name not in self._data or len(self._data[signal_name]) < 5:
            return None

        data = list(self._data[signal_name])
        predictions = [d[0] for d in data]
        actuals = [d[1] for d in data]

        return _spearman_rank_correlation(predictions, actuals)

    def compute_ic_trend(self, signal_name: str) -> str:
        """Determine IC trend for a signal.

        Returns one of: "stable", "decaying", "improving", "unknown".
        """
        if signal_name not in self._data:
            return "unknown"

        data = list(self._data[signal_name])
        if len(data) < self.trend_window * 2:
            return "unknown"

        # Split into recent and earlier halves
        n = len(data)
        recent = data[n // 2:]
        earlier = data[:n // 2]

        predictions_recent = [d[0] for d in recent]
        actuals_recent = [d[1] for d in recent]
        predictions_earlier = [d[0] for d in earlier]
        actuals_earlier = [d[1] for d in earlier]

        ic_recent = _spearman_rank_correlation(predictions_recent, actuals_recent)
        ic_earlier = _spearman_rank_correlation(predictions_earlier, actuals_earlier)

        if ic_recent is None or ic_earlier is None:
            return "unknown"

        diff = ic_recent - ic_earlier

        if ic_recent < self.decay_threshold:
            return "decaying"
        elif diff > 0.05:
            return "improving"
        elif ic_recent > self.stable_min:
            return "stable"
        else:
            return "decaying"

    def compute_decay_report(self) -> Dict[str, Dict]:
        """Generate a decay report for all tracked signals.

        Returns dict mapping signal_name -> {
            "ic_rolling": float,
            "ic_trend": str,
            "observations": int,
            "status": str,  # "healthy", "warning", "critical"
        }
        """
        report = {}
        for signal_name in self._data:
            ic = self.compute_ic(signal_name)
            trend = self.compute_ic_trend(signal_name)
            n_obs = len(self._data[signal_name])

            # Thin resolved history produces unstable Spearman IC; do not escalate
            # warning/critical (and thus kill HALT) until min_obs_for_status.
            if ic is None or n_obs < self.min_obs_for_status:
                status = "insufficient_data"
            elif ic < self.decay_threshold:
                status = "critical"
            elif ic < self.stable_min or trend == "decaying":
                status = "warning"
            else:
                status = "healthy"

            contract = dict(IC_EVALUATION_CONTRACTS.get(signal_name, {
                "contract_version": IC_EVALUATION_CONTRACT_VERSION,
                "intended_metric_axis": "time_series_rank_correlation",
                "intended_metric_kind": "correlation",
                "target_asset": None,
                "target_basket": None,
                "intended_horizon_sessions": None,
                "prediction_field": None,
                "prediction_transform": None,
                "declared_alignment_status": "undeclared",
                "alignment_reason": "evaluation_contract_missing",
            }))
            declared_alignment = str(
                contract.get("declared_alignment_status") or "undeclared"
            )
            metadata_rows = list(self._observation_metadata.get(signal_name, ()))
            metadata_complete = (
                len(metadata_rows) == n_obs
                and bool(metadata_rows)
                and all(
                    isinstance(metadata_row, Mapping)
                    and _OBSERVATION_METADATA_FIELDS.issubset(metadata_row)
                    for metadata_row in metadata_rows
                )
            )
            if declared_alignment == "provisional" and metadata_rows:
                aligned = metadata_complete and all(
                    row.get("target_asset") == contract.get("target_asset")
                    and row.get("realized_horizon_sessions")
                    >= contract.get("intended_horizon_sessions")
                    and row.get("prediction_field")
                    == contract.get("prediction_field")
                    and row.get("prediction_transform")
                    == contract.get("prediction_transform")
                    and row.get("metric_axis")
                    == contract.get("intended_metric_axis")
                    and row.get("metric_kind")
                    == contract.get("intended_metric_kind")
                    and row.get("contract_version")
                    == IC_OBSERVATION_METADATA_VERSION
                    for row in metadata_rows
                    if isinstance(row, Mapping)
                )
                if aligned:
                    declared_alignment = "aligned"
                    alignment_reason = "metadata_complete_and_contract_aligned"
                elif metadata_complete:
                    declared_alignment = "misaligned"
                    alignment_reason = "observation_metadata_conflicts_with_contract"
                else:
                    alignment_reason = (
                        "observation_metadata_incomplete"
                        if any(isinstance(row, Mapping) for row in metadata_rows)
                        else "legacy_rows_missing_alignment_metadata"
                    )
            else:
                alignment_reason = str(
                    contract.get("alignment_reason") or "evaluation_contract_missing"
                )

            # Control eligibility (Task 2A): only fully contract-aligned rows
            # with complete v2 metadata may drive halt-authoritative IC control
            # decisions. Descriptive status is preserved for operators.
            control_eligible = bool(
                metadata_complete
                and metadata_rows
                and declared_alignment == "aligned"
            )

            if declared_alignment in {"misaligned", "ambiguous"}:
                inference_reason = "label_alignment_mismatch"
            elif declared_alignment == "metric_mismatch":
                inference_reason = "metric_contract_mismatch"
            elif declared_alignment == "undeclared":
                inference_reason = "evaluation_contract_missing"
            elif not metadata_rows or any(row is None for row in metadata_rows):
                inference_reason = "legacy_rows_missing_alignment_metadata"
            elif not metadata_complete:
                inference_reason = "observation_metadata_incomplete"
            else:
                inference_reason = "dependence_not_characterized"

            public_contract = {
                key: value
                for key, value in contract.items()
                if key not in {
                    "observed_prediction_field",
                    "observed_prediction_transform",
                    "declared_alignment_status",
                    "alignment_reason",
                }
            }
            row: Dict[str, Any] = {
                "ic_rolling": round(ic, 4) if ic is not None else None,
                "ic_trend": trend,
                "observations": n_obs,
                "status": status,
                "min_obs_for_status": self.min_obs_for_status,
                "metric_axis": IC_ACTUAL_METRIC_AXIS,
                "metric_kind": IC_ACTUAL_METRIC_KIND,
                "estimate_kind": "descriptive",
                "alignment_status": declared_alignment,
                "alignment_reason": alignment_reason,
                "inference_status": "unavailable",
                "inference_reason": inference_reason,
                "observation_count": n_obs,
                "observation_unit": "pairs",
                "contract_version": IC_EVALUATION_CONTRACT_VERSION,
                "evaluation_contract": public_contract,
                # Control eligibility (Task 2A): derived strictly from complete
                # contract alignment — never from coefficient magnitude. A row
                # keeps its descriptive status but cannot drive halt-authoritative
                # IC control alerts unless every observation matches the declared
                # v2 contract (axis/kind, target/basket, horizon, field/transform).
                "control_eligible": control_eligible,
                "control_status": "eligible" if control_eligible else "ineligible",
                "control_ineligibility_reason": (
                    None if control_eligible else inference_reason
                ),
            }
            latest_metadata = next(
                (
                    dict(item)
                    for item in reversed(metadata_rows)
                    if isinstance(item, Mapping)
                ),
                None,
            )
            if latest_metadata:
                row["latest_observation_metadata"] = latest_metadata
            report[signal_name] = row

        return report

    def get_signals_needing_attention(self) -> List[str]:
        """Return signal names with 'warning' or 'critical' IC status."""
        report = self.compute_decay_report()
        return [
            name for name, data in report.items()
            if data["status"] in ("warning", "critical")
        ]

    def save_state(self, path: Optional[Path] = None) -> Path:
        """Save current monitor state to JSON for persistence across runs."""
        if path is None:
            path = IC_STATE_PATH
        state: Dict[str, object] = {}
        for signal_name, data in self._data.items():
            state[signal_name] = list(data)
        if self._staged:
            state["__staged_v2__"] = list(self._staged.values())
        if self._observation_metadata:
            state["__state_schema_version__"] = IC_STATE_SCHEMA_VERSION
            state["__observation_metadata__"] = {
                signal_name: list(rows)
                for signal_name, rows in self._observation_metadata.items()
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f)
        logger.info("IC monitor state saved: %s (%d signals)", path, len(self._data))
        return path

    def load_state(self, path: Optional[Path] = None) -> None:
        """Load monitor state from JSON."""
        if path is None:
            path = IC_STATE_PATH
        if not path.exists():
            return
        try:
            with open(path) as f:
                state = json.load(f)
            if not isinstance(state, Mapping):
                raise TypeError("IC monitor state must be a JSON object")
            metadata_state = state.get("__observation_metadata__", {})
            loaded_data: Dict[str, deque] = {}
            loaded_metadata: Dict[str, deque] = {}
            loaded_staged: Dict[str, Dict[str, Any]] = {}
            for key, observations in state.items():
                if key == "__staged_v2__":
                    if observations is not None and not isinstance(observations, (list, Mapping)):
                        raise TypeError("staged IC state must be a JSON object")
                    raw_entries = (
                        list(observations)
                        if isinstance(observations, list)
                        else list(observations.values())
                        if observations
                        else []
                    )
                    for entry in raw_entries:
                        if not isinstance(entry, Mapping):
                            continue
                        signal = str(entry.get("signal") or "").strip()
                        if not signal:
                            continue
                        identity = str(
                            entry.get("identity")
                            or f"{signal}|{entry.get('prediction_date')}"
                        )
                        loaded_staged[identity] = {
                            "signal": signal,
                            "prediction_date": entry.get("prediction_date"),
                            "prediction": entry.get("prediction"),
                            "metadata": dict(entry.get("metadata") or {}),
                            "identity": identity,
                        }
                elif key == "__staged__":
                    # Legacy single-slot staging: migrate to per-signal entries
                    # without rewriting historical paired rows.
                    if observations is not None and isinstance(observations, Mapping):
                        legacy_predictions = observations.get("predictions")
                        legacy_metadata = observations.get("prediction_metadata")
                        legacy_date = observations.get("date")
                        if isinstance(legacy_predictions, Mapping):
                            for signal, value in legacy_predictions.items():
                                signal = str(signal).strip()
                                if not signal:
                                    continue
                                contract = IC_EVALUATION_CONTRACTS.get(signal, {})
                                derived = dict(_OBSERVATION_CONTRACT_DEFAULTS)
                                if isinstance(legacy_metadata, Mapping) and isinstance(
                                    legacy_metadata.get(signal), Mapping
                                ):
                                    derived.update(legacy_metadata[signal])
                                derived.setdefault("prediction_date", legacy_date)
                                derived.setdefault(
                                    "prediction_field",
                                    contract.get("observed_prediction_field"),
                                )
                                derived.setdefault(
                                    "prediction_transform",
                                    contract.get("observed_prediction_transform"),
                                )
                                derived.setdefault(
                                    "intended_horizon_sessions",
                                    contract.get("intended_horizon_sessions"),
                                )
                                derived.setdefault(
                                    "target_asset", contract.get("target_asset")
                                )
                                normalized = _normalize_observation_metadata(derived)
                                metadata = normalized if normalized else derived
                                contract_version = str(
                                    metadata.get("contract_version")
                                    or IC_OBSERVATION_METADATA_VERSION
                                )
                                identity = (
                                    f"{signal}|{legacy_date}|{contract_version}"
                                )
                                loaded_staged[identity] = {
                                    "signal": signal,
                                    "prediction_date": legacy_date,
                                    "prediction": value,
                                    "metadata": metadata,
                                    "identity": identity,
                                }
                elif key in {"__state_schema_version__", "__observation_metadata__"}:
                    continue
                else:
                    if not isinstance(observations, list):
                        raise TypeError(f"IC observations for {key} must be a list")
                    observation_rows = list(observations)
                    loaded_data[key] = deque(maxlen=self.window_size)
                    for observation in observation_rows:
                        if not isinstance(observation, (list, tuple)) or len(observation) != 2:
                            raise ValueError(
                                f"IC observation for {key} must be a prediction/return pair"
                            )
                        pred, actual = observation
                        loaded_data[key].append((pred, actual))
                    raw_metadata = (
                        metadata_state.get(key, [])
                        if isinstance(metadata_state, Mapping)
                        else []
                    )
                    metadata_is_aligned = (
                        isinstance(raw_metadata, list)
                        and len(raw_metadata) == len(observation_rows)
                    )
                    normalized_metadata = (
                        [
                            _normalize_observation_metadata(item)
                            for item in raw_metadata
                        ]
                        if metadata_is_aligned
                        else []
                    )
                    data_len = len(loaded_data[key])
                    normalized_metadata = normalized_metadata[-data_len:]
                    if len(normalized_metadata) < data_len:
                        normalized_metadata = (
                            [None] * (data_len - len(normalized_metadata))
                            + normalized_metadata
                        )
                    loaded_metadata[key] = deque(
                        normalized_metadata,
                        maxlen=self.window_size,
                    )
            self._data = loaded_data
            self._observation_metadata = loaded_metadata
            self._staged = loaded_staged
            logger.info("IC monitor state loaded: %d signals", len(self._data))
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
            logger.warning("Failed to load IC monitor state: %s", e)



def _signal_prediction_date_expr(columns: set[str]) -> str:
    """SQL expression for prediction calendar date across schema variants.

    Production SignalHealthTracker uses ``timestamp`` (ISO datetime). Older /
    test fixtures may use ``prediction_date`` only. Prefer date(timestamp) when
    present so distinct pending dates and oldest unresolved are honest.
    """
    if "prediction_date" in columns:
        return "prediction_date"
    if "timestamp" in columns:
        # date() handles ISO 'YYYY-MM-DD…' prefixes; substr fallback for odd values
        return "COALESCE(date(timestamp), substr(timestamp, 1, 10))"
    return "NULL"


def _signal_prediction_backlog(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Row-level pending backlog from SignalHealthTracker table (not staged IC window).

    pending_predictions in the IC report is staged-date count; pending_rows is the
    full unlabeled prediction history operators confuse with near-green pending=6.
    """
    from src.paths import MARKET_DB, sqlite_connect

    path = Path(db_path) if db_path is not None else MARKET_DB
    empty = {
        "pending_rows": 0,
        "pending_dates": 0,
        "oldest_unresolved_date": None,
        "total_predictions": 0,
        "resolved_predictions": 0,
        "pending_semantics": "signal_predictions.actual_direction IS NULL",
    }
    if not path.exists():
        return empty
    try:
        with sqlite_connect(path) as conn:
            cur = conn.cursor()
            # Fail soft if table missing
            tables = {
                r[0]
                for r in cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "signal_predictions" not in tables:
                return empty
            col_rows = cur.execute("PRAGMA table_info(signal_predictions)").fetchall()
            columns = {str(r[1]) for r in col_rows}
            if "actual_direction" not in columns:
                return empty

            total = int(cur.execute("SELECT COUNT(*) FROM signal_predictions").fetchone()[0])
            resolved = int(
                cur.execute(
                    "SELECT COUNT(*) FROM signal_predictions WHERE actual_direction IS NOT NULL"
                ).fetchone()[0]
            )
            pending_rows = max(0, total - resolved)
            date_expr = _signal_prediction_date_expr(columns)
            row = cur.execute(
                f"""
                SELECT COUNT(DISTINCT {date_expr}), MIN({date_expr})
                FROM signal_predictions
                WHERE actual_direction IS NULL
                """
            ).fetchone()
            pending_dates = int(row[0] or 0) if row else 0
            oldest = row[1] if row else None
        return {
            "pending_rows": pending_rows,
            "pending_dates": pending_dates,
            "oldest_unresolved_date": oldest,
            "total_predictions": total,
            "resolved_predictions": resolved,
            "pending_semantics": (
                "pending_predictions=IC staged window; "
                "pending_rows=signal_predictions unlabeled rows"
            ),
        }
    except Exception as exc:  # noqa: BLE001 — optional enrichment
        logger.warning("signal prediction backlog unavailable: %s", exc)
        return empty


def compute_ic_decay_report() -> Dict[str, Any]:
    """Convenience function: compute IC decay report from saved state.

    Creates an ICMonitor, loads any persisted state, and returns
    the decay report. Used by DashboardGenerator.
    """
    monitor = ICMonitor()
    monitor.load_state()
    signals = monitor.compute_decay_report()
    pending = monitor.get_staged_prediction_count()
    get_staged_names = getattr(monitor, "get_staged_prediction_names", None)
    staged_prediction_names = get_staged_names() if callable(get_staged_names) else []
    # Operator-approved re-baseline trigger (incident 8115a9c1, 2026-08-11):
    # informational re-review signal, never an alert. getattr fallback keeps
    # lightweight fakes compatible.
    get_trigger = getattr(monitor, "rebaseline_trigger_state", None)
    if callable(get_trigger):
        trigger = get_trigger()
    else:
        trigger = {
            "due": False,
            "threshold": IC_MIN_OBS_FOR_STATUS,
            "max_staged_observations": 0,
            "staged_observations_per_signal": {},
        }
    if signals:
        statuses = [row.get("status") for row in signals.values()]
        if any(status == "critical" for status in statuses):
            status = "critical"
        elif any(status == "warning" for status in statuses):
            status = "warning"
        elif all(status == "healthy" for status in statuses):
            status = "healthy"
        else:
            status = "insufficient_resolved_history"
    elif pending:
        status = "waiting_for_forward_returns"
    else:
        status = "no_data"
    backlog = _signal_prediction_backlog()
    return {
        "status": status,
        "signals": signals,
        "resolved_signal_count": len(signals),
        "pending_predictions": pending,
        "pending_scope": "ic_staged_date_window",
        "staged_prediction_names": staged_prediction_names,
        "pending_rows": backlog.get("pending_rows", 0),
        "pending_rows_scope": "historical_db_unlabeled_rows",
        "pending_dates": backlog.get("pending_dates", 0),
        "oldest_unresolved_date": backlog.get("oldest_unresolved_date"),
        "total_predictions": backlog.get("total_predictions", 0),
        "resolved_predictions": backlog.get("resolved_predictions", 0),
        "pending_semantics": backlog.get("pending_semantics")
        or (
            "pending_predictions=IC staged-date window; "
            "pending_rows=signal_predictions unlabeled rows (full history)"
        ),
        "staged_date": monitor.get_staged_date(),
        "label_horizon": "SPY close-to-close forward return from staged market-data date to latest available SPY row",
        "advisory_factor_half_life": advisory_factor_half_life_table(),
        "staged_observations_per_signal": trigger[
            "staged_observations_per_signal"
        ],
        "rebaseline_due": trigger["due"],
        "rebaseline_threshold": trigger["threshold"],
        "max_staged_observations": trigger["max_staged_observations"],
    }


def build_ic_decay_summary(
    report: Mapping[str, Any] | None,
    *,
    evidence_generated_at: Optional[str] = None,
    evidence_freshness: str = "captured_runtime_snapshot",
    control_effect: str = "unknown",
    routing_authority: str = "advisory_only",
    routing_control: str = "unknown",
    kill_switch_level: Optional[str] = None,
) -> Dict[str, Any]:
    """Project IC evidence into a bounded operator-facing quality summary.

    The raw IC report remains available on ``signals.json`` for diagnostics.
    This projection deliberately carries only named signal states and bounded
    counters needed to review the incident. It never contains prediction rows,
    database contents, or a routing decision derived from IC status.
    """
    source = report if isinstance(report, Mapping) else {}
    raw_signals = source.get("signals")
    signals = raw_signals if isinstance(raw_signals, Mapping) else {}

    critical: list[str] = []
    warning: list[str] = []
    insufficient: list[str] = []
    control_eligible_critical: list[str] = []
    control_eligible_warning: list[str] = []
    qualified_count = 0
    minimums: list[int] = []
    signal_evidence: Dict[str, Any] = {}
    for raw_name, raw_row in signals.items():
        if not isinstance(raw_row, Mapping):
            continue
        name = str(raw_name).strip()
        if not name:
            continue
        status = str(raw_row.get("status") or "").lower()
        control_eligible = bool(raw_row.get("control_eligible"))
        if status == "critical":
            critical.append(name)
            qualified_count += 1
            if control_eligible:
                control_eligible_critical.append(name)
        elif status == "warning":
            warning.append(name)
            qualified_count += 1
            if control_eligible:
                control_eligible_warning.append(name)
        elif status == "healthy":
            qualified_count += 1
        elif status == "insufficient_data":
            insufficient.append(name)
        try:
            minimums.append(int(raw_row.get("min_obs_for_status")))
        except (TypeError, ValueError):
            pass
        evidence_fields = {
            "ic_rolling",
            "observations",
            "status",
            "metric_axis",
            "metric_kind",
            "estimate_kind",
            "alignment_status",
            "alignment_reason",
            "inference_status",
            "inference_reason",
            "observation_count",
            "observation_unit",
            "contract_version",
            "evaluation_contract",
            "latest_observation_metadata",
        }
        evidence = {
            key: raw_row.get(key)
            for key in evidence_fields
            if key in raw_row
        }
        if evidence:
            signal_evidence[name] = evidence

    def _bounded_int(key: str, default: int = 0) -> int:
        try:
            return max(0, int(source.get(key) or default))
        except (TypeError, ValueError):
            return default

    staged_names = source.get("staged_prediction_names")
    if isinstance(staged_names, (list, tuple, set)):
        staged_signal_names = sorted(
            {str(name).strip() for name in staged_names if str(name).strip()}
        )
    else:
        staged_signal_names = []

    staged_counts = source.get("staged_observations_per_signal")
    if isinstance(staged_counts, Mapping):
        bounded_staged_counts = {
            str(name).strip(): max(0, int(count))
            for name, count in staged_counts.items()
            if str(name).strip()
        }
    else:
        bounded_staged_counts = {}

    summary: Dict[str, Any] = {
        "status": str(source.get("status") or "unknown"),
        "critical_signals": sorted(set(critical)),
        "warning_signals": sorted(set(warning)),
        "insufficient_data_signals": sorted(set(insufficient)),
        # Control-eligible subsets: only these may drive halt-authoritative IC
        # control alerts. Descriptive critical/warning lists stay complete.
        "control_eligible_critical_signals": sorted(set(control_eligible_critical)),
        "control_eligible_warning_signals": sorted(set(control_eligible_warning)),
        "resolved_signal_count": qualified_count,
        "min_observations": min(minimums) if minimums else IC_MIN_OBS_FOR_STATUS,
        "staged_pending_predictions": _bounded_int("pending_predictions"),
        "staged_pending_signal_names": staged_signal_names,
        "staged_date": source.get("staged_date"),
        "staged_pending_scope": str(
            source.get("pending_scope") or "ic_staged_date_window"
        ),
        # Re-baseline trigger (operator-approved 2026-08-11): bounded per-signal
        # staged counts + due flag; informational, never an alert.
        "staged_observations_per_signal": bounded_staged_counts,
        "rebaseline_due": bool(source.get("rebaseline_due")),
        "rebaseline_threshold": (
            _bounded_int("rebaseline_threshold") or IC_MIN_OBS_FOR_STATUS
        ),
        "max_staged_observations": _bounded_int("max_staged_observations"),
        "historical_unlabeled_rows": _bounded_int("pending_rows"),
        "historical_unlabeled_dates": _bounded_int("pending_dates"),
        "historical_unlabeled_oldest_date": source.get("oldest_unresolved_date"),
        "historical_unlabeled_scope": str(
            source.get("pending_rows_scope") or "historical_db_unlabeled_rows"
        ),
        "evidence_generated_at": evidence_generated_at
        or source.get("generated_at"),
        "evidence_freshness": evidence_freshness,
        "routing_authority": routing_authority,
        "routing_control": routing_control,
        "control_effect": control_effect,
        "signal_evidence": signal_evidence,
    }
    if kill_switch_level is not None:
        summary["kill_switch_level"] = str(kill_switch_level)
    return summary


def ic_control_projection(kill_fields: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Describe the existing paper control alongside advisory IC quality."""
    kill = kill_fields if isinstance(kill_fields, Mapping) else {}
    enabled = bool(kill.get("enabled"))
    mode = str(kill.get("mode") or "").lower()
    return {
        "control_effect": "paper_warning"
        if enabled and mode == "paper"
        else "routing_blocked"
        if enabled
        else "none",
        "routing_control": "routing_blocked" if enabled else "available",
        "kill_switch_level": kill.get("level"),
    }
