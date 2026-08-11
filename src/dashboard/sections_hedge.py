"""Hedge / tsmom / staleness mixin extracted from ``src.dashboard.generator``.

Class-level cluster C8 (10 methods + _last_regime + 4 staleness constants)
moved here by Item 20 (2026-08-12). ``DashboardGenerator`` inherits
``_HedgeSectionsMixin`` (after ``_EnsembleSectionsMixin`` — _apply_staleness_
decay calls C2 helpers). Class-qualified refs rewritten to ``type(self).X``;
SIGNAL_EXCEPTIONS/_log_signal_error resolved lazily through the generator
module (they stay there); datetime.now deferred through the generator module
(FakeDateTime patch seam).
"""

import json
import os
import re
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.paths import DATA_DIR
from src.monitor.signal_ownership import optional_advisory_signals
from src.dashboard.alternative_data import (
    _ENSEMBLE_STALENESS_MAP,
    load_alternative_data_producer_timestamp,
)

logger = logging.getLogger(__name__)


class _HedgeSectionsMixin:
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
            type(self)._last_regime = regime
            return regime.lower() in {"high_vol", "crisis"}
        except Exception as e:
            logger.warning("_is_msm_gated: regime query failed (%s) — using last-known regime '%s'",
                           e, type(self)._last_regime)
            return type(self)._last_regime.lower() in {"high_vol", "crisis"}

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
        from src.dashboard import generator as _generator  # lazy (FakeDateTime patch seam)
        return {
            "available": False,
            "generated_at": _generator.datetime.now(_generator.timezone.utc).isoformat(),
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
        from src.dashboard import generator as _generator  # lazy (FakeDateTime patch seam)
        from src.dashboard.generator import SIGNAL_EXCEPTIONS, _log_signal_error  # lazy (stays in generator)
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
                "generated_at": _generator.datetime.now(_generator.timezone.utc).isoformat(),
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
        from src.dashboard import generator as _generator  # lazy (FakeDateTime patch seam)
        import math as _math

        ttl_seconds = self.SIGNAL_STALENESS_TTL_HOURS * 3600
        tau_hours = self.STALENESS_DECAY_TAU_HOURS
        now = _generator.datetime.now(_generator.timezone.utc)
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
        producer_ts = _generator.load_alternative_data_producer_timestamp(_generator.DATA_DIR)
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
            sh_metrics = type(self)._signal_health_metrics_map()
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
                                la := type(self)._label_alignment_diagnostic(
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
        if type(self)._spc_monitor is None:
            type(self)._spc_monitor = SPCMonitor()
            type(self)._spc_monitor.load_state()

        monitor = type(self)._spc_monitor

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

