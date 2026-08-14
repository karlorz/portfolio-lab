"""EnsembleVoter weights mixin (Item 5 s3 ENSEMBLE-VOTER-MIXINS).
Methods extracted verbatim from src/strategy/ensemble_voter.py.
"""

import logging
from datetime import datetime
from src.paths import BASE_ALLOCATION
from src.paths import ENSEMBLE_CONSENSUS_THRESHOLD
from src.paths import sqlite_connect
from src.signals.regime_spec import REGIME_WEIGHTS
from src.signals.regime_spec import Regime
from src.signals.regime_spec import SignalReading
from src.signals.signal_source import SignalSource
from src.strategy.ensemble_support import EnsembleVote
from src.strategy.ensemble_support import _get_health_tracker
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import json
import numpy as np
import os
import sqlite3
logger = logging.getLogger("src.strategy.ensemble_voter")

class WeightsMixin:
    # Clip each arm to max_weight (0.50 default)
    DEFAULT_PER_SIGNAL_WEIGHT_CAP = 0.50

    @staticmethod
    def _apply_per_signal_weight_cap(
        weights: Dict,
        *,
        max_weight: float = DEFAULT_PER_SIGNAL_WEIGHT_CAP,
        soft_delete: Optional[set] = None,
    ) -> Dict:
        """Clip each arm to ``max_weight`` then water-fill redistrib excess.

        Soft-delete / zero arms stay at 0. When only one positive arm remains,
        concentration is unavoidable — leave at 1.0 (cap cannot diversify).
        """
        from src.strategy.ensemble_voter import EnsembleVoter
        if not weights:
            return weights
        try:
            cap = float(max_weight)
        except (TypeError, ValueError):
            cap = EnsembleVoter.DEFAULT_PER_SIGNAL_WEIGHT_CAP
        if cap <= 0.0 or cap >= 1.0:
            return weights

        soft = soft_delete or set()
        out = {k: max(0.0, float(v or 0.0)) for k, v in weights.items()}
        for src in soft:
            if src in out:
                out[src] = 0.0

        total0 = sum(out.values())
        if total0 <= 0:
            return out
        # Normalize first so mass is a probability simplex
        out = {k: v / total0 for k, v in out.items()}

        max_iter = 16
        for _ in range(max_iter):
            positive = [k for k, v in out.items() if v > 1e-12 and k not in soft]
            if len(positive) <= 1:
                break
            over = [k for k in positive if out[k] > cap + 1e-12]
            if not over:
                break
            excess = 0.0
            for k in over:
                excess += out[k] - cap
                out[k] = cap
            under = [k for k in positive if out[k] < cap - 1e-12]
            if not under:
                # Everyone at/above cap — equal-share residual among positive
                # (feasible only if n*cap >= 1)
                n = len(positive)
                if n * cap + 1e-12 < 1.0:
                    # Impossible to satisfy: keep equal 1/n (may exceed cap)
                    share = 1.0 / n
                    for k in positive:
                        out[k] = share
                break
            under_sum = sum(out[k] for k in under)
            if under_sum <= 0:
                # Degenerate under-set: distribute excess equally among under
                share = excess / len(under)
                for k in under:
                    out[k] = min(cap, out[k] + share)
            else:
                scale = (under_sum + excess) / under_sum
                for k in under:
                    out[k] = min(cap, out[k] * scale)

        # Final soft-delete pin + renorm of non-soft mass
        for src in soft:
            if src in out:
                out[src] = 0.0
        total = sum(out.values())
        if total > 0 and abs(total - 1.0) > 1e-9:
            out = {k: v / total for k, v in out.items()}
        return out

    def _cap_per_signal_weights(
        self,
        weights: Dict,
        regime_name: str,
        *,
        max_weight: Optional[float] = None,
    ) -> Dict:
        """Instance wrapper: apply cap, pin soft-delete, record disclosure."""
        env_cap = os.environ.get("ENSEMBLE_PER_SIGNAL_WEIGHT_CAP", "").strip()
        if env_cap:
            try:
                cap = float(env_cap)
            except (TypeError, ValueError):
                cap = (
                    float(max_weight)
                    if max_weight is not None
                    else self.DEFAULT_PER_SIGNAL_WEIGHT_CAP
                )
        else:
            cap = (
                float(max_weight)
                if max_weight is not None
                else self.DEFAULT_PER_SIGNAL_WEIGHT_CAP
            )
        soft = self._static_zero_baseline_sources(regime_name)
        before = {k: float(v or 0.0) for k, v in weights.items()}
        capped = self._apply_per_signal_weight_cap(
            weights, max_weight=cap, soft_delete=soft
        )
        capped = self._pin_zero_baseline_weights(capped, regime_name)
        max_before = max(before.values()) if before else 0.0
        max_after = max((float(v or 0.0) for v in capped.values()), default=0.0)
        breached = [
            (k.value if hasattr(k, "value") else str(k))
            for k, v in before.items()
            if float(v or 0.0) > cap + 1e-12
        ]
        self._last_per_signal_cap = {
            "cap": cap,
            "applied": bool(breached) or max_before > cap + 1e-12,
            "breached_before": breached,
            "max_weight_before": round(max_before, 5),
            "max_weight_after": round(max_after, 5),
            "policy": "clip_renorm_iterative_soft_delete_pinned",
        }
        if breached:
            logger.info(
                "Per-signal weight cap %.0f%% applied (Batch DN): max %.1f%% → %.1f%%; "
                "breached=%s",
                cap * 100,
                max_before * 100,
                max_after * 100,
                ",".join(breached),
            )
        return capped

    def get_blended_weights(self, regime_name: str) -> dict:
        """Get regime weights blended between static REGIME_WEIGHTS and bandit.

        Starts 100% static (bandit_blend=0.0), gradually shifts toward
        up to 70% bandit after 252 days of observations.

        Batch DK: static-zero soft-delete arms stay at 0 after blend+renorm
        (bandit posterior must not reintroduce vote mass).
        """
        from src.strategy.ensemble_voter import BANDIT_MAX_BLEND, BANDIT_WARMUP_DAYS
        regime_enum = getattr(Regime, regime_name, Regime.NORMAL)
        static = dict(REGIME_WEIGHTS.get(regime_enum, {}))

        # If bandit not initialized (e.g. test fixtures bypassing __init__), fall back
        if not hasattr(self, 'bandit') or self.bandit is None:
            return static

        bandit = self.bandit.get_weights(regime_name)

        if bandit is None:
            return static  # Cold start: 100% static

        # Blend: starts 100% static, shifts to (1-MAX_BLEND)/MAX_BLEND after warmup
        day_steps = int(getattr(self, "bandit_days", 0) or 0)
        if day_steps <= 0 and int(getattr(self, "bandit_observations", 0) or 0) > 0:
            # Legacy states without bandit_days: approximate days from arm updates
            n_sources = max(1, len(list(SignalSource)))
            day_steps = max(1, int(self.bandit_observations) // n_sources)
        blend = min(BANDIT_MAX_BLEND, day_steps / BANDIT_WARMUP_DAYS * BANDIT_MAX_BLEND)

        # Convert static keys from SignalSource enum to string values for matching
        static_by_value = {k.value: v for k, v in static.items()}

        blended = {}
        for sig_value in static_by_value:
            bandit_w = bandit.get(sig_value, 0.0)
            static_w = static_by_value[sig_value]
            # Hard-pin soft-delete: never mix bandit mass into static-zero arms
            if float(static_w or 0.0) <= 0.0:
                blended[sig_value] = 0.0
            else:
                blended[sig_value] = static_w * (1 - blend) + bandit_w * blend

        # Normalize to sum=1.0
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}

        # Convert back to SignalSource keys
        value_to_source = {s.value: s for s in SignalSource}
        out = {value_to_source[k]: v for k, v in blended.items() if k in value_to_source}
        return self._pin_zero_baseline_weights(out, regime_name)

    def get_adaptive_learning_status(self, regime_name: Optional[str] = None) -> Dict[str, Any]:
        """Disclose adaptive-learning branch status without changing weights."""
        from src.strategy.ensemble_voter import BANDIT_MAX_BLEND, BANDIT_WARMUP_DAYS
        if regime_name is None:
            current = getattr(self, "current_regime", Regime.NORMAL)
            regime_name = current.name if hasattr(current, "name") else str(current)

        observations = int(getattr(self, "bandit_observations", 0) or 0)
        reward_days = int(getattr(self, "bandit_days", 0) or 0)
        if reward_days <= 0 and observations > 0:
            n_sources = max(1, len(list(SignalSource)))
            reward_days = max(1, observations // n_sources)
        bandit = getattr(self, "bandit", None)
        bandit_status: Dict[str, Any] = {
            "status": "unavailable",
            "enabled": False,
            "observations": observations,
            "reward_days": reward_days,
            "days": reward_days,
            "warmup_days": BANDIT_WARMUP_DAYS,
            "max_blend": BANDIT_MAX_BLEND,
            "current_blend": 0.0,
            "reason": "bandit_weighter_unavailable",
        }

        if bandit is not None:
            bandit_status.update({
                "enabled": True,
                "status": "non_effective",
                "reason": "cold_start_no_regime_weights",
            })
            try:
                bandit_weights = bandit.get_weights(regime_name)
                if bandit_weights is not None:
                    blend = min(
                        BANDIT_MAX_BLEND,
                        reward_days / BANDIT_WARMUP_DAYS * BANDIT_MAX_BLEND,
                    )
                    bandit_status["current_blend"] = round(blend, 4)
                    if blend > 0:
                        bandit_status["status"] = "active"
                        bandit_status["reason"] = "blending_with_static_weights"
                    else:
                        bandit_status["reason"] = "cold_start_no_observations"
            except (AttributeError, KeyError, ValueError, TypeError, OSError) as e:
                bandit_status.update({
                    "enabled": False,
                    "status": "unavailable",
                    "reason": f"bandit_status_error:{type(e).__name__}",
                })

        use_ic = bool(getattr(self, "_use_ic_weights", False))
        ic_weighter = getattr(self, "_ic_weighter", None)
        try:
            ic_blend_alpha = float(os.environ.get("ENSEMBLE_IC_WEIGHT_BLEND_ALPHA", "0.3"))
        except ValueError:
            ic_blend_alpha = 0.3

        online_ic_status: Dict[str, Any] = {
            "status": "disabled",
            "enabled": use_ic,
            "state_available": ic_weighter is not None,
            "blend_alpha": ic_blend_alpha,
            "reason": "env_disabled",
        }
        if use_ic and ic_weighter is None:
            online_ic_status.update({
                "status": "unavailable",
                "reason": "initialization_failed_or_unavailable",
            })
        elif use_ic and ic_weighter is not None:
            online_ic_status.update({
                "status": "active",
                "reason": "weighter_initialized",
            })

        last_ic_status = getattr(self, "_last_online_ic_learning_status", None)
        if isinstance(last_ic_status, dict):
            online_ic_status.update(last_ic_status)

        return {
            "bandit": bandit_status,
            "online_ic": online_ic_status,
        }

    @staticmethod
    def _compute_asset_biases(
        weighted_signals: List[SignalReading], fallback_consensus: float
    ) -> Dict[str, float]:
        """Compute per-asset weighted bias from signal readings."""
        assets = ['SPY', 'TLT', 'GLD']
        asset_biases = {}
        for asset in assets:
            asset_signals = [
                (r.asset_signals.get(asset, 0), r.weight)
                for r in weighted_signals
                if r.asset_signals and asset in r.asset_signals and not np.isnan(r.asset_signals.get(asset, np.nan))
            ]
            if asset_signals:
                total_w = sum(w for _, w in asset_signals) or 1.0
                asset_biases[asset] = sum(v * w for v, w in asset_signals) / total_w
            else:
                asset_biases[asset] = fallback_consensus
        return asset_biases

    @staticmethod
    def _determine_action(
        regime: Regime, regime_confidence: float, equity_bias: float, agreement: float
    ) -> Tuple[str, float]:
        """Determine portfolio action from regime, equity bias, and agreement.

        Uses regime-conditional consensus thresholds:
        CRISIS 0.50, HIGH_VOL 0.55, RECOVERY 0.60, LOW_VOL 0.67, NORMAL 0.75.
        Falls back to ENSEMBLE_CONSENSUS_THRESHOLD env var for unknown regimes.
        """
        from src.strategy.ensemble_voter import REGIME_CONSENSUS_THRESHOLDS
        if regime == Regime.CRISIS:
            return "risk_off", regime_confidence

        # Regime-specific threshold (falls back to global constant)
        threshold = REGIME_CONSENSUS_THRESHOLDS.get(
            regime.value.upper() if hasattr(regime.value, 'upper') else str(regime.value).upper(),
            ENSEMBLE_CONSENSUS_THRESHOLD,
        )

        if equity_bias > 0.3 and agreement > threshold:
            return "increase_equity", agreement * abs(equity_bias)
        elif equity_bias < -0.3 and agreement > threshold:
            return "decrease_equity", agreement * abs(equity_bias)
        else:
            # Neutral hold conviction tracks agreement × regime confidence —
            # do not hardcode 0.5 (high-agreement hold looked identical to uncertain).
            conf = float(max(0.0, min(1.0, agreement * regime_confidence)))
            return "neutral", conf

    def _compute_consensus(
        self,
        weighted_signals: List[SignalReading],
        regime: Regime,
        regime_confidence: float,
    ) -> '_ConsensusResult':  # noqa: F821  # local nested class
        """Compute weighted consensus, agreement ratio, and asset biases."""
        # Weighted consensus — handle NaN values
        valid_signals = [
            (r.value, r.weight)
            for r in weighted_signals
            if not np.isnan(r.value)
        ]

        if valid_signals:
            total_weight = sum(w for _, w in valid_signals)
            if total_weight == 0:
                total_weight = 1.0
            weighted_consensus = sum(v * w for v, w in valid_signals) / total_weight
        else:
            weighted_consensus = 0.0
            total_weight = 1.0

        # Agreement ratio: % of weighted signals agreeing with consensus
        agreement = sum(
            r.weight for r in weighted_signals
            if np.sign(r.value) == np.sign(weighted_consensus) or abs(r.value) < 0.1
        ) / total_weight

        # Asset-specific consensus
        asset_biases = self._compute_asset_biases(weighted_signals, weighted_consensus)

        # Determine action
        equity_bias = asset_biases.get('SPY', weighted_consensus)
        duration_bias = asset_biases.get('TLT', 0)
        gold_bias = asset_biases.get('GLD', 0)

        action, action_confidence = self._determine_action(
            regime, regime_confidence, equity_bias, agreement
        )

        return self._ConsensusResult(
            weighted_consensus=weighted_consensus,
            agreement=agreement,
            equity_bias=equity_bias,
            duration_bias=duration_bias,
            gold_bias=gold_bias,
            action=action,
            action_confidence=action_confidence,
        )

    def _build_vote(
        self,
        weighted_signals: List[SignalReading],
        consensus: '_ConsensusResult',  # noqa: F821  # local nested class
        regime: Regime,
        regime_confidence: float,
    ) -> EnsembleVote:
        """Build EnsembleVote from weighted signals and consensus result."""
        reasons = [
            f"Regime: {regime.value} (confidence: {regime_confidence:.1%})",
            f"Sources: {len(weighted_signals)}, Consensus: {consensus.weighted_consensus:+.3f}",
            f"Agreement: {consensus.agreement:.1%}",
            f"Equity bias: {consensus.equity_bias:+.3f}, Duration: {consensus.duration_bias:+.3f}, Gold: {consensus.gold_bias:+.3f}"
        ]

        for r in weighted_signals[:3]:
            reasons.append(f"  {r.source.value}: {r.value:+.3f} (w={r.weight:.2f}, conf={r.confidence:.1%})")

        # Compute effective signal count (N_eff) and Shannon entropy on
        # *renormalized* positive weights so incomplete collection does not
        # understate diversification (sleeping-experts: active set sums to 1).
        weights_arr = np.array([r.weight for r in weighted_signals], dtype=float)
        weights_arr = weights_arr[np.isfinite(weights_arr) & (weights_arr > 0)]
        active_weight_mass = float(np.sum(weights_arr)) if len(weights_arr) else 0.0
        if len(weights_arr) > 0 and active_weight_mass > 0:
            w_norm = weights_arr / active_weight_mass
            weight_entropy = float(-np.sum(w_norm * np.log(w_norm)))
            n_eff = float(np.exp(weight_entropy))
        else:
            weight_entropy = 0.0
            n_eff = 0.0
            active_weight_mass = 0.0

        sleep_reasons = dict(getattr(self, "_health_gate_sleep_reasons", None) or {})
        regime_gated = dict(getattr(self, "_regime_gated", None) or {})
        soft_floor = dict(getattr(self, "_health_gate_soft_floor", None) or {})
        adaptive_learning = dict(self.get_adaptive_learning_status(regime.name) or {})
        # Batch DN: disclose final per-signal weight cap application
        cap_info = getattr(self, "_last_per_signal_cap", None)
        if isinstance(cap_info, dict) and cap_info:
            adaptive_learning["per_signal_weight_cap"] = cap_info
        if soft_floor:
            adaptive_learning["health_gate_soft_floor"] = soft_floor
        return EnsembleVote(
            timestamp=str(datetime.now()),
            regime=regime,
            regime_confidence=regime_confidence,
            num_sources=len(weighted_signals),
            weighted_consensus=consensus.weighted_consensus,
            agreement_ratio=consensus.agreement,
            equity_bias=consensus.equity_bias,
            duration_bias=consensus.duration_bias,
            gold_bias=consensus.gold_bias,
            action=consensus.action,
            confidence=consensus.action_confidence,
            reasoning="\n".join(reasons),
            source_votes=weighted_signals,
            n_eff=round(n_eff, 2),
            weight_entropy=round(weight_entropy, 4),
            adaptive_learning=adaptive_learning,
            health_gate_slept=sleep_reasons or None,
            health_gate_freeze=bool(getattr(self, "_health_gate_freeze", False)),
            regime_gated=regime_gated or None,
            health_gate_soft_floor=soft_floor or None,
        )

    def _persist_vote(self, vote: EnsembleVote, weighted_consensus: float) -> None:
        """Persist ensemble decision for regret-weighted cycle and save vote to DB."""
        # Persist ensemble decision for next regret-weighted cycle (v8.03)
        try:
            from src.strategy.regret_weighted_selector import RegretWeightedSelector
            rw_selector = RegretWeightedSelector()
            rw_selector.state.last_ensemble_decision = weighted_consensus
            rw_selector._save_state()
        except (ImportError, OSError, KeyError, ValueError, TypeError, AttributeError) as rw_e:
            logger.warning("Could not persist ensemble decision to regret-weighted state: %s", rw_e)

        # Persist IC weighter state if enabled
        if getattr(self, '_use_ic_weights', False) and getattr(self, '_ic_weighter', None) is not None:
            try:
                ic_state_path = self.data_path / "ic_weighter_state.json"
                ic_state = self._ic_weighter.get_state()
                with open(ic_state_path, "w") as f:
                    json.dump(ic_state, f)
                logger.debug("OnlineICWeighter state saved to %s", ic_state_path)
            except (OSError, TypeError, ValueError) as e:
                logger.warning("Failed to save OnlineICWeighter state: %s", e)

        # Check for IC-based signal decay alerts
        try:
            _tracker = _get_health_tracker()
            if _tracker is not None:
                alerts = _tracker.detect_ic_alerts()
                if alerts:
                    alert_names = [a.source for a in alerts]
                    logger.warning("IC decay alerts detected: %s", alert_names)
        except (KeyError, ValueError, TypeError, AttributeError, OSError, sqlite3.Error) as ic_e:
            logger.warning("IC alert check failed: %s", ic_e)

        # Save to DB
        self._save_vote(vote)

    def _save_vote(self, vote: EnsembleVote):
        """Save vote to database, including per-source readings (v5.70)."""
        with sqlite_connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO ensemble_votes
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                vote.timestamp,
                vote.regime.value,
                vote.regime_confidence,
                vote.num_sources,
                vote.weighted_consensus,
                vote.agreement_ratio,
                vote.equity_bias,
                vote.duration_bias,
                vote.gold_bias,
                vote.action,
                vote.confidence,
                vote.reasoning
            ))

            # v5.70: Save individual source readings for attribution
            for reading in vote.source_votes:
                try:
                    conn.execute("""
                        INSERT INTO source_readings
                        (timestamp, source, value, confidence, weight, regime_fit, explanation)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        vote.timestamp,
                        reading.source.value if hasattr(reading.source, 'value') else str(reading.source),
                        float(reading.value),
                        float(reading.confidence),
                        float(reading.weight),
                        reading.regime_fit or "",
                        (reading.explanation or "")[:500],
                    ))
                except (ValueError, TypeError) as e:
                    logger.warning("Failed to save source reading %s: %s", reading.source, e)

    def recommend_allocation(
        self,
        base_allocation: Dict[str, float] = None,
        vote: Optional[EnsembleVote] = None,
        max_shift: float = 0.10
    ) -> Dict[str, Dict]:
        """
        Generate allocation recommendation based on ensemble vote.
        
        Returns shifts from base allocation for each asset.
        """
        if base_allocation is None:
            base_allocation = BASE_ALLOCATION
        
        if vote is None:
            vote = self.compute_vote()
        
        # Apply shifts based on biases
        shifts = {
            'SPY': np.clip(vote.equity_bias * max_shift, -max_shift, max_shift),
            'TLT': np.clip(vote.duration_bias * max_shift, -max_shift, max_shift),
            'GLD': np.clip(vote.gold_bias * max_shift, -max_shift, max_shift),
        }
        
        # Risk-off override
        if vote.regime == Regime.CRISIS:
            shifts['SPY'] = -max_shift * 0.5  # Reduce equity
            shifts['GLD'] = max_shift * 0.3   # Increase gold
            shifts['TLT'] = max_shift * 0.2   # Increase bonds
        
        result = {}
        total_shift = 0
        
        for asset, base in base_allocation.items():
            shift = shifts.get(asset, 0)
            new_alloc = base + shift
            
            result[asset] = {
                'base': base,
                'shift': shift,
                'new': np.clip(new_alloc, 0.05, 0.95),  # Bounds
                'bias': shifts.get(asset, 0),
            }
            total_shift += shift
        
        # Normalize to sum to 1
        total = sum(r['new'] for r in result.values())
        for asset in result:
            result[asset]['new'] /= total
            result[asset]['normalized_shift'] = result[asset]['new'] - result[asset]['base']
        
        return {
            'assets': result,
            'regime': vote.regime.value,
            'confidence': vote.confidence,
            'action': vote.action,
            'consensus': vote.weighted_consensus,
            'timestamp': vote.timestamp
        }

    def get_bl_views(
        self,
        vote: Optional[EnsembleVote] = None,
        tau: float = 0.15,
        prior: str = "equal",
    ) -> Dict[str, Any]:
        """Generate Black-Litterman views from ensemble vote.

        Maps equity_bias, duration_bias, and gold_bias from the
        current ensemble consensus to BL absolute views, with view
        confidence derived from signal health scores.

        Args:
            vote: Pre-computed vote (default: compute fresh).
            tau: BL tau parameter (view weight). Default 0.15.
            prior: Prior type — "equal" or "market".

        Returns:
            Dict with 'views' (BLViews), 'tau', 'prior', and
            'health_scores_used' keys.
        """
        from src.strategy.black_litterman_mapper import map_biases_to_views

        if vote is None:
            vote = self.compute_vote()

        # Collect health scores from tracker
        health_scores = {}
        tracker = _get_health_tracker()
        if tracker is not None:
            try:
                report = tracker.get_health_report()
                for source_name, data in report.get('sources', {}).items():
                    score = data.get('health_score', 0.5)
                    health_scores[source_name] = score
            except (KeyError, ValueError, TypeError, AttributeError, OSError, sqlite3.Error) as e:
                logger.warning("Could not get health scores for BL views: %s", e)

        views = map_biases_to_views(
            equity_bias=vote.equity_bias,
            duration_bias=vote.duration_bias,
            gold_bias=vote.gold_bias,
            health_scores=health_scores if health_scores else None,
            tau=tau,
            prior=prior,
        )

        return {
            'views': views,
            'tau': tau,
            'prior': prior,
            'health_scores_used': health_scores,
            'equity_bias': vote.equity_bias,
            'duration_bias': vote.duration_bias,
            'gold_bias': vote.gold_bias,
        }
