"""Signal section assembly for signals.json."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

logger = logging.getLogger(__name__)

__all__ = ["SignalSectionBuilder"]

@dataclass(slots=True)
class SignalSectionBuilder:
    """Own signal-section assembly while reusing generator helper seams."""

    owner: Any
    generator_module: ModuleType

    def __getattr__(self, name: str) -> Any:
        return getattr(self.owner, name)

    def build_base_sections(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build the core signal sections before dashboard-level metadata."""
        generator = self.generator_module
        DATA_DIR = generator.DATA_DIR
        DB_PATH = generator.DB_PATH
        DashboardGenerator = generator.DashboardGenerator
        MONITOR_EXCEPTIONS = generator.MONITOR_EXCEPTIONS
        SIGNAL_EXCEPTIONS = generator.SIGNAL_EXCEPTIONS
        _apply_kill_to_smart_rebalance = generator._apply_kill_to_smart_rebalance
        _enrich_duration_allocation_provenance = (
            generator._enrich_duration_allocation_provenance
        )
        _log_signal_error = generator._log_signal_error
        _remaining_budget_display_pct = generator._remaining_budget_display_pct
        _remaining_budget_ratio = generator._remaining_budget_ratio
        load_kill_switch_payload = generator.load_kill_switch_payload
        project_alternative_data_signal = generator.project_alternative_data_signal
        validate_signal = generator.validate_signal
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
        _bs_started = datetime.now()
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
        _bs_elapsed = (datetime.now() - _bs_started).total_seconds()
        if _bs_elapsed >= 2.0:
            logger.warning(
                "behavioral_sentiment section took %.1fs (stall watchdog)", _bs_elapsed
            )

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
            "regime_allocation_diagnostic": self._build_regime_allocation_diagnostic(
                current_regime
            ),
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

    def build_optional_sections(
        self,
        output: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Append optional operational sections that precede staleness checks."""
        generator = self.generator_module
        SIGNAL_EXCEPTIONS = generator.SIGNAL_EXCEPTIONS
        _log_signal_error = generator._log_signal_error
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


    def apply_postprocessors(
        self,
        output: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply staleness, monitoring, alerting, and final signal appenders."""
        generator = self.generator_module
        DATA_DIR = generator.DATA_DIR
        DashboardGenerator = generator.DashboardGenerator
        MONITOR_EXCEPTIONS = generator.MONITOR_EXCEPTIONS
        SIGNAL_EXCEPTIONS = generator.SIGNAL_EXCEPTIONS
        _compact_health_summary = generator._compact_health_summary
        _first_known_value = generator._first_known_value
        _is_predictive_fred_macro = generator._is_predictive_fred_macro
        _load_canonical_health_report = generator._load_canonical_health_report
        _log_signal_error = generator._log_signal_error
        build_signal_health_section = generator.build_signal_health_section
        project_execution_timeline_onto_health = (
            generator.project_execution_timeline_onto_health
        )
        project_paper_return_ssot_onto_health = (
            generator.project_paper_return_ssot_onto_health
        )
        project_reentry_eligibility_onto_health = (
            generator.project_reentry_eligibility_onto_health
        )
        project_repo_public_mirror_lag_onto_health = (
            generator.project_repo_public_mirror_lag_onto_health
        )
        project_smart_rebalance_budget_onto_health = (
            generator.project_smart_rebalance_budget_onto_health
        )
        project_voting_mass_quality_onto_health = (
            generator.project_voting_mass_quality_onto_health
        )
        validate_signal = generator.validate_signal
        _ = context["cursor"]
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
