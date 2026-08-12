"""Ensemble / signal-health section mixin extracted from ``src.dashboard.generator``.

Class-level cluster C2 (17 static methods + 8 class constants) moved here by
Item 19 (2026-08-12). ``DashboardGenerator`` inherits ``_EnsembleSectionsMixin``;
methods referencing ``DashboardGenerator.X`` use a call-time lazy import
(circular-import safe — the generator module imports this mixin for its bases).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.paths import DATA_DIR


class _EnsembleSectionsMixin:
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
        from src.dashboard.generator import DashboardGenerator  # lazy (circular-import safe)
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
        from src.dashboard.generator import DashboardGenerator  # lazy (circular-import safe)
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
        from src.dashboard.generator import DashboardGenerator  # lazy (circular-import safe)
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
        from src.dashboard.generator import DashboardGenerator  # lazy (circular-import safe)
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
        from src.dashboard.generator import DashboardGenerator  # lazy (circular-import safe)
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
        from src.dashboard.generator import DashboardGenerator  # lazy (circular-import safe)
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
        from src.dashboard.generator import DashboardGenerator  # lazy (circular-import safe)
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
        from src.dashboard.generator import DashboardGenerator  # lazy (circular-import safe)
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

