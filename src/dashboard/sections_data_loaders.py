"""Data-loader mixin extracted from ``src.dashboard.generator``.

Class-level cluster C6 (5 helpers: ``_load_broker_data``,
``_load_garch_cvar_data``, ``_load_entropy_data``, ``_get_yield_curve_data``,
``_load_json_file``) moved here by Item 27 (2026-08-12) — the FINAL mixin of
the A8 decomposition. ``DashboardGenerator`` inherits ``_DataLoaderSectionsMixin``.

PATCH-TARGET CONTRACT (critical): ``DATA_DIR`` / ``PUBLIC_DIR`` /
``YIELDS_JSON`` / ``datetime`` / ``timezone`` /
``_yield_source_provenance`` / ``_enrich_duration_allocation_provenance``
are resolved via call-time lazy imports from the generator module so tests
that patch ``src.dashboard.generator.*`` keep working (test_generator.py
patch regions + test_yield_curve_stale_asof.py) — module-top src.paths
imports would read the REAL yields.json / data dir during tests. The
provenance functions stay as generator module fns (direct import at
test_batch_ag_residual_honesty.py:82); this mixin only references them.
``self.conn`` stays a direct attribute access (exception semantics). The
lazy ``BrokerDataLoader`` / ``conformal_risk`` / ``math`` + ``BASE_ALLOCATION``
imports inside methods are preserved as-is.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.utils import safe_get

logger = logging.getLogger(__name__)


class _DataLoaderSectionsMixin:
    def _load_broker_data(self) -> Dict:
        """Load broker position sync and order data for dashboard."""
        from src.dashboard import generator as _generator  # lazy (patch seams)
        from src.dashboard.broker_data_loader import BrokerDataLoader

        return BrokerDataLoader(data_dir=_generator.DATA_DIR).load()

    def _load_garch_cvar_data(self) -> Dict:
        """Load GARCH-filtered CVaR metrics for dashboard (v3.21).

        Also computes a conformal CVaR cross-check (distribution-free)
        as a model-risk validation against the parametric GARCH estimate.
        """
        from src.dashboard import generator as _generator  # lazy (patch seams)
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
            health_file = _generator.DATA_DIR / ".health_report.json"
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
        from src.dashboard import generator as _generator  # lazy (patch seams)
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
            health_file = _generator.DATA_DIR / ".health_report.json"
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

    def _get_yield_curve_data(self) -> Dict:
        """Get yield curve data from yields.json and calculate duration allocation."""
        from src.dashboard import generator as _generator  # lazy (patch seams)
        result = {
            "yield_curve": None,
            "duration_allocation": None
        }

        yields_file = _generator.YIELDS_JSON
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
                    today = _generator.datetime.now(_generator.timezone.utc).date()
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
                    for key, value in _generator._yield_source_provenance(
                        _generator.PUBLIC_DIR
                    ).items()
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
            result["duration_allocation"] = _generator._enrich_duration_allocation_provenance(
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

    @staticmethod
    def _load_json_file(path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
