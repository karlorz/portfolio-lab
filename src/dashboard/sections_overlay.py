"""Overlay / regime-prep mixin extracted from ``src.dashboard.generator``.

Class-level cluster C1 (10 methods: unavailable-payload builders, overlay
data/IC recording, two-stage + BOCD regime generation, VIX coercion,
regime-VIX enrichment, signal-generation context) moved here by Item 23
(2026-08-12). ``DashboardGenerator`` inherits ``_OverlaySectionsMixin``.
Zero class-qualified refs (audit); datetime.now / DATA_DIR deferred through
the generator module (FakeDateTime + DATA_DIR patch seams);
SIGNAL_EXCEPTIONS/MONITOR_EXCEPTIONS/_log_signal_error resolved lazily
through the generator module (they stay there); heavy producers
(OverlayDashboardGenerator / VIXTermStructureSignalGenerator / ICMonitor /
TwoStageKMeansRegime / FredMdFetcher / BOCDDetector) stay function-local
lazy imports.
"""

import json
import logging
from collections import deque
from typing import Any, Dict, Optional

import numpy as np

from src.utils import classify_vix_regime

logger = logging.getLogger(__name__)


class _OverlaySectionsMixin:
    @staticmethod
    def _unavailable_zero_dte_payload() -> Dict[str, Any]:
        """Schema-compatible zero_dte panel when no producer is wired.

        LiveDashboard expects positions/config fields; never publish silent {}.
        Not live order-routing authority.
        """
        from src.dashboard import generator as _generator  # lazy (patch seams)
        now_ts = _generator.datetime.now().isoformat()
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
        from src.dashboard import generator as _generator  # lazy (patch seams)
        now_ts = _generator.datetime.now().isoformat()
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
        from src.dashboard import generator as _generator  # lazy (patch seams)
        from src.dashboard.generator import SIGNAL_EXCEPTIONS, MONITOR_EXCEPTIONS, _log_signal_error  # lazy (stay in generator)
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
            state_file = _generator.DATA_DIR / "vix_overlay_state.json"
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

        Two-phase lifecycle (Task 2B — per-signal):
        1. Resolve: each staged prediction is paired with the forward return of
           its own declared target asset (SPY/GLD/TLT) once its intended
           horizon has elapsed; other entries stay staged.
        2. Stage: store canonical current predictions (equity/gold/duration
           biases, alternative-data SPY-facing value, behavioral normalized
           equity shift) for resolution next runs.

        Consensus, factor rotation, and FRED are NOT staged into correlation
        control history until their basket/outcome/metric contracts are
        implemented; their exclusion is disclosed in the IC summary.
        Saves monitor state to disk so IC data survives across cron runs.
        """
        from src.dashboard import generator as _generator  # lazy (patch seams)
        from src.dashboard.generator import SIGNAL_EXCEPTIONS, MONITOR_EXCEPTIONS, _log_signal_error  # lazy (stay in generator)
        try:
            from src.monitor.ic_decay_monitor import ICMonitor

            monitor = ICMonitor()
            monitor.load_state()
            cursor = self.conn.cursor()

            def _latest_bar(symbol: str):
                cursor.execute(
                    "SELECT date, close FROM prices WHERE symbol = ? "
                    "ORDER BY date DESC LIMIT 1",
                    (symbol,),
                )
                return cursor.fetchone()

            latest_spy_row = _latest_bar("SPY")

            # Phase 1: Resolve previously staged predictions per target asset.
            staged_entries = getattr(monitor, "_staged", None) or {}
            for target_asset in ("SPY", "GLD", "TLT"):
                latest = _latest_bar(target_asset)
                if not latest:
                    continue
                latest_date, latest_close = latest
                staged_dates = sorted(
                    {
                        entry.get("prediction_date")
                        for entry in staged_entries.values()
                        if entry.get("metadata", {}).get("target_asset")
                        in (None, target_asset)
                        and entry.get("prediction_date")
                    }
                )
                if not staged_dates:
                    continue
                for staged_date in staged_dates:
                    cursor.execute(
                        "SELECT date, close FROM prices WHERE symbol = ? "
                        "AND date >= ? ORDER BY date ASC LIMIT 1",
                        (target_asset, staged_date),
                    )
                    start_row = cursor.fetchone()
                    if (
                        not start_row
                        or start_row[0] == latest_date
                        or float(start_row[1]) <= 0
                    ):
                        continue
                    start_price = float(start_row[1])
                    forward_return = (float(latest_close) / start_price) - 1.0
                    cursor.execute(
                        "SELECT MIN(date), COUNT(*) FROM prices WHERE symbol = ? "
                        "AND date > ? AND date <= ?",
                        (target_asset, start_row[0], latest_date),
                    )
                    realized_range_row = cursor.fetchone()
                    realized_start_date = (
                        str(realized_range_row[0])
                        if realized_range_row and realized_range_row[0]
                        else None
                    )
                    realized_horizon_sessions = int(realized_range_row[1] or 0)
                    n_resolved = monitor.resolve_staged(
                        forward_return,
                        resolved_date=str(latest_date),
                        realized_start_date=realized_start_date,
                        target_asset=target_asset,
                        realized_horizon_sessions=realized_horizon_sessions,
                    )
                    if n_resolved:
                        logger.info(
                            "IC decay: resolved %d staged predictions "
                            "(%s → %s %s, forward return=%.4f%%)",
                            n_resolved, staged_date, latest_date, target_asset,
                            forward_return * 100,
                        )

            # Phase 2: Stage canonical current predictions (per-signal upsert).
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

            # Alternative data: canonical SPY-facing value (projection boundary).
            alt = output.get("alternative_data")
            if isinstance(alt, dict) and alt.get("spy_value") is not None:
                predictions["alternative_data"] = float(alt["spy_value"])

            # Behavioral sentiment: normalized equity shift (capped ±5% → [-1, 1]).
            beh = output.get("behavioral_sentiment")
            if isinstance(beh, dict) and beh.get("equity_shift_pct") is not None:
                try:
                    predictions["behavioral_sentiment"] = max(
                        -1.0, min(1.0, float(beh["equity_shift_pct"]) / 5.0)
                    )
                except (TypeError, ValueError):
                    pass

            if predictions:
                monitor.stage_predictions(
                    predictions,
                    str(latest_spy_row[0]) if latest_spy_row else _generator.datetime.now(_generator.timezone.utc).strftime("%Y-%m-%d"),
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
        from src.dashboard import generator as _generator  # lazy (patch seams)
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
            "timestamp": _generator.datetime.now().isoformat(),
        }

    def _generate_bocd_regime(self) -> Optional[Dict]:
        """Generate BOCD (Bayesian Online Changepoint Detection) regime signal.

        Uses Adams & MacKay (2007) for real-time structural break detection
        in daily return series without fixed observation windows.

        Returns:
            Dict with regime, regime_change_prob, changepoint_count, etc.
            None if insufficient data.
        """
        from src.dashboard import generator as _generator  # lazy (patch seams)
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
        bocd_data["timestamp"] = _generator.datetime.now().isoformat()

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
        from src.dashboard import generator as _generator  # lazy (patch seams)
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
        portfolio_state = _generator.DATA_DIR / "portfolio_paper.json"
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
        
        target_alloc = self._resolve_live_target_allocations_for_regime(current_regime)
        
        # Pending orders (tail read only)
        orders = []
        orders_log = _generator.DATA_DIR / "orders.jsonl"
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

