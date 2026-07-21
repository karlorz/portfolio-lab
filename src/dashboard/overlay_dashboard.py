"""
Overlay Dashboard Data Generator - v4.91
Collects signals from all tactical overlays and generates dashboard-ready JSON.

Feeds the frontend with:
- Collar status: strikes, premium, regime, VIX level
- Crypto allocation: BTC/ETH weight, momentum, vol regime
- Bond duration: TLT/IEF/SHY split, curve regime
- Calendar: urgency modifier, active windows, next window
- Kurtosis: regime, KER, strategy routing
- Unified: composite portfolio recommendation

Usage:
    python -m src.dashboard.overlay_dashboard generate
    python -m src.dashboard.overlay_dashboard status
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any, Tuple

from pathlib import Path

from src.paths import DATA_DIR
from src.backtest.metrics import save_results_json
from src.dashboard.kill_authority import load_kill_switch_payload

logger = logging.getLogger(__name__)


def _crypto_status_text(
    *,
    composite: float,
    btc_pf: float,
    eth_pf: float,
    btc_mom: float,
    eth_mom: float,
) -> str:
    """Build status_text that headlines the active sleeve asset(s).

    When portfolio notional is ETH-only (btc_pf≈0), do not lead with BTC
    momentum — that misleads operators reading the crypto panel.
    """
    head = f"Crypto: {composite:.1%}"
    eps = 1e-9
    btc_on = btc_pf > eps
    eth_on = eth_pf > eps
    if btc_on and eth_on:
        return f"{head}, BTC {btc_mom:+.1%} 6m / ETH {eth_mom:+.1%} 6m"
    if eth_on and not btc_on:
        return f"{head}, ETH {eth_mom:+.1%} 6m"
    if btc_on and not eth_on:
        return f"{head}, BTC {btc_mom:+.1%} 6m"
    # Flat sleeve — still show both for context
    return f"{head}, BTC {btc_mom:+.1%} 6m / ETH {eth_mom:+.1%} 6m"


@dataclass
class OverlayDashboardData:
    """Complete overlay dashboard data ready for frontend consumption."""
    timestamp: str
    generated_at: str

    # Collar overlay (v4.60)
    collar: Dict[str, Any]

    # Crypto tactical (v4.70)
    crypto: Dict[str, Any]

    # Bond duration rotation (v4.80)
    bond_duration: Dict[str, Any]

    # Calendar seasonality (v3.50)
    calendar: Dict[str, Any]

    # Kurtosis regime (v4.91)
    kurtosis: Dict[str, Any]

    # Mean reversion (v4.81)
    mean_reversion: Dict[str, Any]

    # Unified orchestrator (v4.90)
    unified: Dict[str, Any]

    # Summary
    active_overlays: int
    total_overlays: int
    portfolio_risk: str  # low, moderate, elevated, high
    alerts: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


class OverlayDashboardGenerator:
    """
    Generates dashboard JSON for all tactical overlays.

    Collects live signals from each overlay module, formats them
    for frontend consumption, and saves to a single JSON file.
    """

    OUTPUT_PATH = DATA_DIR / "dashboard" / "overlay_dashboard.json"

    def __init__(self, data_dir: Path | str | None = None):
        # Authority artifacts (kill_switch.json) live under data_dir.
        # Injectable for tests so live halt cannot pollute pure overlay scoring.
        self.data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
        self._ensure_dirs()

    def _ensure_dirs(self):
        self.OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _stamp_freshness(block: Dict[str, Any], produced_at: str | None = None) -> Dict[str, Any]:
        """Ensure overlay sections carry generated_at for signal-staleness TTL.

        signals.json maps these blocks into collar/crypto_allocation/etc. and
        marks optional sections without a freshness field as unavailable, which
        blocks all-fresh PASS and keeps sticky kill incidents alive even when
        producers just ran successfully.
        """
        ts = produced_at or datetime.now(timezone.utc).isoformat()
        block.setdefault("timestamp", ts)
        block.setdefault("generated_at", ts)
        return block

    def _load_collar_signal_file(self) -> Optional[Dict[str, Any]]:
        """Load SIGNALS_DIR/collar_signal.json SSOT when present."""
        candidates = []
        try:
            from src.signals.collar_signal import CollarSignalGenerator
            candidates.append(Path(CollarSignalGenerator.OUTPUT_PATH))
        except Exception:
            pass
        try:
            from src.paths import SIGNALS_DIR, DATA_DIR
            candidates.append(Path(SIGNALS_DIR) / "collar_signal.json")
            candidates.append(Path(DATA_DIR) / "signals" / "collar_signal.json")
        except Exception:
            pass
        seen = set()
        for path in candidates:
            try:
                key = str(path.resolve()) if path.exists() else str(path)
            except OSError:
                key = str(path)
            if key in seen:
                continue
            seen.add(key)
            try:
                if not path.exists():
                    continue
                data = json.loads(path.read_text())
                if not isinstance(data, dict):
                    continue
                if data.get("call_strike") is None and data.get("put_strike") is None:
                    continue
                return data
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
        return None

    def _live_spy_mark(self) -> Optional[float]:
        """Current SPY mark from price SSOT (prices.json / public)."""
        try:
            from src.paths import PUBLIC_DATA_DIR, PRICES_JSON

            for path in (
                Path(PRICES_JSON),
                Path(PUBLIC_DATA_DIR) / "prices.json",
                Path(DATA_DIR) / "prices.json",
            ):
                if not path.is_file():
                    continue
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    continue
                if not isinstance(data, dict):
                    continue
                # shapes: {"SPY": {"close": x}} or {"SPY": x} or nested latest
                spy = data.get("SPY") or data.get("spy")
                if isinstance(spy, (int, float)) and float(spy) > 0:
                    return float(spy)
                if isinstance(spy, dict):
                    for k in ("close", "price", "last", "mark"):
                        v = spy.get(k)
                        if v is not None and float(v) > 0:
                            return float(v)
                # series form
                if isinstance(spy, list) and spy:
                    last = spy[-1]
                    if isinstance(last, (int, float)):
                        return float(last)
                    if isinstance(last, dict):
                        for k in ("close", "price", "last"):
                            if last.get(k) is not None:
                                return float(last[k])
        except Exception:  # noqa: BLE001
            return None
        return None

    def _get_collar_data(self) -> Dict[str, Any]:
        """Collect collar overlay data; refresh underlying from live SPY marks.

        Saved collar_signal.json can stick to a stale underlying while SPY
        marks advance. Prefer live generate when mark drift or age exceeds
        thresholds; otherwise stamp status and demote active when stale.
        """
        STALE_AGE_MINUTES = 30
        SPY_TICK_THRESHOLD = 0.50  # dollars — |underlying − SPY| above this is stale
        try:
            live_spy = self._live_spy_mark()
            saved = self._load_collar_signal_file()
            if saved is not None:
                strikes = saved.get("strikes") if isinstance(saved.get("strikes"), dict) else {}
                call = saved.get("call_strike", strikes.get("call_strike"))
                put = saved.get("put_strike", strikes.get("put_strike"))
                net = strikes.get("net_premium", saved.get("net_premium", 0.0))
                is_cashless = strikes.get("is_cashless", saved.get("is_cashless", False))
                regime = saved.get("regime", "unknown")
                underlying = saved.get("underlying_price")
                try:
                    underlying_f = float(underlying) if underlying is not None else None
                except (TypeError, ValueError):
                    underlying_f = None
                stale_reason = None
                # Age gate
                ts = saved.get("timestamp") or saved.get("generated_at")
                age_min = None
                if ts:
                    try:
                        from datetime import datetime, timezone

                        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                        if t.tzinfo is None:
                            t = t.replace(tzinfo=timezone.utc)
                        age_min = (
                            datetime.now(timezone.utc) - t.astimezone(timezone.utc)
                        ).total_seconds() / 60.0
                        if age_min > STALE_AGE_MINUTES:
                            stale_reason = f"age={age_min:.0f}m>{STALE_AGE_MINUTES}m"
                    except (TypeError, ValueError):
                        pass
                # Mark drift gate — force regenerate when SPY moved
                if (
                    live_spy is not None
                    and underlying_f is not None
                    and abs(live_spy - underlying_f) > SPY_TICK_THRESHOLD
                ):
                    drift = abs(live_spy - underlying_f)
                    stale_reason = (
                        f"|underlying-SPY|={drift:.2f}>{SPY_TICK_THRESHOLD}"
                    )

                # Prefer regenerate when we have a live SPY mark and are stale
                if stale_reason and live_spy is not None:
                    # Regenerate with current SPY mark
                    from src.signals.collar_signal import CollarSignalGenerator

                    gen = CollarSignalGenerator()
                    signal = gen.generate_signal(spot=live_spy)
                    underlying = float(getattr(signal, "underlying_price", 0) or live_spy)
                    payload = {
                        "active": bool(signal.is_valid),
                        "regime": signal.regime,
                        "call_strike": signal.call_strike,
                        "put_strike": signal.put_strike,
                        "net_premium": signal.strikes.net_premium,
                        "is_cashless": signal.strikes.is_cashless,
                        "max_upside_pct": signal.max_upside_pct,
                        "max_downside_pct": signal.max_downside_pct,
                        "vix_level": signal.vix_level,
                        "confidence": signal.confidence,
                        "underlying_price": underlying if underlying > 0 else live_spy,
                        "spy_mark": live_spy,
                        "spot_source": "prices.json_refresh",
                        "source": "collar_refresh_from_spy_marks",
                        "live_authoritative": False,
                        "role": "advisory_overlay",
                        "stale_reason_cleared": stale_reason,
                        "status_text": (
                            f"Collar refreshed from SPY ${live_spy:.2f}: "
                            f"{signal.regime}, call ${signal.call_strike:.0f}, "
                            f"put ${signal.put_strike:.0f}"
                        ),
                    }
                    return self._stamp_freshness(
                        payload, getattr(signal, "timestamp", None)
                    )

                active = bool(saved.get("is_valid", True))
                status = (
                    f"Collar: {regime}, call ${float(call):.0f}, put ${float(put):.0f}"
                )
                if stale_reason:
                    active = False
                    status = f"STALE ({stale_reason}) — {status}"
                if live_spy is not None and underlying_f is not None:
                    status += f", underlying ${underlying_f:.2f} vs SPY ${live_spy:.2f}"
                return self._stamp_freshness({
                    "active": active,
                    "regime": regime,
                    "call_strike": call,
                    "put_strike": put,
                    "net_premium": net,
                    "is_cashless": is_cashless,
                    "max_upside_pct": saved.get("max_upside_pct"),
                    "max_downside_pct": saved.get("max_downside_pct"),
                    "vix_level": saved.get("vix_level"),
                    "confidence": saved.get("confidence"),
                    "underlying_price": underlying_f,
                    "spy_mark": live_spy,
                    "source": "collar_signal.json",
                    "stale": bool(stale_reason),
                    "stale_reason": stale_reason,
                    "live_authoritative": False,
                    "role": "advisory_overlay",
                    "status_text": status,
                }, saved.get("timestamp") or saved.get("generated_at"))

            from src.signals.collar_signal import generate_collar_signal
            # Live fetch inside generator — do not hardcode spot=550 / vix=16
            # Prefer explicit SPY mark when available
            if live_spy is not None:
                from src.signals.collar_signal import CollarSignalGenerator

                signal = CollarSignalGenerator().generate_signal(spot=live_spy)
            else:
                signal = generate_collar_signal()
            underlying = float(getattr(signal, "underlying_price", 0) or 0)
            return self._stamp_freshness({
                "active": signal.is_valid,
                "regime": signal.regime,
                "call_strike": signal.call_strike,
                "put_strike": signal.put_strike,
                "net_premium": signal.strikes.net_premium,
                "is_cashless": signal.strikes.is_cashless,
                "max_upside_pct": signal.max_upside_pct,
                "max_downside_pct": signal.max_downside_pct,
                "vix_level": signal.vix_level,
                "confidence": signal.confidence,
                "underlying_price": underlying if underlying > 0 else None,
                "spy_mark": live_spy,
                "spot_source": "collar_signal_generate",
                "source": "generate_collar_signal",
                "live_authoritative": False,
                "role": "advisory_overlay",
                "status_text": f"Collar: {signal.regime}, "
                               f"call ${signal.call_strike:.0f}, "
                               f"put ${signal.put_strike:.0f}"
                               + (f", spot ${underlying:.0f}" if underlying > 0 else ""),
            }, getattr(signal, "timestamp", None))
        except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            return {"active": False, "error": str(e)}

    def _get_crypto_data(self) -> Dict[str, Any]:
        """Collect crypto tactical data."""
        try:
            from src.signals.crypto_momentum import (
                CryptoMomentumCalculator,
                generate_crypto_signal,
            )
            signal = generate_crypto_signal()
            # target_weight is sleeve share (0–1 within crypto sleeve), not a
            # portfolio fraction. Scale by BASE then renormalize to composite so
            # btc_weight + eth_weight == total_crypto.
            base_w = float(CryptoMomentumCalculator.BASE_CRYPTO_WEIGHT)
            btc_sleeve = float(signal.btc_signal.target_weight)
            eth_sleeve = float(signal.eth_signal.target_weight)
            raw_btc = btc_sleeve * base_w
            raw_eth = eth_sleeve * base_w
            raw_sum = raw_btc + raw_eth
            composite = float(signal.composite_weight)
            if raw_sum > 0 and composite > 0:
                scale = composite / raw_sum
                btc_pf = raw_btc * scale
                eth_pf = raw_eth * scale
            else:
                btc_pf = 0.0
                eth_pf = 0.0
            return self._stamp_freshness({
                "active": signal.is_valid,
                # Portfolio fractions (sum ≈ total_crypto)
                "btc_weight": round(btc_pf, 6),
                "eth_weight": round(eth_pf, 6),
                "total_crypto": composite,
                # Sleeve shares preserved for diagnostics
                "btc_sleeve_share": round(btc_sleeve, 4),
                "eth_sleeve_share": round(eth_sleeve, 4),
                "weight_unit": "portfolio_fraction",
                "live_authoritative": False,
                "role": "advisory_non_routed",
                "btc_momentum_6m": signal.btc_signal.momentum_6m,
                "eth_momentum_6m": signal.eth_signal.momentum_6m,
                "btc_vol_regime": signal.btc_signal.vol_regime,
                "eth_vol_regime": signal.eth_signal.vol_regime,
                "confidence": signal.confidence,
                "status_text": _crypto_status_text(
                    composite=composite,
                    btc_pf=btc_pf,
                    eth_pf=eth_pf,
                    btc_mom=float(signal.btc_signal.momentum_6m or 0),
                    eth_mom=float(signal.eth_signal.momentum_6m or 0),
                ),
            }, getattr(signal, "timestamp", None))
        except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            return {"active": False, "error": str(e)}

    def _get_bond_duration_data(self) -> Dict[str, Any]:
        """Collect bond duration rotation data."""
        try:
            from src.signals.bond_duration_signal import generate_bond_duration_signal
            signal = generate_bond_duration_signal()
            # MagicMock auto-attrs are truthy; only treat explicit True as defaults.
            _ud = getattr(signal, "using_defaults", False)
            using_defaults = _ud is True
            return self._stamp_freshness({
                "active": bool(signal.is_valid) and not using_defaults,
                "yield_10y": signal.yield_10y,
                "yield_2y": signal.yield_2y,
                "spread": signal.spread_10y2y,
                "curve_regime": signal.curve_regime,
                "rate_direction": signal.rate_direction,
                "tlt_weight": signal.tlt_weight,
                "ief_weight": signal.ief_weight,
                "shy_weight": signal.shy_weight,
                "weight_unit": "sleeve_fraction",
                "live_authoritative": False,
                "role": "advisory_non_routed",
                "effective_duration": signal.effective_duration,
                "position": signal.position,
                "confidence": signal.confidence,
                "using_defaults": using_defaults,
                "source_mode": getattr(signal, "source_mode", "live")
                if not isinstance(getattr(signal, "source_mode", None), type(signal))
                else "live",
                "source_status": getattr(signal, "source_status", "ok")
                if isinstance(getattr(signal, "source_status", "ok"), str)
                else "ok",
                "status_text": f"Bonds: {signal.position} "
                               f"({signal.curve_regime}/{signal.rate_direction}), "
                               f"dur {signal.effective_duration:.0f}yr",
            }, getattr(signal, "timestamp", None)
            if isinstance(getattr(signal, "timestamp", None), str)
            else None)
        except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            return {"active": False, "error": str(e)}

    def _get_calendar_data(self) -> Dict[str, Any]:
        """Collect calendar seasonality data."""
        try:
            from src.signals.calendar_seasonality import check_calendar
            signal = check_calendar()
            # Production freshness must be wall-clock time of this generate(),
            # not assessment_date midnight — midnight stamps false-stale by
            # mid-afternoon and re-arm signal_staleness kills every day-roll.
            assessment = getattr(signal, "assessment_date", None)
            modifier = float(signal.urgency_modifier)
            applied_to_targets = False  # advisory only — paper targets unscaled
            block = {
                "active": signal.is_trading_day,
                "modifier": modifier,
                "active_windows": signal.active_windows,
                "next_window": signal.next_window,
                "days_to_next": signal.days_to_next_window,
                "recommendation": signal.recommendation,
                "effect": signal.effect,
                "applies_to_target_allocations": applied_to_targets,
                "role": "advisory_non_routed",
                "live_authoritative": False,
                "status_text": (
                    f"Calendar: {modifier:.2f}x (not applied to target_allocations), "
                    f"{len(signal.active_windows)} windows active"
                    if modifier != 1.0
                    else f"Calendar: {modifier:.2f}x, "
                    f"{len(signal.active_windows)} windows active"
                ),
            }
            if assessment is not None:
                block["assessment_date"] = assessment
            # Wall-clock production stamp (default in _stamp_freshness).
            return self._stamp_freshness(block)
        except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            return {"active": False, "error": str(e)}

    def _get_kurtosis_data(self) -> Dict[str, Any]:
        """Collect kurtosis regime data."""
        try:
            from src.regime.kurtosis_regime import detect_kurtosis_regime
            signal = detect_kurtosis_regime()
            return self._stamp_freshness({
                "active": True,
                "kurtosis_20d": signal.kurtosis_20d,
                "kurtosis_60d": signal.kurtosis_60d,
                "ker_ratio": signal.ker_ratio,
                "regime": signal.regime,
                "transitioning": signal.is_transitioning,
                "strategy_preference": signal.strategy_preference,
                "tsom_weight": signal.tsom_weight,
                "mr_weight": signal.mr_weight,
                "fat_tail_risk": signal.fat_tail_risk,
                "live_authoritative": False,
                "role": "advisory_non_routed",
                "status_text": f"Kurtosis: {signal.regime} "
                               f"(k={signal.kurtosis_60d:.1f}, KER={signal.ker_ratio:.2f})",
            }, getattr(signal, "timestamp", None))
        except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            return {"active": False, "error": str(e)}

    def _get_mean_reversion_data(self) -> Dict[str, Any]:
        """Mean reversion overlay removed v9.38."""
        return {"active": False, "status_text": "MR: disabled"}

    def _get_unified_data(self) -> Dict[str, Any]:
        """Collect unified orchestrator data."""
        try:
            from src.strategy.unified_orchestrator import get_unified_recommendation
            rec = get_unified_recommendation()
            return {
                "active": True,
                "spy": rec.spy,
                "gld": rec.gld,
                "tlt": rec.tlt,
                "ief": rec.ief,
                "shy": rec.shy,
                "btc": rec.btc,
                "eth": rec.eth,
                "estimated_sharpe": rec.estimated_sharpe,
                "conflict_count": rec.conflict_count,
                "calendar_modifier": rec.calendar_modifier,
                "execution_rec": rec.execution_recommendation,
                "status_text": f"Unified: SPY {rec.spy:.1%}, GLD {rec.gld:.1%}, "
                               f"Sharpe est {rec.estimated_sharpe:.3f}",
            }
        except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            return {"active": False, "error": str(e)}

    def _assess_portfolio_risk(self, data: Dict) -> Tuple[str, List[str]]:
        """Assess overall portfolio risk level and generate alerts."""
        alerts = []
        risk_score = 0

        # Kill authority (multi-surface honesty): never claim normal under halt.
        kill = load_kill_switch_payload(self.data_dir)
        if isinstance(kill, dict) and kill.get("enabled"):
            level = str(kill.get("level") or "unknown").lower()
            reason = kill.get("reason")
            message = kill.get("message")
            human = (
                message
                if isinstance(message, str) and message.strip()
                else (str(reason) if reason is not None else "kill switch enabled")
            )
            alerts.append(f"Kill switch {level}: {human}")
            if level == "halt":
                risk_score += 5
            elif level in {"restrict", "liquidate"}:
                risk_score += 4
            else:
                # warning / advisory enabled kill
                risk_score += 3

        # Collar risk
        collar = data.get("collar", {})
        if collar.get("vix_level", 0) > 30:
            alerts.append(f"VIX elevated ({collar['vix_level']:.0f}) — collar active")
            risk_score += 2
        elif collar.get("vix_level", 0) > 25:
            risk_score += 1

        # Crypto risk
        crypto = data.get("crypto", {})
        btc_vol = crypto.get("btc_vol_regime", "")
        if btc_vol == "extreme":
            alerts.append("BTC vol extreme — crypto position exited")
            risk_score += 2
        elif btc_vol == "high":
            risk_score += 1

        # Kurtosis risk
        kurt = data.get("kurtosis", {})
        if kurt.get("fat_tail_risk", 0) > 0.7:
            alerts.append(f"Fat tail risk elevated ({kurt['fat_tail_risk']:.1%})")
            risk_score += 2

        # Bond risk
        bond = data.get("bond_duration", {})
        if bond.get("curve_regime") == "inverted":
            alerts.append("Yield curve inverted — defensive bond posture")
            risk_score += 1

        # Unified conflicts
        unified = data.get("unified", {})
        if unified.get("conflict_count", 0) > 0:
            alerts.append(f"{unified['conflict_count']} overlay conflict(s) detected")
            risk_score += unified["conflict_count"]

        if risk_score >= 5:
            risk_level = "high"
        elif risk_score >= 3:
            risk_level = "elevated"
        elif risk_score >= 1:
            risk_level = "moderate"
        else:
            risk_level = "low"

        return risk_level, alerts

    def generate(self) -> OverlayDashboardData:
        """Generate complete dashboard data."""
        data = {
            "collar": self._get_collar_data(),
            "crypto": self._get_crypto_data(),
            "bond_duration": self._get_bond_duration_data(),
            "calendar": self._get_calendar_data(),
            "kurtosis": self._get_kurtosis_data(),
            "mean_reversion": self._get_mean_reversion_data(),
            "unified": self._get_unified_data(),
        }

        risk_level, alerts = self._assess_portfolio_risk(data)

        active = sum(1 for v in data.values() if v.get("active"))
        total = len(data)

        return OverlayDashboardData(
            timestamp=datetime.now(timezone.utc).isoformat(),
            generated_at=datetime.now(timezone.utc).isoformat(),
            collar=data["collar"],
            crypto=data["crypto"],
            bond_duration=data["bond_duration"],
            calendar=data["calendar"],
            kurtosis=data["kurtosis"],
            mean_reversion=data["mean_reversion"],
            unified=data["unified"],
            active_overlays=active,
            total_overlays=total,
            portfolio_risk=risk_level,
            alerts=alerts,
        )

    def save(self, dashboard: OverlayDashboardData):
        payload = dashboard.to_dict()
        # Operator lag detection: stamp code tip when available
        try:
            from src.dashboard.generator import _stamp_generator_git_sha

            payload = _stamp_generator_git_sha(payload)
        except Exception:  # noqa: BLE001 — never block overlay save on stamp
            pass

        private_path = Path(self.OUTPUT_PATH)
        public_path = None
        paths_identical = True
        try:
            from src.paths import PUBLIC_DATA_DIR

            public_path = Path(PUBLIC_DATA_DIR) / "overlay_dashboard.json"
            try:
                paths_identical = private_path.resolve() == public_path.resolve()
            except OSError:
                paths_identical = False
        except Exception:  # noqa: BLE001 — PUBLIC_DATA_DIR may be unavailable
            public_path = None
            paths_identical = True

        # Intent stamp before private write (mirrors rebalance_health dual-write)
        try:
            from src.dashboard.generator import _attach_dual_write_provenance

            payload = _attach_dual_write_provenance(
                payload,
                private_path=private_path,
                public_path=public_path,
                dual_write_attempted=bool(public_path) and not paths_identical,
                dual_write_ok=None if (public_path and not paths_identical) else True,
                paths_identical=paths_identical,
            )
        except Exception:  # noqa: BLE001 — never block overlay save on provenance
            pass

        private_path.parent.mkdir(parents=True, exist_ok=True)
        save_results_json(payload, output_path=str(private_path))

        # Dual-write live PUBLIC SSOT when distinct from private dashboard tree.
        if public_path is not None and not paths_identical:
            try:
                public_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    from src.dashboard.generator import _attach_dual_write_provenance

                    payload = _attach_dual_write_provenance(
                        payload,
                        private_path=private_path,
                        public_path=public_path,
                        dual_write_attempted=True,
                        dual_write_ok=True,
                        paths_identical=False,
                    )
                    # Keep private in sync with success completeness block
                    save_results_json(payload, output_path=str(private_path))
                except Exception:  # noqa: BLE001
                    pass
                save_results_json(payload, output_path=str(public_path))
            except OSError as e:
                logger.warning("overlay public dual-write skipped: %s", e)
                try:
                    from src.dashboard.generator import _attach_dual_write_provenance

                    payload = _attach_dual_write_provenance(
                        payload,
                        private_path=private_path,
                        public_path=public_path,
                        dual_write_attempted=True,
                        dual_write_ok=False,
                        paths_identical=False,
                        note=str(e),
                    )
                    save_results_json(payload, output_path=str(private_path))
                except Exception:  # noqa: BLE001
                    pass
        logger.info("Dashboard saved to %s", self.OUTPUT_PATH)


def generate_overlay_dashboard() -> OverlayDashboardData:
    """Convenience function."""
    gen = OverlayDashboardGenerator()
    return gen.generate()


def main():
    import sys
    gen = OverlayDashboardGenerator()
    dashboard = gen.generate()

    logger.info("=" * 60)
    logger.info("OVERLAY DASHBOARD v4.91")
    logger.info("=" * 60)
    logger.info("Generated: %s", dashboard.generated_at)
    logger.info("Active: %s/%s", dashboard.active_overlays, dashboard.total_overlays)
    logger.info("Risk Level: %s", dashboard.portfolio_risk.upper())
    logger.info("")

    for name, data in [
        ("Collar", dashboard.collar),
        ("Crypto", dashboard.crypto),
        ("Bond Duration", dashboard.bond_duration),
        ("Calendar", dashboard.calendar),
        ("Kurtosis", dashboard.kurtosis),
        ("Mean Reversion", dashboard.mean_reversion),
        ("Unified", dashboard.unified),
    ]:
        status_text = data.get("status_text", data.get("error", "N/A"))
        flag = "[active]" if data.get("active") else "[inactive]"
        logger.info("  %s %-18s %s", flag, name, status_text)

    logger.info("")
    if dashboard.alerts:
        logger.info("Alerts:")
        for alert in dashboard.alerts:
            logger.info("  [alert] %s", alert)
    else:
        logger.info("No alerts — all systems normal")
    logger.info("=" * 60)

    if "--save" in sys.argv:
        gen.save(dashboard)
        logger.info("Saved to %s", gen.OUTPUT_PATH)


if __name__ == "__main__":
    from src.utils.log_config import configure_logging
    configure_logging()
    main()
