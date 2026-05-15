"""
Cashless Collar Overlay - v4.60 Implementation
Tactical options overlay for portfolio drawdown protection.

Writes a cashless collar (OTM call + OTM put) on SPY allocation monthly.
Integrates with EnsembleVoter and SmartRebalanceGate.

Usage:
    python -m src.strategy.collar_overlay status
    python -m src.strategy.collar_overlay recommend
    python -m src.strategy.collar_overlay backtest --start 2006-01-01
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, date, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np

from src.signals.collar_signal import (
    CollarSignalGenerator,
    CollarSignal,
    CollarStrikes,
    CollarRegime,
    CollarState,
    BlackScholesPricer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CollarOverlayStatus(Enum):
    ACTIVE = "active"
    ROLLING = "rolling"
    FROZEN = "frozen"        # During market stress
    DISABLED = "disabled"


@dataclass
class CollarOverlayDecision:
    """Tactical decision from collar overlay."""
    timestamp: str
    status: str

    # Strikes
    call_strike: float
    put_strike: float
    underlying_price: float

    # Allocation shifts
    spy_shift: float
    cash_shift: float        # Premium collected or paid

    # Risk metrics
    max_upside: float        # Capped upside (pct)
    max_downside: float      # Protected floor (pct)
    vix_level: float
    regime: str

    # Meta
    confidence: float
    recommendation: str
    is_actionable: bool

    def to_dict(self) -> dict:
        return asdict(self)


class CollarOverlay:
    """
    Cashless collar overlay manager.

    Monthly cycle:
    1. Generate collar signal (strikes, premiums)
    2. Check for roll (within 5 days of expiry or new month)
    3. Apply allocation shifts
    4. Track P&L attribution
    """

    STATE_FILE = Path(__file__).parent.parent.parent / "data" / "collar_overlay_state.json"
    ATTRIBUTION_FILE = Path(__file__).parent.parent.parent / "data" / "collar_attribution.json"

    # Collar parameters
    ROLL_DAYS_BEFORE = 5       # Start rolling 5 days before expiry
    MIN_HOLDING_DAYS = 20      # Minimum days between rolls
    CRISIS_FREEZE_VIX = 40.0   # Freeze collar above VIX 40

    # Integration weights
    ENSEMBLE_WEIGHT = 0.10     # 10% weight in ensemble voter

    def __init__(self, state_file: Optional[Path] = None):
        self._signal_gen = CollarSignalGenerator()
        self._pricer = BlackScholesPricer()
        self.state_file = state_file or self.STATE_FILE
        self.attribution_file = self.ATTRIBUTION_FILE
        self._state = self._load_state()

    def _load_state(self) -> Dict:
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return self._default_state()

    def _default_state(self) -> Dict:
        return {
            "status": "disabled",
            "last_roll_date": None,
            "current_call_strike": None,
            "current_put_strike": None,
            "current_expiry": None,
            "ytd_premium_collected": 0.0,
            "ytd_premium_paid": 0.0,
            "total_rolls": 0,
            "frozen_until": None,
        }

    def _save_state(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self._state, f, indent=2)

    def get_status(self) -> CollarOverlayStatus:
        status = self._state.get("status", "disabled")
        return CollarOverlayStatus(status)

    def recommend(
        self,
        spot: Optional[float] = None,
        vix: Optional[float] = None,
    ) -> CollarOverlayDecision:
        """Generate collar overlay recommendation."""
        signal = self._signal_gen.generate_signal(spot=spot, vix=vix)

        if not signal.is_valid:
            return CollarOverlayDecision(
                timestamp=datetime.now().isoformat(),
                status=CollarOverlayStatus.DISABLED.value,
                call_strike=0, put_strike=0, underlying_price=signal.underlying_price,
                spy_shift=0, cash_shift=0, max_upside=0, max_downside=0,
                vix_level=signal.vix_level, regime=signal.regime, confidence=0,
                recommendation="Collar disabled: " + signal.reason,
                is_actionable=False,
            )

        # Check if roll is needed
        should_roll = self._should_roll(signal)
        status = CollarOverlayStatus.ROLLING if should_roll else CollarOverlayStatus.ACTIVE

        # Calculate net premium impact on portfolio cash
        notional = signal.collar_notional_pct
        cash_shift = signal.strikes.net_premium / signal.underlying_price * notional

        if should_roll and signal.is_valid:
            recommendation = (
                f"Roll collar: sell {signal.call_strike} call, "
                f"buy {signal.put_strike} put. "
                f"Net: ${signal.strikes.net_premium:.2f}/share"
            )
            self._state["last_roll_date"] = datetime.now().isoformat()
            self._state["total_rolls"] += 1
            self._state["current_call_strike"] = signal.call_strike
            self._state["current_put_strike"] = signal.put_strike
            self._save_state()
        else:
            recommendation = f"Hold collar: {signal.reason}"

        return CollarOverlayDecision(
            timestamp=datetime.now().isoformat(),
            status=status.value,
            call_strike=signal.call_strike,
            put_strike=signal.put_strike,
            underlying_price=signal.underlying_price,
            spy_shift=signal.spy_shift,
            cash_shift=round(cash_shift * 100, 4),  # as percentage
            max_upside=signal.max_upside_pct,
            max_downside=signal.max_downside_pct,
            vix_level=signal.vix_level,
            regime=signal.regime,
            confidence=signal.confidence,
            recommendation=recommendation,
            is_actionable=should_roll,
        )

    def _should_roll(self, signal: CollarSignal) -> bool:
        """Determine if collar should be rolled."""
        last_roll = self._state.get("last_roll_date")

        if last_roll is None:
            return True

        try:
            last_date = datetime.fromisoformat(last_roll)
            days_since_roll = (datetime.now() - last_date).days
        except (ValueError, TypeError):
            return True

        if days_since_roll < self.MIN_HOLDING_DAYS:
            return False

        # Roll if within ROLL_DAYS_BEFORE of estimated expiry
        current_expiry = self._state.get("current_expiry")
        if current_expiry:
            try:
                expiry_date = datetime.fromisoformat(current_expiry)
                days_to_expiry = (expiry_date - datetime.now()).days
                if days_to_expiry <= self.ROLL_DAYS_BEFORE:
                    return True
            except (ValueError, TypeError):
                pass

        # Roll monthly (every ~30 days)
        if days_since_roll >= 30:
            return True

        return False

    def get_allocation_shifts(self, signal_value: float) -> Dict[str, float]:
        """
        Map collar signal to allocation shifts.

        The collar itself doesn't shift allocations directly — it overlays on SPY.
        This provides the effective equity delta after collar.
        """
        if signal_value > 0.7:
            return {"spy": 0.0, "gld": 0.0, "tlt": 0.0}  # collar neutral in risk-on
        elif signal_value > 0.3:
            return {"spy": -0.01, "gld": 0.0, "tlt": 0.01}  # slight defensive
        elif signal_value > -0.3:
            return {"spy": 0.0, "gld": 0.0, "tlt": 0.0}   # neutral
        elif signal_value > -0.7:
            return {"spy": -0.02, "gld": 0.01, "tlt": 0.01}  # defensive
        else:
            return {"spy": -0.03, "gld": 0.02, "tlt": 0.01}  # strongly defensive

    def backtest_collar(
        self,
        prices: Dict[str, List[float]],
        dates: List[str],
        vix_history: Optional[List[float]] = None,
    ) -> Dict:
        """
        Backtest the collar overlay on historical data.

        Simulates monthly collar on SPY allocation:
        - Each month: write OTM call, buy OTM put
        - Track premium P&L
        - Compare hedged vs unhedged returns
        """
        spy_prices = prices.get("SPY", [])
        if len(spy_prices) < 30:
            return {"error": "Insufficient price history"}

        results = {
            "dates": [],
            "unhedged_returns": [],
            "hedged_returns": [],
            "collar_premiums": [],
            "drawdowns_hedged": [],
            "drawdowns_unhedged": [],
        }

        monthly_returns_unhedged = []
        monthly_returns_hedged = []
        peak_hedged = spy_prices[0]
        peak_unhedged = spy_prices[0]

        # Simulate monthly rolls
        roll_interval = 21  # trading days per month
        current_call_strike = None
        current_put_strike = None
        current_net_premium = 0.0

        for i in range(len(spy_prices)):
            date_str = dates[i] if i < len(dates) else str(i)
            spot = spy_prices[i]

            # VIX proxy from history or default
            vix = 16.0
            if vix_history and i < len(vix_history):
                vix = vix_history[i]

            # Roll every 21 days
            if i % roll_interval == 0 or current_call_strike is None:
                strikes = self._signal_gen.calculate_strikes(spot, vix, days_to_expiry=30)
                if strikes.is_cashless or strikes.regime != CollarRegime.CRISIS.value:
                    current_call_strike = strikes.call_strike
                    current_put_strike = strikes.put_strike
                    current_net_premium = strikes.net_premium

            # Calculate hedged return
            if i > 0:
                prev_spot = spy_prices[i - 1]
                unhedged_ret = (spot / prev_spot - 1) * 100

                # Hedged: cap upside at call, floor at put
                if current_call_strike and current_put_strike:
                    capped_spot = min(spot, current_call_strike)
                    floored_spot = max(capped_spot, current_put_strike)
                    hedged_ret = (floored_spot / prev_spot - 1) * 100
                    # Add premium
                    hedged_ret += (current_net_premium / prev_spot) * 100 / 21  # daily accrual
                else:
                    hedged_ret = unhedged_ret

                peak_hedged = max(peak_hedged, spot if not current_put_strike else
                                  min(spot, current_call_strike or float("inf")))
                peak_unhedged = max(peak_unhedged, spot)

                dd_hedged = (spot / peak_hedged - 1) * 100
                dd_unhedged = (spot / peak_unhedged - 1) * 100

                monthly_returns_unhedged.append(unhedged_ret)
                monthly_returns_hedged.append(hedged_ret)

                results["unhedged_returns"].append(unhedged_ret)
                results["hedged_returns"].append(hedged_ret)
                results["drawdowns_hedged"].append(dd_hedged)
                results["drawdowns_unhedged"].append(dd_unhedged)

            results["dates"].append(date_str)
            results["collar_premiums"].append(current_net_premium)

        # Compute summary stats
        if monthly_returns_hedged:
            results["summary"] = {
                "cagr_unhedged": round(np.mean(monthly_returns_unhedged) * 12, 2) if monthly_returns_unhedged else 0,
                "cagr_hedged": round(np.mean(monthly_returns_hedged) * 12, 2),
                "vol_unhedged": round(np.std(monthly_returns_unhedged) * np.sqrt(12), 2) if monthly_returns_unhedged else 0,
                "vol_hedged": round(np.std(monthly_returns_hedged) * np.sqrt(12), 2),
                "max_dd_unhedged": round(min(results["drawdowns_unhedged"]), 2) if results["drawdowns_unhedged"] else 0,
                "max_dd_hedged": round(min(results["drawdowns_hedged"]), 2) if results["drawdowns_hedged"] else 0,
                "sharpe_unhedged": round(
                    np.mean(monthly_returns_unhedged) / np.std(monthly_returns_unhedged) * np.sqrt(12), 3
                ) if monthly_returns_unhedged and np.std(monthly_returns_unhedged) > 0 else 0,
                "sharpe_hedged": round(
                    np.mean(monthly_returns_hedged) / np.std(monthly_returns_hedged) * np.sqrt(12), 3
                ) if np.std(monthly_returns_hedged) > 0 else 0,
            }

        return results


class CollarOverlayIntegrator:
    """
    Integrates collar overlay with the ensemble voter and portfolio system.
    """

    INTEGRATION_WEIGHT = 0.10  # 10% of composite signal

    def __init__(self):
        self.overlay = CollarOverlay()

    def get_ensemble_signal(self) -> Dict:
        """Get collar signal for ensemble voter integration."""
        decision = self.overlay.recommend()

        # Normalize to [-1, 1] for ensemble voter
        if decision.status == CollarOverlayStatus.DISABLED.value:
            signal_value = 0.0
        elif decision.confidence > 80:
            signal_value = 0.3 if decision.max_downside > 3 else 0.1
        elif decision.confidence > 50:
            signal_value = 0.15 if decision.max_downside > 2 else 0.05
        else:
            signal_value = 0.0

        return {
            "source": "collar_overlay",
            "signal": signal_value,
            "weight": self.INTEGRATION_WEIGHT,
            "confidence": decision.confidence,
            "recommendation": decision.recommendation,
            "is_actionable": decision.is_actionable,
        }


def calculate_collar_overlay(
    spot: Optional[float] = None,
    vix: Optional[float] = None,
) -> CollarOverlayDecision:
    """Convenience function for collar overlay recommendations."""
    overlay = CollarOverlay()
    return overlay.recommend(spot=spot, vix=vix)


def get_collar_summary() -> Dict:
    """Get current collar status summary."""
    overlay = CollarOverlay()
    decision = overlay.recommend()
    return {
        "status": decision.status,
        "call_strike": decision.call_strike,
        "put_strike": decision.put_strike,
        "net_premium": decision.spy_shift,
        "confidence": decision.confidence,
        "recommendation": decision.recommendation,
    }


def main():
    """CLI entry point."""
    import sys

    overlay = CollarOverlay()

    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        print("Backtest mode — loading price data...")
        # Quick backtest using real data if available
        db_path = Path(__file__).parent.parent.parent / "data" / "market.db"
        if db_path.exists():
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            df = None
            try:
                import pandas as pd
                df = pd.read_sql_query(
                    "SELECT date, symbol, close FROM prices WHERE symbol IN ('SPY', 'VIX') ORDER BY date",
                    conn
                )
            except Exception:
                pass
            conn.close()

            if df is not None and not df.empty:
                spy_data = df[df["symbol"] == "SPY"].set_index("date")["close"]
                vix_data = df[df["symbol"] == "VIX"].set_index("date")["close"]
                common_dates = spy_data.index.intersection(vix_data.index)

                results = overlay.backtest_collar(
                    prices={"SPY": spy_data[common_dates].values.tolist()},
                    dates=common_dates.tolist(),
                    vix_history=vix_data[common_dates].values.tolist(),
                )

                if "summary" in results:
                    s = results["summary"]
                    print(f"\nCollar Backtest Results:")
                    print(f"  CAGR Unhedged: {s['cagr_unhedged']:.2f}%")
                    print(f"  CAGR Hedged:   {s['cagr_hedged']:.2f}%")
                    print(f"  Vol Unhedged:  {s['vol_unhedged']:.2f}%")
                    print(f"  Vol Hedged:    {s['vol_hedged']:.2f}%")
                    print(f"  Max DD Unhedged: {s['max_dd_unhedged']:.2f}%")
                    print(f"  Max DD Hedged:   {s['max_dd_hedged']:.2f}%")
                    print(f"  Sharpe Unhedged: {s['sharpe_unhedged']:.3f}")
                    print(f"  Sharpe Hedged:   {s['sharpe_hedged']:.3f}")
                else:
                    print(f"Backtest error: {results.get('error', 'Unknown')}")
                return

        print("No price data available for backtest")

    # Status mode (default)
    decision = overlay.recommend()

    print("=" * 60)
    print("CASHLESS COLLAR OVERLAY v4.60")
    print("=" * 60)
    print(f"Status: {decision.status}")
    print(f"Regime: {decision.regime}")
    print(f"VIX: {decision.vix_level:.1f}")
    print()
    if decision.is_actionable:
        print(f"Call Strike (short): ${decision.call_strike:.2f}")
        print(f"Put Strike  (long):  ${decision.put_strike:.2f}")
        print(f"Max Upside:  +{decision.max_upside:.1f}%")
        print(f"Max Downside: -{decision.max_downside:.1f}%")
        print()
    print(f"SPY Shift: {decision.spy_shift:+.2f}%")
    print(f"Confidence: {decision.confidence:.0f}%")
    print(f"Recommendation: {decision.recommendation}")
    print(f"Actionable: {decision.is_actionable}")
    print("=" * 60)


if __name__ == "__main__":
    main()
