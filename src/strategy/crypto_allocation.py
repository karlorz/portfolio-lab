"""
Crypto Tactical Allocation Overlay - v4.70 Implementation
BTC/ETH momentum-based tactical sleeve funded from GLD.

Rules:
- Entry: 6-month momentum positive + vol regime normal/low
- Exit: momentum negative OR vol extreme (>100% ann.)
- Vol-scaling: target 40% ann. vol, adjust position size
- Hard cap: 5% of portfolio
- Funding: reduces GLD allocation

Usage:
    python -m src.strategy.crypto_allocation recommend
    python -m src.strategy.crypto_allocation status
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Optional, Dict

import numpy as np

from src.signals.crypto_momentum import (
    CryptoMomentumSignalGenerator,
    CryptoCompositeSignal,
    CryptoSignalState,
    CryptoVolRegime,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CryptoAllocationStatus(Enum):
    ACTIVE = "active"
    REDUCED = "reduced"
    FLAT = "flat"
    DISABLED = "disabled"


@dataclass
class CryptoAllocationDecision:
    """Tactical crypto allocation decision."""
    timestamp: str
    status: str

    # Allocation
    btc_weight: float        # % of total portfolio
    eth_weight: float        # % of total portfolio
    total_crypto: float      # % of total portfolio
    gld_reduction: float     # % of total portfolio

    # Risk
    btc_vol: float
    eth_vol: float
    vol_scale: float
    btc_momentum_6m: float
    eth_momentum_6m: float

    # Meta
    confidence: float
    recommendation: str
    is_actionable: bool

    def to_dict(self) -> dict:
        return asdict(self)


class CryptoAllocationOverlay:
    """
    Crypto tactical allocation overlay.

    Operates as a satellite sleeve: 0-5% funded from GLD.
    Not a standalone portfolio — always used within the 46/38/16 framework.
    """

    STATE_FILE = Path(__file__).parent.parent.parent / "data" / "crypto_allocation_state.json"

    # Integration weight
    ENSEMBLE_WEIGHT = 0.05    # 5% weight (lower — crypto is satellite)

    def __init__(self, state_file: Optional[Path] = None):
        self._signal_gen = CryptoMomentumSignalGenerator()
        self.state_file = state_file or self.STATE_FILE
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
            "status": "flat",
            "last_signal_date": None,
            "btc_weight": 0.0,
            "eth_weight": 0.0,
            "total_crypto": 0.0,
            "gld_reduction": 0.0,
            "entry_date": None,
            "exit_date": None,
        }

    def _save_state(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self._state, f, indent=2)

    def get_status(self) -> CryptoAllocationStatus:
        status = self._state.get("status", "flat")
        return CryptoAllocationStatus(status)

    def recommend(self) -> CryptoAllocationDecision:
        """Generate crypto allocation recommendation."""
        signal = self._signal_gen.generate_signal()

        btc_w = signal.btc_signal.target_weight * signal.composite_weight
        eth_w = signal.eth_signal.target_weight * signal.composite_weight

        # Normalize so btc + eth = composite_weight
        if btc_w + eth_w > 0:
            scale = signal.composite_weight / (btc_w + eth_w) if (btc_w + eth_w) > 0 else 0
            btc_w *= scale
            eth_w *= scale

        if signal.signal_state == "flat":
            status = CryptoAllocationStatus.FLAT
            btc_w, eth_w = 0.0, 0.0
            recommendation = f"Crypto flat: {signal.reason}"
            actionable = False
        elif signal.composite_weight <= 0.01:
            status = CryptoAllocationStatus.FLAT
            btc_w, eth_w = 0.0, 0.0
            recommendation = "Crypto allocation below minimum threshold (1%)"
            actionable = False
        elif "reduced" in signal.btc_signal.signal_state or "reduced" in signal.eth_signal.signal_state:
            status = CryptoAllocationStatus.REDUCED
            recommendation = f"Reduced crypto: {signal.reason}"
            actionable = True
        else:
            status = CryptoAllocationStatus.ACTIVE
            recommendation = f"Active crypto: {signal.reason}"
            actionable = True

        # Update state
        self._state["status"] = status.value
        self._state["last_signal_date"] = datetime.now().isoformat()
        self._state["btc_weight"] = round(btc_w, 4)
        self._state["eth_weight"] = round(eth_w, 4)
        self._state["total_crypto"] = round(btc_w + eth_w, 4)
        self._state["gld_reduction"] = round(signal.gld_reduction, 4)
        self._save_state()

        return CryptoAllocationDecision(
            timestamp=datetime.now().isoformat(),
            status=status.value,
            btc_weight=round(btc_w, 4),
            eth_weight=round(eth_w, 4),
            total_crypto=round(btc_w + eth_w, 4),
            gld_reduction=round(signal.gld_reduction, 4),
            btc_vol=signal.btc_signal.vol_30d,
            eth_vol=signal.eth_signal.vol_30d,
            vol_scale=signal.vol_scale_factor,
            btc_momentum_6m=signal.btc_signal.momentum_6m,
            eth_momentum_6m=signal.eth_signal.momentum_6m,
            confidence=signal.confidence,
            recommendation=recommendation,
            is_actionable=actionable,
        )

    def get_allocation_shifts(self) -> Dict[str, float]:
        """Get allocation shifts for portfolio integration."""
        decision = self.recommend()

        return {
            "btc": decision.btc_weight,
            "eth": decision.eth_weight,
            "gld": -decision.gld_reduction,
            "spy": 0.0,
            "tlt": 0.0,
        }

    def backtest(
        self,
        btc_prices: list,
        eth_prices: list,
        gld_prices: list,
        spy_prices: list,
        tlt_prices: list,
        dates: list,
    ) -> Dict:
        """
        Backtest crypto tactical allocation.

        Simulates monthly rebalancing with momentum-based entry/exit.
        Compares baseline 46/38/16 vs 46/(38-crypto)/(16-crypto)/(crypto).
        """
        n = len(dates)
        if n < 180:
            return {"error": "Insufficient data (need >180 days)"}

        results = {
            "dates": [],
            "baseline_returns": [],
            "crypto_returns": [],
            "crypto_weights": [],
            "btc_weights": [],
            "eth_weights": [],
        }

        baseline_values = [1.0]
        crypto_values = [1.0]

        for i in range(180, n):  # Need 180 days for 6m momentum
            date_str = dates[i]

            # Compute signals
            btc_hist = btc_prices[i-180:i+1]
            eth_hist = eth_prices[i-180:i+1]

            # Daily returns
            btc_rets = [(btc_hist[j] / btc_hist[j-1] - 1) for j in range(1, len(btc_hist))]
            eth_rets = [(eth_hist[j] / eth_hist[j-1] - 1) for j in range(1, len(eth_hist))]

            calc = self._signal_gen.calculator
            btc_sig = calc.assess_asset_signal("BTC", btc_hist[-1], btc_hist, btc_rets)
            eth_sig = calc.assess_asset_signal("ETH", eth_hist[-1], eth_hist, eth_rets)

            # Determine crypto weight
            if btc_sig.signal_state == "flat" and eth_sig.signal_state == "flat":
                crypto_w = 0.0
            elif btc_sig.vol_regime == "extreme" or eth_sig.vol_regime == "extreme":
                crypto_w = 0.0
            else:
                btc_contrib = btc_sig.target_weight * 0.03  # base 3%
                eth_contrib = eth_sig.target_weight * 0.03
                crypto_w = min(btc_contrib + eth_contrib, 0.05)

            btc_w = btc_sig.target_weight * crypto_w if crypto_w > 0 else 0
            eth_w = eth_sig.target_weight * crypto_w if crypto_w > 0 else 0

            # Monthly returns (approximate: use daily)
            spy_ret = spy_prices[i] / spy_prices[i-1] - 1 if i > 0 else 0
            gld_ret = gld_prices[i] / gld_prices[i-1] - 1 if i > 0 else 0
            tlt_ret = tlt_prices[i] / tlt_prices[i-1] - 1 if i > 0 else 0
            btc_ret = btc_prices[i] / btc_prices[i-1] - 1 if i > 0 else 0
            eth_ret = eth_prices[i] / eth_prices[i-1] - 1 if i > 0 else 0

            # Baseline: 46/38/16
            base_ret = 0.46 * spy_ret + 0.38 * gld_ret + 0.16 * tlt_ret

            # Crypto overlay: fund from GLD
            crypto_ret = (
                0.46 * spy_ret +
                (0.38 - crypto_w) * gld_ret +
                0.16 * tlt_ret +
                (btc_w + eth_w) * (btc_w/(btc_w+eth_w)*btc_ret + eth_w/(btc_w+eth_w)*eth_ret)
                if (btc_w + eth_w) > 0
                else base_ret
            )

            baseline_values.append(baseline_values[-1] * (1 + base_ret))
            crypto_values.append(crypto_values[-1] * (1 + crypto_ret))

            results["dates"].append(date_str)
            results["baseline_returns"].append(base_ret * 100)
            results["crypto_returns"].append(crypto_ret * 100)
            results["crypto_weights"].append(crypto_w)
            results["btc_weights"].append(btc_w)
            results["eth_weights"].append(eth_w)

        # Compute summary statistics
        base_rets = results["baseline_returns"]
        crypto_rets = results["crypto_returns"]

        if len(base_rets) > 0:
            results["summary"] = {
                "cagr_baseline": round(np.mean(base_rets) * 252, 2),
                "cagr_crypto": round(np.mean(crypto_rets) * 252, 2),
                "vol_baseline": round(np.std(base_rets) * np.sqrt(252), 2),
                "vol_crypto": round(np.std(crypto_rets) * np.sqrt(252), 2),
                "sharpe_baseline": round(
                    np.mean(base_rets) / np.std(base_rets) * np.sqrt(252), 3
                ) if np.std(base_rets) > 0 else 0,
                "sharpe_crypto": round(
                    np.mean(crypto_rets) / np.std(crypto_rets) * np.sqrt(252), 3
                ) if np.std(crypto_rets) > 0 else 0,
                "avg_crypto_weight": round(np.mean(results["crypto_weights"]) * 100, 2),
                "max_crypto_weight": round(max(results["crypto_weights"]) * 100, 2),
            }

        return results


def calculate_crypto_allocation() -> CryptoAllocationDecision:
    """Convenience function."""
    overlay = CryptoAllocationOverlay()
    return overlay.recommend()


def get_crypto_summary() -> Dict:
    """Get current crypto allocation summary."""
    overlay = CryptoAllocationOverlay()
    decision = overlay.recommend()
    return {
        "status": decision.status,
        "btc_weight": decision.btc_weight,
        "eth_weight": decision.eth_weight,
        "total_crypto": decision.total_crypto,
        "confidence": decision.confidence,
        "recommendation": decision.recommendation,
    }


def main():
    import sys
    overlay = CryptoAllocationOverlay()
    decision = overlay.recommend()

    print("=" * 60)
    print("CRYPTO TACTICAL ALLOCATION v4.70")
    print("=" * 60)
    print(f"Status: {decision.status}")
    print(f"BTC Allocation: {decision.btc_weight:.2%}")
    print(f"ETH Allocation: {decision.eth_weight:.2%}")
    print(f"Total Crypto: {decision.total_crypto:.2%}")
    print(f"GLD Reduction: {decision.gld_reduction:.2%}")
    print()
    print(f"BTC 6m Momentum: {decision.btc_momentum_6m:.1%}")
    print(f"ETH 6m Momentum: {decision.eth_momentum_6m:.1%}")
    print(f"BTC 30d Vol: {decision.btc_vol:.1%}")
    print(f"ETH 30d Vol: {decision.eth_vol:.1%}")
    print(f"Vol Scale: {decision.vol_scale:.2f}x")
    print()
    print(f"Confidence: {decision.confidence:.0f}%")
    print(f"Recommendation: {decision.recommendation}")
    print(f"Actionable: {decision.is_actionable}")
    print("=" * 60)


if __name__ == "__main__":
    main()
