"""
Portfolio-Lab v7.02: Staking Yield Integration for Crypto Sleeve

Enhances existing crypto allocation (v4.70) with staking yield modeling.
ETH staking yields (~3-5% APY) turn crypto from pure appreciation play
to income-generating component.

Model:
- Track ETH staking yield from public sources / estimation models
- BTC yield = 0 (no native staking)
- Adjust ETH/BTC split based on staking yield attractiveness
- Report "crypto carry" as portfolio yield enhancement

Key Insight (2026 context):
Institutional crypto allocation is shifting from speculative to structured.
ETH staking yields (~3-5% APY) above risk-free rate provide genuine carry.
For our 5% crypto sleeve, staking yield adds ~15-25 bps to portfolio return.

Usage:
    python -m src.strategy.crypto_staking status    # Current staking yields
    python -m src.strategy.crypto_staking estimate  # Yield estimates
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Optional
from enum import Enum

from src.paths import DATA_DIR



__all__ = ['StakingSource', 'ETHStakingMetrics', 'StakingAllocationInfluence', 'ETHStakingModel', 'get_staking_status', 'get_carry_summary']

logger = logging.getLogger(__name__)

STATE_FILE = DATA_DIR / "crypto_staking_state.json"


class StakingSource(Enum):
    """Source of staking yield data."""
    LIVE = "live"           # Fetched from beacon chain / API
    ESTIMATED = "estimated"  # Estimated from staking ratio model
    FALLBACK = "fallback"    # Hardcoded default
    NONE = "none"           # No data available


@dataclass
class ETHStakingMetrics:
    """ETH staking yield metrics."""
    annual_yield: float           # Current annualized staking yield (e.g., 0.035 = 3.5%)
    staking_ratio: float          # % of ETH supply staked (e.g., 0.28 = 28%)
    total_staked_eth: float       # Total ETH staked (in millions)
    source: StakingSource
    timestamp: str
    confidence: float             # 0-1

    # Derived
    real_yield: float             # After inflation (yield - US CPI)
    excess_over_rfr: float        # yield - risk_free_rate
    is_attractive: bool           # yield > risk_free_rate + 2%


@dataclass
class StakingAllocationInfluence:
    """How staking yields affect crypto allocation."""
    eth_preference: float          # -1 (avoid ETH) to +1 (prefer ETH)
    eth_btc_ratio: float           # ETH/(BTC+ETH) target (e.g., 0.50 = 50%)
    btc_weight: float              # Adjusted BTC weight
    eth_weight: float              # Adjusted ETH weight
    total_crypto: float            # Total crypto allocation
    yield_contribution_bps: float  # Staking yield contribution to portfolio (in bps)
    recommendation: str


class ETHStakingModel:
    """
    ETH staking yield estimation model.

    ETH staking yield is determined by:
    1. Total ETH staked (staking ratio)
    2. Total issuance (inflation)
    3. Transaction fees (priority fees + MEV)
    4. Burn rate (EIP-1559)

    Simplified model:
        yield ≈ (issuance_rate + fee_revenue) / staking_ratio

    Where:
    - issuance_rate ~ 0.7% of total supply (post-merge)
    - fee_revenue ~ 2-4% of total supply (varies with network activity)
    - staking_ratio ~ 25-30% (trending up)
    """

    # Base model parameters (can be tuned)
    BASE_ISSUANCE_RATE = 0.007      # 0.7% post-merge ETH issuance (more accurate: ~0.6-0.8%)
    BASE_FEE_REVENUE = 0.005        # 0.5% from tx fees + MEV (realistic for current network activity)
    MIN_STAKING_RATIO = 0.10
    MAX_STAKING_RATIO = 1.0
    DEFAULT_STAKING_RATIO = 0.28    # ~28% as of mid-2026
    FALLBACK_YIELD = 0.035          # 3.5% if model fails

    # Conservative estimates by quarter (2026)
    ESTIMATED_RATIOS = {
        "2026-Q1": 0.26,
        "2026-Q2": 0.28,
        "2026-Q3": 0.30,
        "2026-Q4": 0.32,
    }

    def __init__(self):
        self._state = self._load_state()
        self._risk_free_rate = 0.043  # Approximate 10yr yield
        self._cpi_rate = 0.030        # Approximate CPI

    def _load_state(self) -> Dict:
        """Load persistent state."""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE) as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logger.warning("Failed to load staking state: %s", e)
        return {}

    def _save_state(self):
        """Persist current state."""
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(self._state, f, indent=2)

    def estimate_yield(
        self,
        staking_ratio: Optional[float] = None,
        fee_revenue: Optional[float] = None,
        issuance_rate: Optional[float] = None,
        source: StakingSource = StakingSource.ESTIMATED,
    ) -> ETHStakingMetrics:
        """
        Estimate ETH staking yield from model parameters.

        Args:
            staking_ratio: % of ETH supply staked (None = auto from quarter)
            fee_revenue: Annualized fee+MEV as % of total supply
            issuance_rate: Annual issuance rate

        Returns:
            ETHStakingMetrics with yield estimate
        """
        # Auto-detect staking ratio from current quarter
        if staking_ratio is None:
            now = datetime.now()
            quarter = f"{now.year}-Q{(now.month - 1) // 3 + 1}"
            staking_ratio = self.ESTIMATED_RATIOS.get(quarter, self.DEFAULT_STAKING_RATIO)

        staking_ratio = max(self.MIN_STAKING_RATIO, min(staking_ratio, self.MAX_STAKING_RATIO))

        if issuance_rate is None:
            issuance_rate = self.BASE_ISSUANCE_RATE
        if fee_revenue is None:
            fee_revenue = self.BASE_FEE_REVENUE

        # Total annualized rewards as % of total supply
        total_rewards = issuance_rate + fee_revenue

        # Staking yield = rewards / fraction staked
        annual_yield = total_rewards / staking_ratio if staking_ratio > 0 else 0

        # Clamp to realistic range
        annual_yield = max(0.01, min(annual_yield, 0.08))

        total_staked_eth = 120.0 * staking_ratio  # ~120M ETH total supply

        real_yield = annual_yield - self._cpi_rate
        excess_over_rfr = annual_yield - self._risk_free_rate
        is_attractive = excess_over_rfr > 0.02  # >2% above risk-free rate

        return ETHStakingMetrics(
            annual_yield=round(annual_yield, 4),
            staking_ratio=round(staking_ratio, 4),
            total_staked_eth=round(total_staked_eth, 2),
            source=source,
            timestamp=datetime.now().isoformat(),
            confidence=0.7 if source == StakingSource.ESTIMATED else 0.4,
            real_yield=round(real_yield, 4),
            excess_over_rfr=round(excess_over_rfr, 4),
            is_attractive=is_attractive,
        )

    def get_live_yield(self) -> ETHStakingMetrics:
        """
        Attempt to fetch live staking yield.

        Current implementation uses estimation model.
        Future: integrate beaconcha.in API or similar.
        """
        # For now, use estimated model with slightly higher confidence
        return self.estimate_yield(source=StakingSource.ESTIMATED)

    def get_btc_metrics(self) -> Dict:
        """BTC has no native staking yield."""
        return {
            "yield": 0.0,
            "source": StakingSource.NONE.value,
            "note": "BTC has no native staking yield",
        }

    def compute_crypto_carry(
        self,
        btc_weight: float,
        eth_weight: float,
        eth_metrics: Optional[ETHStakingMetrics] = None,
    ) -> Dict:
        """
        Compute portfolio staking yield contribution.

        Args:
            btc_weight: BTC weight in portfolio (e.g., 0.03 = 3%)
            eth_weight: ETH weight in portfolio (e.g., 0.02 = 2%)
            eth_metrics: ETH staking metrics (fetched if None)

        Returns:
            Dict with yield contribution details
        """
        if eth_metrics is None:
            eth_metrics = self.get_live_yield()

        # Staking yield on ETH portion only
        eth_staking_contribution = eth_weight * eth_metrics.annual_yield
        btc_contribution = 0.0  # BTC no yield

        total_carry = eth_staking_contribution + btc_contribution
        total_crypto = btc_weight + eth_weight

        carry_bps = total_carry * 10000  # Convert to bps of portfolio

        return {
            "eth_staking_yield_pct": eth_metrics.annual_yield * 100,
            "eth_staking_ratio_pct": eth_metrics.staking_ratio * 100,
            "btc_yield_pct": 0.0,
            "eth_carry_bps": eth_staking_contribution * 10000,
            "btc_carry_bps": 0.0,
            "total_carry_bps": round(carry_bps, 2),
            "total_crypto_pct": total_crypto * 100,
            "yield_enhancement_bps": round(carry_bps / total_crypto * 100, 2)
            if total_crypto > 0 else 0,
            "is_attractive": eth_metrics.is_attractive,
            "excess_over_rfr_pct": eth_metrics.excess_over_rfr * 100,
            "real_yield_pct": eth_metrics.real_yield * 100,
            "note": f"ETH staking adds ~{carry_bps:.1f} bps to portfolio return"
            if carry_bps > 0 else "No staking yield contribution",
        }

    def compute_allocation_influence(
        self,
        current_btc_weight: float = 0.03,
        current_eth_weight: float = 0.02,
        base_btc_split: float = 0.60,
        base_eth_split: float = 0.40,
    ) -> StakingAllocationInfluence:
        """
        Compute how staking yields should influence BTC/ETH split.

        Factors:
        - Staking yield attractiveness (yield vs risk-free rate)
        - Higher yield → tilt toward ETH
        - Yield collapse → revert to base split
        """
        eth_metrics = self.get_live_yield()
        total_crypto = current_btc_weight + current_eth_weight

        # Base split
        if total_crypto <= 0:
            return StakingAllocationInfluence(
                eth_preference=0.0,
                eth_btc_ratio=base_eth_split,
                btc_weight=0.0,
                eth_weight=0.0,
                total_crypto=0.0,
                yield_contribution_bps=0.0,
                recommendation="No crypto allocation",
            )

        # Tilt factor based on staking yield attractiveness
        if eth_metrics.is_attractive:
            # Strong staking yield → tilt toward ETH
            tilt = min(eth_metrics.excess_over_rfr / 0.05, 1.0)  # Normalize to [0, 1]
            # Max tilt: 70% ETH / 30% BTC (up from 60/40 base)
            eth_split = base_eth_split + (0.70 - base_eth_split) * tilt  # 0.40 → up to 0.70
            eth_split = min(eth_split, 0.70)
            btc_split = 1.0 - eth_split
            preference = tilt
            rec = f"Tilt toward ETH: staking yield ({eth_metrics.annual_yield*100:.1f}%) "
            rec += f"exceeds risk-free rate by {eth_metrics.excess_over_rfr*100:.1f}pp"
        else:
            # Weak staking yield → back to base
            eth_split = base_eth_split
            btc_split = base_btc_split
            preference = 0.0
            rec = f"Neutral split: staking yield ({eth_metrics.annual_yield*100:.1f}%) "
            rec += "not sufficiently attractive vs risk-free rate"

        eth_w = total_crypto * eth_split
        btc_w = total_crypto * btc_split

        carry = self.compute_crypto_carry(btc_w, eth_w, eth_metrics)

        return StakingAllocationInfluence(
            eth_preference=round(preference, 4),
            eth_btc_ratio=round(eth_split, 4),
            btc_weight=round(btc_w, 4),
            eth_weight=round(eth_w, 4),
            total_crypto=round(total_crypto, 4),
            yield_contribution_bps=round(carry["total_carry_bps"], 2),
            recommendation=rec,
        )

    def set_external_staking_ratio(self, ratio: float):
        """Set external staking ratio (e.g., from beaconchain API)."""
        self._state["last_external_ratio"] = ratio
        self._state["ratio_update_time"] = datetime.now().isoformat()
        self._save_state()

    def set_risk_free_rate(self, rate: float):
        """Update risk-free rate (e.g., from FRED feed)."""
        self._risk_free_rate = rate


# ─── Standalone Functions ─────────────────────────────────────────────────────


def get_staking_status() -> Dict:
    """Get current staking yield status."""
    model = ETHStakingModel()
    eth = model.get_live_yield()
    btc = model.get_btc_metrics()
    return {
        "eth": asdict(eth),
        "btc": btc,
        "risk_free_rate": model._risk_free_rate,
        "cpi_rate": model._cpi_rate,
    }


def get_carry_summary(crypto_weight: float = 0.05, eth_share: float = 0.40) -> Dict:
    """Get crypto carry contribution summary."""
    model = ETHStakingModel()
    return model.compute_crypto_carry(
        btc_weight=crypto_weight * (1 - eth_share),
        eth_weight=crypto_weight * eth_share,
    )


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Crypto Staking Yield Model (v7.02)"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="status",
        choices=["status", "estimate", "carry", "influence"],
        help="Mode: status (default), estimate, carry, influence",
    )
    parser.add_argument(
        "--ratio", type=float, help="Staking ratio override (e.g., 0.30)"
    )
    parser.add_argument(
        "--btc-w", type=float, default=0.03, help="BTC weight (default: 0.03)"
    )
    parser.add_argument(
        "--eth-w", type=float, default=0.02, help="ETH weight (default: 0.02)"
    )

    args = parser.parse_args()
    model = ETHStakingModel()

    if args.mode == "status":
        eth = model.get_live_yield()
        print("=" * 65)
        print("v7.02 CRYPTO STAKING YIELD MODEL")
        print("=" * 65)
        print(f"ETH Staking Yield:    {eth.annual_yield:.2%}")
        print(f"Staking Ratio:        {eth.staking_ratio:.1%}")
        print(f"Real Yield (ex-CPI):  {eth.real_yield:.2%}")
        print(f"Excess over RFR:      {eth.excess_over_rfr:.2%}")
        print(f"Attractive:           {eth.is_attractive}")
        print(f"Source:               {eth.source.value}")
        print(f"Confidence:           {eth.confidence:.0%}")
        print()
        print("BTC:")
        print("  Staking Yield: 0.00% (no native staking)")

    elif args.mode == "estimate":
        ratio = args.ratio if args.ratio else None
        eth = model.estimate_yield(staking_ratio=ratio)
        print(f"ETH Staking Yield (ratio={eth.staking_ratio:.1%}): {eth.annual_yield:.2%}")
        print(f"  Real Yield: {eth.real_yield:.2%}")
        print(f"  Excess over RFR ({model._risk_free_rate:.1%}): {eth.excess_over_rfr:.2%}")
        print(f"  Attractive: {eth.is_attractive}")

        # Parameter scan
        print("\nSensitivity (staking ratio × yield):")
        for ratio_test in [0.20, 0.25, 0.28, 0.30, 0.35, 0.40]:
            r = model.estimate_yield(staking_ratio=ratio_test)
            print(f"  Ratio {ratio_test:.0%} → Yield {r.annual_yield:.2%}  "
                  f"(real: {r.real_yield:.2%}, attractive: {r.is_attractive})")

    elif args.mode == "carry":
        carry = model.compute_crypto_carry(
            btc_weight=args.btc_w,
            eth_weight=args.eth_w,
        )
        print("=" * 65)
        print("CRYPTO CARRY CONTRIBUTION")
        print("=" * 65)
        print(f"ETH Staking Yield:    {carry['eth_staking_yield_pct']:.2f}%")
        print(f"ETH Staking Ratio:    {carry['eth_staking_ratio_pct']:.1f}%")
        print(f"ETH Carry:            {carry['eth_carry_bps']:.2f} bps")
        print(f"BTC Carry:            {carry['btc_carry_bps']:.2f} bps")
        print(f"Total Carry:          {carry['total_carry_bps']:.2f} bps")
        print(f"Is Attractive:        {carry['is_attractive']}")
        print()
        print(f"Note: {carry['note']}")

    elif args.mode == "influence":
        inf = model.compute_allocation_influence(
            current_btc_weight=args.btc_w,
            current_eth_weight=args.eth_w,
        )
        print("=" * 65)
        print("STAKING YIELD ALLOCATION INFLUENCE")
        print("=" * 65)
        print(f"ETH Preference:        {inf.eth_preference:+.2f}")
        print(f"Target ETH/BTC Ratio:  {inf.eth_btc_ratio:.1%} ETH")
        print(f"Adjusted BTC Weight:   {inf.btc_weight:.2%}")
        print(f"Adjusted ETH Weight:   {inf.eth_weight:.2%}")
        print(f"Yield Contribution:    {inf.yield_contribution_bps:.1f} bps")
        print()
        print(f"Recommendation:")
        print(f"  {inf.recommendation}")


if __name__ == "__main__":
    main()
