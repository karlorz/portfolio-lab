"""
ETF Transaction Cost Table — centralized per-symbol cost data.

Replaces duplicated cost constants in smart_rebalancer.py and
BacktestConfig with a single source of truth.
"""

from typing import Dict, Optional


__all__ = ['get_cost_bps', 'estimate_round_trip_bps', 'estimate_cost_bps']

# One-way transaction costs in basis points.
# Source: typical retail spread + commission for marketable orders.
ETF_COST_BPS: Dict[str, float] = {
    'SPY': 2.0,   # Most liquid US equity ETF
    'QQQ': 2.0,   # Near-SPY liquidity
    'GLD': 5.0,   # Gold — wider spread
    'TLT': 8.0,   # Long-duration Treasury — thinnest book
    'IEF': 6.0,   # Intermediate Treasury
    'EFA': 5.0,   # International equity
    'VXUS': 5.0,  # Ex-US equity
    'MTUM': 4.0,  # Factor ETF
    'VLUE': 5.0,  # Value factor
    'USMV': 4.0,  # Min-vol factor
    'DBC': 10.0,  # Commodities — widest spreads
}

DEFAULT_COST_BPS: float = 5.0  # Fallback for unknown symbols


def get_cost_bps(symbol: str) -> float:
    """Get one-way transaction cost for a symbol in basis points."""
    return ETF_COST_BPS.get(symbol, DEFAULT_COST_BPS)


def estimate_round_trip_bps(symbol: str) -> float:
    """Estimate round-trip (buy + sell) transaction cost in bps."""
    return get_cost_bps(symbol) * 2


# Regime cost multipliers — market stress widens spreads.
REGIME_COST_MULTIPLIER: Dict[str, float] = {
    'low_vol': 0.8,
    'normal': 1.0,
    'high_vol': 1.3,
    'crisis': 1.8,
}

DEFAULT_REGIME_MULTIPLIER: float = 1.0


def estimate_cost_bps(
    symbol: str,
    regime: Optional[str] = None,
) -> float:
    """
    Estimate one-way execution cost for a symbol, optionally adjusted
    for market regime.

    Args:
        symbol: ETF ticker (e.g. 'SPY', 'TLT').
        regime: Market regime for spread adjustment.

    Returns:
        Estimated cost in basis points.
    """
    base = get_cost_bps(symbol)
    mult = REGIME_COST_MULTIPLIER.get(regime or 'normal', DEFAULT_REGIME_MULTIPLIER)
    return round(base * mult, 2)
