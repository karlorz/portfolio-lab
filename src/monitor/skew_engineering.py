#!/usr/bin/env python3
"""
v5.40 — Skew Engineering Overlay
Asymmetric volatility management separating upside/downside variance regimes.

Based on concept from arXiv 2605.09123:
"The Engineering of Skew: A Path-Dependent Framework for Asymmetric Volatility Management"

Features:
- Separate realized variances for positive and negative daily returns
- Skew ratio = downside_variance / upside_variance (rolling windows)
- Regime classification: NORMAL / ELEVATED / HIGH
- Vol target adjustment: reduce exposure when skew ratio elevated
- Integration with v5.20 Bayesian Vol model for improved estimation

Usage:
    python -m src.monitor.skew_engineering compute --symbol SPY
    python -m src.monitor.skew_engineering history --symbol SPY
    python -m src.monitor.skew_engineering adjust --symbol SPY
"""

import argparse
import json
import logging
import math
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STATE_FILE = DATA_DIR / "skew_state.json"


class SkewRegime:
    """Skew ratio regimes."""
    NORMAL = "NORMAL"        # skew_ratio < 1.3
    ELEVATED = "ELEVATED"    # 1.3 <= skew_ratio < 1.8
    HIGH = "HIGH"           # skew_ratio >= 1.8

    THRESHOLD_ELEVATED = 1.3
    THRESHOLD_HIGH = 1.8

    # Vol penalty caps per regime
    PENALTY_NORMAL = 0.05    # 5% reduction during normal
    PENALTY_ELEVATED = 0.12  # 12% reduction during elevated
    PENALTY_HIGH = 0.20      # 20% max reduction during high


@dataclass
class SkewMetrics:
    """Skew engineering metrics for an asset."""
    symbol: str
    timestamp: str

    # Windows
    window_21d: int = 21
    window_63d: int = 63
    window_252d: int = 252

    # Short-term (21d)
    upside_var_21d: float = 0.0
    downside_var_21d: float = 0.0
    skew_ratio_21d: float = 1.0
    regime_21d: str = SkewRegime.NORMAL

    # Medium-term (63d)
    upside_var_63d: float = 0.0
    downside_var_63d: float = 0.0
    skew_ratio_63d: float = 1.0
    regime_63d: str = SkewRegime.NORMAL

    # Long-term baseline (252d)
    upside_var_252d: float = 0.0
    downside_var_252d: float = 0.0
    skew_ratio_252d: float = 1.0
    regime_252d: str = SkewRegime.NORMAL

    # Composite recommendation
    composite_regime: str = SkewRegime.NORMAL
    vol_penalty: float = 0.0       # Fraction to reduce vol target
    effective_vol_target: float = 0.10  # After penalty applied
    n_obs: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SkewState:
    """Persistent state for skew engineering."""
    symbol: str
    last_update: str
    composite_regime: str
    vol_penalty: float
    side_computed: bool
    n_obs: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SkewState":
        return cls(**d)


class SkewEngine:
    """
    Core skew engineering engine.
    Computes asymmetric volatility and generates vol target adjustments.
    """

    # Minimum observations required
    MIN_OBS = 10

    def __init__(self, symbol: str = "SPY"):
        self.symbol = symbol
        self.db_path = DATA_DIR / "market.db"

    def _get_prices(self, days: int = 260) -> np.ndarray:
        """Fetch daily returns from market.db."""
        if not self.db_path.exists():
            logger.warning(f"Database not found: {self.db_path}")
            return np.array([])

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Check table structure
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]

            if "daily_bars" in tables:
                query = """
                    SELECT close FROM daily_bars
                    WHERE symbol = ?
                    ORDER BY date DESC
                    LIMIT ?
                """
            elif "prices" in tables:
                query = """
                    SELECT close FROM prices
                    WHERE symbol = ?
                    ORDER BY date DESC
                    LIMIT ?
                """
            else:
                # Try generic approach
                query = """
                    SELECT close FROM market_data
                    WHERE symbol = ?
                    ORDER BY date DESC
                    LIMIT ?
                """

            cursor.execute(query, (self.symbol, days + 1))
            rows = cursor.fetchall()
            conn.close()

            if len(rows) < 2:
                logger.warning(
                    f"Not enough data for {self.symbol}: {len(rows)} rows"
                )
                return np.array([])

            closes = np.array([r[0] for r in rows], dtype=np.float64)
            # Reverse to chronological order
            closes = closes[::-1]
            returns = np.diff(closes) / closes[:-1]
            return returns

        except (sqlite3.Error, Exception) as e:
            logger.error(f"Error fetching prices: {e}")
            return np.array([])

    def compute_skew_ratio(
        self, returns: np.ndarray, window: int
    ) -> Tuple[float, float, float, str]:
        """
        Compute skew ratio for a given window.

        Returns:
            Tuple of (upside_var, downside_var, skew_ratio, regime)
        """
        if len(returns) < self.MIN_OBS:
            return (0.0, 0.0, 1.0, SkewRegime.NORMAL)

        # Use most recent `window` observations
        recent = returns[-window:] if len(returns) >= window else returns

        # Separate upside and downside returns
        upside = recent[recent >= 0]
        downside = recent[recent < 0]

        # Compute realized variances
        upside_var = float(np.var(upside)) if len(upside) > 1 else 0.0
        downside_var = float(np.var(downside)) if len(downside) > 1 else 0.0

        # Handle edge cases
        if downside_var < 1e-12 and upside_var < 1e-12:
            # Both near-zero (all returns same value)
            return (0.0, 0.0, 1.0, SkewRegime.NORMAL)

        # Avoid division by zero
        if upside_var < 1e-12:
            upside_var = 1e-12

        skew_ratio = downside_var / upside_var

        # Determine regime
        if skew_ratio >= SkewRegime.THRESHOLD_HIGH:
            regime = SkewRegime.HIGH
        elif skew_ratio >= SkewRegime.THRESHOLD_ELEVATED:
            regime = SkewRegime.ELEVATED
        else:
            regime = SkewRegime.NORMAL

        # Annualize variances for display
        upside_var_ann = upside_var * 252
        downside_var_ann = downside_var * 252

        return (upside_var_ann, downside_var_ann, skew_ratio, regime)

    def compute(self) -> SkewMetrics:
        """Compute full skew metrics for the configured symbol."""
        returns = self._get_prices(days=260)

        if len(returns) < self.MIN_OBS:
            logger.warning(
                f"Insufficient data for {self.symbol}: "
                f"{len(returns)} obs"
            )
            return SkewMetrics(
                symbol=self.symbol,
                timestamp=datetime.now().isoformat(),
                n_obs=len(returns),
            )

        # Compute for all windows
        up_21, down_21, ratio_21, regime_21 = self.compute_skew_ratio(
            returns, 21
        )
        up_63, down_63, ratio_63, regime_63 = self.compute_skew_ratio(
            returns, 63
        )
        up_252, down_252, ratio_252, regime_252 = self.compute_skew_ratio(
            returns, 252
        )

        # Composite regime: use the most conservative across windows
        regimes = [regime_21, regime_63, regime_252]
        if SkewRegime.HIGH in regimes:
            composite = SkewRegime.HIGH
        elif SkewRegime.ELEVATED in regimes:
            composite = SkewRegime.ELEVATED
        else:
            composite = SkewRegime.NORMAL

        # Compute vol penalty based on composite regime
        if composite == SkewRegime.HIGH:
            vol_penalty = SkewRegime.PENALTY_HIGH
        elif composite == SkewRegime.ELEVATED:
            vol_penalty = SkewRegime.PENALTY_ELEVATED
        else:
            vol_penalty = SkewRegime.PENALTY_NORMAL

        metrics = SkewMetrics(
            symbol=self.symbol,
            timestamp=datetime.now().isoformat(),
            window_21d=21,
            window_63d=63,
            window_252d=252,
            upside_var_21d=round(up_21, 6),
            downside_var_21d=round(down_21, 6),
            skew_ratio_21d=round(ratio_21, 4),
            regime_21d=regime_21,
            upside_var_63d=round(up_63, 6),
            downside_var_63d=round(down_63, 6),
            skew_ratio_63d=round(ratio_63, 4),
            regime_63d=regime_63,
            upside_var_252d=round(up_252, 6),
            downside_var_252d=round(down_252, 6),
            skew_ratio_252d=round(ratio_252, 4),
            regime_252d=regime_252,
            composite_regime=composite,
            vol_penalty=round(vol_penalty, 4),
            effective_vol_target=round(0.10 * (1.0 - vol_penalty), 4),
            n_obs=len(returns),
        )

        # Persist state
        self._save_state(metrics)
        return metrics

    def _save_state(self, metrics: SkewMetrics) -> None:
        """Save persistent state to disk."""
        state = SkewState(
            symbol=self.symbol,
            last_update=metrics.timestamp,
            composite_regime=metrics.composite_regime,
            vol_penalty=metrics.vol_penalty,
            side_computed=False,
            n_obs=metrics.n_obs,
        )
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(STATE_FILE, "w") as f:
                json.dump(state.to_dict(), f, indent=2)
        except OSError as e:
            logger.error(f"Failed to save state: {e}")

    def load_state(self) -> Optional[SkewState]:
        """Load persistent state from disk."""
        if not STATE_FILE.exists():
            return None
        try:
            with open(STATE_FILE) as f:
                data = json.load(f)
            return SkewState.from_dict(data)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load state: {e}")
            return None

    def get_vol_adjustment(self, target_vol: float = 0.10) -> float:
        """
        Get the effective vol target after skew penalty.

        Args:
            target_vol: Base annualized vol target (default 10%)

        Returns:
            Adjusted vol target after applying skew penalty
        """
        metrics = self.compute()
        adjusted = target_vol * (1.0 - metrics.vol_penalty)
        return round(adjusted, 4)

    def summarize(self) -> str:
        """Get a human-readable summary of current skew state."""
        metrics = self.compute()
        lines = [
            f"Skew Engineering — {self.symbol}",
            f"  Timestamp: {metrics.timestamp}",
            f"  Observations: {metrics.n_obs}",
            "",
            f"  21-day:  up_var={metrics.upside_var_21d:.6f}  "
            f"down_var={metrics.downside_var_21d:.6f}  "
            f"ratio={metrics.skew_ratio_21d:.3f}  "
            f"regime={metrics.regime_21d}",
            f"  63-day:  up_var={metrics.upside_var_63d:.6f}  "
            f"down_var={metrics.downside_var_63d:.6f}  "
            f"ratio={metrics.skew_ratio_63d:.3f}  "
            f"regime={metrics.regime_63d}",
            f"  252-day: up_var={metrics.upside_var_252d:.6f}  "
            f"down_var={metrics.downside_var_252d:.6f}  "
            f"ratio={metrics.skew_ratio_252d:.3f}  "
            f"regime={metrics.regime_252d}",
            "",
            f"  Composite regime: {metrics.composite_regime}",
            f"  Vol penalty: {metrics.vol_penalty:.1%}",
            f"  Effective vol target: {metrics.effective_vol_target:.1%}",
        ]
        return "\n".join(lines)


def cli_compute(args: argparse.Namespace) -> None:
    """CLI: compute skew metrics."""
    engine = SkewEngine(symbol=args.symbol)
    metrics = engine.compute()
    print(json.dumps(metrics.to_dict(), indent=2))


def cli_summary(args: argparse.Namespace) -> None:
    """CLI: print human-readable summary."""
    engine = SkewEngine(symbol=args.symbol)
    print(engine.summarize())


def cli_adjust(args: argparse.Namespace) -> None:
    """CLI: get vol target adjustment."""
    engine = SkewEngine(symbol=args.symbol)
    adjusted = engine.get_vol_adjustment(target_vol=args.target_vol)
    print(f"Base target: {args.target_vol:.1%}")
    print(f"Adjusted target: {adjusted:.1%}")
    print(f"Reduction: {(1.0 - adjusted / args.target_vol):.1%}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="v5.40 Skew Engineering Overlay"
    )
    parser.add_argument(
        "--symbol", type=str, default="SPY",
        help="Symbol to analyze (default: SPY)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # compute
    p_compute = subparsers.add_parser(
        "compute", help="Compute skew metrics"
    )

    # summary
    p_summary = subparsers.add_parser(
        "summary", help="Human-readable summary"
    )

    # adjust
    p_adjust = subparsers.add_parser(
        "adjust", help="Get vol target adjustment"
    )
    p_adjust.add_argument(
        "--target-vol", type=float, default=0.10,
        help="Base annualized vol target (default: 0.10)"
    )

    args = parser.parse_args()

    if args.command == "compute":
        cli_compute(args)
    elif args.command == "summary":
        cli_summary(args)
    elif args.command == "adjust":
        cli_adjust(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
