#!/usr/bin/env python3
"""
v5.71: Cross-Asset Relative Value Scanner

Scans for extreme divergence between correlated asset pairs and generates
mean-reversion signals when z-scores exceed threshold.

Pairs:
- SPY vs QQQ (US equity vs tech)
- SPY vs EFA (US vs international)
- GLD vs BTC (gold vs digital gold)
- TLT vs IEF (long vs intermediate duration)
- SPY vs GLD (equity vs gold — traditional hedge)

Approach:
- Rolling 60-day z-score of return differential
- Entry: z-score > 2.0 (extreme divergence — bet on convergence)
- Exit: z-score < 0.5 (convergence achieved)
- Vol-scaling for position sizing

Usage:
    python -m src.signals.cross_asset_relative_value scan
    python -m src.signals.cross_asset_relative_value pairs
    python -m src.signals.cross_asset_relative_value signal --pair spy_gld
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

from src.paths import DATA_DIR, PRICES_JSON
from src.backtest.metrics import save_results_json


__all__ = ['ZSCORE_ENTRY', 'ZSCORE_EXIT', 'LOOKBACK', 'MIN_HISTORY', 'PairReading', 'CrossAssetRVSignal', 'CrossAssetRVScanner', 'print_scan']

# Cross-asset pairs with (symbol_a, symbol_b, interpretation)
# Z-score positive means A outperforming B (A overvalued vs B)
CROSS_ASSET_PAIRS: Dict[str, Tuple[str, str, str]] = {
    "spy_qqq": ("SPY", "QQQ", "US equity vs tech — tech outperforming → mean-revert"),
    "spy_efa": ("SPY", "EFA", "US vs international — US strong → international catch-up"),
    "gld_btc": ("GLD", "BTC", "Gold vs digital gold — diverging stores of value"),
    "tlt_ief": ("TLT", "IEF", "Long vs intermediate duration — steepener/flattener"),
    "spy_gld": ("SPY", "GLD", "Equity vs gold — risk-on vs risk-off divergence"),
}

# Trading parameters
ZSCORE_ENTRY = 2.0   # Enter when z-score exceeds this
ZSCORE_EXIT = 0.5    # Exit when z-score falls below this
LOOKBACK = 60         # Rolling window for z-score computation
MIN_HISTORY = 20      # Minimum data points


@dataclass
class PairReading:
    """Current state of a cross-asset pair."""
    pair_name: str
    symbol_a: str
    symbol_b: str

    # Returns
    return_a_60d: float
    return_b_60d: float
    return_differential: float  # A - B over 60d

    # Z-score
    z_score: float
    z_score_mean: float
    z_score_std: float

    # Signal
    signal_value: float  # -1 (short A/long B) to +1 (long A/short B)
    regime: str          # "diverged_bull", "diverged_bear", "converged", "neutral"
    conviction: float    # 0-1 signal strength

    # Metadata
    active: bool         # Currently in position?
    days_active: int     # Days since entry
    entry_zscore: float  # Z-score at entry

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CrossAssetRVSignal:
    """Aggregate signal from all pairs."""
    timestamp: str
    pairs: Dict[str, PairReading]

    # Composite signal
    avg_z_score: float
    max_divergence: float
    num_diverged: int
    total_pairs: int

    # Directional bias
    risk_on_score: float    # Positive = favor risk assets
    duration_score: float   # Positive = favor long duration

    # Confidence
    overall_conviction: float

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "pairs": {k: v.to_dict() for k, v in self.pairs.items()},
            "avg_z_score": self.avg_z_score,
            "max_divergence": self.max_divergence,
            "num_diverged": self.num_diverged,
            "total_pairs": self.total_pairs,
            "risk_on_score": self.risk_on_score,
            "duration_score": self.duration_score,
            "overall_conviction": self.overall_conviction,
        }


class CrossAssetRVScanner:
    """Cross-asset relative value scanner engine."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.prices: Dict[str, np.ndarray] = {}
        self.dates: List[str] = []

        # State persistence
        self.state_dir = self.data_dir / "signals"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "cross_asset_rv_state.json"

    def _load_price_data(self) -> bool:
        """Load price data from JSON file."""
        prices_path = self.data_dir.parent / "public" / "data" / "prices.json"
        if not prices_path.exists():
            prices_path = self.data_dir / "prices.json"
        if not prices_path.exists():
            # Try project root
            prices_path = PRICES_JSON

        if not prices_path.exists():
            logger.error("Price data not found: %s", prices_path)
            return False

        try:
            with open(prices_path) as f:
                raw = json.load(f)

            # Build date-indexed arrays for each symbol
            symbol_data: Dict[str, Dict[str, float]] = {}
            all_dates: set = set()

            for symbol, entries in raw.items():
                if isinstance(entries, list) and len(entries) > 0 and isinstance(entries[0], dict):
                    for entry in entries:
                        d = entry.get("d", "")
                        p = entry.get("p", None)
                        if d and p is not None:
                            if symbol not in symbol_data:
                                symbol_data[symbol] = {}
                            symbol_data[symbol][d] = float(p)
                            all_dates.add(d)

            if not all_dates:
                return False

            sorted_dates = sorted(all_dates)
            self.dates = sorted_dates

            # Build arrays for each pair symbol
            needed_symbols = set()
            for sym_a, sym_b, _ in CROSS_ASSET_PAIRS.values():
                needed_symbols.add(sym_a)
                needed_symbols.add(sym_b)

            for sym in needed_symbols:
                if sym in symbol_data:
                    self.prices[sym] = np.array([
                        symbol_data[sym].get(d, np.nan) for d in sorted_dates
                    ])
                else:
                    logger.warning("Symbol %s not found in price data", sym)
                    self.prices[sym] = np.full(len(sorted_dates), np.nan)

            return True

        except (KeyError, ValueError, TypeError, ZeroDivisionError, AttributeError, RuntimeError) as e:
            logger.error("Error loading price data: %s", e)
            return False

    def _compute_returns(self, prices: np.ndarray, period: int = 60) -> np.ndarray:
        """Compute rolling period returns."""
        if len(prices) < period + 1:
            return np.full_like(prices, np.nan)
        returns = np.full_like(prices, np.nan)
        returns[period:] = prices[period:] / prices[:-period] - 1
        return returns

    def _compute_z_score(
        self, values: np.ndarray, window: int = LOOKBACK
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute rolling z-score of a series."""
        if len(values) < window:
            return np.full_like(values, np.nan), np.full_like(values, np.nan), np.full_like(values, np.nan)

        z_scores = np.full_like(values, np.nan)
        means = np.full_like(values, np.nan)
        stds = np.full_like(values, np.nan)

        for i in range(window, len(values)):
            segment = values[i - window:i]
            seg_clean = segment[~np.isnan(segment)]
            if len(seg_clean) < MIN_HISTORY:
                continue
            m = np.mean(seg_clean)
            s = np.std(seg_clean)
            if s > 0:
                z_scores[i] = (values[i] - m) / s
                means[i] = m
                stds[i] = s

        return z_scores, means, stds

    def scan_pair(
        self, pair_name: str, current_idx: int = -1
    ) -> Optional[PairReading]:
        """Scan a single cross-asset pair for divergence."""
        if pair_name not in CROSS_ASSET_PAIRS:
            logger.warning("Unknown pair: %s", pair_name)
            return None

        sym_a, sym_b, interpretation = CROSS_ASSET_PAIRS[pair_name]

        if sym_a not in self.prices or sym_b not in self.prices:
            return None

        prices_a = self.prices[sym_a]
        prices_b = self.prices[sym_b]

        if len(prices_a) < LOOKBACK + 1 or len(prices_b) < LOOKBACK + 1:
            return None

        # Compute return differential
        ret_a = self._compute_returns(prices_a, LOOKBACK)
        ret_b = self._compute_returns(prices_b, LOOKBACK)
        diff = ret_a - ret_b

        # Compute z-score of differential
        z_scores, means, stds = self._compute_z_score(diff)

        if current_idx < 0:
            current_idx = len(diff) - 1
        if current_idx >= len(diff):
            current_idx = len(diff) - 1

        current_diff = diff[current_idx] if not np.isnan(diff[current_idx]) else 0.0
        current_z = z_scores[current_idx] if not np.isnan(z_scores[current_idx]) else 0.0
        current_mean = means[current_idx] if not np.isnan(means[current_idx]) else 0.0
        current_std = stds[current_idx] if not np.isnan(stds[current_idx]) else 0.0

        ret_a_val = ret_a[current_idx] if not np.isnan(ret_a[current_idx]) else 0.0
        ret_b_val = ret_b[current_idx] if not np.isnan(ret_b[current_idx]) else 0.0

        # Determine regime and signal
        abs_z = abs(current_z)
        if abs_z > ZSCORE_ENTRY:
            if current_z > ZSCORE_ENTRY:
                # A outperforming B — bet on mean reversion: short A, long B
                regime = "diverged_bull"  # A is overbought vs B
                signal_value = -np.clip(abs_z / 4.0, 0.5, 1.0)
            else:
                # B outperforming A — bet on mean reversion: long A, short B
                regime = "diverged_bear"  # A is oversold vs B
                signal_value = np.clip(abs_z / 4.0, 0.5, 1.0)
            conviction = min(abs_z / 3.0, 1.0)
            active = True
        elif abs_z < ZSCORE_EXIT:
            regime = "converged"
            signal_value = 0.0
            conviction = 0.0
            active = False
        else:
            regime = "neutral"
            signal_value = -current_z / ZSCORE_ENTRY  # Gradual signal
            conviction = abs_z / ZSCORE_ENTRY * 0.5
            active = False

        # Load saved state for days_active/entry_zscore
        days_active = 0
        entry_zscore = 0.0
        saved_state = self._load_state()
        if pair_name in saved_state:
            saved = saved_state[pair_name]
            if saved.get("active", False) and active:
                days_active = saved.get("days_active", 0) + 1
                entry_zscore = saved.get("entry_zscore", current_z)
            elif not active:
                days_active = 0
                entry_zscore = 0.0

        # Save updated state
        saved_state[pair_name] = {
            "active": active,
            "days_active": days_active,
            "entry_zscore": entry_zscore if active else current_z if active else 0.0,
            "last_zscore": current_z,
            "last_scan": str(datetime.now()),
        }
        self._save_state(saved_state)

        return PairReading(
            pair_name=pair_name,
            symbol_a=sym_a,
            symbol_b=sym_b,
            return_a_60d=round(float(ret_a_val) * 100, 2),
            return_b_60d=round(float(ret_b_val) * 100, 2),
            return_differential=round(float(current_diff) * 100, 2),
            z_score=round(float(current_z), 4),
            z_score_mean=round(float(current_mean), 4),
            z_score_std=round(float(current_std), 4),
            signal_value=round(float(signal_value), 4),
            regime=regime,
            conviction=round(float(conviction), 4),
            active=active,
            days_active=days_active,
            entry_zscore=round(float(entry_zscore), 4),
        )

    def scan_all(self) -> CrossAssetRVSignal:
        """Scan all cross-asset pairs."""
        if not self.prices:
            if not self._load_price_data():
                return self._empty_signal()

        readings: Dict[str, PairReading] = {}
        z_scores = []
        diverged = 0
        risk_on = 0.0
        duration = 0.0

        for pair_name in CROSS_ASSET_PAIRS:
            reading = self.scan_pair(pair_name)
            if reading is not None:
                readings[pair_name] = reading
                z_scores.append(reading.z_score)
                if abs(reading.z_score) > ZSCORE_ENTRY:
                    diverged += 1
                # Aggregate directional biases
                if pair_name == "spy_qqq":
                    risk_on -= reading.signal_value
                elif pair_name == "spy_efa":
                    risk_on -= reading.signal_value * 0.5
                elif pair_name == "spy_gld":
                    risk_on -= reading.signal_value * 0.3
                elif pair_name == "tlt_ief":
                    duration += reading.signal_value

        avg_z = float(np.mean(z_scores)) if z_scores else 0.0
        max_div = float(max(abs(z) for z in z_scores)) if z_scores else 0.0
        avg_conviction = float(np.mean([
            r.conviction for r in readings.values()
        ])) if readings else 0.0

        return CrossAssetRVSignal(
            timestamp=str(datetime.now()),
            pairs=readings,
            avg_z_score=round(avg_z, 4),
            max_divergence=round(max_div, 4),
            num_diverged=diverged,
            total_pairs=len(CROSS_ASSET_PAIRS),
            risk_on_score=round(risk_on, 4),
            duration_score=round(duration, 4),
            overall_conviction=round(avg_conviction, 4),
        )

    def get_ensemble_signal(self) -> Dict:
        """Get aggregate signal suitable for EnsembleVoter integration.

        Returns dict with signal_value, confidence, and per-asset breakdown.
        """
        signal = self.scan_all()
        if not signal.pairs:
            return {"signal_value": 0.0, "confidence": 0.0, "pairs": {}}

        # Average signal across all diverged pairs
        diverged = [r for r in signal.pairs.values() if abs(r.z_score) > ZSCORE_ENTRY]
        if diverged:
            avg_signal = float(np.mean([r.signal_value for r in diverged]))
            confidence = float(np.mean([r.conviction for r in diverged]))
        else:
            avg_signal = float(np.mean([r.signal_value for r in signal.pairs.values()]))
            confidence = signal.overall_conviction * 0.5

        # Per-asset consensus from pairs
        spy_bias = 0.0
        gld_bias = 0.0
        tlt_bias = 0.0

        if "spy_qqq" in signal.pairs:
            spy_bias += signal.pairs["spy_qqq"].signal_value * 0.4
        if "spy_efa" in signal.pairs:
            spy_bias += signal.pairs["spy_efa"].signal_value * 0.3
        if "spy_gld" in signal.pairs:
            spy_bias += signal.pairs["spy_gld"].signal_value * 0.3
            gld_bias -= signal.pairs["spy_gld"].signal_value * 0.6
        if "gld_btc" in signal.pairs:
            gld_bias -= signal.pairs["gld_btc"].signal_value * 0.4
        if "tlt_ief" in signal.pairs:
            tlt_bias += signal.pairs["tlt_ief"].signal_value * 0.5

        return {
            "signal_value": round(avg_signal, 4),
            "confidence": round(min(confidence, 1.0), 4),
            "timestamp": signal.timestamp,
            "pairs": {k: v.to_dict() for k, v in signal.pairs.items()},
            "asset_signals": {
                "SPY": round(spy_bias, 4),
                "GLD": round(gld_bias, 4),
                "TLT": round(tlt_bias, 4),
            },
            "avg_z_score": signal.avg_z_score,
            "num_diverged": signal.num_diverged,
            "total_pairs": signal.total_pairs,
        }

    def get_signal_snapshot(self):
        """Return signal as canonical SignalSnapshot for typed pipeline consumption."""
        from src.signals.signal_snapshot import SignalSnapshot
        raw = self.get_ensemble_signal()
        raw["source"] = "cross_asset_rv"
        raw["is_active"] = raw.get("signal_value", 0) != 0.0
        raw["regime_fit"] = "all"
        raw["explanation"] = (
            f"Cross-asset RV: z={raw.get('avg_z_score', 0):+.2f}, "
            f"diverged={raw.get('num_diverged', 0)}/{raw.get('total_pairs', 0)} pairs"
        )
        return SignalSnapshot.from_dict(raw)

    def _empty_signal(self) -> CrossAssetRVSignal:
        return CrossAssetRVSignal(
            timestamp=str(datetime.now()),
            pairs={},
            avg_z_score=0.0,
            max_divergence=0.0,
            num_diverged=0,
            total_pairs=len(CROSS_ASSET_PAIRS),
            risk_on_score=0.0,
            duration_score=0.0,
            overall_conviction=0.0,
        )

    def _load_state(self) -> Dict:
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logger.warning("Failed to load relative value state: %s", e)
        return {}

    def _save_state(self, state: Dict) -> None:
        try:
            save_results_json(state, output_path=str(self.state_path))
        except (OSError, KeyError, ValueError, TypeError) as e:
            logger.warning("Failed to save state: %s", e)


def print_scan(signal: CrossAssetRVSignal):
    """Pretty-print scan results."""
    logger.info("\n" + "=" * 72)
    logger.info("  CROSS-ASSET RELATIVE VALUE SCAN")
    logger.info("=" * 72)
    logger.info(f"  Timestamp: {signal.timestamp}")
    logger.info(f"  Pairs: {signal.num_diverged}/{signal.total_pairs} diverged")
    logger.info(f"  Avg Z-Score: {signal.avg_z_score:+.2f}")
    logger.info(f"  Max Divergence: {signal.max_divergence:.2f}")
    logger.info(f"  Risk-On Score: {signal.risk_on_score:+.2f}")
    logger.info(f"  Duration Score: {signal.duration_score:+.2f}")
    logger.info(f"  Conviction: {signal.overall_conviction:.1%}")
    logger.info("")

    if signal.pairs:
        logger.info(f"  {'Pair':18} {'Z-Score':>9} {'Ret A':>7} {'Ret B':>7} {'Signal':>9} {'Regime':16} {'Active':>6}")
        logger.info("  " + "-" * 72)
        for name, reading in signal.pairs.items():
            active = "✅" if reading.active else " "
            logger.info(
                f"  {name:18}"
                f" {reading.z_score:>+8.2f}"
                f" {reading.return_a_60d:>+6.1f}%"
                f" {reading.return_b_60d:>+6.1f}%"
                f" {reading.signal_value:>+8.2f}"
                f" {reading.regime:16}"
                f" {active:>6}"
            )

    logger.info("")
    logger.info("  Legend:")
    logger.info("    Z-Score > +2.0: A overvalued vs B → signal negative (short A, long B)")
    logger.info("    Z-Score < -2.0: A undervalued vs B → signal positive (long A, short B)")
    logger.info("    Z-Score < 0.5:  Converged → no signal")
    logger.info("=" * 72)
    logger.info("")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Cross-Asset Relative Value Scanner")
    subparsers = parser.add_subparsers(dest="command")

    scan_parser = subparsers.add_parser("scan", help="Scan all pairs")
    scan_parser.add_argument("--pair", help="Scan specific pair (default: all)")
    scan_parser.add_argument("--json", action="store_true", help="Output as JSON")

    pairs_parser = subparsers.add_parser("pairs", help="List available pairs")

    signal_parser = subparsers.add_parser("signal", help="Get ensemble signal")
    signal_parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()
    scanner = CrossAssetRVScanner()

    if args.command == "scan":
        signal = scanner.scan_all()
        if args.json:
            print(json.dumps(signal.to_dict(), indent=2, default=str))
        else:
            print_scan(signal)

    elif args.command == "pairs":
        print("\nAvailable Cross-Asset Pairs:")
        print("-" * 60)
        for name, (a, b, desc) in CROSS_ASSET_PAIRS.items():
            print(f"  {name:15} {a:5} vs {b:5}  — {desc}")
        print(f"\n  Entry threshold: |z-score| > {ZSCORE_ENTRY}")
        print(f"  Exit threshold:  |z-score| < {ZSCORE_EXIT}")
        print(f"  Lookback:        {LOOKBACK} days")
        print()

    elif args.command == "signal":
        ensemble_sig = scanner.get_ensemble_signal()
        if args.json:
            print(json.dumps(ensemble_sig, indent=2, default=str))
        else:
            print_scan(scanner.scan_all())
            print(f"  Ensemble signal: {ensemble_sig['signal_value']:+.4f}")
            print(f"  Confidence:       {ensemble_sig['confidence']:.1%}")
            print(f"  Asset biases:")
            for asset, bias in ensemble_sig.get("asset_signals", {}).items():
                print(f"    {asset}: {bias:+.4f}")
            print()

    else:
        parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
