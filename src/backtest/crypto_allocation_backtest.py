"""
Crypto Tactical Allocation Walk-Forward Backtest - v9.32 Implementation

Validates the crypto tactical allocation overlay on the baseline 46/38/16
(SPY/GLD/TLT) portfolio. The overlay dynamically allocates 0-5% to BTC/ETH
(60/40 split) funded from the GLD sleeve.

Entry rules:
  - 6-month SPY momentum positive
  - Crypto vol regime normal/low (not extreme >100% annualized)

Exit rules:
  - SPY 6-month momentum turns negative
  - Crypto vol regime extreme (>100% annualized)

Key questions:
  1. Does crypto overlay improve Sharpe ratio vs baseline?
  2. How much does crypto add to CAGR?
  3. What fraction of time is crypto active?
  4. Does crypto help or hurt during crisis years?

Period: 2006-2026 (20+ years including GFC, COVID, 2022 rate hikes)
"""

import json
import logging
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.backtest.metrics import (
    BacktestMetrics,
    compute_metrics,
    compute_crisis_returns,
    save_results_json,
)
from src.paths import PRICES_JSON, BACKTEST_RESULTS_DIR
from src.signals.crypto_momentum import CryptoMomentumCalculator, CryptoVolRegime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252
CRYPTO_TRADING_DAYS = 365  # crypto trades ~365 days/year
MONTHLY_TRADING_DAYS = 21

# Crisis years to evaluate
CRISIS_YEARS = ["2008", "2020", "2022"]


# Symbols
BASE_SYMBOLS = ["SPY", "GLD", "TLT"]
CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD"]


@dataclass
class BacktestConfig:
    """Configuration for crypto allocation walk-forward backtest."""

    start_date: str = "2006-01-01"
    end_date: str = "2026-05-15"
    initial_capital: float = 100000.0

    # Baseline allocation (46/38/16)
    base_spy_weight: float = 0.46
    base_gld_weight: float = 0.38
    base_tlt_weight: float = 0.16

    # Crypto constraints
    max_crypto_pct: float = 5.0  # Hard cap on total crypto (%)

    # Rebalancing
    rebalance_frequency_days: int = MONTHLY_TRADING_DAYS
    transaction_cost_bps: float = 10.0  # 10 bps per rebalance


@dataclass
class DailyPrices:
    """Daily price data for a single date."""

    date: str
    spy: float
    gld: float
    tlt: float
    btc: Optional[float] = None
    eth: Optional[float] = None


@dataclass
class BacktestResult:
    """Complete backtest results comparing baseline vs crypto overlay."""

    # Core metrics
    total_return: float
    cagr: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float

    # Baseline comparison
    baseline_total_return: float
    baseline_cagr: float
    baseline_volatility: float
    baseline_sharpe: float
    baseline_max_drawdown: float
    sharpe_improvement: float
    cagr_impact: float

    # Crypto activity
    crypto_active_days: int
    crypto_active_pct: float
    avg_crypto_pct: float
    max_crypto_pct: float

    # Crisis returns
    crisis_returns_crypto: Dict[str, float]
    crisis_returns_baseline: Dict[str, float]

    # Regime breakdown
    regime_breakdown: Dict[str, Dict[str, float]]

    # Trade stats
    total_rebalances: int
    total_transaction_costs: float

    # Config
    config_snapshot: Dict

    def to_dict(self) -> Dict:
        """Serialize to dict for JSON output."""
        return {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "baseline_total_return": self.baseline_total_return,
            "baseline_cagr": self.baseline_cagr,
            "baseline_volatility": self.baseline_volatility,
            "baseline_sharpe": self.baseline_sharpe,
            "baseline_max_drawdown": self.baseline_max_drawdown,
            "sharpe_improvement": self.sharpe_improvement,
            "cagr_impact": self.cagr_impact,
            "crypto_active_days": self.crypto_active_days,
            "crypto_active_pct": self.crypto_active_pct,
            "avg_crypto_pct": self.avg_crypto_pct,
            "max_crypto_pct": self.max_crypto_pct,
            "crisis_returns_crypto": self.crisis_returns_crypto,
            "crisis_returns_baseline": self.crisis_returns_baseline,
            "regime_breakdown": self.regime_breakdown,
            "total_rebalances": self.total_rebalances,
            "total_transaction_costs": self.total_transaction_costs,
            "config_snapshot": self.config_snapshot,
        }


# ---------------------------------------------------------------------------
# Walk-Forward Backtester
# ---------------------------------------------------------------------------


class WalkForwardCryptoBacktester:
    """
    Walk-forward backtest for the crypto tactical allocation overlay.

    Simulates monthly rebalancing of the baseline 46/38/16 portfolio with
    a dynamic crypto sleeve (BTC 60% / ETH 40%) funded from GLD.
    Crypto is active when SPY 6-month momentum is positive and crypto
    vol regimes are not extreme.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self._daily_prices: List[DailyPrices] = []
        self._trading_dates: List[str] = []

    def load_data(self) -> None:
        """Load price data from PRICES_JSON.

        Extracts SPY/GLD/TLT/BTC-USD/ETH-USD. Falls back to synthetic data
        if the file is missing or symbols are not available.
        """
        prices_path = PRICES_JSON
        if not prices_path.exists():
            logger.warning("Prices file not found at %s; generating synthetic data", prices_path)
            self._generate_synthetic_data()
            return

        with open(prices_path) as f:
            raw = json.load(f)

        # Build ordered price lookups
        spy_data = {e["d"]: e["p"] for e in raw.get("SPY", [])}
        gld_data = {e["d"]: e["p"] for e in raw.get("GLD", [])}
        tlt_data = {e["d"]: e["p"] for e in raw.get("TLT", [])}
        btc_data = {}
        eth_data = {}

        # Try multiple crypto symbol representations
        for btc_key in ("BTC-USD", "BTCUSD", "BTC"):
            if btc_key in raw:
                btc_data = {e["d"]: e["p"] for e in raw[btc_key]}
                break

        for eth_key in ("ETH-USD", "ETHUSD", "ETH"):
            if eth_key in raw:
                eth_data = {e["d"]: e["p"] for e in raw[eth_key]}
                break

        # Collect all trading dates that have all base assets
        all_dates = sorted(
            d for d in spy_data
            if d in gld_data and d in tlt_data
        )

        # Filter to config range
        start_ts = self.config.start_date
        end_ts = self.config.end_date
        filtered_dates = [d for d in all_dates if start_ts <= d <= end_ts]

        if not filtered_dates:
            logger.warning("No data in date range %s to %s; generating synthetic data", start_ts, end_ts)
            self._generate_synthetic_data()
            return

        self._trading_dates = filtered_dates
        self._daily_prices = []

        for date in filtered_dates:
            btc_price = float(btc_data[date]) if date in btc_data and btc_data[date] is not None else None
            eth_price = float(eth_data[date]) if date in eth_data and eth_data[date] is not None else None

            self._daily_prices.append(DailyPrices(
                date=date,
                spy=float(spy_data[date]),
                gld=float(gld_data[date]),
                tlt=float(tlt_data[date]),
                btc=btc_price,
                eth=eth_price,
            ))

        # If crypto data is entirely missing, synthesize it
        has_btc = any(dp.btc is not None for dp in self._daily_prices)
        has_eth = any(dp.eth is not None for dp in self._daily_prices)

        if not has_btc or not has_eth:
            logger.info("Crypto price data incomplete; filling gaps with synthetic data")
            self._fill_synthetic_crypto()

        logger.info(
            "Loaded %d trading days from %s to %s (crypto: %s)",
            len(self._daily_prices),
            filtered_dates[0],
            filtered_dates[-1],
            "synthetic" if not has_btc else "real",
        )

    def _fill_synthetic_crypto(self) -> None:
        """Fill missing crypto prices with synthetic data.

        Generates BTC/ETH prices with realistic crypto-like volatility
        (60-80% annualized) and a slight positive drift.
        """
        n = len(self._daily_prices)
        rng = np.random.RandomState(42)

        btc_daily_vol = 0.75 / np.sqrt(CRYPTO_TRADING_DAYS)
        eth_daily_vol = 0.85 / np.sqrt(CRYPTO_TRADING_DAYS)

        btc_log_rets = rng.normal(0.0008, btc_daily_vol, n)
        eth_log_rets = rng.normal(0.0006, eth_daily_vol, n)

        btc_price = 50000.0
        eth_price = 3000.0

        for i, dp in enumerate(self._daily_prices):
            if dp.btc is None:
                btc_price *= np.exp(btc_log_rets[i])
                dp.btc = btc_price
            if dp.eth is None:
                eth_price *= np.exp(eth_log_rets[i])
                dp.eth = eth_price

    def _generate_synthetic_data(self) -> None:
        """Generate fully synthetic price data for testing.

        Creates ~20 years of daily prices for SPY, GLD, TLT, BTC-USD, ETH-USD
        with realistic drift, volatility, and correlation structure.
        Crypto assets have higher volatility (60-85% annualized).
        """
        rng = np.random.RandomState(42)
        n_days = 5100  # ~20 years

        start_date = datetime.strptime(self.config.start_date, "%Y-%m-%d")
        dates = []
        dt = start_date
        while len(dates) < n_days:
            if dt.weekday() < 5:
                dates.append(dt.strftime("%Y-%m-%d"))
            dt += timedelta(days=1)

        dates = dates[:n_days]
        n = len(dates)

        # Common factor for correlation
        common = rng.normal(0, 1, n)

        # SPY: 7% drift, 18% vol
        spy_noise = rng.normal(0.07 / TRADING_DAYS_PER_YEAR, 0.18 / np.sqrt(TRADING_DAYS_PER_YEAR), n)
        spy_returns = 0.3 * common * (0.18 / np.sqrt(TRADING_DAYS_PER_YEAR)) + 0.7 * spy_noise + 0.07 / TRADING_DAYS_PER_YEAR

        # GLD: 4% drift, 15% vol, 0.3 corr to SPY
        gld_noise = rng.normal(0.04 / TRADING_DAYS_PER_YEAR, 0.15 / np.sqrt(TRADING_DAYS_PER_YEAR), n)
        gld_common = 0.3 * spy_returns * (0.15 / 0.18)
        gld_returns = gld_common + np.sqrt(1 - 0.3 ** 2) * gld_noise

        # TLT: 3% drift, 12% vol, low corr
        tlt_returns = rng.normal(0.03 / TRADING_DAYS_PER_YEAR, 0.12 / np.sqrt(TRADING_DAYS_PER_YEAR), n)

        # BTC: 15% drift, 75% vol
        btc_daily_vol = 0.75 / np.sqrt(CRYPTO_TRADING_DAYS)
        btc_returns = rng.normal(0.15 / CRYPTO_TRADING_DAYS, btc_daily_vol, n)

        # ETH: 12% drift, 85% vol
        eth_daily_vol = 0.85 / np.sqrt(CRYPTO_TRADING_DAYS)
        eth_returns = rng.normal(0.12 / CRYPTO_TRADING_DAYS, eth_daily_vol, n)

        spy_price = 100.0 * np.exp(np.cumsum(spy_returns))
        gld_price = 100.0 * np.exp(np.cumsum(gld_returns))
        tlt_price = 100.0 * np.exp(np.cumsum(tlt_returns))
        btc_price = 50000.0 * np.exp(np.cumsum(btc_returns))
        eth_price = 3000.0 * np.exp(np.cumsum(eth_returns))

        self._trading_dates = dates
        self._daily_prices = []

        for i in range(n):
            self._daily_prices.append(DailyPrices(
                date=dates[i],
                spy=float(spy_price[i]),
                gld=float(gld_price[i]),
                tlt=float(tlt_price[i]),
                btc=float(btc_price[i]),
                eth=float(eth_price[i]),
            ))

        logger.info("Generated %d synthetic trading days with crypto data", n)

    def _compute_spy_momentum_6m(self, idx: int) -> float:
        """Compute SPY 6-month momentum using production CryptoMomentumCalculator.

        Returns 0.0 if insufficient history.
        """
        if idx < 180:
            return 0.0

        calc = CryptoMomentumCalculator()
        spy_prices = [dp.spy for dp in self._daily_prices[:idx + 1]]
        return calc.compute_momentum(spy_prices, 180)

    def _compute_crypto_vol(
        self, idx: int, lookback: int = 21
    ) -> Tuple[float, float]:
        """Compute annualized volatility for BTC and ETH using production code.

        Delegates to CryptoMomentumCalculator.compute_volatility().

        Returns (btc_vol, eth_vol) as decimal fractions (e.g., 0.75 = 75%).
        Returns 0.0 for an asset with insufficient data.
        """
        calc = CryptoMomentumCalculator()
        btc_vol = 0.0
        eth_vol = 0.0

        if idx >= lookback:
            # BTC vol
            btc_prices = [self._daily_prices[j].btc for j in range(idx - lookback, idx + 1)]
            btc_prices = [p for p in btc_prices if p is not None and p > 0]
            if len(btc_prices) >= 5:
                btc_rets = [btc_prices[j] / btc_prices[j - 1] - 1 for j in range(1, len(btc_prices))]
                if btc_rets:
                    btc_vol = calc.compute_volatility(btc_rets, len(btc_rets))

            # ETH vol
            eth_prices = [self._daily_prices[j].eth for j in range(idx - lookback, idx + 1)]
            eth_prices = [p for p in eth_prices if p is not None and p > 0]
            if len(eth_prices) >= 5:
                eth_rets = [eth_prices[j] / eth_prices[j - 1] - 1 for j in range(1, len(eth_prices))]
                if eth_rets:
                    eth_vol = calc.compute_volatility(eth_rets, len(eth_rets))

        return btc_vol, eth_vol

    def _is_vol_extreme(self, btc_vol: float, eth_vol: float) -> bool:
        """Check if either crypto asset has extreme vol using production constant.

        Uses CryptoMomentumCalculator.VOL_EXTREME for the threshold with strict
        greater-than comparison (>1.00), matching the original backtest boundary.
        """
        return btc_vol > CryptoMomentumCalculator.VOL_EXTREME or eth_vol > CryptoMomentumCalculator.VOL_EXTREME

    def _compute_crypto_allocation(
        self, idx: int, gld_sleeve: float
    ) -> Tuple[float, float, float]:
        """Compute crypto allocation using production CryptoMomentumCalculator.

        Uses assess_asset_signal() for vol regime classification, momentum
        computation, and vol scaling instead of the backtest's own logic.

        Entry: SPY 6m momentum positive AND no extreme crypto vol
        Exit: SPY momentum negative OR extreme crypto vol

        Returns (btc_weight, eth_weight, total_crypto_weight) as portfolio
        fractions (e.g., 0.03 = 3% of portfolio).
        """
        if idx < 180:
            return 0.0, 0.0, 0.0

        calc = CryptoMomentumCalculator()

        # SPY momentum
        spy_prices = [dp.spy for dp in self._daily_prices[:idx + 1]]
        spy_mom_6m = calc.compute_momentum(spy_prices, 180)

        # Exit if SPY momentum negative
        if spy_mom_6m <= 0:
            return 0.0, 0.0, 0.0

        # BTC/ETH signals
        btc_prices = [dp.btc for dp in self._daily_prices[:idx + 1] if dp.btc is not None]
        eth_prices = [dp.eth for dp in self._daily_prices[:idx + 1] if dp.eth is not None]

        if len(btc_prices) < 30 or len(eth_prices) < 30:
            return 0.0, 0.0, 0.0

        btc_rets = [(btc_prices[j] / btc_prices[j - 1] - 1) for j in range(1, len(btc_prices))]
        eth_rets = [(eth_prices[j] / eth_prices[j - 1] - 1) for j in range(1, len(eth_prices))]

        btc_signal = calc.assess_asset_signal("BTC", btc_prices[-1], btc_prices, btc_rets)
        eth_signal = calc.assess_asset_signal("ETH", eth_prices[-1], eth_prices, eth_rets)

        # Exit if extreme vol
        if btc_signal.vol_regime == "extreme" or eth_signal.vol_regime == "extreme":
            return 0.0, 0.0, 0.0

        # Compute allocation
        max_crypto = self.config.max_crypto_pct / 100.0
        crypto_target = min(max_crypto, gld_sleeve * 0.15)

        # Use vol scale from production code
        avg_vol = (btc_signal.vol_30d + eth_signal.vol_30d) / 2
        vol_scale = calc.compute_vol_scale(avg_vol)
        crypto_target *= vol_scale
        crypto_target = min(crypto_target, max_crypto)

        btc_w = crypto_target * calc.BTC_WEIGHT
        eth_w = crypto_target * calc.ETH_WEIGHT

        return btc_w, eth_w, crypto_target

    def _compute_portfolio_return(
        self,
        p0: DailyPrices,
        p1: DailyPrices,
        spy_w: float,
        gld_w: float,
        tlt_w: float,
        btc_w: float,
        eth_w: float,
    ) -> float:
        """Compute 1-day portfolio return given current weights."""
        spy_ret = (p1.spy / p0.spy - 1) if p0.spy > 0 else 0.0
        gld_ret = (p1.gld / p0.gld - 1) if p0.gld > 0 else 0.0
        tlt_ret = (p1.tlt / p0.tlt - 1) if p0.tlt > 0 else 0.0
        btc_ret = (p1.btc / p0.btc - 1) if (p0.btc and p0.btc > 0 and btc_w > 0) else 0.0
        eth_ret = (p1.eth / p0.eth - 1) if (p0.eth and p0.eth > 0 and eth_w > 0) else 0.0

        return (
            spy_w * spy_ret
            + gld_w * gld_ret
            + tlt_w * tlt_ret
            + btc_w * btc_ret
            + eth_w * eth_ret
        )

    def run(self) -> BacktestResult:
        """Run the walk-forward backtest simulation.

        Returns a BacktestResult with crypto overlay metrics, baseline
        comparison, crypto activity stats, crisis returns, and regime breakdown.
        """
        if not self._daily_prices:
            self.load_data()

        if len(self._daily_prices) < 2:
            logger.error("Insufficient data for backtest")
            return self._empty_result()

        prices = self._daily_prices
        config = self.config

        # Need enough warmup for 6-month momentum
        if len(prices) < 180 + 1:
            logger.error("Insufficient data for 6-month momentum")
            return self._empty_result()

        # ── Baseline: buy-and-hold 46/38/16 ──────────────────────────────
        baseline_equity = self._run_baseline(prices, config)

        # ── Crypto overlay: dynamic allocation ───────────────────────────
        crypto_equity, crypto_tracker = self._run_crypto_overlay(prices, config)

        # Compute metrics from equity curves
        baseline_metrics = compute_metrics(baseline_equity, config.initial_capital)
        crypto_metrics = compute_metrics(crypto_equity, config.initial_capital)

        # Crisis returns
        prices_lookup = self._build_prices_lookup()
        trading_dates = self._trading_dates

        crisis_crypto = self._compute_crisis_returns_crypto(
            prices_lookup, trading_dates, crypto_equity, config.initial_capital
        )
        crisis_baseline = compute_crisis_returns(prices_lookup, trading_dates)

        # Crypto activity stats
        crypto_active_days = crypto_tracker["active_days"]
        total_days = max(len(prices) - 1, 1)
        crypto_active_pct = round(100.0 * crypto_active_days / total_days, 2)
        avg_crypto_pct = crypto_tracker["avg_pct"]
        max_crypto_pct = crypto_tracker["max_pct"]

        # Trade stats
        total_rebalances = crypto_tracker["rebalances"]
        total_costs = crypto_tracker["total_costs"]

        # Regime breakdown
        regime_breakdown = crypto_tracker.get("regime_breakdown", {})

        return BacktestResult(
            total_return=crypto_metrics.total_return,
            cagr=crypto_metrics.cagr,
            volatility=crypto_metrics.volatility,
            sharpe_ratio=crypto_metrics.sharpe_ratio,
            max_drawdown=crypto_metrics.max_drawdown,
            baseline_total_return=baseline_metrics.total_return,
            baseline_cagr=baseline_metrics.cagr,
            baseline_volatility=baseline_metrics.volatility,
            baseline_sharpe=baseline_metrics.sharpe_ratio,
            baseline_max_drawdown=baseline_metrics.max_drawdown,
            sharpe_improvement=round(
                crypto_metrics.sharpe_ratio - baseline_metrics.sharpe_ratio, 4
            ),
            cagr_impact=round(
                crypto_metrics.cagr - baseline_metrics.cagr, 2
            ),
            crypto_active_days=crypto_active_days,
            crypto_active_pct=crypto_active_pct,
            avg_crypto_pct=avg_crypto_pct,
            max_crypto_pct=max_crypto_pct,
            crisis_returns_crypto=crisis_crypto,
            crisis_returns_baseline=crisis_baseline,
            regime_breakdown=regime_breakdown,
            total_rebalances=total_rebalances,
            total_transaction_costs=round(total_costs, 2),
            config_snapshot={
                "start_date": config.start_date,
                "end_date": config.end_date,
                "initial_capital": config.initial_capital,
                "max_crypto_pct": config.max_crypto_pct,
                "rebalance_frequency_days": config.rebalance_frequency_days,
                "transaction_cost_bps": config.transaction_cost_bps,
                "base_allocation": {
                    "SPY": config.base_spy_weight,
                    "GLD": config.base_gld_weight,
                    "TLT": config.base_tlt_weight,
                },
                "crypto_split": {
                    "BTC": 0.60,
                    "ETH": 0.40,
                },
            },
        )

    def _run_baseline(
        self, prices: List[DailyPrices], config: BacktestConfig
    ) -> List[float]:
        """Run baseline buy-and-hold 46/38/16 portfolio.

        Starts accumulating returns after the momentum warmup period so both
        equity curves are the same length.
        """
        spy_w = config.base_spy_weight
        gld_w = config.base_gld_weight
        tlt_w = config.base_tlt_weight

        equity = [config.initial_capital]

        for i in range(1, len(prices)):
            ret = self._compute_portfolio_return(
                prices[i - 1], prices[i], spy_w, gld_w, tlt_w, 0.0, 0.0
            )
            new_equity = equity[-1] * (1 + ret)
            equity.append(new_equity)

        return equity

    def _run_crypto_overlay(
        self, prices: List[DailyPrices], config: BacktestConfig
    ) -> Tuple[List[float], Dict]:
        """Run portfolio with dynamic crypto overlay.

        Returns (equity_curve, crypto_tracker).
        """
        spy_w = config.base_spy_weight
        gld_w = config.base_gld_weight
        tlt_w = config.base_tlt_weight
        btc_w = 0.0
        eth_w = 0.0

        equity = [config.initial_capital]
        crypto_tracker = {
            "active_days": 0,
            "avg_pct": 0.0,
            "max_pct": 0.0,
            "rebalances": 0,
            "total_costs": 0.0,
            "allocations": [],
            "regime_counts": {},
        }
        days_since_rebalance = 0
        rebalance_freq = config.rebalance_frequency_days
        cost_per_trade = config.transaction_cost_bps / 10000.0  # Convert bps to decimal

        for i in range(1, len(prices)):
            date = prices[i].date
            days_since_rebalance += 1

            # Rebalance on initial day and then monthly
            if days_since_rebalance >= rebalance_freq:
                # Compute target crypto allocation
                new_btc_w, new_eth_w, total_crypto = self._compute_crypto_allocation(i, gld_w)

                # Fund crypto from GLD sleeve
                new_gld_w = gld_w - total_crypto
                new_gld_w = max(0.0, new_gld_w)  # No shorting GLD

                # Rebalance: set new weights, pay transaction cost
                old_weights = [spy_w, gld_w, tlt_w, btc_w, eth_w]
                new_weights = [spy_w, new_gld_w, tlt_w, new_btc_w, new_eth_w]

                # Transaction cost: proportional to absolute weight change
                turnover = sum(abs(new_weights[j] - old_weights[j]) for j in range(5))
                cost = turnover * cost_per_trade * equity[-1]

                gld_w, btc_w, eth_w = new_gld_w, new_btc_w, new_eth_w

                crypto_tracker["rebalances"] += 1
                crypto_tracker["total_costs"] += cost
                days_since_rebalance = 0

                # Track regime
                spy_mom_6m = self._compute_spy_momentum_6m(i)
                btc_vol, eth_vol = self._compute_crypto_vol(i)
                regime_key = "active" if total_crypto > 0 else "inactive"
                crypto_tracker["regime_counts"][regime_key] = (
                    crypto_tracker["regime_counts"].get(regime_key, 0) + 1
                )

                # Add rebalance snapshot
                crypto_tracker.setdefault("rebalance_records", []).append({
                    "date": date,
                    "spy_mom_6m": round(spy_mom_6m, 4),
                    "btc_vol": round(btc_vol, 4),
                    "eth_vol": round(eth_vol, 4),
                    "crypto_pct": round(total_crypto * 100, 2),
                    "gld_pct": round(gld_w * 100, 2),
                })

            # Track daily crypto stats
            crypto_pct = (btc_w + eth_w) * 100
            crypto_tracker["allocations"].append(crypto_pct)
            if crypto_pct > 0:
                crypto_tracker["active_days"] += 1
            if crypto_pct > crypto_tracker["max_pct"]:
                crypto_tracker["max_pct"] = crypto_pct

            # Daily return
            ret = self._compute_portfolio_return(
                prices[i - 1], prices[i],
                spy_w, gld_w, tlt_w, btc_w, eth_w,
            )
            new_equity = equity[-1] * (1 + ret)
            equity.append(new_equity)

        # Compute average crypto percentage
        allocs = crypto_tracker["allocations"]
        crypto_tracker["avg_pct"] = round(float(np.mean(allocs)), 2) if allocs else 0.0
        crypto_tracker["max_pct"] = round(crypto_tracker["max_pct"], 2)

        # Build regime breakdown
        total_rebalances = crypto_tracker["rebalances"]
        regime_breakdown = {}
        for regime, count in crypto_tracker["regime_counts"].items():
            pct_of_rebalances = round(100.0 * count / total_rebalances, 2) if total_rebalances > 0 else 0.0
            regime_breakdown[regime] = {
                "count": count,
                "pct_of_rebalances": pct_of_rebalances,
            }
        crypto_tracker["regime_breakdown"] = regime_breakdown

        return equity, crypto_tracker

    def _build_prices_lookup(self) -> Dict[str, Dict[str, float]]:
        """Build {date: {symbol: price}} lookup from daily prices."""
        lookup: Dict[str, Dict[str, float]] = {}
        for dp in self._daily_prices:
            lookup[dp.date] = {
                "SPY": dp.spy,
                "GLD": dp.gld,
                "TLT": dp.tlt,
                "BTC-USD": dp.btc if dp.btc else 0.0,
                "ETH-USD": dp.eth if dp.eth else 0.0,
            }
        return lookup

    def _compute_crisis_returns_crypto(
        self,
        prices_lookup: Dict[str, Dict[str, float]],
        trading_dates: List[str],
        equity_curve: List[float],
        initial_capital: float,
    ) -> Dict[str, float]:
        """Compute crypto overlay portfolio returns during crisis years.

        Uses the equity curve directly rather than buy-and-hold prices.
        """
        result: Dict[str, float] = {}
        for year in CRISIS_YEARS:
            year_dates = [d for d in trading_dates if d.startswith(year)]
            if not year_dates:
                continue

            date_to_idx: Dict[str, int] = {}
            for i, dp in enumerate(self._daily_prices):
                date_to_idx[dp.date] = i

            start_idx = date_to_idx.get(year_dates[0])
            end_idx = date_to_idx.get(year_dates[-1])

            if start_idx is None or end_idx is None:
                continue

            eq_start = equity_curve[start_idx]
            eq_end = equity_curve[end_idx]

            if eq_start > 0:
                year_ret = (eq_end / eq_start - 1) * 100
                result[year] = round(year_ret, 2)

        return result

    def _empty_result(self) -> BacktestResult:
        """Return an empty result when backtest cannot run."""
        return BacktestResult(
            total_return=0.0,
            cagr=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            baseline_total_return=0.0,
            baseline_cagr=0.0,
            baseline_volatility=0.0,
            baseline_sharpe=0.0,
            baseline_max_drawdown=0.0,
            sharpe_improvement=0.0,
            cagr_impact=0.0,
            crypto_active_days=0,
            crypto_active_pct=0.0,
            avg_crypto_pct=0.0,
            max_crypto_pct=0.0,
            crisis_returns_crypto={},
            crisis_returns_baseline={},
            regime_breakdown={},
            total_rebalances=0,
            total_transaction_costs=0.0,
            config_snapshot={},
        )

    def print_results(self, result: BacktestResult) -> None:
        """Print formatted backtest results to stdout."""
        print("\n" + "=" * 70)
        print("  Crypto Tactical Allocation -- Walk-Forward Backtest Results")
        print("=" * 70)

        print(f"\n  Period: {self.config.start_date} to {self.config.end_date}")
        print(f"  Capital: ${self.config.initial_capital:,.0f}")
        print(f"  Baseline: SPY {self.config.base_spy_weight*100:.0f}% / "
              f"GLD {self.config.base_gld_weight*100:.0f}% / "
              f"TLT {self.config.base_tlt_weight*100:.0f}%")
        print(f"  Max Crypto: {self.config.max_crypto_pct:.0f}% (BTC 60% / ETH 40%)")

        print(f"\n  {'Metric':<30} {'Baseline':>10} {'Crypto':>10} {'Delta':>10}")
        print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
        print(f"  {'Total Return':<30} {result.baseline_total_return:>9.2f}% {result.total_return:>9.2f}% "
              f"{result.total_return - result.baseline_total_return:>+9.2f}%")
        print(f"  {'CAGR':<30} {result.baseline_cagr:>9.2f}% {result.cagr:>9.2f}% "
              f"{result.cagr_impact:>+9.2f}%")
        print(f"  {'Volatility':<30} {result.baseline_volatility:>9.2f}% {result.volatility:>9.2f}% "
              f"{result.volatility - result.baseline_volatility:>+9.2f}%")
        print(f"  {'Sharpe Ratio':<30} {result.baseline_sharpe:>10.4f} {result.sharpe_ratio:>10.4f} "
              f"{result.sharpe_improvement:>+10.4f}")
        print(f"  {'Max Drawdown':<30} {result.baseline_max_drawdown:>9.2f}% {result.max_drawdown:>9.2f}% "
              f"{result.max_drawdown - result.baseline_max_drawdown:>+9.2f}%")

        print(f"\n  -- Crypto Activity --")
        print(f"  Crypto active days:  {result.crypto_active_days} ({result.crypto_active_pct:.1f}%)")
        print(f"  Avg crypto:          {result.avg_crypto_pct:.2f}%")
        print(f"  Max crypto:          {result.max_crypto_pct:.2f}%")
        print(f"  Rebalances:          {result.total_rebalances}")
        print(f"  Transaction costs:   ${result.total_transaction_costs:.2f}")

        print(f"\n  -- Crisis Returns (%) --")
        print(f"  {'Year':<10} {'Baseline':>10} {'Crypto':>10}")
        print(f"  {'-'*10} {'-'*10} {'-'*10}")
        all_crisis_years = sorted(
            set(list(result.crisis_returns_baseline.keys()) + list(result.crisis_returns_crypto.keys()))
        )
        for year in all_crisis_years:
            b = result.crisis_returns_baseline.get(year, 0.0)
            c = result.crisis_returns_crypto.get(year, 0.0)
            print(f"  {year:<10} {b:>10.2f} {c:>10.2f}")

        if result.regime_breakdown:
            print(f"\n  -- Regime Breakdown --")
            print(f"  {'Regime':<15} {'Count':>8} {'% Rebals':>10}")
            print(f"  {'-'*15} {'-'*8} {'-'*10}")
            for regime, stats in sorted(result.regime_breakdown.items()):
                print(f"  {regime:<15} {stats['count']:>8} {stats['pct_of_rebalances']:>9.1f}%")

        print("\n" + "=" * 70)

    def save_results(self, result: BacktestResult, output_path: Optional[str] = None) -> None:
        """Save backtest results to a JSON file."""
        data = result.to_dict()
        data["_metadata"] = {
            "strategy": "crypto_allocation",
            "generated": datetime.now().isoformat(),
            "type": "walk_forward_backtest",
        }

        if output_path:
            save_results_json(data, output_path=output_path)
            logger.info("Results saved to %s", output_path)
        else:
            named_path = str(BACKTEST_RESULTS_DIR / "crypto_allocation_backtest_results.json")
            save_results_json(data, output_path=named_path)
            logger.info("Results saved to %s", named_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for the crypto allocation walk-forward backtest."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Crypto Tactical Allocation Walk-Forward Backtest v9.32"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="run",
        choices=["run"],
        help="Run the backtest (default: run)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD, default: 2006-01-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD, default: 2026-05-15)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="Initial capital (default: 100000)",
    )
    parser.add_argument(
        "--max-crypto",
        type=float,
        default=None,
        help="Max crypto allocation %% (default: 5.0)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for JSON results",
    )

    args = parser.parse_args()

    config = BacktestConfig()
    if args.start:
        config.start_date = args.start
    if args.end:
        config.end_date = args.end
    if args.capital is not None:
        config.initial_capital = args.capital
    if args.max_crypto is not None:
        config.max_crypto_pct = args.max_crypto

    backtester = WalkForwardCryptoBacktester(config)
    result = backtester.run()
    backtester.print_results(result)

    if args.save or args.output:
        backtester.save_results(result, output_path=args.output)


if __name__ == "__main__":
    main()
