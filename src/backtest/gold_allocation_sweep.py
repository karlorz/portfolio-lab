"""
Gold Allocation Sweep — v4.95 Analysis

Sweep GLD weight from 20-55% to find optimal gold allocation.
Motivated by 2025 research:
- BlackRock: stock-bond correlation shifted positive; gold replaces bond diversifier
- Goldman Sachs: "strategic tilting" — world portfolio should hold more gold
- BofA: recommends 40% gold allocation
- World Gold Council: at inflation >2.5%, optimal gold weight rises significantly

Champion baseline: SPY/GLD/TLT 46/38/16, Sharpe 0.79 (2005-2026)

Tests two funding strategies:
1. Fund GLD increase from TLT (bonds → gold shift)
2. Fund GLD increase from SPY (equity → gold shift)
3. Replace TLT with IEF (shorter duration) while shifting to gold

Also tests IEF replacing some or all TLT to address the positive
stock-bond correlation concern.

Usage:
    python -m src.backtest.gold_allocation_sweep run
    python -m src.backtest.gold_allocation_sweep run --output data/gold_sweep_results.json
"""

import json
import logging
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.paths import DATA_DIR, BASE_ALLOCATION, PRICES_JSON
from src.backtest.metrics import save_results_json

logger = logging.getLogger(__name__)


@dataclass
class GoldSweepRow:
    """Single row in the gold allocation sweep."""
    spy_pct: float
    gld_pct: float
    tlt_pct: float
    ief_pct: float
    label: str
    cagr: float
    vol: float
    sharpe: float
    max_dd: float
    sharpe_delta: float      # vs baseline 46/38/16
    year_2008: float
    year_2020: float
    year_2022: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GoldSweepResult:
    """Complete gold allocation sweep results."""
    timestamp: str
    data_range: str
    n_days: int
    baseline_cagr: float
    baseline_vol: float
    baseline_sharpe: float
    baseline_max_dd: float

    rows: List[GoldSweepRow]

    best_sharpe_row: Optional[dict]
    best_drawdown_row: Optional[dict]
    best_2022_row: Optional[dict]

    recommendation: str

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


__all__ = ['GoldSweepRow', 'GoldSweepResult', 'GoldAllocationSweep', 'run_gold_sweep']


class GoldAllocationSweep:
    """
    Sweep gold allocation weights using real Yahoo Finance price data.

    Tests GLD weight from 20% to 55% in 2% increments, funded from TLT, SPY,
    or with IEF replacing TLT.
    """

    BASELINE = {k.lower(): v for k, v in BASE_ALLOCATION.items()}

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or PRICES_JSON
        self.prices: Dict[str, List[float]] = {}
        self.dates: List[str] = []

    def _load_prices(self) -> None:
        """Load real price data from prices.json."""
        from src.data.price_cache import get_prices
        raw = get_prices()

        for symbol in ["SPY", "GLD", "TLT", "IEF"]:
            if symbol in raw:
                self.prices[symbol] = [entry["p"] for entry in raw[symbol]]
                if not self.dates:
                    self.dates = [entry["d"] for entry in raw[symbol]]

        # Validate alignment
        n = len(self.dates)
        for sym, vals in self.prices.items():
            assert len(vals) == n, f"{sym} has {len(vals)} prices but {n} dates"

    def _compute_returns(self, prices: List[float]) -> List[float]:
        return [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]

    def _simulate_portfolio(
        self,
        weights: Dict[str, float],
    ) -> Tuple[float, float, float, float, Dict[str, float]]:
        """Simulate portfolio with given weights, return (cagr, vol, sharpe, max_dd, year_returns)."""
        spy_rets = self._compute_returns(self.prices["SPY"])
        gld_rets = self._compute_returns(self.prices["GLD"])
        tlt_rets = self._compute_returns(self.prices["TLT"])
        ief_rets = self._compute_returns(self.prices.get("IEF", self.prices["TLT"]))

        n = min(len(spy_rets), len(gld_rets), len(tlt_rets), len(ief_rets))

        values = [1.0]
        peak = 1.0
        daily_rets = []
        yearly_rets: Dict[str, List[float]] = {}

        for i in range(n):
            ret = (
                weights.get("spy", 0) * spy_rets[i] +
                weights.get("gld", 0) * gld_rets[i] +
                weights.get("tlt", 0) * tlt_rets[i] +
                weights.get("ief", 0) * ief_rets[i]
            )
            values.append(values[-1] * (1 + ret))
            daily_rets.append(ret)
            peak = max(peak, values[-1])

            # Track yearly returns
            year = self.dates[i + 1][:4] if i + 1 < len(self.dates) else "unknown"
            if year not in yearly_rets:
                yearly_rets[year] = []
            yearly_rets[year].append(ret)

        cagr = np.mean(daily_rets) * 252 * 100
        vol = np.std(daily_rets) * math.sqrt(252) * 100
        sharpe = cagr / vol if vol > 0 else 0
        max_dd = min((v / peak - 1) * 100 for v in values) if values else 0

        # Year-level returns
        year_total = {}
        for y, rets in yearly_rets.items():
            year_total[y] = (np.prod([1 + r for r in rets]) - 1) * 100

        return round(cagr, 2), round(vol, 2), round(sharpe, 3), round(max_dd, 2), year_total

    def run_sweep(self) -> GoldSweepResult:
        """Run full gold allocation sweep."""
        self._load_prices()

        baseline_w = {"spy": 0.46, "gld": 0.38, "tlt": 0.16, "ief": 0.0}
        b_cagr, b_vol, b_sharpe, b_dd, b_yr = self._simulate_portfolio(baseline_w)

        rows = []
        best_sharpe = b_sharpe
        best_sharpe_row = None
        best_dd = b_dd
        best_dd_row = None
        best_2022 = b_yr.get("2022", 0)
        best_2022_row = None

        def make_row(w: Dict[str, float], label: str) -> GoldSweepRow:
            nonlocal best_sharpe, best_sharpe_row, best_dd, best_dd_row, best_2022, best_2022_row

            cagr, vol, sharpe, dd, yr = self._simulate_portfolio(w)
            row = GoldSweepRow(
                spy_pct=w.get("spy", 0) * 100,
                gld_pct=w.get("gld", 0) * 100,
                tlt_pct=w.get("tlt", 0) * 100,
                ief_pct=w.get("ief", 0) * 100,
                label=label,
                cagr=cagr, vol=vol, sharpe=sharpe, max_dd=dd,
                sharpe_delta=round(sharpe - b_sharpe, 3),
                year_2008=round(yr.get("2008", 0), 2),
                year_2020=round(yr.get("2020", 0), 2),
                year_2022=round(yr.get("2022", 0), 2),
            )

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_sharpe_row = row.to_dict()
            if dd > best_dd:  # less negative = better
                best_dd = dd
                best_dd_row = row.to_dict()
            if yr.get("2022", -999) > best_2022:
                best_2022 = yr.get("2022", -999)
                best_2022_row = row.to_dict()

            return row

        # === Strategy 1: Fund GLD increase from TLT ===
        for gld_pct in range(20, 56, 2):
            gld_w = gld_pct / 100
            tlt_w = max(0, 0.16 - (gld_w - 0.38))  # fund from TLT
            spy_w = 1.0 - gld_w - tlt_w
            if spy_w < 0.1 or tlt_w < 0:
                continue
            rows.append(make_row(
                {"spy": spy_w, "gld": gld_w, "tlt": tlt_w, "ief": 0},
                f"GLD+from TLT"
            ))

        # === Strategy 2: Fund GLD increase from SPY ===
        for gld_pct in range(20, 56, 2):
            gld_w = gld_pct / 100
            spy_w = max(0, 0.46 - (gld_w - 0.38))  # fund from SPY
            tlt_w = 1.0 - spy_w - gld_w
            if tlt_w < 0 or spy_w < 0.1:
                continue
            rows.append(make_row(
                {"spy": spy_w, "gld": gld_w, "tlt": tlt_w, "ief": 0},
                f"GLD+from SPY"
            ))

        # === Strategy 3: Replace TLT with IEF + increase GLD ===
        for gld_pct in range(30, 56, 2):
            gld_w = gld_pct / 100
            for ief_pct in [4, 8, 12, 16]:
                ief_w = ief_pct / 100
                spy_w = 1.0 - gld_w - ief_w
                if spy_w < 0.1:
                    continue
                rows.append(make_row(
                    {"spy": spy_w, "gld": gld_w, "tlt": 0, "ief": ief_w},
                    f"GLD+IEF (no TLT)"
                ))

        # === Strategy 4: Partial IEF replacement (TLT + IEF mix) ===
        for gld_pct in [38, 40, 42, 44, 46]:
            gld_w = gld_pct / 100
            for ief_mix in [0.25, 0.5, 0.75, 1.0]:
                bond_total = 1.0 - 0.46 - gld_w
                if bond_total < 0:
                    continue
                ief_w = bond_total * ief_mix
                tlt_w = bond_total * (1 - ief_mix)
                rows.append(make_row(
                    {"spy": 0.46, "gld": gld_w, "tlt": tlt_w, "ief": ief_w},
                    f"GLD{gld_pct}/TLT+IEF mix"
                ))

        # Baseline row
        baseline_row = GoldSweepRow(
            spy_pct=46, gld_pct=38, tlt_pct=16, ief_pct=0,
            label="BASELINE (champion)",
            cagr=b_cagr, vol=b_vol, sharpe=b_sharpe, max_dd=b_dd,
            sharpe_delta=0.0,
            year_2008=round(b_yr.get("2008", 0), 2),
            year_2020=round(b_yr.get("2020", 0), 2),
            year_2022=round(b_yr.get("2022", 0), 2),
        )

        # Sort by Sharpe
        all_rows = [baseline_row] + rows
        all_rows.sort(key=lambda r: r.sharpe, reverse=True)

        # Recommendation
        if best_sharpe_row and best_sharpe > b_sharpe:
            rec = (
                f"Best Sharpe: SPY {best_sharpe_row['spy_pct']:.0f}/"
                f"GLD {best_sharpe_row['gld_pct']:.0f}/"
                f"TLT {best_sharpe_row['tlt_pct']:.0f}/"
                f"IEF {best_sharpe_row['ief_pct']:.0f} "
                f"(Sharpe {best_sharpe:.3f}, +{best_sharpe - b_sharpe:.3f} vs baseline). "
                f"Research-aligned: BlackRock/Goldman/BofA recommend higher gold weight."
            )
        else:
            rec = (
                f"Champion 46/38/16 (Sharpe {b_sharpe:.3f}) remains optimal. "
                f"Higher gold weights tested but none improve Sharpe. "
                f"Consider IEF for duration reduction without Sharpe cost."
            )

        return GoldSweepResult(
            timestamp=datetime.now().isoformat(),
            data_range=f"{self.dates[0]} to {self.dates[-1]}" if self.dates else "unknown",
            n_days=len(self.dates),
            baseline_cagr=b_cagr,
            baseline_vol=b_vol,
            baseline_sharpe=b_sharpe,
            baseline_max_dd=b_dd,
            rows=all_rows,
            best_sharpe_row=best_sharpe_row,
            best_drawdown_row=best_dd_row,
            best_2022_row=best_2022_row,
            recommendation=rec,
        )


def run_gold_sweep(output: Optional[str] = None) -> GoldSweepResult:
    """Run gold allocation sweep and optionally save results."""
    sweep = GoldAllocationSweep()
    result = sweep.run_sweep()

    if output:
        out_path = Path(output)
        save_results_json(result.to_dict(), out_path)
        logger.info("Gold sweep results saved to %s", out_path)
    else:
        out_path = DATA_DIR / "gold_allocation_sweep.json"
        save_results_json(result.to_dict(), out_path)
        logger.info("Gold sweep results saved to %s", out_path)

    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info(f"GOLD ALLOCATION SWEEP — {result.data_range} ({result.n_days} days)")
    logger.info(f"{'='*70}")
    logger.info(f"Baseline: SPY/GLD/TLT 46/38/16 — Sharpe {result.baseline_sharpe:.3f}")
    logger.info(f"{'='*70}")

    # Top 10 by Sharpe
    logger.info(f"\nTop 10 by Sharpe Ratio:")
    logger.info(f"{'SPY':>5} {'GLD':>5} {'TLT':>5} {'IEF':>5} | {'CAGR':>6} {'Vol':>6} {'Sharpe':>7} {'MaxDD':>7} {'Δ':>6} | {'2008':>7} {'2020':>7} {'2022':>7} | Label")
    logger.info("-" * 110)
    for row in result.rows[:10]:
        logger.info(
            f"{row.spy_pct:5.0f} {row.gld_pct:5.0f} {row.tlt_pct:5.0f} {row.ief_pct:5.0f} | "
            f"{row.cagr:5.1f}% {row.vol:5.1f}% {row.sharpe:7.3f} {row.max_dd:6.1f}% {row.sharpe_delta:+5.3f} | "
            f"{row.year_2008:6.1f}% {row.year_2020:6.1f}% {row.year_2022:6.1f}% | {row.label}"
        )

    logger.info(f"\n{result.recommendation}")

    return result


if __name__ == "__main__":
    import argparse
    from src.utils.log_config import configure_logging
    configure_logging()

    parser = argparse.ArgumentParser(description="Gold Allocation Sweep")
    parser.add_argument("command", choices=["run"], help="Command to run")
    parser.add_argument("--output", help="Output JSON file path")
    args = parser.parse_args()

    if args.command == "run":
        run_gold_sweep(output=args.output)
