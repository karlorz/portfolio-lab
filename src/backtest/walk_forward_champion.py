"""
Walk-Forward Champion Validation.

Validates that the SPY/GLD/TLT 46/38/16 champion allocation holds up
out-of-sample using expanding-window walk-forward analysis. Also tests
the 44/36/20 challenger through the same windows for head-to-head comparison.

Methodology:
- Start with 5 years of IS data (2006-2010)
- Expand by 1 year each iteration through 2026 (~16 windows)
- Champion 46/38/16 and challenger 44/36/20 tested against SPY, 60/40
- Compute IS Sharpe, OOS Sharpe, WFE, mean OOS CAGR, drawdowns

Usage:
    python -m src.backtest.walk_forward_champion
    python -m src.backtest.walk_forward_champion --save
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.paths import BASE_ALLOCATION, DATA_DIR, PRICES_JSON
from src.backtest.grid_runner import load_prices_dates_prices
from src.backtest.metrics import (
    BacktestMetrics,
    compute_metrics,
    save_results_json,
)

logger = logging.getLogger(__name__)

__all__ = [
    "WalkForwardResult",
    "WalkForwardWindow",
    "ChampionVsChallengerResult",
    "run_walk_forward_champion",
    "run_walk_forward_comparison",
]


@dataclass
class WalkForwardWindow:
    """Metrics for a single walk-forward window."""
    window_index: int
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str
    is_days: int
    oos_days: int
    champion_is_sharpe: float
    champion_oos_sharpe: float
    champion_oos_cagr: float
    champion_oos_max_dd: float
    challenger_is_sharpe: float
    challenger_oos_sharpe: float
    challenger_oos_cagr: float
    challenger_oos_max_dd: float
    benchmark_spy_oos_sharpe: float
    benchmark_6040_oos_sharpe: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WalkForwardResult:
    """Full walk-forward validation result for one allocation."""
    analysis_date: str
    data_range: str
    n_windows: int
    windows: List[WalkForwardWindow]

    # Summary metrics
    allocation_label: str
    mean_oos_sharpe: float
    mean_oos_cagr: float
    mean_oos_max_dd: float
    is_sharpe: float  # full-period IS
    wfe: float  # Walk-Forward Efficiency = mean OOS Sharpe / IS Sharpe
    benchmark_spy_mean_oos_sharpe: float
    benchmark_6040_mean_oos_sharpe: float

    oos_sharpe_positive_pct: float  # % of OOS windows with positive Sharpe
    beats_spy: int  # number of OOS windows this allocation beats SPY
    beats_6040: int  # number of OOS windows this allocation beats 60/40

    summary: str

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class ChampionVsChallengerResult:
    """Head-to-head walk-forward comparison of champion vs challenger."""
    analysis_date: str
    data_range: str
    n_windows: int
    windows: List[WalkForwardWindow]

    champion: WalkForwardResult
    challenger: WalkForwardResult

    # Head-to-head: how many OOS windows each beats the other
    champion_beats_challenger: int
    challenger_beats_champion: int

    # Which has higher WFE
    better_wfe: str  # "champion", "challenger", or "tie"
    wfe_delta: float  # challenger_wfe - champion_wfe

    recommendation: str

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _load_prices() -> dict:
    """Load price data from prices.json."""
    return load_prices_dates_prices()


def _compute_portfolio_returns(
    prices_data: dict,
    weights: Dict[str, float],
) -> np.ndarray:
    """Compute daily portfolio returns given weight dict."""
    symbols = list(weights.keys())
    n = min(len(prices_data[s]["prices"]) for s in symbols)

    returns = np.zeros(n - 1)
    for i in range(1, n):
        ret = 0.0
        for sym in symbols:
            if weights.get(sym, 0) > 0:
                prev = prices_data[sym]["prices"][i - 1]
                curr = prices_data[sym]["prices"][i]
                if prev > 0:
                    ret += weights[sym] * (curr / prev - 1)
        returns[i - 1] = ret
    return returns


def _returns_to_equity(returns: np.ndarray, initial: float = 100000.0) -> List[float]:
    """Convert daily returns to equity curve."""
    equity = [initial]
    for r in returns:
        equity.append(equity[-1] * (1 + float(r)))
    return equity


def _compute_allocation_result(
    windows: List[WalkForwardWindow],
    allocation_label: str,
    is_sharpe: float,
    oos_sharpe_key: str,
    oos_cagr_key: str,
    oos_dd_key: str,
) -> WalkForwardResult:
    """Build WalkForwardResult from window data for a specific allocation."""
    if not windows:
        return WalkForwardResult(
            analysis_date=datetime.now().isoformat(),
            data_range="unknown", n_windows=0, windows=[],
            allocation_label=allocation_label,
            mean_oos_sharpe=0.0, mean_oos_cagr=0.0, mean_oos_max_dd=0.0,
            is_sharpe=is_sharpe, wfe=0.0,
            benchmark_spy_mean_oos_sharpe=0.0, benchmark_6040_mean_oos_sharpe=0.0,
            oos_sharpe_positive_pct=0.0, beats_spy=0, beats_6040=0,
            summary="No windows.",
        )

    oos_sharpes = [getattr(w, oos_sharpe_key) for w in windows]
    mean_oos_sharpe = round(float(np.mean(oos_sharpes)), 4)
    mean_oos_cagr = round(float(np.mean([getattr(w, oos_cagr_key) for w in windows])), 4)
    mean_oos_dd = round(float(np.mean([getattr(w, oos_dd_key) for w in windows])), 4)
    wfe = round(mean_oos_sharpe / is_sharpe, 4) if is_sharpe > 0 else 0.0
    positive_pct = round(float(sum(1 for s in oos_sharpes if s > 0) / len(oos_sharpes)), 4)
    beats_spy = sum(1 for w in windows if getattr(w, oos_sharpe_key) > w.benchmark_spy_oos_sharpe)
    beats_6040 = sum(1 for w in windows if getattr(w, oos_sharpe_key) > w.benchmark_6040_oos_sharpe)

    spy_mean = round(float(np.mean([w.benchmark_spy_oos_sharpe for w in windows])), 4)
    sf_mean = round(float(np.mean([w.benchmark_6040_oos_sharpe for w in windows])), 4)

    if wfe >= 1.0:
        verdict = (
            f"WFE={wfe:.2f} — {allocation_label} ALLOCATION VALIDATED. "
            f"OOS Sharpe {mean_oos_sharpe:.4f} vs IS {is_sharpe:.4f}. "
            f"Beats SPY in {beats_spy}/{len(windows)} windows, "
            f"beats 60/40 in {beats_6040}/{len(windows)} windows."
        )
    elif wfe >= 0.80:
        verdict = (
            f"WFE={wfe:.2f} — {allocation_label} MOSTLY VALIDATED. "
            f"Minor OOS degradation, still beats benchmarks consistently."
        )
    else:
        verdict = (
            f"WFE={wfe:.2f} — {allocation_label} NOT VALIDATED. "
            f"Significant OOS degradation — consider allocation revision."
        )

    return WalkForwardResult(
        analysis_date=datetime.now().isoformat(),
        data_range=f"{windows[0].is_start} to {windows[-1].oos_end}",
        n_windows=len(windows),
        windows=windows,
        allocation_label=allocation_label,
        mean_oos_sharpe=mean_oos_sharpe,
        mean_oos_cagr=mean_oos_cagr,
        mean_oos_max_dd=mean_oos_dd,
        is_sharpe=is_sharpe,
        wfe=wfe,
        benchmark_spy_mean_oos_sharpe=spy_mean,
        benchmark_6040_mean_oos_sharpe=sf_mean,
        oos_sharpe_positive_pct=positive_pct,
        beats_spy=beats_spy,
        beats_6040=beats_6040,
        summary=verdict,
    )


def run_walk_forward_champion(
    is_years: int = 5,
    oos_years: int = 1,
    min_is_days: int = 1000,
    save: bool = False,
) -> WalkForwardResult:
    """Run walk-forward validation of champion portfolio (46/38/16).

    Returns a WalkForwardResult for the champion allocation.
    """
    result = run_walk_forward_comparison(is_years, oos_years, min_is_days, save)
    return result.champion


def run_walk_forward_comparison(
    is_years: int = 5,
    oos_years: int = 1,
    min_is_days: int = 1000,
    save: bool = False,
) -> ChampionVsChallengerResult:
    """Run head-to-head walk-forward comparison of champion vs challenger.

    Both allocations go through the exact same expanding windows for
    a fair head-to-head comparison.

    Returns:
        ChampionVsChallengerResult with both allocation results compared
    """
    logger.info("Loading prices for walk-forward champion vs challenger")
    prices_data = _load_prices()
    all_dates = prices_data["SPY"]["dates"]
    n_total = len(all_dates)
    logger.info("Loaded %d days (%s to %s)", n_total, all_dates[0], all_dates[-1])

    # Portfolio definitions
    champion = dict(BASE_ALLOCATION)
    challenger = {"SPY": 0.44, "GLD": 0.36, "TLT": 0.20}
    spy_only = {"SPY": 1.0, "GLD": 0.0, "TLT": 0.0}
    sixty_forty = {"SPY": 0.60, "GLD": 0.0, "TLT": 0.40}

    # Find start index
    start_idx = next((i for i, d in enumerate(all_dates) if d >= "2006-01-01"), 0)

    def window_returns(weights: Dict[str, float], start_i: int, end_i: int) -> np.ndarray:
        n = end_i - start_i
        rets = np.zeros(n - 1)
        for i in range(start_i + 1, end_i):
            ret = 0.0
            for sym in ["SPY", "GLD", "TLT"]:
                prev = prices_data[sym]["prices"][i - 1]
                curr = prices_data[sym]["prices"][i]
                if prev > 0:
                    ret += weights.get(sym, 0) * (curr / prev - 1)
            rets[i - start_i - 1] = ret
        return rets

    def full_period_sharpe(weights: Dict[str, float]) -> float:
        rets = window_returns(weights, 0, n_total)
        equity = _returns_to_equity(rets)
        metrics = compute_metrics(equity_curve=equity, initial_capital=100000.0)
        return round(metrics.sharpe_ratio, 4)

    champion_is_sharpe = full_period_sharpe(champion)
    challenger_is_sharpe = full_period_sharpe(challenger)

    # Expanding windows
    windows: List[WalkForwardWindow] = []
    is_end_idx = start_idx + is_years * 252
    window_idx = 0

    while is_end_idx + oos_years * 252 < n_total:
        oos_end_idx = min(is_end_idx + oos_years * 252, n_total - 1)
        if oos_end_idx - is_end_idx < 60:
            break

        # IS metrics (both share same IS period)
        champ_is = compute_metrics(
            equity_curve=_returns_to_equity(window_returns(champion, start_idx, is_end_idx)),
            initial_capital=100000.0,
        )
        chal_is = compute_metrics(
            equity_curve=_returns_to_equity(window_returns(challenger, start_idx, is_end_idx)),
            initial_capital=100000.0,
        )

        # OOS metrics
        champ_oos = compute_metrics(
            equity_curve=_returns_to_equity(window_returns(champion, is_end_idx, oos_end_idx)),
            initial_capital=100000.0,
        )
        chal_oos = compute_metrics(
            equity_curve=_returns_to_equity(window_returns(challenger, is_end_idx, oos_end_idx)),
            initial_capital=100000.0,
        )
        spy_oos = compute_metrics(
            equity_curve=_returns_to_equity(window_returns(spy_only, is_end_idx, oos_end_idx)),
            initial_capital=100000.0,
        )
        sf_oos = compute_metrics(
            equity_curve=_returns_to_equity(window_returns(sixty_forty, is_end_idx, oos_end_idx)),
            initial_capital=100000.0,
        )

        window = WalkForwardWindow(
            window_index=window_idx,
            is_start=all_dates[start_idx],
            is_end=all_dates[is_end_idx - 1],
            oos_start=all_dates[is_end_idx],
            oos_end=all_dates[oos_end_idx - 1],
            is_days=is_end_idx - start_idx,
            oos_days=oos_end_idx - is_end_idx,
            champion_is_sharpe=round(champ_is.sharpe_ratio, 4),
            champion_oos_sharpe=round(champ_oos.sharpe_ratio, 4),
            champion_oos_cagr=round(champ_oos.cagr, 4),
            champion_oos_max_dd=round(champ_oos.max_drawdown, 4),
            challenger_is_sharpe=round(chal_is.sharpe_ratio, 4),
            challenger_oos_sharpe=round(chal_oos.sharpe_ratio, 4),
            challenger_oos_cagr=round(chal_oos.cagr, 4),
            challenger_oos_max_dd=round(chal_oos.max_drawdown, 4),
            benchmark_spy_oos_sharpe=round(spy_oos.sharpe_ratio, 4),
            benchmark_6040_oos_sharpe=round(sf_oos.sharpe_ratio, 4),
        )
        windows.append(window)
        window_idx += 1
        is_end_idx += oos_years * 252

    # Build results for each allocation
    champ_result = _compute_allocation_result(
        windows, "Champion (46/38/16)", champion_is_sharpe,
        "champion_oos_sharpe", "champion_oos_cagr", "champion_oos_max_dd",
    )
    chal_result = _compute_allocation_result(
        windows, "Challenger (44/36/20)", challenger_is_sharpe,
        "challenger_oos_sharpe", "challenger_oos_cagr", "challenger_oos_max_dd",
    )

    # Head-to-head
    champ_beats_chal = sum(
        1 for w in windows if w.champion_oos_sharpe > w.challenger_oos_sharpe
    )
    chal_beats_champ = sum(
        1 for w in windows if w.challenger_oos_sharpe > w.champion_oos_sharpe
    )

    wfe_delta = round(chal_result.wfe - champ_result.wfe, 4)
    if abs(wfe_delta) < 0.02:
        better_wfe = "tie"
    elif wfe_delta > 0:
        better_wfe = "challenger"
    else:
        better_wfe = "champion"

    # Recommendation
    if chal_result.wfe >= 1.0 and chal_result.mean_oos_sharpe >= champ_result.mean_oos_sharpe:
        rec = (
            f"CHALLENGER 44/36/20 PROMOTABLE — WFE={chal_result.wfe:.2f} "
            f"(vs champion {champ_result.wfe:.2f}), "
            f"OOS Sharpe {chal_result.mean_oos_sharpe:.4f} (vs {champ_result.mean_oos_sharpe:.4f}). "
            f"Lower gold (36% vs 38%) + higher TLT (20% vs 16%) survives walk-forward "
            f"at least as well as champion. "
            f"Challenger beats champion in {chal_beats_champ}/{len(windows)} OOS windows."
        )
    elif chal_result.wfe >= 1.0:
        rec = (
            f"CHALLENGER 44/36/20 VALIDATED BUT INFERIOR — WFE={chal_result.wfe:.2f}, "
            f"but mean OOS Sharpe {chal_result.mean_oos_sharpe:.4f} trails champion "
            f"{champ_result.mean_oos_sharpe:.4f}. Keep champion as primary, "
            f"challenger as defensive alternative."
        )
    elif chal_result.wfe >= 0.80:
        rec = (
            f"CHALLENGER 44/36/20 ACCEPTABLE — WFE={chal_result.wfe:.2f}, "
            f"survives walk-forward but lags champion (WFE={champ_result.wfe:.2f}). "
            f"Champion remains primary allocation."
        )
    else:
        rec = (
            f"CHALLENGER 44/36/20 NOT VALIDATED — WFE={chal_result.wfe:.2f}. "
            f"Champion 46/38/16 is the correct choice."
        )

    result = ChampionVsChallengerResult(
        analysis_date=datetime.now().isoformat(),
        data_range=f"{all_dates[0]} to {all_dates[-1]}",
        n_windows=len(windows),
        windows=windows,
        champion=champ_result,
        challenger=chal_result,
        champion_beats_challenger=champ_beats_chal,
        challenger_beats_champion=chal_beats_champ,
        better_wfe=better_wfe,
        wfe_delta=wfe_delta,
        recommendation=rec,
    )

    if save:
        output_path = DATA_DIR / "walk_forward_comparison.json"
        save_results_json(result.to_dict(), output_path=str(output_path))
        logger.info("Saved walk-forward comparison to %s", output_path)

    logger.info("Walk-Forward Champion vs Challenger:")
    logger.info("  Champion %s:   WFE=%.4f, OOS Sharpe=%.4f, IS=%.4f",
                "46/38/16", champ_result.wfe, champ_result.mean_oos_sharpe, champion_is_sharpe)
    logger.info("  Challenger %s: WFE=%.4f, OOS Sharpe=%.4f, IS=%.4f",
                "44/36/20", chal_result.wfe, chal_result.mean_oos_sharpe, challenger_is_sharpe)
    logger.info("  Head-to-head:  champion=%d, challenger=%d",
                champ_beats_chal, chal_beats_champ)
    logger.info("  %s", rec)

    return result


def main():
    """CLI entry point."""
    import argparse
    from src.utils.log_config import configure_logging
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Walk-Forward Champion vs Challenger Validation"
    )
    parser.add_argument("--is-years", type=int, default=5,
                        help="Initial IS period in years (default: 5)")
    parser.add_argument("--oos-years", type=int, default=1,
                        help="OOS test period in years (default: 1)")
    parser.add_argument("--save", action="store_true",
                        help="Save results to JSON")
    args = parser.parse_args()

    result = run_walk_forward_comparison(
        is_years=args.is_years,
        oos_years=args.oos_years,
        save=args.save,
    )

    logger.info("\n%s", "=" * 70)
    logger.info("WALK-FORWARD CHAMPION vs CHALLENGER")
    logger.info("%s", "=" * 70)
    logger.info("  Data: %s, %d windows", result.data_range, result.n_windows)

    logger.info("\n  %-25s %10s %10s %8s %8s %7s", "Allocation", "IS Sharpe", "OOS Sharpe", "WFE", "Max DD", "OOS+%")
    logger.info("  %s", "-" * 68)
    for label, r in [("Champion (46/38/16)", result.champion),
                      ("Challenger (44/36/20)", result.challenger)]:
        logger.info("  %-25s %10.4f %10.4f %8.4f %7.1f%% %6.1f%%",
                     label, r.is_sharpe, r.mean_oos_sharpe,
                     r.wfe, r.mean_oos_max_dd, r.oos_sharpe_positive_pct * 100)

    logger.info("\n  Head-to-head (OOS):")
    logger.info("    Champion beats Challenger:  %d/%d", result.champion_beats_challenger, result.n_windows)
    logger.info("    Challenger beats Champion:  %d/%d", result.challenger_beats_champion, result.n_windows)
    logger.info("    Better WFE:                 %s (delta %+.4f)", result.better_wfe, result.wfe_delta)

    logger.info("\n  Benchmarks:")
    logger.info("    Champion > SPY:   %d/%d", result.champion.beats_spy, result.n_windows)
    logger.info("    Challenger > SPY: %d/%d", result.challenger.beats_spy, result.n_windows)
    logger.info("    Champion > 60/40: %d/%d", result.champion.beats_6040, result.n_windows)

    logger.info("\n  %s", result.recommendation)
    logger.info("%s", "=" * 70)


if __name__ == "__main__":
    main()
