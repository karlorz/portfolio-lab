#!/usr/bin/env python3
"""
Regime Gating Gap Analysis — v961

Analyzes regime-performance characteristics for 3 unexamined signals:
1. CROSS_ASSET_RV (mean-reversion) — may underperform in strong trends
2. UNIFIED_OVERLAY (composite) — sub-component weaknesses unknown
3. CRYPTO_MOMENTUM — correlation breakdown in CRISIS

Also examines RECOVERY regime and LOW_VOL detection effectiveness.
"""
import json
import logging
import sys
from collections import defaultdict
from enum import Enum
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))



class Regime(str, Enum):
    LOW_VOL = "low_vol"
    NORMAL = "normal"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"
    RECOVERY = "recovery"


# Thresholds matched to ensemble_voter.py EnsembleVoter class
CRISIS_VOL_THRESHOLD = 0.30
CRISIS_DRAWDOWN_THRESHOLD = -0.10
HIGH_VOL_VOL_THRESHOLD = 0.20
HIGH_VOL_DRAWDOWN_THRESHOLD = -0.05
HIGH_VOL_MOM_THRESHOLD = 0.0
LOW_VOL_VOL_THRESHOLD = 0.12
LOW_VOL_MOM_THRESHOLD = 0.01
RECOVERY_DRAWDOWN_THRESHOLD = -0.03
RECOVERY_MOM_THRESHOLD = 0.02


def load_prices(path="public/data/prices.json") -> dict:
    with open(SRC_DIR / path) as f:
        data = json.load(f)
    return data


def prices_to_returns(prices: dict) -> dict:
    """Convert price dict {symbol: [{d: ..., p: ...}, ...]} to daily returns."""
    result = {}
    for sym, series in prices.items():
        p = np.array([entry["p"] for entry in series], dtype=float)
        dates = [entry["d"] for entry in series]
        ret = p[1:] / p[:-1] - 1.0
        dates = dates[1:]  # align dates
        result[sym] = {"returns": ret, "dates": dates}
    return result


def detect_regime(vol_20d: float, drawdown: float, mom_20d: float) -> str:
    """Replicate ensemble_voter.py detect_regime logic."""
    if vol_20d > CRISIS_VOL_THRESHOLD or drawdown < CRISIS_DRAWDOWN_THRESHOLD:
        return "crisis"
    elif vol_20d > HIGH_VOL_VOL_THRESHOLD or (
        drawdown < HIGH_VOL_DRAWDOWN_THRESHOLD and mom_20d < HIGH_VOL_MOM_THRESHOLD
    ):
        return "high_vol"
    elif drawdown < RECOVERY_DRAWDOWN_THRESHOLD and mom_20d > RECOVERY_MOM_THRESHOLD:
        return "recovery"
    elif vol_20d < LOW_VOL_VOL_THRESHOLD and mom_20d > LOW_VOL_MOM_THRESHOLD:
        return "low_vol"
    else:
        return "normal"


def compute_regime_labels(spy_returns: np.ndarray) -> list:
    """Compute regime labels for each trading day with lookback."""
    regimes = []
    for i in range(len(spy_returns)):
        if i < 20:
            regimes.append("normal")
            continue
        window = spy_returns[max(0, i - 19): i + 1]
        vol_20d = np.std(window) * np.sqrt(252)
        cum_ret = np.cumprod(1 + spy_returns[:i + 1])
        running_max = np.maximum.accumulate(cum_ret)
        drawdown = (cum_ret[-1] / running_max[-1]) - 1
        mom_20d = np.sum(window)
        regimes.append(detect_regime(vol_20d, drawdown, mom_20d))
    return regimes


# --- Signal Simulations ---

def simulate_cross_asset_rv(spy_returns: np.ndarray, tlt_returns: np.ndarray,
                             gld_returns: np.ndarray) -> np.ndarray:
    """
    Cross-asset relative value signal.
    Mean-reversion on cross-asset relationships.
    Signal is positive when assets diverge from 20-day z-score relationship.
    """
    signal = np.zeros(len(spy_returns))
    for i in range(20, len(spy_returns)):
        window = slice(i - 19, i + 1)
        r_spy = spy_returns[window]
        r_tlt = tlt_returns[window]
        r_gld = gld_returns[window]

        # Z-score of recent cross-asset divergence
        rel_spy_tlt = r_spy - r_tlt
        rel_spy_gld = r_spy - r_gld
        combined = 0.5 * (rel_spy_tlt + rel_spy_gld)

        z = (combined[-1] - np.mean(combined)) / (np.std(combined) + 1e-10)
        # Mean-reversion: negative z-score → positive signal (reversion expected)
        signal[i] = -np.clip(z, -1, 1)
    return signal


def simulate_unified_overlay(spy_returns: np.ndarray, tlt_returns: np.ndarray,
                              gld_returns: np.ndarray) -> np.ndarray:
    """
    Unified overlay: composite of collar + bond_duration + crypto_tactical + calendar.
    Simplification: uses a weighted combination of regime-aware signals.
    """
    signal = np.zeros(len(spy_returns))
    for i in range(20, len(spy_returns)):
        window = slice(i - 19, i + 1)

        # Bond duration signal: positive when bonds outperform
        r_tlt = tlt_returns[window]
        bond_mom = np.sum(r_tlt)
        bond_sig = np.clip(bond_mom * 5, -0.3, 0.3)

        # Collar-like: reduce equity when vol is high
        r_spy = spy_returns[window]
        vol_regime = np.std(r_spy) * np.sqrt(252)
        collar_sig = -np.clip((vol_regime - 0.15) * 2, -0.3, 0.3)

        # Calendar seasonality: slight positive bias Nov-Apr
        cal_sig = 0.05  # small constant positive

        # Gold hedge: positive in crisis
        spy_dd = (np.cumprod(1 + r_spy)[-1] / np.maximum.accumulate(
            np.cumprod(1 + r_spy))[-1]) - 1
        gold_sig = np.clip(abs(spy_dd) * 2, 0, 0.2) if spy_dd < -0.03 else 0.0

        signal[i] = np.clip(bond_sig + collar_sig + cal_sig + gold_sig, -0.5, 0.5)
    return signal


def simulate_crypto_momentum(spy_returns: np.ndarray) -> np.ndarray:
    """
    Crypto momentum proxy.
    Uses a high-beta momentum strategy with regime-dependent behavior.
    In CRISIS (2020-style correlation breakdown), crypto momentum fails.
    """
    signal = np.zeros(len(spy_returns))
    for i in range(20, len(spy_returns)):
        window = slice(i - 19, i + 1)
        mom = np.sum(spy_returns[window])
        # Crypto momentum amplifies equity momentum
        signal[i] = np.clip(mom * 3, -0.5, 0.5)
    return signal


def signal_pnl(signal: np.ndarray, spy_returns: np.ndarray) -> np.ndarray:
    """
    Compute daily PnL from signal as portfolio tilt.
    signal * daily_return = overlay PnL.
    """
    aligned_len = min(len(signal), len(spy_returns))
    return signal[:aligned_len] * spy_returns[:aligned_len]


def regime_sharpe_analysis(spy_returns: np.ndarray, signal_pnls: np.ndarray,
                            regime_labels: list) -> dict:
    """Compute Sharpe ratio per regime for a signal."""
    min_len = min(len(spy_returns), len(signal_pnls), len(regime_labels))
    regime_returns = defaultdict(list)
    total_returns = []

    for i in range(min_len):
        regime = regime_labels[i]
        regime_returns[regime].append(signal_pnls[i])
        total_returns.append(signal_pnls[i])

    results = {}
    for regime, rets in sorted(regime_returns.items()):
        rets_arr = np.array(rets)
        if len(rets_arr) < 5:
            sharpe = 0.0
        else:
            sharpe = np.mean(rets_arr) / (np.std(rets_arr) + 1e-10) * np.sqrt(252)
        results[regime] = {
            "sharpe": round(float(sharpe), 3),
            "n_days": len(rets_arr),
            "mean_daily_return": round(float(np.mean(rets_arr)), 6),
            "std_daily_return": round(float(np.std(rets_arr)), 6),
        }

    # Overall
    total_arr = np.array(total_returns)
    if len(total_arr) >= 5:
        overall_sharpe = np.mean(total_arr) / (np.std(total_arr) + 1e-10) * np.sqrt(252)
    else:
        overall_sharpe = 0.0
    results["overall"] = {
        "sharpe": round(float(overall_sharpe), 3),
        "n_days": len(total_arr),
        "mean_daily_return": round(float(np.mean(total_arr)), 6),
        "std_daily_return": round(float(np.std(total_arr)), 6),
    }

    return results


def regime_distribution_analysis(regime_labels: list) -> dict:
    """Analyze the distribution of regimes over the full period."""
    counts = defaultdict(int)
    for r in regime_labels:
        counts[r] += 1
    total = len(regime_labels)
    return {
        "total_days": total,
        "distribution": {
            k: {"count": v, "pct": round(v / total * 100, 1)}
            for k, v in sorted(counts.items())
        }
    }


def analyze_recovery_regime(spy_returns: np.ndarray, regime_labels: list) -> dict:
    """Analyze RECOVERY regime transition characteristics."""
    recovery_entries = []
    prev_regime = None
    for i, regime in enumerate(regime_labels):
        if regime == "recovery" and prev_regime is not None and prev_regime != "recovery":
            # Entry day statistics
            _ = spy_returns[max(0, i - 5):i + 6]
            recovery_entries.append({
                "day": i,
                "prev_regime": prev_regime,
                "entry_5d_return_before": round(float(np.sum(spy_returns[max(0, i-5):i])), 4),
                "exit_5d_return_after": round(float(np.sum(spy_returns[i:min(len(spy_returns), i+6)])), 4),
            })
        prev_regime = regime

    # Compute transition matrix
    transition_matrix = defaultdict(lambda: defaultdict(int))
    for i in range(1, len(regime_labels)):
        prev = regime_labels[i - 1]
        curr = regime_labels[i]
        if prev != curr:
            transition_matrix[prev][curr] += 1

    # Convert to serializable
    tm_serial = {}
    for src, dsts in sorted(transition_matrix.items()):
        tm_serial[src] = {}
        for dst, count in sorted(dsts.items()):
            tm_serial[src][dst] = count

    # Recovery performance: average return during recovery periods
    recovery_rets = [spy_returns[i] for i in range(len(regime_labels))
                     if regime_labels[i] == "recovery"]

    return {
        "n_recovery_days": sum(1 for r in regime_labels if r == "recovery"),
        "pct_of_time": round(sum(1 for r in regime_labels if r == "recovery") / len(regime_labels) * 100, 1),
        "n_recovery_entries": len(recovery_entries),
        "transition_matrix": tm_serial,
        "recovery_mean_return": round(float(np.mean(recovery_rets)) * 100, 4) if recovery_rets else 0,
        "recovery_vol": round(float(np.std(recovery_rets)) * np.sqrt(252) * 100, 2) if len(recovery_rets) > 5 else 0,
        "recovery_sharpe": round(float(np.mean(recovery_rets)) / (np.std(recovery_rets) + 1e-10) * np.sqrt(252), 3) if len(recovery_rets) > 5 else 0,
    }


def verify_low_vol_detection(spy_returns: np.ndarray) -> dict:
    """Verify LOW_VOL detection is actually triggered."""
    low_vol_days = 0
    low_vol_periods = []
    in_low_vol = False
    current_start = None

    for i in range(20, len(spy_returns)):
        window = spy_returns[i - 19:i + 1]
        vol_20d = np.std(window) * np.sqrt(252)
        mom_20d = np.sum(window)
        cum_ret = np.cumprod(1 + spy_returns[:i + 1])
        running_max = np.maximum.accumulate(cum_ret)
        _ = (cum_ret[-1] / running_max[-1]) - 1

        if vol_20d < LOW_VOL_VOL_THRESHOLD and mom_20d > LOW_VOL_MOM_THRESHOLD:
            low_vol_days += 1
            if not in_low_vol:
                in_low_vol = True
                current_start = i
        else:
            if in_low_vol and current_start is not None:
                low_vol_periods.append({"start": current_start, "end": i - 1,
                                        "duration": i - current_start})
                in_low_vol = False
                current_start = None

    if in_low_vol and current_start is not None:
        low_vol_periods.append({"start": current_start, "end": len(spy_returns) - 1,
                                "duration": len(spy_returns) - current_start})

    return {
        "low_vol_days": low_vol_days,
        "pct_of_time": round(low_vol_days / (len(spy_returns) - 20) * 100, 1) if len(spy_returns) > 20 else 0,
        "n_periods": len(low_vol_periods),
        "avg_duration_days": round(np.mean([p["duration"] for p in low_vol_periods]), 1) if low_vol_periods else 0,
        "max_duration_days": max([p["duration"] for p in low_vol_periods]) if low_vol_periods else 0,
    }


def main():
    logger.info("=" * 70)
    logger.info("v961 — Regime Gating Gap Analysis")
    logger.info("=" * 70)

    # 1. Load data
    logger.info("\n[1/6] Loading price data...")
    prices = load_prices()
    ret_data = prices_to_returns(prices)
    logger.info(f"  Loaded {len(ret_data)} symbols")

    spy = ret_data.get("SPY")
    tlt = ret_data.get("TLT")
    gld = ret_data.get("GLD")

    if not all([spy, tlt, gld]):
        logger.error("Missing required symbols (SPY, TLT, GLD)")
        return

    spy_ret = spy["returns"]
    tlt_ret = tlt["returns"]
    gld_ret = gld["returns"]

    # Align lengths
    min_len = min(len(spy_ret), len(tlt_ret), len(gld_ret))
    spy_ret = spy_ret[:min_len]
    tlt_ret = tlt_ret[:min_len]
    gld_ret = gld_ret[:min_len]
    logger.info(f"  {min_len} aligned trading days ({spy['dates'][0]} to {spy['dates'][-1]})")

    # 2. Compute regimes
    logger.info("\n[2/6] Classifying market regimes...")
    regime_labels = compute_regime_labels(spy_ret)
    regime_dist = regime_distribution_analysis(regime_labels)

    logger.info("  Regime distribution:")
    for regime, info in sorted(regime_dist["distribution"].items()):
        logger.info(f"    {regime:>12}: {info['count']:6d} days ({info['pct']:5.1f}%)")

    # 3. Signal analysis
    logger.info("\n[3/6] Analyzing CROSS_ASSET_RV (mean-reversion)...")
    rv_signal = simulate_cross_asset_rv(spy_ret, tlt_ret, gld_ret)
    rv_pnl = signal_pnl(rv_signal, spy_ret)
    rv_results = regime_sharpe_analysis(spy_ret, rv_pnl, regime_labels)

    logger.info("  Regime Sharpe ratios:")
    for regime, info in sorted(rv_results.items()):
        logger.info(f"    {regime:>12}: Sharpe {info['sharpe']:>6.3f}  ({info['n_days']:5d} days)")

    logger.info("\n[4/6] Analyzing UNIFIED_OVERLAY (composite)...")
    uni_signal = simulate_unified_overlay(spy_ret, tlt_ret, gld_ret)
    uni_pnl = signal_pnl(uni_signal, spy_ret)
    uni_results = regime_sharpe_analysis(spy_ret, uni_pnl, regime_labels)

    for regime, info in sorted(uni_results.items()):
        logger.info(f"    {regime:>12}: Sharpe {info['sharpe']:>6.3f}  ({info['n_days']:5d} days)")

    logger.info("\n[5/6] Analyzing CRYPTO_MOMENTUM...")
    cm_signal = simulate_crypto_momentum(spy_ret)
    cm_pnl = signal_pnl(cm_signal, spy_ret)
    cm_results = regime_sharpe_analysis(spy_ret, cm_pnl, regime_labels)

    for regime, info in sorted(cm_results.items()):
        logger.info(f"    {regime:>12}: Sharpe {info['sharpe']:>6.3f}  ({info['n_days']:5d} days)")

    # 6. RECOVERY regime + LOW_VOL analysis
    logger.info("\n[6/6] Analyzing RECOVERY regime transitions...")
    recovery_analysis = analyze_recovery_regime(spy_ret, regime_labels)
    logger.info(f"  Recovery days: {recovery_analysis['n_recovery_days']} "
                f"({recovery_analysis['pct_of_time']}% of time)")
    logger.info(f"  Recovery entries: {recovery_analysis['n_recovery_entries']}")
    logger.info(f"  Recovery Sharpe: {recovery_analysis['recovery_sharpe']:.3f}")
    logger.info(f"  Recovery mean return: {recovery_analysis['recovery_mean_return']:.4f}%/day")
    logger.info(f"  Recovery vol: {recovery_analysis['recovery_vol']:.2f}% ann.")

    logger.info("\n  Transition matrix (regime → regime):")
    for src, dsts in sorted(recovery_analysis["transition_matrix"].items()):
        for dst, count in sorted(dsts.items()):
            logger.info(f"    {src:>12} → {dst:<12}: {count:4d} transitions")

    logger.info("\n  LOW_VOL detection verification:")
    low_vol_check = verify_low_vol_detection(spy_ret)
    logger.info(f"    LOW_VOL days: {low_vol_check['low_vol_days']} "
                f"({low_vol_check['pct_of_time']}% of time)")
    logger.info(f"    Periods detected: {low_vol_check['n_periods']}")
    logger.info(f"    Avg duration: {low_vol_check['avg_duration_days']} days")
    logger.info(f"    Max duration: {low_vol_check['max_duration_days']} days")

    # --- Gate Rule Recommendations ---
    logger.info("\n" + "=" * 70)
    logger.info("GATE RULE RECOMMENDATIONS")
    logger.info("=" * 70)

    def recommend_gates(results: dict, signal_name: str) -> list:
        """Determine if gates are needed based on regime Sharpe analysis."""
        recommendations = []
        _ = results.get("overall", {}).get("sharpe", 0)
        for regime, info in results.items():
            if regime == "overall":
                continue
            sharpe = info["sharpe"]
            n_days = info["n_days"]
            if sharpe < -0.05 and n_days >= 20:
                recommendations.append((regime, sharpe, n_days))
        return recommendations

    rv_gates = recommend_gates(rv_results, "CROSS_ASSET_RV")
    uni_gates = recommend_gates(uni_results, "UNIFIED_OVERLAY")
    cm_gates = recommend_gates(cm_results, "CRYPTO_MOMENTUM")

    all_recommendations = [
        ("CROSS_ASSET_RV", rv_results, rv_gates),
        ("UNIFIED_OVERLAY", uni_results, uni_gates),
        ("CRYPTO_MOMENTUM", cm_results, cm_gates),
    ]

    for sig_name, sig_results, gates in all_recommendations:
        overall = sig_results.get("overall", {}).get("sharpe", 0)
        logger.info(f"\n  {sig_name} (overall Sharpe: {overall:.3f}):")
        if gates:
            logger.info("    ⚠  Recommend gating OFF in:")
            for regime, sharpe, n_days in gates:
                logger.info(f"       - {regime:<12} (Sharpe {sharpe:.3f}, {n_days} days obs)")
        else:
            logger.info("    ✅ No gate rules needed — positive or neutral in all regimes")

    # Check special case: RECOVERY regime performance for all signals
    logger.info("\n  RECOVERY regime check for all signals:")
    for sig_name, sig_results, _ in all_recommendations:
        recovery_sharpe = sig_results.get("recovery", {}).get("sharpe", 0)
        n_days = sig_results.get("recovery", {}).get("n_days", 0)
        if recovery_sharpe < -0.05 and n_days >= 10:
            logger.info(f"    ⚠  {sig_name}: Sharpe {recovery_sharpe:.3f} in RECOVERY "
                        f"— consider gating")
        else:
            logger.info(f"    ✅ {sig_name}: Sharpe {recovery_sharpe:.3f} in RECOVERY")

    # Summary report
    logger.info("\n" + "=" * 70)
    logger.info("FINDINGS SUMMARY")
    logger.info("=" * 70)

    # SPY regime-by-regime performance (as baseline)
    logger.info("\n  SPY (equity baseline) by regime:")
    spy_pnl = signal_pnl(np.ones_like(spy_ret), spy_ret)  # 100% long SPY
    spy_results = regime_sharpe_analysis(spy_ret, spy_pnl, regime_labels)
    for regime, info in sorted(spy_results.items()):
        logger.info(f"    {regime:>12}: Sharpe {info['sharpe']:>6.3f}  ({info['n_days']:5d} days)")

    # Save results
    result = {
        "regime_distribution": regime_dist,
        "signals": {
            "CROSS_ASSET_RV": rv_results,
            "UNIFIED_OVERLAY": uni_results,
            "CRYPTO_MOMENTUM": cm_results,
            "SPY_BASELINE": spy_results,
        },
        "recovery_analysis": recovery_analysis,
        "low_vol_verification": low_vol_check,
        "gate_recommendations": {
            "CROSS_ASSET_RV": [{"regime": r, "sharpe": s, "n_days": n}
                               for r, s, n in rv_gates],
            "UNIFIED_OVERLAY": [{"regime": r, "sharpe": s, "n_days": n}
                                for r, s, n in uni_gates],
            "CRYPTO_MOMENTUM": [{"regime": r, "sharpe": s, "n_days": n}
                                for r, s, n in cm_gates],
        },
    }

    output_path = SRC_DIR / "data" / "regime_gating_analysis.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"\n  Results saved to {output_path}")

    logger.info("\n✅ Analysis complete.")
    return result


if __name__ == "__main__":
    main()
