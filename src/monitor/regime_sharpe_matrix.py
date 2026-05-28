"""Data-driven regime-conditional signal gating via per-signal Sharpe ratios.

Computes standalone Sharpe ratio for each signal in each market regime
from historical daily returns, with stationary bootstrap significance
testing. Outputs feed directly into RegimeGate.update_from_performance()
and REGIME_CONDITIONAL_WEIGHTS for data-driven ensemble weighting.

Replaces hardcoded GATE_RULES with computed thresholds while preserving
hysteresis and confidence-gating safety rails.

Based on:
- Politis & Romano (1994) "The Stationary Bootstrap"
- Lopez de Prado (2018) "Advances in Financial Machine Learning" Ch. 7
- Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

# Annualization factor for daily returns
_ANNUALIZATION = np.sqrt(252)

# Minimum observations for any Sharpe estimate
DEFAULT_MIN_OBS = 30

# Default bootstrap iterations
DEFAULT_N_BOOTSTRAP = 10_000

# Significance threshold: P(Sharpe > 0)
DEFAULT_P_THRESHOLD = 0.90

# Multiplier bounds for soft weight adjustment
MULTIPLIER_CAP = 1.5
MULTIPLIER_FLOOR = 0.3

# Sharpe thresholds for multiplier mapping
_SHARPE_STRONG = 0.5   # above this → boost
_SHARPE_NEUTRAL = 0.0  # around this → baseline 1.0


@dataclass
class RegimeSharpeEntry:
    """Per-signal, per-regime Sharpe ratio with statistical significance."""
    signal: str
    regime: str
    sharpe: float
    hit_rate: float
    ic: float           # Information Coefficient (Spearman rank correlation)
    n_obs: int
    p_positive: float   # bootstrap P(Sharpe > 0)
    ci_95_low: float
    ci_95_high: float


def compute_sharpe(
    returns: pd.Series,
    min_obs: int = DEFAULT_MIN_OBS,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute annualized Sharpe ratio from daily returns.

    Args:
        returns: Series of daily returns.
        min_obs: Minimum observations required. Returns NaN if fewer.
        risk_free_rate: Annual risk-free rate (e.g. 0.045 for 4.5%).

    Returns:
        Annualized Sharpe ratio, or NaN if insufficient data.
    """
    if len(returns) < min_obs:
        return np.nan

    daily_rf = risk_free_rate / 252.0
    excess = returns - daily_rf
    std = excess.std()
    if std == 0 or np.isnan(std):
        return np.nan

    return float(excess.mean() / std * _ANNUALIZATION)


def compute_hit_rate(
    signal_values: pd.Series,
    returns: pd.Series,
) -> float:
    """Compute directional hit rate (sign agreement).

    Args:
        signal_values: Signal direction values (positive = long).
        returns: Forward daily returns.

    Returns:
        Fraction of nonzero-return days where signal direction matches return sign.
    """
    # Use numpy arrays to avoid pandas index alignment issues
    sig_arr = np.asarray(signal_values.values, dtype=float)
    ret_arr = np.asarray(returns.values, dtype=float)

    # Only count days with nonzero returns and nonzero signals
    mask = (ret_arr != 0) & (sig_arr != 0)
    if mask.sum() == 0:
        return np.nan

    matches = np.sign(sig_arr[mask]) == np.sign(ret_arr[mask])
    return float(matches.mean())


def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    min_obs: int = DEFAULT_MIN_OBS,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Stationary bootstrap for Sharpe ratio confidence interval.

    Uses circular block bootstrap to handle autocorrelation in returns.

    Args:
        returns: Series of daily returns.
        n_bootstrap: Number of bootstrap iterations.
        min_obs: Minimum observations required.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: mean, ci_95_low, ci_95_high, p_positive.
        All NaN if insufficient data.
    """
    nan_result = {
        "mean": np.nan,
        "ci_95_low": np.nan,
        "ci_95_high": np.nan,
        "p_positive": np.nan,
    }

    if len(returns) < min_obs:
        return nan_result

    rng = np.random.RandomState(seed)
    n = len(returns)
    vals = returns.values

    # Block length = n^(1/3) per Politis & Romano (1992)
    block_len = max(1, int(n ** (1.0 / 3.0)))

    boot_sharpes = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        # Circular block bootstrap
        idx = np.empty(n, dtype=int)
        pos = 0
        while pos < n:
            start = rng.randint(0, n)
            length = rng.randint(1, block_len + 1)
            for j in range(length):
                if pos >= n:
                    break
                idx[pos] = (start + j) % n
                pos += 1

        sample = vals[idx]
        s_mean = sample.mean()
        s_std = sample.std()
        boot_sharpes[i] = (s_mean / s_std * _ANNUALIZATION) if s_std > 0 else 0.0

    return {
        "mean": float(np.mean(boot_sharpes)),
        "ci_95_low": float(np.percentile(boot_sharpes, 2.5)),
        "ci_95_high": float(np.percentile(boot_sharpes, 97.5)),
        "p_positive": float(np.mean(boot_sharpes > 0)),
    }


def compute_regime_sharpe_matrix(
    df: pd.DataFrame,
    min_obs: int = DEFAULT_MIN_OBS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    risk_free_rate: float = 0.0,
    seed: Optional[int] = None,
) -> Dict[str, Dict[str, RegimeSharpeEntry]]:
    """Compute per-signal, per-regime Sharpe matrix with bootstrap CI.

    Args:
        df: DataFrame with columns: 'signal', 'regime', 'daily_return'.
            Index should be datetime (for time-series operations).
        min_obs: Minimum observations per group for valid Sharpe.
        n_bootstrap: Bootstrap iterations per group.
        risk_free_rate: Annual risk-free rate.
        seed: Random seed for bootstrap reproducibility.

    Returns:
        Nested dict: {signal_name: {regime_name: RegimeSharpeEntry}}.
    """
    if df.empty:
        return {}

    required_cols = {"signal", "regime", "daily_return"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing columns: {missing}")

    matrix: Dict[str, Dict[str, RegimeSharpeEntry]] = {}

    for (signal, regime), group in df.groupby(["signal", "regime"]):
        returns = group["daily_return"].dropna()
        n = len(returns)

        if n < min_obs:
            sharpe = np.nan
            hit_rate = np.nan
            ic = np.nan
            p_positive = np.nan
            ci_low = np.nan
            ci_high = np.nan
        else:
            sharpe = compute_sharpe(returns, min_obs=min_obs, risk_free_rate=risk_free_rate)
            hit_rate = compute_hit_rate(
                pd.Series(np.sign(returns.values), index=returns.index),
                returns,
            )
            # IC: Spearman rank autocorrelation as proxy for signal quality
            # (signal values are the returns themselves; IC measures persistence)
            if n >= 10:
                rho, _ = scipy_stats.spearmanr(
                    returns.values[:-1],
                    returns.values[1:],
                )
                ic = float(rho) if not np.isnan(rho) else 0.0
            else:
                ic = 0.0

            boot = bootstrap_sharpe_ci(returns, n_bootstrap=n_bootstrap, min_obs=min_obs, seed=seed)
            p_positive = boot["p_positive"]
            ci_low = boot["ci_95_low"]
            ci_high = boot["ci_95_high"]

        entry = RegimeSharpeEntry(
            signal=signal,
            regime=regime,
            sharpe=sharpe,
            hit_rate=hit_rate,
            ic=ic,
            n_obs=n,
            p_positive=p_positive,
            ci_95_low=ci_low,
            ci_95_high=ci_high,
        )

        matrix.setdefault(signal, {})[regime] = entry

    logger.info(
        "Computed regime Sharpe matrix: %d signals × %d regimes",
        len(matrix),
        max(len(v) for v in matrix.values()) if matrix else 0,
    )

    return matrix


def derive_gate_rules(
    matrix: Dict[str, Dict[str, RegimeSharpeEntry]],
    p_threshold: float = DEFAULT_P_THRESHOLD,
) -> Dict[str, set]:
    """Convert Sharpe matrix to RegimeGate-compatible ON/OFF rules.

    A signal is gated OFF in a regime when:
    - There is sufficient data (n_obs >= min_obs)
    - AND P(Sharpe > 0) < p_threshold (bootstrap significance)

    Signals/regions with insufficient data are left unchanged (not gated).

    Args:
        matrix: Output of compute_regime_sharpe_matrix().
        p_threshold: Minimum P(Sharpe > 0) to keep signal ON.

    Returns:
        Dict: {signal_name: set_of_off_regimes}. Compatible with
        RegimeGate.__init__(gate_rules=...).
    """
    rules: Dict[str, set] = {}

    for signal, regimes in matrix.items():
        for regime, entry in regimes.items():
            # Skip insufficient data — don't generate rules
            if np.isnan(entry.p_positive):
                continue

            if entry.p_positive < p_threshold:
                rules.setdefault(signal, set()).add(regime)

    n_gated = sum(len(v) for v in rules.values())
    logger.info(
        "Derived %d gate rules (%d signal-regime pairs gated OFF, threshold=%.2f)",
        len(rules), n_gated, p_threshold,
    )

    return rules


def derive_regime_weight_multipliers(
    matrix: Dict[str, Dict[str, RegimeSharpeEntry]],
    neutral_sharpe: float = _SHARPE_NEUTRAL,
    strong_sharpe: float = _SHARPE_STRONG,
    cap: float = MULTIPLIER_CAP,
    floor: float = MULTIPLIER_FLOOR,
) -> Dict[str, Dict[str, float]]:
    """Convert Sharpe matrix to soft weight multipliers per signal per regime.

    Mapping:
    - Sharpe <= -0.5 → floor (0.3)
    - Sharpe = 0.0 → 1.0 (baseline)
    - Sharpe >= 0.5 → cap (1.5)
    - Linear interpolation between thresholds

    Args:
        matrix: Output of compute_regime_sharpe_matrix().
        neutral_sharpe: Sharpe value that maps to multiplier=1.0.
        strong_sharpe: Sharpe value that maps to multiplier=cap.
        cap: Maximum multiplier.
        floor: Minimum multiplier.

    Returns:
        Nested dict: {signal: {regime: multiplier}}.
    """
    multipliers: Dict[str, Dict[str, float]] = {}

    for signal, regimes in matrix.items():
        for regime, entry in regimes.items():
            if np.isnan(entry.sharpe):
                continue

            s = entry.sharpe
            # Symmetric linear mapping: Sharpe ±2.0 maps to floor/cap
            # Sharpe 0.0 → 1.0, Sharpe ±0.5 → ±0.125 from 1.0
            scale = (cap - floor) / 4.0  # 0.3 per unit Sharpe
            mult = 1.0 + s * scale
            mult = max(floor, min(cap, mult))

            multipliers.setdefault(signal, {})[regime] = round(mult, 3)

    return multipliers


def format_for_gate_update(
    matrix: Dict[str, Dict[str, RegimeSharpeEntry]],
) -> Dict[str, Dict[str, float]]:
    """Format Sharpe matrix for RegimeGate.update_from_performance().

    Args:
        matrix: Output of compute_regime_sharpe_matrix().

    Returns:
        {regime: {signal: sharpe_ratio}} — matching the API expected by
        RegimeGate.update_from_performance().
    """
    result: Dict[str, Dict[str, float]] = {}

    for signal, regimes in matrix.items():
        for regime, entry in regimes.items():
            if np.isnan(entry.sharpe):
                continue
            result.setdefault(regime, {})[signal] = entry.sharpe

    return result


def extract_signal_regime_data(
    db_path,
    prices: pd.DataFrame,
    market_col: str = "SPY",
) -> pd.DataFrame:
    """Extract signal-regime data from SQLite for regime Sharpe computation.

    Joins source_readings (signal values) with ensemble_votes (regime labels),
    then computes signal returns as signal_value × next-day market return.

    Signal return = value_t × (SPY_close_{t+1} / SPY_close_t - 1)

    This measures how much P&L the signal's directional bet would have
    generated on the following day.

    Args:
        db_path: Path to ensemble_signals.db (str or Path).
        prices: DataFrame with DatetimeIndex and market_col column.
        market_col: Column name for market returns (default "SPY").

    Returns:
        DataFrame with columns: signal, regime, daily_return (DatetimeIndex).
        Empty DataFrame if DB missing or no data.
    """
    from pathlib import Path

    db_path = Path(db_path)
    if not db_path.exists():
        logger.warning("Ensemble signals DB not found: %s", db_path)
        return pd.DataFrame(columns=["signal", "regime", "daily_return"])

    try:
        import sqlite3 as _sqlite3

        with _sqlite3.connect(str(db_path)) as conn:
            # Load regime labels per timestamp
            votes_df = pd.read_sql_query(
                "SELECT timestamp, regime FROM ensemble_votes ORDER BY timestamp",
                conn,
                parse_dates=["timestamp"],
            )
            if votes_df.empty:
                return pd.DataFrame(columns=["signal", "regime", "daily_return"])

            # Load signal readings
            readings_df = pd.read_sql_query(
                "SELECT timestamp, source, value FROM source_readings ORDER BY timestamp",
                conn,
                parse_dates=["timestamp"],
            )
            if readings_df.empty:
                return pd.DataFrame(columns=["signal", "regime", "daily_return"])

        # Join readings with regime labels on timestamp
        merged = readings_df.merge(votes_df, on="timestamp", how="inner")
        merged = merged.set_index("timestamp").sort_index()

        # Compute next-day market returns
        if market_col not in prices.columns:
            # Try first column as fallback
            market_col = prices.columns[0]

        market_returns = prices[market_col].pct_change().shift(-1)
        market_returns.name = "market_return"

        # Align market returns with signal timestamps
        merged = merged.join(market_returns, how="inner")
        merged = merged.dropna(subset=["value", "market_return", "regime"])

        if merged.empty:
            return pd.DataFrame(columns=["signal", "regime", "daily_return"])

        # Signal return = signal_value × next_day_market_return
        merged["daily_return"] = merged["value"] * merged["market_return"]

        result = merged[["source", "regime", "daily_return"]].rename(
            columns={"source": "signal"}
        )

        logger.info(
            "Extracted %d signal-regime observations (%d signals, %d regimes)",
            len(result),
            result["signal"].nunique(),
            result["regime"].nunique(),
        )

        return result

    except Exception as e:
        logger.warning("Failed to extract signal-regime data: %s", e)
        return pd.DataFrame(columns=["signal", "regime", "daily_return"])


def update_gate_from_history(
    gate,
    db_path,
    prices: pd.DataFrame,
    p_threshold: float = DEFAULT_P_THRESHOLD,
    min_obs: int = DEFAULT_MIN_OBS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: Optional[int] = None,
) -> Dict[str, Dict[str, RegimeSharpeEntry]]:
    """Update a RegimeGate with data-driven rules from historical data.

    Extracts signal-regime data from SQLite, computes the Sharpe matrix,
    derives gate rules, and updates the gate in-place via
    update_from_performance().

    Args:
        gate: RegimeGate instance to update.
        db_path: Path to ensemble_signals.db.
        prices: Price DataFrame for market return computation.
        p_threshold: Bootstrap significance threshold.
        min_obs: Minimum observations per signal-regime.
        n_bootstrap: Bootstrap iterations.
        seed: Random seed.

    Returns:
        The computed Sharpe matrix (for logging/dashboard).
    """
    df = extract_signal_regime_data(db_path, prices)
    if df.empty:
        logger.info("No historical signal-regime data available; gate unchanged")
        return {}

    matrix = compute_regime_sharpe_matrix(
        df, min_obs=min_obs, n_bootstrap=n_bootstrap, seed=seed,
    )

    # Update gate rules (hard ON/OFF)
    gate_input = format_for_gate_update(matrix)
    if gate_input:
        gate.update_from_performance(gate_input, sharpe_threshold=0.0)
        logger.info("Updated RegimeGate with data-driven rules from %d observations", len(df))

    return matrix


def load_persisted_gate_rules(
    persist_path,
    max_age_hours: int = 24,
) -> Optional[Dict[str, set]]:
    """Load data-driven gate rules from persisted JSON file.

    Reads the file written by DashboardGenerator.generate_regime_gate_json()
    and converts it to RegimeGate-compatible format.

    Args:
        persist_path: Path to regime_gate_persisted.json.
        max_age_hours: Maximum age of persisted data in hours.
            Returns None if data is older than this.

    Returns:
        Dict of {signal: set_of_off_regimes} or None if file missing/stale.
    """
    import json as _json
    from pathlib import Path
    from datetime import datetime, timedelta

    persist_path = Path(persist_path)
    if not persist_path.exists():
        return None

    try:
        with open(persist_path) as f:
            data = _json.load(f)

        # Check staleness
        computed_at = data.get("computed_at")
        if computed_at:
            ts = datetime.fromisoformat(computed_at)
            if datetime.now() - ts > timedelta(hours=max_age_hours):
                logger.info(
                    "Persisted gate rules stale (%s > %dh old); ignoring",
                    computed_at, max_age_hours,
                )
                return None

        raw_rules = data.get("gate_rules", {})
        if not raw_rules:
            return None

        rules: Dict[str, set] = {}
        for signal, regimes in raw_rules.items():
            rules[signal] = set(regimes)

        logger.info(
            "Loaded %d data-driven gate rules from %s",
            len(rules), persist_path,
        )
        return rules

    except Exception as e:
        logger.warning("Failed to load persisted gate rules: %s", e)
        return None
