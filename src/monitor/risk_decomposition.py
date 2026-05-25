"""
Portfolio-Lab v6.03: Risk Factor Decomposition (Barra-Style)

Maps each portfolio asset to risk factor betas via rolling OLS regression
and reports % of total portfolio variance from each factor.

Risk Factors:
- Equity Beta: SPY returns → equity market factor
- Duration: TLT returns → interest rate/bond factor
- Gold Beta: GLD returns → commodity/precious metal factor
- Crypto Beta: equal-weight BTC+ETH returns → digital asset factor
- FX Beta: EFA returns → non-USD currency factor

Usage:
    from src.monitor.risk_decomposition import RiskDecomposer, decompose_portfolio

    decomposer = RiskDecomposer(window=60)
    result = decomposer.decompose(weights={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16})
    print(result.to_dict())

CLI:
    python -m src.monitor.risk_decomposition decompose --weights 46/38/16
    python -m src.monitor.risk_decomposition factors
    python -m src.monitor.risk_decomposition check --asset SPY
"""

import json
import sys
import logging
from src.paths import BASE_ALLOCATION as DEFAULT_WEIGHTS
from src.utils import safe_get
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# project root, logging
# ---------------------------------------------------------------------------


__all__ = ['FactorBeta', 'AssetRiskDecomposition', 'PortfolioRiskDecomposition', 'RiskDecomposer', 'decompose_portfolio']

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# factor definitions
# ---------------------------------------------------------------------------

# Factor definitions: each factor has a display name and asset weights for
# constructing the factor return series from portfolio or pipeline assets.
FACTOR_DEFINITIONS: Dict[str, Dict] = {
    "equity": {
        "name": "Equity Beta",
        "symbols": {"SPY": 1.0},
        "description": "Equity market risk (S&P 500)",
        "color": "#2196F3",
    },
    "duration": {
        "name": "Duration",
        "symbols": {"TLT": 1.0},
        "description": "Interest rate / bond duration risk",
        "color": "#FF9800",
    },
    "gold": {
        "name": "Gold Beta",
        "symbols": {"GLD": 1.0},
        "description": "Commodity / precious metal factor",
        "color": "#FFC107",
    },
    "crypto": {
        "name": "Crypto Beta",
        "symbols": {"BTC-USD": 0.6, "ETH-USD": 0.4},
        "description": "Digital asset / cryptocurrency factor",
        "color": "#9C27B0",
    },
    "fx": {
        "name": "FX Beta",
        "symbols": {"EFA": 1.0},
        "description": "Non-USD currency / international equity factor",
        "color": "#4CAF50",
    },
}

# Default portfolio weights (champion 46/38/16)
# Default rolling window for regression
DEFAULT_WINDOW: int = 60


# ---------------------------------------------------------------------------
# data classes
# ---------------------------------------------------------------------------


@dataclass
class FactorBeta:
    """Estimated beta coefficient for a single risk factor on one asset."""

    factor_name: str
    beta: float
    t_stat: float
    p_value: float
    significant: bool  # p < 0.05


@dataclass
class AssetRiskDecomposition:
    """Risk decomposition for a single portfolio asset."""

    symbol: str
    weight: float
    r_squared: float  # How much of asset variance is explained by factors
    factor_betas: Dict[str, FactorBeta]  # factor_name -> beta estimate
    idiosyncratic_var: float  # Un-explained variance
    systematic_var: float  # Factor-explained variance
    total_var: float  # Total variance of this asset


@dataclass
class PortfolioRiskDecomposition:
    """Top-level portfolio risk decomposition."""

    timestamp: str
    portfolio_weights: Dict[str, float]
    total_portfolio_variance: float
    total_portfolio_volatility: float  # annualized (sqrt(252) * std)
    factor_contributions: Dict[str, float]  # factor -> % of total variance
    systematic_pct: float  # % of risk from factors
    idiosyncratic_pct: float  # % of risk from asset-specific noise
    asset_decompositions: Dict[str, AssetRiskDecomposition]
    window: int  # rolling window used
    num_observations: int  # actual data points used
    factor_correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None

    def to_dict(self) -> Dict:
        """Serialize to JSON-compatible dict."""
        result = asdict(self)
        # asdict() already converts nested dataclasses, but FactorBeta objects
        # may have been stored as dicts during re-serialization. Ensure all
        # factor_betas entries are plain dicts.
        for sym, ad in result["asset_decompositions"].items():
            betas = ad["factor_betas"]
            if betas and isinstance(next(iter(betas.values())), dict):
                pass  # Already dicts
            else:
                ad["factor_betas"] = {
                    k: asdict(v) if hasattr(v, "__dataclass_fields__") else v
                    for k, v in betas.items()
                }
        return result

    def summary_string(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Risk Factor Decomposition (window={self.window}d, {self.num_observations} obs)",
            f"Timestamp: {self.timestamp}",
            f"Total Portfolio Vol: {self.total_portfolio_volatility:.2%} ann.",
            f"Systematic Risk: {self.systematic_pct:.1f}% | Idiosyncratic: {self.idiosyncratic_pct:.1f}%",
            "",
            "Factor Contributions (% of total variance):",
        ]
        # Sort by contribution descending
        sorted_factors = sorted(
            self.factor_contributions.items(), key=lambda x: x[1], reverse=True
        )
        for fname, contrib in sorted_factors:
            bar = "█" * int(contrib / 2)
            lines.append(f"  {safe_get(FACTOR_DEFINITIONS, fname, 'name', default=fname):20s} {contrib:5.1f}% {bar}")
        lines.append("")
        lines.append("Asset-Level Factor Betas:")
        for sym, ad in self.asset_decompositions.items():
            lines.append(f"  {sym:6s} (w={ad.weight:.0%})  R²={ad.r_squared:.2f}")
            for fname, beta in sorted(
                ad.factor_betas.items(), key=lambda x: abs(x[1].beta), reverse=True
            ):
                sig = "*" if beta.significant else " "
                lines.append(f"           {safe_get(FACTOR_DEFINITIONS, fname, 'name', default=fname):20s} β={beta.beta:+.3f}{sig}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_prices_from_pipeline() -> Dict[str, np.ndarray]:
    """Load daily close prices from the existing data pipeline (TTL-cached).

    Returns:
        dict mapping symbol -> 1D numpy array of daily close prices (oldest first)
    """
    from src.data.price_cache import get_prices_df

    try:
        df = get_prices_df()
    except FileNotFoundError:
        logger.warning("Prices file not found")
        return {}

    result: Dict[str, np.ndarray] = {}
    for symbol in df.columns:
        result[symbol] = df[symbol].dropna().values

    return result


def _compute_returns(prices: np.ndarray) -> np.ndarray:
    """Compute daily log returns from price series."""
    if len(prices) < 2:
        return np.array([])
    return np.diff(np.log(prices))


def _ols_beta(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Simple OLS regression: y = beta * x (no intercept for factor model).

    Returns:
        (beta, t_stat, p_value)
    """
    n = len(x)
    if n < 3:
        return 0.0, 0.0, 1.0

    # Ensure 1D
    x = x.flatten()
    y = y.flatten()

    # Check for zero variance in x (no explanatory power)
    if np.std(x) < 1e-12:
        return 0.0, 0.0, 1.0

    # beta = (X'X)^(-1) X'y
    beta = np.dot(x, y) / np.dot(x, x) if np.dot(x, x) > 1e-12 else 0.0

    # Residuals
    residuals = y - beta * x

    # Check for perfect fit (all residuals near zero)
    mse = np.dot(residuals, residuals) / (n - 1)
    if mse < 1e-15:
        # Perfect fit — infinite t-stat
        return beta, 999.0, 0.0

    se = np.sqrt(mse / np.dot(x, x)) if np.dot(x, x) > 1e-12 else 0.0

    t_stat = beta / se if se > 1e-12 else 0.0

    # Two-sided p-value from t-distribution (approximation using normal for n>30)
    from scipy.stats import t as t_dist

    p_value = 2.0 * t_dist.sf(abs(t_stat), df=n - 1)

    return beta, t_stat, p_value


def _build_factor_returns(
    prices: Dict[str, np.ndarray],
    factor_defs: Dict[str, Dict] = None,
) -> Dict[str, np.ndarray]:
    """Build factor return series from price data.

    Args:
        prices: dict symbol -> price array (oldest first)
        factor_defs: factor definitions dict

    Returns:
        dict factor_key -> daily return array
    """
    if factor_defs is None:
        factor_defs = FACTOR_DEFINITIONS

    factor_returns: Dict[str, np.ndarray] = {}

    for fkey, fdef in factor_defs.items():
        components = []
        min_len = None

        for sym, weight in fdef["symbols"].items():
            if sym not in prices:
                logger.warning("Factor %s: missing price data for %s", fkey, sym)
                continue
            rets = _compute_returns(prices[sym])
            if len(rets) == 0:
                continue
            components.append(weight * rets)
            if min_len is None or len(rets) < min_len:
                min_len = len(rets)

        if not components or min_len is None:
            logger.warning("Factor %s: no data available", fkey)
            factor_returns[fkey] = np.array([])
            continue

        # Truncate all to same length
        truncated = [c[-min_len:] for c in components]
        factor_returns[fkey] = np.sum(truncated, axis=0)

    return factor_returns


def _align_series(
    asset_rets: np.ndarray,
    factor_rets: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Align asset and factor return series to common timeline.

    Args:
        asset_rets: asset daily returns (oldest first)
        factor_rets: dict of factor returns (oldest first)

    Returns:
        (asset_returns_aligned, factor_matrix) where factor_matrix is
        (n_obs, n_factors)
    """
    # Find min length across all series with data
    lengths = [len(asset_rets)] + [
        len(frets) for frets in factor_rets.values() if len(frets) > 0
    ]

    if len(lengths) < 2:
        # No factor data available
        return np.array([]), np.empty((0, len(factor_rets)))

    min_len = min(lengths)

    if min_len < 2:
        return np.array([]), np.empty((0, len(factor_rets)))

    # Truncate to common length (keep most recent)
    asset_aligned = asset_rets[-min_len:]
    factor_matrix = np.column_stack([
        frets[-min_len:] if len(frets) >= min_len else np.zeros(min_len)
        for frets in factor_rets.values()
    ])
    return asset_aligned, factor_matrix


# ---------------------------------------------------------------------------
# main decomposer
# ---------------------------------------------------------------------------


class RiskDecomposer:
    """Decompose portfolio risk into factor contributions using OLS regression.

    Uses rolling-window OLS to estimate each asset's beta to 5 risk factors,
    then computes portfolio-level variance decomposition.
    """

    def __init__(
        self,
        window: int = DEFAULT_WINDOW,
        factor_definitions: Optional[Dict[str, Dict]] = None,
        prices_data: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Args:
            window: Rolling regression window in trading days (default: 60)
            factor_definitions: Custom factor definitions (default: FACTOR_DEFINITIONS)
            prices_data: Pre-loaded price data (auto-loads if None)
        """
        self.window = window
        self.factor_defs = factor_definitions or FACTOR_DEFINITIONS
        self.factor_keys = list(self.factor_defs)
        self.n_factors = len(self.factor_keys)

        # Load price data
        if prices_data is None:
            self.prices = _load_prices_from_pipeline()
        else:
            self.prices = prices_data

        # Pre-compute factor returns from available data
        self.factor_returns: Dict[str, np.ndarray] = _build_factor_returns(
            self.prices, self.factor_defs
        )

        logger.info(
            "RiskDecomposer initialized: %d factors, "
            "window=%dd, %d symbols available",
            self.n_factors, window, len(self.prices),
        )

    def estimate_asset_betas(
        self, symbol: str, returns: Optional[np.ndarray] = None
    ) -> Dict[str, FactorBeta]:
        """Estimate factor betas for a single asset using rolling-window OLS.

        Uses the most recent `window` trading days of data.

        Args:
            symbol: Asset symbol
            returns: Pre-computed asset returns (auto-computes if None)

        Returns:
            dict factor_key -> FactorBeta
        """
        # Get asset returns
        if returns is None:
            if symbol not in self.prices:
                logger.warning("No price data for %s", symbol)
                return {}
            returns = _compute_returns(self.prices[symbol])

        if len(returns) < self.window:
            logger.warning(
                "Not enough data for %s: %d obs < window=%d",
                symbol, len(returns), self.window,
            )
            # Use all available data
            window = len(returns)
        else:
            window = self.window

        # Use most recent window observations
        asset_window = returns[-window:]

        # Build aligned factor matrix for the same window
        aligned_asset, factor_matrix = _align_series(
            asset_window, self.factor_returns
        )

        if len(aligned_asset) < 3:
            logger.warning("Insufficient aligned data for %s", symbol)
            return {}

        len(aligned_asset)
        betas: Dict[str, FactorBeta] = {}

        for i, fkey in enumerate(self.factor_keys):
            x = factor_matrix[:, i]
            beta_val, t_stat, p_val = _ols_beta(x, aligned_asset)
            betas[fkey] = FactorBeta(
                factor_name=self.factor_defs[fkey]["name"],
                beta=beta_val,
                t_stat=t_stat,
                p_value=p_val,
                significant=p_val < 0.05,
            )

        return betas

    def decompose_asset(
        self, symbol: str, weight: float
    ) -> Optional[AssetRiskDecomposition]:
        """Decompose a single asset's risk into factor contributions.

        Args:
            symbol: Asset symbol
            weight: Portfolio weight (0-1)

        Returns:
            AssetRiskDecomposition or None if insufficient data
        """
        if symbol not in self.prices:
            logger.warning("No price data for %s, skipping", symbol)
            return None

        returns = _compute_returns(self.prices[symbol])
        if len(returns) < 3:
            return None

        # Get betas
        betas = self.estimate_asset_betas(symbol, returns)
        if not betas:
            return None

        # Use most recent window
        window = min(self.window, len(returns))
        asset_window = returns[-window:]
        aligned_asset, factor_matrix = _align_series(asset_window, self.factor_returns)
        n_obs = len(aligned_asset)

        if n_obs < 3:
            return None

        # Explained returns (systematic)
        explained = np.zeros(n_obs)
        for i, fkey in enumerate(self.factor_keys):
            if fkey in betas:
                explained += betas[fkey].beta * factor_matrix[:, i]

        residuals = aligned_asset - explained

        # Variance components (annualized approximation)
        # Use traditional mean-centered variance for R² (most interpretable)
        total_var = np.var(aligned_asset, ddof=1)
        residual_var = np.var(residuals, ddof=1)
        r_squared = 1.0 - (residual_var / total_var) if total_var > 1e-12 else 0.0
        r_squared = min(max(r_squared, 0.0), 1.0)  # Clip to [0, 1]
        systematic_var = max(total_var - residual_var, 0.0)

        return AssetRiskDecomposition(
            symbol=symbol,
            weight=weight,
            r_squared=r_squared,
            factor_betas=betas,
            idiosyncratic_var=residual_var,
            systematic_var=systematic_var,
            total_var=total_var,
        )

    def decompose(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> PortfolioRiskDecomposition:
        """Decompose full portfolio risk into factor contributions.

        Args:
            weights: Dict mapping symbol -> weight. Sums to 1.
                     Defaults to champion 46/38/16.

        Returns:
            PortfolioRiskDecomposition with full breakdown
        """
        if weights is None:
            weights = DEFAULT_WEIGHTS

        # Normalize weights
        total_w = sum(weights.values())
        if total_w == 0:
            raise ValueError("Portfolio weights must sum to > 0")
        norm_weights = {k: v / total_w for k, v in weights.items()}

        # Decompose each asset
        asset_decomps: Dict[str, AssetRiskDecomposition] = {}
        for sym, w in norm_weights.items():
            ad = self.decompose_asset(sym, w)
            if ad is not None:
                asset_decomps[sym] = ad

        if not asset_decomps:
            raise ValueError("No assets could be decomposed (check price data)")

        # --- Portfolio-level variance decomposition ---
        # Identify factors with data (skip empty factors like crypto without BTC/ETH)
        active_factors = [
            fkey for fkey in self.factor_keys
            if len(self.factor_returns.get(fkey, [])) > 0
        ]

        if len(active_factors) < 2:
            raise ValueError("Insufficient factor return data (need at least 2 factors with data)")

        # Build aligned factor matrix using only active factors
        min_factor_len = min(
            len(self.factor_returns[fkey]) for fkey in active_factors
        )

        factor_matrix = np.column_stack([
            self.factor_returns[fkey][-min_factor_len:]
            for fkey in active_factors
        ])

        # Factor covariance matrix (n_factors x n_factors)
        factor_cov = np.cov(factor_matrix, rowvar=False)

        # Build weight x beta matrix (only for active factors)
        n_active = len(active_factors)
        portfolio_factor_beta = np.zeros(n_active)
        for sym, ad in asset_decomps.items():
            w = norm_weights[sym]
            for i, fkey in enumerate(active_factors):
                if fkey in ad.factor_betas:
                    portfolio_factor_beta[i] += w * ad.factor_betas[fkey].beta

        # Systematic variance: β^T Σ_factors β
        systematic_var = portfolio_factor_beta @ factor_cov @ portfolio_factor_beta

        # Idiosyncratic variance: sum(w_i^2 * σ²_idio_i)
        # (assuming no cross-asset correlated residuals)
        idio_var = sum(
            (norm_weights[ad.symbol] ** 2) * ad.idiosyncratic_var
            for ad in asset_decomps.values()
        )

        total_portfolio_var = systematic_var + idio_var

        # Factor contributions: how much of systematic variance comes from each factor
        # Use factor marginal contribution: β_i * (Σ_factors β)_i
        factor_betas_weighted = factor_cov @ portfolio_factor_beta
        factor_contribs_raw = portfolio_factor_beta * factor_betas_weighted

        # Normalize to percentages - fill ALL factors (active ones get their share, inactive get 0)
        total_sys = max(sum(factor_contribs_raw), 1e-12)
        factor_contributions: Dict[str, float] = {}
        for i, fkey in enumerate(active_factors):
            factor_contributions[fkey] = (
                (factor_contribs_raw[i] / total_sys) * systematic_var / max(total_portfolio_var, 1e-12) * 100
            )

        # Fill in missing/inactive factors with 0
        for fkey in self.factor_keys:
            if fkey not in factor_contributions:
                factor_contributions[fkey] = 0.0

        systematic_pct = (
            systematic_var / max(total_portfolio_var, 1e-12) * 100
        )
        idiosyncratic_pct = (
            idio_var / max(total_portfolio_var, 1e-12) * 100
        )

        # Annualized volatility
        annualized_vol = np.sqrt(total_portfolio_var * 252)

        # Factor correlation matrix (only active factors)
        try:
            corr_matrix = np.corrcoef(factor_matrix, rowvar=False)
            factor_corr: Dict[str, Dict[str, float]] = {}
            for i, fkey1 in enumerate(active_factors):
                fname1 = self.factor_defs[fkey1]["name"]
                factor_corr[fname1] = {}
                for j, fkey2 in enumerate(active_factors):
                    fname2 = self.factor_defs[fkey2]["name"]
                    factor_corr[fname1][fname2] = round(float(corr_matrix[i, j]), 4)
        except (ValueError, ZeroDivisionError):
            logger.exception("Failed to compute factor correlation matrix")
            factor_corr = None

        return PortfolioRiskDecomposition(
            timestamp=datetime.now().isoformat(),
            portfolio_weights=norm_weights,
            total_portfolio_variance=total_portfolio_var,
            total_portfolio_volatility=annualized_vol,
            factor_contributions=factor_contributions,
            systematic_pct=systematic_pct,
            idiosyncratic_pct=idiosyncratic_pct,
            asset_decompositions=asset_decomps,
            window=self.window,
            num_observations=min_factor_len,
            factor_correlation_matrix=factor_corr,
        )

    def check_asset_factor_exposure(self, symbol: str) -> str:
        """Get a human-readable factor exposure report for a single asset.

        Args:
            symbol: Asset symbol (e.g., 'SPY', 'GLD', 'QQQ')

        Returns:
            Formatted string report
        """
        betas = self.estimate_asset_betas(symbol)
        if not betas:
            return f"No data available for {symbol}"

        lines = [
            f"Factor Exposure Report: {symbol}",
            f"Window: {self.window} trading days",
            "",
            "Factor         Beta     t-stat   p-value  Significant",
            "-" * 55,
        ]
        for fkey in self.factor_keys:
            if fkey in betas:
                b = betas[fkey]
                sig = " *" if b.significant else "  "
                lines.append(
                    f"{self.factor_defs[fkey]['name']:15s} {b.beta:+.4f}  {b.t_stat:+7.2f}  {b.p_value:.4f}{sig}"
                )
        return "\n".join(lines)

    def get_factor_correlations(self) -> Dict[str, Dict[str, float]]:
        """Compute correlation matrix of available risk factors."""
        active_factors = [
            fkey for fkey in self.factor_keys
            if len(self.factor_returns.get(fkey, [])) > 0
        ]
        if len(active_factors) < 2:
            return {}

        min_len = min(len(self.factor_returns[fkey]) for fkey in active_factors)
        if min_len < 2:
            return {}

        factor_matrix = np.column_stack([
            self.factor_returns[fkey][-min_len:]
            for fkey in active_factors
        ])
        corr = np.corrcoef(factor_matrix, rowvar=False)

        result: Dict[str, Dict[str, float]] = {}
        for i, fkey1 in enumerate(active_factors):
            result[self.factor_defs[fkey1]["name"]] = {}
            for j, fkey2 in enumerate(active_factors):
                result[self.factor_defs[fkey1]["name"]][self.factor_defs[fkey2]["name"]] = round(
                    float(corr[i, j]), 4
                )
        return result


# ---------------------------------------------------------------------------
# convenience function
# ---------------------------------------------------------------------------


def decompose_portfolio(
    weights: Optional[Dict[str, float]] = None,
    window: int = DEFAULT_WINDOW,
) -> PortfolioRiskDecomposition:
    """One-shot portfolio risk decomposition.

    Args:
        weights: Portfolio weights dict (default: 46/38/16)
        window: Rolling regression window (default: 60)

    Returns:
        PortfolioRiskDecomposition
    """
    decomposer = RiskDecomposer(window=window)
    return decomposer.decompose(weights=weights)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="v6.03 Risk Factor Decomposition (Barra-Style)"
    )
    sub = parser.add_subparsers(dest="command", help="Sub-command")

    # decompose
    p_decomp = sub.add_parser("decompose", help="Decompose portfolio risk")
    p_decomp.add_argument(
        "--weights",
        default="46/38/16",
        help="Portfolio weights as slash-separated values in SPY/GLD/TLT order (default: 46/38/16)",
    )
    p_decomp.add_argument(
        "--window", type=int, default=DEFAULT_WINDOW, help="Rolling window (default: 60)"
    )
    p_decomp.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )

    # check
    p_check = sub.add_parser("check", help="Check factor exposure for an asset")
    p_check.add_argument("symbol", help="Asset symbol (e.g., SPY, QQQ, IEF)")
    p_check.add_argument("--window", type=int, default=DEFAULT_WINDOW)

    # factors
    p_factors = sub.add_parser("factors", help="Show factor correlation matrix")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    try:
        if args.command == "decompose":
            parts = [float(x) for x in args.weights.split("/")]
            symbols = ["SPY", "GLD", "TLT"]
            if len(parts) != len(symbols):
                # Try to infer from length
                if len(parts) == 2:
                    symbols = ["SPY", "GLD"]
                elif len(parts) == 4:
                    symbols = ["SPY", "GLD", "TLT", "IEF"]
                elif len(parts) == 5:
                    symbols = ["SPY", "GLD", "TLT", "IEF", "SHY"]
                else:
                    symbols = [f"ASSET{i+1}" for i in range(len(parts))]

            weights = dict(zip(symbols, [p / 100.0 for p in parts]))
            result = decompose_portfolio(weights=weights, window=args.window)

            if args.json:
                print(json.dumps(result.to_dict(), indent=2, default=str))
            else:
                print(result.summary_string())

        elif args.command == "check":
            decomposer = RiskDecomposer(window=args.window)
            print(decomposer.check_asset_factor_exposure(args.symbol.upper()))

        elif args.command == "factors":
            decomposer = RiskDecomposer()
            corr = decomposer.get_factor_correlations()
            print("Factor Correlation Matrix:\n")
            factors = list(corr)
            # Header
            print(f"{'':20s}", end="")
            for f in factors:
                print(f"{f:15s}", end="")
            print()
            for f1 in factors:
                print(f"{f1:20s}", end="")
                for f2 in factors:
                    print(f"{corr[f1][f2]:+8.4f}    ", end="")
                print()

        else:
            parser.print_help()

    except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as e:
        logger.error("Error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
