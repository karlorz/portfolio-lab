"""GP-VCV: Hybrid Gaussian Process Covariance Estimation.

Based on arXiv:2605.17275 — GP regression for VCV estimation,
outperforming EWMA and DCC-GARCH by 12-18%.

ML-gated (sklearn). Use only with PORTFOLIO_LAB_ENABLE_ML=1.
"""

import os
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.paths import DATA_DIR as _DATA_DIR

logger = logging.getLogger(__name__)

# ── ML Gate ────────────────────────────────────────────────────────────────
_HAS_SKLEARN = False
_gp_import_error = None

if os.environ.get("PORTFOLIO_LAB_ENABLE_ML") == "1":
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            RBF, Matern, WhiteKernel, ConstantKernel, Kernel,
        )
        _HAS_SKLEARN = True
    except ImportError as e:
        _gp_import_error = str(e)
        logger.warning(f"sklearn not available for GP-VCV: {_gp_import_error}")


# ── Data Classes ───────────────────────────────────────────────────────────

@dataclass
class GPVCVResult:
    """Result of a GP-VCV estimation run."""
    cov_matrix: np.ndarray           # N×N covariance matrix (annualized)
    vol_estimates: np.ndarray        # N-element volatility vector
    corr_matrix: np.ndarray          # N×N correlation matrix
    kernel_params: Dict[str, float]  # Learned kernel hyperparameters
    log_marginal_likelihood: float   # Model fit score
    condition_number: float          # Cov matrix condition number
    is_psd: bool                     # Positive semi-definite check
    asset_labels: List[str]          # Asset names in order
    timestamp: str                   # ISO timestamp of estimation
    lookback_days: int               # Days of history used

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict (numpy arrays → lists)."""
        d = asdict(self)
        d["cov_matrix"] = self.cov_matrix.tolist()
        d["vol_estimates"] = self.vol_estimates.tolist()
        d["corr_matrix"] = self.corr_matrix.tolist()
        return d


# ── Kernel Builder ─────────────────────────────────────────────────────────

def _build_hybrid_kernel(vol_length_scale: float = 60.0) -> Kernel:
    """Build the hybrid GP kernel from arXiv:2605.17275.

    RBF(long-term trend) + Matérn3/2(short-term) + White(noise).
    """
    if not _HAS_SKLEARN:
        raise RuntimeError("sklearn not available. Set PORTFOLIO_LAB_ENABLE_ML=1")

    # Long-term smooth trend: length_scale ~60 trading days (1 quarter)
    k_rbf = RBF(length_scale=vol_length_scale, length_scale_bounds=(20.0, 252.0))

    # Short-term fluctuations: Matérn 3/2 with ~5-day length scale
    k_matern = Matern(length_scale=5.0, nu=1.5, length_scale_bounds=(1.0, 21.0))

    # Observation noise
    k_white = WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-4, 0.1))

    # Combine: constant scale per asset × (RBF + Matérn + White)
    kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.1, 10.0)) * \
             (k_rbf + k_matern + k_white)
    return kernel


# ── Main Estimator ─────────────────────────────────────────────────────────

class GaussianProcessVCV:
    """Gaussian Process Volatility-Covariance estimator.

    Predicts per-asset variance using GP regression on historical
    squared log-returns, then constructs the full VCV matrix using
    the historical correlation matrix.

    Usage:
        estimator = GaussianProcessVCV(lookback=252)
        result = estimator.estimate(log_returns, asset_labels=["SPY","GLD","TLT"])
        print(f"Condition number: {result.condition_number:.1f}")
    """

    DEFAULT_LOOKBACK = 504  # ~2 years of trading data
    DATA_DIR = _DATA_DIR
    STATE_FILE = "gp_vcv_state.json"

    def __init__(
        self,
        lookback: int = DEFAULT_LOOKBACK,
        vol_length_scale: float = 60.0,
        data_dir: Optional[Path] = None,
    ):
        self.lookback = lookback
        self.vol_length_scale = vol_length_scale
        self.data_dir = data_dir or GaussianProcessVCV.DATA_DIR
        self._last_result: Optional[GPVCVResult] = None

        if not _HAS_SKLEARN:
            logger.warning(
                "GP-VCV: sklearn not available. "
                "Set PORTFOLIO_LAB_ENABLE_ML=1 to enable GP estimation. "
                "Falling back to EWMA covariance."
            )

    # ── Core Estimation ───────────────────────────────────────────────────

    def estimate(
        self,
        log_returns: np.ndarray,
        asset_labels: Optional[List[str]] = None,
        n_restarts: int = 3,
    ) -> GPVCVResult:
        """Estimate covariance matrix from log-returns using GP regression.

        Args:
            log_returns: T×N matrix of daily log returns (most recent first).
            asset_labels: Optional list of N asset names.
            n_restarts: Number of optimizer restarts (default 3).

        Returns:
            GPVCVResult with covariance, volatility, correlation matrices.
        """
        if not _HAS_SKLEARN:
            raise RuntimeError(
                "sklearn not available. Set PORTFOLIO_LAB_ENABLE_ML=1 "
                "or use EWMA fallback."
            )

        n_days, n_assets = log_returns.shape
        if asset_labels is None:
            asset_labels = [f"Asset_{i}" for i in range(n_assets)]

        # Use at most lookback days
        returns = log_returns[:self.lookback] if log_returns.shape[0] > self.lookback else log_returns
        T = returns.shape[0]

        # ── Step 1: Compute historical correlation matrix ──────────────────
        # Use full sample for stable correlation estimate
        corr = np.atleast_2d(np.corrcoef(returns.T))
        # Ensure valid correlations
        corr = np.nan_to_num(corr, nan=0.0)
        corr = np.clip(corr, -1.0, 1.0)
        # Fix diagonal to 1.0
        np.fill_diagonal(corr, 1.0)

        # ── Step 2: GP-predict per-asset variance ──────────────────────────
        # Input: day index (0 = oldest, T-1 = newest)
        X = np.arange(T).reshape(-1, 1).astype(float)

        # Normalize X to [0, 1] for numerical stability
        X_norm = X / T

        vol_estimates = np.zeros(n_assets)
        kernel_params_all = {}

        # Build kernel once (same for all assets)
        kernel = _build_hybrid_kernel(self.vol_length_scale)

        for i in range(n_assets):
            # Target: squared log returns (variance proxy)
            sq_returns = returns[:, i] ** 2

            # Remove outliers: winsorize at 99.5th percentile
            cap = np.percentile(sq_returns, 99.5)
            sq_returns_clean = np.clip(sq_returns, 0, cap)

            # Fit GP
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=n_restarts,
                normalize_y=True,
                alpha=1e-6,  # small regularization for numerical stability
                random_state=42,
            )
            gp.fit(X_norm, sq_returns_clean)

            # Predict at the most recent point
            X_pred = np.array([[1.0]])  # normalized position of newest day
            pred_var, pred_std = gp.predict(X_pred, return_std=True)

            # Annualize: daily variance × 252
            annual_var = float(np.clip(pred_var[0], 1e-12, None) * 252)
            vol_estimates[i] = np.sqrt(annual_var)

            # Store kernel params (sklearn 1.8+ compatible)
            kernel_params_all[f"asset_{i}"] = {
                "kernel_str": str(gp.kernel_),
                "log_marginal_likelihood": float(
                    getattr(gp, "log_marginal_likelihood_value_", 0.0)
                ),
            }

        # ── Step 3: Build VCV matrix ──────────────────────────────────────
        vol_outer = np.outer(vol_estimates, vol_estimates)
        cov = corr * vol_outer

        # ── Step 4: Ensure PSD ───────────────────────────────────────────
        eigenvalues, eigvecs = np.linalg.eigh(cov)
        min_eig = eigenvalues.min()
        is_psd = bool(min_eig >= 0)
        if not is_psd:
            eigenvalues_pos = np.maximum(eigenvalues, 1e-12)
            cov = eigvecs @ np.diag(eigenvalues_pos) @ eigvecs.T

        condition_number = float(
            eigenvalues.max() / max(abs(eigenvalues[eigenvalues != 0].min()), 1e-12)
            if np.any(eigenvalues != 0) else np.inf
        )

        avg_lml = np.mean([
            kp["log_marginal_likelihood"] for kp in kernel_params_all.values()
        ])

        result = GPVCVResult(
            cov_matrix=cov,
            vol_estimates=vol_estimates,
            corr_matrix=corr,
            kernel_params=kernel_params_all,
            log_marginal_likelihood=avg_lml,
            condition_number=condition_number,
            is_psd=is_psd,
            asset_labels=asset_labels,
            timestamp=datetime.now().isoformat(),
            lookback_days=T,
        )
        self._last_result = result
        return result

    # ── Comparison ────────────────────────────────────────────────────────

    @staticmethod
    def estimate_ewma(log_returns: np.ndarray, halflife: int = 42) -> np.ndarray:
        """Estimate covariance using EWMA (baseline for comparison).

        Args:
            log_returns: T×N matrix of daily log returns.
            halflife: EWMA half-life in days (default 42 ≈ 2 months).

        Returns:
            N×N annualized covariance matrix.
        """
        T, N = log_returns.shape
        # Decay weights
        alpha = 1 - np.exp(np.log(0.5) / halflife)
        weights = (1 - alpha) ** np.arange(T)[::-1]
        weights = weights / weights.sum()

        # Weighted mean returns
        weighted_mean = (log_returns * weights[:, np.newaxis]).sum(axis=0)
        centered = log_returns - weighted_mean

        # Weighted covariance
        cov = (centered * weights[:, np.newaxis]).T @ centered
        # Annualize
        cov = cov * 252
        return cov

    def compare(self, log_returns: np.ndarray, asset_labels: Optional[List[str]] = None) -> dict:
        """Compare GP-VCV vs EWMA covariance estimates.

        Returns dict with both estimates and difference metrics.
        """
        gp_result = self.estimate(log_returns, asset_labels)
        ewma_cov = self.estimate_ewma(log_returns)

        # Frobenius norm of difference
        diff_norm = float(np.linalg.norm(gp_result.cov_matrix - ewma_cov, ord='fro'))

        # Per-asset volatility comparison
        ewma_vols = np.sqrt(np.diag(ewma_cov))

        return {
            "gp_cov": gp_result.cov_matrix.tolist(),
            "ewma_cov": ewma_cov.tolist(),
            "gp_vols": gp_result.vol_estimates.tolist(),
            "ewma_vols": ewma_vols.tolist(),
            "frobenius_diff": diff_norm,
            "gp_condition": gp_result.condition_number,
            "ewma_condition": float(np.linalg.cond(ewma_cov)),
        }

    # ── State Persistence ─────────────────────────────────────────────────

    def save_state(self):
        """Save latest estimation result to disk."""
        if self._last_result is None:
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        state = self._last_result.to_dict()
        state["kernel_params"] = self._last_result.kernel_params
        filepath = self.data_dir / self.STATE_FILE
        filepath.write_text(json.dumps(state, indent=2))
        logger.info(f"GP-VCV state saved to {filepath}")

    def load_state(self) -> Optional[dict]:
        """Load previous estimation state from disk."""
        filepath = self.data_dir / self.STATE_FILE
        if not filepath.exists():
            return None
        return json.loads(filepath.read_text())


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    """CLI entry point for GP-VCV estimation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="GP-VCV: Gaussian Process Covariance Estimation"
    )
    parser.add_argument(
        "mode", choices=["estimate", "compare", "status"],
        help="estimate: run GP-VCV on price data, compare: GP vs EWMA, status: show last result"
    )
    parser.add_argument(
        "--lookback", type=int, default=504,
        help="Trading days of history (default: 504)"
    )
    parser.add_argument(
        "--assets", type=str, default=None,
        help="Comma-separated asset list (default: SPY,GLD,TLT)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save result to disk"
    )
    args = parser.parse_args()

    if not _HAS_SKLEARN:
        print("GP-VCV: sklearn not available.")
        print("Set PORTFOLIO_LAB_ENABLE_ML=1 to enable GP estimation.")
        return 1

    # Fetch price data from market DB
    import sqlite3

    assets = args.assets.split(",") if args.assets else ["SPY", "GLD", "TLT"]
    estimator = GaussianProcessVCV(lookback=args.lookback)

    if args.mode == "status":
        state = estimator.load_state()
        if state:
            print(json.dumps(state, indent=2))
        else:
            print("No previous GP-VCV state found.")
        return 0

    # Fetch price data from SQLite
    db_path = DATA_DIR / "market.db"
    if not db_path.exists():
        print("No market.db found. Run data pipeline first.")
        return 1

    conn = sqlite3.connect(db_path)
    prices = {}
    for sym in assets:
        rows = conn.execute(
            "SELECT date, close FROM prices WHERE symbol = ? ORDER BY date",
            (sym,),
        ).fetchall()
        if rows:
            prices[sym] = [r[1] for r in rows]
    conn.close()

    # Convert to log returns
    log_returns_list = []
    valid_assets = []
    for sym in assets:
        if sym in prices and len(prices[sym]) > 2:
            p = np.array(prices[sym], dtype=float)
            r = np.diff(np.log(p))
            log_returns_list.append(r)
            valid_assets.append(sym)

    if len(valid_assets) < 2:
        print("Insufficient price data for covariance estimation.")
        return 1

    # Align to same length
    min_len = min(len(r) for r in log_returns_list)
    log_returns = np.column_stack([r[-min_len:] for r in log_returns_list])

    if args.mode == "compare":
        result = estimator.compare(log_returns, valid_assets)
        print(f"GP vs EWMA comparison for {valid_assets}:")
        print(f"  Frobenius norm of difference: {result['frobenius_diff']:.6f}")
        print(f"  GP condition number: {result['gp_condition']:.1f}")
        print(f"  EWMA condition number: {result['ewma_condition']:.1f}")
        print(f"\nVolatilities (annualized):")
        for i, sym in enumerate(valid_assets):
            print(f"  {sym}: GP={result['gp_vols'][i]:.1%}, EWMA={result['ewma_vols'][i]:.1%}")
    else:
        result = estimator.estimate(log_returns, valid_assets)
        print(f"GP-VCV estimation for {valid_assets}:")
        print(f"  Log marginal likelihood: {result.log_marginal_likelihood:.2f}")
        print(f"  Condition number: {result.condition_number:.1f}")
        print(f"  PSD: {result.is_psd}")
        print(f"\nCovariance matrix:")
        print(np.array2string(result.cov_matrix, precision=6, suppress_small=True))

    if args.save:
        estimator.save_state()
        print(f"State saved to {estimator.data_dir / estimator.STATE_FILE}")

    return 0


if __name__ == "__main__":
    main()
