"""Neutral metrics IO + return-metrics helpers (layer-leak fix, Items A2+A3).

Moved from ``src/backtest/metrics.py`` so data/regime layers can use
``save_results_json`` / ``compute_metrics_from_returns`` without importing the
backtest layer. ``src/backtest/metrics.py`` re-exports these names — all
existing importers keep working unchanged.

Top-level imports are stdlib + numpy only (no src.backtest edge), so this
module cannot participate in an import cycle.
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Dashboard / Caddy-served JSON must be world-readable (Batch HZ residual after
# multi-dest fchmod on signals). save_results_json still uses open()+write.
_PUBLIC_JSON_MODE = 0o644

TRADING_DAYS_PER_YEAR: int = 252


def compute_metrics_from_returns(
    returns: List[float],
    risk_free_rate: Optional[float] = None,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Dict[str, float]:
    """Compute core metrics directly from daily returns.

    Lightweight alternative to compute_metrics() when you have raw
    returns instead of an equity curve. Returns a flat dict for easy
    integration into backtest scripts.

    Args:
        returns: List or array of daily returns (e.g., [0.01, -0.005, ...]).
        risk_free_rate: Annual risk-free rate (default from RISK_FREE_RATE).
        trading_days_per_year: Annualization factor (default 252).

    Returns:
        Dict with keys: total_return, cagr, volatility, sharpe, max_drawdown, calmar.
        Values are decimals (not percentages) for direct use in calculations.
    """
    if risk_free_rate is None:
        from src.paths import RISK_FREE_RATE
        risk_free_rate = RISK_FREE_RATE / 100

    returns_arr = np.array(returns, dtype=float)
    n = len(returns_arr)

    if n == 0:
        return {
            'total_return': 0.0, 'cagr': 0.0, 'volatility': 0.0,
            'sharpe': 0.0, 'max_drawdown': 0.0, 'calmar': 0.0,
        }

    total_return = float(np.prod(1 + returns_arr) - 1)
    years = n / trading_days_per_year
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    daily_vol = float(np.std(returns_arr, ddof=1)) if n > 1 else 0.0
    annualized_vol = daily_vol * np.sqrt(trading_days_per_year)

    # Sharpe: (CAGR - Rf) / vol
    sharpe = (cagr - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0.0

    # Max drawdown from cumulative returns
    cumulative = np.cumprod(1 + returns_arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_dd = float(np.min(drawdown))

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    return {
        'total_return': round(total_return, 6),
        'cagr': round(cagr, 6),
        'volatility': round(annualized_vol, 6),
        'sharpe': round(sharpe, 4),
        'max_drawdown': round(max_dd, 6),
        'calmar': round(calmar, 4),
    }


def save_results_json(
    data: dict,
    output_path: str = None,
    default_dir: Path = None,
    validator: Callable[[dict], dict] = None,
    experiment_manifest: Optional[Dict[str, Any]] = None,
    data_snapshot: Optional[Mapping[str, Any]] = None,
):
    """Save results dict to JSON file.

    Args:
        data: Dict to serialize.
        output_path: Explicit output path (overrides default_dir).
        default_dir: Directory for auto-named output.
        validator: Optional validation function. If provided, data is passed
            through this function before serialization. Should return validated
            data or the original data on validation failure.
        experiment_manifest: Optional provenance config for experiment result
            artifacts. When provided, must include ``experiment_id`` and may
            include ``manifest_mode`` (embedded or sidecar), command, module,
            config_snapshot, env_keys, and input_paths. Normal JSON writes are
            unchanged when omitted.
        data_snapshot: Optional historical data snapshot provenance to embed
            under ``_data_snapshot``. Normal JSON writes are unchanged when
            omitted.
    """
    if data_snapshot is not None:
        data = dict(data)
        data["_data_snapshot"] = dict(data_snapshot)

    if validator is not None:
        try:
            data = validator(data)
        except Exception as e:
            logger.warning("Validation callback failed: %s", e)

    if output_path:
        path = Path(output_path)
    elif default_dir:
        default_dir.mkdir(parents=True, exist_ok=True)
        path = default_dir / "backtest_results.json"
    else:
        return

    # Public artifacts have a smaller disclosure surface than private monitor
    # files.  Apply the shared projection here as a last-mile guard for legacy
    # producers that still call save_results_json directly.
    public_output = False
    try:
        from src.dashboard.public_projection import (
            is_public_output_path,
            prepare_payload_for_write,
        )

        public_output = is_public_output_path(path)
        if public_output:
            data = prepare_payload_for_write(data, path, public=True)
    except Exception as projection_exc:  # noqa: BLE001 - preserve legacy saves
        logger.warning("Public payload projection failed for %s: %s", path, projection_exc)

    if experiment_manifest is not None:
        from src.research.experiment_manifest import save_experiment_result_json

        manifest_config = dict(experiment_manifest)
        experiment_id = manifest_config.pop("experiment_id")
        save_experiment_result_json(data, path, experiment_id=experiment_id, **manifest_config)
        _maybe_record_backtest_experiment(data, path, experiment_manifest)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from src.monitor.signal_authority import serialize_json_payload

        path.write_text(
            serialize_json_payload(
                data,
                output_path=path,
                public=public_output,
            ),
            encoding="utf-8",
        )
        # Batch HZ: normalize mode so public dashboard dual-writes via
        # save_results_json never leave sticky 0600 (Caddy 403). Safe for
        # private backtest artifacts on lab hosts (not secrets).
        try:
            os.chmod(path, _PUBLIC_JSON_MODE)
        except OSError as chmod_exc:
            logger.warning("chmod %s after save_results_json failed: %s", path, chmod_exc)
    except (OSError, TypeError, ValueError) as e:
        logger.error("Failed to save results to %s: %s", path, e)
        raise

    _maybe_record_backtest_experiment(data, path, experiment_manifest)


def _maybe_record_backtest_experiment(
    data: dict,
    path: Path,
    experiment_manifest: Optional[Dict[str, Any]],
) -> None:
    """Append experiment row to decision registry when saving result JSON."""
    if experiment_manifest is None:
        return
    experiment_id = experiment_manifest.get("experiment_id")
    if not experiment_id:
        return
    try:
        from src.monitor.decision_registry import record_backtest_experiment

        record_backtest_experiment(
            data,
            experiment_id=str(experiment_id),
            output_path=path,
            name=str(experiment_manifest.get("name") or experiment_id),
            hypothesis=str(experiment_manifest.get("hypothesis") or ""),
            tags=["experiment_manifest"],
        )
    except (ImportError, ValueError, OSError, TypeError) as e:
        logger.warning("Decision registry backtest record skipped: %s", e)
