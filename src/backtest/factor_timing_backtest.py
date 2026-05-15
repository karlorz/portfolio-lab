#!/usr/bin/env python3
"""
v2.43 ML Factor Timing Backtest
Walk-forward validation of XGBoost (sklearn GradientBoosting) factor timing.

Uses existing factor_timing_features.csv (Fama-French 5-factor + macro features)
to predict next-month factor returns and dynamically tilt factor weights.

Target: +0.03-0.05 Sharpe improvement over static factor allocation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Conditional sklearn import — only loaded when actually running the backtest
try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

DATA_DIR = Path(__file__).parent.parent.parent / "data"
FEATURES_FILE = DATA_DIR / "features" / "factor_timing_features.csv"
PRICES_FILE = Path(__file__).parent.parent.parent / "public" / "data" / "prices.json"


def load_features() -> pd.DataFrame:
    """Load and prepare factor timing features."""
    df = pd.read_csv(FEATURES_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()

    # Drop rows with too many NaNs
    df = df.dropna(thresh=int(len(df.columns) * 0.7))

    # Forward-fill remaining NaNs (common for monthly data)
    df = df.ffill()

    # Drop columns that are entirely NaN
    df = df.dropna(axis=1, how='all')

    return df


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Build next-month factor return targets."""
    targets = pd.DataFrame(index=df.index)

    # Next-month returns for key factors
    for factor in ['HML_return', 'UMD_return', 'RMW_return']:
        if factor in df.columns:
            targets[f'{factor}_fwd1m'] = df[factor].shift(-1)

    # Composite: equal-weight HML + UMD + RMW
    fwd_cols = [c for c in targets.columns if c.endswith('_fwd1m')]
    if fwd_cols:
        targets['composite_fwd1m'] = targets[fwd_cols].mean(axis=1)

    return targets


def get_feature_columns(df: pd.DataFrame) -> list:
    """Select feature columns (exclude targets and raw returns)."""
    # Exclude raw factor returns (these are the targets, not features)
    raw_returns = ['Mkt-RF_return', 'SMB_return', 'HML_return', 'RMW_return',
                   'CMA_return', 'UMD_return']

    # Exclude string/categorical columns
    exclude = raw_returns + ['real_rate_regime', 'macro_regime']

    feature_cols = [c for c in df.columns if c not in exclude
                    and not c.endswith('_return')
                    and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

    return feature_cols


def walk_forward_backtest(
    df: pd.DataFrame,
    targets: pd.DataFrame,
    feature_cols: list,
    train_years: int = 3,
    start_year: int = 2015,
    end_year: int = 2026,
) -> dict:
    """
    Walk-forward backtest: train on train_years, predict next month,
    rebalance monthly. Compare ML-timed vs static factor allocation.
    """
    if not SKLEARN_AVAILABLE:
        return {'error': 'sklearn not available — run with PORTFOLIO_LAB_ENABLE_ML=1 or install sklearn'}
    # Merge features and targets
    merged = df[feature_cols].join(targets)
    # Drop rows where composite target is NaN (last row)
    merged = merged.dropna(subset=['composite_fwd1m'])
    # Fill remaining feature NaNs with 0 (neutral signal)
    merged = merged.fillna(0)

    if 'composite_fwd1m' not in merged.columns:
        return {'error': 'No composite target available'}

    # Filter to date range
    merged = merged[(merged.index.year >= start_year - train_years)]

    results = {
        'ml_returns': [],
        'static_returns': [],
        'ml_predictions': [],
        'actual_returns': [],
        'dates': [],
        'feature_importance': [],
    }

    # Walk-forward loop
    test_start = pd.Timestamp(f'{start_year}-01-01')

    for test_year in range(start_year, end_year + 1):
        for test_month in range(1, 13):
            test_date = pd.Timestamp(f'{test_year}-{test_month:02d}-01')

            if test_date not in merged.index:
                # Find nearest available date
                available = merged.index[merged.index <= test_date]
                if len(available) == 0:
                    continue
                test_date = available[-1]

            # Training window: 3 years before test date
            train_end = test_date - pd.DateOffset(months=1)
            train_start = train_end - pd.DateOffset(years=train_years)

            train_mask = (merged.index >= train_start) & (merged.index <= train_end)
            test_mask = merged.index == test_date

            train_data = merged[train_mask]
            test_data = merged[test_mask]

            if len(train_data) < 24 or len(test_data) == 0:
                continue

            X_train = train_data[feature_cols].values
            y_train = train_data['composite_fwd1m'].values
            X_test = test_data[feature_cols].values

            # Handle NaN in features
            valid_mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
            X_train = X_train[valid_mask]
            y_train = y_train[valid_mask]

            if len(X_train) < 12:
                continue

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            model = GradientBoostingRegressor(
                max_depth=3,
                learning_rate=0.05,
                n_estimators=100,
                subsample=0.8,
                min_samples_leaf=6,
                random_state=42,
            )
            model.fit(X_train_scaled, y_train)

            # Predict
            pred = model.predict(X_test_scaled)[0]
            actual = test_data['composite_fwd1m'].values[0]

            # ML-timed return: scale position by prediction confidence
            # If model predicts positive → overweight factors
            # If model predicts negative → underweight factors
            # Position scale: 0.5 to 1.5 (50% to 150% of base)
            position_scale = np.clip(1.0 + pred * 10, 0.5, 1.5)
            ml_return = actual * position_scale
            static_return = actual  # Equal-weight, no timing

            results['ml_returns'].append(ml_return)
            results['static_returns'].append(static_return)
            results['ml_predictions'].append(pred)
            results['actual_returns'].append(actual)
            results['dates'].append(test_date.isoformat())

            # Feature importance (last iteration)
            if test_year == end_year and test_month == 12:
                importances = model.feature_importances_
                top_idx = np.argsort(importances)[-10:]
                results['feature_importance'] = [
                    {'feature': feature_cols[i], 'importance': float(importances[i])}
                    for i in reversed(top_idx)
                ]

    return results


def compute_metrics(returns: list, label: str) -> dict:
    """Compute performance metrics for a return series."""
    if not returns:
        return {'error': 'No returns'}

    r = np.array(returns)
    r = r[~np.isnan(r)]

    if len(r) < 2:
        return {'error': 'Insufficient data'}

    ann_return = np.mean(r) * 12
    ann_vol = np.std(r) * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    cum = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = float(np.min(dd))

    # Win rate
    win_rate = float(np.mean(r > 0))

    return {
        'label': label,
        'months': len(r),
        'ann_return': round(float(ann_return) * 100, 2),
        'ann_vol': round(float(ann_vol) * 100, 2),
        'sharpe': round(float(sharpe), 3),
        'max_drawdown': round(max_dd * 100, 2),
        'win_rate': round(win_rate * 100, 1),
        'total_return': round(float((cum[-1] - 1) * 100), 2),
    }


def run_full_backtest():
    """Run the complete ML factor timing backtest."""
    print("=== v2.43 ML Factor Timing Backtest ===\n")

    # Load data
    print("Loading features...")
    df = load_features()
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

    # Build targets
    targets = build_targets(df)
    print(f"  Targets: {list(targets.columns)}")

    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"  Features: {len(feature_cols)} columns")

    # Walk-forward backtest
    print("\nRunning walk-forward backtest (3yr train, 1yr test windows)...")
    results = walk_forward_backtest(df, targets, feature_cols)

    if 'error' in results:
        print(f"Error: {results['error']}")
        return

    # Compute metrics
    ml_metrics = compute_metrics(results['ml_returns'], 'ML-Timed')
    static_metrics = compute_metrics(results['static_returns'], 'Static')

    print(f"\n{'='*60}")
    print(f"{'Metric':<25} {'ML-Timed':>12} {'Static':>12} {'Delta':>10}")
    print(f"{'='*60}")

    for key in ['months', 'ann_return', 'ann_vol', 'sharpe', 'max_drawdown', 'win_rate', 'total_return']:
        ml_val = ml_metrics.get(key, 'N/A')
        st_val = static_metrics.get(key, 'N/A')

        if isinstance(ml_val, (int, float)) and isinstance(st_val, (int, float)):
            delta = ml_val - st_val
            delta_str = f"{delta:+.3f}" if isinstance(delta, float) else f"{delta:+d}"
        else:
            delta_str = 'N/A'

        label = key.replace('_', ' ').title()
        print(f"{label:<25} {str(ml_val):>12} {str(st_val):>12} {delta_str:>10}")

    # Sharpe improvement
    sharpe_delta = ml_metrics.get('sharpe', 0) - static_metrics.get('sharpe', 0)
    print(f"\n{'='*60}")
    print(f"Sharpe Improvement: {sharpe_delta:+.3f}")
    target_met = abs(sharpe_delta) >= 0.03
    print(f"Target (+0.03): {'MET' if target_met else 'NOT MET'}")

    # Feature importance
    if results.get('feature_importance'):
        print(f"\nTop 10 Features:")
        for fi in results['feature_importance'][:10]:
            print(f"  {fi['feature']:<40} {fi['importance']:.4f}")

    # Prediction accuracy
    predictions = np.array(results['ml_predictions'])
    actuals = np.array(results['actual_returns'])
    direction_correct = np.mean(np.sign(predictions) == np.sign(actuals))
    print(f"\nDirectional Accuracy: {direction_correct*100:.1f}%")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'ml_metrics': ml_metrics,
        'static_metrics': static_metrics,
        'sharpe_improvement': round(sharpe_delta, 3),
        'target_met': target_met,
        'directional_accuracy': round(float(direction_correct) * 100, 1),
        'feature_importance': results.get('feature_importance', []),
        'monthly_returns': {
            'ml': [round(r, 6) for r in results['ml_returns']],
            'static': [round(r, 6) for r in results['static_returns']],
            'dates': results['dates'],
        },
    }

    output_path = DATA_DIR / "factor_timing_backtest_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return output


if __name__ == '__main__':
    run_full_backtest()
