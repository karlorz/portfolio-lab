#!/usr/bin/env python3
"""
Portfolio-Lab v2.43 Phase 1.3: Combined Factor Timing Feature Store

Integrates factor returns and macro features into unified feature store
for ML factor timing models.

Usage:
    python -m src.features.factor_timing_pipeline build
    python -m src.features.factor_timing_pipeline current

Author: Autonomous Agent
Version: v2.43 Phase 1.3
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import feature modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.factor_data_fetcher import FactorDataFetcher
from src.features.macro_features import MacroFeatureEngineer

FEATURE_DIR = Path("~/projects/portfolio-lab/data/features").expanduser()
FEATURE_DIR.mkdir(parents=True, exist_ok=True)


class FactorTimingPipeline:
    """Combined feature pipeline for ML factor timing."""
    
    def __init__(self):
        self.feature_dir = FEATURE_DIR
        self.feature_file = self.feature_dir / "factor_timing_features.csv"
        self.metadata_file = self.feature_dir / "feature_metadata.json"
        
        self.factor_fetcher = FactorDataFetcher()
        self.macro_engineer = MacroFeatureEngineer()
    
    def build_feature_dataset(self, use_synthetic: bool = False) -> pd.DataFrame:
        """Build combined factor timing feature dataset."""
        logger.info("Building factor timing feature dataset...")
        
        # 1. Load factor returns
        if use_synthetic:
            factor_data = self.factor_fetcher.generate_synthetic_factor_data()
        else:
            factor_data = self.factor_fetcher.load_factor_data()
        
        # 2. Load macro features
        if use_synthetic:
            macro_data = self.macro_engineer._generate_synthetic_macro_data()
            vix_data = self.macro_engineer._generate_synthetic_vix_data()
            macro_features = self.macro_engineer.engineer_features(macro_data, vix_data)
        else:
            macro_features = self.macro_engineer.engineer_features()
        
        # 3. Engineer factor-specific features
        factor_features = self._engineer_factor_features(factor_data)
        
        # 4. Merge datasets
        logger.info(f"Merging {len(factor_features)} factor rows with {len(macro_features)} macro rows")
        
        # Resample factor features to monthly
        factor_monthly = factor_features.resample('ME').last()
        
        # Align indices
        combined = factor_monthly.join(macro_features, how='inner')
        
        logger.info(f"Combined dataset: {len(combined)} rows, {len(combined.columns)} features")
        
        return combined
    
    def _engineer_factor_features(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """Engineer lagged factor features for prediction."""
        features = pd.DataFrame(index=factor_data.index)
        
        # Factor returns (target variables)
        for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD']:
            if col in factor_data.columns:
                features[f'{col}_return'] = factor_data[col]
                
                # Lagged returns (predictive features)
                for lag in [1, 2, 3, 6, 12]:  # 1-12 month lags
                    features[f'{col}_lag_{lag}m'] = factor_data[col].shift(lag)
                
                # Rolling statistics
                for window in [3, 6, 12]:
                    features[f'{col}_ma_{window}m'] = factor_data[col].rolling(window).mean()
                    features[f'{col}_vol_{window}m'] = factor_data[col].rolling(window).std()
                
                # Cumulative returns (momentum) - use apply with np.prod
                def rolling_prod_return(x):
                    return np.prod(1 + x) - 1
                features[f'{col}_cumret_12m'] = factor_data[col].rolling(12).apply(rolling_prod_return, raw=True)
                
                # Reversal indicator (5-year percentile rank)
                features[f'{col}_percentile_5y'] = factor_data[col].rolling(60).apply(
                    lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-6),
                    raw=False
                )
        
        # Factor correlations (regime detection)
        if all(c in factor_data.columns for c in ['Mkt-RF', 'HML', 'UMD']):
            # Value-momentum correlation
            features['hml_umd_corr_12m'] = factor_data['HML'].rolling(12).corr(factor_data['UMD'])
            # Market-factor correlation
            features['mkt_hml_corr_12m'] = factor_data['Mkt-RF'].rolling(12).corr(factor_data['HML'])
        
        # Cross-sectional dispersion (factor opportunity)
        factor_cols = [c for c in factor_data.columns if c not in ['RF']]
        if len(factor_cols) >= 3:
            features['factor_dispersion'] = factor_data[factor_cols].std(axis=1)
        
        logger.info(f"Engineered {len(features.columns)} factor features")
        return features
    
    def save_dataset(self, features: pd.DataFrame, metadata: Optional[Dict] = None):
        """Save feature dataset with metadata."""
        # Save features as CSV (parquet requires pyarrow/fastparquet)
        features.to_csv(self.feature_file)
        logger.info(f"Saved features to {self.feature_file}")
        
        # Save metadata
        meta = {
            'created': datetime.now().isoformat(),
            'rows': len(features),
            'columns': list(features.columns),
            'numeric_columns': list(features.select_dtypes(include=[np.number]).columns),
            'date_range': {
                'start': features.index.min().isoformat() if len(features) > 0 else None,
                'end': features.index.max().isoformat() if len(features) > 0 else None,
            },
            'target_columns': [c for c in features.columns if '_return' in c and '_lag' not in c],
            'feature_columns': [c for c in features.columns if '_return' not in c],
        }
        
        if metadata:
            meta.update(metadata)
        
        import json
        with open(self.metadata_file, 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        
        logger.info(f"Saved metadata to {self.metadata_file}")
    
    def load_dataset(self) -> Optional[pd.DataFrame]:
        """Load cached feature dataset."""
        if self.feature_file.exists():
            df = pd.read_csv(self.feature_file, index_col=0, parse_dates=True)
            return df
        return None
    
    def get_feature_summary(self) -> Dict:
        """Get summary of feature dataset."""
        features = self.load_dataset()
        
        if features is None:
            return {'error': 'No feature dataset found'}
        
        # Calculate statistics
        numeric = features.select_dtypes(include=[np.number])
        
        summary = {
            'rows': len(features),
            'columns': len(features.columns),
            'date_range': {
                'start': features.index.min().isoformat(),
                'end': features.index.max().isoformat(),
            },
            'targets': [c for c in features.columns if '_return' in c and '_lag' not in c],
            'feature_count': len([c for c in features.columns if '_return' not in c]),
            'correlation_matrix': numeric.corr().to_dict(),
        }
        
        return summary
    
    def show_current_features(self):
        """Display current feature values."""
        features = self.load_dataset()
        
        if features is None or len(features) == 0:
            print("No feature dataset available")
            return
        
        current = features.iloc[-1]
        
        print("\n" + "="*70)
        print("FACTOR TIMING FEATURES (v2.43)")
        print("="*70)
        print(f"Date: {features.index[-1].strftime('%Y-%m-%d')}")
        print()
        
        # Target factors
        targets = [c for c in features.columns if '_return' in c and '_lag' not in c]
        if targets:
            print("Target Factor Returns (Last Month):")
            for t in targets[:6]:
                val = current[t] * 100
                print(f"  {t:20s}: {val:>7.2f}%")
            print()
        
        # Macro regime
        if 'macro_regime' in features.columns:
            regime = current['macro_regime']
            print(f"Macro Regime: {regime}")
        
        # Key predictive features
        key_features = [
            'vix_level', 'yield_curve_slope', 'real_yield_10y',
            'HML_cumret_12m', 'UMD_cumret_12m',
            'hml_umd_corr_12m', 'factor_dispersion'
        ]
        
        print("\nKey Predictive Features:")
        for f in key_features:
            if f in features.columns:
                val = current[f]
                if isinstance(val, (int, float)):
                    if 'return' in f or 'cumret' in f:
                        print(f"  {f:20s}: {val*100:>7.2f}%")
                    else:
                        print(f"  {f:20s}: {val:>7.3f}")
                else:
                    print(f"  {f:20s}: {val}")
        
        # Factor percentile positions
        print("\nFactor Valuation (5-year percentile):")
        for factor in ['HML', 'UMD', 'SMB', 'RMW']:
            col = f'{factor}_percentile_5y'
            if col in features.columns:
                val = current[col]
                bar = '█' * int(val * 10) + '░' * (10 - int(val * 10))
                print(f"  {factor:10s}: {bar} {val*100:>5.1f}%")
        
        print("="*70)


def main():
    """CLI interface for factor timing pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Factor Timing Feature Pipeline v2.43")
    parser.add_argument('command', choices=['build', 'current', 'stats', 'synthetic'])
    parser.add_argument('--refresh', action='store_true', help='Force refresh from sources')
    
    args = parser.parse_args()
    
    pipeline = FactorTimingPipeline()
    
    if args.command == 'build':
        features = pipeline.build_feature_dataset(use_synthetic=False)
        pipeline.save_dataset(features, {'source': 'live_data', 'version': 'v2.43'})
        print(f"\n✓ Feature dataset built: {len(features)} rows, {len(features.columns)} columns")
        
    elif args.command == 'synthetic':
        features = pipeline.build_feature_dataset(use_synthetic=True)
        pipeline.save_dataset(features, {'source': 'synthetic', 'version': 'v2.43'})
        print(f"\n✓ Synthetic feature dataset built: {len(features)} rows, {len(features.columns)} columns")
        
    elif args.command == 'current':
        pipeline.show_current_features()
        
    elif args.command == 'stats':
        summary = pipeline.get_feature_summary()
        import json
        print(json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
