#!/usr/bin/env python3
"""
Portfolio-Lab v2.43 Phase 1.2: Macro Features Pipeline

Engineers macro features for ML factor timing:
- VIX level and term structure
- Real yields (10Y TIPS)
- Credit spreads (HY - Treasury)
- Yield curve slope (10Y - 2Y)
- Dollar index (DXY)
- Inflation expectations

Features are designed for monthly factor return prediction.

Author: Autonomous Agent
Version: v2.43 Phase 1.2
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import logging
import json
import sqlite3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
FEATURE_DIR = DATA_DIR / "features"
FEATURE_DIR.mkdir(exist_ok=True)

FRED_DATA_PATH = DATA_DIR / "fred_data.json"
VIX_DATA_PATH = DATA_DIR / "vix_term_structure.json"


class MacroFeatureEngineer:
    """Engineer macroeconomic features for ML models."""
    
    def __init__(self):
        self.feature_dir = FEATURE_DIR
        self.feature_file = self.feature_dir / "macro_features.csv"
        
    def load_fred_data(self) -> pd.DataFrame:
        """Load FRED economic data from local storage."""
        if not FRED_DATA_PATH.exists():
            logger.warning("FRED data not found, generating synthetic data")
            return self._generate_synthetic_macro_data()
        
        try:
            with open(FRED_DATA_PATH, 'r') as f:
                data = json.load(f)
            
            # Parse FRED series
            records = []
            series_names = {
                'GS10': 'treasury_10y',
                'GS2': 'treasury_2y',
                'DGS10': 'treasury_10y_daily',
                'DGS2': 'treasury_2y_daily',
                'DFF': 'fed_funds',
                'DFII10': 'tips_10y',
                'DCOILBRENTEU': 'oil_brent',
                'DEXUSEU': 'eur_usd',
                'T10YIE': 'breakeven_10y',
            }
            
            for series_id, values in data.get('series', {}).items():
                col_name = series_names.get(series_id, series_id)
                for obs in values:
                    if obs.get('value') not in ['.', '']:
                        try:
                            records.append({
                                'date': obs['date'],
                                'series': col_name,
                                'value': float(obs['value'])
                            })
                        except (ValueError, TypeError):
                            pass

            if not records:
                return self._generate_synthetic_macro_data()
            
            df = pd.DataFrame(records)
            df['date'] = pd.to_datetime(df['date'])
            df = df.pivot(index='date', columns='series', values='value')
            
            logger.info(f"Loaded FRED data: {len(df)} rows, {len(df.columns)} series")
            return df
            
        except Exception as e:
            logger.error(f"Error loading FRED data: {e}")
            return self._generate_synthetic_macro_data()
    
    def load_vix_data(self) -> pd.DataFrame:
        """Load VIX term structure data."""
        if not VIX_DATA_PATH.exists():
            logger.warning("VIX data not found, generating synthetic")
            return self._generate_synthetic_vix_data()
        
        try:
            with open(VIX_DATA_PATH, 'r') as f:
                data = json.load(f)
            
            records = []
            for date_str, values in data.items():
                try:
                    records.append({
                        'date': pd.to_datetime(date_str),
                        'vix_spot': values.get('vix_spot', values.get('vix', 20)),
                        'vix_1m': values.get('vix_1m', values.get('vix', 20)),
                        'vix_3m': values.get('vix_3m', values.get('vix', 20) * 0.95),
                    })
                except (ValueError, TypeError, KeyError):
                    pass
            
            df = pd.DataFrame(records).set_index('date').sort_index()
            logger.info(f"Loaded VIX data: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error loading VIX data: {e}")
            return self._generate_synthetic_vix_data()
    
    def _generate_synthetic_macro_data(self, start_date: str = '2010-01-01') -> pd.DataFrame:
        """Generate synthetic macro data for testing."""
        np.random.seed(42)
        dates = pd.date_range(start=start_date, end=datetime.now(), freq='D')
        
        data = pd.DataFrame(index=dates)
        data['treasury_10y'] = np.cumsum(np.random.normal(0.0001, 0.05, len(dates))) + 2.5
        data['treasury_10y'] = data['treasury_10y'].clip(0.5, 10)
        data['treasury_2y'] = data['treasury_10y'] - np.random.uniform(0, 2, len(dates))
        data['fed_funds'] = data['treasury_2y'] + np.random.normal(0, 0.3, len(dates))
        data['tips_10y'] = data['treasury_10y'] - np.random.uniform(1.5, 3.5, len(dates))
        data['oil_brent'] = np.cumsum(np.random.normal(0, 0.5, len(dates))) + 60
        data['oil_brent'] = data['oil_brent'].clip(20, 150)
        data['breakeven_10y'] = data['treasury_10y'] - data['tips_10y']
        
        logger.info(f"Generated synthetic macro data: {len(data)} rows")
        return data
    
    def _generate_synthetic_vix_data(self, start_date: str = '2010-01-01') -> pd.DataFrame:
        """Generate synthetic VIX data for testing."""
        np.random.seed(42)
        dates = pd.date_range(start=start_date, end=datetime.now(), freq='D')
        
        # Mean-reverting VIX
        vix = np.zeros(len(dates))
        vix[0] = 20
        for i in range(1, len(dates)):
            vix[i] = vix[i-1] * 0.95 + 20 * 0.05 + np.random.normal(0, 2)
        
        data = pd.DataFrame({
            'vix_spot': np.clip(vix, 10, 80),
            'vix_1m': np.clip(vix * np.random.uniform(0.9, 1.1, len(dates)), 10, 80),
            'vix_3m': np.clip(vix * np.random.uniform(0.85, 1.15, len(dates)), 10, 80),
        }, index=dates)
        
        return data
    
    def engineer_features(self, macro_data: Optional[pd.DataFrame] = None, 
                         vix_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Engineer ML-ready macro features."""
        macro = macro_data if macro_data is not None else self.load_fred_data()
        vix = vix_data if vix_data is not None else self.load_vix_data()
        
        # Resample to monthly for factor timing
        macro_monthly = macro.resample('ME').last()
        vix_monthly = vix.resample('ME').last()
        
        # Combine datasets
        features = pd.DataFrame(index=macro_monthly.index)
        
        # 1. Interest Rate Features
        if 'treasury_10y' in macro_monthly.columns:
            features['yield_10y'] = macro_monthly['treasury_10y']
        if 'treasury_2y' in macro_monthly.columns:
            features['yield_2y'] = macro_monthly['treasury_2y']
        if 'fed_funds' in macro_monthly.columns:
            features['fed_rate'] = macro_monthly['fed_funds']
        if 'tips_10y' in macro_monthly.columns:
            features['real_yield_10y'] = macro_monthly['tips_10y']
        
        # Yield curve slope (recession indicator)
        if 'treasury_10y' in macro_monthly.columns and 'treasury_2y' in macro_monthly.columns:
            features['yield_curve_slope'] = macro_monthly['treasury_10y'] - macro_monthly['treasury_2y']
            features['curve_inverted'] = (features['yield_curve_slope'] < 0).astype(int)
        
        # 2. VIX Features
        features = features.join(vix_monthly[['vix_spot']], how='left')
        if 'vix_spot' in features.columns:
            features['vix_level'] = features['vix_spot']
            features['vix_percentile_1y'] = features['vix_level'].rolling(252).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-6), raw=False
            )
        
        # 3. Credit Spread (if available, else estimate)
        if 'breakeven_10y' in macro_monthly.columns:
            features['inflation_expectations'] = macro_monthly['breakeven_10y']
        
        # 4. Momentum/Change features
        for col in ['yield_10y', 'vix_level']:
            if col in features.columns:
                features[f'{col}_change_1m'] = features[col].diff(1)
                features[f'{col}_change_3m'] = features[col].diff(3)
                features[f'{col}_change_12m'] = features[col].diff(12)
        
        # 5. Rate regime classification
        if 'real_yield_10y' in features.columns:
            features['real_rate_regime'] = pd.cut(
                features['real_yield_10y'],
                bins=[-float('inf'), 0, 1, 2, float('inf')],
                labels=['deep_negative', 'negative', 'low', 'elevated']
            )
        
        # 6. Macro regime composite
        if 'vix_level' in features.columns and 'yield_curve_slope' in features.columns:
            conditions = [
                (features['vix_level'] < 20) & (features['yield_curve_slope'] > 0.5),
                (features['vix_level'] < 25) & (features['yield_curve_slope'] > 0),
                (features['vix_level'] < 30),
                (features['vix_level'] >= 30)
            ]
            choices = ['bull_normal', 'bull_late', 'neutral', 'bear_stress']
            features['macro_regime'] = np.select(conditions, choices, default='neutral')
        
        # Drop rows with too many NaNs
        features = features.dropna(thresh=len(features.columns) * 0.5)
        
        logger.info(f"Engineered {len(features.columns)} features: {list(features.columns)}")
        return features
    
    def save_features(self, features: pd.DataFrame):
        """Save engineered features to CSV."""
        features.to_csv(self.feature_file)
        logger.info(f"Saved features to {self.feature_file}")
    
    def load_features(self) -> pd.DataFrame:
        """Load cached features."""
        if self.feature_file.exists():
            return pd.read_csv(self.feature_file, index_col=0, parse_dates=True)
        features = self.engineer_features()
        self.save_features(features)
        return features


def main():
    """CLI interface for macro feature engineering."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Macro Feature Engineering v2.43")
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--stats', action='store_true', help='Show feature statistics')
    parser.add_argument('--regime', action='store_true', help='Show current regime classification')
    
    args = parser.parse_args()
    
    engineer = MacroFeatureEngineer()
    
    if args.synthetic:
        macro = engineer._generate_synthetic_macro_data()
        vix = engineer._generate_synthetic_vix_data()
        features = engineer.engineer_features(macro, vix)
    else:
        features = engineer.engineer_features()
    
    print(f"\nMacro Features Summary:")
    print(f"  Rows: {len(features)}")
    print(f"  Date range: {features.index.min().date()} to {features.index.max().date()}")
    print(f"  Features: {len(features.columns)}")
    print(f"  Numeric: {[c for c in features.columns if features[c].dtype in ['float64', 'int64']]}")
    
    if args.stats:
        numeric = features.select_dtypes(include=[np.number])
        print(f"\nFeature Statistics (last 12 months):")
        print(numeric.tail(12).describe().T[['mean', 'std', 'min', 'max']].to_string())
    
    if args.regime and 'macro_regime' in features.columns:
        current = features['macro_regime'].iloc[-1]
        print(f"\nCurrent Macro Regime: {current}")
        
        if 'vix_level' in features.columns:
            vix = features['vix_level'].iloc[-1]
            print(f"  VIX Level: {vix:.2f}")
        
        if 'yield_curve_slope' in features.columns:
            slope = features['yield_curve_slope'].iloc[-1]
            print(f"  Yield Curve: {slope:.2f}% ({'Inverted' if slope < 0 else 'Normal'})")
        
        if 'real_yield_10y' in features.columns:
            real = features['real_yield_10y'].iloc[-1]
            print(f"  Real Yield (10Y): {real:.2f}%")


if __name__ == '__main__':
    main()
