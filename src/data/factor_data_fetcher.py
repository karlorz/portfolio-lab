#!/usr/bin/env python3
"""
Portfolio-Lab v2.43 Phase 1: Factor Data Fetcher

Fetches Fama-French 5-factor data and AQR factor zoo for ML factor timing.
Sources:
- Ken French Data Library (free): http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- AQR Factor Data (free): https://www.aqr.com/Insights/Datasets

Author: Autonomous Agent
Version: v2.43 Phase 1
"""

import pandas as pd
import numpy as np
import requests
import zipfile
import io
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
FACTOR_DIR = DATA_DIR / "factors"
FACTOR_DIR.mkdir(exist_ok=True)

# Fama-French 5 Factor URLs
FF_URLS = {
    'daily_5_factors': 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip',
    'monthly_5_factors': 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip',
    'momentum': 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip',
}

AQR_URLS = {
    'quality': 'https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Quality-Minus-Junk-Factors-Daily.xlsx',
    'betting_against_beta': 'https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Betting-Against-Beta-Equity-Factors-Daily.xlsx',
}


class FactorDataFetcher:
    """Fetch and store factor return data from academic sources."""
    
    def __init__(self):
        self.data_dir = FACTOR_DIR
        self.cache_file = self.data_dir / "factor_returns.csv"
        
    def download_fama_french_daily(self) -> Optional[pd.DataFrame]:
        """Download Fama-French 5-factor daily returns."""
        try:
            url = FF_URLS['daily_5_factors']
            logger.info(f"Downloading Fama-French 5 factors from {url}")
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Extract CSV from zip
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Find the CSV file (usually named similarly to zip)
                csv_name = [f for f in z.namelist() if f.endswith('.CSV')][0]
                with z.open(csv_name) as f:
                    # Skip header rows (Ken French files have descriptive headers)
                    lines = f.read().decode('utf-8').split('\n')
                    # Find the data start (usually after a blank line)
                    data_start = 0
                    for i, line in enumerate(lines):
                        if line.startswith('Date'):
                            data_start = i
                            break
                    
                    # Read the data
                    from io import StringIO
                    df = pd.read_csv(
                        StringIO('\n'.join(lines[data_start:])),
                        skip_blank_lines=True
                    )
                    
                    # Clean column names
                    df.columns = [c.strip() for c in df.columns]
                    
                    # Parse date
                    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d', errors='coerce')
                    df = df.dropna(subset=['Date'])
                    df = df.set_index('Date')
                    
                    # Convert to decimal returns (from percentages)
                    factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
                    for col in factor_cols:
                        if col in df.columns:
                            df[col] = df[col] / 100.0
                    
                    logger.info(f"Loaded FF5 data: {len(df)} rows, {df.index.min()} to {df.index.max()}")
                    return df
                    
        except Exception as e:
            logger.error(f"Failed to download Fama-French data: {e}")
            return None
    
    def download_momentum_factor(self) -> Optional[pd.DataFrame]:
        """Download Carhart momentum factor (UMD)."""
        try:
            url = FF_URLS['momentum']
            logger.info(f"Downloading momentum factor from {url}")
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_name = [f for f in z.namelist() if f.endswith('.CSV')][0]
                with z.open(csv_name) as f:
                    lines = f.read().decode('utf-8').split('\n')
                    
                    data_start = 0
                    for i, line in enumerate(lines):
                        if line.startswith('Date'):
                            data_start = i
                            break
                    
                    from io import StringIO
                    df = pd.read_csv(StringIO('\n'.join(lines[data_start:])))
                    df.columns = [c.strip() for c in df.columns]
                    
                    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d', errors='coerce')
                    df = df.dropna(subset=['Date'])
                    df = df.set_index('Date')
                    
                    # Momentum factor usually named 'Mom' or 'MOM'
                    mom_col = [c for c in df.columns if 'Mom' in c][0]
                    df[mom_col] = df[mom_col] / 100.0
                    df = df.rename(columns={mom_col: 'UMD'})
                    
                    logger.info(f"Loaded momentum data: {len(df)} rows")
                    return df[['UMD']]
                    
        except Exception as e:
            logger.error(f"Failed to download momentum data: {e}")
            return None
    
    def generate_synthetic_factor_data(self, start_date: str = '2010-01-01') -> pd.DataFrame:
        """Generate synthetic factor data for testing when downloads fail."""
        logger.warning("Generating synthetic factor data for testing")
        
        dates = pd.date_range(start=start_date, end=datetime.now(), freq='D')
        np.random.seed(42)
        
        # Factor statistics based on historical averages
        factors = {
            'Mkt-RF': {'mean': 0.0003, 'std': 0.012, 'auto': 0.0},   # Market excess
            'SMB': {'mean': 0.0001, 'std': 0.008, 'auto': 0.1},      # Size
            'HML': {'mean': 0.0002, 'std': 0.009, 'auto': 0.15},    # Value
            'RMW': {'mean': 0.0002, 'std': 0.007, 'auto': 0.1},     # Profitability
            'CMA': {'mean': 0.0001, 'std': 0.006, 'auto': 0.1},     # Conservative investment
            'RF': {'mean': 0.0001, 'std': 0.0001, 'auto': 0.9},     # Risk-free
            'UMD': {'mean': 0.0003, 'std': 0.011, 'auto': 0.2},     # Momentum
        }
        
        data = {}
        for factor, params in factors.items():
            returns = np.random.normal(params['mean'], params['std'], len(dates))
            # Add autocorrelation
            for i in range(1, len(returns)):
                returns[i] = params['auto'] * returns[i-1] + (1 - params['auto']) * returns[i]
            data[factor] = returns
        
        df = pd.DataFrame(data, index=dates)
        df.index.name = 'Date'
        
        logger.info(f"Generated synthetic data: {len(df)} rows, {len(factors)} factors")
        return df
    
    def fetch_all_factors(self) -> pd.DataFrame:
        """Fetch and combine all factor data sources."""
        ff_data = self.download_fama_french_daily()
        mom_data = self.download_momentum_factor()
        
        if ff_data is None:
            logger.warning("Using synthetic factor data")
            return self.generate_synthetic_factor_data()
        
        # Combine datasets
        combined = ff_data.copy()
        
        if mom_data is not None:
            # Align and merge momentum
            combined = combined.join(mom_data, how='left')
        
        # Fill missing values with 0 (assumes neutral on missing days)
        combined = combined.fillna(0)
        
        # Save to cache
        combined.to_csv(self.cache_file)
        logger.info(f"Saved factor data to {self.cache_file}")
        
        return combined
    
    def load_factor_data(self, refresh: bool = False) -> pd.DataFrame:
        """Load factor data from cache or fetch fresh."""
        if not refresh and self.cache_file.exists():
            logger.info(f"Loading cached factor data from {self.cache_file}")
            return pd.read_csv(self.cache_file, index_col=0, parse_dates=True)
        
        return self.fetch_all_factors()
    
    def get_factor_stats(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calculate factor statistics."""
        df = data if data is not None else self.load_factor_data()
        
        stats = pd.DataFrame({
            'mean_daily': df.mean(),
            'std_daily': df.std(),
            'sharpe_annual': (df.mean() / df.std()) * np.sqrt(252),
            'skew': df.skew(),
            'kurtosis': df.kurtosis(),
            'max_dd': (df.cumsum() - df.cumsum().cummax()).min(),
        })
        
        return stats


def main():
    """CLI interface for factor data fetching."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Factor Data Fetcher v2.43")
    parser.add_argument('--refresh', action='store_true', help='Refresh data from sources')
    parser.add_argument('--stats', action='store_true', help='Show factor statistics')
    parser.add_argument('--test', action='store_true', help='Generate test/synthetic data')
    
    args = parser.parse_args()
    
    fetcher = FactorDataFetcher()
    
    if args.test:
        data = fetcher.generate_synthetic_factor_data()
    else:
        data = fetcher.load_factor_data(refresh=args.refresh)
    
    print(f"\nFactor Data Summary:")
    print(f"  Rows: {len(data)}")
    print(f"  Date range: {data.index.min().date()} to {data.index.max().date()}")
    print(f"  Factors: {list(data.columns)}")
    
    if args.stats:
        stats = fetcher.get_factor_stats(data)
        print(f"\nFactor Statistics (Annualized):")
        print(stats[['mean_daily', 'sharpe_annual', 'max_dd']].to_string())


if __name__ == '__main__':
    main()
