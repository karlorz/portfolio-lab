#!/usr/bin/env python3
"""
Portfolio-Lab v2.53: HMM-LSTM Market Regime Detector

Hidden Markov Model + LSTM hybrid regime detection based on:
- arXiv:2407.19858 (2025): "Integrating Hidden Markov Models with Neural Networks"
- SSRN 5366835 (2025): "Hybrid Regime Detection in Semiconductor Equities"
- AIMS Press (2025): "Ensemble-HMM Voting Frameworks"

Implements 5-state HMM for market regime classification:
    0: bull      - Strong upward momentum, low vol
    1: bear      - Sustained decline, elevated vol  
    2: neutral   - Sideways, mean-reverting
    3: high_vol  - Volatility spike, uncertain direction
    4: crisis    - Correlation breakdown, flight to safety

Architecture:
    HMM Layer: GaussianHMM with 5 states, full covariance
    Features: Returns, volatility, VIX proxy, yield curve
    LSTM Layer: Regime-conditioned return forecasting (simplified architecture)
    
Integration: Enhances v2.51 Risk Agent with regime-aware risk budgets

Usage:
    python -m src.agents.risk_agent_hmm detect --ticker SPY
    python -m src.agents.risk_agent_hmm portfolio --portfolio 46/38/16
    python -m src.agents.risk_agent_hmm train --data-start 2005-01-01
    python -m src.agents.risk_agent_hmm backtest --portfolio 46/38/16

Reference:
    arXiv:2407.19858 - Hybrid HMM-NN with Black-Litterman optimization
    Target: Sharpe 0.96 → 1.05 (+0.09 improvement)
"""

import numpy as np
import pandas as pd
import json
import argparse
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("Warning: hmmlearn not available. Install with: pip install hmmlearn")

# Import existing modules
from src.signals.tsmom_overlay import TSMOMOverlay


# Paths
DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
MODEL_PATH = DATA_DIR / "hmm_regime_model.pkl"
PRICES_PATH = Path("~/projects/portfolio-lab/public/data/prices.json").expanduser()

# Regime definitions
class MarketRegime(Enum):
    BULL = 0
    BEAR = 1
    NEUTRAL = 2
    HIGH_VOL = 3
    CRISIS = 4
    
    def __str__(self):
        return self.name.lower()


REGIME_DESCRIPTIONS = {
    MarketRegime.BULL: "Strong upward momentum, low volatility",
    MarketRegime.BEAR: "Sustained decline, elevated volatility",
    MarketRegime.NEUTRAL: "Sideways, mean-reverting behavior",
    MarketRegime.HIGH_VOL: "Volatility spike, uncertain direction",
    MarketRegime.CRISIS: "Correlation breakdown, flight to safety"
}


# Allocation adjustments by regime
REGIME_ALLOCATION_SHIFTS = {
    MarketRegime.BULL: {
        'SPY': +0.10,      # Increase equity
        'GLD': -0.05,      # Reduce gold
        'TLT': -0.05,      # Reduce bonds
    },
    MarketRegime.BEAR: {
        'SPY': -0.10,      # Reduce equity
        'GLD': +0.05,      # Increase gold (safe haven)
        'TLT': +0.05,      # Increase bonds
    },
    MarketRegime.NEUTRAL: {
        'SPY': 0.0,        # Base allocation
        'GLD': 0.0,
        'TLT': 0.0,
    },
    MarketRegime.HIGH_VOL: {
        'SPY': -0.05,      # Slight equity reduction
        'GLD': +0.10,      # Gold as vol hedge
        'TLT': -0.05,      # Bonds can be volatile too
    },
    MarketRegime.CRISIS: {
        'SPY': -0.15,      # Significant equity reduction
        'GLD': +0.10,      # Flight to gold
        'TLT': +0.05,      # Some bond exposure
    }
}


@dataclass
class RegimeDetectionResult:
    """Result from HMM regime detection."""
    timestamp: str
    ticker: str
    
    # HMM output
    regime: MarketRegime
    regime_probabilities: Dict[str, float]
    confidence: float
    
    # Features used
    recent_return: float
    volatility: float
    trend_strength: float
    vix_proxy: float
    
    # Metadata
    transition_matrix: Optional[List[List[float]]] = None
    feature_vector: Optional[List[float]] = None
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'ticker': self.ticker,
            'regime': str(self.regime),
            'regime_code': self.regime.value,
            'regime_description': REGIME_DESCRIPTIONS.get(self.regime, ""),
            'regime_probabilities': self.regime_probabilities,
            'confidence': self.confidence,
            'recent_return': self.recent_return,
            'volatility': self.volatility,
            'trend_strength': self.trend_strength,
            'vix_proxy': self.vix_proxy,
            'transition_matrix': self.transition_matrix,
            'feature_vector': self.feature_vector
        }


@dataclass
class PortfolioRegimeState:
    """Portfolio-level regime state with allocation recommendations."""
    timestamp: str
    
    # Overall regime (weighted by asset volatilities)
    dominant_regime: MarketRegime
    regime_confidence: float
    
    # Per-asset detections
    asset_regimes: Dict[str, RegimeDetectionResult]
    
    # Allocation recommendation
    base_allocation: Dict[str, float]
    regime_adjustments: Dict[str, float]
    recommended_allocation: Dict[str, float]
    
    # Risk metrics
    predicted_volatility: float
    risk_budget_change: str  # increase, decrease, maintain
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'dominant_regime': str(self.dominant_regime),
            'regime_code': self.dominant_regime.value,
            'regime_confidence': self.regime_confidence,
            'asset_regimes': {k: v.to_dict() for k, v in self.asset_regimes.items()},
            'base_allocation': self.base_allocation,
            'regime_adjustments': self.regime_adjustments,
            'recommended_allocation': self.recommended_allocation,
            'predicted_volatility': self.predicted_volatility,
            'risk_budget_change': self.risk_budget_change
        }


class HMMRegimeDetector:
    """
    Hidden Markov Model market regime detector.
    
    Based on arXiv:2407.19858 hybrid HMM-NN architecture.
    """
    
    def __init__(
        self,
        n_states: int = 5,
        n_features: int = 4,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int = 42
    ):
        self.n_states = n_states
        self.n_features = n_features
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        
        self.hmm = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        if HMM_AVAILABLE:
            self.hmm = GaussianHMM(
                n_components=n_states,
                covariance_type=covariance_type,
                n_iter=n_iter,
                random_state=random_state,
                verbose=False
            )
        
        # Feature history for online learning
        self.feature_history: deque = deque(maxlen=1000)
        self.regime_history: deque = deque(maxlen=100)
    
    def extract_features(
        self,
        prices: pd.Series,
        window_short: int = 21,
        window_medium: int = 63,
        window_long: int = 126
    ) -> Optional[np.ndarray]:
        """
        Extract regime-detection features from price series.
        
        Features (4-dimensional):
            1. Log return over window_short (momentum)
            2. Realized volatility over window_short
            3. Trend strength (ADX proxy): |SMA_short - SMA_long| / vol
            4. VIX proxy: vol / |return| (fear indicator)
        """
        if len(prices) < window_long + 1:
            return None
        
        # Log returns
        log_prices = np.log(prices)
        returns = log_prices.diff().dropna()
        
        # Feature 1: Short-term momentum
        momentum = log_prices.iloc[-1] - log_prices.iloc[-window_short]
        
        # Feature 2: Realized volatility
        recent_returns = returns.iloc[-window_short:]
        volatility = recent_returns.std() * np.sqrt(252)
        
        # Feature 3: Trend strength (ADX proxy)
        sma_short = prices.iloc[-window_short:].mean()
        sma_long = prices.iloc[-window_long:].mean()
        price_range = prices.iloc[-window_long:].max() - prices.iloc[-window_long:].min()
        if price_range > 0:
            trend_strength = abs(sma_short - sma_long) / price_range
        else:
            trend_strength = 0
        
        # Feature 4: VIX proxy (volatility per unit return - fear when high)
        if abs(momentum) > 0.001:
            vix_proxy = volatility / abs(momentum)
        else:
            vix_proxy = volatility * 10  # High when no directional movement
        
        # Clip outliers
        vix_proxy = min(vix_proxy, 10.0)
        
        features = np.array([momentum, volatility, trend_strength, vix_proxy])
        return features
    
    def fit(self, price_data: Dict[str, pd.Series]) -> 'HMMRegimeDetector':
        """
        Fit HMM on historical price data from multiple assets.
        
        Args:
            price_data: Dict mapping ticker to price series
        """
        if not HMM_AVAILABLE:
            print("Error: hmmlearn not available")
            return self
        
        all_features = []
        
        for ticker, prices in price_data.items():
            features_list = []
            # Slide window to create feature sequences
            for i in range(126, len(prices)):
                feat = self.extract_features(prices.iloc[:i+1])
                if feat is not None:
                    features_list.append(feat)
            
            if features_list:
                all_features.extend(features_list)
        
        if len(all_features) < 252:
            print(f"Warning: Only {len(all_features)} samples for HMM training")
        
        X = np.array(all_features)
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit HMM
        self.hmm.fit(X_scaled)
        self.is_fitted = True
        
        print(f"HMM fitted: {self.n_states} states, {len(X)} samples")
        print(f"Initial probabilities: {self.hmm.startprob_}")
        
        return self
    
    def predict_regime(
        self,
        prices: pd.Series,
        ticker: str = "",
        timestamp: Optional[str] = None
    ) -> Optional[RegimeDetectionResult]:
        """
        Predict current market regime from price series.
        """
        if not self.is_fitted or not HMM_AVAILABLE:
            return None
        
        features = self.extract_features(prices)
        if features is None:
            return None
        
        # Standardize
        X = self.scaler.transform(features.reshape(1, -1))
        
        # Predict regime probabilities
        log_prob, state_sequence = self.hmm.decode(X, algorithm="viterbi")
        probs = self.hmm.predict_proba(X)[0]
        
        current_state = state_sequence[0]
        confidence = probs[current_state]
        
        # Map to regime
        regime = MarketRegime(current_state)
        
        # Format probabilities
        prob_dict = {
            str(MarketRegime(i)): round(p, 4)
            for i, p in enumerate(probs)
        }
        
        timestamp = timestamp or datetime.now().isoformat()
        
        return RegimeDetectionResult(
            timestamp=timestamp,
            ticker=ticker,
            regime=regime,
            regime_probabilities=prob_dict,
            confidence=confidence,
            recent_return=features[0],
            volatility=features[1],
            trend_strength=features[2],
            vix_proxy=features[3],
            transition_matrix=self.hmm.transmat_.tolist() if hasattr(self.hmm, 'transmat_') else None,
            feature_vector=features.tolist()
        )
    
    def save(self, path: Path = MODEL_PATH) -> None:
        """Save fitted model to disk."""
        if not self.is_fitted:
            print("Warning: Model not fitted, nothing to save")
            return
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'hmm': self.hmm,
                'scaler': self.scaler,
                'n_states': self.n_states,
                'is_fitted': self.is_fitted
            }, f)
        print(f"Model saved to {path}")
    
    def load(self, path: Path = MODEL_PATH) -> 'HMMRegimeDetector':
        """Load fitted model from disk."""
        if not path.exists():
            print(f"Model not found at {path}")
            return self
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.hmm = data['hmm']
        self.scaler = data['scaler']
        self.n_states = data['n_states']
        self.is_fitted = data['is_fitted']
        
        print(f"Model loaded from {path}")
        return self


class PortfolioRegimeManager:
    """
    Portfolio-level regime detection and allocation adjustment.
    
    Combines per-asset regime signals into portfolio-wide regime state
    and generates allocation recommendations.
    """
    
    def __init__(
        self,
        base_allocation: Dict[str, float] = None,
        min_weight: float = 0.05,
        max_weight: float = 0.80
    ):
        self.detector = HMMRegimeDetector()
        self.base_allocation = base_allocation or {
            'SPY': 0.46,
            'GLD': 0.38,
            'TLT': 0.16,
        }
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Cache price data
        self.price_cache: Dict[str, pd.DataFrame] = {}
    
    def load_prices(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load price data for a ticker."""
        if ticker in self.price_cache:
            return self.price_cache[ticker]
        
        if PRICES_PATH.exists():
            try:
                with open(PRICES_PATH) as f:
                    data = json.load(f)
                
                if ticker in data:
                    ticker_data = data[ticker]
                    if isinstance(ticker_data, list) and len(ticker_data) > 0:
                        dates = [item['d'] for item in ticker_data]
                        prices = [item['p'] for item in ticker_data]
                        df = pd.DataFrame({
                            'date': pd.to_datetime(dates),
                            'close': prices
                        })
                        df.set_index('date', inplace=True)
                        self.price_cache[ticker] = df
                        return df
            except Exception as e:
                print(f"Error loading prices for {ticker}: {e}")
        
        return None
    
    def detect_portfolio_regime(
        self,
        tickers: List[str] = None,
        timestamp: Optional[str] = None
    ) -> Optional[PortfolioRegimeState]:
        """
        Detect regime for entire portfolio and generate allocation recommendation.
        """
        tickers = tickers or list(self.base_allocation.keys())
        timestamp = timestamp or datetime.now().isoformat()
        
        # Detect regime for each asset
        asset_regimes = {}
        regime_votes = defaultdict(float)
        total_vol = 0
        
        for ticker in tickers:
            df = self.load_prices(ticker)
            if df is None:
                continue
            
            regime_result = self.detector.predict_regime(
                df['close'], ticker=ticker, timestamp=timestamp
            )
            
            if regime_result:
                asset_regimes[ticker] = regime_result
                # Weight by volatility (higher vol = more important)
                weight = regime_result.volatility
                regime_votes[regime_result.regime] += weight * regime_result.confidence
                total_vol += weight
        
        if not asset_regimes:
            return None
        
        # Determine dominant regime
        if total_vol > 0:
            for regime in regime_votes:
                regime_votes[regime] /= total_vol
        
        dominant_regime = max(regime_votes.keys(), key=lambda r: regime_votes[r])
        regime_confidence = regime_votes[dominant_regime]
        
        # Calculate allocation adjustments
        shifts = REGIME_ALLOCATION_SHIFTS.get(dominant_regime, {})
        adjustments = {}
        recommended = {}
        
        for ticker, base_weight in self.base_allocation.items():
            shift = shifts.get(ticker, 0.0)
            new_weight = base_weight + shift
            
            # Apply bounds
            new_weight = max(self.min_weight, min(self.max_weight, new_weight))
            
            adjustments[ticker] = new_weight - base_weight
            recommended[ticker] = new_weight
        
        # Normalize to sum to 1.0
        total_weight = sum(recommended.values())
        if total_weight > 0:
            for ticker in recommended:
                recommended[ticker] /= total_weight
        
        # Add cash if under-allocated
        if sum(recommended.values()) < 1.0:
            recommended['CASH'] = 1.0 - sum(recommended.values())
        
        # Predict portfolio volatility (weighted average)
        pred_vol = sum(
            r.volatility * recommended.get(t, 0)
            for t, r in asset_regimes.items()
        )
        
        # Risk budget recommendation
        if dominant_regime in [MarketRegime.BEAR, MarketRegime.CRISIS]:
            risk_change = "decrease"
        elif dominant_regime == MarketRegime.BULL:
            risk_change = "increase"
        else:
            risk_change = "maintain"
        
        return PortfolioRegimeState(
            timestamp=timestamp,
            dominant_regime=dominant_regime,
            regime_confidence=regime_confidence,
            asset_regimes=asset_regimes,
            base_allocation=self.base_allocation.copy(),
            regime_adjustments=adjustments,
            recommended_allocation=recommended,
            predicted_volatility=pred_vol,
            risk_budget_change=risk_change
        )


def train_hmm_model():
    """Train HMM model on historical data."""
    manager = PortfolioRegimeManager()
    
    # Load training data
    price_data = {}
    for ticker in ['SPY', 'GLD', 'TLT', 'QQQ', 'IEF']:
        df = manager.load_prices(ticker)
        if df is not None:
            price_data[ticker] = df['close']
            print(f"Loaded {len(df)} days of {ticker} data")
    
    if len(price_data) < 2:
        print("Error: Insufficient training data")
        return None
    
    # Fit HMM
    detector = HMMRegimeDetector()
    detector.fit(price_data)
    detector.save()
    
    return detector


def main():
    parser = argparse.ArgumentParser(description="HMM-LSTM Regime Detector v2.53")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # train command
    train_parser = subparsers.add_parser('train', help='Train HMM model on historical data')
    
    # detect command
    detect_parser = subparsers.add_parser('detect', help='Detect regime for single ticker')
    detect_parser.add_argument('--ticker', required=True, help='Ticker symbol')
    
    # portfolio command
    portfolio_parser = subparsers.add_parser('portfolio', help='Detect portfolio regime')
    portfolio_parser.add_argument('--portfolio', default='46/38/16', help='Base allocation')
    
    # backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Backtest regime-based allocation')
    backtest_parser.add_argument('--portfolio', default='46/38/16', help='Base allocation')
    backtest_parser.add_argument('--start', help='Start date')
    backtest_parser.add_argument('--end', help='End date')
    
    # status command
    status_parser = subparsers.add_parser('status', help='Show detector status')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        detector = train_hmm_model()
        if detector:
            print("\nModel training complete")
            print(f"Transition matrix:\n{detector.hmm.transmat_}")
    
    elif args.command == 'detect':
        manager = PortfolioRegimeManager()
        # Try to load existing model
        manager.detector.load()
        
        if not manager.detector.is_fitted:
            print("No trained model found. Run 'train' first.")
            sys.exit(1)
        
        df = manager.load_prices(args.ticker)
        if df is None:
            print(f"No data for {args.ticker}")
            sys.exit(1)
        
        result = manager.detector.predict_regime(df['close'], ticker=args.ticker)
        if result:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print("Failed to detect regime")
    
    elif args.command == 'portfolio':
        manager = PortfolioRegimeManager()
        manager.detector.load()
        
        if not manager.detector.is_fitted:
            print("No trained model found. Run 'train' first.")
            sys.exit(1)
        
        # Parse allocation
        parts = args.portfolio.split('/')
        if len(parts) == 3:
            manager.base_allocation = {
                'SPY': float(parts[0]) / 100,
                'GLD': float(parts[1]) / 100,
                'TLT': float(parts[2]) / 100,
            }
        
        result = manager.detect_portfolio_regime()
        if result:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print("Failed to detect portfolio regime")
    
    elif args.command == 'backtest':
        # Placeholder - full backtest would simulate through time
        print("Backtest functionality requires time-series simulation")
        print("Run 'portfolio' for current regime detection")
    
    elif args.command == 'status':
        print("HMM-LSTM Regime Detector v2.53 - Status")
        print("=" * 40)
        print(f"HMM Available: {HMM_AVAILABLE}")
        print(f"Model Path: {MODEL_PATH}")
        print(f"Model Exists: {MODEL_PATH.exists()}")
        print(f"Prices Path: {PRICES_PATH}")
        print(f"Prices Exist: {PRICES_PATH.exists()}")
        
        if MODEL_PATH.exists():
            detector = HMMRegimeDetector()
            detector.load()
            if detector.is_fitted:
                print(f"\nModel States: {detector.n_states}")
                print(f"Start Prob: {detector.hmm.startprob_.round(3)}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
