"""
Wasserstein HMM Regime Detector (v220 Phase 1)

Implementation of Hidden Markov Model with Wasserstein distance template tracking
for regime detection based on arXiv 2603.04441v1 research.

Research-backed performance: Sharpe 2.18, Max DD -5.43%, turnover 0.0079
"""

import os
import json
import sqlite3
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import sys
from collections import deque
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

_ML_ENABLED = os.environ.get("PORTFOLIO_LAB_ENABLE_ML", "0") == "1"
if _ML_ENABLED:
    try:
        from hmmlearn.hmm import GaussianHMM
        HMM_AVAILABLE = True
    except ImportError:
        HMM_AVAILABLE = False
else:
    HMM_AVAILABLE = False

# Database path
DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
DB_PATH = DATA_DIR / "market.db"


@dataclass
class RegimeState:
    """Represents a detected market regime state"""
    timestamp: str
    regime_label: str  # bull, bear, neutral, crisis
    regime_id: int     # HMM state index
    probability: float
    
    # Feature values that led to this classification
    vix_level: float
    vix_change: float
    yield_spread: float
    momentum_20d: float
    momentum_60d: float
    correlation_proxy: float
    
    # Wasserstein template matching
    template_distance: float
    template_confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass  
class WassersteinTemplate:
    """Template for a regime state with historical distribution"""
    regime_label: str
    mean_vector: np.ndarray
    cov_matrix: np.ndarray
    sample_count: int
    last_updated: str
    
    def wasserstein_distance(self, observations: np.ndarray) -> float:
        """
        Calculate 2-Wasserstein distance between template and observations.
        
        For Gaussian distributions: W2 = ||μ1 - μ2||^2 + Tr(Σ1 + Σ2 - 2√(Σ1Σ2))
        """
        if observations.ndim == 1:
            obs_mean = observations
            obs_cov = np.eye(len(observations)) * 0.01
        else:
            obs_mean = np.mean(observations, axis=0)
            obs_cov = np.cov(observations.T) if observations.shape[0] > 1 else np.eye(len(obs_mean)) * 0.01
        
        # Mean difference term
        mean_diff = np.sum((self.mean_vector - obs_mean) ** 2)
        
        # Covariance term (simplified)
        cov_diff = np.sum(np.diag(self.cov_matrix)) + np.sum(np.diag(obs_cov))
        
        return np.sqrt(mean_diff + cov_diff)


class WassersteinHMMDetector:
    """
    Hidden Markov Model regime detector with Wasserstein template tracking.
    
    Prevents label switching by maintaining stable regime templates based on
    economic fundamentals rather than arbitrary state ordering.
    """
    
    # Feature names for regime detection
    FEATURES = ['vix_level', 'vix_change', 'yield_spread', 
                'momentum_20d', 'momentum_60d', 'hyg_spread_proxy']
    
    # Regime definitions with characteristic feature profiles
    REGIME_TEMPLATES = {
        'bull': {
            'mean': [14.0, 0.0, 1.5, 0.02, 0.08, 3.5],
            'description': 'Low volatility, positive momentum, normal spreads'
        },
        'bear': {
            'mean': [25.0, 0.5, -0.5, -0.02, -0.05, 5.0],
            'description': 'High volatility, negative momentum, flat/inverted curve'
        },
        'neutral': {
            'mean': [18.0, 0.0, 1.0, 0.0, 0.03, 4.0],
            'description': 'Moderate volatility, mixed momentum'
        },
        'crisis': {
            'mean': [35.0, 2.0, -1.0, -0.05, -0.10, 8.0],
            'description': 'Extreme volatility, sharply negative momentum'
        }
    }
    
    def __init__(
        self,
        n_states: int = 4,
        lookback_days: int = 252,
        template_window: int = 63,
        random_state: int = 42
    ):
        self.n_states = n_states
        self.lookback_days = lookback_days
        self.template_window = template_window
        self.random_state = random_state
        
        self.model: Optional[Any] = None
        self.templates: Dict[int, WassersteinTemplate] = {}
        self.regime_history: deque = deque(maxlen=252)
        self.feature_history: deque = deque(maxlen=lookback_days)
        
        self.state_to_regime: Dict[int, str] = {}
        self.regime_to_state: Dict[str, int] = {}
        
        self._last_training_date: Optional[str] = None
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        
    def _fetch_data_from_db(self, symbol: str, days: int) -> List[Dict]:
        """Fetch price data from SQLite database."""
        if not DB_PATH.exists():
            return []
        
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Build date filter directly in SQL (can't use parameter for date arithmetic)
            date_filter = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            cursor.execute("""
                SELECT date, close, volume 
                FROM prices 
                WHERE symbol = ? 
                AND date >= ?
                ORDER BY date
            """, (symbol, date_filter))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [
                {'date': r[0], 'price': r[1], 'close': r[1], 'volume': r[2]}
                for r in rows
            ]
        except Exception as e:
            print(f"DB error fetching {symbol}: {e}")
            return []
    
    def _calculate_features(self) -> List[Dict]:
        """Calculate regime detection features from market data."""
        # Fetch data for all required symbols
        vix_data = self._fetch_data_from_db('^VIX', int(self.lookback_days * 1.5))
        spy_data = self._fetch_data_from_db('SPY', int(self.lookback_days * 1.5))
        hyg_data = self._fetch_data_from_db('HYG', int(self.lookback_days * 1.5))
        tlt_data = self._fetch_data_from_db('TLT', int(self.lookback_days * 1.5))
        
        if len(spy_data) < 60:
            return []
        
        # Build date-indexed structure
        results = []
        
        for i in range(60, len(spy_data)):
            spy_slice = spy_data[i-60:i+1]
            current_price = spy_slice[-1]['price']
            
            # Momentum calculations
            price_20d = spy_slice[-20]['price'] if len(spy_slice) >= 20 else spy_slice[0]['price']
            price_60d = spy_slice[-60]['price'] if len(spy_slice) >= 60 else spy_slice[0]['price']
            
            mom_20d = (current_price - price_20d) / price_20d if price_20d > 0 else 0
            mom_60d = (current_price - price_60d) / price_60d if price_60d > 0 else 0
            
            # VIX features - find by date
            date = spy_slice[-1]['date']
            vix_level = 20.0
            vix_change = 0.0
            
            for j, v in enumerate(vix_data):
                if v['date'] == date:
                    vix_level = v['price']
                    # Find VIX 5 days ago
                    if j >= 5:
                        vix_change = v['price'] - vix_data[j-5]['price']
                    break
            
            # Yield spread proxy from TLT (inverse relationship)
            yield_spread = 1.5
            for j, t in enumerate(tlt_data):
                if t['date'] == date and j >= 20:
                    tlt_current = t['price']
                    tlt_20d = tlt_data[j-20]['price']
                    tlt_change = (tlt_current - tlt_20d) / tlt_20d
                    yield_spread = max(-2.0, min(5.0, 3.0 - (tlt_change * 100)))
                    break
            
            # Credit spread proxy from HYG
            hyg_spread = 3.5
            for j, h in enumerate(hyg_data):
                if h['date'] == date and j >= 20:
                    hyg_current = h['price']
                    hyg_20d = hyg_data[j-20]['price']
                    hyg_change = (hyg_current - hyg_20d) / hyg_20d
                    hyg_spread = max(1.0, min(15.0, 3.5 + (-hyg_change * 100)))
                    break
            
            results.append({
                'date': date,
                'vix_level': vix_level,
                'vix_change': vix_change,
                'yield_spread': yield_spread,
                'momentum_20d': mom_20d,
                'momentum_60d': mom_60d,
                'hyg_spread_proxy': hyg_spread
            })
        
        return results
    
    def _prepare_features(self, data: List[Dict]) -> np.ndarray:
        """Prepare feature matrix for HMM training."""
        feature_cols = self.FEATURES
        X = np.array([[d.get(f, 0.0) for f in feature_cols] for d in data])
        
        # Standardize features
        self.feature_means = np.mean(X, axis=0)
        self.feature_stds = np.std(X, axis=0)
        self.feature_stds[self.feature_stds == 0] = 1
        
        X_scaled = (X - self.feature_means) / self.feature_stds
        return X_scaled
    
    def _initialize_templates(self, X: np.ndarray, hidden_states: np.ndarray):
        """Initialize Wasserstein templates from HMM states."""
        for state_id in range(self.n_states):
            mask = hidden_states == state_id
            if mask.sum() < 10:
                continue
            
            state_data = X[mask]
            
            mean_vec = np.mean(state_data, axis=0)
            cov_mat = np.cov(state_data.T) if state_data.shape[0] > 1 else np.eye(len(mean_vec)) * 0.1
            
            best_regime = self._match_to_template(mean_vec)
            
            self.templates[state_id] = WassersteinTemplate(
                regime_label=best_regime,
                mean_vector=mean_vec,
                cov_matrix=cov_mat,
                sample_count=int(mask.sum()),
                last_updated=datetime.now().isoformat()
            )
            
            self.state_to_regime[state_id] = best_regime
            self.regime_to_state[best_regime] = state_id
    
    def _match_to_template(self, mean_vector: np.ndarray) -> str:
        """Match feature mean vector to closest regime template."""
        best_regime = 'neutral'
        best_distance = float('inf')
        
        for regime_name, template in self.REGIME_TEMPLATES.items():
            template_mean = np.array(template['mean'])
            if self.feature_means is not None:
                template_scaled = (template_mean - self.feature_means) / self.feature_stds
            else:
                template_scaled = template_mean
            
            distance = np.sum((mean_vector - template_scaled) ** 2)
            
            if distance < best_distance:
                best_distance = distance
                best_regime = regime_name
        
        return best_regime
    
    def fit(self, force_retrain: bool = False) -> bool:
        """Train HMM on historical data."""
        if not HMM_AVAILABLE:
            return False
        
        # Check if we need to retrain
        if self._last_training_date and not force_retrain:
            last_train = datetime.fromisoformat(self._last_training_date)
            if datetime.now() - last_train < timedelta(days=7):
                return True
        
        # Fetch features
        data = self._calculate_features()
        if len(data) < 126:
            return False
        
        # Prepare features
        X = self._prepare_features(data)
        
        # Train HMM
        try:
            self.model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",
                n_iter=100,
                random_state=self.random_state,
                init_params='st'
            )
            
            self.model.fit(X)
            
            hidden_states = self.model.predict(X)
            self._initialize_templates(X, hidden_states)
            
            self._last_training_date = datetime.now().isoformat()
            
            print(f"HMM trained on {len(X)} observations")
            print(f"Regime mapping: {self.state_to_regime}")
            
            return True
            
        except Exception as e:
            print(f"HMM training failed: {e}")
            return False
    
    def detect_current_regime(self) -> Optional[RegimeState]:
        """Detect current market regime using HMM + Wasserstein templates."""
        if self.model is None:
            success = self.fit()
            if not success:
                return self._fallback_regime_detection()
        
        # Get current features
        data = self._calculate_features()
        if len(data) == 0:
            return self._fallback_regime_detection()
        
        latest = data[-1]
        
        # Prepare feature vector
        feature_values = [latest.get(f, 0.0) for f in self.FEATURES]
        X_current = np.array([feature_values])
        
        # Scale features
        if self.feature_means is not None:
            X_scaled = (X_current - self.feature_means) / self.feature_stds
        else:
            X_scaled = X_current
        
        # Predict regime
        if HMM_AVAILABLE and self.model:
            state_probs = self.model.predict_proba(X_scaled)[0]
            predicted_state = int(np.argmax(state_probs))
            confidence = float(state_probs[predicted_state])
            
            regime_label = self.state_to_regime.get(predicted_state, 'neutral')
            
            if predicted_state in self.templates:
                template = self.templates[predicted_state]
                w_dist = template.wasserstein_distance(X_scaled[0])
                template_conf = 1.0 / (1.0 + w_dist)
            else:
                w_dist = 0.0
                template_conf = 0.5
            
        else:
            regime_label, confidence = self._rule_based_regime(latest)
            predicted_state = -1
            w_dist = 0.0
            template_conf = confidence
        
        regime = RegimeState(
            timestamp=datetime.now().isoformat(),
            regime_label=regime_label,
            regime_id=predicted_state,
            probability=confidence,
            vix_level=float(latest.get('vix_level', 20.0)),
            vix_change=float(latest.get('vix_change', 0.0)),
            yield_spread=float(latest.get('yield_spread', 1.5)),
            momentum_20d=float(latest.get('momentum_20d', 0.0)),
            momentum_60d=float(latest.get('momentum_60d', 0.0)),
            correlation_proxy=float(latest.get('hyg_spread_proxy', 3.5)),
            template_distance=w_dist,
            template_confidence=template_conf
        )
        
        self.regime_history.append(regime)
        return regime
    
    def _rule_based_regime(self, features: Dict) -> Tuple[str, float]:
        """Fallback rule-based regime detection."""
        vix = features.get('vix_level', 20.0)
        mom20 = features.get('momentum_20d', 0.0)
        mom60 = features.get('momentum_60d', 0.0)
        
        if vix > 30 and mom20 < -0.03:
            return 'crisis', 0.8
        elif vix > 25 or (mom20 < -0.02 and mom60 < -0.05):
            return 'bear', 0.7
        elif vix < 16 and mom60 > 0.05:
            return 'bull', 0.75
        else:
            return 'neutral', 0.6
    
    def _fallback_regime_detection(self) -> RegimeState:
        """Return neutral regime when detection fails."""
        return RegimeState(
            timestamp=datetime.now().isoformat(),
            regime_label='neutral',
            regime_id=-1,
            probability=0.5,
            vix_level=20.0,
            vix_change=0.0,
            yield_spread=1.5,
            momentum_20d=0.0,
            momentum_60d=0.0,
            correlation_proxy=3.5,
            template_distance=0.0,
            template_confidence=0.5
        )
    
    def get_regime_stats(self) -> Dict[str, Any]:
        """Get statistics about regime history."""
        if not self.regime_history:
            return {
                'total_detections': 0,
                'regime_distribution': {},
                'average_confidence': 0.0
            }
        
        regimes = [r.regime_label for r in self.regime_history]
        confidences = [r.probability for r in self.regime_history]
        
        from collections import Counter
        distribution = Counter(regimes)
        
        return {
            'total_detections': len(regimes),
            'regime_distribution': dict(distribution),
            'average_confidence': float(np.mean(confidences)),
            'current_regime': regimes[-1] if regimes else 'unknown',
            'last_detection': self.regime_history[-1].timestamp if self.regime_history else None
        }
    
    def save_state(self, filepath: Optional[str] = None):
        """Save detector state to file."""
        if filepath is None:
            filepath = project_root / 'data' / 'hmm_state.json'
        
        state = {
            'n_states': self.n_states,
            'last_training': self._last_training_date,
            'regime_mapping': self.state_to_regime,
            'templates': {
                str(k): {
                    'regime_label': v.regime_label,
                    'sample_count': v.sample_count,
                    'last_updated': v.last_updated
                }
                for k, v in self.templates.items()
            },
            'regime_stats': self.get_regime_stats(),
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"HMM state saved to {filepath}")
    
    def load_state(self, filepath: Optional[str] = None):
        """Load detector state from file."""
        if filepath is None:
            filepath = project_root / 'data' / 'hmm_state.json'
        
        if not Path(filepath).exists():
            return False
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.n_states = state.get('n_states', 4)
            self._last_training_date = state.get('last_training')
            self.state_to_regime = {int(k): v for k, v in state.get('regime_mapping', {}).items()}
            self.regime_to_state = {v: int(k) for k, v in self.state_to_regime.items()}
            
            return True
            
        except Exception as e:
            print(f"Failed to load HMM state: {e}")
            return False


class HMMRegimeCLI:
    """Command-line interface for HMM regime detection."""
    
    def __init__(self):
        self.detector = WassersteinHMMDetector()
    
    def status(self):
        """Show current regime status."""
        self.detector.load_state()
        regime = self.detector.detect_current_regime()
        
        if regime:
            output = {
                'timestamp': regime.timestamp,
                'regime': {
                    'label': regime.regime_label,
                    'id': regime.regime_id,
                    'confidence': f"{regime.probability:.2%}",
                    'template_confidence': f"{regime.template_confidence:.2%}"
                },
                'features': {
                    'vix_level': round(regime.vix_level, 2),
                    'vix_change_5d': round(regime.vix_change, 2),
                    'yield_spread': round(regime.yield_spread, 2),
                    'momentum_20d': f"{regime.momentum_20d:.2%}",
                    'momentum_60d': f"{regime.momentum_60d:.2%}",
                    'credit_proxy': round(regime.correlation_proxy, 2)
                },
                'wasserstein': {
                    'template_distance': round(regime.template_distance, 4),
                    'label_stability': 'stable' if regime.template_confidence > 0.7 else 'uncertain'
                },
                'stats': self.detector.get_regime_stats()
            }
            print(json.dumps(output, indent=2))
        else:
            print(json.dumps({'error': 'Failed to detect regime'}))
    
    def history(self, days: int = 30):
        """Show regime history."""
        stats = self.detector.get_regime_stats()
        print(json.dumps(stats, indent=2))
    
    def predict(self, horizon_days: int = 5):
        """Predict regime transition probabilities."""
        self.detector.fit()
        current = self.detector.detect_current_regime()
        
        if not current or not self.detector.model:
            print(json.dumps({'error': 'Cannot predict without trained model'}))
            return
        
        transmat = self.detector.model.transmat_
        current_state = current.regime_id
        if current_state < 0:
            print(json.dumps({'error': 'Invalid state for prediction'}))
            return
        
        prob_vector = np.zeros(self.detector.n_states)
        prob_vector[current_state] = 1.0
        
        for _ in range(horizon_days):
            prob_vector = prob_vector @ transmat
        
        predictions = {}
        for state_id, prob in enumerate(prob_vector):
            regime = self.detector.state_to_regime.get(state_id, f'state_{state_id}')
            predictions[regime] = f"{prob:.2%}"
        
        output = {
            'current_regime': current.regime_label,
            'horizon_days': horizon_days,
            'predicted_distribution': predictions,
            'most_likely': max(predictions, key=lambda k: float(predictions[k].rstrip('%'))),
            'transition_matrix': transmat.tolist() if hasattr(transmat, 'tolist') else str(transmat)
        }
        
        print(json.dumps(output, indent=2))
    
    def train(self):
        """Force retrain HMM model."""
        success = self.detector.fit(force_retrain=True)
        if success:
            self.detector.save_state()
            print(json.dumps({'status': 'training_complete', 'success': True}))
        else:
            print(json.dumps({'status': 'training_failed', 'success': False}))


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Wasserstein HMM Regime Detector')
    parser.add_argument('command', choices=['status', 'history', 'predict', 'train'],
                       help='Command to execute')
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument('--horizon', type=int, default=5)
    
    args = parser.parse_args()
    
    cli = HMMRegimeCLI()
    
    if args.command == 'status':
        cli.status()
    elif args.command == 'history':
        cli.history(args.days)
    elif args.command == 'predict':
        cli.predict(args.horizon)
    elif args.command == 'train':
        cli.train()


if __name__ == '__main__':
    main()
