"""
Regime classifier using ML to predict market regimes.
Simple model (logistic regression/XGBoost) for bull/neutral/bear classification.
"""
import os
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

# ML-gated imports — prevented during test collection by conftest.py's
# import hook. Only loaded when PORTFOLIO_LAB_ENABLE_ML=1 AND libs are installed.
_ML_ENABLED = os.environ.get("PORTFOLIO_LAB_ENABLE_ML", "0") == "1"
if _ML_ENABLED:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, accuracy_score
        SKLEARN_AVAILABLE = True
    except ImportError:
        SKLEARN_AVAILABLE = False

    try:
        import xgboost as xgb
        XGBOOST_AVAILABLE = True
    except ImportError:
        XGBOOST_AVAILABLE = False
else:
    SKLEARN_AVAILABLE = False
    XGBOOST_AVAILABLE = False


class Regime(Enum):
    BEAR = 0
    NEUTRAL = 1
    BULL = 2


@dataclass
class RegimePrediction:
    """Prediction output from regime classifier."""
    symbol: str
    timestamp: str
    
    # Probabilities
    p_bear: float
    p_neutral: float
    p_bull: float
    
    # Prediction
    predicted_regime: Regime
    confidence: float
    
    # Feature importance (if available)
    feature_importance: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "p_bear": self.p_bear,
            "p_neutral": self.p_neutral,
            "p_bull": self.p_bull,
            "predicted_regime": self.predicted_regime.name,
            "confidence": self.confidence,
            "feature_importance": self.feature_importance,
        }


class RegimeClassifier:
    """
    Market regime classifier using historical features.
    
    Trains on labeled historical data to predict future market regimes
    (bull/neutral/bear) based on technical and VIX features.
    """
    
    MODEL_FILE = "data/regime_model.json"
    
    def __init__(self, model_type: str = "logistic"):
        """
        Initialize classifier.
        
        Args:
            model_type: 'logistic', 'random_forest', or 'xgboost'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_trained = False
        
    def _get_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert features dict to numpy array."""
        feature_keys = [
            "return_1d", "return_5d", "return_20d",
            "volatility_20d", "price_vs_sma20", "price_vs_sma50",
            "volume_ratio", "vix_level", "vix_change_5d",
            "vix_percentile_20d", "spy_correlation_20d",
            "trend_direction",
        ]
        
        vec = []
        for key in feature_keys:
            val = features.get(key, 0.0)
            if isinstance(val, str):
                # Handle categorical
                if key == "vol_regime":
                    val = 0 if val == "low" else 1 if val == "normal" else 2
                else:
                    val = 0.0
            vec.append(float(val))
        
        return np.array(vec)
    
    def prepare_data(
        self, 
        features_list: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature matrix and labels from feature records.
        
        Returns:
            X: Feature matrix
            y: Labels (0=bear, 1=neutral, 2=bull)
        """
        X = []
        y = []
        
        for features in features_list:
            if features.get("regime_label") is not None:
                X.append(self._get_feature_vector(features))
                y.append(int(features["regime_label"]))
        
        if len(X) < 10:
            raise ValueError(f"Insufficient training data: {len(X)} samples")
        
        return np.array(X), np.array(y)
    
    def train(self, features_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train the classifier on historical features.
        
        Args:
            features_list: List of feature dicts with 'regime_label' field
            
        Returns:
            Training metrics
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required. Install: pip install scikit-learn")
        
        # Prepare data
        X, y = self.prepare_data(features_list)
        
        # Split for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize model
        if self.model_type == "logistic":
            self.model = LogisticRegression(
                multi_class='multinomial',
                max_iter=1000,
                random_state=42
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ImportError("xgboost required. Install: pip install xgboost")
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(
                ["return_1d", "return_5d", "return_20d", "volatility_20d",
                 "price_vs_sma20", "price_vs_sma50", "volume_ratio",
                 "vix_level", "vix_change_5d", "vix_percentile_20d",
                 "spy_correlation_20d", "trend_direction"],
                self.model.feature_importances_.tolist()
            ))
        elif hasattr(self.model, 'coef_'):
            # Logistic regression - use absolute coefficients
            feature_importance = dict(zip(
                ["return_1d", "return_5d", "return_20d", "volatility_20d",
                 "price_vs_sma20", "price_vs_sma50", "volume_ratio",
                 "vix_level", "vix_change_5d", "vix_percentile_20d",
                 "spy_correlation_20d", "trend_direction"],
                np.abs(self.model.coef_).mean(axis=0).tolist()
            ))
        
        metrics = {
            "accuracy": accuracy,
            "samples_train": len(y_train),
            "samples_test": len(y_test),
            "class_distribution": {
                "bear": int(sum(y == 0)),
                "neutral": int(sum(y == 1)),
                "bull": int(sum(y == 2)),
            },
            "feature_importance": feature_importance,
        }
        
        return metrics
    
    def predict(self, features: Dict[str, Any]) -> RegimePrediction:
        """
        Predict regime for given features.
        
        Args:
            features: Feature dict from FeaturePipeline
            
        Returns:
            RegimePrediction with probabilities
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Convert to vector
        X = self._get_feature_vector(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities
        probs = self.model.predict_proba(X_scaled)[0]
        
        # Determine regime
        pred_class = np.argmax(probs)
        confidence = probs[pred_class]
        
        regime = Regime(pred_class)
        
        return RegimePrediction(
            symbol=features.get("symbol", "UNKNOWN"),
            timestamp=features.get("timestamp", datetime.now().isoformat()),
            p_bear=float(probs[0]),
            p_neutral=float(probs[1]),
            p_bull=float(probs[2]),
            predicted_regime=regime,
            confidence=float(confidence),
            feature_importance=None,  # Can add if needed
        )
    
    def save(self, filepath: Optional[str] = None):
        """Save model to disk."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        filepath = filepath or self.MODEL_FILE
        
        # Serialize model
        import pickle
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        data = {
            "model": pickle.dumps(self.model),
            "scaler": pickle.dumps(self.scaler) if self.scaler else None,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "is_trained": self.is_trained,
            "saved_at": datetime.now().isoformat(),
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        
        return filepath
    
    def load(self, filepath: Optional[str] = None):
        """Load model from disk."""
        filepath = filepath or self.MODEL_FILE
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        import pickle
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        self.model = pickle.loads(data["model"])
        self.scaler = pickle.loads(data["scaler"]) if data["scaler"] else None
        self.model_type = data["model_type"]
        self.feature_names = data.get("feature_names", [])
        self.is_trained = data.get("is_trained", True)
        
        return self


class WeeklyGridSearch:
    """
    Automated grid search for strategy parameter optimization.
    Runs weekly to find optimal allocations near current holdings.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.results_file = os.path.join(data_dir, "grid_search_results.jsonl")
        
    def run_search(
        self,
        symbols: List[str],
        base_allocations: Dict[str, float],
        grid_steps: int = 5,
        max_deviation: float = 0.10,
        min_weight: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """
        Run grid search around current allocations.
        
        Args:
            symbols: Assets to include
            base_allocations: Current target weights
            grid_steps: Number of steps in each direction
            max_deviation: Max deviation from base allocation
            min_weight: Minimum weight for any asset
            
        Returns:
            List of evaluated configurations sorted by Sharpe
        """
        import random
        from datetime import datetime
        
        results = []
        
        # Generate grid points
        for i in range(grid_steps * len(symbols)):
            # Randomly perturb allocations
            perturbations = {
                s: random.uniform(-max_deviation, max_deviation)
                for s in symbols
            }
            
            # Apply perturbations
            new_alloc = {}
            for s in symbols:
                base = base_allocations.get(s, 1.0 / len(symbols))
                new_alloc[s] = max(min_weight, min(1.0, base + perturbations[s]))
            
            # Normalize to 100%
            total = sum(new_alloc.values())
            new_alloc = {s: w / total for s, w in new_alloc.items()}
            
            # Calculate mock metrics (would be from backtest in real impl)
            mock_sharpe = random.uniform(0.3, 0.8)
            mock_volatility = random.uniform(0.08, 0.15)
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "allocations": new_alloc,
                "base_allocations": base_allocations,
                "sharpe": mock_sharpe,
                "volatility": mock_volatility,
                "perturbation": perturbations,
            }
            results.append(result)
        
        # Sort by Sharpe
        results.sort(key=lambda x: x["sharpe"], reverse=True)
        
        # Save
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self.results_file, "a") as f:
            for r in results[:5]:  # Save top 5
                f.write(json.dumps(r) + "\n")
        
        return results


def main():
    """CLI for regime classifier."""
    import sys
    
    classifier = RegimeClassifier()
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == "train":
            # Load features from file
            features_file = sys.argv[2] if len(sys.argv) > 2 else "data/features.jsonl"
            
            if not os.path.exists(features_file):
                print(f"Features file not found: {features_file}")
                print("Generate features first: python3 -m research.features batch")
                return
            
            # Load features
            features_list = []
            with open(features_file, "r") as f:
                for line in f:
                    try:
                        features_list.append(json.loads(line))
                    except (json.JSONDecodeError, OSError):
                        continue
            
            print(f"Training on {len(features_list)} samples...")
            
            try:
                metrics = classifier.train(features_list)
                print(json.dumps(metrics, indent=2))
                
                # Save model
                path = classifier.save()
                print(f"Model saved to {path}")
                
            except Exception as e:
                print(f"Training failed: {e}")
                
        elif cmd == "predict":
            # Load model and predict
            features_file = sys.argv[2] if len(sys.argv) > 2 else "data/features.jsonl"
            
            try:
                classifier.load()
            except FileNotFoundError:
                print("Model not found. Train first: python3 -m research.regime_classifier train")
                return
            
            # Load latest features and predict
            with open(features_file, "r") as f:
                lines = f.readlines()
                if not lines:
                    print("No features available")
                    return
                
                latest = json.loads(lines[-1])
                prediction = classifier.predict(latest)
                print(json.dumps(prediction.to_dict(), indent=2))
                
        elif cmd == "grid":
            # Run grid search
            grid = WeeklyGridSearch()
            base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}  # Champion allocation
            results = grid.run_search(
                symbols=["SPY", "GLD", "TLT"],
                base_allocations=base,
                grid_steps=3,
            )
            print(f"Grid search complete. Top result:")
            print(json.dumps(results[0], indent=2))
            
        else:
            print(f"Unknown command: {cmd}")
            print("Commands: train [features_file], predict [features_file], grid")
    else:
        print("Regime Classifier")
        print("Commands: train, predict, grid")
        print(f"sklearn available: {SKLEARN_AVAILABLE}")
        print(f"xgboost available: {XGBOOST_AVAILABLE}")


if __name__ == "__main__":
    main()
