"""
Transformer Regime Detector - v3.18 Implementation
Lightweight transformer for market regime classification from price sequences.

Architecture:
- Input: 60-day sequences of (return, realized_vol, volume_ratio)
- Model: Small transformer encoder (d_model=64, 2 layers, 4 heads, ~50k params)
- Output: 5 regime classes + confidence

Regimes:
- TREND_UP: persistent positive drift, low vol
- TREND_DOWN: persistent negative drift, low vol
- MEAN_REVERT: oscillating around mean, moderate vol
- HIGH_VOL: elevated volatility, uncertain direction
- CRISIS: extreme vol, sharp drawdowns, fat tails

ML-gated: requires PORTFOLIO_LAB_ENABLE_ML=1 and PyTorch installed.

Usage:
    PORTFOLIO_LAB_ENABLE_ML=1 python -m src.regime.transformer_regime detect
    PORTFOLIO_LAB_ENABLE_ML=1 python -m src.regime.transformer_regime train
"""

import json
import logging
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ML gate
_ML_ENABLED = os.environ.get("PORTFOLIO_LAB_ENABLE_ML") == "1"
_TORCH = None
_F = None
_nn = None
_optim = None

if _ML_ENABLED:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        _TORCH = torch
        _F = F
        _nn = nn
        _optim = optim
        logger.info("PyTorch loaded for transformer regime detector")
    except ImportError:
        logger.warning("PyTorch not available — transformer regime detector disabled")
        _ML_ENABLED = False


class TransformerRegime(Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    MEAN_REVERT = "mean_revert"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"


@dataclass
class RegimePrediction:
    """Transformer regime prediction result."""
    timestamp: str
    regime: str
    confidence: float
    probabilities: Dict[str, float]  # Per-class probabilities

    # Ensemble integration
    signal_value: float       # -1 to +1 for ensemble voter
    trend_strength: float     # 0-1
    vol_regime: str           # low, normal, high, extreme

    explanation: str

    def to_dict(self) -> dict:
        return asdict(self)


class RegimeDataGenerator:
    """
    Generates synthetic training data with known regime labels.

    Each sample: 60-day sequence of (return, volatility, volume_ratio)
    with a known regime label for supervised training.
    """

    SEQ_LEN = 60
    N_FEATURES = 3

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_samples(self, n_per_regime: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """Generate balanced training data across all 5 regimes."""
        X_list, y_list = [], []

        for regime_idx, regime in enumerate(TransformerRegime):
            for _ in range(n_per_regime):
                seq = self._generate_sequence(regime)
                X_list.append(seq)
                y_list.append(regime_idx)

        X = np.stack(X_list)  # (N, 60, 3)
        y = np.array(y_list)
        return X, y

    def _generate_sequence(self, regime: TransformerRegime) -> np.ndarray:
        """Generate a 60-day sequence for a given regime."""
        n = self.SEQ_LEN
        seq = np.zeros((n, self.N_FEATURES))

        if regime == TransformerRegime.TREND_UP:
            returns = self.rng.normal(0.0015, 0.008, n)  # Positive drift
            vol = np.full(n, 0.12)
        elif regime == TransformerRegime.TREND_DOWN:
            returns = self.rng.normal(-0.0015, 0.008, n)  # Negative drift
            vol = np.full(n, 0.12)
        elif regime == TransformerRegime.MEAN_REVERT:
            # Oscillating with mean reversion
            raw = self.rng.normal(0, 0.015, n)
            returns = -0.3 * np.roll(raw, 1) + raw  # Negative autocorrelation
            vol = np.full(n, 0.15)
        elif regime == TransformerRegime.HIGH_VOL:
            returns = self.rng.normal(0, 0.025, n)
            vol = np.full(n, 0.30)
        else:  # CRISIS
            returns = self.rng.normal(-0.003, 0.04, n)
            # Add crash days
            crash_idx = self.rng.choice(n, size=5, replace=False)
            returns[crash_idx] = self.rng.normal(-0.03, 0.02, 5)
            vol = np.full(n, 0.50)

        # Add noise to vol
        vol += self.rng.normal(0, 0.02, n)
        vol = np.abs(vol)

        # Volume ratio: higher in high vol / crisis
        if regime in (TransformerRegime.HIGH_VOL, TransformerRegime.CRISIS):
            volume = self.rng.normal(1.5, 0.3, n)
        else:
            volume = self.rng.normal(1.0, 0.2, n)
        volume = np.abs(volume)

        seq[:, 0] = returns
        seq[:, 1] = vol
        seq[:, 2] = volume
        return seq


class TransformerRegimeModel(nn.Module):
    """
    Lightweight transformer encoder for regime classification.

    Architecture:
    - Input projection: 3 → 64
    - Positional encoding (learned)
    - 2x TransformerEncoderLayer (d=64, heads=4, ff=128)
    - Global mean pooling
    - Classifier: 64 → 5
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, n_classes: int = 5,
                 seq_len: int = 60, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        self.input_proj = nn.Linear(3, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=128,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        # x: (batch, seq_len, 3)
        x = self.input_proj(x)  # (B, S, d)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.encoder(x)     # (B, S, d)
        x = x.mean(dim=1)       # Global pooling (B, d)
        x = self.classifier(x)  # (B, 5)
        return x

    def predict(self, x: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """Predict regime from a single sequence."""
        self.eval()
        with torch.no_grad():
            t = torch.FloatTensor(x).unsqueeze(0)  # (1, S, 3)
            logits = self.forward(t)
            probs = F.softmax(logits, dim=-1).squeeze().numpy()
            pred = int(np.argmax(probs))
            conf = float(probs[pred])
        return pred, conf, probs


class TransformerRegimeDetector:
    """
    Transformer-based regime detector for portfolio integration.

    Classifies market regimes from 60-day price sequences using a
    lightweight transformer (~50k parameters).

    Model is trained on synthetic regime data and can be fine-tuned
    on real market data.
    """

    MODEL_PATH = Path(__file__).parent.parent.parent / "data" / "transformer_regime_model.pt"
    N_CLASSES = 5
    SEQ_LEN = 60

    REGIME_TO_SIGNAL = {
        0: 0.5,   # TREND_UP → bullish
        1: -0.5,  # TREND_DOWN → bearish
        2: 0.0,   # MEAN_REVERT → neutral
        3: -0.2,  # HIGH_VOL → slightly bearish
        4: -0.8,  # CRISIS → strongly bearish
    }

    def __init__(self):
        if not _ML_ENABLED:
            raise RuntimeError(
                "ML disabled. Set PORTFOLIO_LAB_ENABLE_ML=1 to use transformer regime detector."
            )
        self._model: Optional[TransformerRegimeModel] = None
        self._generator = RegimeDataGenerator()
        self._load_or_train()

    def _load_or_train(self):
        if self.MODEL_PATH.exists():
            try:
                self._model = TransformerRegimeModel()
                self._model.load_state_dict(
                    torch.load(self.MODEL_PATH, weights_only=True)
                )
                self._model.eval()
                logger.info(f"Loaded trained model from {self.MODEL_PATH}")
                return
            except Exception as e:
                logger.warning(f"Failed to load model: {e}, retraining...")

        logger.info("Training new transformer regime model...")
        self.train()
        self.save()

    def train(self, n_per_regime: int = 500, epochs: int = 20,
              batch_size: int = 64, lr: float = 0.001):
        """Train the transformer on synthetic regime data."""
        if not _ML_ENABLED:
            return

        self._model = TransformerRegimeModel()
        X, y = self._generator.generate_samples(n_per_regime)

        # Convert to tensors
        X_t = torch.FloatTensor(X)
        y_t = torch.LongTensor(y)

        # Train/val split
        n = len(X)
        idx = np.random.RandomState(42).permutation(n)
        split = int(n * 0.8)
        train_idx = idx[:split]
        val_idx = idx[split:]

        X_train, y_train = X_t[train_idx], y_t[train_idx]
        X_val, y_val = X_t[val_idx], y_t[val_idx]

        optimizer = optim.AdamW(self._model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        for epoch in range(epochs):
            self._model.train()
            total_loss = 0.0
            n_batches = 0

            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                optimizer.zero_grad()
                logits = self._model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            # Validation
            self._model.eval()
            with torch.no_grad():
                val_logits = self._model(X_val)
                val_preds = val_logits.argmax(dim=-1)
                val_acc = (val_preds == y_val).float().mean().item()

            if val_acc > best_acc:
                best_acc = val_acc

            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}: loss={total_loss/n_batches:.4f}, "
                           f"val_acc={val_acc:.3f}")

        logger.info(f"Training complete: best val acc = {best_acc:.3f}")

    def save(self):
        if self._model is not None:
            self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._model.state_dict(), self.MODEL_PATH)
            logger.info(f"Model saved to {self.MODEL_PATH}")

    def detect(self, returns: List[float],
               vols: Optional[List[float]] = None,
               volumes: Optional[List[float]] = None) -> RegimePrediction:
        """Detect current market regime from recent return sequence."""
        if not _ML_ENABLED or self._model is None:
            return RegimePrediction(
                timestamp=datetime.now().isoformat(),
                regime="unknown", confidence=0.0, probabilities={},
                signal_value=0.0, trend_strength=0.0,
                vol_regime="unknown",
                explanation="ML disabled or model not trained",
            )

        # Prepare sequence
        n = min(len(returns), self.SEQ_LEN)
        seq = np.zeros((self.SEQ_LEN, 3))

        recent = returns[-n:]
        seq[-n:, 0] = recent

        if vols and len(vols) >= n:
            seq[-n:, 1] = vols[-n:]
        else:
            seq[-n:, 1] = np.std(recent) * math.sqrt(252) if n > 1 else 0.16

        if volumes and len(volumes) >= n:
            seq[-n:, 2] = volumes[-n:]
        else:
            seq[-n:, 2] = 1.0

        # Predict
        pred_idx, conf, probs = self._model.predict(seq)

        regime_names = [r.value for r in TransformerRegime]
        regime = regime_names[pred_idx]
        probs_dict = {regime_names[i]: float(p) for i, p in enumerate(probs)}

        signal = self.REGIME_TO_SIGNAL.get(pred_idx, 0.0)
        trend_strength = max(probs[0], probs[1])  # TREND_UP or TREND_DOWN prob

        # Vol regime from realized vol
        realized_vol = np.std(recent) * math.sqrt(252) if n > 1 else 0.16
        if realized_vol > 0.40:
            vol_regime = "extreme"
        elif realized_vol > 0.25:
            vol_regime = "high"
        elif realized_vol > 0.15:
            vol_regime = "normal"
        else:
            vol_regime = "low"

        return RegimePrediction(
            timestamp=datetime.now().isoformat(),
            regime=regime,
            confidence=round(conf, 3),
            probabilities=probs_dict,
            signal_value=round(signal, 3),
            trend_strength=round(trend_strength, 3),
            vol_regime=vol_regime,
            explanation=(
                f"Transformer: {regime} (conf={conf:.2f}), "
                f"vol={realized_vol:.1%}, "
                f"trend={trend_strength:.2f}"
            ),
        )


def detect_transformer_regime(
    returns: List[float],
    vols: Optional[List[float]] = None,
    volumes: Optional[List[float]] = None,
) -> RegimePrediction:
    """Convenience function for transformer regime detection."""
    if not _ML_ENABLED:
        return RegimePrediction(
            timestamp=datetime.now().isoformat(),
            regime="unknown", confidence=0.0, probabilities={},
            signal_value=0.0, trend_strength=0.0,
            vol_regime="unknown",
            explanation="ML disabled — set PORTFOLIO_LAB_ENABLE_ML=1",
        )
    detector = TransformerRegimeDetector()
    return detector.detect(returns, vols, volumes)


def main():
    import sys

    if not _ML_ENABLED:
        print("ML disabled. Set PORTFOLIO_LAB_ENABLE_ML=1 to use.")
        return

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        detector = TransformerRegimeDetector()
        detector.train(n_per_regime=500, epochs=30)
        detector.save()
        print("Model trained and saved.")
        return

    # Detect mode
    detector = TransformerRegimeDetector()

    # Generate test sequence
    rng = np.random.RandomState(42)
    returns = list(rng.normal(0.001, 0.01, 60))
    result = detector.detect(returns)

    print("=" * 60)
    print("TRANSFORMER REGIME DETECTOR v3.18")
    print("=" * 60)
    print(f"Regime: {result.regime}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Signal Value: {result.signal_value:+.3f}")
    print(f"Trend Strength: {result.trend_strength:.3f}")
    print(f"Vol Regime: {result.vol_regime}")
    print()
    print("Class Probabilities:")
    for regime, prob in sorted(result.probabilities.items(),
                                key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        print(f"  {regime:<15} {prob:.3f} {bar}")
    print()
    print(f"Explanation: {result.explanation}")
    print("=" * 60)


if __name__ == "__main__":
    main()
