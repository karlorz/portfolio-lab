"""
Tests for Transformer Regime Detector (v3.18)
ML-gated — requires PORTFOLIO_LAB_ENABLE_ML=1
"""

import os
import pytest
import numpy as np

ML_ENABLED = os.environ.get("PORTFOLIO_LAB_ENABLE_ML") == "1"


@pytest.mark.skipif(not ML_ENABLED, reason="ML disabled")
class TestTransformerRegimeModel:
    """Test transformer model architecture."""

    @pytest.fixture
    def model(self):
        from src.regime.transformer_regime import TransformerRegimeModel
        return TransformerRegimeModel()

    def test_model_creation(self, model):
        assert model is not None
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params < 200000  # Should be lightweight (<200k params)
        assert n_params > 1000

    def test_forward_pass(self, model):
        import torch
        batch = torch.randn(4, 60, 3)
        out = model(batch)
        assert out.shape == (4, 5)

    def test_predict(self, model):
        import torch
        seq = np.random.RandomState(42).normal(0, 0.01, (60, 3))
        pred_idx, conf, probs = model.predict(seq)
        assert 0 <= pred_idx < 5
        assert 0.0 <= conf <= 1.0
        assert len(probs) == 5
        assert abs(sum(probs) - 1.0) < 0.01


@pytest.mark.skipif(not ML_ENABLED, reason="ML disabled")
class TestRegimeDataGenerator:
    """Test synthetic data generation."""

    @pytest.fixture
    def gen(self):
        from src.regime.transformer_regime import RegimeDataGenerator
        return RegimeDataGenerator(seed=42)

    def test_generates_all_regimes(self, gen):
        X, y = gen.generate_samples(n_per_regime=50)
        assert X.shape == (250, 60, 3)  # 5 regimes * 50 samples
        assert len(y) == 250
        assert set(y) == {0, 1, 2, 3, 4}  # All 5 regimes present

    def test_sequence_values_in_range(self, gen):
        X, _ = gen.generate_samples(n_per_regime=30)
        # Returns should be roughly in [-0.1, 0.1] range for daily
        assert -0.15 < X[:, :, 0].mean() < 0.15
        # Vol should be positive
        assert (X[:, :, 1] > 0).all()

    def test_regime_separation(self, gen):
        """Different regimes should have distinguishable characteristics."""
        X, y = gen.generate_samples(n_per_regime=40)
        # Crisis should have higher vol than trend_up
        crisis_idx = 4  # CRISIS
        trend_idx = 0   # TREND_UP
        crisis_vol = X[y == crisis_idx, :, 1].mean()
        trend_vol = X[y == trend_idx, :, 1].mean()
        assert crisis_vol > trend_vol

    def test_trend_regimes_have_opposite_sign(self, gen):
        """TREND_UP and TREND_DOWN should have opposite mean returns."""
        X, y = gen.generate_samples(n_per_regime=50)
        up_mean = X[y == 0, :, 0].mean()   # TREND_UP
        down_mean = X[y == 1, :, 0].mean()  # TREND_DOWN
        assert up_mean > down_mean


@pytest.mark.skipif(not ML_ENABLED, reason="ML disabled")
class TestTransformerTraining:
    """Test model training pipeline."""

    def test_train_small(self, tmp_path):
        """Quick training run on small data."""
        import torch
        from src.regime.transformer_regime import (
            TransformerRegimeModel,
            RegimeDataGenerator,
        )

        gen = RegimeDataGenerator(seed=42)
        X, y = gen.generate_samples(n_per_regime=100)  # 500 samples
        X_t = torch.FloatTensor(X)
        y_t = torch.LongTensor(y)

        model = TransformerRegimeModel()
        import torch.optim as optim
        import torch.nn as nn
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for _ in range(5):
            optimizer.zero_grad()
            logits = model(X_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

        # After training, should have decent accuracy
        model.eval()
        with torch.no_grad():
            preds = model(X_t).argmax(dim=-1)
            acc = (preds == y_t).float().mean().item()

        assert acc > 0.3  # Better than random (0.2)
        assert loss.item() < 2.0


@pytest.mark.skipif(not ML_ENABLED, reason="ML disabled")
class TestTransformerRegimeDetector:
    """Test detector integration."""

    @pytest.fixture
    def detector(self):
        from src.regime.transformer_regime import TransformerRegimeDetector
        return TransformerRegimeDetector()

    def test_detect_normal_sequence(self, detector):
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0.001, 0.01, 80))
        result = detector.detect(returns)
        assert result.regime is not None
        assert result.confidence > 0
        assert len(result.probabilities) == 5

    def test_detect_crisis_sequence(self, detector):
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0.001, 0.01, 50))
        # Add crash events
        returns += list(rng.normal(-0.03, 0.03, 10))
        result = detector.detect(returns)
        assert result.regime is not None

    def test_signal_value_in_range(self, detector):
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 70))
        result = detector.detect(returns)
        assert -1.0 <= result.signal_value <= 1.0

    def test_vol_regime_detected(self, detector):
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.02, 70))  # High vol
        result = detector.detect(returns)
        assert result.vol_regime in ("low", "normal", "high", "extreme")

    def test_serializable(self, detector):
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 70))
        result = detector.detect(returns)
        d = result.to_dict()
        assert "regime" in d
        assert "probabilities" in d
        assert "signal_value" in d

    def test_model_save_load(self, detector, tmp_path):
        import torch
        detector.MODEL_PATH = tmp_path / "test_model.pt"
        detector.save()
        assert detector.MODEL_PATH.exists()

        # Load in new detector
        from src.regime.transformer_regime import TransformerRegimeDetector
        detector2 = TransformerRegimeDetector()
        # Should load from saved path (revert after test)
        import src.regime.transformer_regime as tr
        detector2.MODEL_PATH = tmp_path / "test_model.pt"
        # Load manually
        detector2._model = tr.TransformerRegimeModel()
        detector2._model.load_state_dict(
            torch.load(tmp_path / "test_model.pt", weights_only=True)
        )

        # Both should give same prediction
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 70))
        r1 = detector.detect(returns)
        r2 = detector2.detect(returns)
        assert r1.regime == r2.regime


@pytest.mark.skipif(not ML_ENABLED, reason="ML disabled")
class TestConvenienceFunction:
    """Test convenience function for ensemble integration."""

    def test_detect_with_ml_enabled(self):
        from src.regime.transformer_regime import detect_transformer_regime

        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 70))
        result = detect_transformer_regime(returns)
        assert result.regime is not None
        assert result.signal_value is not None
