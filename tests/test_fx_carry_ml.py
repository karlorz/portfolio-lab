"""Tests for ML-enhanced FX carry timing v3.19."""
import pytest
import numpy as np
import json
from datetime import datetime
from pathlib import Path


class TestFeatureEngineering:
    """Feature computation from price history."""

    def test_compute_features_basic(self):
        from src.signals.fx_carry_ml import compute_features

        prices = [
            {"d": f"2026-{m:02d}-{d:02d}", "p": 25.0 + i * 0.1}
            for i, (m, d) in enumerate([(1, 15), (1, 16), (1, 17)] * 50)
            for _ in range(1)
        ][:200]  # 200 data points

        features = compute_features(prices, "UUP", label_unwinds=False)
        assert len(features) > 0
        f = features[-1]
        assert f.ticker == "UUP"
        assert isinstance(f.momentum_1m, float)
        assert isinstance(f.momentum_3m, float)
        assert isinstance(f.volatility_1m, float)
        assert isinstance(f.carry_signal, float)
        assert isinstance(f.trend_strength, float)
        assert isinstance(f.rate_differential, float)

    def test_compute_features_insufficient_data(self):
        from src.signals.fx_carry_ml import compute_features

        prices = [{"d": "2026-05-14", "p": 25.0}]
        features = compute_features(prices, "UUP")
        assert len(features) == 0

    def test_compute_features_with_labels(self):
        from src.signals.fx_carry_ml import compute_features, CARRY_SIGNAL_WINDOW, UNWIND_WINDOW

        # Generate 150 data points (enough for CARRY_SIGNAL_WINDOW + UNWIND_WINDOW)
        n = 150
        # Create a clear unwind pattern: drop in last 5 days
        prices = [{"d": f"2026-01-{min(d, 28):02d}", "p": 25.0} for d in range(1, n + 1)]
        # Make last 5 days show a sharp drop (>2%)
        for i in range(n - 5, n):
            prices[i]["p"] = 25.0 - (i - (n - 5)) * 0.15

        features = compute_features(prices, "UUP", label_unwinds=True)

        assert len(features) > 0
        # Some features near the end should have unwind_label=1
        labels = [f.unwind_label for f in features]
        assert 1 in labels, f"Expected at least one unwind event, got labels: {labels[-20:]}"

    def test_features_to_array(self):
        from src.signals.fx_carry_ml import FXCarryMLFeatures, features_to_array

        features = [
            FXCarryMLFeatures(
                ticker="UUP", date="2026-05-14",
                momentum_1m=1.0, momentum_3m=2.0, volatility_1m=15.0,
                carry_signal=0.5, trend_strength=1.2, rate_differential=-1.0,
                unwind_label=0
            ),
            FXCarryMLFeatures(
                ticker="UUP", date="2026-05-15",
                momentum_1m=-1.0, momentum_3m=-2.0, volatility_1m=18.0,
                carry_signal=-0.5, trend_strength=-1.2, rate_differential=1.0,
                unwind_label=1
            ),
        ]

        X, y = features_to_array(features)
        assert X.shape == (2, 6)
        assert y.tolist() == [0, 1]
        assert X[0, 0] == 1.0
        assert X[1, 0] == -1.0

    def test_features_to_array_no_labels(self):
        from src.signals.fx_carry_ml import FXCarryMLFeatures, features_to_array

        features = [
            FXCarryMLFeatures(
                ticker="UUP", date="2026-05-14",
                momentum_1m=0.0, momentum_3m=0.0, volatility_1m=0.0,
                carry_signal=0.0, trend_strength=0.0, rate_differential=0.0,
            )
        ]
        X, y = features_to_array(features)
        assert X.shape == (1, 6)
        assert y[0] == 0


class TestModelTraining:
    """RandomForest model training."""

    def test_train_model_with_mock_data(self, tmp_path):
        from src.signals.fx_carry_ml import train_model
        import json

        # Create mock price data with enough history
        n = 200
        prices = [25.0 + np.sin(i / 20) * 2 for i in range(n)]
        mock_data = {
            "UUP": [{"d": f"2026-01-{min(d, 28):02d}", "p": p}
                    for d, p in enumerate(prices, 1)]
        }
        prices_path = tmp_path / "prices.json"
        prices_path.write_text(json.dumps(mock_data))

        model_path = tmp_path / "model.pkl"

        result = train_model(
            prices_path=str(prices_path),
            model_path=str(model_path),
            tickers=["UUP"]
        )

        assert result["n_samples"] > 0
        assert "cv_f1_mean" in result
        # F1 score may be NaN if mock data has no unwind events — that's OK
        if not np.isnan(result["cv_f1_mean"]):
            assert result["cv_f1_mean"] > 0
        assert "feature_importance" in result
        assert len(result["feature_importance"]) == 6
        assert model_path.exists()

    def test_train_model_no_tickers(self):
        from src.signals.fx_carry_ml import train_model
        import json
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
            json.dump({}, f)
            f.flush()
            with pytest.raises(ValueError, match="No training data"):
                train_model(prices_path=f.name, tickers=["UUP"])


class TestPrediction:
    """Unwind risk prediction."""

    def test_predict_unwind_risk_no_model(self):
        from src.signals.fx_carry_ml import predict_unwind_risk
        pred = predict_unwind_risk("UUP", model_path="/nonexistent/model.pkl")
        assert pred.risk_level == "unknown"
        assert pred.carry_allowed is True
        assert pred.unwind_risk == 0.5

    def test_predict_with_trained_model(self, tmp_path):
        from src.signals.fx_carry_ml import train_model, predict_unwind_risk
        import json

        n = 200
        prices = [25.0 + np.sin(i / 20) * 2 for i in range(n)]
        mock_data = {
            "UUP": [{"d": f"2026-01-{min(d, 28):02d}", "p": p}
                    for d, p in enumerate(prices, 1)]
        }
        prices_path = tmp_path / "prices.json"
        prices_path.write_text(json.dumps(mock_data))
        model_path = tmp_path / "model.pkl"

        train_model(prices_path=str(prices_path), model_path=str(model_path), tickers=["UUP"])

        pred = predict_unwind_risk("UUP", prices_path=str(prices_path), model_path=str(model_path))
        assert pred.ticker == "UUP"
        assert isinstance(pred.unwind_risk, float)
        assert 0.0 <= pred.unwind_risk <= 1.0
        assert pred.risk_level in ("low", "medium", "high", "unknown")
        assert pred.features_used == 6

    def test_predict_unknown_ticker_with_model(self, tmp_path):
        """With a valid model, unknown ticker raises ValueError."""
        from src.signals.fx_carry_ml import train_model, predict_unwind_risk
        import json

        n = 200
        np.random.seed(42)
        prices = [25.0]
        for i in range(1, n):
            change = np.random.normal(0.001, 0.02)
            if i % 30 == 0 and i > 100:
                change -= 0.03
            prices.append(prices[-1] * (1 + change))

        mock_data = {
            "UUP": [{"d": f"2026-{((d-1)//28)+1:02d}-{((d-1)%28)+1:02d}", "p": round(p, 4)}
                    for d, p in enumerate(prices, 1)]
        }
        prices_path = tmp_path / "prices.json"
        prices_path.write_text(json.dumps(mock_data))
        model_path = tmp_path / "model.pkl"

        train_model(prices_path=str(prices_path), model_path=str(model_path), tickers=["UUP"])

        with pytest.raises(ValueError, match="No price data"):
            predict_unwind_risk("UNKNOWN", prices_path=str(prices_path), model_path=str(model_path))


class TestAllocation:
    """Carry allocation gating."""

    def test_get_carry_allocation_low_risk(self):
        from src.signals.fx_carry_ml import get_carry_allocation, FXCarryMLPrediction
        pred = FXCarryMLPrediction(
            ticker="UUP", unwind_risk=0.1, risk_level="low",
            carry_allowed=True, features_used=6,
            timestamp=datetime.now()
        )
        assert get_carry_allocation(pred, base_weight=5.0) == 5.0

    def test_get_carry_allocation_medium_risk(self):
        from src.signals.fx_carry_ml import get_carry_allocation, FXCarryMLPrediction
        pred = FXCarryMLPrediction(
            ticker="UUP", unwind_risk=0.4, risk_level="medium",
            carry_allowed=True, features_used=6,
            timestamp=datetime.now()
        )
        assert get_carry_allocation(pred, base_weight=5.0) == 3.75

    def test_get_carry_allocation_high_risk(self):
        from src.signals.fx_carry_ml import get_carry_allocation, FXCarryMLPrediction
        pred = FXCarryMLPrediction(
            ticker="UUP", unwind_risk=0.7, risk_level="high",
            carry_allowed=False, features_used=6,
            timestamp=datetime.now()
        )
        assert get_carry_allocation(pred, base_weight=5.0) == 0.0

    def test_get_carry_allocation_unknown(self):
        from src.signals.fx_carry_ml import get_carry_allocation, FXCarryMLPrediction
        pred = FXCarryMLPrediction(
            ticker="UUP", unwind_risk=0.5, risk_level="unknown",
            carry_allowed=True, features_used=0,
            timestamp=datetime.now()
        )
        assert get_carry_allocation(pred, base_weight=5.0) == 2.5

    def test_get_carry_allocation_custom_weight(self):
        from src.signals.fx_carry_ml import get_carry_allocation, FXCarryMLPrediction
        pred = FXCarryMLPrediction(
            ticker="UUP", unwind_risk=0.1, risk_level="low",
            carry_allowed=True, features_used=6,
            timestamp=datetime.now()
        )
        assert get_carry_allocation(pred, base_weight=3.0) == 3.0


class TestIntegration:
    """Integration with existing v3.15 FX carry infrastructure."""

    def test_imports_from_fx_carry_signal(self):
        """Verify we can import from the existing v3.15 module."""
        from src.signals.fx_carry_signal import FXCarrySignalGenerator, FXCarrySignal
        assert FXCarrySignalGenerator is not None
        assert FXCarrySignal is not None

    def test_feature_computation_from_real_data(self):
        """Compute features from real UUP data if available."""
        from src.signals.fx_carry_ml import compute_features, PRICES_PATH
        import json

        if not PRICES_PATH.exists():
            pytest.skip("prices.json not available")

        with open(PRICES_PATH) as f:
            data = json.load(f)

        if "UUP" not in data:
            pytest.skip("UUP not in prices.json")

        prices = sorted(data["UUP"], key=lambda x: x["d"])
        features = compute_features(prices, "UUP", label_unwinds=False)

        assert len(features) > 0, f"Should extract features from {len(prices)} UUP data points"
        f = features[-1]
        assert f.ticker == "UUP"
        assert isinstance(f.momentum_1m, float)

    def test_train_and_predict_e2e(self, tmp_path):
        """End-to-end: train model then predict."""
        from src.signals.fx_carry_ml import train_model, predict_unwind_risk, get_carry_allocation
        import json

        n = 200
        # Generate price data with a trend + noise + periodic drops
        np.random.seed(42)
        prices = [25.0]
        for i in range(1, n):
            change = np.random.normal(0.001, 0.02)
            # Every ~30 days add a mini crash
            if i % 30 == 0 and i > 100:
                change -= 0.03
            prices.append(prices[-1] * (1 + change))

        mock_data = {
            "UUP": [{"d": f"2026-{((d-1)//28)+1:02d}-{((d-1)%28)+1:02d}", "p": round(p, 4)}
                    for d, p in enumerate(prices, 1)]
        }
        prices_path = tmp_path / "prices.json"
        prices_path.write_text(json.dumps(mock_data))
        model_path = tmp_path / "model.pkl"

        # Train
        result = train_model(
            prices_path=str(prices_path),
            model_path=str(model_path),
            tickers=["UUP"]
        )
        assert result["n_samples"] > 0

        # Predict
        pred = predict_unwind_risk(
            "UUP",
            prices_path=str(prices_path),
            model_path=str(model_path)
        )
        assert pred.ticker == "UUP"
        assert 0.0 <= pred.unwind_risk <= 1.0

        # Get allocation
        alloc = get_carry_allocation(pred, base_weight=5.0)
        assert 0.0 <= alloc <= 5.0
