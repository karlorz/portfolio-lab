"""Tests for per-signal walk-forward validation module."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.monitor.signal_walk_forward import (
    SignalWalkForwardValidator,
    SignalWFEResult,
    SignalWFEWindow,
    compute_signal_wfe_report,
)


class TestSignalWalkForwardValidator:
    """Test the per-signal walk-forward validator."""

    def test_validated_signal_with_good_predictions(self):
        """Signal with consistent predictive power should be 'validated'."""
        validator = SignalWalkForwardValidator(
            n_splits=5, test_size=30, gap=5, min_ic=0.03,
        )
        # 500 observations with strong positive IC
        rng = np.random.RandomState(42)
        predictions = rng.randn(500).tolist()
        actual_returns = [p * 0.5 + rng.normal(0, 0.1) for p in predictions]

        result = validator.validate_signal("good_signal", predictions, actual_returns)
        assert result.status in ("validated", "weak")
        assert result.n_windows >= 1
        assert result.mean_oos_ic > 0

    def test_unvalidated_signal_with_random_predictions(self):
        """Signal with random predictions should be 'unvalidated'."""
        validator = SignalWalkForwardValidator(
            n_splits=5, test_size=30, gap=5, min_ic=0.03,
        )
        rng = np.random.RandomState(123)
        predictions = rng.randn(500).tolist()
        actual_returns = rng.randn(500).tolist()  # No correlation

        result = validator.validate_signal("random_signal", predictions, actual_returns)
        # Random signal should not be validated (IC near 0)
        assert result.status in ("unvalidated", "weak")
        assert abs(result.mean_oos_ic) < 0.3  # IC should be near zero

    def test_insufficient_data_returns_early(self):
        """Too few observations should return insufficient_data status."""
        validator = SignalWalkForwardValidator(test_size=100, gap=10)
        predictions = [0.1, 0.2, 0.3, 0.4, 0.5]
        actual_returns = [0.01, 0.02, 0.03, 0.04, 0.05]

        result = validator.validate_signal("short_signal", predictions, actual_returns)
        assert result.status == "insufficient_data"
        assert result.n_windows == 0

    def test_mismatched_lengths_raises(self):
        """Mismatched prediction/return lengths should raise ValueError."""
        validator = SignalWalkForwardValidator()
        with pytest.raises(ValueError, match="same length"):
            validator.validate_signal("bad", [1, 2, 3], [1, 2])

    def test_wfe_computed_correctly(self):
        """WFE should be OOS_IC / IS_IC ratio."""
        validator = SignalWalkForwardValidator(
            n_splits=3, test_size=50, gap=10, min_ic=0.01,
        )
        rng = np.random.RandomState(42)
        # Good predictions — consistent positive IC
        predictions = list(range(500))
        actual_returns = [p * 0.01 + rng.normal(0, 0.001) for p in predictions]

        result = validator.validate_signal("wfe_signal", predictions, actual_returns)
        assert result.wfe is not None
        if result.mean_is_ic != 0:
            expected_wfe = result.mean_oos_ic / result.mean_is_ic
            assert abs(result.wfe - round(expected_wfe, 4)) < 0.01

    def test_positive_oos_ratio_computed(self):
        """positive_oos_ratio should reflect fraction of positive OOS IC windows."""
        validator = SignalWalkForwardValidator(
            n_splits=3, test_size=50, gap=10, min_ic=0.01,
        )
        rng = np.random.RandomState(42)
        predictions = rng.randn(500).tolist()
        actual_returns = [p * 0.3 + rng.normal(0, 0.1) for p in predictions]

        result = validator.validate_signal("ratio_signal", predictions, actual_returns)
        assert 0.0 <= result.positive_oos_ratio <= 1.0

    def test_windows_populated(self):
        """Per-window results should be populated."""
        validator = SignalWalkForwardValidator(
            n_splits=3, test_size=50, gap=10, min_ic=0.01,
        )
        rng = np.random.RandomState(42)
        predictions = list(range(500))
        actual_returns = [p * 0.01 + rng.normal(0, 0.001) for p in predictions]

        result = validator.validate_signal("windowed_signal", predictions, actual_returns)
        assert len(result.windows) >= 1
        for w in result.windows:
            assert isinstance(w, SignalWFEWindow)
            assert -1.0 <= w.is_ic <= 1.0
            assert -1.0 <= w.oos_ic <= 1.0
            assert w.train_days > 0
            assert w.test_days > 0

    def test_constant_windows_are_missing_evidence_not_zero_ic(self):
        validator = SignalWalkForwardValidator(
            n_splits=3, test_size=30, gap=5, min_ic=0.01,
        )
        predictions = [1.0] * 300
        actual_returns = np.linspace(-0.1, 0.1, 300).tolist()

        result = validator.validate_signal(
            "constant_signal", predictions, actual_returns
        )

        assert result.status == "insufficient_data"
        assert result.n_windows == 0
        assert result.windows == []


class TestValidateFromICMonitor:
    """Test batch validation from ICMonitor data."""

    def test_multiple_signals_validated(self):
        """Should validate multiple signals from IC monitor data."""
        validator = SignalWalkForwardValidator(
            n_splits=3, test_size=30, gap=5, min_ic=0.01,
        )
        rng = np.random.RandomState(42)
        ic_data = {
            "good_signal": [(float(i), float(i) * 0.01 + rng.normal(0, 0.001)) for i in range(300)],
            "random_signal": [(rng.randn(), rng.randn() * 0.01) for _ in range(300)],
        }

        results = validator.validate_from_ic_monitor(ic_data)
        assert "good_signal" in results
        assert "random_signal" in results
        assert results["good_signal"]["status"] in ("validated", "weak", "unvalidated")
        assert results["random_signal"]["status"] in ("validated", "weak", "unvalidated")

    def test_empty_signal_returns_insufficient_data(self):
        """Empty signal data should return insufficient_data."""
        validator = SignalWalkForwardValidator()
        results = validator.validate_from_ic_monitor({"empty": []})
        assert results["empty"]["status"] == "insufficient_data"


class TestWFEPersistence:
    """Test save/load state persistence."""

    def test_save_and_load_state(self, tmp_path):
        """Save then load should preserve results."""
        validator = SignalWalkForwardValidator()
        results = {
            "test_signal": {
                "signal_name": "test_signal",
                "wfe": 0.92,
                "status": "validated",
            },
        }
        path = tmp_path / "wfe_state.json"
        validator.save_state(results, path=path)

        loaded = validator.load_state(path=path)
        assert loaded["test_signal"]["wfe"] == 0.92
        assert loaded["test_signal"]["status"] == "validated"

    def test_load_nonexistent_returns_empty(self, tmp_path):
        """Loading nonexistent file should return empty dict."""
        validator = SignalWalkForwardValidator()
        result = validator.load_state(path=tmp_path / "nonexistent.json")
        assert result == {}

    def test_load_corrupt_json_returns_empty(self, tmp_path):
        """Loading corrupt JSON should return empty dict."""
        path = tmp_path / "corrupt.json"
        path.write_text("not valid json{{{")
        validator = SignalWalkForwardValidator()
        result = validator.load_state(path=path)
        assert result == {}

    def test_save_creates_parent_dirs(self, tmp_path):
        """Save should create parent directories."""
        path = tmp_path / "sub" / "dir" / "wfe.json"
        validator = SignalWalkForwardValidator()
        validator.save_state({"sig": {"wfe": 0.5}}, path=path)
        assert path.exists()


class TestComputeSignalWFEReport:
    """Test the convenience function."""

    def test_convenience_function_returns_dict(self):
        """compute_signal_wfe_report should return a dict without crashing."""
        report = compute_signal_wfe_report()
        assert isinstance(report, dict)

    def test_staged_ic_state_reports_waiting_for_resolved_history(
        self, tmp_path, monkeypatch
    ):
        """WFE should not emit an empty report when predictions are label-pending."""
        import src.monitor.ic_decay_monitor as icm
        import src.monitor.signal_walk_forward as swf

        wfe_state = tmp_path / "signal_wfe_state.json"
        ic_state = tmp_path / "ic_monitor_state.json"
        ic_state.write_text(json.dumps({
            "__staged__": {
                "date": "2026-07-02",
                "predictions": {"ensemble_equity": 0.4},
            }
        }))
        monkeypatch.setattr(swf, "WFV_STATE_PATH", wfe_state)
        monkeypatch.setattr(icm, "IC_STATE_PATH", ic_state)

        report = compute_signal_wfe_report()

        assert report["status"] == "waiting_for_forward_returns"
        assert report["pending_predictions"] == 1
        assert report["signals"] == {}
