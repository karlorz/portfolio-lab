"""
Tests for src/experimental.py — ML feature flag and require_ml decorator.
No ML deps — tests env var gating logic only.
"""
import pytest
import os


class TestMLFeatureFlag:
    """ML_ENABLED depends on PORTFOLIO_LAB_ENABLE_ML env var."""

    def test_default_disabled(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("PORTFOLIO_LAB_ENABLE_ML", raising=False)
            from src.experimental import _is_ml_enabled
            assert _is_ml_enabled() is False

    def test_explicitly_enabled(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("PORTFOLIO_LAB_ENABLE_ML", "1")
            from src.experimental import _is_ml_enabled
            assert _is_ml_enabled() is True

    def test_explicitly_disabled(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("PORTFOLIO_LAB_ENABLE_ML", "0")
            from src.experimental import _is_ml_enabled
            assert _is_ml_enabled() is False

    def test_any_other_value_is_false(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("PORTFOLIO_LAB_ENABLE_ML", "yes")
            from src.experimental import _is_ml_enabled
            assert _is_ml_enabled() is False

    def test_module_level_constant(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("PORTFOLIO_LAB_ENABLE_ML", "1")
            import importlib
            import src.experimental
            importlib.reload(src.experimental)
            assert src.experimental.ML_ENABLED is True
            mp.setenv("PORTFOLIO_LAB_ENABLE_ML", "0")
            importlib.reload(src.experimental)
            assert src.experimental.ML_ENABLED is False

    def test_require_ml_decorator_allows_when_enabled(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("PORTFOLIO_LAB_ENABLE_ML", "1")
            import importlib
            import src.experimental
            importlib.reload(src.experimental)

            called = []
            @src.experimental.require_ml
            def test_func():
                called.append(True)
                return 42

            result = test_func()
            assert result == 42
            assert len(called) == 1

    def test_require_ml_decorator_blocks_when_disabled(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("PORTFOLIO_LAB_ENABLE_ML", "0")
            import importlib
            import src.experimental
            importlib.reload(src.experimental)

            called = []
            @src.experimental.require_ml
            def test_func():
                called.append(True)
                return 42

            result = test_func()
            assert result is None
            assert len(called) == 0
