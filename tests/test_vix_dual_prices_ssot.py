"""VIX dual-threshold default prices path is public SSOT."""
import os
from pathlib import Path


def test_historical_prices_defaults_to_public_prices_json(monkeypatch):
    monkeypatch.delenv("VIX_DUAL_HISTORICAL_PRICES_JSON", raising=False)
    # Re-import module constants
    import importlib
    import src.backtest.vix_dual_threshold_backtest as mod
    importlib.reload(mod)
    from src.paths import PRICES_JSON
    assert Path(mod.HISTORICAL_PRICES_JSON).resolve() == Path(PRICES_JSON).resolve()


def test_historical_prices_env_override(monkeypatch, tmp_path):
    snap = tmp_path / "snapshot_prices.json"
    snap.write_text("{}")
    monkeypatch.setenv("VIX_DUAL_HISTORICAL_PRICES_JSON", str(snap))
    import importlib
    import src.backtest.vix_dual_threshold_backtest as mod
    importlib.reload(mod)
    assert Path(mod.HISTORICAL_PRICES_JSON).resolve() == snap.resolve()
