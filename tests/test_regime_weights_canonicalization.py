"""Regression tests for canonical regime-weight input to legacy integrator weights."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from src.signals import integrator
from src.strategy import ensemble_voter


def test_integrator_regime_weights_keep_legacy_public_contract():
    assert set(integrator.REGIME_WEIGHTS) == {
        "bull",
        "bear",
        "neutral",
        "crisis",
        "high_vol",
    }
    assert integrator.REGIME_WEIGHTS["neutral"] is integrator.BASE_WEIGHTS

    for regime, weights in integrator.REGIME_WEIGHTS.items():
        assert set(weights).issubset(integrator.BASE_WEIGHTS), regime


def test_integrator_bull_weights_project_from_canonical_low_vol_weights():
    canonical_low_vol = {
        source.value: weight
        for source, weight in ensemble_voter.REGIME_WEIGHTS[
            ensemble_voter.Regime.LOW_VOL
        ].items()
    }

    bull_weights = integrator.REGIME_WEIGHTS["bull"]

    assert bull_weights["multi_speed"] == canonical_low_vol["multi_speed_momentum"]
    assert bull_weights["value"] == canonical_low_vol["cross_asset_rv"]
    assert bull_weights["momentum"] == (
        canonical_low_vol["international_momentum"]
        + canonical_low_vol["multi_timeframe_fusion"]
    )
    assert "network_momentum" not in bull_weights
    assert bull_weights["sentiment"] == (
        canonical_low_vol["alternative_data"] + canonical_low_vol["google_trends"]
    )


def test_multi_timeframe_fusion_does_not_project_to_network_momentum():
    legacy_type = integrator._CANONICAL_SOURCE_TO_LEGACY_TYPE[
        ensemble_voter.SignalSource.MULTI_TIMEFRAME_FUSION
    ]

    assert legacy_type != "network_momentum"


def test_integrator_regime_weights_respect_ensemble_weights_file_override(tmp_path):
    weights_path = tmp_path / "ensemble_weights.json"
    projected_sources = {
        "international_momentum": 0.40,
        "alternative_data": 0.20,
        "google_trends": 0.10,
        "vix_term_structure": 0.30,
    }
    custom_weights = {
        regime.value: projected_sources
        for regime in ensemble_voter.Regime
    }
    weights_path.write_text(json.dumps(custom_weights), encoding="utf-8")

    env = {
        **os.environ,
        "PORTFOLIO_LAB_ENABLE_ML": "0",
        "ENSEMBLE_WEIGHTS_FILE": str(weights_path),
    }
    script = (
        "import json; "
        "from src.signals.integrator import REGIME_WEIGHTS; "
        "print(json.dumps(REGIME_WEIGHTS['bull'], sort_keys=True))"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout) == {
        "macro": 0.30,
        "momentum": 0.40,
        "sentiment": 0.30,
    }
