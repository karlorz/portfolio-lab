"""Regression tests for canonical regime-weight exposure."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from src.signals import integrator
from src.strategy import ensemble_voter


def _projected_weights(regime: ensemble_voter.Regime) -> dict[str, float]:
    return {
        source.value: weight
        for source, weight in ensemble_voter.REGIME_WEIGHTS[regime].items()
    }


def test_integrator_regime_weights_project_canonical_ensemble_weights():
    for regime in ensemble_voter.Regime:
        assert integrator.REGIME_WEIGHTS[regime.value] == _projected_weights(regime)


def test_integrator_legacy_neutral_alias_uses_canonical_normal_weights():
    assert integrator.REGIME_WEIGHTS["neutral"] == _projected_weights(
        ensemble_voter.Regime.NORMAL
    )


def test_integrator_regime_weights_respect_ensemble_weights_file_override(tmp_path):
    weights_path = tmp_path / "ensemble_weights.json"
    custom_weights = {
        regime.value: {"multi_speed_momentum": 0.42}
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
        "print(json.dumps(REGIME_WEIGHTS['normal'], sort_keys=True))"
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
    assert json.loads(result.stdout) == {"multi_speed_momentum": 0.42}
