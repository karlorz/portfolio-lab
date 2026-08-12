"""Batch BJ residual honesty: last_full stamp trail + adaptive empty/ghost guards."""

from __future__ import annotations

import json
from unittest.mock import patch


def test_stamp_generator_git_sha_retains_last_full_when_tip_moves():
    from src.dashboard.generator import _stamp_generator_git_sha

    with patch(
        "src.dashboard.generator._generator_git_sha_short",
        return_value="newtipsha1234",
    ):
        out = _stamp_generator_git_sha(
            {
                "generator_git_sha": "oldtipsha5678",
                "generator_git_sha_status": "full_generate",
            }
        )
    assert out["generator_git_sha"] == "newtipsha1234"
    assert out["generator_git_sha_status"] == "full_generate"
    assert out["last_full_generator_git_sha"] == "oldtipsha5678"


def test_stamp_generator_git_sha_preserves_existing_last_full():
    from src.dashboard.generator import _stamp_generator_git_sha

    with patch(
        "src.dashboard.generator._generator_git_sha_short",
        return_value="newtipsha1234",
    ):
        out = _stamp_generator_git_sha(
            {
                "generator_git_sha": None,
                "generator_git_sha_status": "partial_patch",
                "last_full_generator_git_sha": "de54dadc3592",
            }
        )
    assert out["generator_git_sha"] == "newtipsha1234"
    assert out["last_full_generator_git_sha"] == "de54dadc3592"


def test_finalize_signal_metadata_stamps_status_and_last_full():
    from src.dashboard.generator import _finalize_signal_metadata

    with patch(
        "src.dashboard.generator._generator_git_sha_short",
        return_value="finalsha99999",
    ):
        out = _finalize_signal_metadata(
            {"generator_git_sha": "priorfull0001", "signals": {}}
        )
    assert out["generator_git_sha"] == "finalsha99999"
    assert out["generator_git_sha_status"] == "full_generate"
    assert out["last_full_generator_git_sha"] == "priorfull0001"
    assert "generated_at" in out


def test_adaptive_refuses_empty_overwrite_of_nonempty_state(tmp_path):
    from src.strategy.adaptive_ensemble_weights import AdaptiveEnsembleWeights

    state = tmp_path / "adaptive_weights_state.json"
    state.write_text(
        json.dumps(
            {
                "regime": "normal",
                "adjusted_weights": {"multi_speed_momentum": 1.0},
                "multipliers": {"multi_speed_momentum": 1.0},
                "baseline_weights": {"multi_speed_momentum": 1.0},
                "history": [],
            }
        ),
        encoding="utf-8",
    )
    aew = AdaptiveEnsembleWeights(base_weights={}, state_file=state)
    aew.adjusted_weights = {}
    aew.multipliers = {}
    aew._save_state()
    assert aew._last_save_refused == "refuse_empty_overwrite_of_nonempty_state"
    on_disk = json.loads(state.read_text(encoding="utf-8"))
    assert on_disk["adjusted_weights"]["multi_speed_momentum"] == 1.0


def test_adaptive_skips_ghost_sources_and_renorms_baseline(tmp_path):
    from src.strategy.adaptive_ensemble_weights import AdaptiveEnsembleWeights

    state = tmp_path / "adaptive_weights_state.json"
    base = {
        "multi_speed_momentum": 0.6,
        "cross_asset_rv": 0.4,
    }
    aew = AdaptiveEnsembleWeights(base_weights=base, state_file=state)
    attr = {
        "timestamp": "now",
        "sources": {
            "multi_speed_momentum": {
                "total_readings": 50,
                "sharpe_contribution": 0.8,
            },
            "cross_asset_rv": {
                "total_readings": 50,
                "sharpe_contribution": 0.5,
            },
            "factor_rotation": {
                "total_readings": 1,
                "sharpe_contribution": None,
                "avg_weight": 0.01,
            },
            "macro_momentum": {
                "total_readings": 1,
                "sharpe_contribution": None,
                "avg_weight": 0.01,
            },
        },
    }
    adapted = aew.update_weights(attr, "normal")
    assert "factor_rotation" not in adapted
    assert "macro_momentum" not in adapted
    assert abs(sum(adapted.values()) - 1.0) < 0.01
    assert set(adapted) <= set(base)


def test_stamp_full_generate_sets_last_full_self_trail_when_missing():
    """Batch CB: full_generate with empty last_full gets self-trail for lag forensics."""
    from src.dashboard import generator as gen

    with patch.object(gen, "_generator_git_sha_short", return_value="newtifulsha01"):
        out = gen._stamp_generator_git_sha(
            {"foo": 1, "generator_git_sha": None},
            status="full_generate",
        )
    assert out["generator_git_sha"] == "newtifulsha01"
    assert out["generator_git_sha_status"] == "full_generate"
    assert out["last_full_generator_git_sha"] == "newtifulsha01"
