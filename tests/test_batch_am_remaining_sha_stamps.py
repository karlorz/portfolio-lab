"""Batch AM remaining: generator_git_sha on six public dashboard surfaces."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def public_dir(tmp_path, monkeypatch):
    public = tmp_path / "public"
    public.mkdir()
    monkeypatch.setattr("src.dashboard.generator.PUBLIC_DIR", public)
    return public


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_vixy_hedge_json_stamps_generator_git_sha(public_dir, monkeypatch):
    from src.dashboard.generator import DashboardGenerator

    gen = DashboardGenerator.__new__(DashboardGenerator)

    class FakeSizer:
        def status(self):
            return {"hedge_weight": 0.0, "vix_level": 15.0}

    with patch(
        "src.strategy.vixy_hedge_sizing.VIXYHedgeSizer",
        FakeSizer,
    ), patch(
        "src.dashboard.generator._generator_git_sha_short",
        return_value="vixyhedge01",
    ):
        out = DashboardGenerator.generate_vixy_hedge_json(gen)

    assert out is not None
    payload = _read(Path(out))
    assert payload["generator_git_sha"] == "vixyhedge01"
    assert payload["generator_git_sha_status"] == "full_generate"


def test_turnover_validator_json_stamps_generator_git_sha(public_dir):
    from src.dashboard.generator import DashboardGenerator

    gen = DashboardGenerator.__new__(DashboardGenerator)

    class FakeValidator:
        def get_state_diagnostics(self):
            return {}

    with patch(
        "src.strategy.turnover_validator.TurnoverValidator",
        FakeValidator,
    ), patch(
        "src.dashboard.generator._generator_git_sha_short",
        return_value="turnover01",
    ):
        out = DashboardGenerator.generate_turnover_validator_json(gen)

    assert out is not None
    payload = _read(Path(out))
    assert payload["generator_git_sha"] == "turnover01"
    assert payload["generator_git_sha_status"] == "full_generate"


def test_tsmom_json_stamps_generator_git_sha(public_dir):
    from src.dashboard.generator import DashboardGenerator

    gen = DashboardGenerator.__new__(DashboardGenerator)
    gen._is_msm_gated = lambda: False

    class FakeSig:
        ticker = "SPY"
        base_weight = 0.46
        signal = 0.1
        adjustment = 0.0
        realized_vol = 0.15
        vol_scaled_position = 0.1

    class FakeOverlay:
        def compute_signal(self, ticker):
            return FakeSig()

    with patch(
        "src.signals.tsmom_overlay.TSMOMOverlay",
        FakeOverlay,
    ), patch(
        "src.dashboard.generator._generator_git_sha_short",
        return_value="tsmomsha01",
    ):
        out = DashboardGenerator.generate_tsmom_json(gen)

    assert out is not None
    payload = _read(Path(out))
    assert payload["generator_git_sha"] == "tsmomsha01"
    assert payload["generator_git_sha_status"] == "full_generate"


def test_cross_asset_rv_json_stamps_generator_git_sha(public_dir):
    from src.dashboard.generator import DashboardGenerator

    gen = DashboardGenerator.__new__(DashboardGenerator)
    gen.conn = MagicMock()
    gen.conn.cursor.return_value.fetchone.return_value = ("normal",)

    class FakePair:
        def to_dict(self):
            return {"pair": "SPY/TLT", "z": 0.0}

    class FakeSignal:
        risk_on_score = 0.1
        pairs = {"a": FakePair()}
        avg_z_score = 0.0
        max_divergence = 0.0
        num_diverged = 0
        total_pairs = 1
        available_pair_count = 1
        unavailable_pair_count = 0
        unavailable_pairs = []
        missing_symbols = []
        duration_score = 0.0
        overall_conviction = 0.1

    class FakeScanner:
        def scan_all(self):
            return FakeSignal()

    with patch(
        "src.signals.cross_asset_relative_value.CrossAssetRVScanner",
        FakeScanner,
    ), patch(
        "src.dashboard.generator._generator_git_sha_short",
        return_value="crossrvsha1",
    ):
        out = DashboardGenerator.generate_cross_asset_rv_json(gen)

    assert out is not None
    payload = _read(Path(out))
    assert payload["generator_git_sha"] == "crossrvsha1"
    assert payload["generator_git_sha_status"] == "full_generate"


def test_black_litterman_json_stamps_generator_git_sha(public_dir):
    from src.dashboard.generator import DashboardGenerator

    gen = DashboardGenerator.__new__(DashboardGenerator)
    gen.conn = MagicMock()
    gen.conn.execute.return_value.fetchall.return_value = []
    gen._canonicalize_public_weights = lambda w, **k: {
        "weights": dict(w) if isinstance(w, dict) else {},
        "excluded_assets": [],
        "zero_weight_assets": [],
    }
    gen._build_advisory_allocation_artifact_role = lambda **k: {
        "role": "advisory",
        "live_authoritative": False,
    }
    gen._flatten_advisory_authority = lambda a: a

    class FakeViews:
        symbols = ["SPY", "GLD", "TLT"]
        absolute_views = {}
        view_confidences = []

    class FakeVoter:
        def get_bl_views(self):
            return {
                "views": FakeViews(),
                "tau": 0.15,
                "health_scores_used": {},
                "equity_bias": 0.0,
                "duration_bias": 0.0,
                "gold_bias": 0.0,
            }

    with patch(
        "src.strategy.ensemble_voter.EnsembleVoter",
        FakeVoter,
    ), patch(
        "src.dashboard.generator._generator_git_sha_short",
        return_value="blsha00001",
    ), patch(
        "src.dashboard.generator.BASE_ALLOCATION",
        {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
    ):
        out = DashboardGenerator.generate_black_litterman_json(gen)

    assert out is not None
    payload = _read(Path(out))
    assert payload["generator_git_sha"] == "blsha00001"
    assert payload["generator_git_sha_status"] == "full_generate"


def test_regime_gate_json_stamps_generator_git_sha(public_dir, monkeypatch):
    from src.dashboard.generator import DashboardGenerator

    gen = DashboardGenerator.__new__(DashboardGenerator)
    gen._resolve_current_regime_for_gate = lambda: ("normal", 0.8, "test")
    gen._persist_regime_state = lambda *a, **k: None
    gen._dedupe_preserve_order = lambda xs: list(dict.fromkeys(xs))
    gen._load_price_data = lambda: None

    class FakeGate:
        min_dwell_days = 5

        def get_gate_summary(self):
            return {"tsmom": set()}

        def get_active_signal_names(self, all_signals, regime_name):
            return list(all_signals)

    with patch(
        "src.signals.regime_gate.RegimeGate",
        FakeGate,
    ), patch(
        "src.dashboard.generator._generator_git_sha_short",
        return_value="regimegate1",
    ), patch(
        "src.dashboard.generator.DATA_DIR",
        public_dir.parent / "data",
    ):
        (public_dir.parent / "data").mkdir(exist_ok=True)
        out = DashboardGenerator.generate_regime_gate_json(gen)

    assert out is not None
    payload = _read(Path(out))
    assert payload["generator_git_sha"] == "regimegate1"
    assert payload["generator_git_sha_status"] == "full_generate"
