"""Authority dual-write gate: refuse hollow signals; same-bytes multi-dest fan-out."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


def test_validate_authority_payload_rejects_hollow():
    from src.monitor.signal_authority import (
        AuthorityValidationError,
        validate_authority_payload,
    )

    with pytest.raises(AuthorityValidationError):
        validate_authority_payload({})
    with pytest.raises(AuthorityValidationError):
        validate_authority_payload({"target_allocations": {}})
    with pytest.raises(AuthorityValidationError):
        validate_authority_payload({"target_allocations": {"SPY": 1.0}})


def test_validate_authority_payload_accepts_champion():
    from src.monitor.signal_authority import validate_authority_payload

    payload = {
        "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        "health": {"status": "ok"},
    }
    validate_authority_payload(payload)  # does not raise


def test_champion_policy_accepts_exact_baseline():
    from src.monitor.signal_authority import is_champion_target_allocations

    payload = {"target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}}

    assert is_champion_target_allocations(payload) is True


def test_champion_policy_rejects_vol_spike_override():
    from src.monitor.signal_authority import is_champion_target_allocations

    payload = {"target_allocations": {"SPY": 0.30, "GLD": 0.45, "TLT": 0.25}}

    assert is_champion_target_allocations(payload) is False


def test_write_signals_multi_dest_refuses_hollow(tmp_path):
    from src.monitor.signal_authority import (
        AuthorityValidationError,
        write_signals_multi_dest,
    )

    public = tmp_path / "www" / "signals.json"
    private = tmp_path / "data" / "signals.json"
    repo = tmp_path / "repo" / "signals.json"
    public.parent.mkdir()
    private.parent.mkdir()
    repo.parent.mkdir()
    good = {"target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}}
    public.write_text(json.dumps(good), encoding="utf-8")
    private.write_text(json.dumps(good), encoding="utf-8")
    before_pub = public.read_text(encoding="utf-8")
    before_priv = private.read_text(encoding="utf-8")

    with pytest.raises(AuthorityValidationError):
        write_signals_multi_dest(
            {"health": {"status": "ok"}},  # no TA
            public_path=public,
            private_path=private,
            repo_path=repo,
        )
    assert public.read_text(encoding="utf-8") == before_pub
    assert private.read_text(encoding="utf-8") == before_priv
    assert not repo.exists() or repo.read_text(encoding="utf-8") != "{}"


def test_write_signals_multi_dest_same_bytes_all_dests(tmp_path):
    from src.monitor.signal_authority import write_signals_multi_dest

    public = tmp_path / "www" / "signals.json"
    private = tmp_path / "data" / "signals.json"
    repo = tmp_path / "repo" / "public" / "data" / "signals.json"
    public.parent.mkdir(parents=True)
    private.parent.mkdir(parents=True)
    repo.parent.mkdir(parents=True)

    payload = {
        "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        "content_patch_source": "test_patch",
        "alternative_data_projection": {"source": "test"},
    }
    result = write_signals_multi_dest(
        payload,
        public_path=public,
        private_path=private,
        repo_path=repo,
    )
    assert result.wrote_public is True
    assert result.wrote_private is True
    assert result.wrote_repo is True
    pub_text = public.read_text(encoding="utf-8")
    assert private.read_text(encoding="utf-8") == pub_text
    assert repo.read_text(encoding="utf-8") == pub_text
    for path in (public, private, repo):
        d = json.loads(path.read_text(encoding="utf-8"))
        assert d["target_allocations"] == {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}


def test_serialize_signals_payload_emits_strict_browser_json():
    from src.monitor.signal_authority import serialize_signals_payload

    payload = {
        "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        "diagnostics": {
            "infinite_half_life": float("inf"),
            "missing_score": float("nan"),
            "finite_score": 1.25,
        },
    }

    body = serialize_signals_payload(payload)

    assert "Infinity" not in body
    assert "NaN" not in body
    assert json.loads(body)["diagnostics"] == {
        "infinite_half_life": None,
        "missing_score": None,
        "finite_score": 1.25,
    }


def test_health_kill_refresh_preserves_ta_on_private_twin(tmp_path, monkeypatch):
    """Case B: partial health kill refresh keeps TA on private even if it was hollow."""
    from src.monitor import health_check as hc

    public_dir = tmp_path / "www"
    data_dir = tmp_path / "data"
    public_dir.mkdir()
    data_dir.mkdir()
    full = {
        "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        "health": {"status": "ok", "kill_switch_enabled": False},
        "generator_git_sha": "fulltipsha001",
        "generator_git_sha_status": "full_generate",
    }
    (public_dir / "signals.json").write_text(json.dumps(full), encoding="utf-8")
    # Hollow private (historical wipe)
    (data_dir / "signals.json").write_text(json.dumps({"health": {}}), encoding="utf-8")

    monkeypatch.setattr(hc, "PUBLIC_DATA_DIR", public_dir)
    monkeypatch.setattr(hc, "DATA_DIR", data_dir)

    with patch.object(
        hc,
        "_disk_kill_and_open_incidents",
        return_value=({"enabled": False, "level": None}, {"open_count": 0, "status": "ok"}),
    ), patch(
        "src.dashboard.kill_authority.project_compact_kill_fields",
        return_value={"kill_switch_enabled": False, "open_incidents_count": 0},
    ):
        hc.refresh_signals_health_kill_fields(
            {"status": "ok"},
            public_dir=public_dir,
            data_dir=data_dir,
        )

    priv = json.loads((data_dir / "signals.json").read_text(encoding="utf-8"))
    pub = json.loads((public_dir / "signals.json").read_text(encoding="utf-8"))
    assert priv["target_allocations"] == {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
    assert pub["target_allocations"] == {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
    assert priv.get("content_patch_source") == "health_kill_refresh"
    # same-bytes authority twin
    assert priv["target_allocations"] == pub["target_allocations"]


def test_alt_data_projection_dual_writes_private_and_preserves_ta(tmp_path, monkeypatch):
    """Case F: alt-data partial must dual-write private + keep TA."""
    from src.dashboard import generator as gen_mod

    data = tmp_path / "data"
    public = tmp_path / "www"
    repo_public = tmp_path / "repo" / "public" / "data"
    data.mkdir()
    public.mkdir()
    repo_public.mkdir(parents=True)
    (data / "signals").mkdir()
    full = {
        "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        "alternative_data": {"old": True},
        "generator_git_sha": "oldfull",
        "generator_git_sha_status": "full_generate",
    }
    (public / "signals.json").write_text(json.dumps(full), encoding="utf-8")
    (data / "signals.json").write_text(json.dumps(full), encoding="utf-8")
    producer = {
        "timestamp": "2026-07-22T18:40:00+00:00",
        "signals": {},
        "confidence": 0.5,
    }
    (data / "signals" / "alternative_data_latest.json").write_text(
        json.dumps(producer), encoding="utf-8"
    )

    monkeypatch.setattr(gen_mod, "DATA_DIR", data)
    monkeypatch.setattr(gen_mod, "PUBLIC_DIR", public)
    # Soft-mirror dest: repo checkout public/data under project root mock
    monkeypatch.setattr(
        "src.monitor.signal_authority.default_repo_signals_path",
        lambda: repo_public / "signals.json",
    )

    with patch(
        "src.dashboard.generator.project_alternative_data_signal",
        return_value={"timestamp": producer["timestamp"], "bias": 0.1},
    ):
        ok = gen_mod.refresh_public_alternative_data_projection(
            data_dir=data, public_dir=public
        )
    assert ok is True
    pub = json.loads((public / "signals.json").read_text(encoding="utf-8"))
    priv = json.loads((data / "signals.json").read_text(encoding="utf-8"))
    assert pub["target_allocations"] == {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
    assert priv["target_allocations"] == pub["target_allocations"]
    assert pub.get("content_patch_source") == "bounded_alt_data_refresh"
    assert priv.get("content_patch_source") == "bounded_alt_data_refresh"
    assert "alternative_data_projection" in pub
    assert "alternative_data_projection" in priv
    # same serialized body
    assert (public / "signals.json").read_text(encoding="utf-8") == (
        data / "signals.json"
    ).read_text(encoding="utf-8")


def test_atomic_write_text_mode_644_not_600(tmp_path):
    """Case CU: multi-dest atomic write must leave world-readable 0o644 (not mkstemp 0600)."""
    from src.monitor.signal_authority import write_signals_multi_dest

    public = tmp_path / "www" / "signals.json"
    private = tmp_path / "data" / "signals.json"
    repo = tmp_path / "repo" / "signals.json"
    public.parent.mkdir(parents=True)
    private.parent.mkdir(parents=True)
    repo.parent.mkdir(parents=True)

    payload = {
        "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        "content_patch_source": "mode_test",
    }
    write_signals_multi_dest(
        payload,
        public_path=public,
        private_path=private,
        repo_path=repo,
    )
    modes = [(p.stat().st_mode & 0o777) for p in (public, private, repo)]
    assert modes == [0o644, 0o644, 0o644], list(map(oct, modes))
    # Content equality still holds
    body = public.read_bytes()
    assert private.read_bytes() == body
    assert repo.read_bytes() == body


def test_generate_signals_json_serialize_once_multi_dest(tmp_path, monkeypatch):
    """Case CV: full generate uses multi-dest once (public==private==repo, 644)."""
    from src.dashboard import generator as gen_mod

    public = tmp_path / "www"
    data = tmp_path / "data"
    repo = tmp_path / "repo" / "public" / "data"
    public.mkdir()
    data.mkdir()
    repo.mkdir(parents=True)

    monkeypatch.setattr(gen_mod, "PUBLIC_DIR", public)
    monkeypatch.setattr(gen_mod, "DATA_DIR", data)
    monkeypatch.setattr(
        "src.monitor.signal_authority.default_repo_signals_path",
        lambda: repo / "signals.json",
    )

    champion = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
    fake_output = {
        "target_allocations": champion,
        "regime": {"regime": "normal"},
        "generator_git_sha_status": "full_generate",
    }

    class _Stub:
        def _load_signal_generation_context(self):
            return {}

        def _build_base_signal_sections(self, context):
            return dict(fake_output)

        def _build_optional_signal_sections(self, output, context):
            return output

        def _apply_signal_postprocessors(self, output, context):
            return output

    # Bind stub methods onto a real DashboardGenerator instance lightly
    gen = gen_mod.DashboardGenerator.__new__(gen_mod.DashboardGenerator)
    gen._load_signal_generation_context = _Stub()._load_signal_generation_context
    gen._build_base_signal_sections = _Stub()._build_base_signal_sections
    gen._build_optional_signal_sections = _Stub()._build_optional_signal_sections
    gen._apply_signal_postprocessors = _Stub()._apply_signal_postprocessors

    with patch(
        "src.dashboard.generator._attach_signal_metadata", side_effect=lambda o: o
    ), patch(
        "src.dashboard.generator._finalize_signal_metadata", side_effect=lambda o: o
    ), patch(
        "src.dashboard.generator.validate_all_signals", side_effect=lambda o: o
    ), patch(
        "src.monitor.decision_registry.record_dashboard_cycle_decision",
        side_effect=ImportError("skip"),
    ):
        out = gen.generate_signals_json()

    assert out == public / "signals.json"
    pub_b = (public / "signals.json").read_bytes()
    priv_b = (data / "signals.json").read_bytes()
    repo_b = (repo / "signals.json").read_bytes()
    assert pub_b == priv_b == repo_b
    for p in (public / "signals.json", data / "signals.json", repo / "signals.json"):
        assert (p.stat().st_mode & 0o777) == 0o644
        d = json.loads(p.read_text(encoding="utf-8"))
        assert d["target_allocations"] == champion
