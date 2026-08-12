#!/usr/bin/env python3
"""
Regression tests for the C3 regime/authority mixin extracted by Item 21
(2026-08-12): ``src/dashboard/sections_regime.py`` ``_RegimeAuthorityMixin``
(test file owed by the TEST-GAP coverage gap — module has zero direct test
references).

A1: getattr smoke — all 9 moved names resolve via BOTH ``DashboardGenerator``
    (MRO) and ``_RegimeAuthorityMixin``.
A2: behavior-equality — canned fixtures for the pure statics
    (``_canonicalize_public_weights``, ``_build_advisory_allocation_artifact_role``,
    ``_flatten_advisory_authority``, ``_dedupe_preserve_order``), the champion
    baseline resolver, the diagnostic/authority builders, the kill-switch-aware
    surface roles (tmp data_dir, both disabled and enabled), and the
    @classmethod ``_update_regime_authority_availability`` (mixin surface for
    unavailable/stale paths; generator surface for present/error paths —
    ``cls._is_unavailable_signal_block`` lives on the hedge mixin).
"""
from src.dashboard.generator import DashboardGenerator
from src.dashboard.sections_regime import _RegimeAuthorityMixin
from src.paths import BASE_ALLOCATION

REGIME_NAMES = (
    "_build_allocation_surface_roles",
    "_build_advisory_allocation_artifact_role",
    "_flatten_advisory_authority",
    "_canonicalize_public_weights",
    "_resolve_live_target_allocations_for_regime",
    "_build_regime_allocation_diagnostic",
    "_build_regime_authority",
    "_dedupe_preserve_order",
    "_update_regime_authority_availability",
)


def test_a1_getattr_resolution_via_both_surfaces():
    """All 9 C3 names resolve via DashboardGenerator MRO and the mixin."""
    for name in REGIME_NAMES:
        assert hasattr(DashboardGenerator, name), name
        assert hasattr(_RegimeAuthorityMixin, name), name


def test_a2_canonicalize_public_weights_canned_inputs():
    """Uppercase keys, float coercion, excluded + zero-weight diagnostics."""
    for surface in (_RegimeAuthorityMixin, DashboardGenerator):
        assert surface._canonicalize_public_weights(
            {"spy": 0.46, "GLD": "0.38", "tlt": 0.16}
        ) == {
            "weights": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            "excluded_assets": [],
            "zero_weight_assets": [],
        }
        assert surface._canonicalize_public_weights(
            {"SPY": 0.46, "QQQ": "garbage", "GLD": 0.0}
        ) == {
            "weights": {"SPY": 0.46, "GLD": 0.0, "TLT": 0.0},
            "excluded_assets": ["QQQ"],
            "zero_weight_assets": ["GLD", "TLT"],
        }
        for empty in (None, {}):
            assert surface._canonicalize_public_weights(empty) == {
                "weights": {"SPY": 0.0, "GLD": 0.0, "TLT": 0.0},
                "excluded_assets": [],
                "zero_weight_assets": ["SPY", "GLD", "TLT"],
            }


def test_a2_build_advisory_allocation_artifact_role_canned():
    """Standalone artifacts are always advisory/non-routed (both)."""
    for surface in (_RegimeAuthorityMixin, DashboardGenerator):
        result = surface._build_advisory_allocation_artifact_role(
            "black_litterman", "weights"
        )
        assert result["schema_version"] == "allocation-artifact-role/v1"
        assert result["surface"] == "black_litterman"
        assert result["allocation_field"] == "weights"
        assert result["runtime_role"] == "advisory_non_routed"
        assert result["live_authoritative"] is False
        assert result["routed"] is False
        assert result["routed_by"] is None
        assert result["canonical_controller"] == "signals.json.target_allocations"
        assert result["routed_surface"] == "target_allocations"
        assert (
            result["routed_surface_path"] == "public/data/signals.json#target_allocations"
        )
        assert "black_litterman" in result["description"]


def test_a2_flatten_advisory_authority_canned():
    """Top-level mirrors of the nested authority block (both)."""
    for surface in (_RegimeAuthorityMixin, DashboardGenerator):
        nested = {
            "runtime_role": "advisory_non_routed",
            "live_authoritative": False,
            "routed": False,
            "routed_by": None,
            "canonical_controller": "signals.json.target_allocations",
            "routed_surface": "target_allocations",
        }
        assert surface._flatten_advisory_authority(nested) == nested
        assert surface._flatten_advisory_authority({}) == {
            "runtime_role": None,
            "live_authoritative": None,
            "routed": None,
            "routed_by": None,
            "canonical_controller": None,
            "routed_surface": None,
        }


def test_a2_resolve_live_target_allocations_for_regime():
    """Only champion baseline routes, regardless of regime; fresh copy."""
    for surface in (_RegimeAuthorityMixin, DashboardGenerator):
        for regime in ("normal", "crisis", "high_vol"):
            result = surface._resolve_live_target_allocations_for_regime(regime)
            assert result == {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}, regime
            assert result == dict(BASE_ALLOCATION)
            assert result is not BASE_ALLOCATION  # no aliasing


def test_a2_dedupe_preserve_order_canned():
    """First-occurrence dedupe with string keys (both)."""
    for surface in (_RegimeAuthorityMixin, DashboardGenerator):
        assert surface._dedupe_preserve_order(["a", "b", "a", "c", "b"]) == [
            "a",
            "b",
            "c",
        ]
        assert surface._dedupe_preserve_order([]) == []
        assert surface._dedupe_preserve_order([1, "1", 2, 2]) == ["1", "2"]


def test_a2_build_regime_allocation_diagnostic_canned():
    """Diagnostic candidates mirror REGIME_OVERRIDES; never routed (both)."""
    for surface in (_RegimeAuthorityMixin, DashboardGenerator):
        normal = surface._build_regime_allocation_diagnostic("normal")
        assert normal["schema_version"] == "regime-allocation-diagnostic/v1"
        assert normal["role"] == "advisory_non_routed"
        assert normal["live_authoritative"] is False
        assert normal["routed"] is False
        assert normal["allocation_regime"] == "normal"
        assert normal["candidate_target_allocations"] == dict(BASE_ALLOCATION)

        crisis = surface._build_regime_allocation_diagnostic("crisis")
        assert crisis["allocation_regime"] == "crisis"
        assert crisis["candidate_target_allocations"] == {
            "SPY": 0.20,
            "GLD": 0.50,
            "TLT": 0.30,
        }

        # vol_spike alias normalizes to high_vol; candidate still its own row.
        spike = surface._build_regime_allocation_diagnostic("vol_spike")
        assert spike["allocation_regime"] == "high_vol"
        assert spike["candidate_target_allocations"] == {
            "SPY": 0.30,
            "GLD": 0.45,
            "TLT": 0.25,
        }

        # Unknown regime: candidate falls back to champion baseline.
        unknown = surface._build_regime_allocation_diagnostic("foo")
        assert unknown["allocation_regime"] == "foo"
        assert unknown["candidate_target_allocations"] == dict(BASE_ALLOCATION)


def test_a2_build_regime_authority_canned():
    """Authority block pins: controller, regime, shadow signals (both)."""
    for surface in (_RegimeAuthorityMixin, DashboardGenerator):
        result = surface._build_regime_authority(
            "high_vol", {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        )
        assert result["schema_version"] == "regime-authority/v1"
        assert result["live_controller"] == "signals.json.target_allocations"
        assert result["live_controller_module"] == "src.broker.order_router"
        assert result["live_regime"] == "high_vol"
        assert result["allocation_regime"] == "high_vol"
        assert result["routed_surface"] == "target_allocations"
        assert result["target_allocations"] == {
            "SPY": 0.46,
            "GLD": 0.38,
            "TLT": 0.16,
        }
        assert result["regime_controller"] == "classify_vix_regime"
        assert result["regime_routed"] is False
        assert set(result["advanced_regime_signals"]) == {
            "two_stage_regime",
            "bocd_regime",
            "regime_transition",
        }
        for entry in result["advanced_regime_signals"].values():
            assert entry["role"] == "advisory_shadow"
            assert entry["routed"] is False
            assert entry["availability"] == "unknown"
            assert entry["published"] is False


def test_a2_build_allocation_surface_roles_no_kill(tmp_path):
    """No kill_switch.json → target_allocations stays execution_routed."""
    for surface in (_RegimeAuthorityMixin, DashboardGenerator):
        roles = surface._build_allocation_surface_roles(data_dir=tmp_path)
        assert roles["schema_version"] == "allocation-surface-roles/v1"
        assert roles["routed_surface"] == "target_allocations"
        assert roles["routed_by"] == "src.broker.order_router"

        target = roles["surfaces"]["target_allocations"]
        assert target["role"] == "execution_routed"
        assert target["routed"] is True
        assert target["live_authoritative"] is True
        assert target["canonical_controller"] == "signals.json.target_allocations"
        assert "execution_blocked" not in target

        advisory = roles["surfaces"]["ensemble_voting"]
        assert advisory["role"] == "advisory_non_routed"
        assert advisory["routed"] is False
        assert advisory["live_authoritative"] is False
        assert roles["surfaces"]["calendar_seasonality"][
            "applies_to_target_allocations"
        ] is False
        assert roles["surfaces"]["factor_rotation"]["allocation_field"] == "allocation"


def test_a2_build_allocation_surface_roles_kill_enabled(tmp_path):
    """Kill switch on → routed surface disclosed execution-blocked."""
    kill_file = tmp_path / "kill_switch.json"
    kill_file.write_text(
        '{"enabled": true, "level": "halt"}', encoding="utf-8"
    )
    for surface in (_RegimeAuthorityMixin, DashboardGenerator):
        roles = surface._build_allocation_surface_roles(data_dir=tmp_path)
        target = roles["surfaces"]["target_allocations"]
        assert target["live_authoritative"] is False
        assert target["execution_blocked"] is True
        assert target["kill_switch_enabled"] is True
        assert target["kill_switch_level"] == "halt"
        assert target["role"] == "execution_blocked"
        # Still the routing surface — blocked, not replaced.
        assert target["routed"] is True
        assert target["description"].startswith(
            "Order routing blocked by active kill switch (level=halt)."
        )
        assert roles["routed_surface"] == "target_allocations"


def test_a2_update_regime_authority_availability_noop():
    """Missing regime_authority / advanced block → silent no-op."""
    assert (
        _RegimeAuthorityMixin._update_regime_authority_availability({}) is None
    )
    assert (
        _RegimeAuthorityMixin._update_regime_authority_availability(
            {"regime_authority": {"advanced_regime_signals": "not-a-dict"}}
        )
        is None
    )


def test_a2_update_regime_authority_availability_unavailable_and_stale():
    """Mixin surface: unavailable/stale/None signals get typed disclosure."""
    output = {
        "regime_authority": {
            "advanced_regime_signals": {
                "two_stage_regime": {},
                "bocd_regime": {},
                "regime_transition": {},
            }
        },
        "staleness": {
            "unavailable_signals": ["two_stage_regime"],
            "stale_signals": ["bocd_regime"],
        },
        # Present block still lands unavailable when listed as such.
        "two_stage_regime": {"status": "ok"},
        "bocd_regime": {"status": "ok"},
    }
    _RegimeAuthorityMixin._update_regime_authority_availability(output)
    advanced = output["regime_authority"]["advanced_regime_signals"]
    assert advanced["two_stage_regime"] == {
        "availability": "unavailable",
        "published": False,
        "description": "Unavailable in this snapshot; not live order-routing authority.",
    }
    assert advanced["bocd_regime"]["availability"] == "stale"
    assert advanced["bocd_regime"]["published"] is False
    assert advanced["regime_transition"]["availability"] == "unavailable"
    assert advanced["regime_transition"]["published"] is False


def test_a2_update_regime_authority_availability_present_and_error():
    """Generator surface: fresh → present; degraded placeholder → error."""
    output = {
        "regime_authority": {
            "advanced_regime_signals": {
                "two_stage_regime": {},
                "bocd_regime": {},
            }
        },
        "staleness": {"unavailable_signals": [], "stale_signals": []},
        "two_stage_regime": {"status": "ok", "generated_at": "2026-07-06T10:00:00Z"},
        "bocd_regime": {"status": "unavailable"},
    }
    DashboardGenerator._update_regime_authority_availability(output)
    advanced = output["regime_authority"]["advanced_regime_signals"]
    assert advanced["two_stage_regime"]["availability"] == "present"
    assert advanced["two_stage_regime"]["published"] is True
    assert advanced["bocd_regime"]["availability"] == "error"
    assert advanced["bocd_regime"]["published"] is False
