"""Unit tests for src.monitor.signal_ownership."""

from __future__ import annotations

from src.monitor.signal_ownership import (
    OPTIONAL_ADVISORY_SIGNALS,
    SIGNAL_OWNERSHIP,
    annotate_unavailable_signals,
    blocks_all_fresh,
    optional_advisory_signals,
    recovery_summary,
    signal_criticality,
)


class TestOptionalAdvisorySignals:
    """Test OPTIONAL_ADVISORY_SIGNALS set and getters."""

    def test_immutable_frozenset(self):
        assert isinstance(OPTIONAL_ADVISORY_SIGNALS, frozenset)
        assert optional_advisory_signals() == OPTIONAL_ADVISORY_SIGNALS

    def test_expected_signals_present(self):
        expected = {
            "behavioral_sentiment",
            "calendar_seasonality",
            "crypto_allocation",
            "factor_rotation",
            "stacking_ensemble",
            "convexity_harvest",
            "llm_sentiment",
            "sector_rotation",
            "kurtosis_regime",
            "volatility_parity",
            "collar",
            "bond_momentum",
            "risk_decomposition",
            "two_stage_regime",
            "bocd_regime",
            "regime_transition",
            "hedge_selector",
            "fred_macro",
        }
        assert expected.issubset(OPTIONAL_ADVISORY_SIGNALS)


class TestSignalOwnershipStructure:
    """Test schema and completeness of SIGNAL_OWNERSHIP registry."""

    def test_all_keys_have_required_fields(self):
        for key, entry in SIGNAL_OWNERSHIP.items():
            assert "job" in entry
            assert "make_target" in entry
            assert "module" in entry
            assert "recovery" in entry
            assert entry["job"].startswith("portfolio-lab-") or entry["job"] != ""


class TestCriticalityAndFreshnessGuards:
    """Test signal_criticality and blocks_all_fresh."""

    def test_optional_advisory_signal_criticality(self):
        assert signal_criticality("behavioral_sentiment") == "optional_advisory"
        assert signal_criticality("fred_macro") == "optional_advisory"
        assert blocks_all_fresh("behavioral_sentiment") is False
        assert blocks_all_fresh("fred_macro") is False

    def test_required_signals_and_unknown_fail_closed(self):
        # Known required signals
        assert signal_criticality("ensemble_voting") == "required"
        assert signal_criticality("garch_cvar") == "required"
        assert blocks_all_fresh("ensemble_voting") is True
        assert blocks_all_fresh("garch_cvar") is True

        # Unknown signals fail closed to required
        assert signal_criticality("completely_unknown_signal_xyz") == "required"
        assert blocks_all_fresh("completely_unknown_signal_xyz") is True


class TestAnnotateUnavailableSignals:
    """Test annotate_unavailable_signals with ML and FRED flags."""

    def test_empty_or_none_returns_empty_list(self):
        assert annotate_unavailable_signals(None) == []
        assert annotate_unavailable_signals([]) == []

    def test_ml_flag_handling(self):
        # When ML is off
        rows_off = annotate_unavailable_signals(["behavioral_sentiment", "stacking_ensemble"], ml_enabled=False)
        for r in rows_off:
            assert r["intentional_when_ml_off"] is True
            assert r["intentional_lab_gap"] is True

        # When ML is on
        rows_on = annotate_unavailable_signals(["behavioral_sentiment", "stacking_ensemble"], ml_enabled=True)
        for r in rows_on:
            assert r["intentional_when_ml_off"] is False
            assert r["intentional_lab_gap"] is False

    def test_fred_flag_handling(self, monkeypatch):
        # FRED unconfigured
        rows_unconf = annotate_unavailable_signals(["two_stage_regime", "fred_macro"], fred_configured=False)
        for r in rows_unconf:
            assert r["intentional_when_fred_unconfigured"] is True
            assert r["intentional_lab_gap"] is True

        # FRED configured
        rows_conf = annotate_unavailable_signals(["two_stage_regime", "fred_macro"], fred_configured=True)
        for r in rows_conf:
            assert r["intentional_when_fred_unconfigured"] is False
            assert r["intentional_lab_gap"] is False

        # Environment variable fallback
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        rows_env_off = annotate_unavailable_signals(["fred_macro"], fred_configured=None)
        assert rows_env_off[0]["intentional_when_fred_unconfigured"] is True

        monkeypatch.setenv("FRED_API_KEY", "test_key")
        rows_env_on = annotate_unavailable_signals(["fred_macro"], fred_configured=None)
        assert rows_env_on[0]["intentional_when_fred_unconfigured"] is False

    def test_unknown_signal_annotation(self):
        rows = annotate_unavailable_signals(["unknown_sig_abc"])
        assert len(rows) == 1
        r = rows[0]
        assert r["signal"] == "unknown_sig_abc"
        assert r["job"] == "unknown"
        assert r["make_target"] == "unknown"
        assert r["criticality"] == "required"
        assert r["blocks_all_fresh"] is True
        assert "make ops-regen" in r["recovery"]


class TestRecoverySummary:
    """Test recovery_summary aggregation logic."""

    def test_empty_rows_summary(self):
        summary = recovery_summary([])
        assert summary["actionable_unavailable_count"] == 0
        assert summary["optional_advisory_unavailable_count"] == 0
        assert summary["intentional_ml_off_count"] == 0
        assert summary["intentional_lab_gap_count"] == 0
        assert summary["jobs_to_rerun"] == []
        assert summary["make_targets"] == []
        assert summary["suggested_commands"] == ["make overlay-signals", "make ops-regen"]

    def test_mixed_rows_aggregation(self):
        rows = annotate_unavailable_signals(
            ["ensemble_voting", "garch_cvar", "behavioral_sentiment", "fred_macro"],
            ml_enabled=False,
            fred_configured=False,
        )
        summary = recovery_summary(rows)
        # ensemble_voting and garch_cvar are required/actionable
        # behavioral_sentiment and fred_macro are optional/intentional lab gaps
        assert summary["actionable_unavailable_count"] == 2
        assert summary["optional_advisory_unavailable_count"] == 2
        assert summary["intentional_ml_off_count"] == 1
        assert summary["intentional_lab_gap_count"] == 2
        assert summary["jobs_to_rerun"] == ["portfolio-lab-dashboard", "portfolio-lab-garch-risk"]
        assert summary["make_targets"] == ["dashboard", "garch-risk"]
        assert summary["suggested_commands"] == ["make dashboard", "make garch-risk"]
        assert "kill_switch" in summary["note"]
