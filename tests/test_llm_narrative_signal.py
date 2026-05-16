"""
Tests for v7.01 LLM Macro/Narrative Signal Generator.

Tests the rule-based macro signal scoring, calendar, and FOMC tone analysis.
No ML dependencies — safe to run anytime.
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.signals.llm_narrative_signal import (
    LLMNarrativeSignalGenerator,
    MacroDataType,
    NarrativeTone,
    MacroReleaseReading,
    NarrativeSignal,
    MACRO_CALENDAR_2026,
)


class TestMacroDataType:
    """Test macro data type enum."""

    def test_enum_values(self):
        assert MacroDataType.CPI.value == "cpi"
        assert MacroDataType.NFP.value == "nfp"
        assert MacroDataType.GDP.value == "gdp"
        assert MacroDataType.FOMC.value == "fomc"
        assert MacroDataType.ISM_MANUFACTURING.value == "ism_manufacturing"

    def test_all_types_defined(self):
        expected = [
            "cpi", "core_cpi", "ppi", "nfp", "unemployment",
            "gdp", "fomc", "fomc_minutes", "ism_manufacturing",
            "ism_services", "retail_sales", "industrial_production",
            "consumer_confidence", "housing_starts", "jolts",
        ]
        actual = [m.value for m in MacroDataType]
        for e in expected:
            assert e in actual, f"Missing {e}"


class TestNarrativeTone:
    """Test narrative tone enum."""

    def test_enum_values(self):
        assert NarrativeTone.HAWKISH.value == "hawkish"
        assert NarrativeTone.DOVISH.value == "dovish"
        assert NarrativeTone.NEUTRAL.value == "neutral"
        assert NarrativeTone.POSITIVE.value == "positive"
        assert NarrativeTone.CAUTIOUS.value == "cautious"


class TestMacroReleaseReading:
    """Test macro release reading dataclass."""

    def test_default_creation(self):
        reading = MacroReleaseReading(
            data_type=MacroDataType.CPI,
            release_date="2026-05-13",
            actual=3.2,
            consensus=3.0,
            surprise=0.0667,
            surprise_z=0.44,
            tone=NarrativeTone.NEUTRAL,
            narrative_score=0.0,
            confidence=0.5,
            note="Inflation in line",
        )
        assert reading.data_type == MacroDataType.CPI
        assert reading.narrative_score == 0.0
        assert reading.confidence == 0.5

    def test_minimal_creation(self):
        reading = MacroReleaseReading(
            data_type=MacroDataType.NFP,
            release_date="2026-05-08",
            actual=None,
            consensus=None,
            surprise=None,
            surprise_z=None,
            tone=NarrativeTone.NEUTRAL,
            narrative_score=0.0,
            confidence=0.0,
        )
        assert reading.note == ""


class TestMacroCalendar:
    """Test the hardcoded macro calendar."""

    def test_has_all_types(self):
        expected_types = ["cpi", "nfp", "fomc", "gdp", "ism_manufacturing"]
        for t in expected_types:
            assert t in MACRO_CALENDAR_2026, f"Missing {t}"

    def test_cpi_12_entries(self):
        assert len(MACRO_CALENDAR_2026["cpi"]) == 12

    def test_nfp_12_entries(self):
        assert len(MACRO_CALENDAR_2026["nfp"]) == 12

    def test_fomc_8_entries(self):
        """FOMC meets ~8 times per year."""
        assert len(MACRO_CALENDAR_2026["fomc"]) == 8

    def test_gdp_12_entries(self):
        """4 quarters × 3 releases (advance, revised, final)."""
        assert len(MACRO_CALENDAR_2026["gdp"]) == 12

    def test_ism_12_entries(self):
        assert len(MACRO_CALENDAR_2026["ism_manufacturing"]) == 12

    def test_all_dates_in_2026(self):
        for data_type, releases in MACRO_CALENDAR_2026.items():
            for r in releases:
                assert "date" in r
                assert "name" in r
                assert r["date"].startswith("2026"), f"Non-2026 date: {r['date']}"


class TestLLMNarrativeSignalGenerator:
    """Test the narrative signal generator with state isolation."""

    def _make_gen(self):
        """Create a generator with fresh state (no loaded data)."""
        gen = LLMNarrativeSignalGenerator()
        gen._recent_releases = []
        gen._latest_fomc_tone = NarrativeTone.NEUTRAL
        return gen

    def test_initial_state(self):
        gen = self._make_gen()
        assert gen._latest_fomc_tone == NarrativeTone.NEUTRAL
        assert gen._recent_releases == []

    def test_signal_no_data(self):
        gen = self._make_gen()
        signal = gen.generate_signal()
        assert signal.composite_score == 0.0
        assert signal.confidence == 0.2
        assert signal.num_releases_analyzed == 0
        assert signal.macro_health == "unknown"

    def test_ingest_cpi_below_consensus(self):
        gen = self._make_gen()
        reading = gen.ingest_macro_release("cpi", 2.8, 3.0)
        assert reading.data_type == MacroDataType.CPI
        # CPI 2.8 vs 3.0: -6.7% surprise → z-score -0.44 → below -0.3 → dovish
        assert reading.narrative_score > 0
        assert reading.tone == NarrativeTone.DOVISH
        assert reading.note != ""

    def test_ingest_cpi_above_consensus(self):
        gen = self._make_gen()
        reading = gen.ingest_macro_release("cpi", 3.8, 3.0)
        assert reading.narrative_score < 0  # Hawkish → negative
        assert reading.tone == NarrativeTone.HAWKISH

    def test_ingest_nfp_beat(self):
        gen = self._make_gen()
        reading = gen.ingest_macro_release("nfp", 250000, 200000)
        assert reading.narrative_score > 0  # Positive surprise
        assert reading.tone == NarrativeTone.POSITIVE

    def test_ingest_nfp_miss(self):
        gen = self._make_gen()
        reading = gen.ingest_macro_release("nfp", 150000, 200000)
        assert reading.narrative_score < 0  # Negative surprise

    def test_ingest_gdp_beat(self):
        gen = self._make_gen()
        reading = gen.ingest_macro_release("gdp", 3.5, 2.5)
        assert reading.narrative_score > 0
        assert reading.tone == NarrativeTone.POSITIVE

    def test_ingest_gdp_miss(self):
        gen = self._make_gen()
        reading = gen.ingest_macro_release("gdp", 1.5, 2.5)
        assert reading.narrative_score < 0

    def test_ingest_ism_expansion_improving(self):
        gen = self._make_gen()
        reading = gen.ingest_macro_release("ism_manufacturing", 54.0, 52.0)
        assert reading.narrative_score > 0
        assert reading.tone == NarrativeTone.POSITIVE

    def test_ingest_ism_contraction_deepening(self):
        gen = self._make_gen()
        reading = gen.ingest_macro_release("ism_manufacturing", 44.0, 47.0)
        assert reading.narrative_score < 0
        assert reading.tone == NarrativeTone.CAUTIOUS

    def test_multiple_releases_composite(self):
        gen = self._make_gen()
        gen.ingest_macro_release("nfp", 250000, 200000)  # Positive
        gen.ingest_macro_release("gdp", 3.5, 2.5)  # Positive
        signal = gen.generate_signal()
        assert signal.composite_score > 0
        assert signal.num_releases_analyzed == 2
        assert signal.macro_health in ("expansion", "slowdown_growth")

    def test_multiple_releases_conflicting(self):
        gen = self._make_gen()
        gen.ingest_macro_release("cpi", 3.8, 3.0)  # Hawkish (negative)
        gen.ingest_macro_release("nfp", 250000, 200000)  # Positive
        signal = gen.generate_signal()
        assert signal.num_releases_analyzed == 2

    def test_fomc_dovish_statement(self):
        gen = self._make_gen()
        statement = (
            "The Committee decided to lower the federal funds rate. "
            "The labor market has weakened. Inflation has eased. "
            "The Committee is patient and data-dependent. "
            "There is spare capacity in the economy."
        )
        tone = gen.ingest_fomc_statement(statement)
        assert tone in (NarrativeTone.DOVISH, NarrativeTone.NEUTRAL_DOVISH)

    def test_fomc_hawkish_statement(self):
        gen = self._make_gen()
        statement = (
            "The Committee decided to raise the federal funds rate. "
            "Inflation remains persistently above target. "
            "The labor market is overheating. "
            "Further firming may be appropriate. "
            "The Committee remains vigilant."
        )
        tone = gen.ingest_fomc_statement(statement)
        assert tone in (NarrativeTone.HAWKISH, NarrativeTone.NEUTRAL_HAWKISH)

    def test_fomc_neutral_statement(self):
        gen = self._make_gen()
        statement = (
            "The Committee decided to maintain the federal funds rate. "
            "Economic activity continues to expand at a moderate pace. "
            "The Committee will continue to assess incoming data."
        )
        tone = gen.ingest_fomc_statement(statement)
        assert tone not in (NarrativeTone.DOVISH, NarrativeTone.HAWKISH)

    def test_signal_with_fomc_and_macro(self):
        gen = self._make_gen()
        gen.ingest_macro_release("nfp", 250000, 200000)
        gen.ingest_fomc_statement("The Committee decided to lower rates. Inflation is easing.")
        signal = gen.generate_signal()
        assert signal.num_releases_analyzed == 1

    def test_equity_signal_mapping(self):
        gen = self._make_gen()
        gen.ingest_macro_release("nfp", 250000, 200000)
        gen.ingest_macro_release("gdp", 3.5, 2.5)
        signal = gen.generate_signal()
        # Positive macro → positive equity
        assert signal.equity_signal > 0

    def test_bond_signal_inverse(self):
        gen = self._make_gen()
        gen.ingest_macro_release("cpi", 4.0, 3.0)  # Strongly hawkish (bad)
        signal = gen.generate_signal()
        # Negative macro → flight to safety → positive bonds
        assert signal.bond_signal > 0

    def test_gold_signal_uncertainty(self):
        gen = self._make_gen()
        gen.ingest_macro_release("cpi", 4.0, 3.0)  # Hawkish → uncertainty
        signal = gen.generate_signal()

    def test_get_ensemble_signal(self):
        gen = self._make_gen()
        gen.ingest_macro_release("cpi", 2.8, 3.0)
        ensemble = gen.get_ensemble_signal()
        assert "source" in ensemble
        assert ensemble["source"] == "llm_narrative"
        assert "value" in ensemble
        assert "confidence" in ensemble
        assert "asset_signals" in ensemble
        assert "SPY" in ensemble["asset_signals"]
        assert "TLT" in ensemble["asset_signals"]
        assert "GLD" in ensemble["asset_signals"]

    def test_get_ensemble_signal_no_data(self):
        gen = self._make_gen()
        ensemble = gen.get_ensemble_signal()
        assert ensemble["value"] == 0.0

    def test_upcoming_events(self):
        gen = self._make_gen()
        events = gen.get_upcoming_events(365)
        assert isinstance(events, list)
        if events:
            assert "date" in events[0]
            assert "type" in events[0]
            assert "name" in events[0]

    def test_data_freshness_no_data(self):
        gen = self._make_gen()
        signal = gen.generate_signal()
        assert signal.data_freshness_days == 999

    def test_data_freshness_with_data(self):
        gen = self._make_gen()
        gen.ingest_macro_release("cpi", 3.0, 3.0)
        signal = gen.generate_signal()
        assert signal.data_freshness_days == 0

    def test_signal_state_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            original_path = LLMNarrativeSignalGenerator.STATE_FILE
            LLMNarrativeSignalGenerator.STATE_FILE = Path(tmpdir) / "narrative_state.json"
            try:
                gen1 = LLMNarrativeSignalGenerator()
                gen1._recent_releases = []
                gen1.ingest_macro_release("cpi", 3.8, 3.0)
                gen1.ingest_macro_release("nfp", 150000, 200000)

                # New instance should load persisted state
                gen2 = LLMNarrativeSignalGenerator()
                assert len(gen2._recent_releases) == 2
            finally:
                LLMNarrativeSignalGenerator.STATE_FILE = original_path

    def test_fomc_tone_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            original_path = LLMNarrativeSignalGenerator.STATE_FILE
            LLMNarrativeSignalGenerator.STATE_FILE = Path(tmpdir) / "narrative_state.json"
            try:
                gen1 = LLMNarrativeSignalGenerator()
                gen1._recent_releases = []
                statement = (
                    "The Committee decided to raise rates. "
                    "Inflation remains persistent. "
                    "Overheating in the labor market."
                )
                gen1.ingest_fomc_statement(statement)

                gen2 = LLMNarrativeSignalGenerator()
                assert gen2._latest_fomc_tone in (
                    NarrativeTone.HAWKISH, NarrativeTone.NEUTRAL_HAWKISH
                )
            finally:
                LLMNarrativeSignalGenerator.STATE_FILE = original_path

    def test_explain_mode(self):
        gen = self._make_gen()
        gen.ingest_macro_release("cpi", 2.8, 3.0)
        signal = gen.generate_signal()
        assert signal.explanation != ""
        assert signal.macro_health in (
            "expansion", "slowdown_growth", "moderate_slowdown",
            "contraction", "unknown"
        )

    def test_compute_macro_health_no_data(self):
        gen = self._make_gen()
        assert gen.compute_macro_health() == "unknown"

    def test_compute_macro_health_expansion(self):
        gen = self._make_gen()
        gen.ingest_macro_release("nfp", 250000, 200000)
        gen.ingest_macro_release("gdp", 3.5, 2.5)
        health = gen.compute_macro_health()
        assert health == "expansion"

    def test_compute_macro_health_contraction(self):
        gen = self._make_gen()
        gen.ingest_macro_release("cpi", 4.5, 3.0)
        gen.ingest_macro_release("nfp", 100000, 200000)
        gen.ingest_macro_release("ism_manufacturing", 42.0, 48.0)
        health = gen.compute_macro_health()
        assert health == "contraction"

    def test_signal_confidence_decay(self):
        gen = self._make_gen()
        gen.ingest_macro_release("cpi", 3.0, 3.0)
        signal = gen.generate_signal()
        fresh_confidence = signal.confidence

        # Simulate stale data
        if gen._recent_releases:
            gen._recent_releases[-1].release_date = (
                datetime.now() - timedelta(days=90)
            ).strftime("%Y-%m-%d")
        stale_signal = gen.generate_signal()
        assert stale_signal.confidence <= fresh_confidence

    def test_signal_in_range(self):
        gen = self._make_gen()
        gen.ingest_macro_release("cpi", 4.5, 3.0)
        gen.ingest_macro_release("nfp", 100000, 200000)
        gen.ingest_macro_release("gdp", 1.0, 2.5)
        gen.ingest_macro_release("ism_manufacturing", 40.0, 48.0)
        signal = gen.generate_signal()
        assert -1.0 <= signal.composite_score <= 1.0
        assert -1.0 <= signal.equity_signal <= 1.0
        assert -1.0 <= signal.bond_signal <= 1.0
        assert -1.0 <= signal.gold_signal <= 1.0
        assert 0.0 <= signal.confidence <= 1.0

    def test_invalid_macro_type(self):
        gen = self._make_gen()
        with pytest.raises(ValueError):
            gen.ingest_macro_release("nonexistent_type", 1.0, 1.0)

    def test_multiple_cpi_readings(self):
        gen = self._make_gen()
        gen.ingest_macro_release("cpi", 3.0, 3.0)  # Neutral
        gen.ingest_macro_release("cpi", 3.0, 3.0)  # Neutral
        signal = gen.generate_signal()
        # Two neutral CPI readings → near-zero score
        assert abs(signal.composite_score) < 0.1

    def test_get_signal_reading_format(self):
        gen = self._make_gen()
        gen.ingest_macro_release("cpi", 3.0, 3.0)
        reading = gen.get_signal_reading()
        assert "composite_score" in reading
        assert "confidence" in reading
        assert "macro_health" in reading
        assert "fomc_tone" in reading
        assert "equity_signal" in reading
        assert "bond_signal" in reading
        assert "gold_signal" in reading
        assert "upcoming_events" in reading

    def test_standalone_function(self):
        from src.signals.llm_narrative_signal import get_narrative_signal
        result = get_narrative_signal()
        assert "source" in result
        assert result["source"] == "llm_narrative"

    def test_macro_health_thresholds(self):
        """Test macro health transitions at boundaries."""
        gen = self._make_gen()
        # Single moderately negative release
        gen.ingest_macro_release("cpi", 3.6, 3.0)
        # surprise_z = 0.6/3.0 / 0.15 = 1.33 → hawkish with score -0.6
        health = gen.compute_macro_health()
        assert health in ("moderate_slowdown", "slowdown_growth", "contraction")

    def test_multiple_aggregation(self):
        """Verify proper weighted aggregation pattern."""
        gen = self._make_gen()
        # Three releases with same score
        gen.ingest_macro_release("cpi", 2.5, 3.0)  # Dovish (positive)
        gen.ingest_macro_release("cpi", 2.5, 3.0)  # Dovish (positive)
        gen.ingest_macro_release("cpi", 2.5, 3.0)  # Dovish (positive)
        signal = gen.generate_signal()
        assert signal.composite_score > 0

    def test_calendar_cli_output(self):
        """Verify calendar mode works."""
        gen = self._make_gen()
        events = gen.get_upcoming_events(60)
        assert isinstance(events, list)
        for ev in events:
            assert isinstance(ev["days_until"], int)
            assert ev["days_until"] >= 0

    def test_fomc_tone_factor_weight(self):
        """FOMC dovish should contribute positively to composite."""
        gen = self._make_gen()
        gen.ingest_fomc_statement(
            "The Committee decided to lower rates. Easing cycle continues."
        )
        signal = gen.generate_signal()
        # FOMC dovish contributes positively (30% weight)
        # Since no macro data, the composite is entirely FOMC factor
        assert signal.composite_score > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
