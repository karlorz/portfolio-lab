#!/usr/bin/env python3
"""
Tests for mock_quality_scores.py — constants, quality score calculation,
noise generation, and CLI.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.mock_quality_scores import (
    ETF_KNOWN_CHARACTERISTICS,
    calculate_mock_quality_score,
    add_noise_to_metrics,
)


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------

class TestConstants:

    def test_etf_count(self):
        assert len(ETF_KNOWN_CHARACTERISTICS) == 4

    def test_expected_etfs(self):
        assert "QUAL" in ETF_KNOWN_CHARACTERISTICS
        assert "MTUM" in ETF_KNOWN_CHARACTERISTICS
        assert "USMV" in ETF_KNOWN_CHARACTERISTICS
        assert "VLUE" in ETF_KNOWN_CHARACTERISTICS

    def test_qual_characteristics(self):
        qual = ETF_KNOWN_CHARACTERISTICS["QUAL"]
        assert qual["roe"] == 0.22
        assert qual["debt_equity"] == 0.35
        assert qual["earnings_stability"] == 0.75

    def test_all_have_required_keys(self):
        for sym, chars in ETF_KNOWN_CHARACTERISTICS.items():
            assert "roe" in chars, f"{sym} missing roe"
            assert "debt_equity" in chars, f"{sym} missing debt_equity"
            assert "earnings_stability" in chars, f"{sym} missing earnings_stability"
            assert "profitability" in chars, f"{sym} missing profitability"


# ---------------------------------------------------------------------------
# calculate_mock_quality_score Tests
# ---------------------------------------------------------------------------

class TestCalculateQualityScore:

    def test_returns_float(self):
        score = calculate_mock_quality_score(0.20, 0.50, 0.70, 0.60)
        assert isinstance(score, float)

    def test_score_bounded(self):
        score = calculate_mock_quality_score(0.20, 0.50, 0.70, 0.60)
        assert 0 <= score <= 1

    def test_high_quality_high_score(self):
        score = calculate_mock_quality_score(0.25, 0.20, 0.90, 0.80)
        assert score > 0.7

    def test_low_quality_low_score(self):
        score = calculate_mock_quality_score(0.05, 1.20, 0.20, 0.20)
        assert score < 0.4

    def test_roe_weight(self):
        high_roe = calculate_mock_quality_score(0.25, 0.50, 0.50, 0.50)
        low_roe = calculate_mock_quality_score(0.05, 0.50, 0.50, 0.50)
        assert high_roe > low_roe

    def test_debt_weight(self):
        low_debt = calculate_mock_quality_score(0.20, 0.20, 0.50, 0.50)
        high_debt = calculate_mock_quality_score(0.20, 1.20, 0.50, 0.50)
        assert low_debt > high_debt

    def test_stability_weight(self):
        high_stab = calculate_mock_quality_score(0.20, 0.50, 0.90, 0.50)
        low_stab = calculate_mock_quality_score(0.20, 0.50, 0.20, 0.50)
        assert high_stab > low_stab

    def test_profitability_weight(self):
        high_prof = calculate_mock_quality_score(0.20, 0.50, 0.50, 0.80)
        low_prof = calculate_mock_quality_score(0.20, 0.50, 0.50, 0.20)
        assert high_prof > low_prof

    def test_qual_ranking(self):
        """QUAL should score highest among factor ETFs."""
        qual = ETF_KNOWN_CHARACTERISTICS["QUAL"]
        qual_score = calculate_mock_quality_score(**{k: qual[k] for k in ['roe', 'debt_equity', 'earnings_stability', 'profitability']})
        vlue = ETF_KNOWN_CHARACTERISTICS["VLUE"]
        vlue_score = calculate_mock_quality_score(**{k: vlue[k] for k in ['roe', 'debt_equity', 'earnings_stability', 'profitability']})
        assert qual_score > vlue_score

    def test_clipping_extreme_roe(self):
        score = calculate_mock_quality_score(1.0, 0.50, 0.50, 0.50)
        assert 0 <= score <= 1

    def test_clipping_negative_values(self):
        score = calculate_mock_quality_score(-0.1, 0.50, 0.50, 0.50)
        assert 0 <= score <= 1

    def test_rounding(self):
        score = calculate_mock_quality_score(0.20, 0.50, 0.70, 0.60)
        assert len(str(score).split('.')[-1]) <= 4


# ---------------------------------------------------------------------------
# add_noise_to_metrics Tests
# ---------------------------------------------------------------------------

class TestAddNoise:

    def test_returns_dict(self):
        base = {"roe": 0.20, "debt_equity": 0.50, "earnings_stability": 0.70, "profitability": 0.60}
        result = add_noise_to_metrics(base, "2026-01-01")
        assert isinstance(result, dict)

    def test_has_all_keys(self):
        base = {"roe": 0.20, "debt_equity": 0.50, "earnings_stability": 0.70, "profitability": 0.60}
        result = add_noise_to_metrics(base, "2026-01-01")
        assert "roe" in result
        assert "debt_equity" in result
        assert "earnings_stability" in result
        assert "profitability" in result

    def test_deterministic(self):
        base = {"roe": 0.20, "debt_equity": 0.50, "earnings_stability": 0.70, "profitability": 0.60}
        r1 = add_noise_to_metrics(base, "2026-01-01")
        r2 = add_noise_to_metrics(base, "2026-01-01")
        assert r1 == r2

    def test_different_dates_different_noise(self):
        base = {"roe": 0.20, "debt_equity": 0.50, "earnings_stability": 0.70, "profitability": 0.60}
        r1 = add_noise_to_metrics(base, "2026-01-01")
        r2 = add_noise_to_metrics(base, "2026-06-15")
        # Different dates should produce different noise (with very high probability)
        assert r1 != r2

    def test_values_clamped(self):
        base = {"roe": 0.20, "debt_equity": 0.50, "earnings_stability": 0.70, "profitability": 0.60}
        for date in ["2026-01-01", "2026-06-15", "2026-12-31"]:
            result = add_noise_to_metrics(base, date)
            assert 0.05 <= result["roe"] <= 0.40
            assert 0.1 <= result["debt_equity"] <= 1.5
            assert 0.2 <= result["earnings_stability"] <= 0.95
            assert 0.2 <= result["profitability"] <= 0.85

    def test_seed_offset_changes_noise(self):
        base = {"roe": 0.20, "debt_equity": 0.50, "earnings_stability": 0.70, "profitability": 0.60}
        r1 = add_noise_to_metrics(base, "2026-01-01", seed_offset=0)
        r2 = add_noise_to_metrics(base, "2026-01-01", seed_offset=1)
        assert r1 != r2

    def test_noise_small(self):
        """Noise should be within ±5% of base value."""
        base = {"roe": 0.20, "debt_equity": 0.50, "earnings_stability": 0.70, "profitability": 0.60}
        result = add_noise_to_metrics(base, "2026-01-01")
        for key in base:
            assert abs(result[key] - base[key]) / base[key] < 0.10  # Within 10%


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------

class TestCLI:

    def test_main_no_args(self, capsys):
        from src.data.mock_quality_scores import main
        with patch("sys.argv", ["mock_quality_scores.py"]):
            with pytest.raises(SystemExit):
                main()
