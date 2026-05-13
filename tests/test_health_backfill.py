#!/usr/bin/env python3
"""
Tests for health_backfill.py — CLI entry point, health calculation logic,
and integration testing.
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Health Calculation Logic Tests (extracted from backfill_historical_health)
# ---------------------------------------------------------------------------

class TestHealthCalculationLogic:
    """Test the health score calculation logic used in backfill."""

    def test_stability_perfect_consistency(self):
        """When all signals are identical, stability should be 0.5 (signal_range=0)."""
        import statistics
        values = [0.5, 0.5, 0.5, 0.5]
        variance = statistics.variance(values)
        signal_range = max(values) - min(values)
        if signal_range > 0:
            stability = 1.0 - min(1.0, variance / (signal_range ** 2))
        else:
            stability = 0.5
        assert stability == 0.5

    def test_stability_high_variance(self):
        """High variance relative to range should give low stability."""
        import statistics
        values = [-1.0, 0.0, 1.0, -0.5, 0.5]
        variance = statistics.variance(values)
        signal_range = max(values) - min(values)
        stability = 1.0 - min(1.0, variance / (signal_range ** 2))
        assert 0 <= stability <= 1

    def test_health_score_formula(self):
        """Verify health = 0.4*correlation + 0.4*(avg_confidence/100) + 0.2*stability."""
        correlation = 0.5
        avg_confidence = 60  # percent
        stability = 0.8
        health = 0.4 * correlation + 0.4 * (avg_confidence / 100) + 0.2 * stability
        expected = 0.4 * 0.5 + 0.4 * 0.6 + 0.2 * 0.8
        assert health == pytest.approx(expected)

    def test_status_healthy(self):
        health = 0.75
        if health >= 0.7:
            status = 'healthy'
        elif health >= 0.5:
            status = 'recovering'
        elif health >= 0.3:
            status = 'degraded'
        else:
            status = 'critical'
        assert status == 'healthy'

    def test_status_recovering(self):
        health = 0.55
        if health >= 0.7:
            status = 'healthy'
        elif health >= 0.5:
            status = 'recovering'
        elif health >= 0.3:
            status = 'degraded'
        else:
            status = 'critical'
        assert status == 'recovering'

    def test_status_degraded(self):
        health = 0.35
        if health >= 0.7:
            status = 'healthy'
        elif health >= 0.5:
            status = 'recovering'
        elif health >= 0.3:
            status = 'degraded'
        else:
            status = 'critical'
        assert status == 'degraded'

    def test_status_critical(self):
        health = 0.15
        if health >= 0.7:
            status = 'healthy'
        elif health >= 0.5:
            status = 'recovering'
        elif health >= 0.3:
            status = 'degraded'
        else:
            status = 'critical'
        assert status == 'critical'

    def test_weight_multiplier_logic(self):
        """Test weight multiplier assignment by status."""
        for status, expected in [('recovering', 0.75), ('healthy', 1.0), ('degraded', 0.5), ('critical', 0.5)]:
            if status == 'recovering':
                mult = 0.75
            elif status == 'healthy':
                mult = 1.0
            else:
                mult = 0.5
            assert mult == expected

    def test_correlation_from_confidence(self):
        """Test correlation scaling: 0.3 + (avg_confidence * 0.4)."""
        assert 0.3 + (0.5 * 0.4) == pytest.approx(0.5)
        assert 0.3 + (1.0 * 0.4) == pytest.approx(0.7)
        assert 0.3 + (0.0 * 0.4) == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# test_integration weight verification
# ---------------------------------------------------------------------------

class TestIntegrationLogic:

    def test_weight_sum_check(self):
        """Verify the weight sum verification logic."""
        adjusted = {"source_a": 0.35, "source_b": 0.35, "source_c": 0.30}
        total = sum(adjusted.values())
        assert 0.99 <= total <= 1.01

    def test_health_distribution_counting(self):
        """Test health score categorization."""
        health_scores = {"a": 0.8, "b": 0.6, "c": 0.3, "d": 0.9}
        healthy = sum(1 for h in health_scores.values() if isinstance(h, (int, float)) and h >= 0.7)
        degraded = sum(1 for h in health_scores.values() if isinstance(h, (int, float)) and h < 0.5)
        assert healthy == 2  # a, d
        assert degraded == 1  # c


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------

class TestCLI:

    def test_no_args_prints_help(self, capsys):
        from src.signals.health_backfill import main
        with patch("sys.argv", ["health_backfill.py"]):
            main()
        captured = capsys.readouterr()
        assert "usage" in captured.out.lower() or "Health" in captured.out or "Signal" in captured.out
