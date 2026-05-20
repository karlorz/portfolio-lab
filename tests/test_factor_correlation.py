#!/usr/bin/env python3
"""
Tests for Factor Correlation Matrix Calculator (src/analysis/factor_correlation.py).
Tests price loading, return calculation, correlation computation, matrix building,
redundancy analysis, and report generation.
"""
import sys
import os
import json
import sqlite3
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.analysis.factor_correlation import (
    FACTOR_ETFS, load_factor_prices, calculate_returns,
    calculate_correlation, build_correlation_matrix,
    analyze_redundancy, generate_report,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _populate_factor_db(db_path: Path, n_days: int = 100):
    """Create a factor_prices table with synthetic data."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS factor_prices (
                date TEXT, symbol TEXT, close REAL,
                PRIMARY KEY (date, symbol)
            )
        """)
        import random
        random.seed(42)
        base_prices = {'MTUM': 100.0, 'VLUE': 80.0, 'QUAL': 90.0, 'USMV': 70.0}
        for day in range(n_days):
            date = f"2025-01-{day+1:02d}"
            for sym, base in base_prices.items():
                base *= (1 + random.gauss(0.001, 0.015))
                conn.execute(
                    "INSERT OR REPLACE INTO factor_prices (date, symbol, close) VALUES (?, ?, ?)",
                    (date, sym, round(base, 4)),
                )
        conn.commit()


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

class TestConstants:
    def test_factor_etfs(self):
        assert FACTOR_ETFS == ['MTUM', 'VLUE', 'QUAL', 'USMV']


# ---------------------------------------------------------------------------
# calculate_returns tests
# ---------------------------------------------------------------------------

class TestCalculateReturns:
    def test_basic_returns(self):
        prices = [100.0, 101.0, 102.0, 100.5]
        returns = calculate_returns(prices)
        assert len(returns) == 3
        assert abs(returns[0] - 0.01) < 1e-6
        assert abs(returns[1] - 1.0/101.0) < 1e-6

    def test_empty_prices(self):
        assert calculate_returns([]) == []

    def test_single_price(self):
        assert calculate_returns([100.0]) == []

    def test_constant_prices(self):
        returns = calculate_returns([100.0] * 10)
        assert all(r == 0.0 for r in returns)


# ---------------------------------------------------------------------------
# calculate_correlation tests
# ---------------------------------------------------------------------------

class TestCalculateCorrelation:
    def test_perfect_positive(self):
        r1 = [0.01, 0.02, -0.01, 0.005, 0.015] * 10  # 50 elements
        r2 = [x * 2 for x in r1]
        corr = calculate_correlation(r1, r2)
        assert abs(corr - 1.0) < 0.01

    def test_perfect_negative(self):
        r1 = [0.01, 0.02, -0.01, 0.005, 0.015] * 10
        r2 = [-x for x in r1]
        corr = calculate_correlation(r1, r2)
        assert abs(corr + 1.0) < 0.01

    def test_insufficient_data(self):
        r1 = [0.01] * 29
        r2 = [0.01] * 29
        assert calculate_correlation(r1, r2) == 0.0

    def test_zero_variance(self):
        r1 = [0.01] * 50
        r2 = [0.01, 0.02] * 25
        assert calculate_correlation(r1, r2) == 0.0

    def test_different_lengths(self):
        r1 = [0.01 * i for i in range(50)]
        r2 = [0.02 * i for i in range(40)]
        corr = calculate_correlation(r1, r2)
        # Should use min length; floating-point may slightly exceed [-1, 1]
        assert -1.01 <= corr <= 1.01


# ---------------------------------------------------------------------------
# load_factor_prices tests
# ---------------------------------------------------------------------------

class TestLoadFactorPrices:
    def test_loads_from_db(self, tmp_path):
        db = tmp_path / "factors.db"
        _populate_factor_db(db, n_days=50)
        prices = load_factor_prices(db)
        assert set(prices.keys()) == set(FACTOR_ETFS)
        for sym in FACTOR_ETFS:
            assert len(prices[sym]) == 50

    def test_empty_db(self, tmp_path):
        db = tmp_path / "empty.db"
        with sqlite3.connect(db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS factor_prices (
                    date TEXT, symbol TEXT, close REAL,
                    PRIMARY KEY (date, symbol)
                )
            """)
        prices = load_factor_prices(db)
        for sym in FACTOR_ETFS:
            assert prices[sym] == []


# ---------------------------------------------------------------------------
# build_correlation_matrix tests
# ---------------------------------------------------------------------------

class TestBuildCorrelationMatrix:
    def test_matrix_structure(self):
        import random
        random.seed(42)
        prices = {}
        for sym in FACTOR_ETFS:
            p = [100.0]
            for _ in range(100):
                p.append(p[-1] * (1 + random.gauss(0.001, 0.015)))
            prices[sym] = p

        matrix = build_correlation_matrix(prices)
        assert set(matrix.keys()) == set(FACTOR_ETFS)
        for sym in FACTOR_ETFS:
            assert set(matrix[sym].keys()) == set(FACTOR_ETFS)

    def test_diagonal_is_one(self):
        import random
        random.seed(42)
        prices = {}
        for sym in FACTOR_ETFS:
            p = [100.0]
            for _ in range(100):
                p.append(p[-1] * (1 + random.gauss(0.001, 0.015)))
            prices[sym] = p

        matrix = build_correlation_matrix(prices)
        for sym in FACTOR_ETFS:
            assert matrix[sym][sym] == 1.0

    def test_symmetric_matrix(self):
        import random
        random.seed(42)
        prices = {}
        for sym in FACTOR_ETFS:
            p = [100.0]
            for _ in range(100):
                p.append(p[-1] * (1 + random.gauss(0.001, 0.015)))
            prices[sym] = p

        matrix = build_correlation_matrix(prices)
        for s1 in FACTOR_ETFS:
            for s2 in FACTOR_ETFS:
                assert abs(matrix[s1][s2] - matrix[s2][s1]) < 0.01

    def test_correlations_bounded(self):
        import random
        random.seed(42)
        prices = {}
        for sym in FACTOR_ETFS:
            p = [100.0]
            for _ in range(100):
                p.append(p[-1] * (1 + random.gauss(0.001, 0.015)))
            prices[sym] = p

        matrix = build_correlation_matrix(prices)
        for s1 in FACTOR_ETFS:
            for s2 in FACTOR_ETFS:
                assert -1.0 <= matrix[s1][s2] <= 1.0


# ---------------------------------------------------------------------------
# analyze_redundancy tests
# ---------------------------------------------------------------------------

class TestAnalyzeRedundancy:
    def test_no_redundancy(self):
        matrix = {}
        for s1 in FACTOR_ETFS:
            matrix[s1] = {}
            for s2 in FACTOR_ETFS:
                if s1 == s2:
                    matrix[s1][s2] = 1.0
                else:
                    matrix[s1][s2] = 0.5
        redundant = analyze_redundancy(matrix)
        assert redundant == []

    def test_with_redundancy(self):
        matrix = {}
        for s1 in FACTOR_ETFS:
            matrix[s1] = {}
            for s2 in FACTOR_ETFS:
                if s1 == s2:
                    matrix[s1][s2] = 1.0
                else:
                    matrix[s1][s2] = 0.5
        # Make MTUM-VLUE highly correlated
        matrix['MTUM']['VLUE'] = 0.85
        matrix['VLUE']['MTUM'] = 0.85
        redundant = analyze_redundancy(matrix)
        assert len(redundant) >= 1
        assert ('MTUM', 'VLUE', 0.85) in redundant

    def test_negative_correlation_not_flagged(self):
        matrix = {}
        for s1 in FACTOR_ETFS:
            matrix[s1] = {}
            for s2 in FACTOR_ETFS:
                if s1 == s2:
                    matrix[s1][s2] = 1.0
                else:
                    matrix[s1][s2] = -0.5
        redundant = analyze_redundancy(matrix)
        assert redundant == []

    def test_threshold_at_0_8(self):
        matrix = {}
        for s1 in FACTOR_ETFS:
            matrix[s1] = {}
            for s2 in FACTOR_ETFS:
                if s1 == s2:
                    matrix[s1][s2] = 1.0
                else:
                    matrix[s1][s2] = 0.5
        # Exactly 0.8 should be flagged
        matrix['MTUM']['VLUE'] = 0.8
        matrix['VLUE']['MTUM'] = 0.8
        redundant = analyze_redundancy(matrix)
        # 0.8 is NOT > 0.8, so should NOT be flagged
        assert all(pair[2] != 0.8 for pair in redundant) or len(redundant) == 0


# ---------------------------------------------------------------------------
# generate_report tests
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_report_has_title(self):
        matrix = {}
        for s1 in FACTOR_ETFS:
            matrix[s1] = {}
            for s2 in FACTOR_ETFS:
                matrix[s1][s2] = 1.0 if s1 == s2 else 0.5
        report = generate_report(matrix, [])
        assert "Factor ETF Correlation Matrix Report" in report

    def test_report_has_matrix(self):
        matrix = {}
        for s1 in FACTOR_ETFS:
            matrix[s1] = {}
            for s2 in FACTOR_ETFS:
                matrix[s1][s2] = 1.0 if s1 == s2 else 0.5
        report = generate_report(matrix, [])
        for sym in FACTOR_ETFS:
            assert sym in report

    def test_report_with_redundancy(self):
        matrix = {}
        for s1 in FACTOR_ETFS:
            matrix[s1] = {}
            for s2 in FACTOR_ETFS:
                matrix[s1][s2] = 1.0 if s1 == s2 else 0.5
        redundant = [('MTUM', 'VLUE', 0.85)]
        report = generate_report(matrix, redundant)
        assert "WARNING" in report
        assert "MTUM-VLUE" in report

    def test_report_no_redundancy(self):
        matrix = {}
        for s1 in FACTOR_ETFS:
            matrix[s1] = {}
            for s2 in FACTOR_ETFS:
                matrix[s1][s2] = 1.0 if s1 == s2 else 0.5
        report = generate_report(matrix, [])
        assert "diversification benefits confirmed" in report.lower()

    def test_report_has_interpretation(self):
        matrix = {}
        for s1 in FACTOR_ETFS:
            matrix[s1] = {}
            for s2 in FACTOR_ETFS:
                matrix[s1][s2] = 1.0 if s1 == s2 else 0.5
        report = generate_report(matrix, [])
        assert "Interpretation" in report
        assert "diversification" in report.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
