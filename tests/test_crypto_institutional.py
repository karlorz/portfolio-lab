#!/usr/bin/env python3
"""Tests for src/crypto/institutional.py — TokenizedTreasuryStrategy + CryptoRiskManager.

Covers: dataclasses, database init, allocation logic, risk assessment,
compliance checks, rebalance deltas, CLI, and edge cases.
No ML dependencies.
"""

import json
import os
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _tmp_data_dir(monkeypatch, tmp_path):
    """Redirect DATA_DIR and CRYPTO_DB to a temp directory."""
    import src.crypto.institutional as mod

    monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(mod, "CRYPTO_DB", tmp_path / "crypto_allocation.db")
    yield tmp_path


@pytest.fixture()
def mock_circuit_breaker():
    """Return a mock DrawdownCircuitBreaker with green status."""
    cb = MagicMock()
    cb.get_status.return_value = {"status": "green", "drawdown": 0.0}
    return cb


@pytest.fixture()
def strategy(mock_circuit_breaker):
    """TokenizedTreasuryStrategy with mocked circuit breaker."""
    with patch(
        "src.crypto.institutional.DrawdownCircuitBreaker",
        return_value=mock_circuit_breaker,
    ):
        from src.crypto.institutional import TokenizedTreasuryStrategy
        return TokenizedTreasuryStrategy()


@pytest.fixture()
def risk_mgr(mock_circuit_breaker):
    """CryptoRiskManager with mocked circuit breaker."""
    with patch(
        "src.crypto.institutional.DrawdownCircuitBreaker",
        return_value=mock_circuit_breaker,
    ):
        from src.crypto.institutional import CryptoRiskManager
        return CryptoRiskManager()


# ---------------------------------------------------------------------------
# TokenizedProductAllocation dataclass
# ---------------------------------------------------------------------------

class TestTokenizedProductAllocation:
    def test_create(self):
        from src.crypto.institutional import TokenizedProductAllocation
        alloc = TokenizedProductAllocation(
            product_code="BUIDL",
            product_name="BlackRock BUIDL",
            allocation_pct=0.50,
            allocation_usd=5000.0,
            expected_apy=0.0345,
            risk_group="group_1_tokenized",
            blockchains=["ethereum", "solana"],
            liquidity_score=0.90,
            regulatory_clearance=True,
            custody_rating="high",
        )
        assert alloc.product_code == "BUIDL"
        assert alloc.allocation_pct == 0.50
        assert alloc.regulatory_clearance is True

    def test_to_dict(self):
        from src.crypto.institutional import TokenizedProductAllocation
        alloc = TokenizedProductAllocation(
            product_code="FOBXX",
            product_name="Franklin FOBXX",
            allocation_pct=0.35,
            allocation_usd=3500.0,
            expected_apy=0.0325,
            risk_group="group_1_tokenized",
            blockchains=["stellar"],
            liquidity_score=0.80,
            regulatory_clearance=True,
            custody_rating="moderate",
        )
        d = alloc.to_dict()
        assert d["product_code"] == "FOBXX"
        assert d["allocation_usd"] == 3500.0
        assert isinstance(d["timestamp"], str)

    def test_default_timestamp(self):
        from src.crypto.institutional import TokenizedProductAllocation
        alloc = TokenizedProductAllocation(
            product_code="TBT", product_name="OpenEden",
            allocation_pct=0.15, allocation_usd=1500.0,
            expected_apy=0.035, risk_group="group_1_tokenized",
            blockchains=["ethereum"], liquidity_score=0.80,
            regulatory_clearance=True, custody_rating="moderate",
        )
        assert alloc.timestamp  # not empty
        parsed = datetime.fromisoformat(alloc.timestamp)
        assert parsed.year >= 2024


# ---------------------------------------------------------------------------
# CryptoAllocation dataclass
# ---------------------------------------------------------------------------

class TestCryptoAllocation:
    def _make_allocation(self, **overrides):
        from src.crypto.institutional import CryptoAllocation
        defaults = dict(
            portfolio_value=100000, risk_profile="moderate",
            total_crypto_pct=0.03, total_crypto_usd=3000.0,
            group_1_allocation=0.60, group_2a_allocation=0.40,
            group_2b_allocation=0.0, tokenized_treasuries=[],
            weighted_risk_weight=0.52, capital_charge_pct=0.0416,
            expected_yield=0.0341, basel_compliant=True,
            sec_compliant=True, rebalance_needed=False,
        )
        defaults.update(overrides)
        return CryptoAllocation(**defaults)

    def test_create(self):
        alloc = self._make_allocation()
        assert alloc.portfolio_value == 100000
        assert alloc.risk_profile == "moderate"

    def test_to_dict_rounding(self):
        alloc = self._make_allocation(total_crypto_pct=0.03001, total_crypto_usd=3000.003)
        d = alloc.to_dict()
        assert d["total_crypto_pct"] == round(0.03001, 4)
        assert d["total_crypto_usd"] == round(3000.003, 2)

    def test_to_dict_includes_treasuries(self):
        from src.crypto.institutional import TokenizedProductAllocation
        t = TokenizedProductAllocation(
            product_code="BUIDL", product_name="BUIDL",
            allocation_pct=0.50, allocation_usd=900.0,
            expected_apy=0.0345, risk_group="group_1_tokenized",
            blockchains=["ethereum"], liquidity_score=0.90,
            regulatory_clearance=True, custody_rating="high",
        )
        alloc = self._make_allocation(tokenized_treasuries=[t])
        d = alloc.to_dict()
        assert len(d["tokenized_treasuries"]) == 1
        assert d["tokenized_treasuries"][0]["product_code"] == "BUIDL"

    def test_default_rebalance_threshold(self):
        alloc = self._make_allocation()
        assert alloc.rebalance_threshold_pct == 0.005


# ---------------------------------------------------------------------------
# RiskAssessment dataclass
# ---------------------------------------------------------------------------

class TestRiskAssessment:
    def test_create_and_to_dict(self):
        from src.crypto.institutional import RiskAssessment
        ra = RiskAssessment(
            portfolio_value=100000, crypto_allocation_pct=0.03,
            group_1_rwa=36.0, group_2a_rwa=120.0, group_2b_rwa=0.0,
            total_rwa=156.0, required_cet1=12.48, available_cet1=12000.0,
            buffer_pct=0.11988, max_drawdown_2022=-0.70,
            estimated_loss_stress=-165.0, portfolio_impact_stress_pct=0.00165,
            within_sec_limits=True, within_basel_limits=True,
            limiting_factor=None,
        )
        d = ra.to_dict()
        assert d["total_rwa"] == 156.0
        assert d["within_sec_limits"] is True
        assert d["limiting_factor"] is None


# ---------------------------------------------------------------------------
# ComplianceReport dataclass
# ---------------------------------------------------------------------------

class TestComplianceReport:
    def test_create_and_to_dict(self):
        from src.crypto.institutional import ComplianceReport
        cr = ComplianceReport(
            report_date="2026-05-20", investor_type="accredited",
            eligible_products=["BUIDL", "FOBXX", "TBT"],
            restricted_products=[],
            sec_compliant=True, accreditation_status="accredited",
            qualified_purchaser_status=False,
            basel_compliant=True, tier_1_capital_ratio=0.12,
            group_2b_within_limits=True,
            custody_arrangement="coinbase_custody",
            insurance_coverage=250_000_000, audit_trail_complete=True,
        )
        d = cr.to_dict()
        assert d["investor_type"] == "accredited"
        assert d["insurance_coverage"] == 250_000_000

    def test_retail_restricted_products(self):
        from src.crypto.institutional import ComplianceReport
        cr = ComplianceReport(
            report_date="2026-05-20", investor_type="retail",
            eligible_products=["FOBXX"],
            restricted_products=["BUIDL", "TBT"],
            sec_compliant=False, accreditation_status="retail",
            qualified_purchaser_status=False,
            basel_compliant=True, tier_1_capital_ratio=0.12,
            group_2b_within_limits=True,
            custody_arrangement="none",
            insurance_coverage=0, audit_trail_complete=False,
        )
        assert "BUIDL" in cr.restricted_products
        assert cr.sec_compliant is False


# ---------------------------------------------------------------------------
# init_database
# ---------------------------------------------------------------------------

class TestInitDatabase:
    def test_creates_tables(self, _tmp_data_dir):
        from src.crypto.institutional import init_database, CRYPTO_DB
        init_database()
        conn = sqlite3.connect(CRYPTO_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        assert "crypto_allocations" in tables
        assert "risk_assessments" in tables
        assert "compliance_reports" in tables
        assert "product_performance" in tables

    def test_idempotent(self, _tmp_data_dir):
        from src.crypto.institutional import init_database
        init_database()
        init_database()  # second call should not error
        assert True


# ---------------------------------------------------------------------------
# TokenizedTreasuryStrategy
# ---------------------------------------------------------------------------

class TestTokenizedTreasuryStrategy:
    def test_init_creates_db(self, strategy, _tmp_data_dir):
        db_path = _tmp_data_dir / "crypto_allocation.db"
        assert db_path.exists()

    def test_calculate_allocation_moderate(self, strategy):
        alloc = strategy.calculate_allocation(portfolio_value=100000, risk_profile="moderate")
        assert alloc.portfolio_value == 100000
        assert alloc.risk_profile == "moderate"
        assert alloc.total_crypto_pct == pytest.approx(0.03, abs=0.001)
        assert alloc.total_crypto_usd == pytest.approx(3000.0, abs=1.0)
        assert len(alloc.tokenized_treasuries) == 3

    def test_calculate_allocation_conservative(self, strategy):
        alloc = strategy.calculate_allocation(portfolio_value=100000, risk_profile="conservative")
        assert alloc.total_crypto_pct == pytest.approx(0.02, abs=0.001)
        assert alloc.group_1_allocation == 1.0  # 100% Group 1
        assert alloc.group_2a_allocation == 0.0

    def test_calculate_allocation_aggressive(self, strategy):
        alloc = strategy.calculate_allocation(portfolio_value=100000, risk_profile="aggressive")
        assert alloc.total_crypto_pct == pytest.approx(0.05, abs=0.001)
        assert alloc.total_crypto_usd == pytest.approx(5000.0, abs=1.0)

    def test_calculate_allocation_zero_portfolio(self, strategy):
        alloc = strategy.calculate_allocation(portfolio_value=0, risk_profile="moderate")
        assert alloc.total_crypto_usd == 0.0
        assert alloc.total_crypto_pct == pytest.approx(0.03, abs=0.001)

    def test_rebalance_triggered(self, strategy):
        alloc = strategy.calculate_allocation(
            portfolio_value=100000,
            risk_profile="moderate",
            current_allocation_pct=0.05,  # 5% vs target 3% → deviation 2% > 0.5%
        )
        assert alloc.rebalance_needed is True
        assert alloc.total_crypto_pct == pytest.approx(0.03, abs=0.001)

    def test_rebalance_not_triggered(self, strategy):
        alloc = strategy.calculate_allocation(
            portfolio_value=100000,
            risk_profile="moderate",
            current_allocation_pct=0.032,  # 3.2% vs target 3% → deviation 0.2% < 0.5%
        )
        assert alloc.rebalance_needed is False
        # Should keep current allocation
        assert alloc.total_crypto_pct == pytest.approx(0.032, abs=0.001)

    def test_basel_compliant(self, strategy):
        alloc = strategy.calculate_allocation(portfolio_value=100000)
        assert alloc.basel_compliant is True

    def test_sec_compliant(self, strategy):
        alloc = strategy.calculate_allocation(portfolio_value=100000)
        assert alloc.sec_compliant is True

    def test_risk_weight_calculation(self, strategy):
        alloc = strategy.calculate_allocation(portfolio_value=100000, risk_profile="moderate")
        # 60% group_1 (0.20) + 40% group_2a (1.00) = 0.52
        assert alloc.weighted_risk_weight == pytest.approx(0.52, abs=0.01)

    def test_capital_charge(self, strategy):
        alloc = strategy.calculate_allocation(portfolio_value=100000, risk_profile="moderate")
        # risk_weight * 0.08 = 0.52 * 0.08 = 0.0416
        assert alloc.capital_charge_pct == pytest.approx(0.0416, abs=0.001)

    def test_expected_yield_moderate(self, strategy):
        alloc = strategy.calculate_allocation(portfolio_value=100000, risk_profile="moderate")
        # BUIDL 50% @ 3.45% + FOBXX 35% @ 3.25% + TBT 15% @ 3.50%
        # = 0.50*0.0345 + 0.35*0.0325 + 0.15*0.0350 = 0.033875
        assert alloc.expected_yield == pytest.approx(0.033875, abs=0.0001)

    def test_conservative_product_weights(self, strategy):
        alloc = strategy.calculate_allocation(portfolio_value=100000, risk_profile="conservative")
        codes = {t.product_code: t.allocation_pct for t in alloc.tokenized_treasuries}
        assert codes["FOBXX"] == 0.60  # Favor SEC-registered
        assert codes["BUIDL"] == 0.30
        assert codes["TBT"] == 0.10

    def test_moderate_product_weights(self, strategy):
        alloc = strategy.calculate_allocation(portfolio_value=100000, risk_profile="moderate")
        codes = {t.product_code: t.allocation_pct for t in alloc.tokenized_treasuries}
        assert codes["BUIDL"] == 0.50
        assert codes["FOBXX"] == 0.35
        assert codes["TBT"] == 0.15

    def test_circuit_breaker_yellow_reduces_allocation(self, mock_circuit_breaker):
        mock_circuit_breaker.get_status.return_value = {"status": "yellow", "drawdown": -0.10}
        with patch(
            "src.crypto.institutional.DrawdownCircuitBreaker",
            return_value=mock_circuit_breaker,
        ):
            from src.crypto.institutional import TokenizedTreasuryStrategy
            s = TokenizedTreasuryStrategy()
            alloc = s.calculate_allocation(portfolio_value=100000, risk_profile="moderate")
            # 3% * 0.8 = 2.4%
            assert alloc.total_crypto_pct == pytest.approx(0.024, abs=0.001)

    def test_circuit_breaker_red_reduces_allocation(self, mock_circuit_breaker):
        mock_circuit_breaker.get_status.return_value = {"status": "red", "drawdown": -0.20}
        with patch(
            "src.crypto.institutional.DrawdownCircuitBreaker",
            return_value=mock_circuit_breaker,
        ):
            from src.crypto.institutional import TokenizedTreasuryStrategy
            s = TokenizedTreasuryStrategy()
            alloc = s.calculate_allocation(portfolio_value=100000, risk_profile="moderate")
            # 3% * 0.25 = 0.75%
            assert alloc.total_crypto_pct == pytest.approx(0.0075, abs=0.001)

    def test_circuit_breaker_orange_reduces_allocation(self, mock_circuit_breaker):
        mock_circuit_breaker.get_status.return_value = {"status": "orange", "drawdown": -0.15}
        with patch(
            "src.crypto.institutional.DrawdownCircuitBreaker",
            return_value=mock_circuit_breaker,
        ):
            from src.crypto.institutional import TokenizedTreasuryStrategy
            s = TokenizedTreasuryStrategy()
            alloc = s.calculate_allocation(portfolio_value=100000, risk_profile="moderate")
            # 3% * 0.5 = 1.5%
            assert alloc.total_crypto_pct == pytest.approx(0.015, abs=0.001)

    def test_circuit_breaker_black_zero_allocation(self, mock_circuit_breaker):
        mock_circuit_breaker.get_status.return_value = {"status": "black", "drawdown": -0.25}
        with patch(
            "src.crypto.institutional.DrawdownCircuitBreaker",
            return_value=mock_circuit_breaker,
        ):
            from src.crypto.institutional import TokenizedTreasuryStrategy
            s = TokenizedTreasuryStrategy()
            alloc = s.calculate_allocation(portfolio_value=100000, risk_profile="moderate")
            assert alloc.total_crypto_pct == 0.0
            assert alloc.total_crypto_usd == 0.0

    def test_product_performance_no_data(self, strategy, _tmp_data_dir):
        perf = strategy.get_product_performance("BUIDL", days=30)
        assert perf["product"] == "BUIDL"
        assert perf["current_nav"] == 1.00
        assert perf["data_points"] == 0

    def test_product_performance_with_data(self, strategy, _tmp_data_dir):
        from src.crypto.institutional import CRYPTO_DB
        conn = sqlite3.connect(CRYPTO_DB)
        conn.execute(
            "INSERT INTO product_performance (product_code, date, nav, apy_7d, aum_billions) "
            "VALUES (?, date('now'), ?, ?, ?)",
            ("BUIDL", 1.001, 0.0350, 2.50),
        )
        conn.execute(
            "INSERT INTO product_performance (product_code, date, nav, apy_7d, aum_billions) "
            "VALUES (?, date('now', '-1 day'), ?, ?, ?)",
            ("BUIDL", 1.000, 0.0340, 2.44),
        )
        conn.commit()
        conn.close()

        perf = strategy.get_product_performance("BUIDL", days=30)
        assert perf["product"] == "BUIDL"
        assert perf["data_points"] == 2
        assert perf["current_nav"] == 1.001

    def test_unknown_product_performance(self, strategy, _tmp_data_dir):
        perf = strategy.get_product_performance("UNKNOWN", days=30)
        assert perf["product"] == "UNKNOWN"
        assert perf["data_points"] == 0

    def test_invalid_risk_profile_falls_back(self, strategy):
        """Unknown risk_profile falls back to 0.03 (moderate) max allocation."""
        alloc = strategy.calculate_allocation(portfolio_value=100000, risk_profile="ultra_aggressive")
        assert alloc.total_crypto_pct == pytest.approx(0.03, abs=0.001)

    def test_rebalance_boundary_exact_threshold(self, strategy):
        """Deviation just under 0.005 should NOT trigger rebalance (strict >)."""
        alloc = strategy.calculate_allocation(
            portfolio_value=100000,
            risk_profile="moderate",
            current_allocation_pct=0.034,  # 3.4% vs target 3% → deviation 0.4% < 0.5%
        )
        assert alloc.rebalance_needed is False

    def test_calculate_expected_yield_empty(self, strategy):
        assert strategy._calculate_expected_yield([]) == 0.0

    def test_calculate_expected_yield_with_allocs(self, strategy):
        from src.crypto.institutional import TokenizedProductAllocation
        allocs = [
            TokenizedProductAllocation(
                product_code="BUIDL", product_name="BUIDL",
                allocation_pct=0.50, allocation_usd=900.0,
                expected_apy=0.0345, risk_group="group_1_tokenized",
                blockchains=["ethereum"], liquidity_score=0.90,
                regulatory_clearance=True, custody_rating="high",
            ),
            TokenizedProductAllocation(
                product_code="FOBXX", product_name="FOBXX",
                allocation_pct=0.50, allocation_usd=900.0,
                expected_apy=0.0325, risk_group="group_1_tokenized",
                blockchains=["stellar"], liquidity_score=0.80,
                regulatory_clearance=True, custody_rating="moderate",
            ),
        ]
        yield_val = strategy._calculate_expected_yield(allocs)
        # 0.50 * 0.0345 + 0.50 * 0.0325 = 0.0335
        assert yield_val == pytest.approx(0.0335, abs=0.0001)


# ---------------------------------------------------------------------------
# CryptoRiskManager
# ---------------------------------------------------------------------------

class TestCryptoRiskManager:
    def _make_allocation(self, **overrides):
        from src.crypto.institutional import CryptoAllocation
        defaults = dict(
            portfolio_value=100000, risk_profile="moderate",
            total_crypto_pct=0.03, total_crypto_usd=3000.0,
            group_1_allocation=0.60, group_2a_allocation=0.40,
            group_2b_allocation=0.0, tokenized_treasuries=[],
            weighted_risk_weight=0.52, capital_charge_pct=0.0416,
            expected_yield=0.0341, basel_compliant=True,
            sec_compliant=True, rebalance_needed=False,
        )
        defaults.update(overrides)
        return CryptoAllocation(**defaults)

    def test_assess_risk_moderate(self, risk_mgr):
        alloc = self._make_allocation()
        ra = risk_mgr.assess_risk(100000, alloc)
        assert ra.portfolio_value == 100000
        # group_1_rwa = 0.60 * 3000 * 0.20 = 360
        assert ra.group_1_rwa == pytest.approx(360.0, abs=1.0)
        # group_2a_rwa = 0.40 * 3000 * 1.00 = 1200
        assert ra.group_2a_rwa == pytest.approx(1200.0, abs=1.0)
        assert ra.total_rwa == pytest.approx(1560.0, abs=1.0)
        assert ra.within_sec_limits is True
        assert ra.within_basel_limits is True
        assert ra.limiting_factor is None

    def test_assess_risk_stress_loss(self, risk_mgr):
        alloc = self._make_allocation()
        ra = risk_mgr.assess_risk(100000, alloc)
        # stress = (0.60 * -0.05 + 0.40 * -0.10) * 3000 = (-0.03 - 0.04) * 3000 = -210
        assert ra.estimated_loss_stress == pytest.approx(-210.0, abs=1.0)
        assert ra.portfolio_impact_stress_pct == pytest.approx(0.0021, abs=0.0001)

    def test_assess_risk_with_group_2b(self, risk_mgr):
        alloc = self._make_allocation(
            group_1_allocation=0.50, group_2a_allocation=0.30,
            group_2b_allocation=0.20,
        )
        ra = risk_mgr.assess_risk(100000, alloc)
        # group_2b_rwa = 0.20 * 3000 * 12.50 = 7500
        assert ra.group_2b_rwa == pytest.approx(7500.0, abs=1.0)
        # group_2b = 0.20 > 0.01 → not within basel
        assert ra.within_basel_limits is False
        assert ra.limiting_factor == "basel_group_2b_limit"

    def test_assess_risk_cet1_buffer(self, risk_mgr):
        alloc = self._make_allocation()
        ra = risk_mgr.assess_risk(100000, alloc)
        # required_cet1 = total_rwa * 0.08 = 1560 * 0.08 = 124.8
        assert ra.required_cet1 == pytest.approx(124.8, abs=0.1)
        # available_cet1 = 100000 * 0.12 = 12000
        assert ra.available_cet1 == pytest.approx(12000.0, abs=1.0)
        # buffer = (12000 - 124.8) / 100000 ≈ 0.118752
        assert ra.buffer_pct == pytest.approx(0.11875, abs=0.001)

    def test_check_compliance_accredited(self, risk_mgr):
        report = risk_mgr.check_compliance(investor_type="accredited", portfolio_value=100000)
        assert report.sec_compliant is True
        assert "BUIDL" in report.eligible_products
        assert "FOBXX" in report.eligible_products
        assert "TBT" in report.eligible_products
        assert len(report.restricted_products) == 0

    def test_check_compliance_qualified_purchaser(self, risk_mgr):
        report = risk_mgr.check_compliance(investor_type="qualified_purchaser")
        assert report.sec_compliant is True
        assert report.qualified_purchaser_status is True

    def test_check_compliance_institution(self, risk_mgr):
        report = risk_mgr.check_compliance(investor_type="institution")
        assert report.sec_compliant is True
        assert report.qualified_purchaser_status is True
        assert report.basel_compliant is True

    def test_check_compliance_retail(self, risk_mgr):
        report = risk_mgr.check_compliance(investor_type="retail")
        assert report.sec_compliant is False
        assert report.eligible_products == ["FOBXX"]
        assert "BUIDL" in report.restricted_products
        assert "TBT" in report.restricted_products
        assert report.qualified_purchaser_status is False

    def test_rebalance_delta_new_allocation(self, risk_mgr):
        deltas = risk_mgr.calculate_rebalance_delta(
            current_allocations={},
            target_allocation_pct=0.03,
            portfolio_value=100000,
        )
        # No current allocation → split $3000 evenly across 3 products
        assert len(deltas) == 3
        assert sum(deltas.values()) == pytest.approx(3000.0, abs=1.0)

    def test_rebalance_delta_existing(self, risk_mgr):
        deltas = risk_mgr.calculate_rebalance_delta(
            current_allocations={"BUIDL": 2000, "FOBXX": 1000},
            target_allocation_pct=0.03,
            portfolio_value=100000,
        )
        assert len(deltas) == 2
        # target_total = 3000, BUIDL proportion = 2000/3000, target = 3000 * 2/3 = 2000
        # FOBXX proportion = 1000/3000, target = 3000 * 1/3 = 1000
        # deltas = target - current = 0 for each (already proportional)
        assert deltas["BUIDL"] == pytest.approx(0.0, abs=1.0)
        assert deltas["FOBXX"] == pytest.approx(0.0, abs=1.0)

    def test_rebalance_delta_needs_rebalance(self, risk_mgr):
        deltas = risk_mgr.calculate_rebalance_delta(
            current_allocations={"BUIDL": 1000, "FOBXX": 1000},
            target_allocation_pct=0.03,
            portfolio_value=100000,
        )
        # target_total = 3000, BUIDL target = 1500, FOBXX target = 1500
        assert deltas["BUIDL"] == pytest.approx(500.0, abs=1.0)
        assert deltas["FOBXX"] == pytest.approx(500.0, abs=1.0)


# ---------------------------------------------------------------------------
# Constants validation
# ---------------------------------------------------------------------------

class TestConstants:
    def test_basel_risk_weights(self):
        from src.crypto.institutional import BASEL_RISK_WEIGHTS
        assert BASEL_RISK_WEIGHTS["group_1_tokenized"] == 0.20
        assert BASEL_RISK_WEIGHTS["group_2a_stablecoins"] == 1.00
        assert BASEL_RISK_WEIGHTS["group_2b_unbacked"] == 12.50

    def test_max_allocation(self):
        from src.crypto.institutional import MAX_ALLOCATION
        assert MAX_ALLOCATION["conservative"] == 0.02
        assert MAX_ALLOCATION["moderate"] == 0.03
        assert MAX_ALLOCATION["aggressive"] == 0.05

    def test_tokenized_products(self):
        from src.crypto.institutional import TOKENIZED_TREASURY_PRODUCTS
        assert "BUIDL" in TOKENIZED_TREASURY_PRODUCTS
        assert "FOBXX" in TOKENIZED_TREASURY_PRODUCTS
        assert "TBT" in TOKENIZED_TREASURY_PRODUCTS
        assert TOKENIZED_TREASURY_PRODUCTS["BUIDL"]["risk_group"] == "group_1_tokenized"

    def test_regulatory_limits(self):
        from src.crypto.institutional import REGULATORY_LIMITS
        assert REGULATORY_LIMITS["sec_qualified_purchaser"] == 5_000_000
        assert REGULATORY_LIMITS["basel_3_group_2b_cap"] == 0.01

    def test_defi_protocols(self):
        from src.crypto.institutional import DEFI_PROTOCOLS
        assert "aave" in DEFI_PROTOCOLS
        assert "compound" in DEFI_PROTOCOLS
        assert "morpho" in DEFI_PROTOCOLS


# ---------------------------------------------------------------------------
# CLI (main)
# ---------------------------------------------------------------------------

class TestCLI:
    def test_analyze_command(self, _tmp_data_dir):
        from src.crypto.institutional import main
        with patch("sys.argv", ["institutional.py", "analyze", "--portfolio", "100000", "--json"]):
            # capture stdout
            import io
            from contextlib import redirect_stdout
            buf = io.StringIO()
            with redirect_stdout(buf):
                main()
            output = buf.getvalue()
            data = json.loads(output)
            assert "allocation" in data
            assert "risk_assessment" in data
            assert data["allocation"]["portfolio_value"] == 100000

    def test_compliance_check_command(self, _tmp_data_dir):
        from src.crypto.institutional import main
        with patch("sys.argv", ["institutional.py", "compliance-check", "--portfolio", "100000",
                                "--investor-type", "accredited", "--json"]):
            import io
            from contextlib import redirect_stdout
            buf = io.StringIO()
            with redirect_stdout(buf):
                main()
            output = buf.getvalue()
            data = json.loads(output)
            assert data["sec_compliant"] is True
            assert "BUIDL" in data["eligible_products"]

    def test_rebalance_command(self, _tmp_data_dir):
        from src.crypto.institutional import main
        with patch("sys.argv", ["institutional.py", "rebalance", "--portfolio", "100000",
                                "--current-crypto-pct", "2", "--target-crypto-pct", "3", "--json"]):
            import io
            from contextlib import redirect_stdout
            buf = io.StringIO()
            with redirect_stdout(buf):
                main()
            output = buf.getvalue()
            data = json.loads(output)
            assert "BUIDL" in data
            assert "FOBXX" in data
            assert "TBT" in data

    def test_rebalance_missing_target_exits(self, _tmp_data_dir):
        from src.crypto.institutional import main
        with patch("sys.argv", ["institutional.py", "rebalance", "--portfolio", "100000"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_performance_command(self, _tmp_data_dir):
        from src.crypto.institutional import main
        with patch("sys.argv", ["institutional.py", "performance", "--portfolio", "100000"]):
            import io
            from contextlib import redirect_stdout
            buf = io.StringIO()
            with redirect_stdout(buf):
                main()
            output = buf.getvalue()
            assert "BUIDL" in output
            assert "FOBXX" in output
