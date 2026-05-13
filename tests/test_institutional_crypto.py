#!/usr/bin/env python3
"""
Tests for institutional crypto module — data classes, tokenized treasury strategy,
Basel III risk management, compliance checking, and rebalancing.
"""
import sys
import os
import json
import sqlite3
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.crypto.institutional import (
    TokenizedProductAllocation, CryptoAllocation,
    RiskAssessment, ComplianceReport,
    TokenizedTreasuryStrategy, CryptoRiskManager,
    BASEL_RISK_WEIGHTS, MAX_ALLOCATION,
    TOKENIZED_TREASURY_PRODUCTS, REGULATORY_LIMITS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_circuit_breaker():
    """Create a mock circuit breaker that returns green status."""
    cb = MagicMock()
    cb.get_status.return_value = {"status": "green"}
    return cb


def _make_strategy(tmp_path):
    """Create a TokenizedTreasuryStrategy with mocked DB and circuit breaker."""
    with patch('src.crypto.institutional.init_database'):
        with patch('src.crypto.institutional.DrawdownCircuitBreaker', return_value=_mock_circuit_breaker()):
            strategy = TokenizedTreasuryStrategy.__new__(TokenizedTreasuryStrategy)
            strategy.circuit_breaker = _mock_circuit_breaker()
            return strategy


def _make_risk_manager(tmp_path):
    """Create a CryptoRiskManager with mocked DB and circuit breaker."""
    with patch('src.crypto.institutional.init_database'):
        with patch('src.crypto.institutional.DrawdownCircuitBreaker', return_value=_mock_circuit_breaker()):
            risk_mgr = CryptoRiskManager.__new__(CryptoRiskManager)
            risk_mgr.circuit_breaker = _mock_circuit_breaker()
            return risk_mgr


def _make_allocation(portfolio_value=100000, risk_profile="moderate", total_pct=0.03):
    """Create a CryptoAllocation for testing."""
    total_usd = portfolio_value * total_pct
    return CryptoAllocation(
        portfolio_value=portfolio_value,
        risk_profile=risk_profile,
        total_crypto_pct=total_pct,
        total_crypto_usd=total_usd,
        group_1_allocation=0.60,
        group_2a_allocation=0.40,
        group_2b_allocation=0.0,
        tokenized_treasuries=[],
        weighted_risk_weight=0.52,
        capital_charge_pct=0.0416,
        expected_yield=0.034,
        basel_compliant=True,
        sec_compliant=True,
        rebalance_needed=False,
    )


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

class TestConstants:
    """Test module constants."""

    def test_basel_risk_weights_keys(self):
        assert "group_1_tokenized" in BASEL_RISK_WEIGHTS
        assert "group_2a_stablecoins" in BASEL_RISK_WEIGHTS
        assert "group_2b_unbacked" in BASEL_RISK_WEIGHTS

    def test_group_1_weight(self):
        assert BASEL_RISK_WEIGHTS["group_1_tokenized"] == 0.20

    def test_group_2b_weight(self):
        assert BASEL_RISK_WEIGHTS["group_2b_unbacked"] == 12.50

    def test_max_allocation_tiers(self):
        assert "conservative" in MAX_ALLOCATION
        assert "moderate" in MAX_ALLOCATION
        assert "aggressive" in MAX_ALLOCATION

    def test_conservative_most_restrictive(self):
        assert MAX_ALLOCATION["conservative"] < MAX_ALLOCATION["moderate"]
        assert MAX_ALLOCATION["moderate"] < MAX_ALLOCATION["aggressive"]

    def test_tokenized_products_defined(self):
        assert "BUIDL" in TOKENIZED_TREASURY_PRODUCTS
        assert "FOBXX" in TOKENIZED_TREASURY_PRODUCTS
        assert "TBT" in TOKENIZED_TREASURY_PRODUCTS

    def test_buidl_aum(self):
        assert TOKENIZED_TREASURY_PRODUCTS["BUIDL"]["aum_billions"] == 2.44

    def test_regulatory_limits_defined(self):
        assert "sec_qualified_purchaser" in REGULATORY_LIMITS
        assert REGULATORY_LIMITS["sec_qualified_purchaser"] == 5_000_000


# ---------------------------------------------------------------------------
# TokenizedProductAllocation tests
# ---------------------------------------------------------------------------

class TestTokenizedProductAllocation:
    """Test TokenizedProductAllocation dataclass."""

    def test_creation(self):
        alloc = TokenizedProductAllocation(
            product_code="BUIDL",
            product_name="BlackRock BUIDL",
            allocation_pct=0.50,
            allocation_usd=1500.0,
            expected_apy=0.0345,
            risk_group="group_1_tokenized",
            blockchains=["ethereum", "solana"],
            liquidity_score=0.90,
            regulatory_clearance=True,
            custody_rating="high",
        )
        assert alloc.product_code == "BUIDL"
        assert alloc.allocation_pct == 0.50

    def test_to_dict(self):
        alloc = TokenizedProductAllocation(
            product_code="BUIDL",
            product_name="BlackRock BUIDL",
            allocation_pct=0.50,
            allocation_usd=1500.0,
            expected_apy=0.0345,
            risk_group="group_1_tokenized",
            blockchains=["ethereum"],
            liquidity_score=0.90,
            regulatory_clearance=True,
            custody_rating="high",
        )
        d = alloc.to_dict()
        assert d["product_code"] == "BUIDL"
        assert "blockchains" in d


# ---------------------------------------------------------------------------
# CryptoAllocation tests
# ---------------------------------------------------------------------------

class TestCryptoAllocation:
    """Test CryptoAllocation dataclass."""

    def test_creation(self):
        alloc = _make_allocation()
        assert alloc.portfolio_value == 100000
        assert alloc.risk_profile == "moderate"
        assert alloc.total_crypto_pct == 0.03

    def test_to_dict_rounds_values(self):
        alloc = _make_allocation()
        d = alloc.to_dict()
        assert isinstance(d["total_crypto_pct"], float)
        assert isinstance(d["total_crypto_usd"], float)

    def test_to_dict_has_tokenized_treasuries(self):
        alloc = _make_allocation()
        d = alloc.to_dict()
        assert "tokenized_treasuries" in d


# ---------------------------------------------------------------------------
# RiskAssessment tests
# ---------------------------------------------------------------------------

class TestRiskAssessment:
    """Test RiskAssessment dataclass."""

    def test_creation(self):
        risk = RiskAssessment(
            portfolio_value=100000,
            crypto_allocation_pct=0.03,
            group_1_rwa=360.0,
            group_2a_rwa=1200.0,
            group_2b_rwa=0.0,
            total_rwa=1560.0,
            required_cet1=124.8,
            available_cet1=12000.0,
            buffer_pct=0.1188,
            max_drawdown_2022=-0.70,
            estimated_loss_stress=-300.0,
            portfolio_impact_stress_pct=0.003,
            within_sec_limits=True,
            within_basel_limits=True,
            limiting_factor=None,
        )
        assert risk.total_rwa == 1560.0
        assert risk.within_basel_limits is True

    def test_to_dict(self):
        risk = RiskAssessment(
            portfolio_value=100000, crypto_allocation_pct=0.03,
            group_1_rwa=360.0, group_2a_rwa=1200.0, group_2b_rwa=0.0,
            total_rwa=1560.0, required_cet1=124.8, available_cet1=12000.0,
            buffer_pct=0.1188, max_drawdown_2022=-0.70, estimated_loss_stress=-300.0,
            portfolio_impact_stress_pct=0.003, within_sec_limits=True,
            within_basel_limits=True, limiting_factor=None,
        )
        d = risk.to_dict()
        assert "total_rwa" in d
        assert "within_basel_limits" in d


# ---------------------------------------------------------------------------
# ComplianceReport tests
# ---------------------------------------------------------------------------

class TestComplianceReport:
    """Test ComplianceReport dataclass."""

    def test_creation(self):
        report = ComplianceReport(
            report_date="2026-01-01",
            investor_type="accredited",
            eligible_products=["BUIDL", "FOBXX", "TBT"],
            restricted_products=[],
            sec_compliant=True,
            accreditation_status="accredited",
            qualified_purchaser_status=False,
            basel_compliant=True,
            tier_1_capital_ratio=0.12,
            group_2b_within_limits=True,
            custody_arrangement="coinbase_custody",
            insurance_coverage=250_000_000,
            audit_trail_complete=True,
        )
        assert report.sec_compliant is True
        assert len(report.eligible_products) == 3

    def test_to_dict(self):
        report = ComplianceReport(
            report_date="2026-01-01", investor_type="retail",
            eligible_products=["FOBXX"], restricted_products=["BUIDL", "TBT"],
            sec_compliant=False, accreditation_status="retail",
            qualified_purchaser_status=False, basel_compliant=True,
            tier_1_capital_ratio=0.12, group_2b_within_limits=True,
            custody_arrangement="self", insurance_coverage=0,
            audit_trail_complete=True,
        )
        d = report.to_dict()
        assert "eligible_products" in d


# ---------------------------------------------------------------------------
# TokenizedTreasuryStrategy tests
# ---------------------------------------------------------------------------

class TestTokenizedTreasuryStrategy:
    """Test TokenizedTreasuryStrategy."""

    def test_calculate_allocation_moderate(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        alloc = strategy.calculate_allocation(portfolio_value=100000, risk_profile="moderate")
        assert isinstance(alloc, CryptoAllocation)
        assert alloc.total_crypto_pct == 0.03  # moderate max

    def test_calculate_allocation_conservative(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        alloc = strategy.calculate_allocation(portfolio_value=100000, risk_profile="conservative")
        assert alloc.total_crypto_pct == 0.02  # conservative max
        assert alloc.group_1_allocation == 1.0  # 100% Group 1
        assert alloc.group_2a_allocation == 0.0

    def test_calculate_allocation_aggressive(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        alloc = strategy.calculate_allocation(portfolio_value=100000, risk_profile="aggressive")
        assert alloc.total_crypto_pct == 0.05  # aggressive max

    def test_group_2b_always_zero(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        for profile in ["conservative", "moderate", "aggressive"]:
            alloc = strategy.calculate_allocation(portfolio_value=100000, risk_profile=profile)
            assert alloc.group_2b_allocation == 0.0

    def test_tokenized_treasuries_populated(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        alloc = strategy.calculate_allocation(portfolio_value=100000, risk_profile="moderate")
        assert len(alloc.tokenized_treasuries) == 3
        codes = [t.product_code for t in alloc.tokenized_treasuries]
        assert "BUIDL" in codes
        assert "FOBXX" in codes
        assert "TBT" in codes

    def test_conservative_heavier_fobxx(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        alloc = strategy.calculate_allocation(portfolio_value=100000, risk_profile="conservative")
        fobxx = next(t for t in alloc.tokenized_treasuries if t.product_code == "FOBXX")
        buidl = next(t for t in alloc.tokenized_treasuries if t.product_code == "BUIDL")
        assert fobxx.allocation_pct > buidl.allocation_pct  # FOBXX favored in conservative

    def test_expected_yield_positive(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        alloc = strategy.calculate_allocation(portfolio_value=100000, risk_profile="moderate")
        assert alloc.expected_yield > 0
        assert alloc.expected_yield < 0.10  # Reasonable bound

    def test_weighted_risk_weight(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        alloc = strategy.calculate_allocation(portfolio_value=100000, risk_profile="moderate")
        # Group 1 (60%) * 0.20 + Group 2a (40%) * 1.00 = 0.52
        expected = 0.60 * 0.20 + 0.40 * 1.00
        assert abs(alloc.weighted_risk_weight - expected) < 0.01

    def test_basel_compliant_no_group_2b(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        alloc = strategy.calculate_allocation(portfolio_value=100000, risk_profile="moderate")
        assert alloc.basel_compliant is True

    def test_rebalance_needed_with_deviation(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        # Current at 1%, target is 3% (moderate) → deviation > 0.5%
        alloc = strategy.calculate_allocation(
            portfolio_value=100000, risk_profile="moderate",
            current_allocation_pct=0.01,
        )
        assert alloc.rebalance_needed is True

    def test_no_rebalance_small_deviation(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        # Current at 2.8%, target is 3% → deviation < 0.5%
        alloc = strategy.calculate_allocation(
            portfolio_value=100000, risk_profile="moderate",
            current_allocation_pct=0.028,
        )
        assert alloc.rebalance_needed is False

    def test_circuit_breaker_reduces_allocation(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        strategy.circuit_breaker.get_status.return_value = {"status": "orange"}
        alloc = strategy.calculate_allocation(portfolio_value=100000, risk_profile="moderate")
        # moderate 3% * 0.5 (orange) = 1.5%
        assert alloc.total_crypto_pct == pytest.approx(0.015, abs=0.001)

    def test_circuit_breaker_black_zeroes_allocation(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        strategy.circuit_breaker.get_status.return_value = {"status": "black"}
        alloc = strategy.calculate_allocation(portfolio_value=100000, risk_profile="aggressive")
        assert alloc.total_crypto_pct == 0.0
        assert alloc.total_crypto_usd == 0.0

    def test_calculate_expected_yield_empty(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        yield_val = strategy._calculate_expected_yield([])
        assert yield_val == 0.0

    def test_usd_matches_pct(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        alloc = strategy.calculate_allocation(portfolio_value=500000, risk_profile="moderate")
        assert alloc.total_crypto_usd == pytest.approx(500000 * 0.03)


# ---------------------------------------------------------------------------
# CryptoRiskManager tests
# ---------------------------------------------------------------------------

class TestCryptoRiskManager:
    """Test CryptoRiskManager."""

    def test_assess_risk_returns_risk_assessment(self, tmp_path):
        risk_mgr = _make_risk_manager(tmp_path)
        alloc = _make_allocation()
        risk = risk_mgr.assess_risk(100000, alloc)
        assert isinstance(risk, RiskAssessment)

    def test_rwa_non_negative(self, tmp_path):
        risk_mgr = _make_risk_manager(tmp_path)
        alloc = _make_allocation()
        risk = risk_mgr.assess_risk(100000, alloc)
        assert risk.total_rwa >= 0
        assert risk.group_1_rwa >= 0
        assert risk.group_2a_rwa >= 0

    def test_required_cet1_positive(self, tmp_path):
        risk_mgr = _make_risk_manager(tmp_path)
        alloc = _make_allocation()
        risk = risk_mgr.assess_risk(100000, alloc)
        assert risk.required_cet1 >= 0

    def test_stress_loss_negative(self, tmp_path):
        risk_mgr = _make_risk_manager(tmp_path)
        alloc = _make_allocation()
        risk = risk_mgr.assess_risk(100000, alloc)
        assert risk.estimated_loss_stress <= 0  # Loss is negative

    def test_portfolio_impact_non_negative(self, tmp_path):
        risk_mgr = _make_risk_manager(tmp_path)
        alloc = _make_allocation()
        risk = risk_mgr.assess_risk(100000, alloc)
        assert risk.portfolio_impact_stress_pct >= 0

    def test_within_sec_limits_normal(self, tmp_path):
        risk_mgr = _make_risk_manager(tmp_path)
        alloc = _make_allocation(total_pct=0.03)
        risk = risk_mgr.assess_risk(100000, alloc)
        assert risk.within_sec_limits is True

    def test_within_basel_limits_no_group_2b(self, tmp_path):
        risk_mgr = _make_risk_manager(tmp_path)
        alloc = _make_allocation()
        risk = risk_mgr.assess_risk(100000, alloc)
        assert risk.within_basel_limits is True

    def test_check_compliance_accredited(self, tmp_path):
        risk_mgr = _make_risk_manager(tmp_path)
        report = risk_mgr.check_compliance(investor_type="accredited", portfolio_value=100000)
        assert isinstance(report, ComplianceReport)
        assert report.sec_compliant is True
        assert len(report.eligible_products) == 3

    def test_check_compliance_retail(self, tmp_path):
        risk_mgr = _make_risk_manager(tmp_path)
        report = risk_mgr.check_compliance(investor_type="retail", portfolio_value=100000)
        assert report.sec_compliant is False
        assert "FOBXX" in report.eligible_products
        assert "BUIDL" in report.restricted_products

    def test_check_compliance_qp(self, tmp_path):
        risk_mgr = _make_risk_manager(tmp_path)
        report = risk_mgr.check_compliance(investor_type="qualified_purchaser", portfolio_value=10_000_000)
        assert report.qualified_purchaser_status is True
        assert report.sec_compliant is True

    def test_rebalance_delta_new_allocation(self, tmp_path):
        risk_mgr = _make_risk_manager(tmp_path)
        deltas = risk_mgr.calculate_rebalance_delta(
            current_allocations={"BUIDL": 0, "FOBXX": 0, "TBT": 0},
            target_allocation_pct=0.03,
            portfolio_value=100000,
        )
        # Target is $3000, split evenly across 3 products = $1000 each
        assert len(deltas) == 3
        for delta in deltas.values():
            assert delta == pytest.approx(1000.0)

    def test_rebalance_delta_existing(self, tmp_path):
        risk_mgr = _make_risk_manager(tmp_path)
        deltas = risk_mgr.calculate_rebalance_delta(
            current_allocations={"BUIDL": 2000, "FOBXX": 1000, "TBT": 0},
            target_allocation_pct=0.03,
            portfolio_value=100000,
        )
        # Target $3000 total, current $3000 total → proportional rebalance
        total_delta = sum(deltas.values())
        assert abs(total_delta) < 1.0  # Should be ~0

    def test_group_2b_stress_severe(self, tmp_path):
        risk_mgr = _make_risk_manager(tmp_path)
        alloc = _make_allocation(total_pct=0.05)
        # Add group 2b
        alloc.group_2b_allocation = 0.20
        alloc.group_1_allocation = 0.50
        alloc.group_2a_allocation = 0.30
        alloc.total_crypto_usd = 100000 * 0.05
        risk = risk_mgr.assess_risk(100000, alloc)
        # Group 2b has -70% stress → should increase loss
        assert risk.estimated_loss_stress < 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
