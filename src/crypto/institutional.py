#!/usr/bin/env python3
"""
Portfolio-Lab v2.23b: Institutional Crypto Module

Tokenized securities, Basel III risk management, and regulatory compliance
for institutional crypto allocation (0-5% portfolio max).

Based on Q3 2026 research synthesis:
- BlackRock BUIDL: $2.44B AUM, 3.45% 7D APY, 9+ blockchains
- Franklin Templeton FOBXX: First US-registered blockchain mutual fund
- Basel III Group 2b: 1,250% risk weight for unbacked crypto

Usage:
    from src.crypto.institutional import TokenizedTreasuryStrategy, CryptoRiskManager
    
    strategy = TokenizedTreasuryStrategy()
    allocation = strategy.calculate_allocation(portfolio_value=100000, risk_profile="moderate")
    
    risk_mgr = CryptoRiskManager()
    risk_adjusted = risk_mgr.apply_risk_weights(allocation)

CLI:
    python -m src.crypto.institutional analyze --portfolio 100000 --risk-profile moderate
    python -m src.crypto.institutional compliance-check --allocation 0.02
    python -m src.crypto.institutional rebalance --current-spy 0.46 --target-crypto 0.03
"""

import json
import sqlite3
import sys
import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import statistics

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.strategy.circuit_breaker import DrawdownCircuitBreaker

# ---------------------------------------------------------------------------
# Constants and Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
CRYPTO_DB = DATA_DIR / "crypto_allocation.db"

# Basel III Risk Weights (per 2025 standards)
BASEL_RISK_WEIGHTS = {
    "group_1_tokenized": 0.20,       # Tokenized traditional assets (20%)
    "group_2a_stablecoins": 1.00,    # Stabilized crypto (100%)
    "group_2b_unbacked": 12.50,    # Unbacked crypto (1,250%)
}

# Maximum portfolio allocations by risk tier
MAX_ALLOCATION = {
    "conservative": 0.02,   # 0-2%: Group 1 only
    "moderate": 0.03,       # 0-3%: Groups 1+2a
    "aggressive": 0.05,     # 0-5%: All groups
}

# Tokenized Treasury Products (as of Q2 2026)
TOKENIZED_TREASURY_PRODUCTS = {
    "BUIDL": {
        "name": "BlackRock USD Institutional Digital Liquidity Fund",
        "aum_billions": 2.44,
        "apy_7d": 0.0345,
        "nav": 1.00,
        "blockchains": ["ethereum", "solana", "bnb", "arbitrum", "optimism", "polygon", "avalanche", "aptos", "base"],
        "access": "us_qualified_purchasers",
        "risk_group": "group_1_tokenized",
        "custody": "coinbase_custody",
        " auditor": "deloitte",
    },
    "FOBXX": {
        "name": "Franklin OnChain U.S. Government Money Fund",
        "aum_billions": 0.68,  # Estimated
        "apy_7d": 0.0325,
        "nav": 1.00,
        "blockchains": ["stellar", "avalanche", "polygon", "aptos", "ethereum", "solana", "base"],
        "access": "us_registered",
        "risk_group": "group_1_tokenized",
        "custody": "franklin_templeton_custody",
        " auditor": "kpmg",
    },
    "TBT": {
        "name": "OpenEden T-Bills",
        "aum_billions": 0.15,
        "apy_7d": 0.0350,
        "nav": 1.00,
        "blockchains": ["ethereum"],
        "access": "global",
        "risk_group": "group_1_tokenized",
        "custody": "fireblocks",
        " auditor": "pricewaterhouse",
    },
}

# DeFi Lending Protocols (skeleton for future integration)
DEFI_PROTOCOLS = {
    "aave": {
        "chain": "ethereum",
        "tvl_billions": 12.5,
        "avg_apy": 0.045,
        "risk_level": "moderate",
    },
    "compound": {
        "chain": "ethereum",
        "tvl_billions": 3.2,
        "avg_apy": 0.042,
        "risk_level": "moderate",
    },
    "morpho": {
        "chain": "ethereum",
        "tvl_billions": 1.8,
        "avg_apy": 0.055,
        "risk_level": "higher",
    },
}

# Regulatory thresholds
REGULATORY_LIMITS = {
    "sec_qualified_purchaser": 5_000_000,  # $5M investments
    "sec_accredited_investor_income": 200_000,  # Individual
    "sec_accredited_investor_networth": 1_000_000,  # Excluding primary residence
    "basel_3_group_2b_cap": 0.01,  # 1% of Tier 1 capital
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class TokenizedProductAllocation:
    """Allocation to a specific tokenized product."""
    product_code: str
    product_name: str
    allocation_pct: float  # 0.0 to 1.0 of crypto allocation
    allocation_usd: float
    
    # Product details
    expected_apy: float
    risk_group: str
    blockchains: List[str]
    
    # Risk metrics
    liquidity_score: float  # 0.0 to 1.0
    regulatory_clearance: bool
    custody_rating: str  # high, moderate, low
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CryptoAllocation:
    """Complete crypto allocation for a portfolio."""
    portfolio_value: float
    risk_profile: str  # conservative, moderate, aggressive
    
    # Allocation amounts
    total_crypto_pct: float  # Of total portfolio (0-5%)
    total_crypto_usd: float
    
    # By risk group
    group_1_allocation: float  # Tokenized securities
    group_2a_allocation: float  # Stablecoins
    group_2b_allocation: float  # Unbacked crypto
    
    # Product breakdown
    tokenized_treasuries: List[TokenizedProductAllocation]
    
    # Risk metrics
    weighted_risk_weight: float  # Basel III weighted average
    capital_charge_pct: float  # Required capital as % of allocation
    expected_yield: float  # Weighted APY
    
    # Compliance
    basel_compliant: bool
    sec_compliant: bool  # Based on investor accreditation
    
    # Rebalancing
    rebalance_needed: bool
    rebalance_threshold_pct: float = 0.005  # 0.5%
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "portfolio_value": self.portfolio_value,
            "risk_profile": self.risk_profile,
            "total_crypto_pct": round(self.total_crypto_pct, 4),
            "total_crypto_usd": round(self.total_crypto_usd, 2),
            "group_1_allocation": round(self.group_1_allocation, 4),
            "group_2a_allocation": round(self.group_2a_allocation, 4),
            "group_2b_allocation": round(self.group_2b_allocation, 4),
            "weighted_risk_weight": round(self.weighted_risk_weight, 2),
            "capital_charge_pct": round(self.capital_charge_pct, 4),
            "expected_yield": round(self.expected_yield, 4),
            "basel_compliant": self.basel_compliant,
            "sec_compliant": self.sec_compliant,
            "rebalance_needed": self.rebalance_needed,
            "timestamp": self.timestamp,
            "tokenized_treasuries": [t.to_dict() for t in self.tokenized_treasuries],
        }


@dataclass
class RiskAssessment:
    """Risk assessment for crypto allocation."""
    portfolio_value: float
    crypto_allocation_pct: float
    
    # Basel III metrics
    group_1_rwa: float  # Risk-weighted assets
    group_2a_rwa: float
    group_2b_rwa: float
    total_rwa: float
    
    # Capital requirements (assuming 8% CET1 ratio)
    required_cet1: float
    available_cet1: float  # Assume 12% for typical portfolio
    buffer_pct: float  # Surplus/deficit
    
    # Stress test
    max_drawdown_2022: float  # Based on 2022 crypto crash
    estimated_loss_stress: float
    portfolio_impact_stress_pct: float
    
    # Regulatory limits
    within_sec_limits: bool
    within_basel_limits: bool
    limiting_factor: Optional[str]
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComplianceReport:
    """Regulatory compliance report."""
    report_date: str
    investor_type: str  # retail, accredited, qualified_purchaser, institution
    
    # Product eligibility
    eligible_products: List[str]
    restricted_products: List[str]
    
    # SEC requirements
    sec_compliant: bool
    accreditation_status: str
    qualified_purchaser_status: bool
    
    # Basel III (for institutions)
    basel_compliant: bool
    tier_1_capital_ratio: float
    group_2b_within_limits: bool
    
    # Custody
    custody_arrangement: str
    insurance_coverage: float
    audit_trail_complete: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Database Setup
# ---------------------------------------------------------------------------

def init_database():
    """Initialize SQLite database for crypto allocations and tracking."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(CRYPTO_DB)
    cursor = conn.cursor()
    
    # Allocation history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS crypto_allocations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            portfolio_value REAL,
            risk_profile TEXT,
            total_crypto_pct REAL,
            total_crypto_usd REAL,
            group_1_allocation REAL,
            group_2a_allocation REAL,
            group_2b_allocation REAL,
            weighted_risk_weight REAL,
            expected_yield REAL,
            basel_compliant INTEGER,
            sec_compliant INTEGER,
            rebalance_needed INTEGER,
            products TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Risk assessments
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS risk_assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            portfolio_value REAL,
            crypto_allocation_pct REAL,
            total_rwa REAL,
            required_cet1 REAL,
            buffer_pct REAL,
            stress_loss_estimate REAL,
            within_sec_limits INTEGER,
            within_basel_limits INTEGER,
            limiting_factor TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Compliance reports
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS compliance_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_date TEXT NOT NULL,
            investor_type TEXT,
            sec_compliant INTEGER,
            basel_compliant INTEGER,
            eligible_products TEXT,
            restricted_products TEXT,
            custody_arrangement TEXT,
            audit_trail_complete INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Product performance tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS product_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_code TEXT NOT NULL,
            date TEXT NOT NULL,
            nav REAL,
            apy_7d REAL,
            aum_billions REAL,
            yield_30d_avg REAL,
            UNIQUE(product_code, date)
        )
    """)
    
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Tokenized Treasury Strategy
# ---------------------------------------------------------------------------

class TokenizedTreasuryStrategy:
    """
    Tokenized government bond allocation strategy.
    
    Implements BlackRock BUIDL-style and Franklin Templeton FOBXX-style
    allocations with yield optimization and risk management.
    """
    
    def __init__(self):
        init_database()
        self.circuit_breaker = DrawdownCircuitBreaker()
    
    def calculate_allocation(
        self,
        portfolio_value: float,
        risk_profile: str = "moderate",
        current_allocation_pct: Optional[float] = None,
    ) -> CryptoAllocation:
        """
        Calculate optimal crypto allocation for portfolio.
        
        Args:
            portfolio_value: Total portfolio value in USD
            risk_profile: conservative, moderate, or aggressive
            current_allocation_pct: Current crypto allocation (for rebalancing)
        
        Returns:
            CryptoAllocation with product breakdown
        """
        # Check circuit breaker status
        cb_status = self.circuit_breaker.get_status()
        
        # Reduce allocation if circuit breaker triggered
        cb_scalar = {
            "green": 1.0,
            "yellow": 0.8,
            "orange": 0.5,
            "red": 0.25,
            "black": 0.0,
        }.get(cb_status.get("status", "green"), 1.0)
        
        # Determine max allocation based on risk profile and circuit breaker
        max_pct = MAX_ALLOCATION.get(risk_profile, 0.03) * cb_scalar
        
        # Default allocation: 60% Group 1 (tokenized), 40% Group 2a (stablecoin yield)
        # No Group 2b (unbacked crypto) for institutional strategy
        target_group_1 = 0.60
        target_group_2a = 0.40
        target_group_2b = 0.0
        
        # Conservative: 100% Group 1
        if risk_profile == "conservative":
            target_group_1 = 1.0
            target_group_2a = 0.0
        
        # Calculate actual allocation
        if current_allocation_pct is not None:
            # Rebalancing scenario - check if rebalance needed
            deviation = abs(current_allocation_pct - max_pct)
            rebalance_needed = deviation > 0.005  # 0.5% threshold
            target_pct = max_pct if rebalance_needed else current_allocation_pct
        else:
            target_pct = max_pct
            rebalance_needed = False
        
        total_crypto_usd = portfolio_value * target_pct
        
        # Allocate by group
        group_1_usd = total_crypto_usd * target_group_1
        group_2a_usd = total_crypto_usd * target_group_2a
        group_2b_usd = total_crypto_usd * target_group_2b
        
        # Build tokenized treasury allocation
        tokenized_allocs = self._allocate_tokenized_treasuries(
            group_1_usd, risk_profile
        )
        
        # Calculate weighted risk weight (Basel III)
        total_risk_weight = (
            target_group_1 * BASEL_RISK_WEIGHTS["group_1_tokenized"] +
            target_group_2a * BASEL_RISK_WEIGHTS["group_2a_stablecoins"] +
            target_group_2b * BASEL_RISK_WEIGHTS["group_2b_unbacked"]
        )
        
        # Calculate capital charge (assuming 8% CET1 requirement)
        capital_charge = total_risk_weight * 0.08
        
        # Calculate expected yield
        expected_yield = self._calculate_expected_yield(tokenized_allocs)
        
        return CryptoAllocation(
            portfolio_value=portfolio_value,
            risk_profile=risk_profile,
            total_crypto_pct=target_pct,
            total_crypto_usd=total_crypto_usd,
            group_1_allocation=target_group_1,
            group_2a_allocation=target_group_2a,
            group_2b_allocation=target_group_2b,
            tokenized_treasuries=tokenized_allocs,
            weighted_risk_weight=total_risk_weight,
            capital_charge_pct=capital_charge,
            expected_yield=expected_yield,
            basel_compliant=True,  # Group 2b = 0 so always compliant
            sec_compliant=True,    # Assume US investor for now
            rebalance_needed=rebalance_needed,
            timestamp=datetime.now().isoformat(),
        )
    
    def _allocate_tokenized_treasuries(
        self,
        group_1_usd: float,
        risk_profile: str,
    ) -> List[TokenizedProductAllocation]:
        """Allocate Group 1 assets across tokenized treasury products."""
        allocations = []
        
        # Allocation weights by product
        weights = {
            "BUIDL": 0.50,  # 50% to BlackRock (largest, most liquid)
            "FOBXX": 0.35,  # 35% to Franklin Templeton (registered fund)
            "TBT": 0.15,    # 15% to OpenEden (diversification)
        }
        
        # Conservative: heavier weight to registered funds
        if risk_profile == "conservative":
            weights = {
                "BUIDL": 0.30,
                "FOBXX": 0.60,  # Favor SEC-registered
                "TBT": 0.10,
            }
        
        for code, weight in weights.items():
            product = TOKENIZED_TREASURY_PRODUCTS[code]
            alloc_usd = group_1_usd * weight
            alloc_pct = weight
            
            allocation = TokenizedProductAllocation(
                product_code=code,
                product_name=product["name"],
                allocation_pct=alloc_pct,
                allocation_usd=alloc_usd,
                expected_apy=product["apy_7d"],
                risk_group=product["risk_group"],
                blockchains=product["blockchains"],
                liquidity_score=0.90 if code == "BUIDL" else 0.80,
                regulatory_clearance=True,
                custody_rating="high" if product["custody"] == "coinbase_custody" else "moderate",
            )
            
            allocations.append(allocation)
        
        return allocations
    
    def _calculate_expected_yield(
        self,
        tokenized_allocs: List[TokenizedProductAllocation],
    ) -> float:
        """Calculate weighted expected yield."""
        if not tokenized_allocs:
            return 0.0
        
        total_yield = sum(
            a.allocation_pct * a.expected_apy for a in tokenized_allocs
        )
        
        return total_yield
    
    def get_product_performance(self, product_code: str, days: int = 30) -> Dict:
        """Get historical performance for a tokenized product."""
        conn = sqlite3.connect(CRYPTO_DB)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT date, nav, apy_7d, aum_billions
            FROM product_performance
            WHERE product_code = ?
            AND date >= date('now', '-{} days')
            ORDER BY date DESC
        """.format(days), (product_code,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            # Return synthetic data based on product config
            product = TOKENIZED_TREASURY_PRODUCTS.get(product_code, {})
            return {
                "product": product_code,
                "current_nav": product.get("nav", 1.00),
                "current_apy": product.get("apy_7d", 0.035),
                "aum_billions": product.get("aum_billions", 0),
                "apy_30d_avg": product.get("apy_7d", 0.035),
                "data_points": 0,
                "note": "Using configured values - no historical data",
            }
        
        navs = [r[1] for r in rows if r[1]]
        apys = [r[2] for r in rows if r[2]]
        aums = [r[3] for r in rows if r[3]]
        
        return {
            "product": product_code,
            "current_nav": navs[0] if navs else 1.00,
            "current_apy": apys[0] if apys else 0.035,
            "aum_billions": aums[0] if aums else 0,
            "apy_30d_avg": statistics.mean(apys) if apys else 0.035,
            "data_points": len(rows),
        }


# ---------------------------------------------------------------------------
# Crypto Risk Manager
# ---------------------------------------------------------------------------

class CryptoRiskManager:
    """
    Basel III compliant risk management for crypto allocations.
    
    Implements risk weight calculations, capital requirements,
    and stress testing per Basel III 2025 standards.
    """
    
    def __init__(self):
        init_database()
        self.circuit_breaker = DrawdownCircuitBreaker()
    
    def assess_risk(
        self,
        portfolio_value: float,
        crypto_allocation: CryptoAllocation,
    ) -> RiskAssessment:
        """
        Perform comprehensive risk assessment.
        
        Args:
            portfolio_value: Total portfolio value
            crypto_allocation: Current crypto allocation
        
        Returns:
            RiskAssessment with Basel III metrics
        """
        # Calculate RWA by group
        group_1_rwa = (
            crypto_allocation.group_1_allocation *
            crypto_allocation.total_crypto_usd *
            BASEL_RISK_WEIGHTS["group_1_tokenized"]
        )
        
        group_2a_rwa = (
            crypto_allocation.group_2a_allocation *
            crypto_allocation.total_crypto_usd *
            BASEL_RISK_WEIGHTS["group_2a_stablecoins"]
        )
        
        group_2b_rwa = (
            crypto_allocation.group_2b_allocation *
            crypto_allocation.total_crypto_usd *
            BASEL_RISK_WEIGHTS["group_2b_unbacked"]
        )
        
        total_rwa = group_1_rwa + group_2a_rwa + group_2b_rwa
        
        # Capital requirements (8% CET1)
        required_cet1 = total_rwa * 0.08
        
        # Assume 12% CET1 ratio for typical portfolio
        available_cet1 = portfolio_value * 0.12
        
        buffer_pct = (available_cet1 - required_cet1) / portfolio_value
        
        # Stress test (2022 crypto crash scenario)
        # Group 1: -5% (stable)
        # Group 2a: -10% (moderate)
        # Group 2b: -70% (severe)
        stress_loss = (
            crypto_allocation.group_1_allocation * -0.05 +
            crypto_allocation.group_2a_allocation * -0.10 +
            crypto_allocation.group_2b_allocation * -0.70
        ) * crypto_allocation.total_crypto_usd
        
        portfolio_impact = abs(stress_loss) / portfolio_value
        
        # Check limits
        within_sec = crypto_allocation.total_crypto_pct <= MAX_ALLOCATION["aggressive"]
        within_basel = crypto_allocation.group_2b_allocation <= 0.01  # 1% limit
        
        limiting_factor = None
        if not within_basel:
            limiting_factor = "basel_group_2b_limit"
        elif not within_sec:
            limiting_factor = "sec_concentration_limit"
        
        return RiskAssessment(
            portfolio_value=portfolio_value,
            crypto_allocation_pct=crypto_allocation.total_crypto_pct,
            group_1_rwa=group_1_rwa,
            group_2a_rwa=group_2a_rwa,
            group_2b_rwa=group_2b_rwa,
            total_rwa=total_rwa,
            required_cet1=required_cet1,
            available_cet1=available_cet1,
            buffer_pct=buffer_pct,
            max_drawdown_2022=-0.70,
            estimated_loss_stress=stress_loss,
            portfolio_impact_stress_pct=portfolio_impact,
            within_sec_limits=within_sec,
            within_basel_limits=within_basel,
            limiting_factor=limiting_factor,
            timestamp=datetime.now().isoformat(),
        )
    
    def check_compliance(
        self,
        investor_type: str = "accredited",
        portfolio_value: float = 100000,
    ) -> ComplianceReport:
        """
        Generate compliance report for investor type.
        
        Args:
            investor_type: retail, accredited, qualified_purchaser, institution
            portfolio_value: Portfolio value for eligibility checks
        
        Returns:
            ComplianceReport with eligibility and restrictions
        """
        # Determine eligible products
        eligible = []
        restricted = []
        
        if investor_type in ["accredited", "qualified_purchaser", "institution"]:
            eligible = list(TOKENIZED_TREASURY_PRODUCTS.keys())
        else:
            # Retail: Only registered funds (FOBXX)
            eligible = ["FOBXX"]
            restricted = ["BUIDL", "TBT"]
        
        # SEC compliance
        sec_compliant = investor_type in ["accredited", "qualified_purchaser", "institution"]
        
        # QP status
        qp_status = investor_type in ["qualified_purchaser", "institution"]
        
        # Basel compliance (always true for Group 1 only)
        basel_compliant = True
        
        return ComplianceReport(
            report_date=datetime.now().isoformat(),
            investor_type=investor_type,
            eligible_products=eligible,
            restricted_products=restricted,
            sec_compliant=sec_compliant,
            accreditation_status=investor_type,
            qualified_purchaser_status=qp_status,
            basel_compliant=basel_compliant,
            tier_1_capital_ratio=0.12,
            group_2b_within_limits=True,
            custody_arrangement="coinbase_custody",
            insurance_coverage=250_000_000,  # $250M
            audit_trail_complete=True,
        )
    
    def calculate_rebalance_delta(
        self,
        current_allocations: Dict[str, float],
        target_allocation_pct: float,
        portfolio_value: float,
    ) -> Dict[str, float]:
        """
        Calculate rebalancing trades needed.
        
        Args:
            current_allocations: Current product allocations {product: usd}
            target_allocation_pct: Target % of portfolio
            portfolio_value: Total portfolio value
        
        Returns:
            Dict of products and trade amounts (+buy, -sell)
        """
        target_usd = portfolio_value * target_allocation_pct
        
        # Calculate total current
        current_total = sum(current_allocations.values())
        
        # Calculate deltas
        deltas = {}
        
        if current_total == 0:
            # New allocation - distribute evenly
            products = list(TOKENIZED_TREASURY_PRODUCTS.keys())
            per_product = target_usd / len(products)
            for product in products:
                deltas[product] = per_product
        else:
            # Rebalance proportionally
            for product, current in current_allocations.items():
                target_product = target_usd * (current / current_total)
                deltas[product] = target_product - current
        
        return deltas


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Institutional Crypto Strategy v2.23b"
    )
    parser.add_argument(
        "command",
        choices=["analyze", "compliance-check", "rebalance", "performance"]
    )
    parser.add_argument("--portfolio", type=float, required=True, help="Portfolio value in USD")
    parser.add_argument("--risk-profile", default="moderate", choices=["conservative", "moderate", "aggressive"])
    parser.add_argument("--investor-type", default="accredited", 
                        choices=["retail", "accredited", "qualified_purchaser", "institution"])
    parser.add_argument("--current-crypto-pct", type=float, default=0.0, help="Current crypto allocation %")
    parser.add_argument("--target-crypto-pct", type=float, help="Target crypto allocation %")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    if args.command == "analyze":
        strategy = TokenizedTreasuryStrategy()
        allocation = strategy.calculate_allocation(
            portfolio_value=args.portfolio,
            risk_profile=args.risk_profile,
            current_allocation_pct=args.current_crypto_pct / 100 if args.current_crypto_pct else None,
        )
        
        # Risk assessment
        risk_mgr = CryptoRiskManager()
        risk = risk_mgr.assess_risk(args.portfolio, allocation)
        
        if args.json:
            result = {
                "allocation": allocation.to_dict(),
                "risk_assessment": risk.to_dict(),
            }
            print(json.dumps(result, indent=2))
        else:
            print(f"\n💰 Institutional Crypto Allocation v2.23b")
            print(f"   Portfolio Value: ${args.portfolio:,.2f}")
            print(f"   Risk Profile: {args.risk_profile}")
            print(f"   Circuit Breaker: {strategy.circuit_breaker.get_status()['status']}")
            print(f"\n   Target Allocation: {allocation.total_crypto_pct:.2%} (${allocation.total_crypto_usd:,.2f})")
            print(f"   Expected Yield: {allocation.expected_yield:.2%} APY")
            print(f"   Basel III Risk Weight: {allocation.weighted_risk_weight:.0%}")
            print(f"   Capital Charge: {allocation.capital_charge_pct:.2%}")
            print(f"\n   Group Breakdown:")
            print(f"   • Group 1 (Tokenized): {allocation.group_1_allocation:.0%}")
            print(f"   • Group 2a (Stablecoins): {allocation.group_2a_allocation:.0%}")
            print(f"   • Group 2b (Unbacked): {allocation.group_2b_allocation:.0%}")
            print(f"\n   Product Allocations:")
            for product in allocation.tokenized_treasuries:
                print(f"   • {product.product_code}: {product.allocation_pct:.0%} (${product.allocation_usd:,.2f}) @ {product.expected_apy:.2%} APY")
            print(f"\n   Risk Assessment:")
            print(f"   • Stress Loss (2022 scenario): ${risk.estimated_loss_stress:,.2f}")
            print(f"   • Portfolio Impact: {risk.portfolio_impact_stress_pct:.2%}")
            print(f"   • Basel Compliant: {'✅' if risk.within_basel_limits else '❌'}")
            print(f"   • Rebalance Needed: {'Yes' if allocation.rebalance_needed else 'No'}")
    
    elif args.command == "compliance-check":
        risk_mgr = CryptoRiskManager()
        report = risk_mgr.check_compliance(
            investor_type=args.investor_type,
            portfolio_value=args.portfolio,
        )
        
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(f"\n📋 Compliance Report ({report.report_date})")
            print(f"   Investor Type: {args.investor_type}")
            print(f"   SEC Compliant: {'✅' if report.sec_compliant else '❌'}")
            print(f"   Basel Compliant: {'✅' if report.basel_compliant else '❌'}")
            print(f"   QP Status: {'✅' if report.qualified_purchaser_status else '❌'}")
            print(f"\n   Eligible Products:")
            for product in report.eligible_products:
                print(f"   ✅ {product}")
            if report.restricted_products:
                print(f"\n   Restricted Products:")
                for product in report.restricted_products:
                    print(f"   ❌ {product}")
    
    elif args.command == "rebalance":
        if args.target_crypto_pct is None:
            print("Error: --target-crypto-pct required")
            sys.exit(1)
        
        risk_mgr = CryptoRiskManager()
        
        # Calculate deltas
        current = {"BUIDL": 0, "FOBXX": 0, "TBT": 0}  # Simplified
        if args.current_crypto_pct > 0:
            current_usd = args.portfolio * (args.current_crypto_pct / 100)
            current = {
                "BUIDL": current_usd * 0.50,
                "FOBXX": current_usd * 0.35,
                "TBT": current_usd * 0.15,
            }
        
        deltas = risk_mgr.calculate_rebalance_delta(
            current_allocations=current,
            target_allocation_pct=args.target_crypto_pct / 100,
            portfolio_value=args.portfolio,
        )
        
        if args.json:
            print(json.dumps(deltas, indent=2))
        else:
            print(f"\n🔄 Rebalancing Plan")
            print(f"   Target: {args.target_crypto_pct}% of portfolio")
            for product, delta in deltas.items():
                action = "BUY" if delta > 0 else "SELL" if delta < 0 else "HOLD"
                print(f"   {action} {product}: ${abs(delta):,.2f}")
    
    elif args.command == "performance":
        strategy = TokenizedTreasuryStrategy()
        
        print(f"\n📈 Tokenized Treasury Performance")
        for code in TOKENIZED_TREASURY_PRODUCTS.keys():
            perf = strategy.get_product_performance(code, days=30)
            print(f"\n   {code}:")
            print(f"   • NAV: ${perf['current_nav']:.4f}")
            print(f"   • 7D APY: {perf['current_apy']:.2%}")
            print(f"   • 30D Avg APY: {perf['apy_30d_avg']:.2%}")
            print(f"   • AUM: ${perf['aum_billions']:.2f}B")


if __name__ == "__main__":
    main()
