#!/usr/bin/env python3
"""
v2.42c ESG/Climate Integration Module
Portfolio construction with carbon intensity metrics and climate risk

Features:
- WACI (Weighted Average Carbon Intensity) calculation
- Scope 1/2/3 emissions tracking
- Climate scenario analysis (NGFS)
- ESG-tilted allocation optimizer
- Carbon pair trade signals
- Decarbonization pathway tracking
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class EmissionScope(Enum):
    """GHG Protocol emission scopes"""
    SCOPE_1 = 1  # Direct emissions
    SCOPE_2 = 2  # Indirect (energy)
    SCOPE_3 = 3  # Value chain
    ALL = "all"  # Combined


class ESGFactor(Enum):
    """ESG scoring factors"""
    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"
    GOVERNANCE = "governance"
    CLIMATE = "climate"


@dataclass
class CarbonMetrics:
    """Carbon intensity metrics for a holding"""
    symbol: str
    isin: Optional[str] = None
    
    # Emissions (tCO2e)
    scope_1: float = 0.0
    scope_2: float = 0.0
    scope_3: float = 0.0
    
    # Intensity metrics (tCO2e/$M revenue)
    scope_1_intensity: float = 0.0
    scope_2_intensity: float = 0.0
    scope_3_intensity: float = 0.0
    total_intensity: float = 0.0
    
    # Data quality
    scope_3_coverage: float = 0.0  # % of portfolio with Scope 3
    data_quality_score: float = 0.0  # 0-100
    
    # Temperature alignment
    implied_temperature_rise: Optional[float] = None  # °C
    net_zero_target_year: Optional[int] = None
    reduction_pledge: Optional[float] = None  # % reduction target


@dataclass
class ESGScore:
    """Comprehensive ESG scoring"""
    symbol: str
    
    # Sub-scores (0-100)
    environmental: float = 50.0
    social: float = 50.0
    governance: float = 50.0
    climate_risk: float = 50.0
    
    # Weighted composite
    composite: float = 50.0
    
    # Controversies (negative score impact)
    controversy_score: float = 0.0  # 0 = no issues, 100 = severe
    
    # Sector adjustment
    sector: Optional[str] = None
    sector_percentile: float = 50.0  # Relative to sector peers


@dataclass
class PortfolioClimateMetrics:
    """Aggregate climate metrics for portfolio"""
    portfolio_value: float
    
    # WACI - Weighted Average Carbon Intensity
    waci_scope_12: float = 0.0  # tCO2e/$M revenue
    waci_total: float = 0.0  # Including Scope 3
    
    # Total emissions
    total_scope_1: float = 0.0
    total_scope_2: float = 0.0
    total_scope_3: float = 0.0
    total_emissions: float = 0.0
    
    # Coverage
    coverage_pct: float = 0.0  # % of AUM with carbon data
    
    # Temperature alignment
    portfolio_temperature: float = 0.0  # Implied °C warming
    alignment_status: str = "unknown"  # aligned/committed/misaligned
    
    # Decarbonization tracking
    baseline_waci: float = 0.0
    current_waci: float = 0.0
    reduction_achieved_pct: float = 0.0
    target_reduction_pct: float = 0.0


class ClimateScenario(Enum):
    """NGFS climate scenarios"""
    NDC = "ndc"  # Nationally Determined Contributions
    NET_ZERO_2050 = "net_zero_2050"
    BELOW_2C = "below_2c"
    DELAYED_TRANSITION = "delayed_transition"
    CURRENT_POLICIES = "current_policies"


@dataclass
class ScenarioImpact:
    """Climate scenario impact on portfolio"""
    scenario: str
    
    # Asset value impact by 2050
    equity_impact_pct: float = 0.0
    credit_spread_widening: float = 0.0  # bps
    
    # Sector impacts
    sector_impacts: Dict[str, float] = None
    
    # Transition/physical split
    transition_risk_pct: float = 0.0
    physical_risk_pct: float = 0.0
    
    # Opportunities
    green_revenue_opportunity: float = 0.0  # % of revenue


class ESGIntegrator:
    """ESG and climate integration engine"""
    
    # Benchmarks (tCO2e/$M revenue)
    MSCI_ACWI_WACI_SCOPE_12 = 120.0
    MSCI_ACWI_WACI_TOTAL = 450.0
    
    # Temperature targets
    PARIS_TARGET = 1.5
    WELL_BELOW_2C = 2.0
    
    def __init__(self):
        self.sample_data = self._load_sample_data()
        
    def _load_sample_data(self) -> Dict[str, CarbonMetrics]:
        """Load sample carbon data for major ETFs/stocks"""
        return {
            'SPY': CarbonMetrics(
                symbol='SPY',
                scope_1=85000, scope_2=45000, scope_3=1250000,
                scope_1_intensity=35.2, scope_2_intensity=18.6,
                scope_3_intensity=517.0, total_intensity=570.8,
                data_quality_score=85.0,
                implied_temperature_rise=2.8,
                reduction_pledge=0.30
            ),
            'QQQ': CarbonMetrics(
                symbol='QQQ',
                scope_1=45000, scope_2=38000, scope_3=890000,
                scope_1_intensity=28.5, scope_2_intensity=24.1,
                scope_3_intensity=564.2, total_intensity=616.8,
                data_quality_score=78.0,
                implied_temperature_rise=2.4,
                reduction_pledge=0.40
            ),
            'TLT': CarbonMetrics(
                symbol='TLT',  # Treasuries - sovereign emissions
                scope_1=0, scope_2=0, scope_3=120000,
                scope_1_intensity=5.2, scope_2_intensity=0.0,
                scope_3_intensity=45.0, total_intensity=50.2,
                data_quality_score=60.0,
                implied_temperature_rise=1.8,
                reduction_pledge=0.0
            ),
            'GLD': CarbonMetrics(
                symbol='GLD',  # Gold - mining emissions
                scope_1=25000, scope_2=8000, scope_3=45000,
                scope_1_intensity=125.0, scope_2_intensity=40.0,
                scope_3_intensity=225.0, total_intensity=390.0,
                data_quality_score=70.0,
                implied_temperature_rise=2.1,
                reduction_pledge=0.25
            ),
            'IEF': CarbonMetrics(
                symbol='IEF',  # 7-10y Treasuries
                scope_1=0, scope_2=0, scope_3=85000,
                scope_1_intensity=4.8, scope_2_intensity=0.0,
                scope_3_intensity=38.5, total_intensity=43.3,
                data_quality_score=58.0,
                implied_temperature_rise=1.7,
                reduction_pledge=0.0
            ),
            'VXUS': CarbonMetrics(
                symbol='VXUS',  # International developed
                scope_1=95000, scope_2=52000, scope_3=1450000,
                scope_1_intensity=62.5, scope_2_intensity=34.2,
                scope_3_intensity=953.8, total_intensity=1050.5,
                data_quality_score=72.0,
                implied_temperature_rise=3.1,
                reduction_pledge=0.35
            ),
            'MTUM': CarbonMetrics(
                symbol='MTUM',  # Momentum factor
                scope_1=72000, scope_2=41000, scope_3=1150000,
                scope_1_intensity=42.8, scope_2_intensity=24.3,
                scope_3_intensity=683.5, total_intensity=750.6,
                data_quality_score=80.0,
                implied_temperature_rise=2.6,
                reduction_pledge=0.32
            ),
            'VLUE': CarbonMetrics(
                symbol='VLUE',  # Value factor
                scope_1=110000, scope_2=58000, scope_3=1650000,
                scope_1_intensity=78.5, scope_2_intensity=41.4,
                scope_3_intensity=1177.3, total_intensity=1297.2,
                data_quality_score=75.0,
                implied_temperature_rise=3.2,
                reduction_pledge=0.28
            ),
            'BEP': CarbonMetrics(  # Brookfield Renewable
                symbol='BEP',
                scope_1=500, scope_2=1200, scope_3=8500,
                scope_1_intensity=2.5, scope_2_intensity=6.0,
                scope_3_intensity=42.5, total_intensity=51.0,
                data_quality_score=88.0,
                implied_temperature_rise=1.2,
                reduction_pledge=1.0  # Already net zero commitment
            ),
            'HASI': CarbonMetrics(  # Hannon Armstrong
                symbol='HASI',
                scope_1=200, scope_2=350, scope_3=2800,
                scope_1_intensity=1.8, scope_2_intensity=3.2,
                scope_3_intensity=25.5, total_intensity=30.5,
                data_quality_score=92.0,
                implied_temperature_rise=1.1,
                reduction_pledge=1.0
            ),
        }
    
    def calculate_waci(self, holdings: List[Tuple[str, float]], 
                      include_scope_3: bool = True) -> PortfolioClimateMetrics:
        """
        Calculate WACI for portfolio
        holdings: [(symbol, weight), ...]
        """
        total_weight = sum(w for _, w in holdings)
        normalized = [(s, w/total_weight) for s, w in holdings]
        
        waci_scope_12 = 0.0
        waci_total = 0.0
        coverage_weight = 0.0
        temp_sum = 0.0
        
        for symbol, weight in normalized:
            if symbol in self.sample_data:
                data = self.sample_data[symbol]
                coverage_weight += weight
                
                # Scope 1+2 WACI
                s12_intensity = data.scope_1_intensity + data.scope_2_intensity
                waci_scope_12 += s12_intensity * weight
                
                # Total WACI (including Scope 3)
                if include_scope_3:
                    waci_total += data.total_intensity * weight
                else:
                    waci_total += s12_intensity * weight
                    
                # Temperature alignment (weighted)
                if data.implied_temperature_rise:
                    temp_sum += data.implied_temperature_rise * weight
        
        portfolio_temp = temp_sum / coverage_weight if coverage_weight > 0 else 0
        
        # Alignment status
        if portfolio_temp <= self.PARIS_TARGET:
            alignment = "aligned"
        elif portfolio_temp <= self.WELL_BELOW_2C:
            alignment = "committed"
        else:
            alignment = "misaligned"
            
        return PortfolioClimateMetrics(
            portfolio_value=sum(w for _, w in holdings) * 100000,  # Assume $100k base
            waci_scope_12=waci_scope_12,
            waci_total=waci_total,
            coverage_pct=coverage_weight * 100,
            portfolio_temperature=portfolio_temp,
            alignment_status=alignment
        )
    
    def esg_score_portfolio(self, holdings: List[Tuple[str, float]]) -> Dict:
        """Calculate aggregate ESG scores for portfolio"""
        total_weight = sum(w for _, w in holdings)
        
        # Weighted ESG scores
        env_score = 0.0
        soc_score = 0.0
        gov_score = 0.0
        climate_score = 0.0
        coverage = 0.0
        
        # Sample ESG scores
        esg_data = {
            'SPY': ESGScore('SPY', environmental=55, social=62, governance=70, climate_risk=58),
            'QQQ': ESGScore('QQQ', environmental=62, social=68, governance=75, climate_risk=65),
            'TLT': ESGScore('TLT', environmental=72, social=65, governance=80, climate_risk=75),
            'GLD': ESGScore('GLD', environmental=48, social=55, governance=60, climate_risk=52),
            'IEF': ESGScore('IEF', environmental=75, social=68, governance=82, climate_risk=78),
            'VXUS': ESGScore('VXUS', environmental=52, social=58, governance=65, climate_risk=55),
            'BEP': ESGScore('BEP', environmental=92, social=75, governance=78, climate_risk=95),
            'HASI': ESGScore('HASI', environmental=94, social=72, governance=76, climate_risk=96),
            'MTUM': ESGScore('MTUM', environmental=58, social=65, governance=72, climate_risk=60),
            'VLUE': ESGScore('VLUE', environmental=52, social=58, governance=68, climate_risk=55),
        }
        
        for symbol, weight in holdings:
            norm_weight = weight / total_weight
            if symbol in esg_data:
                esg = esg_data[symbol]
                coverage += norm_weight
                env_score += esg.environmental * norm_weight
                soc_score += esg.social * norm_weight
                gov_score += esg.governance * norm_weight
                climate_score += esg.climate_risk * norm_weight
        
        composite = (env_score * 0.35 + soc_score * 0.25 + 
                    gov_score * 0.20 + climate_score * 0.20)
        
        return {
            'environmental': env_score,
            'social': soc_score,
            'governance': gov_score,
            'climate': climate_score,
            'composite': composite,
            'coverage_pct': coverage * 100
        }
    
    def scenario_analysis(self, holdings: List[Tuple[str, float]], 
                         scenario: ClimateScenario) -> ScenarioImpact:
        """Analyze portfolio impact under climate scenario"""
        
        # Simplified scenario impacts (equity value change by 2050)
        scenario_impacts = {
            ClimateScenario.NDC: -5.0,
            ClimateScenario.NET_ZERO_2050: -15.0,  # Transition winners/losers
            ClimateScenario.BELOW_2C: -20.0,  # Faster transition
            ClimateScenario.DELAYED_TRANSITION: -25.0,  # Disorderly late action
            ClimateScenario.CURRENT_POLICIES: -35.0,  # Physical risks dominate
        }
        
        base_impact = scenario_impacts.get(scenario, -10.0)
        
        # Adjust for portfolio composition
        weighted_impact = 0.0
        for symbol, weight in holdings:
            symbol_impact = base_impact
            
            # Green assets benefit from transition scenarios
            if symbol in ['BEP', 'HASI', 'ICLN', 'PBW']:
                if scenario in [ClimateScenario.NET_ZERO_2050, ClimateScenario.BELOW_2C]:
                    symbol_impact = 15.0  # Gain value
                else:
                    symbol_impact = base_impact * 0.3  # Less impacted
            
            # High carbon assets more impacted
            if symbol in ['VLUE', 'XLE', 'XOP']:
                if scenario in [ClimateScenario.NET_ZERO_2050, ClimateScenario.BELOW_2C]:
                    symbol_impact = base_impact * 2.0  # Double impact
                    
            weighted_impact += symbol_impact * (weight / sum(w for _, w in holdings))
        
        # Sector impacts
        sector_impacts = {
            'energy': base_impact * 1.5,
            'utilities': base_impact * 0.8,
            'tech': base_impact * 0.5,
            'healthcare': base_impact * 0.3,
            'financials': base_impact * 0.6,
            'renewables': -base_impact * 0.5 if scenario != ClimateScenario.CURRENT_POLICIES else 0
        }
        
        transition_pct = 60.0 if scenario != ClimateScenario.CURRENT_POLICIES else 30.0
        physical_pct = 100.0 - transition_pct
        
        return ScenarioImpact(
            scenario=scenario.value,
            equity_impact_pct=weighted_impact,
            sector_impacts=sector_impacts,
            transition_risk_pct=transition_pct,
            physical_risk_pct=physical_pct
        )
    
    def optimize_esg_tilt(self, base_weights: List[Tuple[str, float]],
                         esg_target: float = 60.0,
                         tracking_error_limit: float = 2.0) -> List[Tuple[str, float]]:
        """
        Optimize portfolio for ESG tilt within tracking error constraint
        Simple heuristic approach (full optimizer would use quadratic programming)
        """
        esg_data = {
            'SPY': 60, 'QQQ': 67, 'TLT': 74, 'GLD': 54, 'IEF': 76,
            'VXUS': 57, 'BEP': 85, 'HASI': 85, 'MTUM': 63, 'VLUE': 58
        }
        
        # Score each holding
        scored = [(s, w, esg_data.get(s, 50)) for s, w in base_weights]
        
        # Sort by ESG score
        sorted_by_esg = sorted(scored, key=lambda x: x[2], reverse=True)
        
        # Gradually shift weight from low to high ESG
        optimized = []
        total_shift = 0.0
        max_shift = tracking_error_limit / 100.0  # As fraction of portfolio
        
        for i, (symbol, weight, score) in enumerate(sorted_by_esg):
            if score >= esg_target and i < len(sorted_by_esg) / 2:
                # Increase high ESG
                shift = min(weight * 0.15, max_shift - total_shift)
                new_weight = weight + shift
                total_shift += shift
            elif score < esg_target and total_shift > 0:
                # Decrease low ESG
                reduction = min(weight * 0.10, total_shift)
                new_weight = weight - reduction
                total_shift -= reduction
            else:
                new_weight = weight
                
            optimized.append((symbol, new_weight))
        
        # Normalize to sum to 1
        total = sum(w for _, w in optimized)
        return [(s, w/total) for s, w in optimized]
    
    def carbon_pair_signals(self, long_symbol: str, short_symbol: str,
                           lookback: int = 30) -> dict:
        """
        Generate carbon pair trade signals
        Long low carbon, short high carbon
        """
        if long_symbol not in self.sample_data or short_symbol not in self.sample_data:
            return {'error': 'Insufficient carbon data'}
        
        long_data = self.sample_data[long_symbol]
        short_data = self.sample_data[short_symbol]
        
        # Carbon spread
        carbon_spread = short_data.total_intensity - long_data.total_intensity
        
        # Signal strength
        if carbon_spread > 1000:
            signal_strength = "strong"
            conviction = 0.8
        elif carbon_spread > 500:
            signal_strength = "moderate"
            conviction = 0.6
        else:
            signal_strength = "weak"
            conviction = 0.4
        
        # Expected convergence (as carbon pricing spreads)
        annual_return_estimate = carbon_spread / 1000 * 2.0  # Simplified
        
        return {
            'long': long_symbol,
            'short': short_symbol,
            'long_waci': long_data.total_intensity,
            'short_waci': short_data.total_intensity,
            'carbon_spread': carbon_spread,
            'signal_strength': signal_strength,
            'conviction': conviction,
            'expected_annual_return_pct': annual_return_estimate,
            'thesis': f"Long {long_symbol} ({long_data.total_intensity:.0f} tCO2e/$M) vs "
                     f"Short {short_symbol} ({short_data.total_intensity:.0f} tCO2e/$M)"
        }


def cmd_analyze(args):
    """Analyze portfolio ESG/climate metrics"""
    integrator = ESGIntegrator()
    
    # Parse holdings
    holdings = []
    for holding in args.holdings:
        parts = holding.split(':')
        if len(parts) == 2:
            symbol, weight = parts[0], float(parts[1])
            holdings.append((symbol, weight))
    
    print(f"\n🌱 ESG/Climate Portfolio Analysis")
    print(f"{'='*60}")
    print(f"Portfolio: {[s for s, _ in holdings]}")
    
    # WACI calculation
    waci = integrator.calculate_waci(holdings, include_scope_3=args.scope3)
    
    print(f"\n🔋 Carbon Intensity (WACI)")
    print(f"{'-'*60}")
    print(f"Scope 1+2: {waci.waci_scope_12:.1f} tCO2e/$M revenue")
    print(f"Total (inc. Scope 3): {waci.waci_total:.1f} tCO2e/$M revenue")
    print(f"Coverage: {waci.coverage_pct:.1f}%")
    
    # Benchmark comparison
    print(f"\nBenchmark Comparison:")
    print(f"  vs MSCI ACWI Scope 1+2 ({integrator.MSCI_ACWI_WACI_SCOPE_12:.0f}): "
          f"{((waci.waci_scope_12 / integrator.MSCI_ACWI_WACI_SCOPE_12 - 1) * 100):+.1f}%")
    print(f"  vs MSCI ACWI Total ({integrator.MSCI_ACWI_WACI_TOTAL:.0f}): "
          f"{((waci.waci_total / integrator.MSCI_ACWI_WACI_TOTAL - 1) * 100):+.1f}%")
    
    # Temperature alignment
    print(f"\n🌡️  Temperature Alignment")
    print(f"{'-'*60}")
    print(f"Implied Temperature Rise: {waci.portfolio_temperature:.1f}°C")
    print(f"Paris Target: {integrator.PARIS_TARGET}°C")
    print(f"Status: {waci.alignment_status.upper()}")
    
    # ESG scores
    esg = integrator.esg_score_portfolio(holdings)
    
    print(f"\n📊 ESG Scores (0-100)")
    print(f"{'-'*60}")
    print(f"Environmental: {esg['environmental']:.1f}")
    print(f"Social: {esg['social']:.1f}")
    print(f"Governance: {esg['governance']:.1f}")
    print(f"Climate Risk: {esg['climate']:.1f}")
    print(f"Composite: {esg['composite']:.1f}")
    print(f"Coverage: {esg['coverage_pct']:.1f}%")
    
    # Scenario analysis
    if args.scenario:
        print(f"\n⚠️  Climate Scenario: {args.scenario.upper()}")
        print(f"{'-'*60}")
        
        scenario_enum = ClimateScenario(args.scenario.lower().replace('_', '_'))
        impact = integrator.scenario_analysis(holdings, scenario_enum)
        
        print(f"Equity Impact by 2050: {impact.equity_impact_pct:+.1f}%")
        print(f"Transition Risk: {impact.transition_risk_pct:.0f}%")
        print(f"Physical Risk: {impact.physical_risk_pct:.0f}%")


def cmd_optimize(args):
    """Optimize portfolio for ESG tilt"""
    integrator = ESGIntegrator()
    
    # Parse base weights
    holdings = []
    for holding in args.holdings:
        parts = holding.split(':')
        if len(parts) == 2:
            holdings.append((parts[0], float(parts[1])))
    
    print(f"\n🔄 ESG Optimization")
    print(f"{'='*60}")
    print(f"Target ESG Score: {args.target}")
    print(f"Tracking Error Limit: {args.tracking_error}%")
    
    print(f"\nOriginal Weights:")
    total = sum(w for _, w in holdings)
    for s, w in holdings:
        print(f"  {s}: {w/total*100:.1f}%")
    
    optimized = integrator.optimize_esg_tilt(
        holdings, 
        esg_target=args.target,
        tracking_error_limit=args.tracking_error
    )
    
    print(f"\nOptimized Weights:")
    for s, w in optimized:
        orig = next((ow for os, ow in holdings if os == s), 0) / total
        change = (w - orig) * 100
        marker = "↑" if change > 0.5 else ("↓" if change < -0.5 else "=")
        print(f"  {s}: {w*100:.1f}% ({change:+.1f}%) {marker}")


def cmd_pair(args):
    """Carbon pair trade analysis"""
    integrator = ESGIntegrator()
    
    signal = integrator.carbon_pair_signals(args.long_symbol, args.short_symbol)
    
    print(f"\n📈 Carbon Pair Trade: {args.long_symbol} / {args.short_symbol}")
    print(f"{'='*60}")
    print(f"Thesis: {signal['thesis']}")
    print(f"Carbon Spread: {signal['carbon_spread']:.0f} tCO2e/$M")
    print(f"Signal Strength: {signal['signal_strength'].upper()}")
    print(f"Conviction: {signal['conviction']*100:.0f}%")
    print(f"Expected Annual Return: {signal['expected_annual_return_pct']:+.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='v2.42c ESG/Climate Integration Module',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s analyze SPY:0.46 GLD:0.38 TLT:0.16
  %(prog)s analyze SPY:0.5 GLD:0.3 BEP:0.1 HASI:0.1 --scenario net_zero_2050
  %(prog)s optimize SPY:0.46 GLD:0.38 TLT:0.16 --target 70
  %(prog)s pair --long BEP --short VLUE
        """
    )
    
    subparsers = parser.add_subparsers(dest='command')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze ESG/climate')
    analyze_parser.add_argument('holdings', nargs='+', help='Symbol:Weight pairs')
    analyze_parser.add_argument('--scope3', action='store_true', help='Include Scope 3')
    analyze_parser.add_argument('--scenario', type=str, choices=[s.value for s in ClimateScenario],
                               help='Climate scenario for stress test')
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize ESG tilt')
    optimize_parser.add_argument('holdings', nargs='+', help='Symbol:Weight pairs')
    optimize_parser.add_argument('--target', type=float, default=65.0, help='Target ESG score')
    optimize_parser.add_argument('--tracking-error', type=float, default=2.0, 
                              help='Max tracking error %')
    optimize_parser.set_defaults(func=cmd_optimize)
    
    # Pair command
    pair_parser = subparsers.add_parser('pair', help='Carbon pair trade')
    pair_parser.add_argument('--long', dest='long_symbol', required=True, help='Long symbol')
    pair_parser.add_argument('--short', dest='short_symbol', required=True, help='Short symbol')
    pair_parser.set_defaults(func=cmd_pair)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
        
    args.func(args)


if __name__ == '__main__':
    main()
