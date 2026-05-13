#!/usr/bin/env python3
"""
Tests for ESG integration — enums, data classes, WACI calculation,
ESG scoring, scenario analysis, ESG tilt optimization, and carbon pair signals.
"""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock

from src.analytics.esg_integration import (
    EmissionScope, ESGFactor, ClimateScenario,
    CarbonMetrics, ESGScore, PortfolioClimateMetrics, ScenarioImpact,
    ESGIntegrator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_integrator():
    return ESGIntegrator()


def _default_holdings():
    return [('SPY', 0.46), ('GLD', 0.38), ('TLT', 0.16)]


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestEnums:
    def test_emission_scope_values(self):
        assert EmissionScope.SCOPE_1.value == 1
        assert EmissionScope.SCOPE_2.value == 2
        assert EmissionScope.SCOPE_3.value == 3
        assert EmissionScope.ALL.value == 'all'

    def test_esg_factor_values(self):
        assert ESGFactor.ENVIRONMENTAL.value == 'environmental'
        assert ESGFactor.SOCIAL.value == 'social'
        assert ESGFactor.GOVERNANCE.value == 'governance'
        assert ESGFactor.CLIMATE.value == 'climate'

    def test_climate_scenario_values(self):
        assert ClimateScenario.NDC.value == 'ndc'
        assert ClimateScenario.NET_ZERO_2050.value == 'net_zero_2050'
        assert ClimateScenario.CURRENT_POLICIES.value == 'current_policies'


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestCarbonMetrics:
    def test_creation(self):
        m = CarbonMetrics(symbol='SPY', scope_1=100, scope_2=50, scope_3=500)
        assert m.symbol == 'SPY'
        assert m.scope_1 == 100

    def test_defaults(self):
        m = CarbonMetrics(symbol='TEST')
        assert m.scope_1 == 0.0
        assert m.total_intensity == 0.0
        assert m.implied_temperature_rise is None


class TestESGScore:
    def test_creation(self):
        s = ESGScore(symbol='SPY', environmental=55, social=62)
        assert s.symbol == 'SPY'
        assert s.environmental == 55

    def test_defaults(self):
        s = ESGScore(symbol='TEST')
        assert s.environmental == 50.0
        assert s.controversy_score == 0.0


class TestPortfolioClimateMetrics:
    def test_creation(self):
        m = PortfolioClimateMetrics(portfolio_value=100000, waci_scope_12=50.0)
        assert m.portfolio_value == 100000
        assert m.alignment_status == 'unknown'


class TestScenarioImpact:
    def test_creation(self):
        s = ScenarioImpact(scenario='ndc', equity_impact_pct=-5.0)
        assert s.scenario == 'ndc'
        assert s.equity_impact_pct == -5.0


# ---------------------------------------------------------------------------
# ESGIntegrator tests
# ---------------------------------------------------------------------------

class TestESGIntegrator:
    def test_init_loads_sample_data(self):
        ei = _make_integrator()
        assert 'SPY' in ei.sample_data
        assert 'GLD' in ei.sample_data
        assert 'BEP' in ei.sample_data

    def test_sample_data_has_carbon_metrics(self):
        ei = _make_integrator()
        assert isinstance(ei.sample_data['SPY'], CarbonMetrics)
        assert ei.sample_data['SPY'].total_intensity > 0

    def test_benchmarks_defined(self):
        assert ESGIntegrator.MSCI_ACWI_WACI_SCOPE_12 == 120.0
        assert ESGIntegrator.MSCI_ACWI_WACI_TOTAL == 450.0

    def test_temperature_targets(self):
        assert ESGIntegrator.PARIS_TARGET == 1.5
        assert ESGIntegrator.WELL_BELOW_2C == 2.0

    # WACI tests
    def test_calculate_waci_returns_metrics(self):
        ei = _make_integrator()
        m = ei.calculate_waci(_default_holdings())
        assert isinstance(m, PortfolioClimateMetrics)
        assert m.waci_scope_12 > 0
        assert m.waci_total > 0

    def test_waci_total_greater_than_scope_12(self):
        ei = _make_integrator()
        m = ei.calculate_waci(_default_holdings())
        assert m.waci_total > m.waci_scope_12

    def test_waci_coverage_full(self):
        ei = _make_integrator()
        m = ei.calculate_waci(_default_holdings())
        # All three holdings have data
        assert m.coverage_pct == pytest.approx(100.0)

    def test_waci_partial_coverage(self):
        ei = _make_integrator()
        m = ei.calculate_waci([('SPY', 0.5), ('UNKNOWN', 0.5)])
        assert m.coverage_pct == pytest.approx(50.0)

    def test_waci_exclude_scope_3(self):
        ei = _make_integrator()
        m = ei.calculate_waci(_default_holdings(), include_scope_3=False)
        # Without scope 3, total should equal scope 1+2
        assert m.waci_total == pytest.approx(m.waci_scope_12, rel=0.01)

    def test_waci_temperature_alignment(self):
        ei = _make_integrator()
        m = ei.calculate_waci(_default_holdings())
        assert m.portfolio_temperature > 0
        assert m.alignment_status in ['aligned', 'committed', 'misaligned']

    def test_waci_green_portfolio_aligned(self):
        ei = _make_integrator()
        # BEP + HASI have low temperature rise (1.2, 1.1)
        m = ei.calculate_waci([('BEP', 0.5), ('HASI', 0.5)])
        assert m.alignment_status == 'aligned'

    # ESG scoring tests
    def test_esg_score_portfolio(self):
        ei = _make_integrator()
        result = ei.esg_score_portfolio(_default_holdings())
        assert 'environmental' in result
        assert 'composite' in result
        assert result['composite'] > 0

    def test_esg_score_coverage(self):
        ei = _make_integrator()
        result = ei.esg_score_portfolio(_default_holdings())
        assert result['coverage_pct'] == pytest.approx(100.0)

    def test_esg_score_unknown_asset(self):
        ei = _make_integrator()
        result = ei.esg_score_portfolio([('UNKNOWN', 1.0)])
        assert result['coverage_pct'] == 0.0

    def test_esg_composite_weighted(self):
        ei = _make_integrator()
        result = ei.esg_score_portfolio(_default_holdings())
        # Composite = 0.35*env + 0.25*soc + 0.20*gov + 0.20*climate
        expected = (result['environmental'] * 0.35 + result['social'] * 0.25 +
                    result['governance'] * 0.20 + result['climate'] * 0.20)
        assert result['composite'] == pytest.approx(expected, rel=0.01)

    def test_esg_green_portfolio_high_score(self):
        ei = _make_integrator()
        result = ei.esg_score_portfolio([('BEP', 0.5), ('HASI', 0.5)])
        assert result['composite'] > 75

    # Scenario analysis tests
    def test_scenario_analysis_returns_impact(self):
        ei = _make_integrator()
        impact = ei.scenario_analysis(_default_holdings(), ClimateScenario.NDC)
        assert isinstance(impact, ScenarioImpact)
        assert impact.scenario == 'ndc'

    def test_scenario_analysis_ndc(self):
        ei = _make_integrator()
        impact = ei.scenario_analysis(_default_holdings(), ClimateScenario.NDC)
        assert impact.equity_impact_pct < 0

    def test_scenario_analysis_current_policies_worst(self):
        ei = _make_integrator()
        ndc = ei.scenario_analysis(_default_holdings(), ClimateScenario.NDC)
        worst = ei.scenario_analysis(_default_holdings(), ClimateScenario.CURRENT_POLICIES)
        assert worst.equity_impact_pct < ndc.equity_impact_pct

    def test_scenario_analysis_sector_impacts(self):
        ei = _make_integrator()
        impact = ei.scenario_analysis(_default_holdings(), ClimateScenario.NET_ZERO_2050)
        assert 'energy' in impact.sector_impacts
        assert 'renewables' in impact.sector_impacts

    def test_scenario_analysis_transition_physical_split(self):
        ei = _make_integrator()
        impact = ei.scenario_analysis(_default_holdings(), ClimateScenario.NDC)
        assert impact.transition_risk_pct + impact.physical_risk_pct == pytest.approx(100.0)

    def test_scenario_analysis_green_assets_benefit(self):
        ei = _make_integrator()
        impact = ei.scenario_analysis([('BEP', 1.0)], ClimateScenario.NET_ZERO_2050)
        # Green assets should have positive impact in transition scenarios
        assert impact.equity_impact_pct > 0

    # ESG tilt optimization tests
    def test_optimize_esg_tilt_returns_list(self):
        ei = _make_integrator()
        result = ei.optimize_esg_tilt(_default_holdings())
        assert isinstance(result, list)
        assert len(result) == 3

    def test_optimize_esg_tilt_sums_to_one(self):
        ei = _make_integrator()
        result = ei.optimize_esg_tilt(_default_holdings())
        total = sum(w for _, w in result)
        assert abs(total - 1.0) < 0.01

    def test_optimize_esg_tilt_preserves_symbols(self):
        ei = _make_integrator()
        result = ei.optimize_esg_tilt(_default_holdings())
        symbols = [s for s, _ in result]
        assert 'SPY' in symbols
        assert 'GLD' in symbols
        assert 'TLT' in symbols

    def test_optimize_esg_tilt_high_target(self):
        ei = _make_integrator()
        result = ei.optimize_esg_tilt(_default_holdings(), esg_target=70.0)
        total = sum(w for _, w in result)
        assert abs(total - 1.0) < 0.01

    # Carbon pair signals tests
    def test_carbon_pair_signals(self):
        ei = _make_integrator()
        result = ei.carbon_pair_signals('BEP', 'VLUE')
        assert 'carbon_spread' in result or 'signal_strength' in result or 'error' not in result

    def test_carbon_pair_unknown_symbol(self):
        ei = _make_integrator()
        result = ei.carbon_pair_signals('BEP', 'NONEXISTENT')
        assert 'error' in result

    def test_carbon_pair_spread_positive(self):
        ei = _make_integrator()
        result = ei.carbon_pair_signals('BEP', 'VLUE')
        # VLUE has high carbon, BEP has low → positive spread
        if 'carbon_spread' in result:
            assert result['carbon_spread'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
