import { describe, expect, it } from 'bun:test';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import {
  EnsembleVotingPanel,
  formatSourceDirection,
  normalizeSourceVote,
} from '../../src/components/EnsembleVotingPanel';

describe('EnsembleVotingPanel source vote normalization', () => {
  it('maps string directions from signals.json into numeric panel values', () => {
    expect(normalizeSourceVote({
      source: 'MULTI_SPEED_MOM',
      direction: 'bullish',
      strength: 0.4,
      confidence: 0.7,
      weight: 0.12,
    })).toEqual({
      source: 'MULTI_SPEED_MOM',
      direction: 1,
      strength: 0.4,
      confidence: 0.7,
      weight: 0.12,
    });

    expect(formatSourceDirection('bearish')).toBe('-1');
    expect(formatSourceDirection('neutral')).toBe('0');
  });

  it('uses safe numeric fallbacks for malformed source vote fields', () => {
    expect(normalizeSourceVote({
      source: '',
      direction: 'not-a-direction',
      strength: 'bad',
      confidence: null,
      weight: undefined,
    })).toEqual({
      source: 'unknown',
      direction: 0,
      strength: 0,
      confidence: 0,
      weight: 0,
    });
  });

  it('renders adaptive learning branch statuses separately from final weights', () => {
    const html = renderToStaticMarkup(React.createElement(EnsembleVotingPanel, {
      data: {
        regime: 'normal',
        regime_confidence: 0.63,
        weighted_consensus: 0,
        agreement_ratio: 1,
        action: 'neutral',
        confidence: 0.5,
        equity_bias: 0.1,
        duration_bias: -0.1,
        gold_bias: 0.05,
        num_sources: 1,
        adaptive_learning: {
          bandit: {
            status: 'non_effective',
            enabled: true,
            reason: 'cold_start_no_regime_weights',
          },
          online_ic: {
            status: 'disabled',
            enabled: false,
            reason: 'env_disabled',
          },
        },
        source_breakdown: [{
          source: 'alternative_data',
          direction: 'bullish',
          strength: 0.4,
          confidence: 0.6,
          weight: 0.2,
        }],
      },
    }));

    expect(html).toContain('Adaptive Learning');
    expect(html).toContain('Bandit');
    expect(html).toContain('Non-effective');
    expect(html).toContain('Online IC');
    expect(html).toContain('Disabled');
  });

  it('renders collected and contributing source counts without a lone ambiguous sources metric', () => {
    const html = renderToStaticMarkup(React.createElement(EnsembleVotingPanel, {
      data: {
        regime: 'normal',
        regime_confidence: 0.63,
        weighted_consensus: 0,
        agreement_ratio: 1,
        action: 'neutral',
        confidence: 0.5,
        equity_bias: 0.1,
        duration_bias: -0.1,
        gold_bias: 0.05,
        num_sources: 4,
        configured_source_count: 9,
        collected_source_count: 4,
        contributing_source_count: 2,
        inactive_source_count: 2,
        inactive_sources: ['cross_asset_rv', 'multi_speed_momentum'],
        source_breakdown: [{
          source: 'alternative_data',
          direction: 'bullish',
          strength: 0.4,
          confidence: 0.6,
          weight: 0.2,
        }],
      },
    }));

    expect(html).toContain('Configured Sources');
    expect(html).toContain('Collected Sources');
    expect(html).toContain('Contributing Sources');
    expect(html).toContain('Inactive/Zero Weight');
    expect(html).not.toContain('>Sources</span>');
  });

  it('renders configured-but-stale Google Trends disclosure', () => {
    const html = renderToStaticMarkup(React.createElement(EnsembleVotingPanel, {
      data: {
        regime: 'normal',
        regime_confidence: 0.63,
        weighted_consensus: 0,
        agreement_ratio: 1,
        action: 'neutral',
        confidence: 0.5,
        equity_bias: 0.1,
        duration_bias: -0.1,
        gold_bias: 0.05,
        num_sources: 1,
        configured_source_status: [{
          source: 'google_trends',
          label: 'Google Trends',
          configured: true,
          collected: false,
          active: false,
          contributing: false,
          status: 'stale',
          reason: 'Data is 37 days old (max 14)',
          configured_weight: 0.04762,
        }],
        source_breakdown: [{
          source: 'alternative_data',
          direction: 'bullish',
          strength: 0.4,
          confidence: 0.6,
          weight: 0.2,
        }],
      },
    }));

    expect(html).toContain('Configured Source Status');
    expect(html).toContain('Google Trends');
    expect(html).toContain('Stale');
    expect(html).toContain('Data is 37 days old (max 14)');
  });

  it('renders the current non-routed live role when provided', () => {
    const html = renderToStaticMarkup(React.createElement(EnsembleVotingPanel, {
      allocationSurfaceRole: {
        label: 'Ensemble Voting',
        role: 'advisory_non_routed',
        routed: false,
        routed_by: null,
        description: 'Published for diagnostics; current order routing uses target_allocations.',
      },
      data: {
        regime: 'normal',
        regime_confidence: 0.63,
        weighted_consensus: 0,
        agreement_ratio: 1,
        action: 'neutral',
        confidence: 0.5,
        equity_bias: 0.1,
        duration_bias: -0.1,
        gold_bias: 0.05,
        num_sources: 1,
        source_breakdown: [],
      },
    }));

    expect(html).toContain('Live Role');
    expect(html).toContain('Not order-routed');
    expect(html).toContain('target_allocations');
  });
});
