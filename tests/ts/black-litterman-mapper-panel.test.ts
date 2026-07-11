import { describe, expect, it } from 'bun:test';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { BlackLittermanMapperPanel } from '../../src/components/BlackLittermanMapperPanel';

describe('BlackLittermanMapperPanel', () => {
  it('renders advisory non-routed authority disclosure when provided', () => {
    const html = renderToStaticMarkup(React.createElement(BlackLittermanMapperPanel, {
      data: {
        prior_weights: { SPY: 0.46, GLD: 0.38, TLT: 0.16 },
        posterior_weights: { SPY: 0.44, GLD: 0.40, TLT: 0.16 },
        views: [{
          signal_name: 'ensemble_consensus',
          asset: 'SPY',
          direction: 'bullish',
          confidence: 0.7,
          expected_return_delta: 0.01,
        }],
        tau: 0.15,
        view_confidence_method: 'idzorek',
        authority: {
          schema_version: 'allocation-artifact-role/v1',
          surface: 'black_litterman',
          allocation_field: 'posterior_weights',
          runtime_role: 'advisory_non_routed',
          live_authoritative: false,
          routed: false,
          routed_by: null,
          canonical_controller: 'signals.json.target_allocations',
          routed_surface: 'target_allocations',
          routed_surface_path: 'public/data/signals.json#target_allocations',
          description: 'black_litterman is advisory; live order routing continues to consume signals.json.target_allocations.',
        },
      },
    }));

    expect(html).toContain('Black-Litterman Mapper');
    expect(html).toContain('Not order-routed');
    expect(html).toContain('target_allocations');
  });
});
