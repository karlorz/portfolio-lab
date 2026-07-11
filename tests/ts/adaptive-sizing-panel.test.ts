import { describe, expect, it } from 'bun:test';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import {
  AdaptiveSizingPanel,
  normalizeAdaptiveSizingData,
} from '../../src/components/AdaptiveSizingPanel';

describe('AdaptiveSizingPanel data normalization', () => {
  it('normalizes allocation-shaped adaptive sizing artifacts into renderable rows', () => {
    const data = normalizeAdaptiveSizingData({
      base_allocation: { SPY: 0.46, GLD: 0.38, TLT: 0.16 },
      adjusted_allocation: { SPY: 0.44, GLD: 0.39, TLT: 0.17 },
      adjustments: { SPY: -0.02, GLD: 0.01, TLT: 0.01 },
      factors: { regime: 'normal', regime_confidence: 0.8 },
      generated_at: '2026-06-11T10:15:18',
    });

    expect(data?.constraints.viability_floor).toBe(0.5);
    expect(data?.current_regime).toBe('NORMAL');
    expect(data?.total_weight).toBeCloseTo(1.0);
    expect(data?.signals).toEqual([
      {
        name: 'SPY',
        current_weight: 0.46,
        target_weight: 0.44,
        health_score: 0.8,
        regime_adjusted: {},
      },
      {
        name: 'GLD',
        current_weight: 0.38,
        target_weight: 0.39,
        health_score: 0.8,
        regime_adjusted: {},
      },
      {
        name: 'TLT',
        current_weight: 0.16,
        target_weight: 0.17,
        health_score: 0.8,
        regime_adjusted: {},
      },
    ]);
  });

  it('returns null for unusable partial fallback payloads', () => {
    expect(normalizeAdaptiveSizingData({ status: 'error' })).toBeNull();
    expect(normalizeAdaptiveSizingData({ adjusted_allocation: {} })).toBeNull();
  });

  it('renders advisory non-routed authority disclosure when provided', () => {
    const html = renderToStaticMarkup(React.createElement(AdaptiveSizingPanel, {
      data: {
        base_allocation: { SPY: 0.46, GLD: 0.38, TLT: 0.16 },
        adjusted_allocation: { SPY: 0.44, GLD: 0.40, TLT: 0.16 },
        factors: { regime: 'normal', regime_confidence: 0.8 },
        authority: {
          schema_version: 'allocation-artifact-role/v1',
          surface: 'adaptive_sizing',
          allocation_field: 'adjusted_allocation',
          runtime_role: 'advisory_non_routed',
          live_authoritative: false,
          routed: false,
          routed_by: null,
          canonical_controller: 'signals.json.target_allocations',
          routed_surface: 'target_allocations',
          routed_surface_path: 'public/data/signals.json#target_allocations',
          description: 'adaptive_sizing is advisory; live order routing continues to consume signals.json.target_allocations.',
        },
        generated_at: '2026-07-06T00:00:00Z',
      },
    }));

    expect(html).toContain('Not order-routed');
    expect(html).toContain('target_allocations');
  });
});
