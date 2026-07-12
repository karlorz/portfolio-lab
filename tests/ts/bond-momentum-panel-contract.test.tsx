/**
 * Regression: public signals.json.bond_momentum is a bond-duration summary.
 * Risk tab must render recommendation (not "Loading…") for that producer shape.
 */
import { describe, expect, it } from 'bun:test';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import {
  BondMomentumPanel,
  normalizeBondMomentumData,
} from '../../src/components/BondMomentumPanel';
import { BondMomentumSchema } from '../../src/schemas/signals';

/** Shape currently emitted into public/data/signals.json (producer). */
function producerPayload() {
  return {
    active: true,
    yield_10y: 4.5,
    yield_2y: 4.0,
    spread: 0.5,
    curve_regime: 'normal',
    rate_direction: 'stable',
    tlt_weight: 0.2,
    ief_weight: 0.5,
    shy_weight: 0.3,
    effective_duration: 7.3,
    position: 'intermediate',
    confidence: 70.0,
    status_text: 'Bonds: intermediate (normal/stable), dur 7yr',
  };
}

function legacyPayload() {
  return {
    signals: [{
      etf: 'TLT',
      timestamp: '2026-05-26T12:00:00Z',
      signal: 0.5,
      position_size: 0.16,
      formation_return: 0.02,
      realized_vol: 0.12,
      formation_months: 6,
      volatility_target: 0.10,
      confidence: 'moderate' as const,
      action: 'hold' as const,
      weight_delta: 0,
    }],
    timestamp: '2026-05-26T12:00:00Z',
    ensemble: {
      weight: 0.5,
      confidence: 'moderate',
      action: 'hold',
      recommendation: 'Hold current allocation',
    },
  };
}

describe('normalizeBondMomentumData', () => {
  it('maps producer summary keys to summary view-model', () => {
    const view = normalizeBondMomentumData(producerPayload());
    expect(view).not.toBeNull();
    expect(view!.kind).toBe('summary');
    if (view!.kind === 'summary') {
      expect(view.data.active).toBe(true);
      expect(view.data.position).toBe('intermediate');
      expect(view.data.effective_duration).toBeCloseTo(7.3, 1);
      expect(view.data.tlt_weight).toBeCloseTo(0.2, 2);
      expect(view.data.status_text).toContain('intermediate');
    }
  });

  it('preserves legacy overlay signals shape', () => {
    const view = normalizeBondMomentumData(legacyPayload());
    expect(view).not.toBeNull();
    expect(view!.kind).toBe('legacy');
    if (view!.kind === 'legacy') {
      expect(view.data.signals).toHaveLength(1);
      expect(view.data.signals[0].etf).toBe('TLT');
    }
  });

  it('returns null for empty/invalid payloads', () => {
    expect(normalizeBondMomentumData(null)).toBeNull();
    expect(normalizeBondMomentumData({})).toBeNull();
    expect(normalizeBondMomentumData({ foo: 1 })).toBeNull();
  });
});

describe('BondMomentumSchema producer contract', () => {
  it('accepts producer-shaped public artifact payload', () => {
    const result = BondMomentumSchema.safeParse(producerPayload());
    expect(result.success).toBe(true);
  });

  it('still accepts legacy overlay fixture', () => {
    const result = BondMomentumSchema.safeParse(legacyPayload());
    expect(result.success).toBe(true);
  });

  it('fails the stale consumer assumption that signals[] is required', () => {
    // Active summary without signals/timestamp/ensemble must parse (not reject).
    const payload = producerPayload() as Record<string, unknown>;
    expect(payload.signals).toBeUndefined();
    expect(payload.timestamp).toBeUndefined();
    expect(payload.ensemble).toBeUndefined();
    expect(BondMomentumSchema.safeParse(payload).success).toBe(true);
  });
});

describe('BondMomentumPanel producer render', () => {
  it('renders producer payload with recommendation, not Loading', () => {
    const html = renderToStaticMarkup(
      <BondMomentumPanel data={producerPayload()} />,
    );
    expect(html).not.toContain('Loading bond momentum signals');
    expect(html).toContain('intermediate');
    expect(html).toContain('Bonds: intermediate');
    expect(html).toMatch(/7\.3|7yr|duration/i);
  });

  it('renders legacy overlay rows when provided', () => {
    const html = renderToStaticMarkup(
      <BondMomentumPanel data={legacyPayload()} />,
    );
    expect(html).toContain('TLT');
    expect(html).not.toContain('Loading bond momentum signals');
  });

  it('shows unavailable — not Loading — when data is missing', () => {
    const html = renderToStaticMarkup(<BondMomentumPanel data={null} />);
    expect(html).not.toContain('Loading bond momentum signals');
    expect(html.toLowerCase()).toMatch(/no bond|unavailable|not available/);
  });
});
