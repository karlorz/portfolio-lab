/**
 * Regression: producer-shaped signals.json.vix_term_structure must render
 * real VIX levels/signals, not N/A zeros, after panel/schema adaptation.
 */
import { describe, expect, it } from 'bun:test';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import {
  VIXTermStructurePanel,
  normalizeVixTermStructureData,
} from '../../src/components/VIXTermStructurePanel';
import { VIXTermStructureSchema } from '../../src/schemas/signals';

/** Shape currently emitted into public/data/signals.json (producer). */
function producerPayload() {
  return {
    timestamp: '2026-07-12T03:57:38.829941',
    signal_state: 'RISK_ON',
    signal_value: 0.5082544916693613,
    vix_spot: 16.760000228881836,
    vix3m: 20.030000686645508,
    vix6m: null,
    slope_vix3m_vix: 1.1951074232164156,
    regime: 'extreme_contango',
    regime_strength: 0.8007161547761047,
    slope_signal: 0.6503580773880524,
    roll_yield_signal: 0.8162756729069564,
    vix_zscore_signal: 0.22021171243700605,
    curve_shape_signal: 0.0,
    spy_shift: 0.02,
    gld_shift: -0.01,
    tlt_shift: -0.01,
    confidence: 90.0,
    is_valid: true,
    reason: 'VIX=16.76, Slope=1.195, Regime=extreme_contango',
  };
}

function nestedLegacyPayload() {
  return {
    vix: { value: 15.2, timestamp: '2026-05-26T12:00:00Z' },
    vix3m: { value: 17.5, timestamp: '2026-05-26T12:00:00Z' },
    slope: 0.87,
    roll_yield: 0.15,
    composite_signal: 0.35,
    regime: 'mild_contango' as const,
    z_score: 0.5,
  };
}

describe('normalizeVixTermStructureData', () => {
  it('maps producer keys to panel view-model fields', () => {
    const view = normalizeVixTermStructureData(producerPayload());
    expect(view).not.toBeNull();
    expect(view!.vix?.value).toBeCloseTo(16.76, 2);
    expect(view!.vix3m?.value).toBeCloseTo(20.03, 2);
    expect(view!.slope).toBeCloseTo(1.195, 3);
    expect(view!.roll_yield).toBeCloseTo(0.816, 3);
    expect(view!.composite_signal).toBeCloseTo(0.508, 3);
    expect(view!.z_score).toBeCloseTo(0.220, 3);
    expect(view!.regime).toBe('extreme_contango');
  });

  it('preserves nested legacy shape', () => {
    const view = normalizeVixTermStructureData(nestedLegacyPayload());
    expect(view).not.toBeNull();
    expect(view!.vix?.value).toBe(15.2);
    expect(view!.vix3m?.value).toBe(17.5);
    expect(view!.slope).toBe(0.87);
    expect(view!.composite_signal).toBe(0.35);
  });

  it('returns null for empty/invalid payloads', () => {
    expect(normalizeVixTermStructureData(null)).toBeNull();
    expect(normalizeVixTermStructureData({})).toBeNull();
    expect(normalizeVixTermStructureData({ foo: 1 })).toBeNull();
  });
});

describe('VIXTermStructureSchema producer contract', () => {
  it('accepts producer-shaped public artifact payload', () => {
    const result = VIXTermStructureSchema.safeParse(producerPayload());
    expect(result.success).toBe(true);
    if (result.success) {
      // After schema normalize/parse, nested view fields should be present.
      const data = result.data as Record<string, unknown>;
      const vix = data.vix as { value?: number } | undefined;
      expect(vix?.value).toBeCloseTo(16.76, 2);
    }
  });

  it('still accepts nested legacy fixture', () => {
    const result = VIXTermStructureSchema.safeParse(nestedLegacyPayload());
    expect(result.success).toBe(true);
  });
});

describe('VIXTermStructurePanel producer render', () => {
  it('renders producer payload without N/A VIX Spot or zeroed metrics', () => {
    const html = renderToStaticMarkup(
      React.createElement(VIXTermStructurePanel, { data: producerPayload() as never }),
    );

    expect(html).toContain('VIX Spot');
    // Spot must not be N/A (VIX6M may still be N/A when producer emits null).
    expect(html).toMatch(/VIX Spot<\/label><span class="value">16\.76<\/span>/);
    expect(html).toMatch(/VIX3M<\/label><span class="value">20\.03<\/span>/);
    // Spot / 3M levels from producer
    expect(html).toContain('16.76');
    expect(html).toContain('20.03');
    // Slope and composite signal non-zero
    expect(html).toContain('1.195');
    expect(html).toContain('0.508');
    expect(html).toContain('Extreme Contango');
  });

  it('still renders nested legacy fixture', () => {
    const html = renderToStaticMarkup(
      React.createElement(VIXTermStructurePanel, { data: nestedLegacyPayload() as never }),
    );
    expect(html).toContain('15.20');
    expect(html).toContain('17.50');
    expect(html).toContain('0.870');
    expect(html).toContain('Mild Contango');
  });
});
