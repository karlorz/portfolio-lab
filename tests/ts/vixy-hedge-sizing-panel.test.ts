import { describe, expect, it } from 'bun:test';
import {
  normalizeVixyHedgeSizingData,
} from '../../src/components/VixyHedgeSizingPanel';

describe('VixyHedgeSizingPanel data normalization', () => {
  it('normalizes status-shaped hedge artifacts into the panel contract', () => {
    const data = normalizeVixyHedgeSizingData({
      current_allocation_pct: 3,
      target_allocation_pct: 2.5,
      vix_level: 25,
      regime: 'elevated',
      ytd_cost_bps: 195.6,
      ytd_benefit_bps: 0,
      hedge_efficiency: 0.11,
      total_signals: 6,
      generated_at: '2026-06-11T10:15:18',
    });

    expect(data?.vixy_position).toBe(0.03);
    expect(data?.recommendation.allocation).toBe(0.025);
    expect(data?.vix_zone).toBe('ELEVATED');
    expect(data?.vix_level).toBe(25);
    expect(data?.costs.total_pct).toBeCloseTo(0.01956);
    expect(data?.crisis_performance).toEqual([]);
  });

  it('returns null for unusable partial fallback payloads', () => {
    expect(normalizeVixyHedgeSizingData({ status: 'error' })).toBeNull();
    expect(normalizeVixyHedgeSizingData({ vix_level: 25 })).toBeNull();
  });
});
