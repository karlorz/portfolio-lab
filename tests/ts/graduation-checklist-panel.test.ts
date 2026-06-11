import { describe, expect, it } from 'bun:test';
import {
  normalizeGraduationChecklistData,
} from '../../src/components/GraduationChecklistPanel';

describe('GraduationChecklistPanel data normalization', () => {
  it('rejects incomplete fallback payloads before render-time field access', () => {
    expect(normalizeGraduationChecklistData({ status: 'error', message: 'schema mismatch' })).toBeNull();
    expect(normalizeGraduationChecklistData({ criteria: [], readiness_pct: 0, eligible: false })).toBeNull();
  });

  it('accepts complete graduation checklist payloads', () => {
    const data = normalizeGraduationChecklistData({
      criteria: [{ id: 'sharpe', label: 'Sharpe', passed: true, value: '1.0', threshold: '0.7' }],
      paper_trading: {
        start_date: '2026-01-01',
        initial_capital: 100000,
        current_value: 101000,
        days_elapsed: 20,
        days_required: 30,
      },
      readiness_pct: 50,
      eligible: false,
    });

    expect(data?.paper_trading.current_value).toBe(101000);
    expect(data?.criteria).toHaveLength(1);
  });
});
