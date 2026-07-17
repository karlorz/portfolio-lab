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

  it('coerces producer dual-shape numeric criterion values for display', () => {
    const data = normalizeGraduationChecklistData({
      criteria: [
        {
          id: 'min_trading_days',
          name: 'min_trading_days',
          label: 'At least 63 trading days',
          passed: true,
          value: 63,
          required: 63,
          threshold: '63',
          description: 'At least 63 trading days',
        },
      ],
      paper_trading: {
        start_date: '2026-05-01',
        initial_capital: 99918.29,
        current_value: 94208.97,
        days_elapsed: 65,
        days_required: 63,
      },
      readiness_pct: 18.2,
      eligible: false,
      readiness_score: 18.2,
      is_graduation_ready: false,
    });

    expect(data).not.toBeNull();
    expect(data?.criteria[0]?.value).toBe('63');
    expect(data?.criteria[0]?.threshold).toBe('63');
    expect(data?.readiness_pct).toBe(18.2);
  });
});
