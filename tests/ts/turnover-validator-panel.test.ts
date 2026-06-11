import { describe, expect, it } from 'bun:test';
import {
  normalizeTurnoverValidatorData,
} from '../../src/components/TurnoverValidatorPanel';

describe('TurnoverValidatorPanel data normalization', () => {
  it('rejects signal-stability artifacts that do not match turnover budgets', () => {
    expect(normalizeTurnoverValidatorData({
      stable: { periods: 20, mean: 0.5, stability_score: 1 },
      generated_at: '2026-06-11T10:15:18',
    })).toBeNull();
  });

  it('accepts complete turnover budget payloads', () => {
    const data = normalizeTurnoverValidatorData({
      current_turnover_pct: 0.5,
      max_daily_turnover: 5,
      max_monthly_turnover: 20,
      max_annual_turnover: 100,
      daily_budget_used: 0.1,
      monthly_budget_used: 0.2,
      annual_budget_used: 0.3,
      recent_rebalances: [
        { date: '2026-06-01', turnover_pct: 0.5, cost_bps: 1.2, trigger: 'drift' },
      ],
      cost_drag_bps: 4.5,
    });

    expect(data?.current_turnover_pct).toBe(0.5);
    expect(data?.recent_rebalances[0].trigger).toBe('drift');
  });
});
