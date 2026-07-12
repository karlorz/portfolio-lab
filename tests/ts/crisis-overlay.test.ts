import { describe, expect, it } from 'bun:test';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import {
  CrisisOverlay,
  resolveCrisisPeriodsStatus,
} from '../../src/components/AnalyticsCharts';
import { AnalyticsDataSchema } from '../../src/schemas/signals';

describe('CrisisOverlay availability', () => {
  const nullPeriods = [
    {
      name: 'GFC 2008',
      period: '2008-09-01 to 2009-03-31',
      description: 'Global Financial Crisis',
      spy_return: -47,
      portfolio_return: null,
    },
  ];

  it('infers unavailable when all portfolio returns are null', () => {
    expect(resolveCrisisPeriodsStatus(nullPeriods)).toBe('unavailable');
  });

  it('renders degraded banner for all-null portfolio crisis returns', () => {
    const html = renderToStaticMarkup(
      React.createElement(CrisisOverlay, {
        periods: nullPeriods,
        crisisPeriodsStatus: 'unavailable',
        crisisPeriodsReason: 'historical_simulation_unavailable',
      }),
    );
    expect(html).toContain('crisis-overlay--degraded');
    expect(html).toContain('data-crisis-status="unavailable"');
    expect(html).toContain('Portfolio crisis returns unavailable');
    expect(html).toContain('historical_simulation_unavailable');
    expect(html).toContain('Unavailable');
  });

  it('accepts analytics payload with crisis section metadata', () => {
    const result = AnalyticsDataSchema.safeParse({
      status: 'success',
      generated_at: '2026-07-13T00:00:00Z',
      data_points: 10,
      date_range: { start: '2026-01-01', end: '2026-07-01' },
      drawdown: {
        series: [],
        max_drawdown: {
          max_drawdown: -5,
          max_drawdown_date: '2026-03-01',
          recovery_date: null,
          underwater_days: 10,
          peak_value: 100,
          trough_value: 95,
        },
      },
      rolling_metrics: { sharpe_63d: [], sharpe_126d: [], sharpe_252d: [] },
      benchmark_comparison: {
        portfolio: {
          start_date: '2026-01-01',
          end_date: '2026-07-01',
          start_value: 100,
          end_value: 110,
          total_return: 0.1,
          cagr: 0.2,
          volatility: 0.1,
          max_drawdown: -5,
          sharpe: 1,
        },
      },
      crisis_periods: nullPeriods,
      crisis_periods_status: 'unavailable',
      crisis_periods_reason: 'historical_simulation_unavailable',
    });
    expect(result.success).toBe(true);
  });
});
