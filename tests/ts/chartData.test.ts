import { describe, it, expect } from 'bun:test';
import { mergeRollingMetrics } from '../../src/utils/chartData';

describe('mergeRollingMetrics', () => {
  it('merges rolling metrics by date with one indexed lookup per series row', () => {
    const sharpe63d = [
      { date: '2026-01-02', sharpe: 0.4, volatility: 12.0, mean_return: 0.001, window_days: 63 },
      { date: '2026-01-03', sharpe: 0.5, volatility: 11.5, mean_return: 0.002, window_days: 63 },
    ];
    const sharpe126d = [
      { date: '2026-01-03', sharpe: 0.7, volatility: 10.5, mean_return: 0.002, window_days: 126 },
    ];
    const sharpe252d = [
      { date: '2026-01-01', sharpe: 0.9, volatility: 9.5, mean_return: 0.003, window_days: 252 },
    ];

    const rows = mergeRollingMetrics(sharpe63d, sharpe126d, sharpe252d);

    expect(rows).toEqual([
      {
        date: '2026-01-01',
        dateFormatted: '1/1',
        sharpe63: null,
        sharpe126: null,
        sharpe252: 0.9,
        vol63: null,
      },
      {
        date: '2026-01-02',
        dateFormatted: '1/2',
        sharpe63: 0.4,
        sharpe126: null,
        sharpe252: null,
        vol63: 12.0,
      },
      {
        date: '2026-01-03',
        dateFormatted: '1/3',
        sharpe63: 0.5,
        sharpe126: 0.7,
        sharpe252: null,
        vol63: 11.5,
      },
    ]);
  });

  it('does not call Array.find while merging large rolling metric series', () => {
    const originalFind = Array.prototype.find;
    Array.prototype.find = function failFind() {
      throw new Error('mergeRollingMetrics should use date maps, not repeated find scans');
    };
    try {
      const series = Array.from({ length: 1500 }, (_, i) => ({
        date: new Date(Date.UTC(2026, 0, 1 + i)).toISOString().slice(0, 10),
        sharpe: i / 100,
        volatility: i / 10,
        mean_return: 0,
        window_days: 63,
      }));

      const rows = mergeRollingMetrics(series, series, series);

      expect(rows.length).toBe(1500);
      expect(rows[0].sharpe63).toBe(0);
      expect(rows[1499].sharpe252).toBe(14.99);
    } finally {
      Array.prototype.find = originalFind;
    }
  });
});
