import { describe, expect, it } from 'bun:test';
import { BacktestEngine, type PriceData } from '../../src/backtest/engine';

type MovingAverageHarness = {
  calculateMovingAverage(symbol: string, date: string, periods: number): number | null;
};

function makePriceData(symbol: string, prices: number[]): PriceData[] {
  return prices.map((price, index) => ({
    date: new Date(Date.UTC(2026, 0, 1 + index)).toISOString().slice(0, 10),
    symbol,
    price,
  }));
}

describe('BacktestEngine date and moving-average caches', () => {
  it('uses cached sorted dates for repeated missing-date fallback lookups', () => {
    const engine = new BacktestEngine();
    engine.loadData([
      { date: '2026-01-01', symbol: 'SPY', price: 100 },
      { date: '2026-01-03', symbol: 'SPY', price: 103 },
      { date: '2026-01-05', symbol: 'SPY', price: 105 },
    ]);

    const originalSort = Array.prototype.sort;
    Array.prototype.sort = function failSort() {
      throw new Error('getPrice fallback should use loadData date cache, not per-call sort');
    };

    try {
      expect(engine.getPrice('SPY', '2026-01-02')).toBe(100);
      expect(engine.getPrice('SPY', '2026-01-04')).toBe(103);
      expect(engine.getPrice('SPY', '2026-01-06')).toBe(105);
    } finally {
      Array.prototype.sort = originalSort;
    }
  });

  it('computes moving averages equivalent to the historical window contract', () => {
    const engine = new BacktestEngine();
    engine.loadData(makePriceData('SPY', [100, 102, 104, 106, 108, 110]));

    const ma = (engine as unknown as MovingAverageHarness).calculateMovingAverage(
      'SPY',
      '2026-01-05',
      3,
    );

    expect(ma).toBe((104 + 106 + 108) / 3);
  });

  it('uses cached moving-average inputs without sorting or fallback lookups', () => {
    const engine = new BacktestEngine();
    engine.loadData(makePriceData('SPY', [100, 102, 104, 106, 108, 110]));

    const originalSort = Array.prototype.sort;
    Array.prototype.sort = function failSort() {
      throw new Error('calculateMovingAverage should use loadData caches, not per-call sort');
    };

    try {
      const ma = (engine as unknown as MovingAverageHarness).calculateMovingAverage(
        'SPY',
        '2026-01-06',
        4,
      );
      expect(ma).toBe((104 + 106 + 108 + 110) / 4);
    } finally {
      Array.prototype.sort = originalSort;
    }
  });
});
