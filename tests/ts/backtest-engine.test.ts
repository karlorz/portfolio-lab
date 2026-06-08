import { describe, expect, it } from 'bun:test';
import { BacktestEngine, type BacktestResult, type PriceData } from '../../src/backtest/engine';

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

function makeBacktestResult(dates: string[], portfolioValues: number[]): BacktestResult {
  const returns = portfolioValues.map((value, index) => (
    index === 0 ? 0 : (value - portfolioValues[index - 1]) / portfolioValues[index - 1]
  ));

  let peak = portfolioValues[0];
  const drawdowns = portfolioValues.map((value) => {
    peak = Math.max(peak, value);
    return (value - peak) / peak;
  });

  return {
    dates,
    portfolioValues,
    returns,
    drawdowns,
    holdings: [],
    trades: [],
  };
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

describe('BacktestEngine metric semantics', () => {
  it('uses the canonical project risk-free rate as the default for Sharpe and Sortino', () => {
    const engine = new BacktestEngine();
    const result = makeBacktestResult(
      ['2026-01-02', '2026-01-05', '2026-01-06', '2026-01-07', '2026-01-08'],
      [100, 102, 101, 104, 103],
    );

    const defaultMetrics = engine.calculateMetrics(result);
    const canonicalMetrics = engine.calculateMetrics(result, 0.045);
    const lowRiskFreeMetrics = engine.calculateMetrics(result, 0.02);

    expect(defaultMetrics.sharpeRatio).toBeCloseTo(canonicalMetrics.sharpeRatio, 12);
    expect(defaultMetrics.sortinoRatio).toBeCloseTo(canonicalMetrics.sortinoRatio, 12);
    expect(lowRiskFreeMetrics.sharpeRatio).toBeGreaterThan(defaultMetrics.sharpeRatio);
    expect(lowRiskFreeMetrics.sortinoRatio).toBeGreaterThan(defaultMetrics.sortinoRatio);
  });

  it('honors the percent-style RISK_FREE_RATE environment override for default metrics', () => {
    const engine = new BacktestEngine();
    const result = makeBacktestResult(
      ['2026-01-02', '2026-01-05', '2026-01-06', '2026-01-07', '2026-01-08'],
      [100, 102, 101, 104, 103],
    );
    const previousRiskFreeRate = process.env.RISK_FREE_RATE;

    try {
      process.env.RISK_FREE_RATE = '6.0';
      const defaultMetrics = engine.calculateMetrics(result);
      const explicitOverrideMetrics = engine.calculateMetrics(result, 0.06);

      expect(defaultMetrics.sharpeRatio).toBeCloseTo(explicitOverrideMetrics.sharpeRatio, 12);
      expect(defaultMetrics.sortinoRatio).toBeCloseTo(explicitOverrideMetrics.sortinoRatio, 12);
    } finally {
      if (previousRiskFreeRate === undefined) {
        delete process.env.RISK_FREE_RATE;
      } else {
        process.env.RISK_FREE_RATE = previousRiskFreeRate;
      }
    }
  });

  it('counts positiveMonths from aggregated monthly returns instead of positive daily returns', () => {
    const engine = new BacktestEngine();
    const result = makeBacktestResult(
      [
        '2026-01-02',
        '2026-01-05',
        '2026-01-06',
        '2026-01-30',
        '2026-02-02',
        '2026-02-27',
        '2026-03-02',
        '2026-03-31',
      ],
      [100, 125, 120, 95, 95, 110, 110, 100],
    );

    expect(engine.calculateMetrics(result).positiveMonths).toBe(1);
  });
});
