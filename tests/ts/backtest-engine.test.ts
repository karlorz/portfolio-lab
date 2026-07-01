import { describe, expect, it } from 'bun:test';
import { readFileSync } from 'fs';
import { BacktestEngine, type BacktestResult, type PortfolioConfig, type PriceData } from '../../src/backtest/engine';
import { fetchCompactPriceData, symbolsForPortfolios, toBacktestData } from '../../src/backtest/price-loader';

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

  it('computes max drawdown as minimum of drawdown series', () => {
    const engine = new BacktestEngine();
    const result = makeBacktestResult(
      ['2026-01-02', '2026-01-05', '2026-01-06'],
      [100, 80, 90],
    );

    const metrics = engine.calculateMetrics(result);
    expect(metrics.maxDrawdown).toBeCloseTo(-0.2, 8);
    expect(metrics.cagr).toBeGreaterThan(-1);
    expect(Number.isFinite(metrics.volatility)).toBe(true);
  });
});

describe('App backtest price loader', () => {
  const defaultComparisonPortfolios: PortfolioConfig[] = [
    {
      name: 'SPY (S&P 500)',
      allocation: { SPY: 1 },
      rebalanceFrequency: 'none',
    },
    {
      name: 'SPY/GLD/TLT 46/38/16 ★★',
      allocation: { SPY: 0.46, GLD: 0.38, TLT: 0.16 },
      rebalanceFrequency: 'annual',
    },
    {
      name: 'SPY/GLD 55/45',
      allocation: { SPY: 0.55, GLD: 0.45 },
      rebalanceFrequency: 'annual',
    },
  ];

  const compactPrices = {
    SPY: [
      { d: '2026-01-02', p: 100 },
      { d: '2026-01-05', p: 102 },
      { d: '2026-01-06', p: 101 },
      { d: '2026-01-07', p: 104 },
    ],
    GLD: [
      { d: '2026-01-02', p: 200 },
      { d: '2026-01-05', p: 201 },
      { d: '2026-01-06', p: 203 },
      { d: '2026-01-07', p: 204 },
    ],
    TLT: [
      { d: '2026-01-02', p: 90 },
      { d: '2026-01-05', p: 91 },
      { d: '2026-01-06', p: 92 },
      { d: '2026-01-07', p: 90 },
    ],
    QQQ: [
      { d: '2026-01-02', p: 300 },
      { d: '2026-01-05', p: 305 },
      { d: '2026-01-06', p: 304 },
      { d: '2026-01-07', p: 308 },
    ],
  };

  function runMetrics(portfolio: PortfolioConfig, priceData: PriceData[]) {
    const engine = new BacktestEngine();
    engine.loadData(priceData);
    const result = engine.runBacktest(portfolio, '2026-01-02', '2026-01-07', 10000);
    return engine.calculateMetrics(result);
  }

  it('flattens only symbols required by the selected portfolio set', () => {
    const requiredSymbols = symbolsForPortfolios(defaultComparisonPortfolios);
    const filtered = toBacktestData(compactPrices, requiredSymbols);

    expect(requiredSymbols).toEqual(['GLD', 'SPY', 'TLT']);
    expect(new Set(filtered.map(row => row.symbol))).toEqual(new Set(['SPY', 'GLD', 'TLT']));
    expect(filtered.some(row => row.symbol === 'QQQ')).toBe(false);
    expect(filtered).toHaveLength(12);
  });

  it('does not iterate unrelated symbol rows while flattening selected portfolio data', () => {
    const qqqRows = new Proxy(compactPrices.QQQ, {
      get(target, prop, receiver) {
        if (prop === Symbol.iterator || prop === 'length' || prop === '0') {
          throw new Error('unrelated QQQ rows should not be flattened');
        }
        return Reflect.get(target, prop, receiver);
      },
    });
    const pricesWithExplodingUnrelatedRows = {
      ...compactPrices,
      QQQ: qqqRows,
    };

    const filtered = toBacktestData(
      pricesWithExplodingUnrelatedRows,
      symbolsForPortfolios(defaultComparisonPortfolios),
    );

    expect(new Set(filtered.map(row => row.symbol))).toEqual(new Set(['SPY', 'GLD', 'TLT']));
  });

  it('preserves default comparison metrics when unrelated symbols are skipped', () => {
    const fullPriceData = toBacktestData(compactPrices);
    const filteredPriceData = toBacktestData(
      compactPrices,
      symbolsForPortfolios(defaultComparisonPortfolios),
    );

    for (const portfolio of defaultComparisonPortfolios) {
      const fullMetrics = runMetrics(portfolio, fullPriceData);
      const filteredMetrics = runMetrics(portfolio, filteredPriceData);

      expect(filteredMetrics.cagr).toBeCloseTo(fullMetrics.cagr, 12);
      expect(filteredMetrics.volatility).toBeCloseTo(fullMetrics.volatility, 12);
      expect(filteredMetrics.sharpeRatio).toBeCloseTo(fullMetrics.sharpeRatio, 12);
      expect(filteredMetrics.maxDrawdown).toBeCloseTo(fullMetrics.maxDrawdown, 12);
      expect(filteredMetrics.totalReturn).toBeCloseTo(fullMetrics.totalReturn, 12);
    }
  });

  it('prefers compact public price payloads and falls back to the legacy prices endpoint', async () => {
    const compactOnlyRequests: string[] = [];
    const compactOnly = await fetchCompactPriceData(async (url) => {
      compactOnlyRequests.push(url);
      return new Response(JSON.stringify(compactPrices), { status: 200 });
    });

    expect(compactOnlyRequests).toEqual(['/data/prices_compact.json']);
    expect(compactOnly).toEqual(compactPrices);

    const fallbackRequests: string[] = [];
    const fallback = await fetchCompactPriceData(async (url) => {
      fallbackRequests.push(url);
      if (url === '/data/prices_compact.json') {
        return new Response('', { status: 404 });
      }
      return new Response(JSON.stringify(compactPrices), { status: 200 });
    });

    expect(fallbackRequests).toEqual(['/data/prices_compact.json', '/data/prices.json']);
    expect(fallback).toEqual(compactPrices);
  });

  it('preserves comparison metrics when prices are loaded from the compact endpoint path', async () => {
    const fetchedPrices = await fetchCompactPriceData(async () => (
      new Response(JSON.stringify(compactPrices), { status: 200 })
    ));
    const directPriceData = toBacktestData(compactPrices, symbolsForPortfolios(defaultComparisonPortfolios));
    const fetchedPriceData = toBacktestData(fetchedPrices, symbolsForPortfolios(defaultComparisonPortfolios));

    for (const portfolio of defaultComparisonPortfolios) {
      const directMetrics = runMetrics(portfolio, directPriceData);
      const fetchedMetrics = runMetrics(portfolio, fetchedPriceData);

      expect(fetchedMetrics.cagr).toBeCloseTo(directMetrics.cagr, 12);
      expect(fetchedMetrics.volatility).toBeCloseTo(directMetrics.volatility, 12);
      expect(fetchedMetrics.sharpeRatio).toBeCloseTo(directMetrics.sharpeRatio, 12);
      expect(fetchedMetrics.maxDrawdown).toBeCloseTo(directMetrics.maxDrawdown, 12);
      expect(fetchedMetrics.totalReturn).toBeCloseTo(directMetrics.totalReturn, 12);
    }
  });

  it('keeps the legacy prices file and writes the compact mirror in the fetcher CLI', () => {
    const fetcherSource = readFileSync('src/data/fetchYahoo.ts', 'utf8');

    expect(fetcherSource).toContain('../../public/data/prices.json');
    expect(fetcherSource).toContain('../../public/data/prices_compact.json');
  });
});
