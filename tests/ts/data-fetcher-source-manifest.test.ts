import { describe, expect, it } from 'bun:test';
import { mkdtempSync, readFileSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import {
  createYahooFinanceProvider,
  convertToBacktestFormat,
  fetchAllDataWithSummary,
  fetchFredSeriesWithSummary,
  fetchYahooV8,
  fetchYieldCurveDataWithSummary,
  generateFallbackYields,
  type HistoricalPrice,
  type MarketDataProvider,
  type MarketDataProviderSummary,
} from '../../src/data/fetcher';
import {
  DashboardGenerationError,
  buildLastGoodRetentionManifest,
  buildPriceSourceRows,
  buildYieldSourceRow,
  createFredDiskCache,
  runDashboardGeneration,
  shouldWriteLastGoodRetentionManifest,
} from '../../scripts/fetch-data';
import { SYMBOL_UNIVERSE_METADATA } from '../../src/data/symbol_universe';
import { buildMarketDataSourceManifest } from '../../src/data/source_manifest';
import {
  PRICE_DATA_QUALITY_FILENAME,
  PRICE_DATA_QUALITY_SCHEMA_VERSION,
  buildPriceDataQualityReport,
} from '../../src/data/price_quality';
import {
  cleanCompactPrices,
  extremeReturnCompactPrices,
  internalGapCompactPrices,
  splitLikeReturnCompactPrices,
  staleLatestCompactPrices,
} from './compact-price-fixtures';

function yahooPayload(close = 100) {
  return {
    chart: {
      result: [
        {
          timestamp: [1_704_067_200],
          indicators: {
            quote: [
              {
                open: [close - 1],
                high: [close + 1],
                low: [close - 2],
                close: [close],
                volume: [123],
              },
            ],
            adjclose: [{ adjclose: [close] }],
          },
        },
      ],
    },
  };
}

function fredPayload(value: string, date = '2024-01-01') {
  return {
    observations: [
      {
        realtime_start: date,
        realtime_end: date,
        date,
        value,
      },
    ],
  };
}

function memoryFredCache(records: Array<{
  series_id: string;
  start_date: string;
  end_date: string;
  fetched_at: string;
  observations: { date: string; value: number }[];
}> = []) {
  const byKey = new Map(records.map((record) => [
    `${record.series_id}:${record.start_date}:${record.end_date}`,
    record,
  ]));
  return {
    get: async ({ seriesId, startDate, endDate }: { seriesId: string; startDate: string; endDate: string }) =>
      byKey.get(`${seriesId}:${startDate}:${endDate}`) ?? null,
    set: async (
      { seriesId, startDate, endDate }: { seriesId: string; startDate: string; endDate: string },
      record: { fetched_at: string; observations: { date: string; value: number }[] },
    ) => {
      byKey.set(`${seriesId}:${startDate}:${endDate}`, {
        series_id: seriesId,
        start_date: startDate,
        end_date: endDate,
        ...record,
      });
    },
    records: byKey,
  };
}

describe('market data fetcher source provenance', () => {
  const licensedRows: HistoricalPrice[] = [
    {
      date: '2024-01-01',
      open: 100,
      high: 102,
      low: 99,
      close: 101,
      adjClose: 101,
      volume: 1_000,
    },
  ];

  function stubProvider(
    name: string,
    fetchSymbol: MarketDataProvider['fetchSymbol'],
  ): MarketDataProvider {
    return {
      name,
      feed: 'adjusted-eod-fixture',
      sourceMode: 'live',
      fetchSymbol,
    };
  }

  function priceProviderSummary(overrides: Partial<MarketDataProviderSummary> = {}): MarketDataProviderSummary {
    return {
      provider: 'Yahoo Finance',
      feed: 'chart/v8',
      provider_chain: ['Yahoo Finance'],
      primary_provider: 'Yahoo Finance',
      fallback_provider: null,
      status: 'success',
      source_mode: 'live',
      fetched_at: '2026-06-12T00:00:00Z',
      symbols: [],
      failure_counts: {},
      circuit_breaker: {
        opened: false,
        reason: null,
        skipped_symbols: [],
      },
      ...overrides,
    };
  }

  it('retries Yahoo 429 responses and returns eventual success', async () => {
    let calls = 0;
    const fetchImpl = async () => {
      calls += 1;
      if (calls === 1) {
        return new Response('', { status: 429, statusText: 'Too Many Requests' });
      }
      return new Response(JSON.stringify(yahooPayload(501)), { status: 200 });
    };

    const rows = await fetchYahooV8('SPY', '2024-01-01', '2024-01-02', {
      fetchImpl,
      maxAttempts: 2,
      backoffMs: 0,
    });

    expect(calls).toBe(2);
    expect(rows).toEqual([
      {
        date: '2024-01-01',
        open: 500,
        high: 502,
        low: 499,
        close: 501,
        adjClose: 501,
        volume: 123,
      },
    ]);
  });

  it('treats endDate as an inclusive calendar day in Yahoo period2', async () => {
    // Yahoo chart period2 is exclusive of bars at/after the boundary. Requesting
    // endDate at UTC midnight drops same-day session bars (notably ^VIX3M).
    let requestedUrl = '';
    const fetchImpl = async (input: RequestInfo | URL) => {
      requestedUrl = String(input);
      return new Response(JSON.stringify(yahooPayload(100)), { status: 200 });
    };

    await fetchYahooV8('^VIX3M', '2005-01-01', '2026-07-17', {
      fetchImpl,
      maxAttempts: 1,
      backoffMs: 0,
    });

    const period2 = Number(new URL(requestedUrl).searchParams.get('period2'));
    const endMidnightUtc = Math.floor(Date.parse('2026-07-17T00:00:00.000Z') / 1000);
    expect(Number.isFinite(period2)).toBe(true);
    // Must request through end of endDate UTC day (exclusive next midnight).
    expect(period2).toBeGreaterThanOrEqual(endMidnightUtc + 86_400);
  });

  it('classifies malformed Yahoo payloads without live network calls', async () => {
    const fetchImpl = async () => new Response('not-json', { status: 200 });

    await expect(fetchYahooV8('SPY', '2024-01-01', '2024-01-02', {
      fetchImpl,
      maxAttempts: 1,
      backoffMs: 0,
    })).rejects.toMatchObject({ reason: 'malformed_payload' });
  });

  it('opens a provider circuit breaker after repeated Yahoo rate limits', async () => {
    let calls = 0;
    const fetchImpl = async () => {
      calls += 1;
      return new Response('', { status: 429, statusText: 'Too Many Requests' });
    };

    const result = await fetchAllDataWithSummary(['SPY', 'GLD', 'TLT', 'IEF'], '2024-01-01', '2024-01-02', {
      fetchImpl,
      maxAttempts: 1,
      backoffMs: 0,
      delayMs: 0,
      circuitBreakerFailureThreshold: 2,
    });

    expect(calls).toBe(2);
    expect(Object.keys(result.data)).toEqual([]);
    expect(result.summary.status).toBe('failed');
    expect(result.summary.failure_counts.rate_limited).toBe(2);
    expect(result.summary.circuit_breaker).toEqual({
      opened: true,
      reason: 'rate_limited',
      skipped_symbols: ['TLT', 'IEF'],
    });
    expect(result.summary.symbols.map((symbol) => symbol.status)).toEqual(['failed', 'failed', 'skipped', 'skipped']);
  });

  it('uses a configured licensed provider before Yahoo and records provider-chain success metadata', async () => {
    const provider = stubProvider('Licensed Fixture', async () => licensedRows);

    const result = await fetchAllDataWithSummary(['SPY'], '2024-01-01', '2024-01-02', {
      providers: [provider],
      delayMs: 0,
    });

    expect(result.data.SPY).toEqual(licensedRows);
    expect(result.summary).toMatchObject({
      provider: 'Licensed Fixture',
      feed: 'adjusted-eod-fixture',
      status: 'success',
      source_mode: 'live',
      provider_chain: ['Licensed Fixture'],
      primary_provider: 'Licensed Fixture',
      fallback_provider: null,
    });
    expect(result.summary.symbols[0]).toMatchObject({
      symbol: 'SPY',
      provider: 'Licensed Fixture',
      feed: 'adjusted-eod-fixture',
      status: 'success',
      source_mode: 'live',
      rows: 1,
      latest_observation: '2024-01-01',
      provider_chain: ['Licensed Fixture'],
      primary_provider: 'Licensed Fixture',
      fallback_provider: null,
    });
  });

  it('falls back from a licensed provider to Yahoo without changing compact price shape', async () => {
    const primary = stubProvider('Licensed Fixture', async () => {
      throw Object.assign(new Error('licensed provider unavailable'), { reason: 'network_error' });
    });
    const yahoo = createYahooFinanceProvider({
      fetchImpl: async () => new Response(JSON.stringify(yahooPayload(402)), { status: 200 }),
      maxAttempts: 1,
      backoffMs: 0,
    });

    const result = await fetchAllDataWithSummary(['SPY'], '2024-01-01', '2024-01-02', {
      providers: [primary, yahoo],
      delayMs: 0,
    });
    const compact = Object.fromEntries(
      Object.entries(result.data).map(([symbol, prices]) => [
        symbol,
        prices.map((price) => ({ d: price.date, p: price.adjClose })),
      ]),
    );
    const manifestRows = buildPriceSourceRows(result.summary, compact, '2026-06-12T00:00:00Z');

    expect(compact).toEqual({ SPY: [{ d: '2024-01-01', p: 402 }] });
    expect(result.summary).toMatchObject({
      provider: 'Yahoo Finance',
      feed: 'chart/v8',
      status: 'degraded',
      provider_chain: ['Licensed Fixture', 'Yahoo Finance'],
      primary_provider: 'Licensed Fixture',
      fallback_provider: 'Yahoo Finance',
    });
    expect(result.summary.failure_counts.network_error).toBe(1);
    expect(result.summary.symbols[0]).toMatchObject({
      symbol: 'SPY',
      provider: 'Yahoo Finance',
      feed: 'chart/v8',
      status: 'degraded',
      fallback_reason: 'network_error',
      provider_chain: ['Licensed Fixture', 'Yahoo Finance'],
      primary_provider: 'Licensed Fixture',
      fallback_provider: 'Yahoo Finance',
    });
    expect(manifestRows[0]).toMatchObject({
      artifact: 'prices.json',
      provider: 'Yahoo Finance',
      provider_chain: ['Licensed Fixture', 'Yahoo Finance'],
      primary_provider: 'Licensed Fixture',
      fallback_provider: 'Yahoo Finance',
      failure_reason: 'network_error',
      source_mode: 'live',
      row_count: 1,
      symbols: ['SPY'],
    });
    expect(JSON.stringify(manifestRows)).not.toContain('query2.finance.yahoo.com');
  });

  it('converts multi-symbol price rows to chronological backtest rows', () => {
    const rows = convertToBacktestFormat({
      GLD: [
        { date: '2024-01-03', open: 192, high: 194, low: 191, close: 193, adjClose: 193, volume: 1 },
        { date: '2024-01-01', open: 190, high: 192, low: 189, close: 191, adjClose: 191, volume: 1 },
      ],
      SPY: [
        { date: '2024-01-02', open: 470, high: 472, low: 469, close: 471, adjClose: 471, volume: 1 },
      ],
    });

    expect(rows).toEqual([
      { date: '2024-01-01', symbol: 'GLD', price: 191 },
      { date: '2024-01-02', symbol: 'SPY', price: 471 },
      { date: '2024-01-03', symbol: 'GLD', price: 193 },
    ]);
  });

  it('builds an ok price data quality report for clean compact prices', () => {
    const report = buildPriceDataQualityReport(
      cleanCompactPrices(),
      '2026-06-12T00:00:00Z',
      { maxLatestLagDays: Number.POSITIVE_INFINITY },
    );

    expect(PRICE_DATA_QUALITY_FILENAME).toBe('data_quality.json');
    expect(report).toEqual({
      schema_version: PRICE_DATA_QUALITY_SCHEMA_VERSION,
      generated_at: '2026-06-12T00:00:00Z',
      generator_git_sha: expect.any(String),
      generator_git_sha_status: 'full_generate',
      overall_status: 'ok',
      issue_counts: {
        duplicate_dates: 0,
        empty_symbols: 0,
        extreme_returns: 0,
        internal_gaps: 0,
        invalid_dates: 0,
        invalid_prices: 0,
        missing_required_keys: 0,
        non_monotonic_rows: 0,
        non_object_records: 0,
        split_like_returns: 0,
        stale_latest_dates: 0,
        total: 0,
      },
      symbols: [
        {
          symbol: 'GLD',
          status: 'ok',
          row_count: 1,
          first_date: '2026-06-11',
          latest_date: '2026-06-11',
          duplicate_date_count: 0,
          duplicate_dates: [],
          internal_gaps: [],
          invalid_dates: [],
          invalid_prices: [],
          latest_lag_days: 1,
          missing_required_keys: [],
          non_monotonic_rows: [],
          non_object_records: [],
          return_anomaly_count: 0,
          return_anomalies: [],
          stale_latest_date: null,
        },
        {
          symbol: 'SPY',
          status: 'ok',
          row_count: 2,
          first_date: '2026-06-10',
          latest_date: '2026-06-11',
          duplicate_date_count: 0,
          duplicate_dates: [],
          internal_gaps: [],
          invalid_dates: [],
          invalid_prices: [],
          latest_lag_days: 1,
          missing_required_keys: [],
          non_monotonic_rows: [],
          non_object_records: [],
          return_anomaly_count: 0,
          return_anomalies: [],
          stale_latest_date: null,
        },
      ],
    });
  });

  it('keeps aligned cross-sections ok and counts only the weekday since the latest bar', () => {
    const report = buildPriceDataQualityReport(
      {
        SPY: [
          { d: '2026-06-12', p: 612.34 },
          { d: '2026-06-15', p: 614.25 },
        ],
        GLD: [
          { d: '2026-06-12', p: 318.12 },
          { d: '2026-06-15', p: 319.2 },
        ],
        TLT: [
          { d: '2026-06-12', p: 88.75 },
          { d: '2026-06-15', p: 89.01 },
        ],
      },
      '2026-06-16T00:00:00Z',
    );

    expect(report.overall_status).toBe('ok');
    expect(report.issue_counts.internal_gaps).toBe(0);
    expect(report.issue_counts.stale_latest_dates).toBe(0);
    expect(report.symbols.every((symbol) => symbol.internal_gaps.length === 0)).toBe(true);
    expect(report.symbols.every((symbol) => symbol.latest_lag_days === 1)).toBe(true);
  });

  it('flags symbols whose latest date lags the reference calendar beyond threshold', () => {
    const report = buildPriceDataQualityReport(
      staleLatestCompactPrices(),
      '2026-06-13T00:00:00Z',
      { maxLatestLagDays: 0 },
    );

    const gld = report.symbols.find((symbol) => symbol.symbol === 'GLD');
    expect(report.overall_status).toBe('fail');
    expect(report.issue_counts.stale_latest_dates).toBe(1);
    expect(gld).toMatchObject({
      status: 'fail',
      latest_lag_days: 1,
      stale_latest_date: {
        reference_date: '2026-06-12',
        latest_date: '2026-06-11',
      },
    });
  });

  it('flags bounded samples of internal reference-calendar gaps as advisory warn', () => {
    const report = buildPriceDataQualityReport(
      internalGapCompactPrices(),
      '2026-06-13T00:00:00Z',
      { maxMissingDateSamples: 2 },
    );

    const tlt = report.symbols.find((symbol) => symbol.symbol === 'TLT');
    // Sparse / lagging mid-history holes must not fail-close the data job when
    // the series is otherwise valid (see ^VIX3M vs SPY calendar).
    expect(report.overall_status).toBe('warn');
    expect(report.issue_counts.internal_gaps).toBe(1);
    expect(tlt).toMatchObject({
      status: 'warn',
      internal_gaps: [
        {
          missing_count: 1,
          sample_missing_dates: ['2026-06-11'],
        },
      ],
    });
  });

  it('keeps sparse index series current when latest matches SPY despite mid-history holes', () => {
    const report = buildPriceDataQualityReport(
      {
        SPY: [
          { d: '2026-07-10', p: 740 },
          { d: '2026-07-13', p: 742 },
          { d: '2026-07-14', p: 743 },
          { d: '2026-07-15', p: 744 },
          { d: '2026-07-16', p: 750 },
          { d: '2026-07-17', p: 743 },
        ],
        '^VIX3M': [
          { d: '2026-07-10', p: 18.57 },
          // Yahoo often omits mid-week bars for VIX indices while still publishing
          // the latest session — not a staleness failure once latest matches SPY.
          { d: '2026-07-17', p: 20.35 },
        ],
      },
      '2026-07-18T00:00:00Z',
    );

    const vix = report.symbols.find((symbol) => symbol.symbol === '^VIX3M');
    expect(report.issue_counts.stale_latest_dates).toBe(0);
    expect(report.overall_status).not.toBe('fail');
    expect(vix).toMatchObject({
      latest_date: '2026-07-17',
      latest_lag_days: 0,
      stale_latest_date: null,
    });
    expect(vix?.internal_gaps.length ?? 0).toBeGreaterThan(0);
    expect(vix?.status).toBe('warn');
  });

  it('treats sparse-index latest lag as advisory when Yahoo null-pads after last real bar', () => {
    // Live Yahoo chart returns calendar timestamps through SPY's latest day with
    // null closes for ^VIX3M after 2026-07-10; fetcher keeps last non-null bar.
    // That lag must not fail-closed the whole data job / SLO.
    const report = buildPriceDataQualityReport(
      {
        SPY: [
          { d: '2026-07-10', p: 740 },
          { d: '2026-07-13', p: 742 },
          { d: '2026-07-14', p: 743 },
          { d: '2026-07-15', p: 744 },
          { d: '2026-07-16', p: 750 },
          { d: '2026-07-17', p: 743 },
        ],
        '^VIX3M': [
          { d: '2026-07-08', p: 19.46 },
          { d: '2026-07-09', p: 18.99 },
          { d: '2026-07-10', p: 18.57 },
        ],
      },
      '2026-07-18T00:00:00Z',
    );

    const vix = report.symbols.find((symbol) => symbol.symbol === '^VIX3M');
    expect(vix).toMatchObject({
      latest_date: '2026-07-10',
      latest_lag_days: 5,
    });
    // Visibility retained, but not a blocking stale_latest_dates count.
    expect(vix?.stale_latest_date).toEqual({
      reference_date: '2026-07-17',
      latest_date: '2026-07-10',
    });
    expect(report.issue_counts.stale_latest_dates).toBe(0);
    expect(report.overall_status).toBe('warn');
    expect(vix?.status).toBe('warn');
  });

  it('still fails equity symbols that lag the reference calendar', () => {
    const report = buildPriceDataQualityReport(
      {
        SPY: [
          { d: '2026-07-15', p: 744 },
          { d: '2026-07-16', p: 750 },
          { d: '2026-07-17', p: 743 },
        ],
        GLD: [
          { d: '2026-07-15', p: 220 },
        ],
      },
      '2026-07-18T00:00:00Z',
    );

    const gld = report.symbols.find((symbol) => symbol.symbol === 'GLD');
    expect(report.issue_counts.stale_latest_dates).toBe(1);
    expect(report.overall_status).toBe('fail');
    expect(gld?.status).toBe('fail');
    expect(gld?.stale_latest_date).toEqual({
      reference_date: '2026-07-17',
      latest_date: '2026-07-15',
    });
  });

  it('flags invalid prices, non-monotonic rows, missing keys, duplicate dates, and bad dates', () => {
    const report = buildPriceDataQualityReport(
      {
        SPY: [
          { d: '2026-06-11', p: 612.34 },
          { d: '2026-06-10', p: 613.5 },
          { d: '2026-06-12', p: 0 },
          { d: '2026-06-13' },
          { d: 'not-a-date', p: 614 },
          { d: '2026-06-14', p: -1 },
        ],
        GLD: [
          { d: '2026-06-10', p: 318.12 },
          { d: '2026-06-10', p: 319.01 },
        ],
      },
      '2026-06-12T00:00:00Z',
      { maxLatestLagDays: Number.POSITIVE_INFINITY },
    );

    expect(report.overall_status).toBe('fail');
    expect(report.issue_counts).toEqual({
      duplicate_dates: 1,
      empty_symbols: 0,
      extreme_returns: 0,
      internal_gaps: 0,
      invalid_dates: 1,
      invalid_prices: 2,
      missing_required_keys: 1,
      non_monotonic_rows: 1,
      non_object_records: 0,
      split_like_returns: 0,
      stale_latest_dates: 0,
      total: 6,
    });
    expect(report.symbols.find((symbol) => symbol.symbol === 'GLD')).toMatchObject({
      status: 'fail',
      duplicate_dates: ['2026-06-10'],
    });
    expect(report.symbols.find((symbol) => symbol.symbol === 'SPY')).toMatchObject({
      status: 'fail',
      invalid_dates: [{ index: 4 }],
      invalid_prices: [
        { index: 2, date: '2026-06-12' },
        { index: 5, date: '2026-06-14' },
      ],
      missing_required_keys: [{ index: 3, missing_keys: ['p'] }],
      non_monotonic_rows: [{ index: 1, previous_date: '2026-06-11', date: '2026-06-10' }],
    });
  });

  it('bounds duplicate date samples while preserving duplicate issue counts', () => {
    const report = buildPriceDataQualityReport(
      {
        SPY: [
          { d: '2026-06-10', p: 612.34 },
          { d: '2026-06-10', p: 612.35 },
          { d: '2026-06-11', p: 613.1 },
          { d: '2026-06-11', p: 613.2 },
          { d: '2026-06-12', p: 614.1 },
          { d: '2026-06-12', p: 614.2 },
        ],
      },
      '2026-06-12T00:00:00Z',
      {
        maxDuplicateDateSamples: 2,
        maxLatestLagDays: Number.POSITIVE_INFINITY,
      },
    );

    expect(report.issue_counts.duplicate_dates).toBe(3);
    expect(report.issue_counts.total).toBe(3);
    expect(report.symbols[0]).toMatchObject({
      symbol: 'SPY',
      status: 'fail',
      duplicate_date_count: 3,
      duplicate_dates: ['2026-06-10', '2026-06-11'],
    });
  });

  it('reports split-like return jumps as warning-level bounded offender samples', () => {
    const report = buildPriceDataQualityReport(
      splitLikeReturnCompactPrices(),
      '2026-06-13T00:00:00Z',
      {
        criticalReturnPct: 125,
        maxLatestLagDays: Number.POSITIVE_INFINITY,
        maxReturnAnomalySamples: 1,
        splitLikeReturnPct: 40,
      },
    );

    expect(report.overall_status).toBe('warn');
    expect(report.issue_counts.split_like_returns).toBe(1);
    expect(report.issue_counts.extreme_returns).toBe(0);
    expect(report.issue_counts.total).toBe(1);
    expect(report.symbols[0]).toMatchObject({
      symbol: 'SPY',
      status: 'warn',
      return_anomaly_count: 1,
      return_anomalies: [
        {
          type: 'split_like_return',
          severity: 'warning',
          symbol: 'SPY',
          date: '2026-06-11',
          previous_date: '2026-06-10',
          previous_price: 100,
          current_price: 45,
          return_pct: -55,
        },
      ],
    });
  });

  it('reports critical extreme returns as blocking failures with return context', () => {
    const report = buildPriceDataQualityReport(
      extremeReturnCompactPrices(),
      '2026-06-12T00:00:00Z',
      {
        criticalReturnPct: 90,
        maxLatestLagDays: Number.POSITIVE_INFINITY,
        splitLikeReturnPct: 40,
      },
    );

    expect(report.overall_status).toBe('fail');
    expect(report.issue_counts.extreme_returns).toBe(1);
    expect(report.issue_counts.split_like_returns).toBe(0);
    expect(report.symbols[0]).toMatchObject({
      symbol: 'SPY',
      status: 'fail',
      return_anomaly_count: 1,
      return_anomalies: [
        {
          type: 'extreme_return',
          severity: 'critical',
          symbol: 'SPY',
          date: '2026-06-11',
          previous_date: '2026-06-10',
          previous_price: 100,
          current_price: 250,
          return_pct: 150,
        },
      ],
    });
  });

  it('keeps plausible high-volatility returns below configured thresholds clean', () => {
    const report = buildPriceDataQualityReport(
      {
        SPY: [
          { d: '2026-06-10', p: 100 },
          { d: '2026-06-11', p: 125 },
        ],
      },
      '2026-06-12T00:00:00Z',
      {
        criticalReturnPct: 90,
        maxLatestLagDays: Number.POSITIVE_INFINITY,
        splitLikeReturnPct: 40,
      },
    );

    expect(report.overall_status).toBe('ok');
    expect(report.issue_counts.extreme_returns).toBe(0);
    expect(report.issue_counts.split_like_returns).toBe(0);
    expect(report.symbols[0]).toMatchObject({
      status: 'ok',
      return_anomaly_count: 0,
      return_anomalies: [],
    });
  });

  it('skips split_like equity gates on VIX-family crisis jumps (Batch BG)', () => {
    const report = buildPriceDataQualityReport(
      {
        SPY: [
          { d: '2018-02-02', p: 100 },
          { d: '2018-02-05', p: 98 },
        ],
        '^VIX3M': [
          { d: '2018-02-02', p: 17 },
          { d: '2018-02-05', p: 37 }, // ~+118% regime jump, not a split
        ],
      },
      '2018-02-06T00:00:00Z',
      {
        criticalReturnPct: 90,
        maxLatestLagDays: Number.POSITIVE_INFINITY,
        splitLikeReturnPct: 40,
      },
    );

    expect(report.overall_status).toBe('ok');
    expect(report.issue_counts.split_like_returns).toBe(0);
    expect(report.issue_counts.extreme_returns).toBe(0);
    const vix = report.symbols.find((s) => s.symbol === '^VIX3M');
    expect(vix?.status).toBe('ok');
    expect(vix?.return_anomaly_count).toBe(0);
  });

  it('keeps price data quality reports deterministic and compact', () => {
    const payload = {
      TLT: [{ d: '2026-06-11', p: 88.75 }],
      SPY: [{ d: '2026-06-10', p: 612.34 }],
    };

    const first = JSON.stringify(buildPriceDataQualityReport(payload, '2026-06-12T00:00:00Z'));
    const second = JSON.stringify(buildPriceDataQualityReport(payload, '2026-06-12T00:00:00Z'));

    expect(first).toBe(second);
    expect(first).toContain('"symbols":[{"symbol":"SPY"');
    expect(first).toContain('"symbol":"TLT"');
    expect(first).not.toContain('undefined');
    expect(first.length).toBeLessThan(1600);
  });

  it('attaches compact price data quality summaries to price source manifest rows', () => {
    const compact = {
      SPY: [
        { d: '2026-06-10', p: 100 },
        { d: '2026-06-11', p: 45 },
      ],
    };
    const qualityReport = buildPriceDataQualityReport(
      compact,
      '2026-06-12T00:00:00Z',
      { maxLatestLagDays: Number.POSITIVE_INFINITY },
    );

    const rows = buildPriceSourceRows(
      priceProviderSummary({ status: 'degraded' }),
      compact,
      '2026-06-12T00:00:00Z',
      qualityReport,
    );

    expect(rows.map((row) => row.artifact)).toEqual(['prices.json', 'prices_compact.json']);
    expect(rows.every((row) => row.data_quality?.artifact === PRICE_DATA_QUALITY_FILENAME)).toBe(true);
    expect(rows[0].data_quality).toEqual({
      artifact: PRICE_DATA_QUALITY_FILENAME,
      schema_version: PRICE_DATA_QUALITY_SCHEMA_VERSION,
      generated_at: '2026-06-12T00:00:00Z',
      status: 'warn',
      issue_counts: {
        duplicate_dates: 0,
        empty_symbols: 0,
        extreme_returns: 0,
        internal_gaps: 0,
        invalid_dates: 0,
        invalid_prices: 0,
        missing_required_keys: 0,
        non_monotonic_rows: 0,
        non_object_records: 0,
        split_like_returns: 1,
        stale_latest_dates: 0,
        total: 1,
      },
    });
    expect(JSON.stringify(rows[0].data_quality)).not.toContain('"symbols"');
    expect(JSON.stringify(rows[0].data_quality)).not.toContain('return_anomalies');
  });

  it('records structured failure state when every price provider fails', async () => {
    const primary = stubProvider('Licensed Fixture', async () => {
      throw Object.assign(new Error('licensed timeout'), { reason: 'timeout' });
    });
    const fallback = stubProvider('Yahoo Fixture', async () => {
      throw Object.assign(new Error('yahoo returned no rows'), { reason: 'no_data' });
    });

    const result = await fetchAllDataWithSummary(['SPY'], '2024-01-01', '2024-01-02', {
      providers: [primary, fallback],
      delayMs: 0,
    });

    expect(result.data).toEqual({});
    expect(result.summary).toMatchObject({
      provider: 'none',
      feed: 'none',
      status: 'failed',
      provider_chain: ['Licensed Fixture', 'Yahoo Fixture'],
      primary_provider: 'Licensed Fixture',
      fallback_provider: null,
    });
    expect(result.summary.failure_counts.timeout).toBe(1);
    expect(result.summary.failure_counts.no_data).toBe(1);
    expect(result.summary.symbols[0]).toMatchObject({
      symbol: 'SPY',
      provider: null,
      feed: null,
      status: 'failed',
      failure_reason: 'no_data',
      provider_chain: ['Licensed Fixture', 'Yahoo Fixture'],
      primary_provider: 'Licensed Fixture',
      fallback_provider: null,
    });
    expect(result.summary.symbols[0].error).toContain('All price providers failed for SPY');
  });

  it('makes FRED fallback deterministic and labels synthetic source mode', async () => {
    const oldKey = process.env.FRED_API_KEY;
    delete process.env.FRED_API_KEY;
    try {
      const first = generateFallbackYields('DGS10', '2024-01-01', '2024-01-03');
      const second = generateFallbackYields('DGS10', '2024-01-01', '2024-01-03');
      const result = await fetchYieldCurveDataWithSummary('2024-01-01', '2024-01-03');

      expect(first).toEqual(second);
      expect(result.summary.provider).toBe('FRED');
      expect(result.summary.status).toBe('degraded');
      expect(result.summary.source_mode).toBe('synthetic');
      expect(result.summary.series.every((series) => series.failure_reason === 'missing_api_key')).toBe(true);
    } finally {
      if (oldKey === undefined) {
        delete process.env.FRED_API_KEY;
      } else {
        process.env.FRED_API_KEY = oldKey;
      }
    }
  });

  it('returns a fresh cached FRED series without calling the network', async () => {
    let calls = 0;
    const cache = memoryFredCache([
      {
        series_id: 'DGS10',
        start_date: '2024-01-01',
        end_date: '2024-01-01',
        fetched_at: '2026-06-12T00:00:00Z',
        observations: [{ date: '2024-01-01', value: 4.25 }],
      },
    ]);

    const result = await fetchFredSeriesWithSummary('DGS10', '2024-01-01', '2024-01-01', {
      cache,
      cacheTtlMs: 60 * 60 * 1000,
      now: () => new Date('2026-06-12T00:30:00Z'),
      fetchImpl: async () => {
        calls += 1;
        throw new Error('network should not be called for fresh FRED cache');
      },
    });

    expect(calls).toBe(0);
    expect(result.data).toEqual([{ date: '2024-01-01', value: 4.25 }]);
    expect(result.summary).toMatchObject({
      status: 'success',
      source_mode: 'cached',
      failure_reason: undefined,
    });
  });

  it('returns stale cached FRED data when no live refresh is possible', async () => {
    const oldKey = process.env.FRED_API_KEY;
    delete process.env.FRED_API_KEY;
    try {
      const cache = memoryFredCache([
        {
          series_id: 'DGS10',
          start_date: '2024-01-01',
          end_date: '2024-01-01',
          fetched_at: '2026-06-10T00:00:00Z',
          observations: [{ date: '2024-01-01', value: 4.1 }],
        },
      ]);

      const result = await fetchFredSeriesWithSummary('DGS10', '2024-01-01', '2024-01-01', {
        cache,
        cacheTtlMs: 24 * 60 * 60 * 1000,
        now: () => new Date('2026-06-12T00:00:00Z'),
      });

      expect(result.data).toEqual([{ date: '2024-01-01', value: 4.1 }]);
      expect(result.summary).toMatchObject({
        status: 'degraded',
        source_mode: 'stale_cached',
        failure_reason: 'cache_stale',
        fallback_reason: 'missing_api_key',
      });
    } finally {
      if (oldKey === undefined) {
        delete process.env.FRED_API_KEY;
      } else {
        process.env.FRED_API_KEY = oldKey;
      }
    }
  });

  it('retries FRED rate limits and stores eventual live success in cache', async () => {
    const oldKey = process.env.FRED_API_KEY;
    process.env.FRED_API_KEY = 'fixture-key';
    try {
      let calls = 0;
      const cache = memoryFredCache();
      const result = await fetchFredSeriesWithSummary('DGS10', '2024-01-01', '2024-01-01', {
        cache,
        maxAttempts: 2,
        backoffMs: 0,
        fetchImpl: async () => {
          calls += 1;
          if (calls === 1) {
            return new Response('', { status: 429, statusText: 'Too Many Requests' });
          }
          return new Response(JSON.stringify(fredPayload('4.33')), { status: 200 });
        },
      });

      expect(calls).toBe(2);
      expect(result.data).toEqual([{ date: '2024-01-01', value: 4.33 }]);
      expect(result.summary.source_mode).toBe('live');
      expect(cache.records.get('DGS10:2024-01-01:2024-01-01')?.observations).toEqual(result.data);
    } finally {
      if (oldKey === undefined) {
        delete process.env.FRED_API_KEY;
      } else {
        process.env.FRED_API_KEY = oldKey;
      }
    }
  });

  it('returns deterministic synthetic FRED fallback after bounded rate-limit exhaustion', async () => {
    const oldKey = process.env.FRED_API_KEY;
    process.env.FRED_API_KEY = 'fixture-key';
    try {
      let calls = 0;
      const result = await fetchFredSeriesWithSummary('DGS10', '2024-01-01', '2024-01-02', {
        maxAttempts: 2,
        backoffMs: 0,
        fetchImpl: async () => {
          calls += 1;
          return new Response('', { status: 429, statusText: 'Too Many Requests' });
        },
      });

      expect(calls).toBe(2);
      expect(result.summary).toMatchObject({
        status: 'degraded',
        source_mode: 'synthetic',
        failure_reason: 'rate_limited',
      });
      expect(result.data).toEqual(generateFallbackYields('DGS10', '2024-01-01', '2024-01-02'));
    } finally {
      if (oldKey === undefined) {
        delete process.env.FRED_API_KEY;
      } else {
        process.env.FRED_API_KEY = oldKey;
      }
    }
  });

  it('classifies malformed FRED JSON as malformed payload', async () => {
    const oldKey = process.env.FRED_API_KEY;
    process.env.FRED_API_KEY = 'fixture-key';
    try {
      const result = await fetchFredSeriesWithSummary('DGS10', '2024-01-01', '2024-01-01', {
        maxAttempts: 1,
        fetchImpl: async () => new Response('not-json', { status: 200 }),
      });

      expect(result.summary).toMatchObject({
        status: 'degraded',
        source_mode: 'synthetic',
        failure_reason: 'malformed_payload',
      });
    } finally {
      if (oldKey === undefined) {
        delete process.env.FRED_API_KEY;
      } else {
        process.env.FRED_API_KEY = oldKey;
      }
    }
  });

  it('maps stale FRED cache reasons into the yield source manifest row', () => {
    const row = buildYieldSourceRow(
      {
        provider: 'FRED',
        feed: 'series/observations',
        status: 'degraded',
        source_mode: 'stale_cached',
        fetched_at: '2026-06-12T00:00:00Z',
        series: [
          {
            series_id: 'DGS10',
            status: 'degraded',
            source_mode: 'stale_cached',
            rows: 1,
            latest_observation: '2024-01-01',
            fetched_at: '2026-06-12T00:00:00Z',
            failure_reason: 'cache_stale',
            fallback_reason: 'rate_limited',
          },
        ],
      },
      1,
      '2024-01-01',
      '2026-06-12T00:00:00Z',
    );

    expect(row).toMatchObject({
      artifact: 'yields.json',
      source_mode: 'stale_cached',
      status: 'degraded',
      failure_reason: 'cache_stale',
      fallback_reason: 'rate_limited',
    });
  });

  it('preserves concurrent FRED disk cache writes from parallel series fetches', async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), 'portfolio-lab-fred-cache-'));
    try {
      const cachePath = join(tmpDir, 'fred_series_cache.json');
      const cache = createFredDiskCache(cachePath);
      await Promise.all([
        cache.set(
          { seriesId: 'DGS2', startDate: '2024-01-01', endDate: '2024-01-01' },
          { fetched_at: '2026-06-12T00:00:00Z', observations: [{ date: '2024-01-01', value: 4.2 }] },
        ),
        cache.set(
          { seriesId: 'DGS10', startDate: '2024-01-01', endDate: '2024-01-01' },
          { fetched_at: '2026-06-12T00:00:00Z', observations: [{ date: '2024-01-01', value: 4.3 }] },
        ),
        cache.set(
          { seriesId: 'DGS30', startDate: '2024-01-01', endDate: '2024-01-01' },
          { fetched_at: '2026-06-12T00:00:00Z', observations: [{ date: '2024-01-01', value: 4.4 }] },
        ),
      ]);

      const payload = JSON.parse(readFileSync(cachePath, 'utf8')) as Record<string, unknown>;
      expect(Object.keys(payload).sort()).toEqual([
        'DGS10:2024-01-01:2024-01-01',
        'DGS2:2024-01-01:2024-01-01',
        'DGS30:2024-01-01:2024-01-01',
      ]);
    } finally {
      rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it('builds last-good retention manifest without leaking provider URLs', () => {
    const manifest = buildLastGoodRetentionManifest(new Error('HTTP 429'), '2026-06-11T00:00:00Z');

    expect(manifest.schema_version).toBe('market-data-source-manifest/v1');
    expect(manifest.artifacts).toHaveLength(2);
    expect(manifest.artifacts[0]).toMatchObject({
      artifact: 'prices.json',
      provider: 'Yahoo Finance',
      source_mode: 'last_good',
      status: 'failed',
      fallback_reason: 'HTTP 429',
    });
    expect(manifest.artifacts.every((row) => row.data_quality?.status === 'unavailable')).toBe(true);
    expect(manifest.artifacts.every((row) => row.data_quality?.artifact === PRICE_DATA_QUALITY_FILENAME)).toBe(true);
    expect(JSON.stringify(manifest)).not.toContain('query2.finance.yahoo.com');
  });

  it('fails dashboard generation as a post-fetch artifact error', async () => {
    await expect(runDashboardGeneration(async () => {
      throw new Error('dashboard generator exploded');
    })).rejects.toThrow('Dashboard generation failed after market data refresh');
  });

  it('does not rewrite provider retention manifest for dashboard generation failures', () => {
    const error = new DashboardGenerationError(new Error('dashboard generator exploded'));

    expect(shouldWriteLastGoodRetentionManifest(error)).toBe(false);
    expect(shouldWriteLastGoodRetentionManifest(new Error('provider failed'))).toBe(true);
  });

  it('attaches symbol-universe metadata to source manifests', () => {
    const manifest = buildMarketDataSourceManifest([], '2026-06-12T00:00:00Z');

    expect(manifest.symbol_universe).toEqual(SYMBOL_UNIVERSE_METADATA);
  });
});
