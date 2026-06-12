import { describe, expect, it } from 'bun:test';
import {
  createYahooFinanceProvider,
  fetchAllDataWithSummary,
  fetchYahooV8,
  fetchYieldCurveDataWithSummary,
  generateFallbackYields,
  type HistoricalPrice,
  type MarketDataProvider,
} from '../../src/data/fetcher';
import { buildLastGoodRetentionManifest, buildPriceSourceRows } from '../../scripts/fetch-data';

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
    expect(JSON.stringify(manifest)).not.toContain('query2.finance.yahoo.com');
  });
});
