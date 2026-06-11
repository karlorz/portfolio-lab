import { describe, expect, it } from 'bun:test';
import {
  fetchAllDataWithSummary,
  fetchYahooV8,
  fetchYieldCurveDataWithSummary,
  generateFallbackYields,
} from '../../src/data/fetcher';
import { buildLastGoodRetentionManifest } from '../../scripts/fetch-data';

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
