import { describe, expect, it } from 'bun:test';
import { readFileSync } from 'fs';
import {
  assertPriceQualityAllowsSuccess,
  buildPriceSourceRows,
  priceManifestStatusFromQuality,
  PriceDataQualityGateError,
  shouldWriteLastGoodRetentionManifest,
  DashboardGenerationError,
} from '../../scripts/fetch-data';
import {
  buildPriceDataQualityReport,
  type PriceDataQualityReport,
} from '../../src/data/price_quality';
import { staleLatestCompactPrices } from './compact-price-fixtures';

function okProviderSummary() {
  return {
    provider: 'Yahoo Finance',
    feed: 'chart/v8',
    provider_chain: ['Yahoo Finance'],
    primary_provider: 'Yahoo Finance',
    fallback_provider: null,
    source_mode: 'live' as const,
    status: 'success' as const,
    failure_counts: {},
    circuit_breaker: {
      opened: false,
      reason: null,
      skipped_symbols: [] as string[],
    },
  };
}

describe('fetch-data price quality fail-closed gate', () => {
  it('assertPriceQualityAllowsSuccess throws on overall_status fail', () => {
    const report = buildPriceDataQualityReport(staleLatestCompactPrices());
    // stale fixtures may be fail or warn depending on thresholds — force fail
    const failReport: PriceDataQualityReport = {
      ...report,
      overall_status: 'fail',
    };
    expect(() => assertPriceQualityAllowsSuccess(failReport)).toThrow(PriceDataQualityGateError);
  });

  it('assertPriceQualityAllowsSuccess allows ok and warn', () => {
    const ok: PriceDataQualityReport = {
      schema_version: 'price-data-quality/v1',
      generated_at: new Date().toISOString(),
      overall_status: 'ok',
      issue_counts: {
        total: 0,
        missing_required_keys: 0,
        invalid_prices: 0,
        invalid_dates: 0,
        non_monotonic_rows: 0,
        non_object_records: 0,
        stale_latest_dates: 0,
        internal_gaps: 0,
        extreme_returns: 0,
        split_like_returns: 0,
      },
      symbols: [],
    };
    expect(() => assertPriceQualityAllowsSuccess(ok)).not.toThrow();
    expect(() => assertPriceQualityAllowsSuccess({ ...ok, overall_status: 'warn' })).not.toThrow();
  });

  it('priceManifestStatusFromQuality couples quality fail to failed', () => {
    expect(priceManifestStatusFromQuality('success', 'fail')).toBe('failed');
    expect(priceManifestStatusFromQuality('success', 'warn')).toBe('degraded');
    expect(priceManifestStatusFromQuality('success', 'ok')).toBe('success');
    expect(priceManifestStatusFromQuality('failed', 'ok')).toBe('failed');
  });

  it('buildPriceSourceRows marks failed when quality overall_status is fail', () => {
    const failReport: PriceDataQualityReport = {
      schema_version: 'price-data-quality/v1',
      generated_at: new Date().toISOString(),
      overall_status: 'fail',
      issue_counts: {
        total: 1,
        missing_required_keys: 0,
        invalid_prices: 0,
        invalid_dates: 0,
        non_monotonic_rows: 0,
        non_object_records: 0,
        stale_latest_dates: 1,
        internal_gaps: 0,
        extreme_returns: 0,
        split_like_returns: 0,
      },
      symbols: [{ symbol: '^VIX3M', status: 'fail', latest_date: '2026-07-02', row_count: 10 }],
    };
    const rows = buildPriceSourceRows(
      okProviderSummary() as any,
      { SPY: [{ d: '2026-07-10', p: 100 }] },
      new Date().toISOString(),
      failReport,
    );
    for (const row of rows) {
      expect(row.status).toBe('failed');
      expect(row.data_quality?.status).toBe('fail');
      expect(row.notes?.some((n) => n.includes('overall_status=fail'))).toBe(true);
    }
  });

  it('shouldWriteLastGoodRetentionManifest is false for quality gate errors', () => {
    const err = new PriceDataQualityGateError({
      schema_version: 'price-data-quality/v1',
      generated_at: new Date().toISOString(),
      overall_status: 'fail',
      issue_counts: {
        total: 1,
        missing_required_keys: 0,
        invalid_prices: 0,
        invalid_dates: 0,
        non_monotonic_rows: 0,
        non_object_records: 0,
        stale_latest_dates: 1,
        internal_gaps: 0,
        extreme_returns: 0,
        split_like_returns: 0,
      },
      symbols: [],
    });
    expect(shouldWriteLastGoodRetentionManifest(err)).toBe(false);
    expect(shouldWriteLastGoodRetentionManifest(new DashboardGenerationError(new Error('x')))).toBe(
      false,
    );
    expect(shouldWriteLastGoodRetentionManifest(new Error('network'))).toBe(true);
  });

  it('source contract: main asserts quality gate before Done', () => {
    const source = readFileSync('scripts/fetch-data.ts', 'utf8');
    expect(source).toContain('assertPriceQualityAllowsSuccess');
    expect(source).toContain('PriceDataQualityGateError');
    const gateIdx = source.indexOf('assertPriceQualityAllowsSuccess(priceQualityReport)');
    const doneIdx = source.indexOf("console.log('\\nDone.')");
    const dashIdx = source.indexOf('await runDashboardGeneration()');
    expect(gateIdx).toBeGreaterThan(-1);
    expect(doneIdx).toBeGreaterThan(gateIdx);
    expect(dashIdx).toBeGreaterThan(gateIdx);
  });
});
