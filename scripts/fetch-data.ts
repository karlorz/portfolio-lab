#!/usr/bin/env bun
/**
 * Fetch market data from Yahoo Finance + FRED.
 * Saves prices.json, prices_compact.json, and yields.json to public/data/.
 *
 * Usage: bun run fetch-data
 */

import { fetchAllDataWithSummary, fetchYieldCurveDataWithSummary, SYMBOLS } from '../src/data/fetcher';
import {
  MARKET_DATA_SOURCE_MANIFEST_FILENAME,
  buildMarketDataSourceManifest,
  type MarketDataSourceRow,
} from '../src/data/source_manifest';
import { join } from 'path';
import { existsSync, mkdirSync, renameSync, unlinkSync } from 'fs';

const PROJECT_ROOT = join(import.meta.dir, '..');
const DATA_DIR = join(import.meta.dir, '..', 'public', 'data');
const PYTHON_RUNTIME = join(PROJECT_ROOT, 'scripts', 'python_runtime.sh');
const START_DATE = '2005-01-01';
const END_DATE = new Date().toISOString().split('T')[0];

export async function writeJsonAtomic(path: string, payload: unknown): Promise<void> {
  const tmpPath = `${path}.${process.pid}.${Date.now()}.tmp`;
  try {
    await Bun.write(tmpPath, JSON.stringify(payload, null, 2));
    renameSync(tmpPath, path);
  } catch (error) {
    try {
      unlinkSync(tmpPath);
    } catch {
      // Best-effort cleanup only.
    }
    throw error;
  }
}

function latestObservationFromCompact(compact: Record<string, { d: string; p: number }[]>): string | null {
  let latest: string | null = null;
  for (const rows of Object.values(compact)) {
    for (const row of rows) {
      if (latest === null || row.d > latest) {
        latest = row.d;
      }
    }
  }
  return latest;
}

function firstFailureReason(
  failureCounts: Awaited<ReturnType<typeof fetchAllDataWithSummary>>['summary']['failure_counts'],
): string | null {
  return Object.keys(failureCounts)[0] ?? null;
}

export function buildPriceSourceRows(
  priceSummary: Awaited<ReturnType<typeof fetchAllDataWithSummary>>['summary'],
  compact: Record<string, { d: string; p: number }[]>,
  fetchedAt: string,
): MarketDataSourceRow[] {
  const rowCount = Object.values(compact).reduce((sum, rows) => sum + rows.length, 0);
  const latestObservation = latestObservationFromCompact(compact);
  const symbols = Object.keys(compact).sort();
  const failureReason = priceSummary.circuit_breaker.opened
    ? priceSummary.circuit_breaker.reason
    : firstFailureReason(priceSummary.failure_counts);
  return ['prices.json', 'prices_compact.json'].map((artifact) => ({
    artifact,
    provider: priceSummary.provider,
    feed: priceSummary.feed,
    provider_chain: priceSummary.provider_chain,
    primary_provider: priceSummary.primary_provider,
    fallback_provider: priceSummary.fallback_provider,
    source_mode: priceSummary.source_mode,
    status: priceSummary.status,
    fetched_at: fetchedAt,
    latest_observation: latestObservation,
    row_count: rowCount,
    symbols,
    failure_reason: failureReason,
    notes: priceSummary.circuit_breaker.opened
      ? [`Skipped symbols after provider circuit breaker opened: ${priceSummary.circuit_breaker.skipped_symbols.join(', ')}`]
      : [],
  }));
}

function buildYieldSourceRow(
  yieldSummary: Awaited<ReturnType<typeof fetchYieldCurveDataWithSummary>>['summary'],
  rowCount: number,
  latestObservation: string | null,
  fetchedAt: string,
): MarketDataSourceRow {
  const failureReason = yieldSummary.series.find((series) => series.failure_reason)?.failure_reason ?? null;
  return {
    artifact: 'yields.json',
    provider: yieldSummary.provider,
    feed: yieldSummary.feed,
    source_mode: yieldSummary.source_mode,
    status: yieldSummary.status,
    fetched_at: fetchedAt,
    latest_observation: latestObservation,
    row_count: rowCount,
    symbols: yieldSummary.series.map((series) => series.series_id),
    failure_reason: failureReason,
    fallback_reason: yieldSummary.source_mode === 'synthetic' ? failureReason : null,
  };
}

export function buildLastGoodRetentionManifest(error: unknown, generatedAt = new Date().toISOString()) {
  const message = error instanceof Error ? error.message : String(error);
  return buildMarketDataSourceManifest([
    {
      artifact: 'prices.json',
      provider: 'Yahoo Finance',
      feed: 'chart/v8',
      provider_chain: ['Yahoo Finance'],
      primary_provider: 'Yahoo Finance',
      fallback_provider: null,
      source_mode: 'last_good',
      status: 'failed',
      fetched_at: generatedAt,
      latest_observation: null,
      row_count: 0,
      failure_reason: 'unknown',
      fallback_reason: message,
      notes: ['Current provider run failed; retained previous last-good public price artifact.'],
    },
    {
      artifact: 'prices_compact.json',
      provider: 'Yahoo Finance',
      feed: 'chart/v8',
      provider_chain: ['Yahoo Finance'],
      primary_provider: 'Yahoo Finance',
      fallback_provider: null,
      source_mode: 'last_good',
      status: 'failed',
      fetched_at: generatedAt,
      latest_observation: null,
      row_count: 0,
      failure_reason: 'unknown',
      fallback_reason: message,
      notes: ['Current provider run failed; retained previous last-good compact price artifact.'],
    },
  ], generatedAt);
}

async function runPythonModule(moduleName: string): Promise<void> {
  const { execFileSync } = await import('child_process');
  execFileSync(PYTHON_RUNTIME, ['-m', moduleName], {
    cwd: PROJECT_ROOT,
    stdio: 'inherit',
  });
}

export async function main() {
  console.log('=== Portfolio-Lab Data Fetcher ===\n');

  if (!existsSync(DATA_DIR)) {
    mkdirSync(DATA_DIR, { recursive: true });
  }

  // 1. Fetch price data (Yahoo Finance v8)
  console.log(`Fetching ${SYMBOLS.length} symbols from ${START_DATE} to ${END_DATE}...\n`);
  const priceResult = await fetchAllDataWithSummary(SYMBOLS, START_DATE, END_DATE);
  const priceData = priceResult.data;
  const missingSymbols = SYMBOLS.filter(symbol => !priceData[symbol]?.length);
  if (missingSymbols.length > 0) {
    throw new Error(`No price rows returned for configured symbols: ${missingSymbols.join(', ')}`);
  }

  // Convert to compact format: { symbol: [{d, p}, ...] }
  const compact: Record<string, { d: string; p: number }[]> = {};
  let totalDays = 0;
  for (const [symbol, prices] of Object.entries(priceData)) {
    compact[symbol] = prices.map(p => ({ d: p.date, p: p.adjClose }));
    totalDays += prices.length;
  }

  const pricesPath = join(DATA_DIR, 'prices.json');
  const pricesCompactPath = join(DATA_DIR, 'prices_compact.json');
  await writeJsonAtomic(pricesPath, compact);
  await writeJsonAtomic(pricesCompactPath, compact);
  console.log(`\nSaved ${Object.keys(compact).length} symbols (${totalDays} total data points) → ${pricesPath}`);
  console.log(`Saved compact price mirror → ${pricesCompactPath}`);

  // 2. Sync fetched prices into canonical SQLite store for dashboard freshness checks
  console.log('\nSyncing fetched prices into market.db...');
  await runPythonModule('src.data.market_db_sync');

  // 3. Fetch yield curve data (FRED)
  const yieldResult = await fetchYieldCurveDataWithSummary(START_DATE, END_DATE);
  const yieldData = yieldResult.data;
  const yieldsPath = join(DATA_DIR, 'yields.json');
  await writeJsonAtomic(yieldsPath, yieldData);
  console.log(`Saved ${yieldData.length} yield observations → ${yieldsPath}`);

  // 4. Publish source manifest for market data artifacts.
  const manifestPath = join(DATA_DIR, MARKET_DATA_SOURCE_MANIFEST_FILENAME);
  const manifestGeneratedAt = new Date().toISOString();
  const manifest = buildMarketDataSourceManifest([
    ...buildPriceSourceRows(priceResult.summary, compact, manifestGeneratedAt),
    buildYieldSourceRow(
      yieldResult.summary,
      yieldData.length,
      yieldData.length > 0 ? yieldData[yieldData.length - 1].date : null,
      manifestGeneratedAt,
    ),
  ], manifestGeneratedAt);
  await writeJsonAtomic(manifestPath, manifest);
  console.log(`Saved market data source manifest → ${manifestPath}`);

  // 5. Regenerate dashboard JSON
  console.log('\nRegenerating dashboard JSON...');
  try {
    await runPythonModule('src.dashboard.generator');
  } catch (e) {
    console.warn('Dashboard generator failed (Python runtime may not be available):', e);
  }

  console.log('\nDone.');
}

if (import.meta.main) {
  main().catch(async err => {
    console.error('Fetch failed:', err);
    try {
      if (!existsSync(DATA_DIR)) {
        mkdirSync(DATA_DIR, { recursive: true });
      }
      await writeJsonAtomic(
        join(DATA_DIR, MARKET_DATA_SOURCE_MANIFEST_FILENAME),
        buildLastGoodRetentionManifest(err),
      );
    } catch (manifestError) {
      console.error('Failed to write last-good retention manifest:', manifestError);
    }
    process.exit(1);
  });
}
