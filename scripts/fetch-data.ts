#!/usr/bin/env bun
/**
 * Fetch market data from Yahoo Finance + FRED.
 * Saves prices.json, prices_compact.json, and yields.json under PUBLIC_DATA_DIR
 * (default: repo public/data; tasker sets /var/www/portfolio-lab/data).
 *
 * Usage: bun run fetch-data
 *        PUBLIC_DATA_DIR=/var/www/portfolio-lab/data bun run fetch-data
 */

import {
  fetchAllDataWithSummary,
  fetchYieldCurveDataWithSummary,
  type FredCacheKey,
  type FredCacheRecord,
  type FredSeriesCache,
} from '../src/data/fetcher';
import { MARKET_DATA_SYMBOLS as SYMBOLS } from '../src/data/symbol_universe';
import {
  MARKET_DATA_SOURCE_MANIFEST_FILENAME,
  buildMarketDataSourceManifest,
  type MarketDataSourceRow,
} from '../src/data/source_manifest';
import {
  PRICE_DATA_QUALITY_FILENAME,
  buildPriceDataQualityReport,
  type PriceDataQualityReport,
} from '../src/data/price_quality';
import { dirname, join, resolve } from 'path';
import { existsSync, mkdirSync, readFileSync, renameSync, unlinkSync } from 'fs';
import { execFileSync } from 'child_process';

const PROJECT_ROOT = join(import.meta.dir, '..');
const PYTHON_RUNTIME = join(PROJECT_ROOT, 'scripts', 'python_runtime.sh');
const CRON_UPDATE_SCRIPT = join(PROJECT_ROOT, 'scripts', 'cron_update.py');
const FRED_CACHE_PATH = join(PROJECT_ROOT, 'data', 'fred_series_cache.json');
const START_DATE = '2005-01-01';
const END_DATE = new Date().toISOString().split('T')[0];
let atomicWriteCounter = 0;

/**
 * Resolve the public data write/read root for the data job.
 *
 * Tasker / live ops set PUBLIC_DATA_DIR=/var/www/portfolio-lab/data.
 * Offline and bare `bun scripts/fetch-data.ts` default to repo public/data.
 *
 * Inject `env` / `projectRoot` in tests; production uses process.env + PROJECT_ROOT.
 * `path.resolve(projectRoot, configured)` already returns absolute configured paths as-is.
 */
export function resolvePublicDataDir(options?: {
  env?: Record<string, string | undefined>;
  projectRoot?: string;
  livePublicDataDir?: string;
}): string {
  const env = options?.env ?? process.env;
  const projectRoot = options?.projectRoot ?? PROJECT_ROOT;
  const configured = env.PUBLIC_DATA_DIR?.trim();
  if (configured) {
    return resolve(projectRoot, configured);
  }

  // Match Python resolve_runtime_public_data_dir: when live WWW exists and is
  // distinct from repo public/data, prefer the operator tree so data jobs do
  // not refresh checkout prices while WWW signals lag on stale WWW prices.
  const allowRepo = ['1', 'true', 'yes', 'on'].includes(
    (env.PORTFOLIO_LAB_ALLOW_REPO_PUBLIC_DATA ?? '').trim().toLowerCase(),
  );
  const liveRoot = resolve(
    projectRoot,
    options?.livePublicDataDir
      ?? env.PORTFOLIO_LAB_LIVE_PUBLIC_DATA_DIR?.trim()
      ?? '/var/www/portfolio-lab/data',
  );
  const repoPublic = join(projectRoot, 'public', 'data');
  if (!allowRepo && existsSync(liveRoot) && resolve(liveRoot) !== resolve(repoPublic)) {
    return liveRoot;
  }
  return repoPublic;
}

/** Mutable for tests; production main() uses resolvePublicDataDir() once. */
export let DATA_DIR = resolvePublicDataDir();

export async function writeJsonAtomic(path: string, payload: unknown): Promise<void> {
  const tmpPath = `${path}.${process.pid}.${Date.now()}.${atomicWriteCounter++}.tmp`;
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

function priceQualitySourceSummary(
  report: PriceDataQualityReport,
): NonNullable<MarketDataSourceRow['data_quality']> {
  return {
    artifact: PRICE_DATA_QUALITY_FILENAME,
    schema_version: report.schema_version,
    generated_at: report.generated_at,
    status: report.overall_status,
    issue_counts: report.issue_counts,
  };
}

function unavailablePriceQualitySourceSummary(): NonNullable<MarketDataSourceRow['data_quality']> {
  return {
    artifact: PRICE_DATA_QUALITY_FILENAME,
    status: 'unavailable',
  };
}

export function buildPriceSourceRows(
  priceSummary: Awaited<ReturnType<typeof fetchAllDataWithSummary>>['summary'],
  compact: Record<string, { d: string; p: number }[]>,
  fetchedAt: string,
  qualityReport?: PriceDataQualityReport,
): MarketDataSourceRow[] {
  const rowCount = Object.values(compact).reduce((sum, rows) => sum + rows.length, 0);
  const latestObservation = latestObservationFromCompact(compact);
  const symbols = Object.keys(compact).sort();
  const failureReason = priceSummary.circuit_breaker.opened
    ? priceSummary.circuit_breaker.reason
    : firstFailureReason(priceSummary.failure_counts);
  const dataQuality = qualityReport === undefined ? undefined : priceQualitySourceSummary(qualityReport);
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
    ...(dataQuality === undefined ? {} : { data_quality: dataQuality }),
    notes: priceSummary.circuit_breaker.opened
      ? [`Skipped symbols after provider circuit breaker opened: ${priceSummary.circuit_breaker.skipped_symbols.join(', ')}`]
      : [],
  }));
}

function fredCacheKey(key: FredCacheKey): string {
  return `${key.seriesId}:${key.startDate}:${key.endDate}`;
}

function readFredCacheRecords(cachePath: string): Record<string, FredCacheRecord & {
  series_id?: string;
  start_date?: string;
  end_date?: string;
}> {
  if (!existsSync(cachePath)) return {};
  try {
    const payload = JSON.parse(readFileSync(cachePath, 'utf8'));
    return payload && typeof payload === 'object' && !Array.isArray(payload)
      ? payload as Record<string, FredCacheRecord>
      : {};
  } catch {
    return {};
  }
}

export function createFredDiskCache(cachePath: string = FRED_CACHE_PATH): FredSeriesCache {
  let writeQueue: Promise<void> = Promise.resolve();
  return {
    get: async (key) => {
      const record = readFredCacheRecords(cachePath)[fredCacheKey(key)];
      if (!record || !Array.isArray(record.observations) || typeof record.fetched_at !== 'string') {
        return null;
      }
      return {
        fetched_at: record.fetched_at,
        observations: record.observations,
      };
    },
    set: async (key, record) => {
      const writeRecord = async () => {
        mkdirSync(dirname(cachePath), { recursive: true });
        const records = readFredCacheRecords(cachePath);
        records[fredCacheKey(key)] = {
          series_id: key.seriesId,
          start_date: key.startDate,
          end_date: key.endDate,
          fetched_at: record.fetched_at,
          observations: record.observations,
        };
        await writeJsonAtomic(cachePath, records);
      };
      writeQueue = writeQueue.then(writeRecord, writeRecord);
      await writeQueue;
    },
  };
}

export function buildYieldSourceRow(
  yieldSummary: Awaited<ReturnType<typeof fetchYieldCurveDataWithSummary>>['summary'],
  rowCount: number,
  latestObservation: string | null,
  fetchedAt: string,
): MarketDataSourceRow {
  const failureReason = yieldSummary.series.find((series) => series.failure_reason)?.failure_reason ?? null;
  const fallbackReason = yieldSummary.series.find((series) => series.fallback_reason)?.fallback_reason
    ?? (yieldSummary.source_mode === 'synthetic' ? failureReason : null);
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
    fallback_reason: fallbackReason,
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
      data_quality: unavailablePriceQualitySourceSummary(),
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
      data_quality: unavailablePriceQualitySourceSummary(),
      notes: ['Current provider run failed; retained previous last-good compact price artifact.'],
    },
  ], generatedAt);
}

export class DashboardGenerationError extends Error {
  readonly preserveSourceManifest = true;

  constructor(cause: unknown) {
    const message = cause instanceof Error ? cause.message : String(cause);
    super(`Dashboard generation failed after market data refresh: ${message}`);
    this.name = 'DashboardGenerationError';
    this.cause = cause;
  }
}

export function shouldWriteLastGoodRetentionManifest(error: unknown): boolean {
  return !(error instanceof DashboardGenerationError);
}

async function runPythonModule(moduleName: string): Promise<void> {
  execFileSync(PYTHON_RUNTIME, ['-m', moduleName], {
    cwd: PROJECT_ROOT,
    stdio: 'inherit',
  });
}

/**
 * Record that dashboard artifacts were regenerated (any entrypoint).
 *
 * Data pipeline side-effects generator without running the dedicated
 * portfolio-lab-dashboard job; operators still need last_run honesty.
 */
export function recordDashboardCronStatus(options?: {
  status?: string;
  durationSeconds?: number;
  backend?: string;
  triggeredBy?: string;
  runUpdate?: (args: string[]) => void;
}): void {
  const status = options?.status ?? 'ok';
  const durationSeconds = options?.durationSeconds ?? 0;
  const backend =
    options?.backend ?? process.env.CRON_BACKEND ?? process.env.PORTFOLIO_LAB_CRON_BACKEND ?? 'tasker';
  const triggeredBy = options?.triggeredBy ?? 'fetch_data';
  const runUpdate =
    options?.runUpdate ??
    ((args: string[]) => {
      execFileSync(PYTHON_RUNTIME, args, {
        cwd: PROJECT_ROOT,
        stdio: 'inherit',
      });
    });

  runUpdate([
    CRON_UPDATE_SCRIPT,
    'portfolio-lab-dashboard',
    status,
    String(durationSeconds),
    backend,
    `triggered_by=${triggeredBy}`,
  ]);
}

export async function runDashboardGeneration(
  runModule: (moduleName: string) => Promise<void> = runPythonModule,
  recordStatus: typeof recordDashboardCronStatus = recordDashboardCronStatus,
): Promise<void> {
  const started = Date.now();
  try {
    await runModule('src.dashboard.generator');
  } catch (error) {
    throw new DashboardGenerationError(error);
  }
  const durationSeconds = Math.max(0, (Date.now() - started) / 1000);
  try {
    recordStatus({
      status: 'ok',
      durationSeconds,
      triggeredBy: 'fetch_data',
    });
  } catch (error) {
    // Status honesty must not fail a successful artifact write, but surface noise.
    console.error('Failed to update portfolio-lab-dashboard cron status:', error);
  }
}

export async function main() {
  console.log('=== Portfolio-Lab Data Fetcher ===\n');

  // Re-resolve at entry so tasker/env changes after import still apply.
  DATA_DIR = resolvePublicDataDir();
  console.log(`Public data dir: ${DATA_DIR}`);

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
  const priceQualityPath = join(DATA_DIR, PRICE_DATA_QUALITY_FILENAME);
  const priceQualityReport = buildPriceDataQualityReport(compact);
  await writeJsonAtomic(pricesPath, compact);
  await writeJsonAtomic(pricesCompactPath, compact);
  await writeJsonAtomic(priceQualityPath, priceQualityReport);
  console.log(`\nSaved ${Object.keys(compact).length} symbols (${totalDays} total data points) → ${pricesPath}`);
  console.log(`Saved compact price mirror → ${pricesCompactPath}`);
  console.log(`Saved price data quality audit → ${priceQualityPath}`);

  // 2. Sync fetched prices into canonical SQLite store for dashboard freshness checks
  console.log('\nSyncing fetched prices into market.db...');
  await runPythonModule('src.data.market_db_sync');

  // 3. Fetch yield curve data (FRED)
  const yieldResult = await fetchYieldCurveDataWithSummary(START_DATE, END_DATE, {
    cache: createFredDiskCache(),
    cacheTtlMs: 24 * 60 * 60 * 1000,
  });
  const yieldData = yieldResult.data;
  const yieldsPath = join(DATA_DIR, 'yields.json');
  await writeJsonAtomic(yieldsPath, yieldData);
  console.log(`Saved ${yieldData.length} yield observations → ${yieldsPath}`);

  // 4. Publish source manifest for market data artifacts.
  const manifestPath = join(DATA_DIR, MARKET_DATA_SOURCE_MANIFEST_FILENAME);
  const manifestGeneratedAt = new Date().toISOString();
  const manifest = buildMarketDataSourceManifest([
    ...buildPriceSourceRows(priceResult.summary, compact, manifestGeneratedAt, priceQualityReport),
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
  await runDashboardGeneration();

  console.log('\nDone.');
}

if (import.meta.main) {
  main().catch(async err => {
    console.error('Fetch failed:', err);
    try {
      if (shouldWriteLastGoodRetentionManifest(err)) {
        if (!existsSync(DATA_DIR)) {
          mkdirSync(DATA_DIR, { recursive: true });
        }
        await writeJsonAtomic(
          join(DATA_DIR, MARKET_DATA_SOURCE_MANIFEST_FILENAME),
          buildLastGoodRetentionManifest(err),
        );
      }
    } catch (manifestError) {
      console.error('Failed to write last-good retention manifest:', manifestError);
    }
    process.exit(1);
  });
}
