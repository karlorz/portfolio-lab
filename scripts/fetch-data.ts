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
  DEFAULT_ANOMALY_WHITELIST,
  DEFAULT_STALE_DATE_TOLERANCE_DAYS,
  PRICE_DATA_QUALITY_FILENAME,
  buildPriceDataQualityReport,
  type PriceDataQualityReport,
} from '../src/data/price_quality';
import { dirname, join, resolve } from 'path';
import {
  chmodSync,
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  unlinkSync,
} from 'fs';
import { execFileSync } from 'child_process';

const PROJECT_ROOT = join(import.meta.dir, '..');
const PYTHON_RUNTIME = join(PROJECT_ROOT, 'scripts', 'python_runtime.sh');
const CRON_UPDATE_SCRIPT = join(PROJECT_ROOT, 'scripts', 'cron_update.py');
const FRED_CACHE_PATH = join(PROJECT_ROOT, 'data', 'fred_series_cache.json');
/** Private operator twin of market artifacts (not the public SoT). */
const PRIVATE_DATA_DIR = join(PROJECT_ROOT, 'data');
const START_DATE = '2005-01-01';
const END_DATE = new Date().toISOString().split('T')[0];
let atomicWriteCounter = 0;

/** Market basenames dual-written to private data/ after public SoT write (Batch HY). */
export const PRIVATE_MARKET_SOFT_MIRROR_BASENAMES = [
  'prices.json',
  'prices_compact.json',
  'yields.json',
  PRICE_DATA_QUALITY_FILENAME,
  MARKET_DATA_SOURCE_MANIFEST_FILENAME,
] as const;

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
    // Batch HY: Caddy/operator trees need world-readable JSON (mkstemp-style
    // umask can leave 0600 → HTTPS 403 on public market artifacts).
    try {
      chmodSync(tmpPath, 0o644);
    } catch {
      // Best-effort; rename still proceeds.
    }
    renameSync(tmpPath, path);
    try {
      chmodSync(path, 0o644);
    } catch {
      // Best-effort post-replace mode normalize.
    }
  } catch (error) {
    try {
      unlinkSync(tmpPath);
    } catch {
      // Best-effort cleanup only.
    }
    throw error;
  }
}

/**
 * Soft-mirror market-data basenames from public SoT → private repo data/.
 *
 * Live authority remains signals.json.target_allocations only. Private
 * data/prices.json is a convenience twin for offline scripts / operator
 * inspection; PRICES_JSON still points at PUBLIC_DATA_DIR.
 *
 * Skips when source==dest (dev shells writing checkout public/data only) or
 * under pytest isolation paths (plab-pytest / pytest-of-*).
 */
export function softMirrorMarketArtifactsToPrivate(options?: {
  publicRoot?: string;
  privateRoot?: string;
  basenames?: readonly string[];
}): { copied: string[]; skipped: string[]; errors: string[] } {
  const publicRoot = options?.publicRoot ?? DATA_DIR;
  const privateRoot = options?.privateRoot ?? PRIVATE_DATA_DIR;
  const basenames = options?.basenames ?? PRIVATE_MARKET_SOFT_MIRROR_BASENAMES;
  const result = { copied: [] as string[], skipped: [] as string[], errors: [] as string[] };

  const pub = resolve(publicRoot);
  const priv = resolve(privateRoot);
  if (pub === priv) {
    result.skipped.push('same-root');
    return result;
  }
  // Never poison production private SSOT from hermetic test trees.
  const pubText = pub.replace(/\\/g, '/');
  if (
    pubText.includes('plab-pytest')
    || pubText.includes('/pytest-of-')
    || pubText.includes('/tmp/pytest-')
  ) {
    result.skipped.push('ephemeral-public-root');
    return result;
  }
  if (process.env.PYTEST_CURRENT_TEST && !process.env.PORTFOLIO_LAB_ALLOW_LIVE_PUBLIC) {
    // Bun unit tests do not set PYTEST; belt for mixed harnesses.
    result.skipped.push('pytest-guard');
    return result;
  }

  if (!existsSync(priv)) {
    mkdirSync(priv, { recursive: true });
  }

  for (const name of basenames) {
    const src = join(pub, name);
    const dst = join(priv, name);
    if (!existsSync(src)) {
      result.skipped.push(`${name}:missing-source`);
      continue;
    }
    try {
      copyFileSync(src, dst);
      try {
        chmodSync(dst, 0o644);
      } catch {
        // Best-effort mode.
      }
      result.copied.push(name);
    } catch (err) {
      result.errors.push(`${name}:${err instanceof Error ? err.message : String(err)}`);
    }
  }
  return result;
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

/** Default window for prices_compact.json (≈2 trading years). */
export const PRICES_COMPACT_DEFAULT_N_BARS = 504;

export type CompactPriceBar = { d: string; p: number };
export type CompactPriceSeriesMap = Record<string, CompactPriceBar[]>;

/**
 * Build last-N-bars compact payload (Batch BK).
 * Full history stays in prices.json; compact is a true window for CDN/dashboard.
 */
export function buildLastNBarsCompact(
  full: CompactPriceSeriesMap,
  nBars: number = PRICES_COMPACT_DEFAULT_N_BARS,
  options?: { asOf?: string; fullArtifact?: string },
): {
  meta: {
    schema: 'prices/compact-v1';
    as_of: string;
    n_bars: number;
    full_artifact: string;
    symbol_count: number;
    bar_count: number;
  };
  symbols: CompactPriceSeriesMap;
} {
  const limit = Number.isFinite(nBars) && nBars > 0 ? Math.trunc(nBars) : PRICES_COMPACT_DEFAULT_N_BARS;
  const symbols: CompactPriceSeriesMap = {};
  let barCount = 0;
  for (const [symbol, rows] of Object.entries(full)) {
    if (!Array.isArray(rows) || rows.length === 0) {
      symbols[symbol] = [];
      continue;
    }
    // Assume rows are chronological ascending (fetch path); take tail.
    const sliced = rows.length > limit ? rows.slice(-limit) : rows.slice();
    symbols[symbol] = sliced;
    barCount += sliced.length;
  }
  const asOf = options?.asOf
    ?? latestObservationFromCompact(symbols)
    ?? new Date().toISOString().slice(0, 10);
  return {
    meta: {
      schema: 'prices/compact-v1',
      as_of: asOf,
      n_bars: limit,
      full_artifact: options?.fullArtifact ?? 'prices.json',
      symbol_count: Object.keys(symbols).length,
      bar_count: barCount,
    },
    symbols,
  };
}

/** Env override: PRICES_COMPACT_N_BARS (positive int). */
export function resolvePricesCompactNBars(
  env: Record<string, string | undefined> = process.env as Record<string, string | undefined>,
): number {
  const raw = env.PRICES_COMPACT_N_BARS;
  if (raw === undefined || raw.trim() === '') {
    return PRICES_COMPACT_DEFAULT_N_BARS;
  }
  const n = Number(raw);
  if (!Number.isFinite(n) || n <= 0) {
    return PRICES_COMPACT_DEFAULT_N_BARS;
  }
  return Math.trunc(n);
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
  fullPrices: CompactPriceSeriesMap,
  fetchedAt: string,
  qualityReport?: PriceDataQualityReport,
  compactMeta?: { n_bars: number; bar_count: number },
): MarketDataSourceRow[] {
  const fullRowCount = Object.values(fullPrices).reduce((sum, rows) => sum + rows.length, 0);
  const latestObservation = latestObservationFromCompact(fullPrices);
  const symbols = Object.keys(fullPrices).sort();
  const failureReason = priceSummary.circuit_breaker.opened
    ? priceSummary.circuit_breaker.reason
    : firstFailureReason(priceSummary.failure_counts);
  const dataQuality = qualityReport === undefined ? undefined : priceQualitySourceSummary(qualityReport);
  const qualityStatus = qualityReport?.overall_status;
  const rowStatus = priceManifestStatusFromQuality(priceSummary.status, qualityStatus);
  const qualityNotes: string[] = [];
  if (qualityStatus === 'fail') {
    qualityNotes.push(
      `Price data quality overall_status=fail; manifest status forced to failed (was provider status ${priceSummary.status}).`,
    );
  } else if (qualityStatus === 'warn' && priceSummary.status === 'success') {
    qualityNotes.push('Price data quality overall_status=warn; manifest status degraded.');
  }
  const compactRowCount = compactMeta?.bar_count ?? fullRowCount;
  const compactNotes = [
    ...qualityNotes,
    compactMeta
      ? `prices_compact is last-${compactMeta.n_bars} bars/symbol (not a full-history mirror)`
      : 'prices_compact last-N window',
  ];
  return [
    {
      artifact: 'prices.json',
      provider: priceSummary.provider,
      feed: priceSummary.feed,
      provider_chain: priceSummary.provider_chain,
      primary_provider: priceSummary.primary_provider,
      fallback_provider: priceSummary.fallback_provider,
      source_mode: priceSummary.source_mode,
      status: rowStatus,
      fetched_at: fetchedAt,
      latest_observation: latestObservation,
      row_count: fullRowCount,
      symbols,
      failure_reason: failureReason ?? (qualityStatus === 'fail' ? 'price_data_quality_fail' : null),
      ...(dataQuality === undefined ? {} : { data_quality: dataQuality }),
      notes: [
        ...(priceSummary.circuit_breaker.opened
          ? [`Skipped symbols after provider circuit breaker opened: ${priceSummary.circuit_breaker.skipped_symbols.join(', ')}`]
          : []),
        ...qualityNotes,
        'full multi-year history for market.db / backtest archive',
      ],
    },
    {
      artifact: 'prices_compact.json',
      provider: priceSummary.provider,
      feed: priceSummary.feed,
      provider_chain: priceSummary.provider_chain,
      primary_provider: priceSummary.primary_provider,
      fallback_provider: priceSummary.fallback_provider,
      source_mode: priceSummary.source_mode,
      status: rowStatus,
      fetched_at: fetchedAt,
      latest_observation: latestObservation,
      row_count: compactRowCount,
      symbols,
      failure_reason: failureReason ?? (qualityStatus === 'fail' ? 'price_data_quality_fail' : null),
      ...(dataQuality === undefined ? {} : { data_quality: dataQuality }),
      notes: [
        ...(priceSummary.circuit_breaker.opened
          ? [`Skipped symbols after provider circuit breaker opened: ${priceSummary.circuit_breaker.skipped_symbols.join(', ')}`]
          : []),
        ...compactNotes,
      ],
    },
  ];
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

/**
 * Raised when prices were written but data_quality overall_status is fail.
 * Artifacts remain on disk for operators; the data job must not exit 0.
 */
export class PriceDataQualityGateError extends Error {
  readonly qualityStatus: string;
  readonly issueCounts: PriceDataQualityReport['issue_counts'];

  constructor(report: PriceDataQualityReport) {
    const failSymbols = (report.symbols || [])
      .filter((s) => s.status === 'fail')
      .map((s) => s.symbol)
      .slice(0, 8);
    const symbolHint = failSymbols.length
      ? ` failing symbols: ${failSymbols.join(', ')}${report.symbols.filter((s) => s.status === 'fail').length > 8 ? '…' : ''}`
      : '';
    super(
      `Price data quality gate failed (overall_status=${report.overall_status}).` +
        ` issue_counts=${JSON.stringify(report.issue_counts)}.${symbolHint}` +
        ` Artifacts written; job must not report success.`,
    );
    this.name = 'PriceDataQualityGateError';
    this.qualityStatus = report.overall_status;
    this.issueCounts = report.issue_counts;
  }
}

export function shouldWriteLastGoodRetentionManifest(error: unknown): boolean {
  // Quality-gate failures already wrote live prices/quality; do not overwrite
  // with last-good retention. Dashboard gen failures also preserve current tree.
  return !(
    error instanceof DashboardGenerationError
    || error instanceof PriceDataQualityGateError
  );
}

/** Fail-closed gate used by the data job after writing data_quality.json. */
export function assertPriceQualityAllowsSuccess(report: PriceDataQualityReport): void {
  if (report.overall_status === 'fail') {
    throw new PriceDataQualityGateError(report);
  }
}

/** Couple source_manifest price row status to nested data_quality.status. */
export function priceManifestStatusFromQuality(
  providerStatus: MarketDataSourceRow['status'],
  qualityStatus: PriceDataQualityReport['overall_status'] | undefined,
): MarketDataSourceRow['status'] {
  if (qualityStatus === 'fail') {
    return 'failed';
  }
  if (qualityStatus === 'warn' && providerStatus === 'success') {
    return 'degraded';
  }
  return providerStatus;
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

  // Full history: { symbol: [{d, p}, ...] } → prices.json (canonical for market.db)
  const fullPrices: CompactPriceSeriesMap = {};
  let totalDays = 0;
  for (const [symbol, prices] of Object.entries(priceData)) {
    fullPrices[symbol] = prices.map(p => ({ d: p.date, p: p.adjClose }));
    totalDays += prices.length;
  }

  // Batch BK: prices_compact is last-N bars only (not a second full archive)
  const compactNBars = resolvePricesCompactNBars();
  const pricesCompactPayload = buildLastNBarsCompact(fullPrices, compactNBars, {
    fullArtifact: 'prices.json',
  });

  const pricesPath = join(DATA_DIR, 'prices.json');
  const pricesCompactPath = join(DATA_DIR, 'prices_compact.json');
  const priceQualityPath = join(DATA_DIR, PRICE_DATA_QUALITY_FILENAME);
  // Quality audit runs on full history (operators care about full-book anomalies)
  const priceQualityReport = buildPriceDataQualityReport(fullPrices, undefined, {
    anomalyWhitelist: DEFAULT_ANOMALY_WHITELIST,
    staleDateToleranceDays: DEFAULT_STALE_DATE_TOLERANCE_DAYS,
  });
  await writeJsonAtomic(pricesPath, fullPrices);
  await writeJsonAtomic(pricesCompactPath, pricesCompactPayload);
  await writeJsonAtomic(priceQualityPath, priceQualityReport);
  console.log(`\nSaved ${Object.keys(fullPrices).length} symbols (${totalDays} total data points) → ${pricesPath}`);
  console.log(
    `Saved prices_compact last-${compactNBars} bars `
    + `(${pricesCompactPayload.meta.bar_count} points) → ${pricesCompactPath}`,
  );
  console.log(
    `Saved price data quality audit → ${priceQualityPath} (overall_status=${priceQualityReport.overall_status})`,
  );

  // 2. Sync fetched prices into canonical SQLite store for dashboard freshness checks
  console.log('\nSyncing fetched prices into market.db...');
  await runPythonModule('src.data.market_db_sync');

  // 2b. Rebuild vix_term_structure.json from the freshly-synced market.db so
  // the derived VIX history archive stays current with ^VIX/^VIX3M rows.
  // Without this the JSON freezes at the last manual generation while the
  // term-structure signal reads stale levels (FILE_STALE_DAYS fallback never
  // trips for a 1-2 day gap). Script reads market.db only; no network.
  console.log('\nRebuilding vix_term_structure.json from market.db...');
  try {
    execFileSync(PYTHON_RUNTIME, ['scripts/update_vix_term_structure.py'], {
      cwd: PROJECT_ROOT,
      stdio: 'inherit',
      env: {
        ...process.env,
        // Keep PUBLIC_DATA_DIR aligned with the tree fetch-data wrote so the
        // dual-write lands in the operator-visible public data dir.
        PUBLIC_DATA_DIR: DATA_DIR,
      },
    });
  } catch (vixError) {
    // Soft: dashboard gen still runs; term-structure signal has its market.db
    // fallback. Surface noise so operators notice a broken rebuild.
    console.error('vix_term_structure.json rebuild after market_db_sync failed:', vixError);
  }

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
    ...buildPriceSourceRows(
      priceResult.summary,
      fullPrices,
      manifestGeneratedAt,
      priceQualityReport,
      {
        n_bars: pricesCompactPayload.meta.n_bars,
        bar_count: pricesCompactPayload.meta.bar_count,
      },
    ),
    buildYieldSourceRow(
      yieldResult.summary,
      yieldData.length,
      yieldData.length > 0 ? yieldData[yieldData.length - 1].date : null,
      manifestGeneratedAt,
    ),
  ], manifestGeneratedAt);
  await writeJsonAtomic(manifestPath, manifest);
  console.log(`Saved market data source manifest → ${manifestPath}`);

  // Batch HY: soft-mirror market artifacts to private data/ so offline
  // operator twins (data/prices.json) do not freeze at last ad-hoc copy.
  // Public SoT remains PUBLIC_DATA_DIR; live routing still ignores prices.
  const privateMirror = softMirrorMarketArtifactsToPrivate({
    publicRoot: DATA_DIR,
    privateRoot: PRIVATE_DATA_DIR,
  });
  if (privateMirror.copied.length > 0) {
    console.log(
      `Soft-mirrored market artifacts → ${PRIVATE_DATA_DIR}: ${privateMirror.copied.join(', ')}`,
    );
  }
  if (privateMirror.errors.length > 0) {
    console.error('Private market soft-mirror errors:', privateMirror.errors);
  }

  // Batch BY: rebuild index.json immediately after source_manifest so
  // data_pipeline_slo never sees stale_index if dashboard gen lags/fails.
  console.log('\nRefreshing public data index (post source_manifest)...');
  try {
    execFileSync(PYTHON_RUNTIME, [
      'scripts/refresh_public_data_index.py',
      '--reason',
      'source_manifest',
    ], {
      cwd: PROJECT_ROOT,
      stdio: 'inherit',
      env: {
        ...process.env,
        // Keep PUBLIC_DATA_DIR pointing at the same tree fetch-data wrote
        PUBLIC_DATA_DIR: DATA_DIR,
      },
    });
  } catch (indexError) {
    // Soft: dashboard gen still rebuilds index; surface noise for operators.
    console.error('Public index refresh after source_manifest failed:', indexError);
  }

  // Fail-closed quality gate after artifacts are written (operators can inspect
  // data_quality.json / source_manifest) but before claiming job success.
  assertPriceQualityAllowsSuccess(priceQualityReport);

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
