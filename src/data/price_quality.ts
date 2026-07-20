export const PRICE_DATA_QUALITY_SCHEMA_VERSION = 'price-data-quality/v1';
export const PRICE_DATA_QUALITY_FILENAME = 'data_quality.json';

export type PriceDataQualityStatus = 'ok' | 'warn' | 'fail';
export type PriceReturnAnomalySeverity = 'warning' | 'critical';
export type PriceReturnAnomalyType = 'split_like_return' | 'extreme_return';

export interface PriceIssueCounts {
  duplicate_dates: number;
  empty_symbols: number;
  extreme_returns: number;
  internal_gaps: number;
  invalid_dates: number;
  invalid_prices: number;
  missing_required_keys: number;
  non_monotonic_rows: number;
  non_object_records: number;
  split_like_returns: number;
  stale_latest_dates: number;
  total: number;
}

export interface PriceDataQualityIssue {
  index: number;
}

export interface MissingRequiredKeysIssue extends PriceDataQualityIssue {
  missing_keys: string[];
}

export interface InvalidPriceIssue extends PriceDataQualityIssue {
  date: string | null;
}

export interface NonMonotonicRowIssue extends PriceDataQualityIssue {
  previous_date: string;
  date: string;
}

export interface InternalGapIssue {
  missing_count: number;
  sample_missing_dates: string[];
}

export interface StaleLatestDateIssue {
  reference_date: string;
  latest_date: string | null;
}

export interface ReturnAnomalyIssue {
  type: PriceReturnAnomalyType;
  severity: PriceReturnAnomalySeverity;
  symbol: string;
  date: string;
  previous_date: string;
  previous_price: number;
  current_price: number;
  return_pct: number;
}

export interface PriceSymbolQualitySummary {
  symbol: string;
  status: PriceDataQualityStatus;
  row_count: number;
  first_date: string | null;
  latest_date: string | null;
  duplicate_date_count: number;
  duplicate_dates: string[];
  internal_gaps: InternalGapIssue[];
  invalid_dates: PriceDataQualityIssue[];
  invalid_prices: InvalidPriceIssue[];
  latest_lag_days: number;
  missing_required_keys: MissingRequiredKeysIssue[];
  non_monotonic_rows: NonMonotonicRowIssue[];
  non_object_records: PriceDataQualityIssue[];
  return_anomaly_count: number;
  return_anomalies: ReturnAnomalyIssue[];
  stale_latest_date: StaleLatestDateIssue | null;
}

export interface PriceDataQualityReport {
  schema_version: typeof PRICE_DATA_QUALITY_SCHEMA_VERSION;
  generated_at: string;
  overall_status: PriceDataQualityStatus;
  issue_counts: PriceIssueCounts;
  symbols: PriceSymbolQualitySummary[];
}

export interface PriceDataQualityOptions {
  criticalReturnPct?: number;
  referenceSymbol?: string;
  maxDuplicateDateSamples?: number;
  maxLatestLagDays?: number;
  maxMissingDateSamples?: number;
  maxReturnAnomalySamples?: number;
  splitLikeReturnPct?: number;
  /**
   * Symbols whose latest-bar lag vs the reference calendar is advisory only.
   * Yahoo often null-pads sparse VIX-family indices after the last real print
   * while SPY keeps advancing; blocking the whole data job for that lag is wrong.
   */
  sparseIndexSymbols?: readonly string[];
  /**
   * Wall-clock as-of (ISO string or Date). Used to detect universe-wide freeze
   * when all symbols share the same latest bar (peer calendar lag stays 0).
   * Defaults to generatedAt of the report when not set.
   */
  asOfDate?: string | Date;
}

const EMPTY_ISSUE_COUNTS: PriceIssueCounts = {
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
};

interface SymbolAudit {
  summary: PriceSymbolQualitySummary;
  valid_dates: string[];
  valid_prices: PriceObservation[];
}

interface PriceObservation {
  date: string;
  index: number;
  price: number;
}

type EnvHost = typeof globalThis & {
  Bun?: { env?: Record<string, string | undefined> };
  process?: { env?: Record<string, string | undefined> };
};

const DEFAULT_CRITICAL_RETURN_PCT = 90;
const DEFAULT_MAX_RETURN_ANOMALY_SAMPLES = 5;
const DEFAULT_SPLIT_LIKE_RETURN_PCT = 40;

/** Yahoo-sparse indexes: lag vs SPY is advisory, not a data-job hard fail. */
const DEFAULT_SPARSE_INDEX_SYMBOLS: readonly string[] = [
  '^VIX3M',
  '^VIX',
  '^VIX6M',
  'VIX3M',
  'VIX',
  'VIX6M',
];

function isSparseIndexSymbol(
  symbol: string,
  options: PriceDataQualityOptions,
): boolean {
  const configured = options.sparseIndexSymbols ?? DEFAULT_SPARSE_INDEX_SYMBOLS;
  const normalized = symbol.trim().toUpperCase();
  return configured.some((candidate) => candidate.trim().toUpperCase() === normalized);
}

function readEnvNumber(name: string): number | undefined {
  const host = globalThis as EnvHost;
  const raw = host.Bun?.env?.[name] ?? host.process?.env?.[name];
  if (raw === undefined || raw.trim() === '') {
    return undefined;
  }
  const value = Number(raw);
  return Number.isFinite(value) && value > 0 ? value : undefined;
}

function positiveOption(
  value: number | undefined,
  envName: string,
  fallback: number,
): number {
  if (value !== undefined && Number.isFinite(value) && value > 0) {
    return value;
  }
  return readEnvNumber(envName) ?? fallback;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isValidIsoDate(value: unknown): value is string {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    return false;
  }
  const date = new Date(`${value}T00:00:00Z`);
  return !Number.isNaN(date.valueOf()) && date.toISOString().slice(0, 10) === value;
}

function isValidPrice(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0;
}

function increment(counts: PriceIssueCounts, key: Exclude<keyof PriceIssueCounts, 'total'>): void {
  counts[key] += 1;
  counts.total += 1;
}

function emptySymbolSummary(symbol: string): PriceSymbolQualitySummary {
  return {
    symbol,
    status: 'ok',
    row_count: 0,
    first_date: null,
    latest_date: null,
    duplicate_date_count: 0,
    duplicate_dates: [],
    internal_gaps: [],
    invalid_dates: [],
    invalid_prices: [],
    latest_lag_days: 0,
    missing_required_keys: [],
    non_monotonic_rows: [],
    non_object_records: [],
    return_anomaly_count: 0,
    return_anomalies: [],
    stale_latest_date: null,
  };
}

function summarizeSymbol(
  symbol: string,
  rows: unknown[],
  counts: PriceIssueCounts,
  options: PriceDataQualityOptions,
): SymbolAudit {
  const summary = emptySymbolSummary(symbol);
  const validDates: string[] = [];
  const validPrices: PriceObservation[] = [];
  summary.row_count = rows.length;

  if (rows.length === 0) {
    increment(counts, 'empty_symbols');
  }

  const seenDates = new Set<string>();
  const duplicateDates = new Set<string>();
  let previousDate: string | null = null;

  rows.forEach((row, index) => {
    if (!isRecord(row)) {
      summary.non_object_records.push({ index });
      increment(counts, 'non_object_records');
      return;
    }

    const missingKeys = ['d', 'p'].filter((key) => !(key in row));
    if (missingKeys.length > 0) {
      summary.missing_required_keys.push({ index, missing_keys: missingKeys });
      increment(counts, 'missing_required_keys');
      return;
    }

    const date = row.d;
    const price = row.p;

    if (!isValidIsoDate(date)) {
      summary.invalid_dates.push({ index });
      increment(counts, 'invalid_dates');
      return;
    }

    validDates.push(date);

    if (previousDate !== null && date < previousDate) {
      summary.non_monotonic_rows.push({ index, previous_date: previousDate, date });
      increment(counts, 'non_monotonic_rows');
    }
    previousDate = date;

    if (seenDates.has(date)) {
      duplicateDates.add(date);
    }
    seenDates.add(date);

    if (summary.first_date === null || date < summary.first_date) {
      summary.first_date = date;
    }
    if (summary.latest_date === null || date > summary.latest_date) {
      summary.latest_date = date;
    }

    if (!isValidPrice(price)) {
      summary.invalid_prices.push({ index, date });
      increment(counts, 'invalid_prices');
    } else {
      validPrices.push({ date, index, price });
    }
  });

  const duplicateDateList = [...duplicateDates].sort();
  const maxDuplicateDateSamples = Math.max(1, options.maxDuplicateDateSamples ?? 5);
  summary.duplicate_date_count = duplicateDateList.length;
  summary.duplicate_dates = duplicateDateList.slice(0, maxDuplicateDateSamples);
  counts.duplicate_dates += summary.duplicate_date_count;
  counts.total += summary.duplicate_date_count;

  return { summary, valid_dates: validDates, valid_prices: validPrices };
}

function updateStatus(
  summary: PriceSymbolQualitySummary,
  hasCriticalReturnAnomaly = summary.return_anomalies.some((issue) => issue.severity === 'critical'),
  options: PriceDataQualityOptions = {},
): void {
  // Reference-calendar internal gaps are advisory: sparse index series such as
  // ^VIX3M legitimately omit SPY trading days while still being current at the
  // latest bar. Do not fail the data job for mid-history holes alone.
  // Sparse-index latest lag (Yahoo null-pad after last real bar) is also advisory.
  const sparseIndexStaleIsAdvisory = summary.stale_latest_date !== null
    && isSparseIndexSymbol(summary.symbol, options);
  const staleIsBlocking = summary.stale_latest_date !== null
    && !sparseIndexStaleIsAdvisory;
  const hasBlockingIssues = (
    summary.duplicate_date_count > 0
    || summary.invalid_dates.length > 0
    || summary.invalid_prices.length > 0
    || summary.missing_required_keys.length > 0
    || summary.non_monotonic_rows.length > 0
    || summary.non_object_records.length > 0
    || staleIsBlocking
    || summary.row_count === 0
    || hasCriticalReturnAnomaly
  );
  if (hasBlockingIssues) {
    summary.status = 'fail';
    return;
  }
  const hasAdvisoryIssues = (
    summary.return_anomalies.length > 0
    || summary.internal_gaps.length > 0
    || sparseIndexStaleIsAdvisory
  );
  summary.status = hasAdvisoryIssues ? 'warn' : 'ok';
}

function uniqueSortedDates(dates: string[]): string[] {
  return [...new Set(dates)].sort();
}

function computeReturnPct(previousPrice: number, currentPrice: number): number {
  return Number((((currentPrice - previousPrice) / previousPrice) * 100).toFixed(4));
}

function applyReturnAnomalyChecks(
  audits: SymbolAudit[],
  counts: PriceIssueCounts,
  options: PriceDataQualityOptions = {},
): void {
  const criticalReturnPct = positiveOption(
    options.criticalReturnPct,
    'PRICE_QUALITY_CRITICAL_RETURN_PCT',
    DEFAULT_CRITICAL_RETURN_PCT,
  );
  const splitLikeReturnPct = positiveOption(
    options.splitLikeReturnPct,
    'PRICE_QUALITY_SPLIT_LIKE_RETURN_PCT',
    DEFAULT_SPLIT_LIKE_RETURN_PCT,
  );
  const maxReturnAnomalySamples = Math.max(1, Math.trunc(positiveOption(
    options.maxReturnAnomalySamples,
    'PRICE_QUALITY_MAX_RETURN_ANOMALY_SAMPLES',
    DEFAULT_MAX_RETURN_ANOMALY_SAMPLES,
  )));

  for (const audit of audits) {
    const sorted = [...audit.valid_prices].sort((left, right) => (
      left.date.localeCompare(right.date) || left.index - right.index
    ));
    const anomalies: ReturnAnomalyIssue[] = [];

    for (let index = 1; index < sorted.length; index += 1) {
      const previous = sorted[index - 1];
      const current = sorted[index];
      if (current.date === previous.date) {
        continue;
      }

      const returnPct = computeReturnPct(previous.price, current.price);
      const absoluteReturnPct = Math.abs(returnPct);
      if (absoluteReturnPct >= criticalReturnPct) {
        anomalies.push({
          type: 'extreme_return',
          severity: 'critical',
          symbol: audit.summary.symbol,
          date: current.date,
          previous_date: previous.date,
          previous_price: previous.price,
          current_price: current.price,
          return_pct: returnPct,
        });
        counts.extreme_returns += 1;
        counts.total += 1;
      } else if (absoluteReturnPct >= splitLikeReturnPct) {
        anomalies.push({
          type: 'split_like_return',
          severity: 'warning',
          symbol: audit.summary.symbol,
          date: current.date,
          previous_date: previous.date,
          previous_price: previous.price,
          current_price: current.price,
          return_pct: returnPct,
        });
        counts.split_like_returns += 1;
        counts.total += 1;
      }
    }

    const hasCriticalReturnAnomaly = anomalies.some((issue) => issue.severity === 'critical');
    audit.summary.return_anomaly_count = anomalies.length;
    audit.summary.return_anomalies = anomalies.slice(0, maxReturnAnomalySamples);
    updateStatus(audit.summary, hasCriticalReturnAnomaly, options);
  }
}

function hasBlockingIssueCounts(counts: PriceIssueCounts): boolean {
  // internal_gaps / split_like_returns are counted but advisory (overall warn).
  return counts.duplicate_dates > 0
    || counts.empty_symbols > 0
    || counts.extreme_returns > 0
    || counts.invalid_dates > 0
    || counts.invalid_prices > 0
    || counts.missing_required_keys > 0
    || counts.non_monotonic_rows > 0
    || counts.non_object_records > 0
    || counts.stale_latest_dates > 0;
}


/** YYYY-MM-DD in UTC from Date or ISO string. */
export function toUtcDateString(value: string | Date): string {
  const d = typeof value === 'string' ? new Date(value) : value;
  if (Number.isNaN(d.getTime())) {
    return '';
  }
  return d.toISOString().slice(0, 10);
}

/** Count weekdays (Mon–Fri) strictly after startDate up to and including endDate (both YYYY-MM-DD). */
export function countWeekdaysBetween(startDate: string, endDate: string): number {
  if (!startDate || !endDate || endDate <= startDate) {
    return 0;
  }
  const start = new Date(`${startDate}T00:00:00Z`);
  const end = new Date(`${endDate}T00:00:00Z`);
  if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) {
    return 0;
  }
  let count = 0;
  const cur = new Date(start);
  cur.setUTCDate(cur.getUTCDate() + 1);
  while (cur <= end) {
    const dow = cur.getUTCDay(); // 0 Sun .. 6 Sat
    if (dow !== 0 && dow !== 6) {
      count += 1;
    }
    cur.setUTCDate(cur.getUTCDate() + 1);
  }
  return count;
}

function applyReferenceCalendarChecks(
  audits: SymbolAudit[],
  counts: PriceIssueCounts,
  options: PriceDataQualityOptions,
): void {
  const referenceSymbol = options.referenceSymbol ?? 'SPY';
  const referenceAudit = audits.find((audit) => (
    audit.summary.symbol === referenceSymbol && audit.valid_dates.length > 0
  )) ?? audits.find((audit) => audit.valid_dates.length > 0);
  if (!referenceAudit) {
    return;
  }

  const referenceDates = uniqueSortedDates(referenceAudit.valid_dates);
  const referenceLatestDate = referenceDates[referenceDates.length - 1] ?? null;
  if (referenceLatestDate === null) {
    return;
  }

  const maxLatestLagDays = options.maxLatestLagDays ?? 1;
  const maxMissingDateSamples = Math.max(1, options.maxMissingDateSamples ?? 5);

  for (const audit of audits) {
    const { summary } = audit;
    if (summary.first_date === null || summary.latest_date === null) {
      continue;
    }
    const firstDate = summary.first_date;
    const latestDate = summary.latest_date;

    const dateSet = new Set(audit.valid_dates);
    const missingDates = referenceDates.filter((date) => (
      date > firstDate
      && date < latestDate
      && !dateSet.has(date)
    ));
    if (missingDates.length > 0) {
      summary.internal_gaps.push({
        missing_count: missingDates.length,
        sample_missing_dates: missingDates.slice(0, maxMissingDateSamples),
      });
      increment(counts, 'internal_gaps');
    }

    const peerLagDays = referenceDates.filter((date) => date > latestDate).length;
    // Wall-clock trading lag: when the whole universe freezes together, peer lag
    // stays 0; compare latest bar to as-of weekday calendar instead.
    const asOfRaw = options.asOfDate;
    const asOfStr = asOfRaw
      ? toUtcDateString(asOfRaw)
      : '';
    const wallClockLagDays = asOfStr
      ? countWeekdaysBetween(latestDate, asOfStr)
      : 0;
    summary.latest_lag_days = Math.max(peerLagDays, wallClockLagDays);
    const sparseIndex = isSparseIndexSymbol(summary.symbol, options);
    if (summary.latest_lag_days > maxLatestLagDays) {
      const refDate = wallClockLagDays > peerLagDays && asOfStr
        ? asOfStr
        : referenceLatestDate;
      summary.stale_latest_date = {
        reference_date: refDate,
        latest_date: summary.latest_date,
      };
      // Keep visibility on the symbol row, but do not increment the blocking
      // stale_latest_dates counter for known sparse indexes (Yahoo null-pad lag).
      // Still bump total so overall_status becomes warn (not silent ok).
      if (!sparseIndex) {
        increment(counts, 'stale_latest_dates');
      } else {
        counts.total += 1;
      }
    }

    updateStatus(summary, undefined, options);
  }
}

export function buildPriceDataQualityReport(
  payload: unknown,
  generatedAt: string = new Date().toISOString(),
  options: PriceDataQualityOptions = {},
): PriceDataQualityReport {
  const counts: PriceIssueCounts = { ...EMPTY_ISSUE_COUNTS };
  const entries = isRecord(payload)
    ? Object.entries(payload).filter(([symbol]) => symbol.length > 0)
    : [];

  const audits = entries
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([symbol, rows]) => summarizeSymbol(
      symbol,
      Array.isArray(rows) ? rows : [],
      counts,
      options,
    ));
  const asOfOptions: PriceDataQualityOptions = {
    ...options,
    asOfDate: options.asOfDate ?? generatedAt,
  };
  applyReferenceCalendarChecks(audits, counts, asOfOptions);
  applyReturnAnomalyChecks(audits, counts, options);

  return {
    schema_version: PRICE_DATA_QUALITY_SCHEMA_VERSION,
    generated_at: generatedAt,
    overall_status: hasBlockingIssueCounts(counts)
      ? 'fail'
      : counts.total > 0 ? 'warn' : 'ok',
    issue_counts: counts,
    symbols: audits.map((audit) => audit.summary),
  };
}
