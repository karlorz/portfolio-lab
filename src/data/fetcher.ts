/**
 * Data Fetcher - Yahoo Finance v8 Chart API + FRED Yield Data
 * Uses the chart endpoint (no API key required) for prices
 * FRED API for Treasury yield curve data
 */

import type { MarketDataSourceMode, MarketDataSourceStatus } from './source_manifest';

export interface HistoricalPrice {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  adjClose: number;
  volume: number;
}

export interface TreasuryYield {
  date: string;
  dgs2: number;   // 2-Year Treasury Yield
  dgs10: number;  // 10-Year Treasury Yield
  dgs30: number;  // 30-Year Treasury Yield
  spread2s10s: number;  // 2s10s spread (basis points)
  spread10s30s: number; // 10s30s spread (basis points)
}

export type YahooFetchFailureReason =
  | 'rate_limited'
  | 'timeout'
  | 'no_data'
  | 'malformed_payload'
  | 'network_error'
  | 'unknown';

export type MarketDataFetchFailureReason =
  | YahooFetchFailureReason
  | 'provider_unavailable'
  | 'not_configured';

export interface YahooFetchOptions {
  fetchImpl?: typeof fetch;
  timeoutMs?: number;
  maxAttempts?: number;
  backoffMs?: number;
}

export interface MarketDataProvider {
  name: string;
  feed: string;
  sourceMode?: MarketDataSourceMode;
  fetchSymbol: (symbol: string, startDate: string, endDate: string) => Promise<HistoricalPrice[]>;
}

export interface FetchAllDataOptions extends YahooFetchOptions {
  delayMs?: number;
  circuitBreakerFailureThreshold?: number;
  providers?: MarketDataProvider[];
}

export interface SymbolFetchSummary {
  symbol: string;
  provider: string | null;
  feed: string | null;
  provider_chain: string[];
  primary_provider: string | null;
  fallback_provider: string | null;
  status: MarketDataSourceStatus;
  source_mode: MarketDataSourceMode;
  rows: number;
  latest_observation: string | null;
  attempts: number;
  fetched_at: string;
  failure_reason?: MarketDataFetchFailureReason;
  fallback_reason?: MarketDataFetchFailureReason;
  error?: string;
}

export interface MarketDataProviderSummary {
  provider: string;
  feed: string;
  provider_chain: string[];
  primary_provider: string | null;
  fallback_provider: string | null;
  status: MarketDataSourceStatus;
  source_mode: MarketDataSourceMode;
  fetched_at: string;
  symbols: SymbolFetchSummary[];
  failure_counts: Partial<Record<MarketDataFetchFailureReason, number>>;
  circuit_breaker: {
    opened: boolean;
    reason: MarketDataFetchFailureReason | null;
    skipped_symbols: string[];
  };
}

export type YahooProviderSummary = MarketDataProviderSummary;

export interface FetchAllDataResult {
  data: { [symbol: string]: HistoricalPrice[] };
  summary: MarketDataProviderSummary;
}

export type FredFailureReason =
  | 'missing_api_key'
  | 'rate_limited'
  | 'timeout'
  | 'malformed_payload'
  | 'network_error'
  | 'api_error'
  | 'unknown';

export interface FredFetchOptions {
  fetchImpl?: typeof fetch;
}

export interface FredSeriesSummary {
  series_id: string;
  status: MarketDataSourceStatus;
  source_mode: MarketDataSourceMode;
  rows: number;
  latest_observation: string | null;
  fetched_at: string;
  failure_reason?: FredFailureReason;
}

export interface FredYieldSummary {
  provider: 'FRED';
  feed: 'series/observations';
  status: MarketDataSourceStatus;
  source_mode: MarketDataSourceMode;
  fetched_at: string;
  series: FredSeriesSummary[];
}

export interface FetchYieldCurveDataResult {
  data: TreasuryYield[];
  summary: FredYieldSummary;
}

export class YahooFetchError extends Error {
  reason: YahooFetchFailureReason;
  attempts: number;

  constructor(reason: YahooFetchFailureReason, message: string, attempts: number = 1) {
    super(message);
    this.name = 'YahooFetchError';
    this.reason = reason;
    this.attempts = attempts;
  }
}

export class MarketDataProviderError extends Error {
  reason: MarketDataFetchFailureReason;
  attempts: number;

  constructor(reason: MarketDataFetchFailureReason, message: string, attempts: number = 1) {
    super(message);
    this.name = 'MarketDataProviderError';
    this.reason = reason;
    this.attempts = attempts;
  }
}

// Core portfolio symbols
const CORE_SYMBOLS = ['SPY', 'QQQ', 'VTI', 'VBR', 'TLT', 'IEF', 'SHY', 'GLD', 'AGG', 'DBC', 'EFA', 'VXUS', '^VIX3M'];

// Factor ETFs for alternative risk premia harvesting (v4.10)
const FACTOR_ETFS = [
  'MTUM',  // iShares MSCI USA Momentum Factor ETF
  'VLUE',  // iShares MSCI USA Value Factor ETF
  'USMV',  // iShares MSCI USA Min Vol Factor ETF
  'QUAL',  // iShares MSCI USA Quality Factor ETF
];

// Sector ETF symbols (v2.40 - Sector Rotation Momentum)
const SECTOR_ETFS = [
  'XLK',   // Technology
  'XLV',   // Healthcare  
  'XLF',   // Financials
  'XLY',   // Consumer Discretionary
  'XLI',   // Industrials
  'XLE',   // Energy
  'XLP',   // Consumer Staples
  'XLU',   // Utilities
  'XLB',   // Materials
  'XLRE',  // Real Estate
  'XLC',   // Communication Services
];

// Leveraged Treasury ETFs (v2.35 Capital Efficiency)
const LEVERAGED_TREASURY_ETFS = [
  'UBT',   // ProShares Ultra 20+ Year Treasury (2x TLT)
  'TMF',   // Direxion Daily 20+ Year Treasury Bull 3X (3x TLT)
];

// FX Currency ETFs (v3.15, v3.19 ML FX Carry Infrastructure)
const FX_SYMBOLS = [
  'UUP',   // Invesco DB US Dollar Index Bullish Fund
  'UDN',   // Invesco DB US Dollar Index Bearish Fund
  'FXE',   // Invesco CurrencyShares Euro Trust
  'FXY',   // Invesco CurrencyShares Japanese Yen Trust
  'FXB',   // Invesco CurrencyShares British Pound Sterling Trust
  'FXA',   // Invesco CurrencyShares Australian Dollar Trust
  'FXC',   // Invesco CurrencyShares Canadian Dollar Trust
  'FXF',   // Invesco CurrencyShares Swiss Franc Trust
];

// Combined symbol list for backward compatibility
const SYMBOLS = [...CORE_SYMBOLS, ...SECTOR_ETFS, ...LEVERAGED_TREASURY_ETFS, ...FX_SYMBOLS, ...FACTOR_ETFS];
const FRED_SERIES = {
  dgs2: 'DGS2',
  dgs10: 'DGS10',
  dgs30: 'DGS30',
};

function sleep(ms: number): Promise<void> {
  return ms > 0 ? new Promise((resolve) => setTimeout(resolve, ms)) : Promise.resolve();
}

function latestPriceObservation(prices: HistoricalPrice[]): string | null {
  return prices.length > 0 ? prices[prices.length - 1].date : null;
}

function classifyYahooHttpStatus(status: number): YahooFetchFailureReason {
  if (status === 429) return 'rate_limited';
  if (status === 404) return 'no_data';
  return 'unknown';
}

function classifyYahooThrown(error: unknown): YahooFetchError {
  if (error instanceof YahooFetchError) {
    return error;
  }
  if (error instanceof Error && error.name === 'AbortError') {
    return new YahooFetchError('timeout', error.message || 'Yahoo request timed out');
  }
  if (error instanceof TypeError) {
    return new YahooFetchError('network_error', error.message);
  }
  if (error instanceof Error) {
    return new YahooFetchError('unknown', error.message);
  }
  return new YahooFetchError('unknown', String(error));
}

function classifyMarketDataProviderThrown(error: unknown): MarketDataProviderError {
  if (error instanceof MarketDataProviderError) {
    return error;
  }
  if (error instanceof YahooFetchError) {
    return new MarketDataProviderError(error.reason, error.message, error.attempts);
  }
  if (typeof error === 'object' && error !== null && 'reason' in error) {
    const reasonedError = error as { reason: unknown; attempts?: unknown };
    const reason = String(reasonedError.reason) as MarketDataFetchFailureReason;
    const message = error instanceof Error ? error.message : String(error);
    const attempts = typeof reasonedError.attempts === 'number'
      ? reasonedError.attempts
      : 1;
    return new MarketDataProviderError(reason, message, attempts);
  }
  const yahooError = classifyYahooThrown(error);
  return new MarketDataProviderError(yahooError.reason, yahooError.message, yahooError.attempts);
}

async function fetchWithTimeout(
  url: string,
  options: YahooFetchOptions,
): Promise<Response> {
  const fetchImpl = options.fetchImpl ?? fetch;
  const timeoutMs = options.timeoutMs ?? 10_000;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetchImpl(url, {
      headers: { 'User-Agent': 'Mozilla/5.0' },
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timer);
  }
}

async function parseYahooChartResponse(response: Response, symbol: string): Promise<HistoricalPrice[]> {
  let data: { chart?: { result?: Array<{ timestamp?: number[]; indicators?: { quote?: Array<{ open?: (number | null)[]; high?: (number | null)[]; low?: (number | null)[]; close?: (number | null)[]; volume?: (number | null)[] }>; adjclose?: Array<{ adjclose?: (number | null)[] }> } }> } };
  try {
    data = await response.json() as typeof data;
  } catch (error) {
    throw new YahooFetchError('malformed_payload', `Malformed Yahoo payload for ${symbol}: ${error}`);
  }

  const result = data.chart?.result?.[0];
  if (!result) throw new YahooFetchError('no_data', `No data returned for ${symbol}`);

  const timestamps: number[] = result.timestamp || [];
  const quote = result.indicators?.quote?.[0];
  if (!quote) {
    throw new YahooFetchError('malformed_payload', `Yahoo payload missing quote block for ${symbol}`);
  }
  const adjclose = result.indicators?.adjclose?.[0]?.adjclose || [];

  const prices: HistoricalPrice[] = [];
  for (let i = 0; i < timestamps.length; i++) {
    const close = quote.close?.[i];
    if (close == null || isNaN(close)) continue;
    const adj = adjclose[i] ?? close;

    const d = new Date(timestamps[i] * 1000);
    prices.push({
      date: d.toISOString().split('T')[0],
      open: quote.open?.[i] ?? close,
      high: quote.high?.[i] ?? close,
      low: quote.low?.[i] ?? close,
      close,
      adjClose: adj,
      volume: quote.volume?.[i] ?? 0,
    });
  }

  if (prices.length === 0) {
    throw new YahooFetchError('no_data', `No usable price rows returned for ${symbol}`);
  }
  return prices;
}

/**
 * Fetch historical data from Yahoo Finance v8 chart API.
 */
export async function fetchYahooV8(
  symbol: string,
  startDate: string,
  endDate: string,
  options: YahooFetchOptions = {},
): Promise<HistoricalPrice[]> {
  const period1 = Math.floor(new Date(startDate).getTime() / 1000);
  const period2 = Math.floor(new Date(endDate).getTime() / 1000);

  const url = `https://query2.finance.yahoo.com/v8/finance/chart/${symbol}?period1=${period1}&period2=${period2}&interval=1d`;
  const maxAttempts = Math.max(1, options.maxAttempts ?? 3);
  const backoffMs = options.backoffMs ?? 500;
  let lastError: YahooFetchError | null = null;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const response = await fetchWithTimeout(url, options);
      if (!response.ok) {
        throw new YahooFetchError(
          classifyYahooHttpStatus(response.status),
          `HTTP ${response.status}: ${response.statusText}`,
          attempt,
        );
      }
      return await parseYahooChartResponse(response, symbol);
    } catch (error) {
      lastError = classifyYahooThrown(error);
      lastError.attempts = attempt;
      if (attempt < maxAttempts) {
        await sleep(backoffMs * attempt);
      }
    }
  }

  throw lastError ?? new YahooFetchError('unknown', `Yahoo fetch failed for ${symbol}`, maxAttempts);
}

export function createYahooFinanceProvider(options: YahooFetchOptions = {}): MarketDataProvider {
  return {
    name: 'Yahoo Finance',
    feed: 'chart/v8',
    sourceMode: 'live',
    fetchSymbol: (symbol, startDate, endDate) => fetchYahooV8(symbol, startDate, endDate, options),
  };
}

function createConfiguredProviderPlaceholder(providerName: string): MarketDataProvider {
  return {
    name: providerName,
    feed: 'configured-provider',
    sourceMode: 'live',
    fetchSymbol: async () => {
      throw new MarketDataProviderError(
        'not_configured',
        `${providerName} is selected as MARKET_DATA_PRIMARY_PROVIDER but no adapter is configured in this build`,
      );
    },
  };
}

export function resolveMarketDataProviders(options: FetchAllDataOptions = {}): MarketDataProvider[] {
  if (options.providers && options.providers.length > 0) {
    return options.providers;
  }

  const configuredPrimary = process.env.MARKET_DATA_PRIMARY_PROVIDER?.trim();
  const providers: MarketDataProvider[] = [];
  if (
    configuredPrimary
    && !['yahoo', 'yahoo finance'].includes(configuredPrimary.toLowerCase())
  ) {
    providers.push(createConfiguredProviderPlaceholder(configuredPrimary));
  }
  providers.push(createYahooFinanceProvider(options));
  return providers;
}

/**
 * Fetch all symbols for backtesting
 */
export async function fetchAllData(
  symbols: string[] = SYMBOLS,
  startDate: string = '2005-01-01',
  endDate: string = new Date().toISOString().split('T')[0]
): Promise<{ [symbol: string]: HistoricalPrice[] }> {
  return (await fetchAllDataWithSummary(symbols, startDate, endDate)).data;
}

/**
 * Fetch all symbols and return structured provider diagnostics.
 */
export async function fetchAllDataWithSummary(
  symbols: string[] = SYMBOLS,
  startDate: string = '2005-01-01',
  endDate: string = new Date().toISOString().split('T')[0],
  options: FetchAllDataOptions = {},
): Promise<FetchAllDataResult> {
  const providers = resolveMarketDataProviders(options);
  const providerChain = providers.map((provider) => provider.name);
  const primaryProvider = providers[0]?.name ?? null;
  const result: { [symbol: string]: HistoricalPrice[] } = {};
  const summaries: SymbolFetchSummary[] = [];
  const failureCounts: Partial<Record<MarketDataFetchFailureReason, number>> = {};
  const circuitBreakerThreshold = Math.max(1, options.circuitBreakerFailureThreshold ?? 3);
  const delayMs = options.delayMs ?? 300;
  let consecutiveRateLimitFailures = 0;
  let circuitOpened = false;
  const skippedSymbols: string[] = [];
  const fetchedAt = new Date().toISOString();

  console.log(`Fetching data for ${symbols.length} symbols from ${providerChain.join(' -> ')}...`);

  for (let idx = 0; idx < symbols.length; idx++) {
    const symbol = symbols[idx];
    if (circuitOpened) {
      skippedSymbols.push(symbol);
      summaries.push({
        symbol,
        provider: null,
        feed: null,
        provider_chain: providerChain,
        primary_provider: primaryProvider,
        fallback_provider: null,
        status: 'skipped',
        source_mode: 'live',
        rows: 0,
        latest_observation: null,
        attempts: 0,
        fetched_at: new Date().toISOString(),
        failure_reason: 'rate_limited',
        error: 'Yahoo provider circuit breaker open for this run',
      });
      continue;
    }

    console.log(`  Fetching ${symbol}...`);
    const providerFailures: MarketDataProviderError[] = [];
    let symbolFetched = false;

    for (const provider of providers) {
      try {
        const prices = await provider.fetchSymbol(symbol, startDate, endDate);
        result[symbol] = prices;
        const fallbackReason = providerFailures[0]?.reason;
        const fallbackProvider = provider.name !== primaryProvider ? provider.name : null;
        summaries.push({
          symbol,
          provider: provider.name,
          feed: provider.feed,
          provider_chain: providerChain,
          primary_provider: primaryProvider,
          fallback_provider: fallbackProvider,
          status: fallbackProvider ? 'degraded' : 'success',
          source_mode: provider.sourceMode ?? 'live',
          rows: prices.length,
          latest_observation: latestPriceObservation(prices),
          attempts: providerFailures.length + 1,
          fetched_at: new Date().toISOString(),
          fallback_reason: fallbackReason,
        });
        consecutiveRateLimitFailures = 0;
        symbolFetched = true;
        console.log(`  ✓ ${symbol}: ${prices.length} days from ${provider.name}`);
        await sleep(delayMs);
        break;
      } catch (error) {
        const fetchError = classifyMarketDataProviderThrown(error);
        providerFailures.push(fetchError);
        failureCounts[fetchError.reason] = (failureCounts[fetchError.reason] ?? 0) + 1;
        console.warn(`  ${provider.name} failed for ${symbol}: ${fetchError.message}`);
      }
    }

    if (!symbolFetched) {
      const lastError = providerFailures[providerFailures.length - 1]
        ?? new MarketDataProviderError('provider_unavailable', `No price providers configured for ${symbol}`, 0);
      consecutiveRateLimitFailures = lastError.reason === 'rate_limited'
        ? consecutiveRateLimitFailures + 1
        : 0;
      summaries.push({
        symbol,
        provider: null,
        feed: null,
        provider_chain: providerChain,
        primary_provider: primaryProvider,
        fallback_provider: null,
        status: 'failed',
        source_mode: 'live',
        rows: 0,
        latest_observation: null,
        attempts: providerFailures.reduce((sum, failure) => sum + Math.max(1, failure.attempts), 0),
        fetched_at: new Date().toISOString(),
        failure_reason: lastError.reason,
        error: `All price providers failed for ${symbol}: ${lastError.message}`,
      });
      console.error(`  ✗ ${symbol}: ${lastError.message}`);
      if (consecutiveRateLimitFailures >= circuitBreakerThreshold && idx < symbols.length - 1) {
        circuitOpened = true;
      }
    }
  }

  const successCount = summaries.filter((summary) => summary.status === 'success' || summary.status === 'degraded').length;
  const failedCount = summaries.filter((summary) => summary.status === 'failed').length;
  const skippedCount = summaries.filter((summary) => summary.status === 'skipped').length;
  const usedFallback = summaries.some((summary) => summary.fallback_provider !== null);
  const status: MarketDataSourceStatus = failedCount === 0 && skippedCount === 0 && !usedFallback
    ? 'success'
    : successCount > 0
      ? 'degraded'
      : 'failed';
  const successfulSummaries = summaries.filter((summary) => summary.provider !== null);
  const successfulProviders = Array.from(new Set(successfulSummaries.map((summary) => summary.provider as string)));
  const successfulFeeds = Array.from(new Set(successfulSummaries.map((summary) => summary.feed as string)));
  const actualProvider = successfulProviders.length === 0
    ? 'none'
    : successfulProviders.length === 1
      ? successfulProviders[0]
      : 'Mixed';
  const actualFeed = successfulFeeds.length === 0
    ? 'none'
    : successfulFeeds.length === 1
      ? successfulFeeds[0]
      : 'mixed';
  const fallbackProvider = successfulSummaries.find((summary) => summary.provider !== primaryProvider)?.provider ?? null;

  return {
    data: result,
    summary: {
      provider: actualProvider,
      feed: actualFeed,
      provider_chain: providerChain,
      primary_provider: primaryProvider,
      fallback_provider: fallbackProvider,
      status,
      source_mode: 'live',
      fetched_at: fetchedAt,
      symbols: summaries,
      failure_counts: failureCounts,
      circuit_breaker: {
        opened: circuitOpened,
        reason: circuitOpened ? 'rate_limited' : null,
        skipped_symbols: skippedSymbols,
      },
    },
  };
}

/**
 * Convert to backtest engine format
 */
export function convertToBacktestFormat(
  data: { [symbol: string]: HistoricalPrice[] }
): Array<{ date: string; symbol: string; price: number; dividend?: number }> {
  const result: Array<{ date: string; symbol: string; price: number; dividend?: number }> = [];

  for (const [symbol, prices] of Object.entries(data)) {
    for (const p of prices) {
      result.push({
        date: p.date,
        symbol,
        price: p.adjClose,
      });
    }
  }

  return result.sort((a, b) => a.date.localeCompare(b.date));
}

/**
 * Fetch Treasury yield data from FRED API.
 * A FRED_API_KEY is required for live FRED observations.
 */
export async function fetchFredSeries(
  seriesId: string,
  startDate: string,
  endDate: string
): Promise<{ date: string; value: number }[]> {
  return (await fetchFredSeriesWithSummary(seriesId, startDate, endDate)).data;
}

export async function fetchFredSeriesWithSummary(
  seriesId: string,
  startDate: string,
  endDate: string,
  options: FredFetchOptions = {},
): Promise<{ data: { date: string; value: number }[]; summary: FredSeriesSummary }> {
  const url = `https://api.stlouisfed.org/fred/series/observations?series_id=${seriesId}&api_key=FRED_API_KEY&file_type=json&observation_start=${startDate}&observation_end=${endDate}`;

  // In production/tasker mode, set FRED_API_KEY for live observations.
  const apiKey = process.env.FRED_API_KEY || '';
  const fetchedAt = new Date().toISOString();
  if (!apiKey) {
    console.warn(`FRED_API_KEY not set - using deterministic synthetic yield fallback`);
    const fallback = generateFallbackYields(seriesId, startDate, endDate);
    return {
      data: fallback,
      summary: {
        series_id: seriesId,
        status: 'degraded',
        source_mode: 'synthetic',
        rows: fallback.length,
        latest_observation: fallback.length > 0 ? fallback[fallback.length - 1].date : null,
        fetched_at: fetchedAt,
        failure_reason: 'missing_api_key',
      },
    };
  }

  const authUrl = url.replace('FRED_API_KEY', apiKey);

  try {
    const response = await (options.fetchImpl ?? fetch)(authUrl);
    if (!response.ok) {
      const reason: FredFailureReason = response.status === 429 ? 'rate_limited' : 'api_error';
      throw Object.assign(new Error(`FRED API error: ${response.status}`), { reason });
    }

    const data = await response.json() as {
      observations: Array<{ date: string; value: string }>;
    };

    if (!Array.isArray(data.observations)) {
      throw Object.assign(new Error('FRED payload missing observations array'), { reason: 'malformed_payload' });
    }

    const rows = data.observations
      .filter(obs => obs.value !== '.')
      .map(obs => ({
        date: obs.date,
        value: parseFloat(obs.value),
      }));
    return {
      data: rows,
      summary: {
        series_id: seriesId,
        status: 'success',
        source_mode: 'live',
        rows: rows.length,
        latest_observation: rows.length > 0 ? rows[rows.length - 1].date : null,
        fetched_at: fetchedAt,
      },
    };
  } catch (error) {
    console.warn(`FRED fetch failed for ${seriesId}: ${error}`);
    const fallback = generateFallbackYields(seriesId, startDate, endDate);
    const failureReason = typeof error === 'object' && error !== null && 'reason' in error
      ? (error as { reason: FredFailureReason }).reason
      : error instanceof TypeError
        ? 'network_error'
        : 'unknown';
    return {
      data: fallback,
      summary: {
        series_id: seriesId,
        status: 'degraded',
        source_mode: 'synthetic',
        rows: fallback.length,
        latest_observation: fallback.length > 0 ? fallback[fallback.length - 1].date : null,
        fetched_at: fetchedAt,
        failure_reason: failureReason,
      },
    };
  }
}

/**
 * Generate fallback yield data based on historical averages
 * Used when FRED API is unavailable
 */
export function generateFallbackYields(
  seriesId: string,
  startDate: string,
  endDate: string
): { date: string; value: number }[] {
  // Approximate historical averages by period
  const historicalYields: Record<string, Record<string, number>> = {
    DGS2: {
      '2005': 3.5, '2006': 4.5, '2007': 4.0, '2008': 2.0, '2009': 0.8,
      '2010': 0.5, '2011': 0.4, '2012': 0.3, '2013': 0.3, '2014': 0.4,
      '2015': 0.6, '2016': 0.8, '2017': 1.4, '2018': 2.5, '2019': 1.8,
      '2020': 0.2, '2021': 0.2, '2022': 3.0, '2023': 4.5, '2024': 4.2, '2025': 4.0, '2026': 3.8,
    },
    DGS10: {
      '2005': 4.3, '2006': 4.7, '2007': 4.6, '2008': 3.7, '2009': 3.3,
      '2010': 3.2, '2011': 3.0, '2012': 1.8, '2013': 2.3, '2014': 2.5,
      '2015': 2.1, '2016': 1.8, '2017': 2.3, '2018': 2.9, '2019': 2.1,
      '2020': 0.9, '2021': 1.4, '2022': 2.9, '2023': 3.9, '2024': 4.2, '2025': 4.3, '2026': 4.1,
    },
    DGS30: {
      '2005': 4.5, '2006': 4.8, '2007': 4.7, '2008': 4.1, '2009': 4.1,
      '2010': 4.2, '2011': 3.9, '2012': 2.9, '2013': 3.2, '2014': 3.2,
      '2015': 2.8, '2016': 2.5, '2017': 2.8, '2018': 3.1, '2019': 2.4,
      '2020': 1.5, '2021': 1.9, '2022': 3.2, '2023': 4.1, '2024': 4.4, '2025': 4.5, '2026': 4.3,
    },
  };

  const start = new Date(startDate);
  const end = new Date(endDate);
  const results: { date: string; value: number }[] = [];

  for (let d = new Date(start); d <= end; d.setDate(d.getDate() + 1)) {
    const year = d.getFullYear().toString();
    const baseYield = historicalYields[seriesId]?.[year] ?? 3.0;

    const variation = deterministicYieldVariation(seriesId, d.toISOString().split('T')[0]);

    results.push({
      date: d.toISOString().split('T')[0],
      value: Math.max(0.01, baseYield + variation),
    });
  }

  return results;
}

function deterministicYieldVariation(seriesId: string, date: string): number {
  const key = `${seriesId}:${date}`;
  let hash = 0;
  for (let i = 0; i < key.length; i++) {
    hash = (hash * 31 + key.charCodeAt(i)) >>> 0;
  }
  return ((hash % 2001) / 2000 - 0.5) * 0.2;
}

/**
 * Fetch and calculate yield curve data for all dates
 */
export async function fetchYieldCurveData(
  startDate: string,
  endDate: string
): Promise<TreasuryYield[]> {
  return (await fetchYieldCurveDataWithSummary(startDate, endDate)).data;
}

export async function fetchYieldCurveDataWithSummary(
  startDate: string,
  endDate: string,
  options: FredFetchOptions = {},
): Promise<FetchYieldCurveDataResult> {
  console.log('Fetching Treasury yield data from FRED...');

  const [dgs2Result, dgs10Result, dgs30Result] = await Promise.all([
    fetchFredSeriesWithSummary(FRED_SERIES.dgs2, startDate, endDate, options),
    fetchFredSeriesWithSummary(FRED_SERIES.dgs10, startDate, endDate, options),
    fetchFredSeriesWithSummary(FRED_SERIES.dgs30, startDate, endDate, options),
  ]);
  const dgs2Data = dgs2Result.data;
  const dgs10Data = dgs10Result.data;
  const dgs30Data = dgs30Result.data;

  // Merge by date
  const dateMap = new Map<string, Partial<TreasuryYield>>();

  for (const obs of dgs2Data) {
    dateMap.set(obs.date, { ...dateMap.get(obs.date), date: obs.date, dgs2: obs.value });
  }
  for (const obs of dgs10Data) {
    dateMap.set(obs.date, { ...dateMap.get(obs.date), date: obs.date, dgs10: obs.value });
  }
  for (const obs of dgs30Data) {
    dateMap.set(obs.date, { ...dateMap.get(obs.date), date: obs.date, dgs30: obs.value });
  }

  // Calculate spreads and filter complete records
  const yields: TreasuryYield[] = [];
  const entries = Array.from(dateMap.values());
  for (const entry of entries) {
    if (entry.dgs2 !== undefined && entry.dgs10 !== undefined && entry.dgs30 !== undefined) {
      yields.push({
        date: entry.date!,
        dgs2: entry.dgs2,
        dgs10: entry.dgs10,
        dgs30: entry.dgs30,
        spread2s10s: (entry.dgs10 - entry.dgs2) * 100, // Convert to bps
        spread10s30s: (entry.dgs30 - entry.dgs10) * 100,
      });
    }
  }

  console.log(`✓ Yield curve data: ${yields.length} days`);
  const sortedYields = yields.sort((a, b) => a.date.localeCompare(b.date));
  const series = [dgs2Result.summary, dgs10Result.summary, dgs30Result.summary];
  const hasSynthetic = series.some((summary) => summary.source_mode === 'synthetic');
  const hasFailure = series.some((summary) => summary.status !== 'success');
  return {
    data: sortedYields,
    summary: {
      provider: 'FRED',
      feed: 'series/observations',
      status: hasFailure ? 'degraded' : 'success',
      source_mode: hasSynthetic ? 'synthetic' : 'live',
      fetched_at: new Date().toISOString(),
      series,
    },
  };
}

/**
 * Fetch only sector ETF data for sector rotation strategies
 * v2.40 - Sector Rotation Momentum Infrastructure
 */
export async function fetchSectorData(
  startDate: string = '2005-01-01',
  endDate: string = new Date().toISOString().split('T')[0]
): Promise<{ [symbol: string]: HistoricalPrice[] }> {
  console.log(`Fetching sector ETF data for ${SECTOR_ETFS.length} symbols...`);
  return fetchAllData(SECTOR_ETFS, startDate, endDate);
}

/**
 * Fetch core portfolio data without sectors
 */
export async function fetchCoreData(
  startDate: string = '2005-01-01',
  endDate: string = new Date().toISOString().split('T')[0]
): Promise<{ [symbol: string]: HistoricalPrice[] }> {
  console.log(`Fetching core portfolio data for ${CORE_SYMBOLS.length} symbols...`);
  return fetchAllData(CORE_SYMBOLS, startDate, endDate);
}

// Export symbol lists for strategy modules
export { CORE_SYMBOLS, SECTOR_ETFS, FX_SYMBOLS, SYMBOLS };

// CLI usage
if (import.meta.main) {
  const priceData = await fetchAllData();
  const yieldData = await fetchYieldCurveData('2005-01-01', new Date().toISOString().split('T')[0]);

  // Save to project public directory
  const dataDir = new URL('../../public/data', import.meta.url).pathname;

  await Bun.write(`${dataDir}/historical.json`, JSON.stringify(priceData, null, 2));
  console.log(`\nPrice data saved to ${dataDir}/historical.json`);

  await Bun.write(`${dataDir}/yields.json`, JSON.stringify(yieldData, null, 2));
  console.log(`Yield curve data saved to ${dataDir}/yields.json`);
}
