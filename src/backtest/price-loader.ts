import type { PortfolioConfig, PriceData } from './engine';

export interface CompactPriceEntry {
  d: string;
  p: number;
  dividend?: number;
}

export type CompactPriceData = Record<string, ReadonlyArray<CompactPriceEntry>>;

/** Wrapped compact artifact (Batch BK): meta + symbols last-N map. */
export interface CompactPriceArtifactV1 {
  meta?: {
    schema?: string;
    as_of?: string;
    n_bars?: number;
    full_artifact?: string;
    symbol_count?: number;
    bar_count?: number;
  };
  symbols: CompactPriceData;
}

export const COMPACT_PRICE_DATA_ENDPOINT = '/data/prices_compact.json';
export const LEGACY_PRICE_DATA_ENDPOINT = '/data/prices.json';

type PriceDataFetcher = (url: string) => Promise<Response>;

/**
 * Normalize prices_compact.json payloads.
 * Accepts legacy flat `{SPY:[{d,p}]}` and Batch BK `{meta, symbols}`.
 */
export function unwrapCompactPricePayload(payload: unknown): CompactPriceData | null {
  if (payload === null || typeof payload !== 'object' || Array.isArray(payload)) {
    return null;
  }
  const record = payload as Record<string, unknown>;
  // Wrapped form: { meta, symbols: { SPY: [...] } }
  if (
    record.symbols !== undefined
    && typeof record.symbols === 'object'
    && record.symbols !== null
    && !Array.isArray(record.symbols)
  ) {
    return record.symbols as CompactPriceData;
  }
  // Legacy flat map — first value is an array of bars
  const values = Object.values(record);
  if (values.length === 0) {
    return record as CompactPriceData;
  }
  const first = values[0];
  if (Array.isArray(first)) {
    return record as CompactPriceData;
  }
  return null;
}

export function symbolsForPortfolios(portfolios: ReadonlyArray<PortfolioConfig>): string[] {
  const symbols = new Set<string>();
  for (const portfolio of portfolios) {
    for (const symbol of Object.keys(portfolio.allocation)) {
      symbols.add(symbol);
    }
  }
  return Array.from(symbols).sort();
}

export function toBacktestData(
  prices: CompactPriceData,
  symbols?: ReadonlyArray<string>,
): PriceData[] {
  const includedSymbols = symbols ? new Set(symbols) : null;
  const result: PriceData[] = [];

  for (const [symbol, entries] of Object.entries(prices)) {
    if (includedSymbols && !includedSymbols.has(symbol)) {
      continue;
    }

    for (const entry of entries) {
      result.push({
        date: entry.d,
        symbol,
        price: entry.p,
        ...(entry.dividend !== undefined ? { dividend: entry.dividend } : {}),
      });
    }
  }

  return result.sort((a, b) => a.date.localeCompare(b.date));
}

async function fetchPricePayload(
  fetcher: PriceDataFetcher,
  endpoint: string,
): Promise<CompactPriceData | null> {
  const response = await fetcher(endpoint);
  if (!response.ok) {
    return null;
  }
  const raw: unknown = await response.json();
  return unwrapCompactPricePayload(raw);
}

export async function fetchCompactPriceData(
  fetcher: PriceDataFetcher = fetch,
): Promise<CompactPriceData> {
  try {
    const compact = await fetchPricePayload(fetcher, COMPACT_PRICE_DATA_ENDPOINT);
    if (compact) {
      return compact;
    }
  } catch {
    // Fall through to the legacy endpoint; static deploys may not have the new compact mirror yet.
  }

  const legacy = await fetchPricePayload(fetcher, LEGACY_PRICE_DATA_ENDPOINT);
  if (legacy) {
    return legacy;
  }
  throw new Error('Failed to load compact or legacy price data');
}
