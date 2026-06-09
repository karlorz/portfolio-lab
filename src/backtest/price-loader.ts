import type { PortfolioConfig, PriceData } from './engine';

export interface CompactPriceEntry {
  d: string;
  p: number;
  dividend?: number;
}

export type CompactPriceData = Record<string, ReadonlyArray<CompactPriceEntry>>;

export const COMPACT_PRICE_DATA_ENDPOINT = '/data/prices_compact.json';
export const LEGACY_PRICE_DATA_ENDPOINT = '/data/prices.json';

type PriceDataFetcher = (url: string) => Promise<Response>;

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
  return await response.json() as CompactPriceData;
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
