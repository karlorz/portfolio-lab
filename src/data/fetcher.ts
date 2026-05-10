/**
 * Data Fetcher - Yahoo Finance v8 Chart API
 * Uses the chart endpoint (no API key required)
 */

export interface HistoricalPrice {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  adjClose: number;
  volume: number;
}

const SYMBOLS = ['SPY', 'QQQ', 'VTI', 'VBR', 'TLT', 'IEF', 'SHY', 'GLD', 'AGG', 'DBC', 'EFA', 'VXUS', 'MTUM', 'VLUE', 'USMV'];

/**
 * Fetch historical data from Yahoo Finance v8 chart API
 */
export async function fetchYahooV8(
  symbol: string,
  startDate: string,
  endDate: string
): Promise<HistoricalPrice[]> {
  const period1 = Math.floor(new Date(startDate).getTime() / 1000);
  const period2 = Math.floor(new Date(endDate).getTime() / 1000);

  const url = `https://query2.finance.yahoo.com/v8/finance/chart/${symbol}?period1=${period1}&period2=${period2}&interval=1d`;

  const response = await fetch(url, {
    headers: { 'User-Agent': 'Mozilla/5.0' },
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  const data = await response.json() as any;
  const result = data.chart?.result?.[0];
  if (!result) throw new Error(`No data returned for ${symbol}`);

  const timestamps: number[] = result.timestamp || [];
  const quote = result.indicators?.quote?.[0] || {};
  const adjclose = result.indicators?.adjclose?.[0]?.adjclose || [];

  const prices: HistoricalPrice[] = [];
  for (let i = 0; i < timestamps.length; i++) {
    const close = quote.close?.[i];
    const adj = adjclose[i] ?? close;
    if (close == null || isNaN(close)) continue;

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

  return prices;
}

/**
 * Fetch all symbols for backtesting
 */
export async function fetchAllData(
  symbols: string[] = SYMBOLS,
  startDate: string = '2005-01-01',
  endDate: string = new Date().toISOString().split('T')[0]
): Promise<{ [symbol: string]: HistoricalPrice[] }> {
  const result: { [symbol: string]: HistoricalPrice[] } = {};

  console.log(`Fetching data for ${symbols.length} symbols from Yahoo Finance v8...`);

  for (const symbol of symbols) {
    try {
      console.log(`  Fetching ${symbol}...`);
      result[symbol] = await fetchYahooV8(symbol, startDate, endDate);
      console.log(`  ✓ ${symbol}: ${result[symbol].length} days`);
      // Rate limit
      await new Promise(r => setTimeout(r, 300));
    } catch (error) {
      console.error(`  ✗ ${symbol}: ${error}`);
    }
  }

  return result;
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

// CLI usage
if (import.meta.main) {
  const data = await fetchAllData();

  // Save to project public directory
  const outPath = new URL('../../public/data/historical.json', import.meta.url).pathname;
  await Bun.write(outPath, JSON.stringify(data, null, 2));
  console.log(`\nData saved to ${outPath}`);
}
