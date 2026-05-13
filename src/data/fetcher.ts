/**
 * Data Fetcher - Yahoo Finance v8 Chart API + FRED Yield Data
 * Uses the chart endpoint (no API key required) for prices
 * FRED API for Treasury yield curve data
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

export interface TreasuryYield {
  date: string;
  dgs2: number;   // 2-Year Treasury Yield
  dgs10: number;  // 10-Year Treasury Yield
  dgs30: number;  // 30-Year Treasury Yield
  spread2s10s: number;  // 2s10s spread (basis points)
  spread10s30s: number; // 10s30s spread (basis points)
}

// Core portfolio symbols
const CORE_SYMBOLS = ['SPY', 'QQQ', 'VTI', 'VBR', 'TLT', 'IEF', 'SHY', 'GLD', 'AGG', 'DBC', 'EFA', 'VXUS', 'MTUM', 'VLUE', 'USMV', 'QUAL'];

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

// Combined symbol list for backward compatibility
const SYMBOLS = [...CORE_SYMBOLS, ...SECTOR_ETFS, ...LEVERAGED_TREASURY_ETFS];
const FRED_SERIES = {
  dgs2: 'DGS2',
  dgs10: 'DGS10',
  dgs30: 'DGS30',
};

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

/**
 * Fetch Treasury yield data from FRED API
 * FRED API is free, no key required for basic usage
 */
export async function fetchFredSeries(
  seriesId: string,
  startDate: string,
  endDate: string
): Promise<{ date: string; value: number }[]> {
  const url = `https://api.stlouisfed.org/fred/series/observations?series_id=${seriesId}&api_key=FRED_API_KEY&file_type=json&observation_start=${startDate}&observation_end=${endDate}`;

  // Note: For demo/development, we'll use cached data or fallback to simulated yields
  // In production, set FRED_API_KEY environment variable
  const apiKey = process.env.FRED_API_KEY || '';
  if (!apiKey) {
    console.warn(`FRED_API_KEY not set - using fallback yield data`);
    return generateFallbackYields(seriesId, startDate, endDate);
  }

  const authUrl = url.replace('FRED_API_KEY', apiKey);

  try {
    const response = await fetch(authUrl);
    if (!response.ok) throw new Error(`FRED API error: ${response.status}`);

    const data = await response.json() as {
      observations: Array<{ date: string; value: string }>;
    };

    return data.observations
      .filter(obs => obs.value !== '.')
      .map(obs => ({
        date: obs.date,
        value: parseFloat(obs.value),
      }));
  } catch (error) {
    console.warn(`FRED fetch failed for ${seriesId}: ${error}`);
    return generateFallbackYields(seriesId, startDate, endDate);
  }
}

/**
 * Generate fallback yield data based on historical averages
 * Used when FRED API is unavailable
 */
function generateFallbackYields(
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

    // Add small random variation
    const variation = (Math.random() - 0.5) * 0.2;

    results.push({
      date: d.toISOString().split('T')[0],
      value: Math.max(0.01, baseYield + variation),
    });
  }

  return results;
}

/**
 * Fetch and calculate yield curve data for all dates
 */
export async function fetchYieldCurveData(
  startDate: string,
  endDate: string
): Promise<TreasuryYield[]> {
  console.log('Fetching Treasury yield data from FRED...');

  const [dgs2Data, dgs10Data, dgs30Data] = await Promise.all([
    fetchFredSeries(FRED_SERIES.dgs2, startDate, endDate),
    fetchFredSeries(FRED_SERIES.dgs10, startDate, endDate),
    fetchFredSeries(FRED_SERIES.dgs30, startDate, endDate),
  ]);

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
  return yields.sort((a, b) => a.date.localeCompare(b.date));
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
export { CORE_SYMBOLS, SECTOR_ETFS, SYMBOLS };

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
