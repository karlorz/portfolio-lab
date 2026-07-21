// Yahoo Finance v8 chart API fetcher
import { yahooChartPeriodBounds } from './fetcher';

export async function fetchYahooData(symbol: string, start: string, end: string) {
  const { period1, period2 } = yahooChartPeriodBounds(start, end);

  const url = `https://query2.finance.yahoo.com/v8/finance/chart/${symbol}?period1=${period1}&period2=${period2}&interval=1d`;

  const res = await fetch(url, {
    headers: { 'User-Agent': 'Mozilla/5.0' },
  });
  if (!res.ok) throw new Error(`Failed to fetch ${symbol}: ${res.status}`);

  const data = await res.json() as { chart?: { result?: Array<{ timestamp?: number[]; indicators?: { quote?: Array<{ close?: (number | null)[] }>; adjclose?: Array<{ adjclose?: (number | null)[] }> } }> } };
  const result = data.chart?.result?.[0];
  if (!result) throw new Error(`No data for ${symbol}`);

  const timestamps: number[] = result.timestamp || [];
  const adjclose = result.indicators?.adjclose?.[0]?.adjclose || [];
  const quote = result.indicators?.quote?.[0] || {};

  return timestamps.map((ts, i) => ({
    date: new Date(ts * 1000).toISOString().split('T')[0],
    adjClose: adjclose[i] ?? quote.close?.[i] ?? 0,
  })).filter(d => d.adjClose > 0);
}

// Fetch all symbols for backtest
const SYMBOLS = ['SPY', 'QQQ', 'VTI', 'VBR', 'TLT', 'IEF', 'SHY', 'GLD', 'AGG', 'DBC', 'EFA', 'VXUS', 'MTUM', 'VLUE', 'USMV', 'QUAL', 'IJR'];

export async function fetchAllSymbols() {
  const allData: Record<string, Array<{date: string, price: number}>> = {};

  for (const symbol of SYMBOLS) {
    try {
      const data = await fetchYahooData(symbol, '2005-01-01', new Date().toISOString().split('T')[0]);
      allData[symbol] = data.map(d => ({ date: d.date, price: d.adjClose }));
      console.log(`✓ ${symbol}: ${data.length} days`);
      await new Promise(r => setTimeout(r, 300)); // Rate limit
    } catch (e) {
      console.error(`✗ ${symbol}: ${e}`);
    }
  }

  return allData;
}

// CLI: fetch and save prices.json + last-N prices_compact.json (Batch BK)
if (import.meta.main) {
  const { buildLastNBarsCompact, resolvePricesCompactNBars } = await import(
    '../../scripts/fetch-data.ts'
  );
  const data = await fetchAllSymbols();
  // Full history: { SPY: [{d:"2024-01-02",p:123.45}, ...], ... }
  const full: Record<string, Array<{d: string, p: number}>> = {};
  for (const [sym, entries] of Object.entries(data)) {
    full[sym] = entries.map(e => ({ d: e.date, p: e.price }));
  }
  const outPath = new URL('../../public/data/prices.json', import.meta.url).pathname;
  const compactPath = new URL('../../public/data/prices_compact.json', import.meta.url).pathname;
  const nBars = resolvePricesCompactNBars();
  const compactPayload = buildLastNBarsCompact(full, nBars);
  await Bun.write(outPath, JSON.stringify(full));
  await Bun.write(compactPath, JSON.stringify(compactPayload));
  console.log(`\nFull history saved to ${outPath}`);
  console.log(
    `Compact last-${nBars} bars saved to ${compactPath} `
    + `(${compactPayload.meta.bar_count} points)`,
  );
  console.log(`Symbols: ${Object.keys(full).join(', ')}`);
  for (const [k, v] of Object.entries(full)) {
    console.log(`  ${k}: ${v.length} days (${v[0]?.d} to ${v[v.length-1]?.d})`);
  }
}
