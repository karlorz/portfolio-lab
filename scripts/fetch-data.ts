#!/usr/bin/env bun
/**
 * Fetch market data from Yahoo Finance + FRED.
 * Saves prices.json and yields.json to public/data/.
 *
 * Usage: bun run fetch-data
 */

import { fetchAllData, fetchYieldCurveData, SYMBOLS } from '../src/data/fetcher';
import { join } from 'path';
import { existsSync, mkdirSync } from 'fs';

const DATA_DIR = join(import.meta.dir, '..', 'public', 'data');
const START_DATE = '2005-01-01';
const END_DATE = new Date().toISOString().split('T')[0];

async function main() {
  console.log('=== Portfolio-Lab Data Fetcher ===\n');

  if (!existsSync(DATA_DIR)) {
    mkdirSync(DATA_DIR, { recursive: true });
  }

  // 1. Fetch price data (Yahoo Finance v8)
  console.log(`Fetching ${SYMBOLS.length} symbols from ${START_DATE} to ${END_DATE}...\n`);
  const priceData = await fetchAllData(SYMBOLS, START_DATE, END_DATE);

  // Convert to compact format: { symbol: [{d, p}, ...] }
  const compact: Record<string, { d: string; p: number }[]> = {};
  let totalDays = 0;
  for (const [symbol, prices] of Object.entries(priceData)) {
    compact[symbol] = prices.map(p => ({ d: p.date, p: p.adjClose }));
    totalDays += prices.length;
  }

  const pricesPath = join(DATA_DIR, 'prices.json');
  await Bun.write(pricesPath, JSON.stringify(compact, null, 2));
  console.log(`\nSaved ${Object.keys(compact).length} symbols (${totalDays} total data points) → ${pricesPath}`);

  // 2. Fetch yield curve data (FRED)
  const yieldData = await fetchYieldCurveData(START_DATE, END_DATE);
  const yieldsPath = join(DATA_DIR, 'yields.json');
  await Bun.write(yieldsPath, JSON.stringify(yieldData, null, 2));
  console.log(`Saved ${yieldData.length} yield observations → ${yieldsPath}`);

  // 3. Regenerate dashboard JSON
  console.log('\nRegenerating dashboard JSON...');
  try {
    const { execSync } = await import('child_process');
    execSync('python3 -m src.dashboard.generator', {
      cwd: join(import.meta.dir, '..'),
      stdio: 'inherit',
    });
  } catch (e) {
    console.warn('Dashboard generator failed (Python may not be available):', e);
  }

  console.log('\nDone.');
}

main().catch(err => {
  console.error('Fetch failed:', err);
  process.exit(1);
});
