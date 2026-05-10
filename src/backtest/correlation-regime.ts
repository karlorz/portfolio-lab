/**
 * Correlation Regime Analysis — tests whether asset correlations shift during crises
 * Critical question: does the SPY/GLD diversification engine hold up when it matters most?
 */
import { BacktestEngine } from './engine';
import type { PriceData } from './engine';

function toBacktestData(prices: Record<string, Array<{ d: string; p: number }>>): PriceData[] {
  const result: PriceData[] = [];
  for (const [symbol, entries] of Object.entries(prices)) {
    for (const { d, p } of entries) {
      result.push({ date: d, symbol, price: p });
    }
  }
  return result.sort((a, b) => a.date.localeCompare(b.date));
}

function computeReturns(prices: Map<string, Map<string, number>>, symbols: string[], startDate: string, endDate: string): Map<string, number[]> {
  const returns = new Map<string, number[]>();
  // Get all trading days in range
  const allDates = new Set<string>();
  for (const sym of symbols) {
    const symData = prices.get(sym);
    if (!symData) continue;
    for (const d of symData.keys()) {
      if (d >= startDate && d <= endDate) allDates.add(d);
    }
  }
  const dates = Array.from(allDates).sort();

  for (const sym of symbols) {
    const symData = prices.get(sym);
    if (!symData) { returns.set(sym, []); continue; }
    const rets: number[] = [];
    let prevPrice = 0;
    for (const d of dates) {
      const p = symData.get(d);
      if (p && p > 0) {
        if (prevPrice > 0) rets.push((p - prevPrice) / prevPrice);
        prevPrice = p;
      }
    }
    returns.set(sym, rets);
  }
  return returns;
}

function rollingCorrelation(
  returnsA: number[],
  returnsB: number[],
  window: number
): { date: string; corr: number }[] {
  const results: { date: string; corr: number }[] = [];
  const len = Math.min(returnsA.length, returnsB.length);

  for (let i = window; i < len; i++) {
    const aSlice = returnsA.slice(i - window, i);
    const bSlice = returnsB.slice(i - window, i);

    const meanA = aSlice.reduce((s, v) => s + v, 0) / window;
    const meanB = bSlice.reduce((s, v) => s + v, 0) / window;

    let cov = 0, varA = 0, varB = 0;
    for (let j = 0; j < window; j++) {
      const da = aSlice[j] - meanA;
      const db = bSlice[j] - meanB;
      cov += da * db;
      varA += da * da;
      varB += db * db;
    }

    const corr = varA && varB ? cov / Math.sqrt(varA * varB) : 0;
    results.push({ date: '', corr });
  }
  return results;
}

function correlation(returnsA: number[], returnsB: number[]): number {
  const len = Math.min(returnsA.length, returnsB.length);
  if (len < 20) return 0;

  const meanA = returnsA.slice(0, len).reduce((s, v) => s + v, 0) / len;
  const meanB = returnsB.slice(0, len).reduce((s, v) => s + v, 0) / len;

  let cov = 0, varA = 0, varB = 0;
  for (let i = 0; i < len; i++) {
    const da = returnsA[i] - meanA;
    const db = returnsB[i] - meanB;
    cov += da * db;
    varA += da * da;
    varB += db * db;
  }
  return varA && varB ? cov / Math.sqrt(varA * varB) : 0;
}

async function main() {
  console.log('Loading price data...\n');
  const response = await fetch('file://' + new URL('../../public/data/prices.json', import.meta.url).pathname);
  const priceJson = await response.json() as Record<string, Array<{ d: string; p: number }>>;
  const priceData = toBacktestData(priceJson);

  const engine = new BacktestEngine();
  engine.loadData(priceData);

  // Access private priceData via loadData side effect
  // Rebuild the price map from priceData
  const prices = new Map<string, Map<string, number>>();
  for (const p of priceData) {
    if (!prices.has(p.symbol)) prices.set(p.symbol, new Map());
    prices.get(p.symbol)!.set(p.date, p.price);
  }

  const KEY_PAIRS = [
    ['SPY', 'GLD'],
    ['SPY', 'TLT'],
    ['SPY', 'IEF'],
    ['GLD', 'TLT'],
    ['SPY', 'EFA'],
    ['QQQ', 'GLD'],
  ];

  const REGIMES = [
    { name: 'Full Period', start: '2005-01-01', end: '2025-12-31' },
    { name: 'Pre-GFC Calm', start: '2005-01-01', end: '2007-09-30' },
    { name: 'GFC Crisis', start: '2007-10-01', end: '2009-06-30' },
    { name: 'Recovery', start: '2009-07-01', end: '2013-12-31' },
    { name: 'Taper Tantrum', start: '2013-05-01', end: '2013-09-30' },
    { name: '2015-16 Selloff', start: '2015-07-01', end: '2016-06-30' },
    { name: 'Bull Market', start: '2013-01-01', end: '2019-12-31' },
    { name: '2020 COVID Crash', start: '2020-02-01', end: '2020-04-30' },
    { name: 'COVID Recovery', start: '2020-04-01', end: '2021-12-31' },
    { name: '2022 Rate Hikes', start: '2022-01-01', end: '2022-12-31' },
    { name: '2023 Rally', start: '2023-01-01', end: '2023-12-31' },
    { name: '2025 Tariff Crisis', start: '2025-01-01', end: '2025-12-31' },
  ];

  console.log('=== CORRELATION REGIME ANALYSIS ===\n');

  // Full correlation table by regime
  for (const [symA, symB] of KEY_PAIRS) {
    console.log(`\n--- ${symA} / ${symB} Correlation by Regime ---`);
    console.log('| Regime | Correlation | Diversification Value |');
    console.log('|--------|-------------|----------------------|');

    for (const regime of REGIMES) {
      const returns = computeReturns(prices, [symA, symB], regime.start, regime.end);
      const retA = returns.get(symA) || [];
      const retB = returns.get(symB) || [];
      if (retA.length < 20 || retB.length < 20) continue;

      const corr = correlation(retA, retB);
      let value = '';
      if (corr < -0.3) value = '★ Strong hedge';
      else if (corr < 0) value = '✓ Mild hedge';
      else if (corr < 0.3) value = '~ Uncorrelated';
      else if (corr < 0.7) value = '⚠ Partial correlation';
      else value = '✗ Poor diversifier';

      console.log(`| ${regime.name} | ${corr.toFixed(3)} | ${value} |`);
    }
  }

  // Rolling correlation (60-day) for SPY/GLD — key question
  console.log('\n\n=== ROLLING 60-DAY CORRELATION: SPY/GLD ===\n');

  const allReturns = computeReturns(prices, ['SPY', 'GLD'], '2005-01-01', '2025-12-31');
  const spyRets = allReturns.get('SPY') || [];
  const gldRets = allReturns.get('GLD') || [];

  const window = 60;
  const rollingCorrs: number[] = [];
  const len = Math.min(spyRets.length, gldRets.length);

  for (let i = window; i < len; i++) {
    const aSlice = spyRets.slice(i - window, i);
    const bSlice = gldRets.slice(i - window, i);
    const c = correlation(aSlice, bSlice);
    rollingCorrs.push(c);
  }

  // Stats
  const mean = rollingCorrs.reduce((a, b) => a + b, 0) / rollingCorrs.length;
  const negatives = rollingCorrs.filter(c => c < 0).length;
  const positives = rollingCorrs.filter(c => c > 0.5).length;
  const min = Math.min(...rollingCorrs);
  const max = Math.max(...rollingCorrs);

  // Check what happens during known crisis months
  console.log(`Mean rolling correlation: ${mean.toFixed(3)}`);
  console.log(`Range: ${min.toFixed(3)} to ${max.toFixed(3)}`);
  console.log(`Negative correlation: ${((negatives / rollingCorrs.length) * 100).toFixed(1)}% of the time`);
  console.log(`High positive (>0.5): ${((positives / rollingCorrs.length) * 100).toFixed(1)}% of the time`);

  // Key question: does correlation spike during crises?
  console.log('\nCorrelation during crisis vs normal:');

  // Approximate crisis periods by index in the rolling correlation array
  // Rolling corr starts at day 60 of the return series
  // Returns start around day 1 of price data (2005-01-03)
  // So rolling corr index i corresponds to approximately trading day i+60

  // Instead, compute direct period correlations for crisis comparison
  const CRISIS_VS_NORMAL = [
    { name: 'Normal (2013-2019)', start: '2013-01-01', end: '2019-12-31', isCrisis: false },
    { name: 'GFC', start: '2007-10-01', end: '2009-06-30', isCrisis: true },
    { name: 'COVID', start: '2020-02-01', end: '2020-04-30', isCrisis: true },
    { name: '2022 Rates', start: '2022-01-01', end: '2022-12-31', isCrisis: true },
    { name: '2025 Tariff', start: '2025-01-01', end: '2025-12-31', isCrisis: true },
  ];

  console.log('\n| Period | SPY/GLD | SPY/TLT | GLD/TLT | Crisis? |');
  console.log('|--------|---------|---------|---------|---------|');
  for (const period of CRISIS_VS_NORMAL) {
    const rets = computeReturns(prices, ['SPY', 'GLD', 'TLT'], period.start, period.end);
    const spyGLD = correlation(rets.get('SPY') || [], rets.get('GLD') || []);
    const spyTLT = correlation(rets.get('SPY') || [], rets.get('TLT') || []);
    const gldTLT = correlation(rets.get('GLD') || [], rets.get('TLT') || []);
    console.log(`| ${period.name} | ${spyGLD.toFixed(3)} | ${spyTLT.toFixed(3)} | ${gldTLT.toFixed(3)} | ${period.isCrisis ? 'YES' : 'no'} |`);
  }

  console.log('\n=== KEY FINDING ===');
  const normalRets = computeReturns(prices, ['SPY', 'GLD'], '2013-01-01', '2019-12-31');
  const normalCorr = correlation(normalRets.get('SPY') || [], normalRets.get('GLD') || []);

  const crisisCombos = [
    computeReturns(prices, ['SPY', 'GLD'], '2007-10-01', '2009-06-30'),
    computeReturns(prices, ['SPY', 'GLD'], '2020-02-01', '2020-04-30'),
    computeReturns(prices, ['SPY', 'GLD'], '2025-01-01', '2025-12-31'),
  ];
  const crisisCorrs = crisisCombos.map(r => correlation(r.get('SPY') || [], r.get('GLD') || []));
  const avgCrisisCorr = crisisCorrs.reduce((a, b) => a + b, 0) / crisisCorrs.length;

  const diversificationShift = avgCrisisCorr - normalCorr;
  console.log(`Normal SPY/GLD correlation: ${normalCorr.toFixed(3)}`);
  console.log(`Average crisis SPY/GLD correlation: ${avgCrisisCorr.toFixed(3)}`);
  console.log(`Shift during crisis: ${diversificationShift > 0 ? '+' : ''}${diversificationShift.toFixed(3)}`);
  if (diversificationShift < -0.1) {
    console.log('→ Diversification IMPROVES during crises (correlation drops). Gold becomes a better hedge when you need it most.');
  } else if (diversificationShift < 0.1) {
    console.log('→ Diversification is STABLE during crises. Gold hedge works consistently.');
  } else {
    console.log('→ WARNING: Diversification WEAKENS during crises. Gold becomes less effective when needed most.');
  }
}

main().catch(console.error);
