/**
 * Drawdown Recovery Analysis — how long each portfolio stays underwater
 * Critical for FIRE: sequence-of-returns risk means recovery time matters more than max DD
 */
import { BacktestEngine } from './engine';
import type { PortfolioConfig, PriceData } from './engine';

function toBacktestData(prices: Record<string, Array<{ d: string; p: number }>>): PriceData[] {
  const result: PriceData[] = [];
  for (const [symbol, entries] of Object.entries(prices)) {
    for (const { d, p } of entries) {
      result.push({ date: d, symbol, price: p });
    }
  }
  return result.sort((a, b) => a.date.localeCompare(b.date));
}

interface DrawdownEvent {
  peakDate: string;
  troughDate: string;
  recoveryDate: string | null;
  depth: number;
  durationToTrough: number; // trading days
  durationToRecovery: number | null; // trading days (null = not yet recovered)
  stillUnderwater: boolean;
}

function findDrawdownEvents(values: number[], dates: string[], threshold: number = -0.10): DrawdownEvent[] {
  const events: DrawdownEvent[] = [];
  let peak = values[0];
  let peakIdx = 0;
  let inDrawdown = false;
  let troughIdx = 0;

  for (let i = 1; i < values.length; i++) {
    if (values[i] >= peak) {
      if (inDrawdown) {
        const depth = (values[troughIdx] - peak) / peak;
        if (depth <= threshold) {
          events.push({
            peakDate: dates[peakIdx],
            troughDate: dates[troughIdx],
            recoveryDate: dates[i],
            depth,
            durationToTrough: troughIdx - peakIdx,
            durationToRecovery: i - peakIdx,
            stillUnderwater: false,
          });
        }
        inDrawdown = false;
      }
      peak = values[i];
      peakIdx = i;
    } else {
      if (!inDrawdown) {
        inDrawdown = true;
        troughIdx = i;
      } else if (values[i] < values[troughIdx]) {
        troughIdx = i;
      }
    }
  }

  // If still in drawdown at end
  if (inDrawdown) {
    const depth = (values[troughIdx] - peak) / peak;
    if (depth <= threshold) {
      events.push({
        peakDate: dates[peakIdx],
        troughDate: dates[troughIdx],
        recoveryDate: null,
        depth,
        durationToTrough: troughIdx - peakIdx,
        durationToRecovery: null,
        stillUnderwater: true,
      });
    }
  }

  return events;
}

const PORTFOLIOS: PortfolioConfig[] = [
  { name: 'SPY (S&P 500)', allocation: { SPY: 1 }, rebalanceFrequency: 'none' },
  { name: 'SPY/GLD 55/45', allocation: { SPY: 0.55, GLD: 0.45 }, rebalanceFrequency: 'annual' },
  { name: 'SPY/GLD/TLT 46/38/16', allocation: { SPY: 0.46, GLD: 0.38, TLT: 0.16 }, rebalanceFrequency: 'annual' },
  { name: 'SPY/GLD/TLT 50/35/15', allocation: { SPY: 0.50, GLD: 0.35, TLT: 0.15 }, rebalanceFrequency: 'annual' },
  { name: 'All Weather', allocation: { VTI: 0.30, TLT: 0.40, IEF: 0.15, GLD: 0.075, DBC: 0.075 }, rebalanceFrequency: 'annual' },
  { name: 'Golden Butterfly', allocation: { VTI: 0.20, VBR: 0.20, TLT: 0.20, SHY: 0.20, GLD: 0.20 }, rebalanceFrequency: 'annual' },
  { name: '60/40', allocation: { SPY: 0.6, AGG: 0.4 }, rebalanceFrequency: 'annual' },
  { name: 'SPY/EFA/GLD/TLT 36/10/38/16', allocation: { SPY: 0.36, EFA: 0.10, GLD: 0.38, TLT: 0.16 }, rebalanceFrequency: 'annual' },
];

async function main() {
  console.log('Loading price data...\n');
  const response = await fetch('file://' + new URL('../../public/data/prices.json', import.meta.url).pathname);
  const priceJson = await response.json() as Record<string, Array<{ d: string; p: number }>>;
  const priceData = toBacktestData(priceJson);

  const engine = new BacktestEngine();
  engine.loadData(priceData);

  const dates = priceData.map(p => p.date).sort();
  const startDate = dates[0];
  const endDate = dates[dates.length - 1];

  console.log('=== DRAWDOWN RECOVERY ANALYSIS (2005-2026) ===\n');
  console.log('Drawdown events with depth ≥ 10%\n');

  // Per-portfolio drawdown events
  const allEvents = new Map<string, DrawdownEvent[]>();

  for (const config of PORTFOLIOS) {
    const result = engine.runBacktest(config, startDate, endDate, 10000);
    const events = findDrawdownEvents(result.portfolioValues, result.dates);
    allEvents.set(config.name, events);
  }

  // Print all drawdown events per portfolio
  for (const [name, events] of allEvents) {
    console.log(`\n--- ${name}: ${events.length} drawdown events (≥10%) ---`);
    console.log('| Peak Date | Trough Date | Recovery Date | Depth | Days to Trough | Days to Recovery | Years to Recovery |');
    console.log('|-----------|-------------|---------------|-------|----------------|-----------------|-------------------|');

    for (const e of events) {
      const depth = (e.depth * 100).toFixed(1);
      const daysToTrough = e.durationToTrough;
      const daysToRecovery = e.durationToRecovery ?? '—';
      const yearsToRecovery = e.durationToRecovery
        ? (e.durationToRecovery / 252).toFixed(1)
        : '—';
      const recovery = e.recoveryDate ?? 'STILL UNDERWATER';
      console.log(`| ${e.peakDate} | ${e.troughDate} | ${recovery} | ${depth}% | ${daysToTrough} | ${daysToRecovery} | ${yearsToRecovery} |`);
    }
  }

  // Summary comparison
  console.log('\n\n=== RECOVERY TIME SUMMARY ===\n');
  console.log('| Portfolio | # Drawdowns (≥10%) | Avg Recovery (years) | Max Recovery (years) | Worst Depth | Still Underwater? |');
  console.log('|-----------|--------------------|-----------------------|----------------------|-------------|-------------------|');

  for (const [name, events] of allEvents) {
    if (events.length === 0) {
      console.log(`| ${name} | 0 | — | — | — | no |`);
      continue;
    }
    const recoveredEvents = events.filter(e => !e.stillUnderwater && e.durationToRecovery !== null);
    const avgRecovery = recoveredEvents.length > 0
      ? (recoveredEvents.reduce((s, e) => s + (e.durationToRecovery! / 252), 0) / recoveredEvents.length).toFixed(1)
      : '—';
    const maxRecovery = recoveredEvents.length > 0
      ? Math.max(...recoveredEvents.map(e => e.durationToRecovery! / 252)).toFixed(1)
      : '—';
    const worstDepth = Math.min(...events.map(e => e.depth));
    const stillUnderwater = events.some(e => e.stillUnderwater);
    console.log(`| ${name} | ${events.length} | ${avgRecovery} | ${maxRecovery} | ${(worstDepth * 100).toFixed(1)}% | ${stillUnderwater ? 'YES ⚠️' : 'no'} |`);
  }

  // FIRE impact: worst-case scenario
  console.log('\n\n=== FIRE SEQUENCE-OF-RETURNS RISK ===\n');
  console.log('If you retired at the worst possible time (peak before GFC), when would you recover?\n');

  for (const config of PORTFOLIOS) {
    const result = engine.runBacktest(config, startDate, endDate, 10000);
    const events = findDrawdownEvents(result.portfolioValues, result.dates, -0.05);
    const gfcEvent = events.find(e => e.peakDate >= '2007-01-01' && e.peakDate <= '2008-06-30');

    if (gfcEvent) {
      const yearsToRecovery = gfcEvent.durationToRecovery ? (gfcEvent.durationToRecovery / 252).toFixed(1) : 'NEVER';
      const depth = (gfcEvent.depth * 100).toFixed(1);
      console.log(`${config.name}: GFC drawdown ${depth}%, recovery in ${yearsToRecovery} years (peak ${gfcEvent.peakDate} → recovery ${gfcEvent.recoveryDate ?? 'never'})`);
    }
  }

  // Spending during drawdown simulation
  console.log('\n\n=== 4% WITHDRAWAL DURING GFC (starting $1M, peak before crash) ===\n');
  const gfcStart = '2007-10-01';
  const gfcEnd = '2014-06-30'; // Several years after crisis

  for (const config of PORTFOLIOS) {
    const result = engine.runBacktest(config, gfcStart, gfcEnd, 1000000);
    const values = result.portfolioValues;

    // Simulate 4% annual withdrawal (in monthly installments)
    let portfolio = 1000000;
    const annualWithdrawal = 40000;
    let broke = false;
    let brokeDate = '';
    let minValue = Infinity;
    let finalValue = 0;

    for (let i = 0; i < values.length; i++) {
      // Track value change
      if (i > 0) {
        portfolio *= (values[i] / values[i - 1]);
      }
      // Monthly withdrawal
      if (i > 0 && i % 21 === 0) {
        portfolio -= annualWithdrawal / 12;
      }
      if (portfolio < minValue) minValue = portfolio;
      if (portfolio <= 0 && !broke) {
        broke = true;
        brokeDate = result.dates[i];
      }
      finalValue = portfolio;
    }

    const minDrop = ((minValue / 1000000 - 1) * 100).toFixed(1);
    const finalPct = ((finalValue / 1000000 - 1) * 100).toFixed(1);
    console.log(`${config.name}: Low $${(minValue/1000).toFixed(0)}k (${minDrop}%), End $${(finalValue/1000).toFixed(0)}k (${finalPct}%)${broke ? ` ⚠️ BROKE: ${brokeDate}` : ''}`);
  }
}

main().catch(console.error);
