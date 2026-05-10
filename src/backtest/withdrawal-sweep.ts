/**
 * Withdrawal Rate Sweep — test sustainable withdrawal rates for FIRE
 * Answers: is 4% conservative for all-season portfolios? What about 4.5%, 5%, 5.5%?
 * Tests across GFC (worst case) and full period (2005-2026)
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

interface WithdrawalResult {
  portfolio: string;
  rate: number;
  scenario: string;
  startValue: number;
  minValue: number;
  minDropPct: number;
  endValue: number;
  endGainPct: number;
  broke: boolean;
  brokeDate: string;
  yearsToRecover: number | null; // years until portfolio back above start value
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

const WITHDRAWAL_RATES = [0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06];

const SCENARIOS: { name: string; start: string; end: string }[] = [
  { name: 'GFC (2007-2014)', start: '2007-10-01', end: '2014-06-30' },
  { name: 'Full Period (2005-2026)', start: '2005-01-03', end: '2026-05-08' },
  { name: 'Post-GFC (2010-2026)', start: '2010-01-04', end: '2026-05-08' },
  { name: '2020 Crisis (2019-2023)', start: '2019-01-02', end: '2023-12-29' },
];

function simulateWithdrawals(
  engine: BacktestEngine,
  config: PortfolioConfig,
  startDate: string,
  endDate: string,
  initialAmount: number,
  annualRate: number,
): WithdrawalResult {
  const result = engine.runBacktest(config, startDate, endDate, initialAmount);
  const values = result.portfolioValues;

  let portfolio = initialAmount;
  let broke = false;
  let brokeDate = '';
  let minValue = Infinity;
  let finalValue = 0;
  let recovered = false;
  let yearsToRecover: number | null = null;
  const monthlyWithdrawal = (initialAmount * annualRate) / 12;

  for (let i = 0; i < values.length; i++) {
    if (i > 0) {
      portfolio *= (values[i] / values[i - 1]);
    }
    // Monthly withdrawal (every ~21 trading days)
    if (i > 0 && i % 21 === 0) {
      portfolio -= monthlyWithdrawal;
    }
    if (portfolio < minValue) minValue = portfolio;
    if (portfolio <= 0 && !broke) {
      broke = true;
      brokeDate = result.dates[i];
    }
    // Check if portfolio recovered to starting value
    if (!recovered && portfolio >= initialAmount && i > 0) {
      recovered = true;
      yearsToRecover = i / 252;
    }
    finalValue = portfolio;
    if (broke) break;
  }

  return {
    portfolio: config.name,
    rate: annualRate,
    scenario: '',
    startValue: initialAmount,
    minValue,
    minDropPct: (minValue / initialAmount - 1),
    endValue: finalValue,
    endGainPct: (finalValue / initialAmount - 1),
    broke,
    brokeDate,
    yearsToRecover,
  };
}

async function main() {
  console.log('Loading price data...\n');
  const response = await fetch('file://' + new URL('../../public/data/prices.json', import.meta.url).pathname);
  const priceJson = await response.json() as Record<string, Array<{ d: string; p: number }>>;
  const priceData = toBacktestData(priceJson);

  const engine = new BacktestEngine();
  engine.loadData(priceData);

  console.log('=== WITHDRAWAL RATE SWEEP FOR FIRE ===\n');
  console.log('Testing sustainable withdrawal rates across scenarios\n');
  console.log('Starting portfolio: $1,000,000 | Monthly pro-rata withdrawals\n');

  const allResults: WithdrawalResult[] = [];

  for (const scenario of SCENARIOS) {
    console.log(`\n${'='.repeat(70)}`);
    console.log(`SCENARIO: ${scenario.name} (${scenario.start} to ${scenario.end})`);
    console.log('='.repeat(70));

    for (const rate of WITHDRAWAL_RATES) {
      console.log(`\n--- ${(rate * 100).toFixed(1)}% Withdrawal Rate ---`);
      console.log('| Portfolio | Low Value | Worst Drop | End Value | Net Gain | Recovered? | Years to Recover |');
      console.log('|-----------|-----------|------------|-----------|----------|------------|------------------|');

      for (const config of PORTFOLIOS) {
        const r = simulateWithdrawals(engine, config, scenario.start, scenario.end, 1000000, rate);
        r.scenario = scenario.name;
        allResults.push(r);

        const low = `$${(r.minValue / 1000).toFixed(0)}k`;
        const drop = `${(r.minDropPct * 100).toFixed(1)}%`;
        const end = `$${(r.endValue / 1000).toFixed(0)}k`;
        const gain = `${(r.endGainPct * 100).toFixed(1)}%`;
        const recovered = r.broke ? 'BROKE' : (r.yearsToRecover !== null ? 'Yes' : 'No');
        const years = r.yearsToRecover !== null ? r.yearsToRecover.toFixed(1) : (r.broke ? 'N/A' : '—');

        console.log(`| ${r.portfolio} | ${low} | ${drop} | ${end} | ${gain} | ${recovered} | ${years} |`);
      }
    }
  }

  // Summary: maximum safe withdrawal rate per portfolio
  console.log('\n\n' + '='.repeat(70));
  console.log('SUMMARY: MAXIMUM SAFE WITHDRAWAL RATE');
  console.log('(Highest rate where portfolio does NOT go broke in any scenario)');
  console.log('='.repeat(70));

  for (const config of PORTFOLIOS) {
    const portfolioResults = allResults.filter(r => r.portfolio === config.name);
    const safeRates = WITHDRAWAL_RATES.filter(rate =>
      !portfolioResults.some(r => r.rate === rate && r.broke)
    );
    const maxSafe = safeRates.length > 0 ? Math.max(...safeRates) : 0;
    const brokeAt = WITHDRAWAL_RATES.find(rate =>
      portfolioResults.some(r => r.rate === rate && r.broke)
    );

    // For each rate, show worst-case drawdown across all scenarios
    console.log(`\n${config.name}:`);
    for (const rate of WITHDRAWAL_RATES) {
      const rateResults = portfolioResults.filter(r => r.rate === rate);
      const worstCase = rateResults.reduce((worst, r) =>
        r.minDropPct < worst.minDropPct ? r : worst, rateResults[0]);
      const anyBroke = rateResults.some(r => r.broke);
      const marker = rate === maxSafe ? ' ← MAX SAFE' : '';
      const brokeMarker = anyBroke ? ' ⚠️ BROKE' : '';
      console.log(`  ${(rate * 100).toFixed(1)}%: worst drop ${(worstCase.minDropPct * 100).toFixed(1)}% (${worstCase.scenario})${brokeMarker}${marker}`);
    }
  }

  // FIRE-specific: years to recover starting value at 4% and 5%
  console.log('\n\n' + '='.repeat(70));
  console.log('RECOVERY TO STARTING VALUE (years until portfolio > $1M again)');
  console.log('='.repeat(70));

  for (const rate of [0.04, 0.045, 0.05]) {
    console.log(`\n${(rate * 100).toFixed(1)}% withdrawal rate:`);
    console.log('| Portfolio | GFC | Full Period | Post-GFC | 2020 Crisis |');
    console.log('|-----------|-----|-------------|----------|-------------|');

    for (const config of PORTFOLIOS) {
      const rateResults = allResults.filter(r =>
        r.portfolio === config.name && r.rate === rate
      );
      const cols = SCENARIOS.map(s => {
        const r = rateResults.find(x => x.scenario === s.name);
        if (!r) return '—';
        if (r.broke) return 'BROKE';
        if (r.yearsToRecover !== null) return `${r.yearsToRecover.toFixed(1)}yr`;
        return 'no';
      });
      console.log(`| ${config.name} | ${cols.join(' | ')} |`);
    }
  }
}

main().catch(console.error);
