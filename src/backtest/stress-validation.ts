/**
 * Stress Period Holdout Validation
 * Validates portfolio configurations against known crisis periods
 * Never trains on stress periods — pure out-of-sample testing
 */

import { BacktestEngine } from './engine';
import type { PortfolioConfig, PriceData } from './engine';

export interface StressPeriod {
  name: string;
  startDate: string;
  endDate: string;
  description: string;
  maxDDThreshold: number;
}

export interface StressValidationResult {
  portfolioName: string;
  stressPeriod: string;
  return: number;
  maxDrawdown: number;
  passesThreshold: boolean;
  recoveryDays: number | null;
}

// Known crisis periods for validation
export const STRESS_PERIODS: StressPeriod[] = [
  {
    name: 'GFC 2008',
    startDate: '2008-09-01',
    endDate: '2009-03-31',
    description: 'Global Financial Crisis peak',
    maxDDThreshold: -0.30, // 30% max drawdown limit
  },
  {
    name: 'COVID 2020',
    startDate: '2020-02-01',
    endDate: '2020-04-30',
    description: 'COVID-19 market crash',
    maxDDThreshold: -0.25,
  },
  {
    name: 'Rate Hikes 2022',
    startDate: '2022-01-01',
    endDate: '2022-10-31',
    description: 'Fed rate hiking cycle',
    maxDDThreshold: -0.25,
  },
  {
    name: 'Taper Tantrum 2013',
    startDate: '2013-05-01',
    endDate: '2013-06-30',
    description: 'QE tapering scare',
    maxDDThreshold: -0.15,
  },
  {
    name: 'China Devaluation 2015',
    startDate: '2015-08-01',
    endDate: '2015-09-30',
    description: 'CNY devaluation impact',
    maxDDThreshold: -0.15,
  },
  {
    name: 'VIX Spike 2018',
    startDate: '2018-01-01',
    endDate: '2018-03-31',
    description: 'Volmageddon / VIX spike',
    maxDDThreshold: -0.15,
  },
  {
    name: 'Russia-Ukraine 2022',
    startDate: '2022-02-01',
    endDate: '2022-04-30',
    description: 'Geopolitical shock',
    maxDDThreshold: -0.20,
  },
];

/**
 * Validate portfolio configuration against all stress periods
 */
export function validateStressPeriods(
  config: PortfolioConfig,
  prices: Record<string, Array<{ d: string; p: number }>>,
  stressPeriods: StressPeriod[] = STRESS_PERIODS
): StressValidationResult[] {
  const engine = new BacktestEngine();
  engine.loadData(toBacktestData(prices));
  const results: StressValidationResult[] = [];

  for (const period of stressPeriods) {
    try {
      const backtest = engine.runBacktest(config, period.startDate, period.endDate, 10000);
      
      // Calculate return
      const finalValue = backtest.portfolioValues[backtest.portfolioValues.length - 1] ?? 10000;
      const initialValue = backtest.portfolioValues[0] ?? 10000;
      const totalReturn = (finalValue / initialValue) - 1;

      // Calculate max drawdown
      const maxDrawdown = calculateMaxDrawdown(backtest.portfolioValues);

      // Calculate recovery days (if applicable)
      const recoveryDays = calculateRecoveryDays(
        backtest.portfolioValues,
        backtest.dates.map(d => new Date(d))
      );

      results.push({
        portfolioName: config.name,
        stressPeriod: period.name,
        return: totalReturn,
        maxDrawdown,
        passesThreshold: maxDrawdown >= period.maxDDThreshold,
        recoveryDays,
      });
    } catch (error) {
      console.warn(`Stress validation failed for ${config.name} during ${period.name}:`, error);
      results.push({
        portfolioName: config.name,
        stressPeriod: period.name,
        return: 0,
        maxDrawdown: -1,
        passesThreshold: false,
        recoveryDays: null,
      });
    }
  }

  return results;
}

/**
 * Batch validate multiple configurations
 */
export function batchStressValidation(
  configs: PortfolioConfig[],
  prices: Record<string, Array<{ d: string; p: number }>>,
  stressPeriods: StressPeriod[] = STRESS_PERIODS
): StressValidationResult[] {
  const allResults: StressValidationResult[] = [];

  for (const config of configs) {
    const results = validateStressPeriods(config, prices, stressPeriods);
    allResults.push(...results);
  }

  return allResults;
}

/**
 * Generate stress validation summary report
 */
export function generateStressReport(
  results: StressValidationResult[]
): {
  summary: Record<string, { passed: number; failed: number; avgDD: number }>;
  failures: StressValidationResult[];
  recommendations: string[];
} {
  const summary: Record<string, { passed: number; failed: number; avgDD: number }> = {};
  const failures: StressValidationResult[] = [];

  for (const result of results) {
    if (!summary[result.stressPeriod]) {
      summary[result.stressPeriod] = { passed: 0, failed: 0, avgDD: 0 };
    }

    if (result.passesThreshold) {
      summary[result.stressPeriod].passed++;
    } else {
      summary[result.stressPeriod].failed++;
      failures.push(result);
    }
    summary[result.stressPeriod].avgDD += result.maxDrawdown;
  }

  // Average drawdown per period
  for (const period of Object.keys(summary)) {
    const count = summary[period].passed + summary[period].failed;
    summary[period].avgDD = count > 0 ? summary[period].avgDD / count : 0;
  }

  // Generate recommendations
  const recommendations: string[] = [];
  
  const failureCounts = failures.reduce((acc, f) => {
    acc[f.portfolioName] = (acc[f.portfolioName] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  for (const [portfolio, count] of Object.entries(failureCounts)) {
    if (count >= 2) {
      recommendations.push(`${portfolio}: Fails ${count} stress tests — consider reducing equity exposure`);
    }
  }

  const gfcFailures = failures.filter(f => f.stressPeriod === 'GFC 2008');
  if (gfcFailures.length > 0) {
    recommendations.push('GFC 2008 remains challenging — ensure adequate bond/gold allocation');
  }

  return { summary, failures, recommendations };
}

// Helper functions
function toBacktestData(prices: Record<string, Array<{ d: string; p: number }>>): PriceData[] {
  const result: PriceData[] = [];
  for (const [symbol, entries] of Object.entries(prices)) {
    for (const { d, p } of entries) {
      result.push({ date: d, symbol, price: p });
    }
  }
  return result.sort((a, b) => a.date.localeCompare(b.date));
}

function calculateMaxDrawdown(values: number[]): number {
  let peak = values[0] || 0;
  let maxDD = 0;

  for (const value of values) {
    if (value > peak) {
      peak = value;
    }
    const drawdown = (value - peak) / peak;
    if (drawdown < maxDD) {
      maxDD = drawdown;
    }
  }

  return maxDD;
}

function calculateRecoveryDays(values: number[], dates: Date[]): number | null {
  if (values.length < 2 || dates.length !== values.length) return null;

  let peak = values[0];
  let peakIndex = 0;
  let inDrawdown = false;
  let recoveryDays: number | null = null;

  for (let i = 1; i < values.length; i++) {
    if (values[i] > peak) {
      peak = values[i];
      peakIndex = i;
      inDrawdown = false;
    } else {
      inDrawdown = true;
    }

    // Recovery to new high
    if (inDrawdown && values[i] >= peak) {
      const days = (dates[i].getTime() - dates[peakIndex].getTime()) / (1000 * 60 * 60 * 24);
      if (recoveryDays === null || days > recoveryDays) {
        recoveryDays = Math.round(days);
      }
    }
  }

  return recoveryDays;
}
