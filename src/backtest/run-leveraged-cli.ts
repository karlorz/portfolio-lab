/**
 * Execute leveraged treasury backtest and save results
 */
import { generateSampleBacktest, analyzeScenarios, BacktestResult } from './leveraged-treasury-backtest';
import * as fs from 'fs';
import * as path from 'path';

console.log('[INFO] Starting leveraged treasury backtest execution...');

// Run the backtest
const results = generateSampleBacktest();

// Analyze scenarios
const analysis = analyzeScenarios(results);

// Prepare output
const output = {
  timestamp: new Date().toISOString(),
  version: 'v2.35',
  results: results.map((r: BacktestResult) => ({
    scenario: r.scenario,
    cagr: r.cagr,
    volatility: r.volatility,
    sharpe: r.sharpe,
    maxDrawdown: r.maxDrawdown,
    calmar: r.calmar,
    totalReturn: r.totalReturn,
    annualizedDrag: r.annualizedDrag,
    capitalFreed: r.capitalFreed,
    regimeTransitions: r.regimeTransitions,
    stressTests: r.stressTests.map(s => ({
      name: s.name,
      period: s.period,
      tltReturn: s.tltReturn,
      simulatedUBT: s.simulatedUBT,
      simulatedTMF: s.simulatedTMF,
      drawdownTLT: s.drawdownTLT,
      drawdownUBT: s.drawdownUBT,
      drawdownTMF: s.drawdownTMF
    }))
  })),
  recommendation: analysis.recommended,
  reasoning: analysis.reasoning,
  metrics: analysis.metrics
};

// Save to data directory
const outputPath = path.join(process.cwd(), 'data', 'ubt_backtest_results.json');
fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));

console.log(`[SUCCESS] Backtest results saved to: ${outputPath}`);
console.log(`\n=== BACKTEST SUMMARY ===`);
console.log(`Recommended Scenario: ${analysis.recommended}`);
console.log(`Reasoning: ${analysis.reasoning}`);
console.log(`\n=== SCENARIO METRICS ===`);
for (const [scenario, m] of Object.entries(analysis.metrics)) {
  console.log(`${scenario}:`);
  console.log(`  CAGR: ${(m.cagr * 100).toFixed(2)}%`);
  console.log(`  Sharpe: ${m.sharpe.toFixed(2)}`);
  console.log(`  Max DD: ${(m.maxDD * 100).toFixed(1)}%`);
}
