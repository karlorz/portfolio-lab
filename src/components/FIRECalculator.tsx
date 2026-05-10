import React, { useMemo, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';
import type { BacktestResult } from '../backtest/engine';

interface FIRECalculatorProps {
  results: Array<{ name: string; result: BacktestResult; metrics: any; color: string }>;
}

const WITHDRAWAL_RATES = [0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06];

// GFC stress test: Oct 2007 peak through mid-2014
const GFC_START = '2007-10-01';
const GFC_END = '2014-06-30';

function simulateWithdrawals(
  result: BacktestResult,
  initialAmount: number,
  annualRate: number,
): { minValue: number; endValue: number; minDropPct: number; endGainPct: number } {
  const values = result.portfolioValues;
  const monthlyWithdrawal = (initialAmount * annualRate) / 12;
  let portfolio = initialAmount;
  let minValue = Infinity;
  let finalValue = 0;

  for (let i = 0; i < values.length; i++) {
    if (i > 0) portfolio *= (values[i] / values[i - 1]);
    if (i > 0 && i % 21 === 0) portfolio -= monthlyWithdrawal;
    if (portfolio < minValue) minValue = portfolio;
    if (portfolio <= 0) { finalValue = 0; break; }
    finalValue = portfolio;
  }

  return {
    minValue,
    endValue: finalValue,
    minDropPct: (minValue / initialAmount - 1),
    endGainPct: (finalValue / initialAmount - 1),
  };
}

// Simple Monte Carlo with block bootstrap
function runMonteCarlo(
  dailyReturns: number[],
  initialAmount: number,
  annualRate: number,
  simulations: number = 500,
  years: number = 30,
): { successRate: number; p10: number; p25: number; p50: number; p75: number; p90: number } {
  const totalDays = years * 252;
  const blockSize = 20;
  const monthlyWithdrawal = (initialAmount * annualRate) / 12;
  const endValues: number[] = [];
  const n = dailyReturns.length;

  for (let sim = 0; sim < simulations; sim++) {
    let portfolio = initialAmount;
    let broke = false;

    for (let day = 0; day < totalDays; day++) {
      // Block bootstrap: pick random block
      if (day % blockSize === 0) {
        // Pick a random starting point for this block
        var blockStart = Math.floor(Math.random() * (n - blockSize));
      }
      const idx = blockStart + (day % blockSize);
      const ret = dailyReturns[Math.min(idx, n - 1)];
      portfolio *= (1 + ret);

      // Monthly withdrawal
      if (day > 0 && day % 21 === 0) {
        portfolio -= monthlyWithdrawal;
      }

      if (portfolio <= 0) { broke = true; break; }
    }

    endValues.push(broke ? 0 : portfolio);
  }

  endValues.sort((a, b) => a - b);
  const survived = endValues.filter(v => v > 0).length;

  return {
    successRate: survived / simulations,
    p10: endValues[Math.floor(simulations * 0.10)] ?? 0,
    p25: endValues[Math.floor(simulations * 0.25)] ?? 0,
    p50: endValues[Math.floor(simulations * 0.50)] ?? 0,
    p75: endValues[Math.floor(simulations * 0.75)] ?? 0,
    p90: endValues[Math.floor(simulations * 0.90)] ?? 0,
  };
}

export const FIRECalculator: React.FC<FIRECalculatorProps> = ({ results }) => {
  const [scenario, setScenario] = useState<'gfc' | 'full' | 'montecarlo'>('gfc');
  const [withdrawalRate, setWithdrawalRate] = useState(0.04);

  // GFC withdrawal simulation
  const gfcData = useMemo(() => {
    return results.map(({ name, result, color }) => {
      // Find GFC date range within the result
      const startIdx = result.dates.findIndex(d => d >= GFC_START);
      const endIdx = result.dates.findIndex(d => d >= GFC_END);

      if (startIdx === -1 || endIdx === -1) {
        return { name, color, values: null };
      }

      // Extract the GFC sub-result
      const gfcResult: BacktestResult = {
        dates: result.dates.slice(startIdx, endIdx + 1),
        portfolioValues: result.portfolioValues.slice(startIdx, endIdx + 1),
        returns: result.returns.slice(startIdx, endIdx + 1),
        drawdowns: result.drawdowns.slice(startIdx, endIdx + 1),
        holdings: result.holdings.slice(startIdx, endIdx + 1),
        trades: result.trades,
      };

      const sim = simulateWithdrawals(gfcResult, 1000000, withdrawalRate);
      return { name, color, values: sim };
    });
  }, [results, withdrawalRate]);

  // Full period withdrawal simulation
  const fullData = useMemo(() => {
    return results.map(({ name, result, color }) => {
      const sim = simulateWithdrawals(result, 1000000, withdrawalRate);
      return { name, color, values: sim };
    });
  }, [results, withdrawalRate]);

  // Monte Carlo simulation (lighter weight - 200 sims)
  const mcData = useMemo(() => {
    return results.map(({ name, result, color }) => {
      const dailyReturns = result.returns.slice(1);
      const mc = runMonteCarlo(dailyReturns, 1000000, withdrawalRate, 200, 30);
      return { name, color, mc };
    });
  }, [results, withdrawalRate]);

  // Withdrawal rate comparison chart data
  const rateComparisonData = useMemo(() => {
    return WITHDRAWAL_RATES.map(rate => {
      const row: any = { rate: `${(rate * 100).toFixed(1)}%` };
      results.forEach(({ name, result }) => {
        // Use full period for rate comparison
        const sim = simulateWithdrawals(result, 1000000, rate);
        row[name] = sim.endGainPct * 100;
      });
      return row;
    });
  }, [results]);

  const currentData = scenario === 'gfc' ? gfcData : scenario === 'full' ? fullData : null;

  return (
    <div className="chart-container">
      <h3>FIRE Withdrawal Calculator</h3>
      <p style={{ color: '#64748b', fontSize: '0.85rem', marginBottom: 15 }}>
        Starting with $1M, how much can you safely withdraw?
      </p>

      {/* Controls */}
      <div style={{ display: 'flex', gap: 20, marginBottom: 20, flexWrap: 'wrap' }}>
        <div>
          <label style={{ color: '#94a3b8', fontSize: '0.85rem', display: 'block', marginBottom: 5 }}>
            Withdrawal Rate
          </label>
          <div style={{ display: 'flex', gap: 6 }}>
            {WITHDRAWAL_RATES.map(rate => (
              <button
                key={rate}
                onClick={() => setWithdrawalRate(rate)}
                style={{
                  padding: '6px 10px',
                  background: withdrawalRate === rate ? '#3b82f6' : '#0f172a',
                  color: withdrawalRate === rate ? '#fff' : '#94a3b8',
                  border: '1px solid #334155',
                  borderRadius: 4,
                  cursor: 'pointer',
                  fontSize: '0.85rem',
                }}
              >
                {(rate * 100).toFixed(1)}%
              </button>
            ))}
          </div>
        </div>

        <div>
          <label style={{ color: '#94a3b8', fontSize: '0.85rem', display: 'block', marginBottom: 5 }}>
            Scenario
          </label>
          <div style={{ display: 'flex', gap: 6 }}>
            {([
              { key: 'gfc', label: 'GFC Stress Test' },
              { key: 'full', label: 'Full 2005-2026' },
              { key: 'montecarlo', label: 'Monte Carlo (30yr)' },
            ] as const).map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setScenario(key)}
                style={{
                  padding: '6px 10px',
                  background: scenario === key ? '#3b82f6' : '#0f172a',
                  color: scenario === key ? '#fff' : '#94a3b8',
                  border: '1px solid #334155',
                  borderRadius: 4,
                  cursor: 'pointer',
                  fontSize: '0.85rem',
                }}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Results table */}
      {scenario !== 'montecarlo' && currentData && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #334155' }}>
                <th style={{ textAlign: 'left', padding: '10px 12px', color: '#94a3b8' }}>Portfolio</th>
                <th style={{ textAlign: 'right', padding: '10px 12px', color: '#94a3b8' }}>Worst Drop</th>
                <th style={{ textAlign: 'right', padding: '10px 12px', color: '#94a3b8' }}>Low Value</th>
                <th style={{ textAlign: 'right', padding: '10px 12px', color: '#94a3b8' }}>End Value</th>
                <th style={{ textAlign: 'right', padding: '10px 12px', color: '#94a3b8' }}>Net Gain</th>
              </tr>
            </thead>
            <tbody>
              {currentData.map(({ name, color, values }) => values && (
                <tr key={name} style={{ borderBottom: '1px solid #1e293b' }}>
                  <td style={{ padding: '10px 12px', color }}>{name}</td>
                  <td style={{
                    textAlign: 'right', padding: '10px 12px',
                    color: values.minDropPct < -0.3 ? '#ef4444' : values.minDropPct < -0.15 ? '#f59e0b' : '#10b981',
                    fontFamily: 'SF Mono, monospace',
                  }}>
                    {(values.minDropPct * 100).toFixed(1)}%
                  </td>
                  <td style={{ textAlign: 'right', padding: '10px 12px', fontFamily: 'SF Mono, monospace' }}>
                    ${values.minValue < 0 ? '0' : (values.minValue / 1000).toFixed(0)}K
                  </td>
                  <td style={{
                    textAlign: 'right', padding: '10px 12px',
                    color: values.endGainPct >= 0 ? '#10b981' : '#ef4444',
                    fontFamily: 'SF Mono, monospace',
                  }}>
                    ${values.endValue < 0 ? '0' : (values.endValue / 1000).toFixed(0)}K
                  </td>
                  <td style={{
                    textAlign: 'right', padding: '10px 12px',
                    color: values.endGainPct >= 0 ? '#10b981' : '#ef4444',
                    fontFamily: 'SF Mono, monospace',
                  }}>
                    {values.endGainPct >= 0 ? '+' : ''}{(values.endGainPct * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p style={{ color: '#64748b', fontSize: '0.8rem', marginTop: 8 }}>
            Monthly pro-rata withdrawals from $1M starting portfolio
          </p>
        </div>
      )}

      {/* Monte Carlo results */}
      {scenario === 'montecarlo' && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #334155' }}>
                <th style={{ textAlign: 'left', padding: '10px 12px', color: '#94a3b8' }}>Portfolio</th>
                <th style={{ textAlign: 'right', padding: '10px 12px', color: '#94a3b8' }}>Survival%</th>
                <th style={{ textAlign: 'right', padding: '10px 12px', color: '#94a3b8' }}>P10 (worst)</th>
                <th style={{ textAlign: 'right', padding: '10px 12px', color: '#94a3b8' }}>Median</th>
                <th style={{ textAlign: 'right', padding: '10px 12px', color: '#94a3b8' }}>P90 (best)</th>
              </tr>
            </thead>
            <tbody>
              {mcData.map(({ name, color, mc }) => (
                <tr key={name} style={{ borderBottom: '1px solid #1e293b' }}>
                  <td style={{ padding: '10px 12px', color }}>{name}</td>
                  <td style={{
                    textAlign: 'right', padding: '10px 12px',
                    color: mc.successRate >= 0.99 ? '#10b981' : mc.successRate >= 0.95 ? '#f59e0b' : '#ef4444',
                    fontFamily: 'SF Mono, monospace',
                  }}>
                    {(mc.successRate * 100).toFixed(0)}%
                  </td>
                  <td style={{ textAlign: 'right', padding: '10px 12px', fontFamily: 'SF Mono, monospace', color: '#94a3b8' }}>
                    ${mc.p10 <= 0 ? '0' : (mc.p10 / 1000).toFixed(0)}K
                  </td>
                  <td style={{ textAlign: 'right', padding: '10px 12px', fontFamily: 'SF Mono, monospace' }}>
                    ${(mc.p50 / 1000).toFixed(0)}K
                  </td>
                  <td style={{ textAlign: 'right', padding: '10px 12px', fontFamily: 'SF Mono, monospace', color: '#64748b' }}>
                    ${(mc.p90 / 1000).toFixed(0)}K
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p style={{ color: '#64748b', fontSize: '0.8rem', marginTop: 8 }}>
            200 bootstrap simulations, 30-year retirement, block size 20 days
          </p>
        </div>
      )}

      {/* Withdrawal rate comparison chart */}
      <h3 style={{ marginTop: 30 }}>Net Gain by Withdrawal Rate (Full Period)</h3>
      <p style={{ color: '#64748b', fontSize: '0.85rem', marginBottom: 15 }}>
        Higher = more money left after 20 years of withdrawals
      </p>
      <ResponsiveContainer width="100%" height={350}>
        <BarChart data={rateComparisonData} margin={{ top: 10, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
          <XAxis dataKey="rate" stroke="#94a3b8" />
          <YAxis stroke="#94a3b8" tickFormatter={(v) => `${v.toFixed(0)}%`} />
          <Tooltip
            contentStyle={{ background: '#1e293b', border: '1px solid #334155' }}
            formatter={(value: number) => [`${value.toFixed(1)}%`]}
          />
          <ReferenceLine y={0} stroke="#64748b" strokeDasharray="3 3" />
          {results.map(({ name, color }) => (
            <Bar key={name} dataKey={name} fill={color} radius={[4, 4, 0, 0]} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};
