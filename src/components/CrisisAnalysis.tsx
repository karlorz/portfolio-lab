import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';
import type { BacktestResult } from '../backtest/engine';

interface CrisisPeriod {
  name: string;
  start: string;
  end: string;
}

const CRISIS_PERIODS: CrisisPeriod[] = [
  { name: '2008 Financial Crisis', start: '2008-01-01', end: '2009-03-31' },
  { name: '2020 COVID Crash', start: '2020-02-15', end: '2020-05-31' },
  { name: '2022 Rate Hikes', start: '2022-01-01', end: '2022-12-31' },
];

interface CrisisData {
  period: string;
  [portfolio: string]: string | number;
}

interface CrisisAnalysisProps {
  results: Array<{ name: string; result: BacktestResult; color: string }>;
}

function calculateCrisisPerformance(
  result: BacktestResult,
  period: CrisisPeriod
): { return: number; maxDrawdown: number } {
  const startIdx = result.dates.findIndex(d => d >= period.start);
  const endIdx = result.dates.findIndex(d => d >= period.end);
  
  if (startIdx === -1 || endIdx === -1 || startIdx >= result.portfolioValues.length || endIdx >= result.portfolioValues.length) {
    return { return: 0, maxDrawdown: 0 };
  }
  
  const startValue = result.portfolioValues[startIdx];
  const endValue = result.portfolioValues[endIdx];
  const periodValues = result.portfolioValues.slice(startIdx, endIdx + 1);
  const peak = Math.max(...periodValues);
  const trough = Math.min(...periodValues);
  
  return {
    return: (endValue - startValue) / startValue,
    maxDrawdown: (trough - peak) / peak,
  };
}

export const CrisisAnalysis: React.FC<CrisisAnalysisProps> = ({ results }) => {
  const data: CrisisData[] = CRISIS_PERIODS.map(period => {
    const row: CrisisData = { period: period.name };
    results.forEach(({ name, result }) => {
      const perf = calculateCrisisPerformance(result, period);
      row[`${name}_return`] = perf.return * 100;
      row[`${name}_dd`] = perf.maxDrawdown * 100;
    });
    return row;
  });

  // Flatten for display
  const portfolios = results.map(r => r.name);

  return (
    <div className="chart-container">
      <h3>Crisis Period Performance (% Return)</h3>
      <p style={{ color: '#64748b', fontSize: '0.85rem', marginBottom: 15 }}>
        Positive values = portfolio gained value during crisis. Negative = lost value.
      </p>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
          <XAxis dataKey="period" stroke="#94a3b8" />
          <YAxis stroke="#94a3b8" tickFormatter={(v) => `${v.toFixed(0)}%`} />
          <Tooltip
            contentStyle={{ background: '#1e293b', border: '1px solid #334155' }}
            formatter={(value: number, name: string) => [`${value.toFixed(1)}%`, name]}
          />
          <Legend />
          <ReferenceLine y={0} stroke="#64748b" strokeDasharray="3 3" />
          {results.map(({ name, color }) => (
            <Bar
              key={name}
              dataKey={`${name}_return`}
              name={name}
              fill={color}
              radius={[4, 4, 0, 0]}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>

      <h3 style={{ marginTop: 30 }}>Max Drawdown During Crisis Periods</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
          <XAxis dataKey="period" stroke="#94a3b8" />
          <YAxis stroke="#94a3b8" tickFormatter={(v) => `${v.toFixed(0)}%`} domain={['auto', 0]} />
          <Tooltip
            contentStyle={{ background: '#1e293b', border: '1px solid #334155' }}
            formatter={(value: number, name: string) => [`${value.toFixed(1)}%`, name]}
          />
          <Legend />
          {results.map(({ name, color }) => (
            <Bar
              key={name}
              dataKey={`${name}_dd`}
              name={name}
              fill={color}
              radius={[0, 0, 4, 4]}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};
