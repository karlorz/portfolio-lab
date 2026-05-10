import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { BacktestResult, PerformanceMetrics } from '../backtest/engine';

interface EquityCurveProps {
  results: Array<{ name: string; result: BacktestResult; metrics: PerformanceMetrics; color: string }>;
}

export const EquityCurve: React.FC<EquityCurveProps> = ({ results }) => {
  // Transform data for recharts
  const data = results[0]?.result.dates.map((date, i) => {
    const point: Record<string, number | string> = { date };
    results.forEach(({ name, result }) => {
      point[name] = result.portfolioValues[i] / result.portfolioValues[0];
    });
    return point;
  }) || [];

  return (
    <div className="chart-container">
      <h3>Equity Curve (Normalized to 1.0)</h3>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={data}>
          <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
          <XAxis 
            dataKey="date" 
            stroke="#94a3b8"
            tickFormatter={(date) => new Date(date as string).getFullYear().toString()}
            minTickGap={50}
          />
          <YAxis stroke="#94a3b8" domain={['auto', 'auto']} />
          <Tooltip 
            contentStyle={{ background: '#1e293b', border: '1px solid #334155' }}
            labelStyle={{ color: '#e2e8f0' }}
          />
          <Legend />
          {results.map(({ name, color }) => (
            <Line
              key={name}
              type="monotone"
              dataKey={name}
              stroke={color}
              strokeWidth={2}
              dot={false}
              name={name}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};
