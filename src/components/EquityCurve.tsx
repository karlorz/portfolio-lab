import React, { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Brush } from 'recharts';
import type { BacktestResult, PerformanceMetrics } from '../backtest/engine';
import { autoDownsample } from '../utils/lttb';

interface EquityCurveProps {
  results: Array<{ name: string; result: BacktestResult; metrics: PerformanceMetrics; color: string }>;
}

export const EquityCurve: React.FC<EquityCurveProps> = ({ results }) => {
  // Transform and downsample data for recharts
  const data = useMemo(() => {
    const raw = results[0]?.result.dates.map((date, i) => {
      const point: Record<string, number | string> = { date };
      results.forEach(({ name, result }) => {
        point[name] = result.portfolioValues[i] / result.portfolioValues[0];
      });
      return point;
    }) || [];
    // Downsample if >1000 points (LTTB preserves visual shape)
    return autoDownsample(raw, 600, 'date', results[0]?.name || 'value', 1000);
  }, [results]);

  return (
    <div className="chart-container">
      <h3>Equity Curve (Normalized to 1.0)</h3>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={data} isAnimationActive={false}>
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
          <Brush dataKey="date" height={30} stroke="#3b82f6" tickFormatter={(date) => new Date(date as string).getFullYear().toString()} />
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
