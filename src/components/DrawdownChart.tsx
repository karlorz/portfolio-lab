import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { BacktestResult, PerformanceMetrics } from '../backtest/engine';

interface DrawdownProps {
  results: Array<{ name: string; result: BacktestResult; metrics: PerformanceMetrics; color: string }>;
}

export const DrawdownChart: React.FC<DrawdownProps> = ({ results }) => {
  const data = results[0]?.result.dates.map((date, i) => {
    const point: Record<string, number | string> = { date };
    results.forEach(({ name, result }) => {
      point[name] = result.drawdowns[i] * 100;
    });
    return point;
  }) || [];

  return (
    <div className="chart-container">
      <h3>Drawdown History (%)</h3>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={data}>
          <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
          <XAxis 
            dataKey="date" 
            stroke="#94a3b8"
            tickFormatter={(date) => new Date(date as string).getFullYear().toString()}
            minTickGap={50}
          />
          <YAxis stroke="#94a3b8" domain={[-50, 0]} tickFormatter={(v) => `${v}%`} />
          <Tooltip 
            contentStyle={{ background: '#1e293b', border: '1px solid #334155' }}
            formatter={(value: number) => [`${(value as number).toFixed(1)}%`, 'Drawdown']}
          />
          <Legend />
          {results.map(({ name, color }) => (
            <Area
              key={name}
              type="monotone"
              dataKey={name}
              stroke={color}
              fill={color}
              fillOpacity={0.3}
              name={name}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};
