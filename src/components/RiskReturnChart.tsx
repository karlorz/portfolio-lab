import React from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ZAxis, Cell } from 'recharts';
import type { PerformanceMetrics } from '../backtest/engine';

interface RiskReturnProps {
  results: Array<{ name: string; metrics: PerformanceMetrics; color: string }>;
}

interface ScatterPoint {
  name: string;
  x: number;
  y: number;
  z: number;
  color: string;
}

export const RiskReturnChart: React.FC<RiskReturnProps> = ({ results }) => {
  const data: ScatterPoint[] = results.map(({ name, metrics, color }) => ({
    name,
    x: metrics.volatility * 100, // Risk (volatility %)
    y: metrics.cagr * 100,       // Return (CAGR %)
    z: Math.abs(metrics.maxDrawdown) * 100, // Bubble size = drawdown
    color,
  }));

  return (
    <div className="chart-container">
      <h3>Risk-Return Scatter (Bubble = Max Drawdown)</h3>
      <ResponsiveContainer width="100%" height={400}>
        <ScatterChart>
          <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
          <XAxis 
            type="number" 
            dataKey="x" 
            name="Volatility" 
            unit="%" 
            stroke="#94a3b8"
            label={{ value: 'Risk (Volatility %)', position: 'bottom', fill: '#94a3b8' }}
          />
          <YAxis 
            type="number" 
            dataKey="y" 
            name="CAGR" 
            unit="%" 
            stroke="#94a3b8"
            label={{ value: 'Return (CAGR %)', angle: -90, position: 'left', fill: '#94a3b8' }}
          />
          <ZAxis type="number" dataKey="z" range={[50, 400]} />
          <Tooltip 
            cursor={{ strokeDasharray: '3 3' }}
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                const p = payload[0].payload as ScatterPoint;
                return (
                  <div className="tooltip">
                    <p style={{ fontWeight: 'bold', color: p.color }}>{p.name}</p>
                    <p>Return: {p.y.toFixed(1)}%</p>
                    <p>Risk: {p.x.toFixed(1)}%</p>
                    <p>Max DD: -{p.z.toFixed(1)}%</p>
                  </div>
                );
              }
              return null;
            }}
          />
          <Scatter data={data}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
};
