import React from 'react';
import type { PerformanceMetrics } from '../backtest/engine';

interface MetricsProps {
  results: Array<{ name: string; metrics: PerformanceMetrics; color: string }>;
}

const formatPercent = (n: number) => `${(n * 100).toFixed(1)}%`;

interface MetricDef {
  key: keyof PerformanceMetrics;
  label: string;
  format: (n: number) => string;
}

export const MetricsCards: React.FC<MetricsProps> = ({ results }) => {
  const metrics: MetricDef[] = [
    { key: 'cagr', label: 'CAGR', format: formatPercent },
    { key: 'volatility', label: 'Volatility', format: formatPercent },
    { key: 'sharpeRatio', label: 'Sharpe Ratio', format: (n: number) => n.toFixed(2) },
    { key: 'maxDrawdown', label: 'Max Drawdown', format: formatPercent },
    { key: 'calmarRatio', label: 'Calmar Ratio', format: (n: number) => n.toFixed(2) },
  ];

  return (
    <div className="metrics-grid">
      {metrics.map(({ key, label, format }) => (
        <div key={key} className="metric-card" style={{ borderColor: '#3b82f6' }}>
          <h4>{label}</h4>
          {results.map(({ name, metrics, color }) => (
            <div key={name} className="metric-row">
              <span style={{ color }}>{name}</span>
              <span className={key === 'maxDrawdown' && metrics[key] < 0 ? 'negative' : ''}>
                {format(metrics[key] as number)}
              </span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
};
