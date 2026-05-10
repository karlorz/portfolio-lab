import React from 'react';
import type { PerformanceMetrics } from '../backtest/engine';

interface TableProps {
  results: Array<{ name: string; metrics: PerformanceMetrics; color: string }>;
}

const formatPercent = (n: number) => `${(n * 100).toFixed(1)}%`;

export const ComparisonTable: React.FC<TableProps> = ({ results }) => {
  return (
    <div className="chart-container comparison-table">
      <h3>Detailed Comparison</h3>
      <table>
        <thead>
          <tr>
            <th>Portfolio</th>
            <th>CAGR</th>
            <th>Volatility</th>
            <th>Sharpe</th>
            <th>Max Drawdown</th>
            <th>Calmar</th>
            <th>Sortino</th>
          </tr>
        </thead>
        <tbody>
          {results.map(({ name, metrics, color }) => (
            <tr key={name}>
              <td style={{ color, fontWeight: 600 }}>{name}</td>
              <td className={metrics.cagr > 0 ? 'positive' : 'negative'}>
                {formatPercent(metrics.cagr)}
              </td>
              <td>{formatPercent(metrics.volatility)}</td>
              <td>{metrics.sharpeRatio.toFixed(2)}</td>
              <td className="negative">{formatPercent(metrics.maxDrawdown)}</td>
              <td>{metrics.calmarRatio.toFixed(2)}</td>
              <td>{metrics.sortinoRatio.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
