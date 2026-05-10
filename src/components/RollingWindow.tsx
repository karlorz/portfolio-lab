import React, { useMemo } from 'react';
import { BacktestEngine } from '../backtest/engine';
import type { PortfolioConfig, PriceData, PerformanceMetrics } from '../backtest/engine';

interface RollingWindowProps {
  portfolios: PortfolioConfig[];
  priceData: PriceData[];
  colors: string[];
}

const WINDOWS = [
  { name: 'Full (2005-2026)', start: '2005-01-01', end: '2026-12-31' },
  { name: 'GFC (2007-2009)', start: '2007-10-01', end: '2009-06-30' },
  { name: 'Recovery (2009-2013)', start: '2009-03-01', end: '2013-12-31' },
  { name: 'Bull Market (2013-2019)', start: '2013-01-01', end: '2019-12-31' },
  { name: '2020 COVID', start: '2020-01-01', end: '2020-12-31' },
  { name: 'Rate Hikes (2022-23)', start: '2022-01-01', end: '2023-12-31' },
  { name: '2025-2026 YTD', start: '2025-01-01', end: '2026-12-31' },
];

export const RollingWindow: React.FC<RollingWindowProps> = ({ portfolios, priceData, colors }) => {
  const gridData = useMemo(() => {
    const engine = new BacktestEngine();
    engine.loadData(priceData);

    return WINDOWS.map(window => {
      const row: { window: string; [portfolioName: string]: number | string } = { window: window.name };
      for (const portfolio of portfolios) {
        const result = engine.runBacktest(portfolio, window.start, window.end, 10000);
        const metrics = engine.calculateMetrics(result);
        row[portfolio.name + '_sharpe'] = metrics.sharpeRatio;
        row[portfolio.name + '_cagr'] = metrics.cagr * 100;
      }
      return row;
    });
  }, [portfolios, priceData]);

  // Find best Sharpe per window for highlighting
  const bestPerWindow = useMemo(() => {
    return WINDOWS.map((_, wi) => {
      let best = -Infinity;
      let bestName = '';
      for (const p of portfolios) {
        const val = gridData[wi][p.name + '_sharpe'] as number;
        if (val > best) { best = val; bestName = p.name; }
      }
      return bestName;
    });
  }, [gridData, portfolios]);

  return (
    <div className="chart-container">
      <h3>Rolling-Window Sharpe Ratios</h3>
      <p style={{ color: '#64748b', fontSize: '0.85rem', marginBottom: 15 }}>
        Sharpe ratio across economic regimes. Bold = best in that window.
      </p>
      <div className="comparison-table">
        <table>
          <thead>
            <tr>
              <th>Period</th>
              {portfolios.map((p, i) => (
                <th key={p.name} style={{ color: colors[i % colors.length] }}>{p.name}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {gridData.map((row, wi) => (
              <tr key={WINDOWS[wi].name}>
                <td style={{ color: '#e2e8f0', fontWeight: 600 }}>{row.window}</td>
                {portfolios.map((p, pi) => {
                  const sharpe = row[p.name + '_sharpe'] as number;
                  const cagr = row[p.name + '_cagr'] as number;
                  const isBest = bestPerWindow[wi] === p.name;
                  return (
                    <td
                      key={p.name}
                      style={{
                        color: sharpe > 0 ? '#10b981' : '#ef4444',
                        fontWeight: isBest ? 800 : 400,
                        background: isBest ? 'rgba(59,130,246,0.1)' : 'transparent',
                      }}
                    >
                      {sharpe.toFixed(2)} <small style={{ color: '#64748b' }}>({cagr > 0 ? '+' : ''}{cagr.toFixed(1)}%)</small>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
