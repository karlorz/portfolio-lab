import React, { useState, useMemo, useEffect, lazy, Suspense } from 'react';
import { BacktestEngine, PORTFOLIOS } from './backtest/engine';
import type { BacktestResult, PerformanceMetrics, PriceData } from './backtest/engine';
import { PortfolioSelector } from './components/PortfolioSelector';
import { MetricsCards } from './components/MetricsCards';
import { EquityCurve } from './components/EquityCurve';
import { LiveDashboard } from './components/LiveDashboard';
import './App.css';

const RiskReturnChart = lazy(() => import('./components/RiskReturnChart').then(m => ({ default: m.RiskReturnChart })));
const DrawdownChart = lazy(() => import('./components/DrawdownChart').then(m => ({ default: m.DrawdownChart })));
const ComparisonTable = lazy(() => import('./components/ComparisonTable').then(m => ({ default: m.ComparisonTable })));
const CrisisAnalysis = lazy(() => import('./components/CrisisAnalysis').then(m => ({ default: m.CrisisAnalysis })));
const RollingWindow = lazy(() => import('./components/RollingWindow').then(m => ({ default: m.RollingWindow })));
const CorrelationMatrix = lazy(() => import('./components/CorrelationMatrix').then(m => ({ default: m.CorrelationMatrix })));
const FIRECalculator = lazy(() => import('./components/FIRECalculator').then(m => ({ default: m.FIRECalculator })));

interface ResultItem {
  name: string;
  result: BacktestResult;
  metrics: PerformanceMetrics;
  color: string;
}

const COLORS: string[] = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#06b6d4', '#ec4899', '#84cc16', '#f97316', '#a855f7'];

// Convert compact price data to backtest format
function toBacktestData(prices: Record<string, Array<{ d: string; p: number }>>): PriceData[] {
  const result: PriceData[] = [];
  for (const [symbol, entries] of Object.entries(prices)) {
    for (const { d, p } of entries) {
      result.push({ date: d, symbol, price: p });
    }
  }
  return result.sort((a, b) => a.date.localeCompare(b.date));
}

function App() {
  const [selected, setSelected] = useState<string[]>(['SPY (S&P 500)', 'SPY/GLD/TLT 46/38/16 ★★', 'SPY/GLD 55/45']);
  const [priceData, setPriceData] = useState<PriceData[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load real price data
  useEffect(() => {
    fetch('/data/prices.json')
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then(json => {
        setPriceData(toBacktestData(json));
        setLoading(false);
      })
      .catch(err => {
        setError(`Failed to load data: ${err.message}`);
        setLoading(false);
      });
  }, []);

  // Run backtests
  const results = useMemo<ResultItem[]>(() => {
    if (!priceData) return [];

    const engine = new BacktestEngine();
    engine.loadData(priceData);

    // Find date range from data
    const dates = priceData.map(p => p.date).sort();
    const startDate = dates[0];
    const endDate = dates[dates.length - 1];

    return PORTFOLIOS.map((portfolio, i) => {
      const result = engine.runBacktest(portfolio, startDate, endDate, 10000);
      const metrics = engine.calculateMetrics(result);
      return {
        name: portfolio.name,
        result,
        metrics,
        color: COLORS[i % COLORS.length]!,
      };
    });
  }, [priceData]);

  const filteredResults = results.filter(r => selected.includes(r.name));

  const handleToggle = (name: string) => {
    setSelected(prev =>
      prev.includes(name)
        ? prev.filter(n => n !== name)
        : [...prev, name]
    );
  };

  // Find best portfolio by Sharpe ratio
  const bestPortfolio = useMemo(() => {
    if (results.length === 0) return null;
    return results.reduce((best, current) =>
      current.metrics.sharpeRatio > best.metrics.sharpeRatio ? current : best
    );
  }, [results]);

  // Find portfolio with SPY-like returns but lower volatility
  const spyResult = results.find(r => r.name === 'SPY (S&P 500)');
  const efficientAlternative = useMemo(() => {
    if (!spyResult) return null;
    const spyVol = spyResult.metrics.volatility;
    return results.filter(r => r.name !== 'SPY (S&P 500)')
      .find(r => r.metrics.cagr >= spyResult.metrics.cagr * 0.9 && r.metrics.volatility <= spyVol * 0.7);
  }, [results, spyResult]);

  if (loading) {
    return (
      <div className="app">
        <header>
          <h1>Portfolio Lab</h1>
          <p>Loading historical data...</p>
        </header>
      </div>
    );
  }

  if (error) {
    return (
      <div className="app">
        <header>
          <h1>Portfolio Lab</h1>
          <p style={{ color: '#ef4444' }}>{error}</p>
        </header>
      </div>
    );
  }

  return (
    <div className="app">
      <header>
        <h1>Portfolio Lab</h1>
        <p>All-Season Portfolio Strategy Backtest (2005-2026)</p>
      </header>

      <PortfolioSelector
        portfolios={PORTFOLIOS}
        selected={selected}
        onToggle={handleToggle}
        colors={COLORS}
      />

      <LiveDashboard refreshInterval={60} />

      {filteredResults.length > 0 && (
        <>
          <MetricsCards results={filteredResults} />

          <div className="charts">
            <EquityCurve results={filteredResults} />

            <Suspense fallback={<div className="chart-loading">Loading chart...</div>}>
              <div className="chart-row">
                <RiskReturnChart results={filteredResults} />
                <DrawdownChart results={filteredResults} />
              </div>

              <ComparisonTable results={filteredResults} />

              <CrisisAnalysis results={filteredResults} />

              <RollingWindow
                portfolios={PORTFOLIOS}
                priceData={priceData}
                colors={COLORS}
              />

              <CorrelationMatrix priceData={priceData} />

              <FIRECalculator results={filteredResults} />
            </Suspense>
          </div>
        </>
      )}

      <footer>
        {bestPortfolio && (
          <p>
            <strong>Highest Sharpe Ratio:</strong>{' '}
            <span style={{ color: bestPortfolio.color }}>{bestPortfolio.name}</span>
            {' '}({bestPortfolio.metrics.sharpeRatio.toFixed(2)})
          </p>
        )}
        {efficientAlternative && spyResult && (
          <p>
            <strong>≥90% SPY Return with ≤70% Volatility:</strong>{' '}
            <span style={{ color: efficientAlternative.color }}>{efficientAlternative.name}</span>
            {' '}(CAGR: {(efficientAlternative.metrics.cagr * 100).toFixed(1)}%,
            {' '}Vol: {(efficientAlternative.metrics.volatility * 100).toFixed(1)}%)
          </p>
        )}
        <p style={{ marginTop: 15, fontSize: '0.75rem' }}>
          Data: Yahoo Finance historical daily prices 2005-2026.
          <br />
          <code>bun run fetch-data</code> to refresh data from Yahoo Finance
        </p>
      </footer>
    </div>
  );
}

export default App;
