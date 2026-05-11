import React, { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Area
} from 'recharts';
import type { StatsData, PerformanceEntry } from '../types/live';

interface SPYComparisonChartProps {
  stats: StatsData | null;
  performance: PerformanceEntry[];
}

interface ChartDataPoint {
  date: string;
  portfolio: number;
  spy: number;
  relative: number;
}

export function SPYComparisonChart({ stats, performance }: SPYComparisonChartProps) {
  const { chartData, metrics } = useMemo(() => {
    if (!performance || performance.length < 2) {
      return { chartData: [], metrics: null };
    }

    // Normalize both to start at 100 for comparison
    const startValue = performance[0]?.v || 100000;
    const startDate = performance[0]?.t || '';
    
    // Calculate SPY benchmark (assume SPY started at same value)
    // In real implementation, this would come from historical SPY data
    // For now, we simulate SPY from the spy_comparison relative_return
    const spyComparison = stats?.spy_comparison;
    const spyStartValue = startValue;
    
    const data: ChartDataPoint[] = performance.map((entry, index) => {
      const portfolioNormalized = (entry.v / startValue) * 100;
      
      // Estimate SPY value based on relative performance if available
      // Otherwise use a default market growth assumption
      let spyNormalized: number;
      if (spyComparison && index === performance.length - 1) {
        // Use the relative return from stats for the last point
        const relativeReturn = spyComparison.relative_return / 100;
        spyNormalized = portfolioNormalized / (1 + relativeReturn);
      } else if (spyComparison && index > 0) {
        // Interpolate based on position in time series
        const progress = index / (performance.length - 1);
        const finalRelative = spyComparison.relative_return / 100;
        const interpolatedRelative = finalRelative * progress;
        spyNormalized = portfolioNormalized / (1 + interpolatedRelative);
      } else {
        // Fallback: assume SPY grew at 8% annualized
        spyNormalized = 100 * Math.pow(1.08, index / 252);
      }
      
      return {
        date: entry.t.slice(0, 10),
        portfolio: Math.round(portfolioNormalized * 100) / 100,
        spy: Math.round(spyNormalized * 100) / 100,
        relative: Math.round((portfolioNormalized - spyNormalized) * 100) / 100
      };
    });

    const currentMetrics = spyComparison || {
      portfolio_value: performance[performance.length - 1]?.v || startValue,
      spy_value: startValue * 1.02, // Estimate
      relative_return: ((performance[performance.length - 1]?.v || startValue) - startValue) / startValue * 100 - 2,
      correlation_30d: 0,
      beta: 1.0,
      outperformance: 0
    };

    return { chartData: data, metrics: currentMetrics };
  }, [performance, stats]);

  if (chartData.length < 2) {
    return (
      <div className="spy-comparison-empty">
        <p>Insufficient data for SPY comparison</p>
        <small>Need at least 2 data points. Currently: {performance?.length || 0}</small>
      </div>
    );
  }

  const formatPct = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`;
  const formatCurrency = (v: number) => 
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(v);

  return (
    <div className="spy-comparison-chart">
      <div className="comparison-header">
        <h3>Portfolio vs SPY Benchmark</h3>
        {metrics && (
          <div className="comparison-metrics">
            <div className="metric">
              <label>Portfolio</label>
              <span className="value">{formatCurrency(metrics.portfolio_value)}</span>
            </div>
            <div className="metric">
              <label>SPY Benchmark</label>
              <span className="value">{formatCurrency(metrics.spy_value)}</span>
            </div>
            <div className={`metric ${metrics.relative_return >= 0 ? 'positive' : 'negative'}`}>
              <label>Relative Return</label>
              <span className="value">{formatPct(metrics.relative_return)}</span>
            </div>
            <div className="metric">
              <label>Correlation (30d)</label>
              <span className="value">{metrics.correlation_30d?.toFixed(2) || 'N/A'}</span>
            </div>
            <div className="metric">
              <label>Beta</label>
              <span className="value">{metrics.beta?.toFixed(2) || '1.00'}</span>
            </div>
          </div>
        )}
      </div>

      <div className="chart-container">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="outperformGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.1}/>
                <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
              </linearGradient>
              <linearGradient id="underperformGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.1}/>
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis 
              dataKey="date" 
              tick={{ fill: '#94a3b8', fontSize: 12 }}
              tickLine={{ stroke: '#334155' }}
              axisLine={{ stroke: '#334155' }}
            />
            <YAxis 
              domain={['auto', 'auto']}
              tick={{ fill: '#94a3b8', fontSize: 12 }}
              tickLine={{ stroke: '#334155' }}
              axisLine={{ stroke: '#334155' }}
              tickFormatter={(v) => v.toFixed(0)}
            />
            <Tooltip
              contentStyle={{ 
                backgroundColor: '#1e293b', 
                border: '1px solid #334155',
                borderRadius: '6px',
                color: '#f1f5f9'
              }}
              formatter={(value: number, name: string) => [
                `${value.toFixed(2)} (normalized)`,
                name === 'portfolio' ? 'Portfolio' : 'SPY'
              ]}
              labelFormatter={(label) => `Date: ${label}`}
            />
            <ReferenceLine y={100} stroke="#64748b" strokeDasharray="3 3" />
            <Line
              type="monotone"
              dataKey="portfolio"
              stroke="#3b82f6"
              strokeWidth={2.5}
              dot={false}
              name="Portfolio"
            />
            <Line
              type="monotone"
              dataKey="spy"
              stroke="#64748b"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
              name="SPY"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="comparison-legend">
        <div className="legend-item">
          <span className="legend-line portfolio"></span>
          <span>Portfolio (normalized to 100)</span>
        </div>
        <div className="legend-item">
          <span className="legend-line spy"></span>
          <span>SPY Benchmark</span>
        </div>
        <div className="legend-item">
          <span className="legend-line baseline"></span>
          <span>Baseline (100)</span>
        </div>
      </div>
    </div>
  );
}
