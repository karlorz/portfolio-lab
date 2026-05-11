import React, { useMemo } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ComposedChart,
  Line,
} from 'recharts';
import { AlertTriangle, TrendingDown, TrendingUp } from 'lucide-react';

interface DrawdownPoint {
  date: string;
  value: number;
  peak: number;
  drawdown: number;
  days_since_peak: number;
  is_recovery: boolean;
}

interface MaxDrawdown {
  max_drawdown: number;
  max_drawdown_date: string;
  recovery_date: string | null;
  underwater_days: number;
  peak_value: number;
  trough_value: number;
}

interface UnderwaterChartProps {
  series: DrawdownPoint[];
  maxDrawdown: MaxDrawdown;
  height?: number;
}

const formatCurrency = (val: number) => {
  if (val >= 1000) return `$${(val / 1000).toFixed(1)}K`;
  return `$${val.toFixed(0)}`;
};

const formatDate = (dateStr: string) => {
  const d = new Date(dateStr);
  return `${d.getMonth() + 1}/${d.getDate()}`;
};

export const UnderwaterChart: React.FC<UnderwaterChartProps> = ({
  series,
  maxDrawdown,
  height = 320,
}) => {
  const chartData = useMemo(() => {
    return series.map((p) => ({
      ...p,
      dateFormatted: formatDate(p.date),
    }));
  }, [series]);

  const maxDD = maxDrawdown?.max_drawdown || 0;
  const isUnderwater = !maxDrawdown?.recovery_date;

  // Determine severity color based on drawdown depth
  const getSeverityColor = (dd: number) => {
    if (dd <= -20) return '#ef4444'; // Red - severe
    if (dd <= -10) return '#f97316'; // Orange - moderate
    if (dd <= -5) return '#eab308'; // Yellow - mild
    return '#22c55e'; // Green - minimal
  };

  const severityColor = getSeverityColor(maxDD);

  return (
    <div className="underwater-chart">
      <div className="underwater-header">
        <h3 className="underwater-title">
          <TrendingDown size={20} />
          Drawdown (Underwater Curve)
        </h3>
        <div className="underwater-metrics">
          <div className={`metric-card ${maxDD <= -15 ? 'danger' : maxDD <= -10 ? 'warning' : 'good'}`}>
            <span className="metric-label">Max DD</span>
            <span className="metric-value">{maxDD.toFixed(2)}%</span>
          </div>
          <div className="metric-card">
            <span className="metric-label">Underwater Days</span>
            <span className="metric-value">
              {maxDrawdown?.underwater_days || 0}
            </span>
          </div>
          {isUnderwater && (
            <div className="metric-card warning">
              <AlertTriangle size={16} />
              <span>Still Recovering</span>
            </div>
          )}
          {maxDrawdown?.recovery_date && (
            <div className="metric-card good">
              <TrendingUp size={16} />
              <span>Recovered</span>
            </div>
          )}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="colorDrawdown" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={severityColor} stopOpacity={0.4} />
              <stop offset="95%" stopColor={severityColor} stopOpacity={0.1} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            dataKey="dateFormatted"
            tick={{ fontSize: 12, fill: '#9ca3af' }}
            tickLine={false}
            axisLine={{ stroke: '#4b5563' }}
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fontSize: 12, fill: '#9ca3af' }}
            tickLine={false}
            axisLine={{ stroke: '#4b5563' }}
            tickFormatter={(val) => `${val}%`}
            domain={['auto', 0]}
          />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload || !payload[0]) return null;
              const p = payload[0].payload as DrawdownPoint & { dateFormatted: string };
              return (
                <div className="chart-tooltip">
                  <div className="tooltip-date">{p.date}</div>
                  <div className="tooltip-row">
                    <span className="tooltip-label">Value:</span>
                    <span className="tooltip-value">{formatCurrency(p.value)}</span>
                  </div>
                  <div className="tooltip-row">
                    <span className="tooltip-label">Drawdown:</span>
                    <span className={`tooltip-value ${p.drawdown <= -10 ? 'negative' : ''}`}>
                      {p.drawdown.toFixed(2)}%
                    </span>
                  </div>
                  <div className="tooltip-row">
                    <span className="tooltip-label">Peak:</span>
                    <span className="tooltip-value">{formatCurrency(p.peak)}</span>
                  </div>
                  <div className="tooltip-row">
                    <span className="tooltip-label">Days since peak:</span>
                    <span className="tooltip-value">{p.days_since_peak}</span>
                  </div>
                  {p.is_recovery && (
                    <div className="tooltip-row recovery">
                      <TrendingUp size={14} />
                      <span>Recovered to new high</span>
                    </div>
                  )}
                </div>
              );
            }}
          />
          <ReferenceLine y={-10} stroke="#f97316" strokeDasharray="4 4" label={{ value: '10% DD', fill: '#f97316', fontSize: 10 }} />
          <ReferenceLine y={-15} stroke="#ef4444" strokeDasharray="4 4" label={{ value: '15% DD', fill: '#ef4444', fontSize: 10 }} />
          <ReferenceLine y={0} stroke="#22c55e" strokeWidth={2} />
          <Area
            type="monotone"
            dataKey="drawdown"
            stroke={severityColor}
            fill="url(#colorDrawdown)"
            strokeWidth={2}
            dot={false}
            name="Drawdown %"
          />
        </AreaChart>
      </ResponsiveContainer>

      <div className="underwater-legend">
        <div className="legend-item">
          <div className="legend-color" style={{ background: '#ef4444' }} />
          <span>-20% (Severe)</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ background: '#f97316' }} />
          <span>-15% (Moderate)</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ background: '#eab308' }} />
          <span>-10% (Mild)</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ background: '#22c55e' }} />
          <span>0% (New High)</span>
        </div>
      </div>
    </div>
  );
};

interface RollingMetricPoint {
  date: string;
  sharpe: number;
  volatility: number;
  mean_return: number;
  window_days: number;
}

interface RollingMetricsProps {
  sharpe63d: RollingMetricPoint[];
  sharpe126d: RollingMetricPoint[];
  sharpe252d: RollingMetricPoint[];
  height?: number;
}

export const RollingMetricsChart: React.FC<RollingMetricsProps> = ({
  sharpe63d,
  sharpe126d,
  sharpe252d,
  height = 320,
}) => {
  // Combine all windows into single dataset
  const chartData = useMemo(() => {
    // Get all unique dates
    const allDates = new Set<string>([
      ...sharpe63d.map((d) => d.date),
      ...sharpe126d.map((d) => d.date),
      ...sharpe252d.map((d) => d.date),
    ]);

    return Array.from(allDates)
      .sort()
      .map((date) => ({
        date,
        dateFormatted: formatDate(date),
        sharpe63: sharpe63d.find((d) => d.date === date)?.sharpe ?? null,
        sharpe126: sharpe126d.find((d) => d.date === date)?.sharpe ?? null,
        sharpe252: sharpe252d.find((d) => d.date === date)?.sharpe ?? null,
        vol63: sharpe63d.find((d) => d.date === date)?.volatility ?? null,
      }));
  }, [sharpe63d, sharpe126d, sharpe252d]);

  const latest63 = sharpe63d[sharpe63d.length - 1]?.sharpe;
  const latest126 = sharpe126d[sharpe126d.length - 1]?.sharpe;
  const latest252 = sharpe252d[sharpe252d.length - 1]?.sharpe;

  return (
    <div className="rolling-metrics-chart">
      <div className="rolling-header">
        <h3 className="rolling-title">Rolling Sharpe Ratios</h3>
        <div className="rolling-legend">
          {latest63 !== undefined && (
            <div className="legend-item">
              <div className="legend-color" style={{ background: '#3b82f6' }} />
              <span>63d: {latest63?.toFixed(2) || '-'}</span>
            </div>
          )}
          {latest126 !== undefined && (
            <div className="legend-item">
              <div className="legend-color" style={{ background: '#8b5cf6' }} />
              <span>126d: {latest126?.toFixed(2) || '-'}</span>
            </div>
          )}
          {latest252 !== undefined && (
            <div className="legend-item">
              <div className="legend-color" style={{ background: '#ec4899' }} />
              <span>252d: {latest252?.toFixed(2) || '-'}</span>
            </div>
          )}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            dataKey="dateFormatted"
            tick={{ fontSize: 12, fill: '#9ca3af' }}
            tickLine={false}
            axisLine={{ stroke: '#4b5563' }}
            interval="preserveStartEnd"
          />
          <YAxis
            yAxisId="sharpe"
            tick={{ fontSize: 12, fill: '#9ca3af' }}
            tickLine={false}
            axisLine={{ stroke: '#4b5563' }}
            domain={[-1, 2]}
          />
          <YAxis
            yAxisId="vol"
            orientation="right"
            tick={{ fontSize: 12, fill: '#9ca3af' }}
            tickLine={false}
            axisLine={{ stroke: '#4b5563' }}
            tickFormatter={(val) => `${val}%`}
            hide
          />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload) return null;
              const p = payload[0]?.payload;
              if (!p) return null;
              return (
                <div className="chart-tooltip">
                  <div className="tooltip-date">{p.date}</div>
                  {p.sharpe63 !== null && (
                    <div className="tooltip-row">
                      <span className="tooltip-label">Sharpe (63d):</span>
                      <span className={`tooltip-value ${p.sharpe63 > 0 ? 'positive' : 'negative'}`}>
                        {p.sharpe63.toFixed(2)}
                      </span>
                    </div>
                  )}
                  {p.sharpe126 !== null && (
                    <div className="tooltip-row">
                      <span className="tooltip-label">Sharpe (126d):</span>
                      <span className={`tooltip-value ${p.sharpe126 > 0 ? 'positive' : 'negative'}`}>
                        {p.sharpe126.toFixed(2)}
                      </span>
                    </div>
                  )}
                  {p.sharpe252 !== null && (
                    <div className="tooltip-row">
                      <span className="tooltip-label">Sharpe (252d):</span>
                      <span className={`tooltip-value ${p.sharpe252 > 0 ? 'positive' : 'negative'}`}>
                        {p.sharpe252.toFixed(2)}
                      </span>
                    </div>
                  )}
                  {p.vol63 !== null && (
                    <div className="tooltip-row">
                      <span className="tooltip-label">Vol (63d):</span>
                      <span className="tooltip-value">{p.vol63.toFixed(1)}%</span>
                    </div>
                  )}
                </div>
              );
            }}
          />
          <ReferenceLine yAxisId="sharpe" y={0} stroke="#6b7280" strokeDasharray="3 3" />
          <ReferenceLine yAxisId="sharpe" y={0.5} stroke="#22c55e" strokeDasharray="4 4" label={{ value: 'Target 0.5', fill: '#22c55e', fontSize: 10 }} />
          <ReferenceLine yAxisId="sharpe" y={1.0} stroke="#3b82f6" strokeDasharray="4 4" label={{ value: 'Good 1.0', fill: '#3b82f6', fontSize: 10 }} />
          <Line
            yAxisId="sharpe"
            type="monotone"
            dataKey="sharpe63"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={false}
            name="Sharpe (63d)"
            connectNulls
          />
          <Line
            yAxisId="sharpe"
            type="monotone"
            dataKey="sharpe126"
            stroke="#8b5cf6"
            strokeWidth={2}
            dot={false}
            name="Sharpe (126d)"
            connectNulls
          />
          <Line
            yAxisId="sharpe"
            type="monotone"
            dataKey="sharpe252"
            stroke="#ec4899"
            strokeWidth={2}
            dot={false}
            name="Sharpe (252d)"
            connectNulls
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};

interface CrisisPeriod {
  name: string;
  period: string;
  description: string;
  spy_return: number;
  portfolio_return: number | null;
}

interface CrisisOverlayProps {
  periods: CrisisPeriod[];
}

export const CrisisOverlay: React.FC<CrisisOverlayProps> = ({ periods }) => {
  return (
    <div className="crisis-overlay">
      <h3 className="crisis-title">Crisis Period Comparison</h3>
      <div className="crisis-grid">
        {periods.map((period) => (
          <div key={period.name} className="crisis-card">
            <div className="crisis-header">
              <span className="crisis-name">{period.name}</span>
              <span className="crisis-period">{period.period}</span>
            </div>
            <div className="crisis-description">{period.description}</div>
            <div className="crisis-returns">
              <div className="return-row">
                <span className="return-label">SPY</span>
                <span className={`return-value ${period.spy_return < 0 ? 'negative' : 'positive'}`}>
                  {period.spy_return > 0 ? '+' : ''}{period.spy_return.toFixed(1)}%
                </span>
              </div>
              <div className="return-row">
                <span className="return-label">Portfolio</span>
                {period.portfolio_return !== null ? (
                  <span className={`return-value ${period.portfolio_return < 0 ? 'negative' : 'positive'}`}>
                    {period.portfolio_return > 0 ? '+' : ''}{period.portfolio_return.toFixed(1)}%
                  </span>
                ) : (
                  <span className="return-value na">No data</span>
                )}
              </div>
            </div>
            {period.portfolio_return !== null && (
              <div className="crisis-outperformance">
                {period.portfolio_return > period.spy_return ? (
                  <span className="outperformance positive">
                    <TrendingUp size={14} />
                    Outperformed SPY by {(period.portfolio_return - period.spy_return).toFixed(1)}%
                  </span>
                ) : (
                  <span className="outperformance negative">
                    <TrendingDown size={14} />
                    Underperformed SPY by {(period.spy_return - period.portfolio_return).toFixed(1)}%
                  </span>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
