import React from 'react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

interface RegimeEntry {
  d: string;
  r: string;
  v: number | null;
}

interface RegimeTimelineProps {
  history: RegimeEntry[];
}

const REGIME_COLORS: Record<string, string> = {
  crisis: '#ef4444',
  vol_spike: '#f59e0b',
  low_vol: '#10b981',
  normal: '#3b82f6',
  unknown: '#6b7280'
};

const REGIME_LABELS: Record<string, string> = {
  crisis: 'Crisis',
  vol_spike: 'Vol Spike',
  low_vol: 'Low Vol',
  normal: 'Normal',
  unknown: 'Unknown'
};

export function RegimeTimeline({ history }: RegimeTimelineProps) {
  if (!history || history.length === 0) {
    return (
      <div className="regime-timeline empty">
        <p>No regime history available</p>
      </div>
    );
  }

  // Group consecutive same regimes and prepare for display
  const processed = history.map((entry, i) => {
    const prev = i > 0 ? history[i - 1] : null;
    const isTransition = prev && prev.r !== entry.r;
    return {
      ...entry,
      isTransition,
      date: new Date(entry.d).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
    };
  });

  const currentRegime = history[history.length - 1]?.r || 'unknown';

  return (
    <div className="regime-timeline">
      <div className="timeline-header">
        <h4>Regime History (30 days)</h4>
        <div className="current-regime">
          <span>Current:</span>
          <span 
            className="regime-badge"
            style={{ backgroundColor: REGIME_COLORS[currentRegime] }}
          >
            {REGIME_LABELS[currentRegime] || currentRegime.toUpperCase()}
          </span>
        </div>
      </div>

      <div className="timeline-legend">
        {Object.entries(REGIME_COLORS).map(([regime, color]) => (
          <div key={regime} className="legend-item">
            <span className="dot" style={{ backgroundColor: color }}></span>
            <span>{REGIME_LABELS[regime] || regime}</span>
          </div>
        ))}
      </div>

      <div className="timeline-chart">
        <ResponsiveContainer width="100%" height={150}>
          <AreaChart data={processed}>
            <defs>
              {Object.entries(REGIME_COLORS).map(([regime, color]) => (
                <linearGradient key={regime} id={`gradient-${regime}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={color} stopOpacity={0.8}/>
                  <stop offset="95%" stopColor={color} stopOpacity={0.3}/>
                </linearGradient>
              ))}
            </defs>
            <XAxis 
              dataKey="date" 
              tick={{ fontSize: 10 }}
              interval="preserveStartEnd"
              tickCount={6}
            />
            <YAxis hide />
            <Tooltip 
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className="timeline-tooltip">
                      <div className="tooltip-date">{data.d}</div>
                      <div className="tooltip-regime">
                        <span 
                          className="dot"
                          style={{ backgroundColor: REGIME_COLORS[data.r] }}
                        ></span>
                        {REGIME_LABELS[data.r] || data.r}
                      </div>
                      {data.v && <div className="tooltip-vix">VIX: {data.v.toFixed(1)}</div>}
                      {data.isTransition && (
                        <div className="tooltip-transition">← Regime Change</div>
                      )}
                    </div>
                  );
                }
                return null;
              }}
            />
            <Area
              type="stepAfter"
              dataKey={(d: RegimeEntry & { date: string; isTransition: boolean }) => {
                // Map regime to numeric value for stacking
                const values: Record<string, number> = {
                  crisis: 4, vol_spike: 3, low_vol: 2, normal: 1, unknown: 0
                };
                return values[d.r] || 0;
              }}
              stroke="#3b82f6"
              fill="url(#gradient-normal)"
              strokeWidth={2}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Transition markers */}
      <div className="transitions-list">
        <h5>Recent Transitions</h5>
        {processed
          .filter(d => d.isTransition)
          .slice(-5)
          .reverse()
          .map((entry, i) => (
            <div key={i} className="transition-item">
              <span className="date">{entry.d}</span>
              <span className="arrow">→</span>
              <span 
                className="regime-tag"
                style={{ backgroundColor: REGIME_COLORS[entry.r] }}
              >
                {REGIME_LABELS[entry.r] || entry.r}
              </span>
            </div>
          ))
        }
        {processed.filter(d => d.isTransition).length === 0 && (
          <p className="no-transitions">No regime transitions in period</p>
        )}
      </div>
    </div>
  );
}
