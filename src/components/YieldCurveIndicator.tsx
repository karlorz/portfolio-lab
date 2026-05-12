import React, { useMemo } from 'react';
import type { DurationRegime } from '../utils/duration-signals';

interface YieldCurveIndicatorProps {
  spread2s10s: number | null;
  regime: DurationRegime | null;
  spreadHistory?: number[];
  lastUpdate?: string;
}

const REGIME_CONFIG: Record<DurationRegime, { 
  label: string; 
  color: string; 
  bgColor: string;
  description: string;
}> = {
  steep: {
    label: 'STEEP',
    color: '#10b981',
    bgColor: '#d1fae5',
    description: '2s10s > 100bps - Long duration preferred (TLT heavy)'
  },
  normal: {
    label: 'NORMAL',
    color: '#f59e0b',
    bgColor: '#fef3c7',
    description: '2s10s 50-100bps - Moderate duration (Balanced)'
  },
  flat: {
    label: 'FLAT',
    color: '#f97316',
    bgColor: '#ffedd5',
    description: '2s10s 0-50bps - Short duration preferred (SHY increase)'
  },
  inverted: {
    label: 'INVERTED',
    color: '#ef4444',
    bgColor: '#fee2e2',
    description: '2s10s < 0bps - Ultra-short/cash (BIL protection)'
  }
};

export function YieldCurveIndicator({ 
  spread2s10s, 
  regime, 
  spreadHistory = [],
  lastUpdate 
}: YieldCurveIndicatorProps) {
  const config = regime ? REGIME_CONFIG[regime] : null;
  
  // Calculate sparkline points
  const sparklinePoints = useMemo(() => {
    if (spreadHistory.length < 2) return '';
    const recent = spreadHistory.slice(-30); // Last 30 data points
    const min = Math.min(...recent);
    const max = Math.max(...recent);
    const range = max - min || 1;
    
    return recent.map((val, i) => {
      const x = (i / (recent.length - 1)) * 100;
      const y = 100 - ((val - min) / range) * 100;
      return `${x},${y}`;
    }).join(' ');
  }, [spreadHistory]);

  // Calculate momentum (change over last 5 periods)
  const momentum = useMemo(() => {
    if (spreadHistory.length < 6) return null;
    const current = spreadHistory[spreadHistory.length - 1];
    const previous = spreadHistory[spreadHistory.length - 6];
    return current - previous;
  }, [spreadHistory]);

  if (!spread2s10s || !regime) {
    return (
      <div className="yield-curve-indicator loading">
        <h3>Yield Curve Regime</h3>
        <div className="indicator-content">
          <span className="loading-text">Loading yield data...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="yield-curve-indicator">
      <h3>Yield Curve Regime</h3>
      <div className="indicator-content">
        {/* Regime Badge */}
        <div 
          className="regime-badge-large"
          style={{ 
            backgroundColor: config?.bgColor,
            color: config?.color,
            border: `2px solid ${config?.color}`
          }}
        >
          {config?.label}
        </div>

        {/* Spread Value */}
        <div className="spread-display">
          <span className="spread-value" style={{ color: config?.color }}>
            {spread2s10s > 0 ? '+' : ''}{spread2s10s.toFixed(1)} bps
          </span>
          <span className="spread-label">2s10s Spread</span>
        </div>

        {/* Momentum Indicator */}
        {momentum !== null && (
          <div className="momentum-indicator">
            <span className={`momentum-value ${momentum > 0 ? 'positive' : 'negative'}`}>
              {momentum > 0 ? '↗' : '↘'} {Math.abs(momentum).toFixed(1)} bps (5d)
            </span>
          </div>
        )}

        {/* Sparkline */}
        {sparklinePoints && (
          <div className="sparkline-container">
            <svg viewBox="0 0 100 100" className="sparkline" preserveAspectRatio="none">
              <polyline
                points={sparklinePoints}
                fill="none"
                stroke={config?.color}
                strokeWidth="2"
              />
              {/* Zero line reference */}
              {(() => {
                const recent = spreadHistory.slice(-30);
                const min = Math.min(...recent);
                const max = Math.max(...recent);
                const range = max - min || 1;
                if (min < 0 && max > 0) {
                  const zeroY = 100 - ((0 - min) / range) * 100;
                  return (
                    <line
                      x1="0"
                      y1={zeroY}
                      x2="100"
                      y2={zeroY}
                      stroke="#666"
                      strokeWidth="0.5"
                      strokeDasharray="2,2"
                    />
                  );
                }
                return null;
              })()}
            </svg>
            <span className="sparkline-label">30-day trend</span>
          </div>
        )}

        {/* Description */}
        <p className="regime-description">{config?.description}</p>

        {/* Last Update */}
        {lastUpdate && (
          <span className="last-update">Updated: {lastUpdate}</span>
        )}
      </div>

      <style>{`
        .yield-curve-indicator {
          background: #1e293b;
          border-radius: 8px;
          padding: 16px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        }

        .yield-curve-indicator.loading {
          opacity: 0.7;
        }

        .yield-curve-indicator h3 {
          margin: 0 0 12px 0;
          font-size: 14px;
          font-weight: 600;
          color: #94a3b8;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .indicator-content {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .regime-badge-large {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          padding: 8px 16px;
          border-radius: 20px;
          font-size: 14px;
          font-weight: 700;
          width: fit-content;
        }

        .spread-display {
          display: flex;
          flex-direction: column;
          gap: 2px;
        }

        .spread-value {
          font-size: 24px;
          font-weight: 700;
        }

        .spread-label {
          font-size: 12px;
          color: #6b7280;
        }

        .momentum-indicator {
          font-size: 13px;
        }

        .momentum-value.positive {
          color: #10b981;
        }

        .momentum-value.negative {
          color: #ef4444;
        }

        .sparkline-container {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .sparkline {
          width: 100%;
          height: 40px;
          background: #0f172a;
          border-radius: 4px;
        }

        .sparkline-label {
          font-size: 11px;
          color: #6b7280;
          text-align: center;
        }

        .regime-description {
          margin: 0;
          font-size: 13px;
          color: #e2e8f0;
          line-height: 1.4;
        }

        .last-update {
          font-size: 11px;
          color: #6b7280;
        }

        .loading-text {
          color: #6b7280;
          font-size: 14px;
        }
      `}</style>
    </div>
  );
}
