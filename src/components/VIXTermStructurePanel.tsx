import React from 'react';

interface VIXLevel {
  value: number;
  timestamp: string;
}

interface VIXTermStructureData {
  vix?: VIXLevel;
  vix3m?: VIXLevel;
  vix6m?: VIXLevel;
  slope?: number;
  roll_yield?: number;
  composite_signal?: number;
  regime?: 'extreme_contango' | 'steep_contango' | 'mild_contango' | 'flat' | 'backwardation' | 'extreme_backwardation';
  z_score?: number;
  percentile_1y?: number;
}

interface VIXOverlayState {
  allocation: Record<string, number>;
  last_shift_date: string;
  shift_history: Array<{
    date: string;
    shifts: Record<string, number>;
    signal_value: number;
    regime: string;
    new_allocation: Record<string, number>;
  }>;
  disabled_until: string | null;
}

interface VIXTermStructurePanelProps {
  data?: VIXTermStructureData | null;
  overlayState?: VIXOverlayState | null;
}

export function VIXTermStructurePanel({ data, overlayState }: VIXTermStructurePanelProps) {
  if (!data) {
    return (
      <div className="vix-panel">
        <div className="panel-header">
          <h3>VIX Term Structure</h3>
          <span className="status-badge">No Data</span>
        </div>
        <div className="panel-content">
          <p className="empty-state">VIX term structure data not available</p>
        </div>
      </div>
    );
  }

  const getRegimeColor = (regime?: string): string => {
    switch (regime) {
      case 'extreme_contango': return '#10b981'; // green
      case 'steep_contango': return '#34d399'; // light green
      case 'mild_contango': return '#6ee7b7'; // mint
      case 'flat': return '#fbbf24'; // amber
      case 'backwardation': return '#f59e0b'; // orange
      case 'extreme_backwardation': return '#ef4444'; // red
      default: return '#6b7280'; // gray
    }
  };

  const getRegimeLabel = (regime?: string): string => {
    if (!regime) return 'Unknown';
    return regime.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  const getSignalColor = (signal: number): string => {
    if (signal > 0.5) return '#10b981';
    if (signal > 0) return '#34d399';
    if (signal > -0.3) return '#fbbf24';
    if (signal > -0.7) return '#f59e0b';
    return '#ef4444';
  };

  const getSignalLabel = (signal: number): string => {
    if (signal > 0.5) return 'Complacent (+)';
    if (signal > 0) return 'Calm';
    if (signal > -0.3) return 'Cautious';
    if (signal > -0.7) return 'Risk-Off (-)';
    return 'Crisis (--) ';
  };

  const formatDate = (isoString?: string): string => {
    if (!isoString) return 'N/A';
    return new Date(isoString).toLocaleString();
  };

  const slope = data.slope ?? 0;
  const rollYield = data.roll_yield ?? 0;
  const signal = data.composite_signal ?? 0;
  const zScore = data.z_score ?? 0;

  return (
    <div className="vix-panel">
      <div className="panel-header">
        <h3>VIX Term Structure</h3>
        <span 
          className="status-badge"
          style={{ backgroundColor: getRegimeColor(data.regime) }}
        >
          {getRegimeLabel(data.regime)}
        </span>
      </div>

      <div className="panel-content">
        {/* VIX Levels */}
        <div className="vix-levels">
          <div className="vix-level">
            <label>VIX Spot</label>
            <span className="value">{data.vix?.value?.toFixed(2) ?? 'N/A'}</span>
          </div>
          <div className="vix-level">
            <label>VIX3M</label>
            <span className="value">{data.vix3m?.value?.toFixed(2) ?? 'N/A'}</span>
          </div>
          <div className="vix-level">
            <label>VIX6M</label>
            <span className="value">{data.vix6m?.value?.toFixed(2) ?? 'N/A'}</span>
          </div>
        </div>

        {/* Term Structure Visual */}
        <div className="term-structure-viz">
          <h4>Term Structure</h4>
          <div className="structure-bars">
            {data.vix && data.vix3m && (
              <>
                <div 
                  className="structure-bar vix-spot"
                  style={{ 
                    height: `${Math.min((data.vix.value / 40) * 100, 100)}%`,
                    backgroundColor: data.vix.value > 25 ? '#ef4444' : data.vix.value > 20 ? '#f59e0b' : '#3b82f6'
                  }}
                >
                  <span className="bar-label">Spot</span>
                  <span className="bar-value">{data.vix.value.toFixed(1)}</span>
                </div>
                <div 
                  className="structure-bar vix-3m"
                  style={{ 
                    height: `${Math.min((data.vix3m.value / 40) * 100, 100)}%`,
                    backgroundColor: '#6366f1'
                  }}
                >
                  <span className="bar-label">3M</span>
                  <span className="bar-value">{data.vix3m.value.toFixed(1)}</span>
                </div>
                {data.vix6m && (
                  <div 
                    className="structure-bar vix-6m"
                    style={{ 
                      height: `${Math.min((data.vix6m.value / 40) * 100, 100)}%`,
                      backgroundColor: '#8b5cf6'
                    }}
                  >
                    <span className="bar-label">6M</span>
                    <span className="bar-value">{data.vix6m.value.toFixed(1)}</span>
                  </div>
                )}
              </>
            )}
          </div>
          <div className="structure-line" />
        </div>

        {/* Key Metrics */}
        <div className="metrics-row">
          <div className="metric">
            <label>Slope (VIX/VIX3M)</label>
            <span className={`value ${slope > 1 ? 'negative' : 'positive'}`}>
              {slope.toFixed(3)}
            </span>
            <small>{slope > 1 ? 'Backwardated' : 'Contango'}</small>
          </div>
          <div className="metric">
            <label>Roll Yield</label>
            <span className={`value ${rollYield > 0 ? 'positive' : 'negative'}`}>
              {(rollYield * 100).toFixed(2)}%
            </span>
            <small>Annualized</small>
          </div>
          <div className="metric">
            <label>Z-Score (1Y)</label>
            <span className="value">{zScore.toFixed(2)}</span>
            <small>{data.percentile_1y ? `${data.percentile_1y.toFixed(0)}th percentile` : ''}</small>
          </div>
        </div>

        {/* Composite Signal */}
        <div className="signal-section">
          <h4>Composite Signal</h4>
          <div className="signal-gauge">
            <div 
              className="gauge-bar"
              style={{ 
                width: `${((signal + 1) / 2) * 100}%`,
                backgroundColor: getSignalColor(signal)
              }}
            />
            <div className="gauge-labels">
              <span>--</span>
              <span>-</span>
              <span>0</span>
              <span>+</span>
              <span>++</span>
            </div>
          </div>
          <div className="signal-value" style={{ color: getSignalColor(signal) }}>
            <span className="signal-number">{signal.toFixed(3)}</span>
            <span className="signal-label">{getSignalLabel(signal)}</span>
          </div>
        </div>

        {/* Tactical Overlay State */}
        {overlayState && (
          <div className="overlay-section">
            <h4>Tactical Allocation Overlay</h4>
            <div className="allocation-shifts">
              {Object.entries(overlayState.allocation).map(([symbol, weight]) => {
                const baseWeight = symbol === 'SPY' ? 0.46 : symbol === 'GLD' ? 0.38 : 0.16;
                const shift = weight - baseWeight;
                return (
                  <div key={symbol} className="allocation-item">
                    <span className="symbol">{symbol}</span>
                    <div className="allocation-bar">
                      <div 
                        className="base-portion"
                        style={{ width: `${baseWeight * 100}%` }}
                      />
                      <div 
                        className={`shift-portion ${shift > 0 ? 'positive' : shift < 0 ? 'negative' : ''}`}
                        style={{ 
                          width: `${Math.abs(shift) * 100}%`,
                          left: shift > 0 ? `${baseWeight * 100}%` : `${(baseWeight + shift) * 100}%`
                        }}
                      />
                    </div>
                    <span className={`shift-label ${shift > 0 ? 'positive' : shift < 0 ? 'negative' : ''}`}>
                      {shift > 0 ? '+' : ''}{(shift * 100).toFixed(1)}%
                    </span>
                  </div>
                );
              })}
            </div>
            <div className="last-update">
              Last shift: {formatDate(overlayState.last_shift_date)}
            </div>
          </div>
        )}

        {/* Shift History */}
        {overlayState?.shift_history && overlayState.shift_history.length > 0 && (
          <div className="history-section">
            <h4>Recent Shifts</h4>
            <div className="shift-history">
              {overlayState.shift_history.slice(0, 5).map((shift, i) => (
                <div key={i} className="history-item">
                  <span className="history-date">
                    {new Date(shift.date).toLocaleDateString()}
                  </span>
                  <span className="history-regime">{shift.regime}</span>
                  <span className={`history-signal ${shift.signal_value < 0 ? 'negative' : 'positive'}`}>
                    {shift.signal_value.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Interpretation */}
        <div className="interpretation">
          <h4>Signal Interpretation</h4>
          <p>
            {signal > 0.5 
              ? 'Extreme contango suggests complacency. Consider reducing hedges.'
              : signal > 0 
              ? 'Normal contango regime. Standard allocation appropriate.'
              : signal > -0.5 
              ? 'Mild backwardation indicates rising uncertainty. Monitor closely.'
              : 'Strong backwardation signals risk-off. Defensive positioning active.'}
          </p>
          <div className="expected-returns">
            <small>
              Expected SPY return (20d): {signal > 0.5 ? '+0.8%' : signal > 0 ? '+0.9%' : signal > -0.5 ? '+0.3%' : signal > -0.8 ? '-1.2%' : '-3.8%'}
            </small>
          </div>
        </div>
      </div>

      <style>{`
        .vix-panel {
          background: #1f2937;
          border-radius: 8px;
          padding: 16px;
          color: #f3f4f6;
        }
        .panel-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 16px;
          padding-bottom: 12px;
          border-bottom: 1px solid #374151;
        }
        .panel-header h3 {
          margin: 0;
          font-size: 1.1rem;
          font-weight: 600;
        }
        .status-badge {
          padding: 4px 8px;
          border-radius: 4px;
          font-size: 0.75rem;
          font-weight: 600;
          color: white;
        }
        .vix-levels {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 12px;
          margin-bottom: 16px;
        }
        .vix-level {
          text-align: center;
          padding: 8px;
          background: #111827;
          border-radius: 6px;
        }
        .vix-level label {
          display: block;
          font-size: 0.75rem;
          color: #9ca3af;
          margin-bottom: 4px;
        }
        .vix-level .value {
          font-size: 1.25rem;
          font-weight: 600;
          color: #f3f4f6;
        }
        .term-structure-viz {
          margin-bottom: 16px;
          padding: 12px;
          background: #111827;
          border-radius: 6px;
        }
        .term-structure-viz h4 {
          margin: 0 0 12px 0;
          font-size: 0.875rem;
          color: #9ca3af;
        }
        .structure-bars {
          display: flex;
          align-items: flex-end;
          justify-content: center;
          gap: 24px;
          height: 120px;
          padding: 0 16px;
          position: relative;
        }
        .structure-bar {
          width: 48px;
          border-radius: 4px 4px 0 0;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: flex-end;
          padding-bottom: 8px;
          position: relative;
          transition: height 0.3s ease;
        }
        .bar-label {
          font-size: 0.625rem;
          color: rgba(255,255,255,0.8);
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }
        .bar-value {
          font-size: 0.75rem;
          font-weight: 600;
          color: white;
          margin-top: 4px;
        }
        .structure-line {
          position: absolute;
          bottom: 0;
          left: 0;
          right: 0;
          height: 1px;
          background: linear-gradient(90deg, transparent, #4b5563, transparent);
        }
        .metrics-row {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 12px;
          margin-bottom: 16px;
        }
        .metric {
          text-align: center;
          padding: 8px;
          background: #111827;
          border-radius: 6px;
        }
        .metric label {
          display: block;
          font-size: 0.625rem;
          color: #9ca3af;
          text-transform: uppercase;
          margin-bottom: 4px;
        }
        .metric .value {
          display: block;
          font-size: 1rem;
          font-weight: 600;
        }
        .metric .value.positive { color: #10b981; }
        .metric .value.negative { color: #ef4444; }
        .metric small {
          font-size: 0.625rem;
          color: #6b7280;
        }
        .signal-section {
          margin-bottom: 16px;
          padding: 12px;
          background: #111827;
          border-radius: 6px;
        }
        .signal-section h4 {
          margin: 0 0 12px 0;
          font-size: 0.875rem;
          color: #9ca3af;
        }
        .signal-gauge {
          position: relative;
          height: 24px;
          background: #374151;
          border-radius: 12px;
          overflow: hidden;
          margin-bottom: 8px;
        }
        .gauge-bar {
          height: 100%;
          border-radius: 12px;
          transition: width 0.3s ease;
        }
        .gauge-labels {
          display: flex;
          justify-content: space-between;
          font-size: 0.625rem;
          color: #6b7280;
          margin-top: 4px;
        }
        .signal-value {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .signal-number {
          font-size: 1.5rem;
          font-weight: 700;
        }
        .signal-label {
          font-size: 0.875rem;
          font-weight: 500;
        }
        .overlay-section {
          margin-bottom: 16px;
          padding: 12px;
          background: #111827;
          border-radius: 6px;
        }
        .overlay-section h4 {
          margin: 0 0 12px 0;
          font-size: 0.875rem;
          color: #9ca3af;
        }
        .allocation-shifts {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        .allocation-item {
          display: grid;
          grid-template-columns: 60px 1fr 60px;
          align-items: center;
          gap: 12px;
        }
        .allocation-item .symbol {
          font-weight: 600;
          font-size: 0.875rem;
        }
        .allocation-bar {
          position: relative;
          height: 20px;
          background: #1f2937;
          border-radius: 4px;
          overflow: hidden;
        }
        .base-portion {
          position: absolute;
          left: 0;
          height: 100%;
          background: #4b5563;
          opacity: 0.5;
        }
        .shift-portion {
          position: absolute;
          height: 100%;
          opacity: 0.8;
        }
        .shift-portion.positive { background: #10b981; }
        .shift-portion.negative { background: #ef4444; }
        .shift-label {
          font-size: 0.75rem;
          font-weight: 500;
          text-align: right;
        }
        .shift-label.positive { color: #10b981; }
        .shift-label.negative { color: #ef4444; }
        .last-update {
          margin-top: 12px;
          font-size: 0.625rem;
          color: #6b7280;
        }
        .history-section {
          margin-bottom: 16px;
          padding: 12px;
          background: #111827;
          border-radius: 6px;
        }
        .history-section h4 {
          margin: 0 0 12px 0;
          font-size: 0.875rem;
          color: #9ca3af;
        }
        .shift-history {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }
        .history-item {
          display: grid;
          grid-template-columns: 80px 1fr 60px;
          font-size: 0.75rem;
          padding: 4px 0;
          border-bottom: 1px solid #374151;
        }
        .history-item:last-child {
          border-bottom: none;
        }
        .history-date {
          color: #6b7280;
        }
        .history-regime {
          color: #9ca3af;
          text-transform: capitalize;
        }
        .history-signal {
          text-align: right;
          font-weight: 500;
        }
        .history-signal.positive { color: #10b981; }
        .history-signal.negative { color: #ef4444; }
        .interpretation {
          padding: 12px;
          background: #111827;
          border-radius: 6px;
        }
        .interpretation h4 {
          margin: 0 0 8px 0;
          font-size: 0.875rem;
          color: #9ca3af;
        }
        .interpretation p {
          margin: 0 0 8px 0;
          font-size: 0.875rem;
          line-height: 1.5;
          color: #d1d5db;
        }
        .expected-returns {
          font-size: 0.75rem;
          color: #6b7280;
        }
        .empty-state {
          text-align: center;
          color: #6b7280;
          padding: 24px;
        }
      `}</style>
    </div>
  );
}
