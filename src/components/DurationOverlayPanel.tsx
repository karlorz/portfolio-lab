import React from 'react';
import '../App.css';

interface DurationAllocation {
  tlt: number;
  ief: number;
  shy: number;
  bil: number;
}

interface YieldCurveData {
  spread2s10s: number;
  dgs2: number;
  dgs10: number;
  duration_regime: 'inverted' | 'flat' | 'steep' | 'normal';
  spread_history?: number[];
}

interface DurationOverlayProps {
  yieldCurve?: YieldCurveData | null;
  durationAllocation?: DurationAllocation | null;
}

const DurationOverlayPanel: React.FC<DurationOverlayProps> = ({ 
  yieldCurve, 
  durationAllocation 
}) => {
  if (!yieldCurve || !durationAllocation) {
    return (
      <div className="duration-overlay-panel">
        <h3>Duration Overlay</h3>
        <p className="loading-text">Loading yield curve data...</p>
      </div>
    );
  }

  const { spread2s10s, duration_regime, spread_history } = yieldCurve;
  const { tlt, ief, shy, bil } = durationAllocation;

  // Calculate effective duration
  const effectiveDuration = (
    tlt * 18.5 + 
    ief * 7.5 + 
    shy * 1.9 + 
    bil * 0.1
  ) / (tlt + ief + shy + bil || 1);

  // Sparkline for spread history
  const renderSparkline = () => {
    if (!spread_history || spread_history.length < 2) return null;
    
    const min = Math.min(...spread_history);
    const max = Math.max(...spread_history);
    const range = max - min || 1;
    const width = 100;
    const height = 30;
    
    const points = spread_history.map((val, i) => {
      const x = (i / (spread_history.length - 1)) * width;
      const y = height - ((val - min) / range) * height;
      return `${x},${y}`;
    }).join(' ');

    const zeroY = height - ((0 - min) / range) * height;

    return (
      <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} className="sparkline">
        <line x1="0" y1={zeroY} x2={width} y2={zeroY} stroke="#666" strokeWidth="0.5" strokeDasharray="2" />
        <polyline points={points} fill="none" stroke="#3b82f6" strokeWidth="2" />
        <circle cx={width} cy={height - ((spread_history[spread_history.length - 1] - min) / range) * height} r="3" fill="#3b82f6" />
      </svg>
    );
  };

  const getRegimeColor = (regime: string) => {
    switch (regime) {
      case 'inverted': return '#ef4444'; // red
      case 'flat': return '#f59e0b'; // amber
      case 'steep': return '#10b981'; // green
      case 'normal': return '#3b82f6'; // blue
      default: return '#6b7280';
    }
  };

  const getRegimeDescription = (regime: string) => {
    switch (regime) {
      case 'inverted': return 'Short duration bias (recession risk)';
      case 'flat': return 'Neutral duration stance';
      case 'steep': return 'Long duration preference';
      case 'normal': return 'Moderate duration';
      default: return '';
    }
  };

  return (
    <div className="duration-overlay-panel">
      <div className="panel-header">
        <h3>Duration Overlay</h3>
        <span 
          className="regime-badge"
          style={{ backgroundColor: getRegimeColor(duration_regime) }}
        >
          {duration_regime.toUpperCase()}
        </span>
      </div>

      <div className="yield-curve-section">
        <div className="metric-row">
          <span className="metric-label">10Y - 2Y Spread:</span>
          <span className={`metric-value ${spread2s10s < 0 ? 'negative' : 'positive'}`}>
            {spread2s10s > 0 ? '+' : ''}{(spread2s10s / 100).toFixed(2)}%
          </span>
        </div>
        <div className="sparkline-container">
          {renderSparkline()}
        </div>
        <p className="regime-description">{getRegimeDescription(duration_regime)}</p>
      </div>

      <div className="duration-breakdown">
        <h4>Duration Allocation</h4>
        <div className="allocation-bars">
          <div className="allocation-item">
            <span className="allocation-label">TLT (Long)</span>
            <div className="bar-container">
              <div className="bar" style={{ width: `${tlt * 100}%`, backgroundColor: '#10b981' }} />
            </div>
            <span className="allocation-value">{(tlt * 100).toFixed(0)}%</span>
          </div>
          <div className="allocation-item">
            <span className="allocation-label">IEF (Interm)</span>
            <div className="bar-container">
              <div className="bar" style={{ width: `${ief * 100}%`, backgroundColor: '#3b82f6' }} />
            </div>
            <span className="allocation-value">{(ief * 100).toFixed(0)}%</span>
          </div>
          <div className="allocation-item">
            <span className="allocation-label">SHY (Short)</span>
            <div className="bar-container">
              <div className="bar" style={{ width: `${shy * 100}%`, backgroundColor: '#f59e0b' }} />
            </div>
            <span className="allocation-value">{(shy * 100).toFixed(0)}%</span>
          </div>
          <div className="allocation-item">
            <span className="allocation-label">BIL (Cash)</span>
            <div className="bar-container">
              <div className="bar" style={{ width: `${bil * 100}%`, backgroundColor: '#6b7280' }} />
            </div>
            <span className="allocation-value">{(bil * 100).toFixed(0)}%</span>
          </div>
        </div>
      </div>

      <div className="effective-duration">
        <span className="duration-label">Effective Duration:</span>
        <span className="duration-value">{effectiveDuration.toFixed(1)} years</span>
      </div>

      <div className="overlay-note">
        <small>Dynamic duration targeting based on yield curve regime. 
        Backtest: +0.011 Sharpe improvement vs static allocation.</small>
      </div>
    </div>
  );
};

export default DurationOverlayPanel;
