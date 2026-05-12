import React from 'react';
import type { DurationAllocation, DurationRegime } from '../utils/duration-signals';
import { getExpectedAlpha } from '../utils/duration-signals';

interface BondAllocationPanelProps {
  currentAllocation: DurationAllocation | null;
  targetAllocation: DurationAllocation | null;
  regime: DurationRegime | null;
  portfolioValue?: number;
  bondSlicePct?: number; // Percentage of total portfolio in bonds (default 16%)
}

const ETF_INFO: Record<keyof DurationAllocation, { 
  name: string; 
  description: string;
  duration: string;
  color: string;
}> = {
  tlt: {
    name: 'TLT',
    description: 'iShares 20+ Year Treasury',
    duration: '20+ years',
    color: '#3b82f6'
  },
  ief: {
    name: 'IEF',
    description: 'iShares 7-10 Year Treasury',
    duration: '7-10 years',
    color: '#8b5cf6'
  },
  shy: {
    name: 'SHY',
    description: 'iShares 1-3 Year Treasury',
    duration: '1-3 years',
    color: '#10b981'
  },
  bil: {
    name: 'BIL',
    description: 'SPDR Bloomberg 1-3 Month',
    duration: '0-1 year',
    color: '#f59e0b'
  }
};

export function BondAllocationPanel({ 
  currentAllocation, 
  targetAllocation,
  regime,
  portfolioValue = 100000,
  bondSlicePct = 0.16
}: BondAllocationPanelProps) {
  const bondValue = portfolioValue * bondSlicePct;

  // Calculate drift for each ETF
  const calculateDrift = (current: number, target: number) => {
    return Math.abs(current - target);
  };

  // Check if rebalance needed (10% threshold)
  const needsRebalance = React.useMemo(() => {
    if (!currentAllocation || !targetAllocation) return false;
    const threshold = 0.10; // 10% drift threshold
    return (
      calculateDrift(currentAllocation.tlt, targetAllocation.tlt) > threshold ||
      calculateDrift(currentAllocation.ief, targetAllocation.ief) > threshold ||
      calculateDrift(currentAllocation.shy, targetAllocation.shy) > threshold ||
      calculateDrift(currentAllocation.bil, targetAllocation.bil) > threshold
    );
  }, [currentAllocation, targetAllocation]);

  const expectedAlpha = regime ? getExpectedAlpha(regime) : 0;

  if (!targetAllocation) {
    return (
      <div className="bond-allocation-panel loading">
        <h3>Bond Allocation</h3>
        <div className="panel-content">
          <span className="loading-text">Loading allocation data...</span>
        </div>
      </div>
    );
  }

  const etfKeys: (keyof DurationAllocation)[] = ['tlt', 'ief', 'shy', 'bil'];

  return (
    <div className="bond-allocation-panel">
      <h3>Bond Allocation ({(bondSlicePct * 100).toFixed(0)}% of portfolio)</h3>
      
      <div className="panel-content">
        {/* Expected Alpha Badge */}
        {regime && (
          <div className={`alpha-badge ${expectedAlpha > 0 ? 'positive' : 'neutral'}`}>
            <span className="alpha-label">Expected Alpha vs Static:</span>
            <span className="alpha-value">
              {expectedAlpha > 0 ? '+' : ''}{expectedAlpha.toFixed(1)}%
            </span>
          </div>
        )}

        {/* Rebalance Alert */}
        {needsRebalance && (
          <div className="rebalance-alert">
            <span className="alert-icon">⚠️</span>
            <span className="alert-text">Rebalance recommended (drift &gt; 10%)</span>
          </div>
        )}

        {/* Allocation Table */}
        <div className="allocation-table">
          <div className="table-header">
            <span>ETF</span>
            <span>Target</span>
            {currentAllocation && <span>Current</span>}
            {currentAllocation && <span>Drift</span>}
            <span>Value</span>
          </div>
          
          {etfKeys.map(key => {
            const info = ETF_INFO[key];
            const target = targetAllocation[key];
            const current = currentAllocation?.[key] ?? target;
            const drift = calculateDrift(current, target);
            const value = bondValue * target;
            
            return (
              <div key={key} className="allocation-row">
                <div className="etf-info">
                  <span 
                    className="etf-color" 
                    style={{ backgroundColor: info.color }}
                  />
                  <div className="etf-details">
                    <span className="etf-name">{info.name}</span>
                    <span className="etf-desc">{info.duration}</span>
                  </div>
                </div>
                
                <span className="target-pct">{(target * 100).toFixed(1)}%</span>
                
                {currentAllocation && (
                  <span className={`current-pct ${drift > 0.10 ? 'drift-high' : ''}`}>
                    {(current * 100).toFixed(1)}%
                  </span>
                )}
                
                {currentAllocation && (
                  <span className={`drift-pct ${drift > 0.10 ? 'drift-high' : ''}`}>
                    {drift > 0 ? '↔' : '✓'} {(drift * 100).toFixed(1)}%
                  </span>
                )}
                
                <span className="value-amount">
                  ${value.toLocaleString('en-US', { maximumFractionDigits: 0 })}
                </span>
              </div>
            );
          })}
        </div>

        {/* Pie Chart Summary */}
        <div className="allocation-visual">
          <svg viewBox="0 0 100 100" className="pie-chart">
            {(() => {
              let cumulativePct = 0;
              return etfKeys.map((key, i) => {
                const pct = targetAllocation[key];
                if (pct < 0.01) return null; // Skip if < 1%
                
                const startAngle = (cumulativePct * 360) - 90;
                const endAngle = ((cumulativePct + pct) * 360) - 90;
                cumulativePct += pct;
                
                const startRad = (startAngle * Math.PI) / 180;
                const endRad = (endAngle * Math.PI) / 180;
                
                const x1 = 50 + 40 * Math.cos(startRad);
                const y1 = 50 + 40 * Math.sin(startRad);
                const x2 = 50 + 40 * Math.cos(endRad);
                const y2 = 50 + 40 * Math.sin(endRad);
                
                const largeArc = pct > 0.5 ? 1 : 0;
                
                return (
                  <path
                    key={key}
                    d={`M 50 50 L ${x1} ${y1} A 40 40 0 ${largeArc} 1 ${x2} ${y2} Z`}
                    fill={ETF_INFO[key].color}
                    stroke="#fff"
                    strokeWidth="1"
                  />
                );
              });
            })()}
            <circle cx="50" cy="50" r="25" fill="var(--card-bg, #ffffff)" />
            <text x="50" y="45" textAnchor="middle" fontSize="8" fill="#6b7280">
              Total
            </text>
            <text x="50" y="58" textAnchor="middle" fontSize="12" fontWeight="bold" fill="#374151">
              ${(bondValue / 1000).toFixed(1)}K
            </text>
          </svg>
          
          <div className="pie-legend">
            {etfKeys.filter(key => targetAllocation[key] >= 0.01).map(key => (
              <div key={key} className="legend-item">
                <span 
                  className="legend-color" 
                  style={{ backgroundColor: ETF_INFO[key].color }}
                />
                <span className="legend-label">{ETF_INFO[key].name}</span>
                <span className="legend-pct">
                  {(targetAllocation[key] * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <style>{`
        .bond-allocation-panel {
          background: #1e293b;
          border-radius: 8px;
          padding: 16px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        }

        .bond-allocation-panel.loading {
          opacity: 0.7;
        }

        .bond-allocation-panel h3 {
          margin: 0 0 12px 0;
          font-size: 14px;
          font-weight: 600;
          color: #94a3b8;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .panel-content {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }

        .alpha-badge {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 8px 12px;
          border-radius: 6px;
          font-size: 13px;
        }

        .alpha-badge.positive {
          background: #064e3b;
          color: #6ee7b7;
        }

        .alpha-badge.neutral {
          background: #374151;
          color: #9ca3af;
        }

        .alpha-label {
          font-weight: 500;
        }

        .alpha-value {
          font-weight: 700;
        }

        .rebalance-alert {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 10px 12px;
          background: #451a03;
          border: 1px solid #92400e;
          border-radius: 6px;
          color: #fbbf24;
          font-size: 13px;
        }

        .alert-icon {
          font-size: 16px;
        }

        .allocation-table {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .table-header {
          display: grid;
          grid-template-columns: 2fr 1fr 1fr 1fr 1fr;
          gap: 8px;
          padding: 8px;
          font-size: 11px;
          font-weight: 600;
          color: #6b7280;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          border-bottom: 1px solid #334155;
        }

        .allocation-row {
          display: grid;
          grid-template-columns: 2fr 1fr 1fr 1fr 1fr;
          gap: 8px;
          padding: 10px 8px;
          align-items: center;
          border-radius: 4px;
          transition: background 0.2s;
        }

        .allocation-row:hover {
          background: #0f172a;
        }

        .etf-info {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .etf-color {
          width: 12px;
          height: 12px;
          border-radius: 3px;
        }

        .etf-details {
          display: flex;
          flex-direction: column;
        }

        .etf-name {
          font-weight: 600;
          font-size: 13px;
          color: #e2e8f0;
        }

        .etf-desc {
          font-size: 11px;
          color: #6b7280;
        }

        .target-pct {
          font-weight: 600;
          font-size: 13px;
          color: #e2e8f0;
        }

        .current-pct {
          font-size: 13px;
          color: #94a3b8;
        }

        .drift-pct {
          font-size: 12px;
          color: #10b981;
        }

        .drift-pct.drift-high {
          color: #f59e0b;
          font-weight: 600;
        }

        .current-pct.drift-high {
          color: #f59e0b;
          font-weight: 600;
        }

        .value-amount {
          font-size: 13px;
          font-weight: 500;
          color: #e2e8f0;
        }

        .allocation-visual {
          display: flex;
          align-items: center;
          gap: 20px;
          padding-top: 12px;
          border-top: 1px solid #334155;
        }

        .pie-chart {
          width: 100px;
          height: 100px;
          flex-shrink: 0;
        }

        .pie-legend {
          display: flex;
          flex-direction: column;
          gap: 6px;
          flex: 1;
        }

        .legend-item {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 12px;
        }

        .legend-color {
          width: 10px;
          height: 10px;
          border-radius: 2px;
        }

        .legend-label {
          color: #94a3b8;
          flex: 1;
        }

        .legend-pct {
          font-weight: 600;
          color: #e2e8f0;
        }

        .loading-text {
          color: #6b7280;
          font-size: 14px;
        }

        @media (max-width: 768px) {
          .table-header,
          .allocation-row {
            grid-template-columns: 2fr 1fr 1fr;
          }

          .table-header > :nth-child(3),
          .table-header > :nth-child(4),
          .allocation-row > :nth-child(4),
          .allocation-row > :nth-child(5) {
            display: none;
          }

          .allocation-visual {
            flex-direction: column;
            align-items: center;
          }
        }
      `}</style>
    </div>
  );
}
