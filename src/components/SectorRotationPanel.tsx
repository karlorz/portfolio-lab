import React from 'react';

interface SectorEntry {
  symbol: string;
  name: string;
  momentumScore: number;
  allocation: number;
  rank: number;
  longMomentum?: number;
  shortMomentum?: number;
  volatility?: number;
}

interface AllocationSector {
  symbol: string;
  weight: number;
  momentum: number;
  rank: number;
}

interface SectorAllocation {
  spy_core: number;
  spy_total: number;
  sector_overlay: number;
  sectors: AllocationSector[];
}

interface SectorRotationData {
  timestamp: string;
  status: string;
  vix?: number;
  regime?: string | null;
  methodology?: string;
  overlay_pct?: number;
  top_sectors: SectorEntry[];
  allocation?: SectorAllocation;
  rebalanceRecommended?: boolean;
  rebalanceReason?: string;
  // VIX-disabled state
  spAllocation?: number;
  sectorAllocations?: never[];
  totalEquityWeight?: number;
  regimeAdjusted?: boolean;
}

interface SectorRotationPanelProps {
  data: SectorRotationData | null;
}

const SECTOR_COLORS: Record<string, string> = {
  XLK: '#3b82f6',
  XLF: '#10b981',
  XLE: '#f59e0b',
  XLV: '#ef4444',
  XLY: '#8b5cf6',
  XLI: '#06b6d4',
  XLP: '#84cc16',
  XLB: '#f97316',
  XLRE: '#ec4899',
  XLU: '#6366f1',
  XLC: '#14b8a6',
};

function SectorBar({ entry }: { entry: SectorEntry }) {
  const maxMom = 0.20;
  const pct = Math.min(Math.abs(entry.momentumScore) / maxMom * 100, 100);
  const isPositive = entry.momentumScore >= 0;
  const color = SECTOR_COLORS[entry.symbol] || '#6b7280';

  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 2 }}>
        <span style={{ fontSize: 11, color: color }}>
          {entry.rank}. {entry.symbol}
          <span style={{ color: '#94a3b8', marginLeft: 4 }}>{entry.name}</span>
        </span>
        <span style={{
          fontSize: 11,
          color: isPositive ? '#10b981' : '#ef4444',
        }}>
          {entry.momentumScore >= 0 ? '+' : ''}{(entry.momentumScore * 100).toFixed(1)}%
          <span style={{ color: '#64748b', marginLeft: 4 }}>
            {(entry.allocation * 100).toFixed(1)}%
          </span>
        </span>
      </div>
      <div style={{
        height: 6, background: '#1e293b', borderRadius: 3, overflow: 'hidden',
      }}>
        <div style={{
          width: `${pct}%`,
          height: '100%',
          background: isPositive ? '#10b981' : '#ef4444',
          borderRadius: 3,
        }} />
      </div>
    </div>
  );
}

export function SectorRotationPanel({ data }: SectorRotationPanelProps) {
  if (!data) {
    return (
      <div className="panel">
        <h3>Sector Rotation</h3>
        <p className="muted">No sector rotation data available</p>
      </div>
    );
  }

  const isDisabled = data.status !== 'active' || (data.sectorAllocations && data.sectorAllocations.length === 0);

  return (
    <div className="panel">
      <h3>Sector Rotation</h3>

      {isDisabled ? (
        <div>
          <div className="panel-grid">
            <div className="metric">
              <span className="label">Status</span>
              <span className="value" style={{ color: '#f59e0b' }}>Disabled</span>
            </div>
            {data.totalEquityWeight !== undefined && (
              <div className="metric">
                <span className="label">SPY Allocation</span>
                <span className="value">{(data.totalEquityWeight * 100).toFixed(1)}%</span>
              </div>
            )}
          </div>
          {data.rebalanceReason && (
            <p className="muted small" style={{ marginTop: 6 }}>{data.rebalanceReason}</p>
          )}
        </div>
      ) : (
        <div>
          {/* Overview metrics */}
          <div className="panel-grid">
            <div className="metric">
              <span className="label">VIX</span>
              <span className="value" style={{
                color: (data.vix || 0) < 15 ? '#10b981' : (data.vix || 0) < 25 ? '#f59e0b' : '#ef4444',
              }}>
                {(data.vix || 0).toFixed(1)}
              </span>
            </div>
            {data.regime && (
              <div className="metric">
                <span className="label">Regime</span>
                <span className="value">{data.regime.replace(/_/g, ' ')}</span>
              </div>
            )}
            {data.allocation && (
              <div className="metric">
                <span className="label">SPY Core</span>
                <span className="value">{(data.allocation.spy_core * 100).toFixed(1)}%</span>
              </div>
            )}
            {data.allocation && (
              <div className="metric">
                <span className="label">Sector Overlay</span>
                <span className="value">{(data.allocation.sector_overlay * 100).toFixed(1)}%</span>
              </div>
            )}
          </div>

          {/* Top sectors */}
          {data.top_sectors && data.top_sectors.length > 0 && (
            <div style={{ marginTop: 10 }}>
              <span className="label" style={{ display: 'block', marginBottom: 6 }}>Top Sectors</span>
              {data.top_sectors.map((sector) => (
                <SectorBar key={sector.symbol} entry={sector} />
              ))}
            </div>
          )}

          {/* Rebalance recommendation */}
          {data.rebalanceRecommended && (
            <div style={{
              marginTop: 8, padding: '6px 8px', borderRadius: 4,
              background: '#1e3a5f', color: '#93c5fd', fontSize: 12,
            }}>
              Rebalance: {data.rebalanceReason || 'Recommended'}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
