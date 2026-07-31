import React from 'react';
import type { FactorRotationSignalData } from '../types/live';

interface FactorRotationPanelProps {
  data: FactorRotationSignalData | null | undefined;
}

type AllocationEntry = {
  symbol: string;
  pct: number | null;
};

const FACTOR_COLORS = [
  'var(--chart-1, #3b82f6)',
  'var(--chart-2, #8b5cf6)',
  'var(--chart-3, #10b981)',
  'var(--chart-4, #f59e0b)',
  'var(--chart-5, #ec4899)',
];

function finiteNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

function allocationToPct(value: unknown): number | null {
  const parsed = finiteNumber(value);
  if (parsed === null) return null;
  return Math.abs(parsed) <= 1 ? parsed * 100 : parsed;
}

function formatSignedScore(value: unknown): string {
  const parsed = finiteNumber(value);
  if (parsed === null) return 'Unavailable';
  return `${parsed >= 0 ? '+' : ''}${parsed.toFixed(2)}`;
}

function formatPct(value: number | null): string {
  if (value === null) return 'Unavailable';
  return `${value.toFixed(0)}%`;
}

function normalizeAllocation(allocation: FactorRotationSignalData['allocation']): AllocationEntry[] {
  if (!allocation || typeof allocation !== 'object') return [];
  return Object.entries(allocation)
    .map(([symbol, value]) => ({ symbol, pct: allocationToPct(value) }))
    .filter((entry) => entry.symbol.trim() !== '')
    .sort((a, b) => a.symbol.localeCompare(b.symbol));
}

function toneForScore(value: unknown): 'positive' | 'negative' | 'muted' {
  const parsed = finiteNumber(value);
  if (parsed === null || parsed === 0) return 'muted';
  return parsed > 0 ? 'positive' : 'negative';
}

function FactorBars({ entries }: { entries: AllocationEntry[] }) {
  if (entries.length === 0) {
    return <p className="muted small">Allocation unavailable — advisory only.</p>;
  }

  return (
    <div className="factor-bars" aria-label="Factor allocation">
      {entries.map((entry, index) => (
        <div className="factor-bar-row" key={entry.symbol}>
          <div className="factor-bar-label">
            <span
              className="legend-dot"
              style={{ backgroundColor: FACTOR_COLORS[index % FACTOR_COLORS.length] }}
            />
            <span>{entry.symbol}</span>
          </div>
          <div className="factor-bar-track" aria-hidden="true">
            <span
              className="factor-bar-fill"
              style={{
                width: `${Math.max(0, Math.min(entry.pct ?? 0, 100))}%`,
                backgroundColor: FACTOR_COLORS[index % FACTOR_COLORS.length],
              }}
            />
          </div>
          <span className="legend-value">{formatPct(entry.pct)}</span>
        </div>
      ))}
    </div>
  );
}

export function FactorRotationPanel({ data }: FactorRotationPanelProps) {
  if (!data) {
    return (
      <div className="panel signal-card factor-rotation-card">
        <h3>Factor Rotation</h3>
        <p className="muted">No factor rotation data available.</p>
      </div>
    );
  }

  const selectedFactors = Array.isArray(data.selected_factors)
    ? data.selected_factors.filter((factor) => typeof factor === 'string' && factor.trim() !== '')
    : [];
  const allocation = normalizeAllocation(data.allocation);
  const recommendation =
    typeof data.recommendation === 'string' && data.recommendation.trim() !== ''
      ? data.recommendation
      : 'Recommendation unavailable — advisory only.';
  const scoreTone = toneForScore(data.signal_strength);

  return (
    <div className="panel signal-card factor-rotation-card">
      <h3>Factor Rotation</h3>

      <div className="panel-section signal-card-section">
        <div className="metric-row">
          <span className="label">Role</span>
          <span className="badge badge-advisory">Advisory</span>
        </div>
        <div className="metric-row">
          <span className="label">Signal Strength</span>
          <span className={`value ${scoreTone}`}>{formatSignedScore(data.signal_strength)}</span>
        </div>
        <div className="metric-row">
          <span className="label">Recommendation</span>
          <span className="value">{recommendation}</span>
        </div>
      </div>

      <div className="panel-section signal-card-section">
        <h4>Selected Factors</h4>
        {selectedFactors.length > 0 ? (
          <div className="factor-chip-row" aria-label="Selected factors">
            {selectedFactors.map((factor) => (
              <span className="badge" key={factor}>{factor}</span>
            ))}
          </div>
        ) : (
          <p className="muted small">Selected factors unavailable — advisory only.</p>
        )}
      </div>

      <div className="panel-section signal-card-section">
        <h4>Allocation</h4>
        <FactorBars entries={allocation} />
      </div>
    </div>
  );
}
