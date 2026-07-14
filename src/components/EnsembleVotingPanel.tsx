import React from 'react';

interface SourceVote {
  source: string;
  direction: number | string;
  strength: number;
  confidence: number;
  weight: number;
}

interface AdaptiveLearningBranch {
  status: string;
  enabled: boolean;
  reason: string;
  observations?: number;
  warmup_days?: number;
  max_blend?: number;
  current_blend?: number;
  state_available?: boolean;
  blend_alpha?: number;
}

interface AdaptiveLearningDisclosure {
  bandit?: AdaptiveLearningBranch;
  online_ic?: AdaptiveLearningBranch;
}

interface ConfiguredSourceStatus {
  source: string;
  label?: string;
  configured: boolean;
  configured_weight?: number;
  collected: boolean;
  active: boolean;
  contributing: boolean;
  status: string;
  reason?: string;
}

export interface AllocationSurfaceRole {
  label: string;
  role: 'execution_routed' | 'advisory_non_routed' | 'execution_blocked';
  routed: boolean;
  routed_by: string | null;
  description: string;
  live_authoritative?: boolean;
  execution_blocked?: boolean;
  kill_switch_enabled?: boolean;
  kill_switch_level?: string | null;
}

type NormalizedSourceVote = Omit<SourceVote, 'direction'> & {
  direction: number;
};

export interface EnsembleVotingData {
  regime: string;
  regime_confidence: number;
  weighted_consensus: number;
  agreement_ratio: number;
  action: string;
  confidence: number;
  equity_bias: number;
  duration_bias: number;
  gold_bias: number;
  num_sources: number;
  configured_source_count?: number;
  collected_source_count?: number;
  contributing_source_count?: number;
  inactive_source_count?: number;
  inactive_sources?: string[];
  configured_source_status?: ConfiguredSourceStatus[];
  adaptive_learning?: AdaptiveLearningDisclosure;
  source_breakdown: SourceVote[];
}

interface EnsembleVotingPanelProps {
  data: EnsembleVotingData | null;
  allocationSurfaceRole?: AllocationSurfaceRole;
}

const REGIME_COLORS: Record<string, string> = {
  LOW_VOL: '#10b981',
  NORMAL: '#3b82f6',
  HIGH_VOL: '#f59e0b',
  CRISIS: '#ef4444',
  RECOVERY: '#8b5cf6',
};

const ACTION_LABELS: Record<string, string> = {
  increase_equity: 'Risk-On',
  decrease_equity: 'Risk-Off',
  neutral: 'Neutral',
  risk_off: 'Defensive',
};

const SOURCE_LABELS: Record<string, string> = {
  MULTI_SPEED_MOM: 'MSM',
  CROSS_ASSET_RV: 'Cross-RV',
  ALTERNATIVE_DATA: 'Alt Data',
  INTERNATIONAL_MOMENTUM: 'Intl Mom',
  CROSS_ASSET_REGIME_ARB: 'Regime Arb',
  UNIFIED_OVERLAY: 'Unified',
  google_trends: 'Google Trends',
};

const ADAPTIVE_STATUS_COLORS: Record<string, string> = {
  active: '#10b981',
  disabled: '#94a3b8',
  unavailable: '#ef4444',
  non_effective: '#f59e0b',
};

const CONFIGURED_SOURCE_STATUS_COLORS: Record<string, string> = {
  active: '#10b981',
  stale: '#f59e0b',
  missing: '#f59e0b',
  zero_weight: '#94a3b8',
  unavailable: '#ef4444',
  inactive: '#f59e0b',
};

function safeNumber(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function directionToNumber(value: unknown): number {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value !== 'string') return 0;
  const normalized = value.trim().toLowerCase();
  if (normalized === 'bullish' || normalized === 'buy' || normalized === 'positive') return 1;
  if (normalized === 'bearish' || normalized === 'sell' || normalized === 'negative') return -1;
  const numeric = Number(normalized);
  return Number.isFinite(numeric) ? numeric : 0;
}

export function normalizeSourceVote(raw: unknown): NormalizedSourceVote {
  const source = raw && typeof raw === 'object' ? raw as Record<string, unknown> : {};
  return {
    source: typeof source.source === 'string' && source.source.trim() !== '' ? source.source : 'unknown',
    direction: directionToNumber(source.direction),
    strength: safeNumber(source.strength),
    confidence: safeNumber(source.confidence),
    weight: safeNumber(source.weight),
  };
}

export function formatSourceDirection(value: unknown): string {
  const direction = directionToNumber(value);
  return `${direction > 0 ? '+' : ''}${direction.toFixed(0)}`;
}

function formatAdaptiveStatus(status: string): string {
  const normalized = status.replaceAll('_', '-');
  return normalized.charAt(0).toUpperCase() + normalized.slice(1);
}

function formatAdaptiveReason(reason: string): string {
  return reason.replace(/^.*:/, '').replaceAll('_', ' ');
}

export function formatConfiguredSourceStatus(status: string): string {
  const normalized = status.replaceAll('_', ' ');
  return normalized.charAt(0).toUpperCase() + normalized.slice(1);
}

function DirectionBar({ value, maxAbs = 1 }: { value: number; maxAbs?: number }) {
  const pct = Math.min(Math.abs(value) / maxAbs * 100, 100);
  const isPositive = value >= 0;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, width: '100%' }}>
      <div style={{
        flex: 1, height: 6, background: '#1e293b', borderRadius: 3, position: 'relative', overflow: 'hidden',
      }}>
        <div style={{
          position: 'absolute',
          left: isPositive ? '50%' : `${50 - pct}%`,
          width: `${pct / 2}%`,
          height: '100%',
          background: isPositive ? '#10b981' : '#ef4444',
          borderRadius: 3,
        }} />
      </div>
      <span style={{ fontSize: 11, color: isPositive ? '#10b981' : '#ef4444', minWidth: 36, textAlign: 'right' }}>
        {value >= 0 ? '+' : ''}{value.toFixed(2)}
      </span>
    </div>
  );
}

function AllocationSurfaceRoleDisclosure({ role }: { role?: AllocationSurfaceRole }) {
  if (!role) return null;
  const status = role.routed ? 'Order-routed' : 'Not order-routed';
  const routeText = role.routed_by ? ` via ${role.routed_by}` : '';

  return (
    <div
      style={{
        marginTop: 12,
        background: '#0f172a',
        border: '1px solid #334155',
        borderRadius: 6,
        padding: 8,
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, flexWrap: 'wrap' }}>
        <span className="label">Live Role</span>
        <span
          style={{
            color: role.routed ? '#10b981' : '#f59e0b',
            fontSize: 12,
            fontWeight: 600,
            minWidth: 0,
            overflowWrap: 'anywhere',
          }}
        >
          {status}{routeText}
        </span>
      </div>
      <div style={{ color: '#94a3b8', fontSize: 11, marginTop: 4, overflowWrap: 'anywhere' }}>
        {role.description}
      </div>
    </div>
  );
}

export function EnsembleVotingPanel({ data, allocationSurfaceRole }: EnsembleVotingPanelProps) {
  if (!data) {
    return (
      <div className="panel">
        <h3>Ensemble Voting</h3>
        <AllocationSurfaceRoleDisclosure role={allocationSurfaceRole} />
        <p className="muted">No ensemble data available</p>
      </div>
    );
  }

  const regimeColor = REGIME_COLORS[data.regime] || '#6b7280';
  const actionLabel = ACTION_LABELS[data.action] || data.action;
  const consensusColor = data.weighted_consensus > 0.2 ? '#10b981' :
    data.weighted_consensus < -0.2 ? '#ef4444' : '#f59e0b';
  const sourceBreakdown = Array.isArray(data.source_breakdown)
    ? data.source_breakdown.map(normalizeSourceVote)
    : [];
  const collectedSourceCount = safeNumber(data.collected_source_count, data.num_sources);
  const configuredSourceCount = safeNumber(data.configured_source_count, collectedSourceCount);
  const contributingSourceCount = safeNumber(
    data.contributing_source_count,
    sourceBreakdown.filter((src) => src.weight > 0).length,
  );
  const inactiveSourceCount = safeNumber(
    data.inactive_source_count,
    Math.max(collectedSourceCount - contributingSourceCount, 0),
  );
  const configuredSourceStatus = Array.isArray(data.configured_source_status)
    ? data.configured_source_status
    : [];
  const inactiveConfiguredSources = configuredSourceStatus.filter((source) => !source.active);
  const adaptiveBranches = [
    { label: 'Bandit', branch: data.adaptive_learning?.bandit },
    { label: 'Online IC', branch: data.adaptive_learning?.online_ic },
  ].filter((entry): entry is { label: string; branch: AdaptiveLearningBranch } => Boolean(entry.branch));

  return (
    <div className="panel">
      <h3>Ensemble Voting</h3>
      <AllocationSurfaceRoleDisclosure role={allocationSurfaceRole} />
      <div className="panel-grid">
        <div className="metric">
          <span className="label">Regime</span>
          <span className="value" style={{ color: regimeColor }}>
            {data.regime.replace('_', ' ')}
          </span>
        </div>
        <div className="metric">
          <span className="label">Action</span>
          <span className="value" style={{ color: data.action === 'increase_equity' ? '#10b981' : data.action === 'risk_off' ? '#ef4444' : '#f59e0b' }}>
            {actionLabel}
          </span>
        </div>
        <div className="metric">
          <span className="label">Consensus</span>
          <span className="value" style={{ color: consensusColor }}>
            {(data.weighted_consensus * 100).toFixed(0)}%
          </span>
        </div>
        <div className="metric">
          <span className="label">Agreement</span>
          <span className="value">{(data.agreement_ratio * 100).toFixed(0)}%</span>
        </div>
        <div className="metric">
          <span className="label">Confidence</span>
          <span className="value">{(data.confidence * 100).toFixed(0)}%</span>
        </div>
        <div className="metric">
          <span className="label">Configured Sources</span>
          <span className="value">{configuredSourceCount}</span>
        </div>
        <div className="metric">
          <span className="label">Collected Sources</span>
          <span className="value">{collectedSourceCount}</span>
        </div>
        <div className="metric">
          <span className="label">Contributing Sources</span>
          <span className="value">{contributingSourceCount}</span>
        </div>
        <div className="metric">
          <span className="label">Inactive/Zero Weight</span>
          <span className="value">{inactiveSourceCount}</span>
        </div>
      </div>

      {/* Asset Bias Bars */}
      <div style={{ marginTop: 12 }}>
        <div className="metric" style={{ marginBottom: 6 }}>
          <span className="label">Equity (SPY)</span>
          <DirectionBar value={data.equity_bias} />
        </div>
        <div className="metric" style={{ marginBottom: 6 }}>
          <span className="label">Duration (TLT)</span>
          <DirectionBar value={data.duration_bias} />
        </div>
        <div className="metric">
          <span className="label">Gold (GLD)</span>
          <DirectionBar value={data.gold_bias} />
        </div>
      </div>

      {adaptiveBranches.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <span className="label" style={{ marginBottom: 6, display: 'block' }}>
            Adaptive Learning
          </span>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 8 }}>
            {adaptiveBranches.map(({ label, branch }) => (
              <div
                key={label}
                style={{
                  background: '#0f172a',
                  border: '1px solid #1e293b',
                  borderRadius: 6,
                  padding: 8,
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                  <span style={{ color: '#e2e8f0', fontSize: 12, fontWeight: 600 }}>{label}</span>
                  <span
                    style={{
                      color: ADAPTIVE_STATUS_COLORS[branch.status] || '#94a3b8',
                      fontSize: 12,
                      fontWeight: 600,
                    }}
                  >
                    {formatAdaptiveStatus(branch.status)}
                  </span>
                </div>
                <div style={{ color: '#94a3b8', fontSize: 11, marginTop: 4 }}>
                  {formatAdaptiveReason(branch.reason)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {inactiveConfiguredSources.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <span className="label" style={{ marginBottom: 6, display: 'block' }}>
            Configured Source Status
          </span>
          <div style={{ display: 'grid', gap: 8 }}>
            {inactiveConfiguredSources.map((source) => (
              <div
                key={source.source}
                style={{
                  background: '#0f172a',
                  border: '1px solid #1e293b',
                  borderRadius: 6,
                  padding: 8,
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                  <span style={{ color: '#e2e8f0', fontSize: 12, fontWeight: 600 }}>
                    {source.label || SOURCE_LABELS[source.source] || source.source}
                  </span>
                  <span
                    style={{
                      color: CONFIGURED_SOURCE_STATUS_COLORS[source.status] || '#94a3b8',
                      fontSize: 12,
                      fontWeight: 600,
                    }}
                  >
                    {formatConfiguredSourceStatus(source.status)}
                  </span>
                </div>
                {source.reason && (
                  <div style={{ color: '#94a3b8', fontSize: 11, marginTop: 4 }}>
                    {source.reason}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Per-Source Breakdown */}
      {sourceBreakdown.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <span className="label" style={{ marginBottom: 6, display: 'block' }}>Signal Breakdown</span>
          <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ color: '#94a3b8', borderBottom: '1px solid #1e293b' }}>
                <th style={{ textAlign: 'left', padding: '2px 4px' }}>Signal</th>
                <th style={{ textAlign: 'center', padding: '2px 4px' }}>Dir</th>
                <th style={{ textAlign: 'center', padding: '2px 4px' }}>Str</th>
                <th style={{ textAlign: 'center', padding: '2px 4px' }}>Conf</th>
                <th style={{ textAlign: 'right', padding: '2px 4px' }}>Wt</th>
              </tr>
            </thead>
            <tbody>
              {sourceBreakdown.map((src) => (
                <tr key={src.source} style={{ borderBottom: '1px solid #0f172a' }}>
                  <td style={{ padding: '2px 4px', color: '#e2e8f0' }}>
                    {SOURCE_LABELS[src.source] || src.source}
                  </td>
                  <td style={{
                    textAlign: 'center', padding: '2px 4px',
                    color: src.direction > 0 ? '#10b981' : src.direction < 0 ? '#ef4444' : '#94a3b8',
                  }}>
                    {formatSourceDirection(src.direction)}
                  </td>
                  <td style={{ textAlign: 'center', padding: '2px 4px', color: '#e2e8f0' }}>
                    {(src.strength * 100).toFixed(0)}%
                  </td>
                  <td style={{ textAlign: 'center', padding: '2px 4px', color: '#e2e8f0' }}>
                    {(src.confidence * 100).toFixed(0)}%
                  </td>
                  <td style={{ textAlign: 'right', padding: '2px 4px', color: '#94a3b8' }}>
                    {(src.weight * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
