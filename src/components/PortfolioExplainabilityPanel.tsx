import React from 'react';

// ── Types ──────────────────────────────────────────────────────────────

interface LatestDecision {
  timestamp: string;
  period: string;
  regime: string;
  action: string;
  confidence: number;
  reasoning: string;
  total_signals: number;
  consensus_direction: string;
  agreement_ratio: number;
  signals: Array<{
    source: string;
    value: number;
    weight: number;
    confidence: number;
  }>;
  top_drivers: Array<ContributionRow | string>;
  top_opposers: Array<ContributionRow | string>;
}

export interface ContributionRow {
  source: string;
  contribution: number | null;
  direction: string;
}

interface SignalDeepDive {
  source: string;
  display_name: string;
  category: string;
  total_observations: number;
  avg_value: number;
  avg_confidence: number;
  avg_weight: number;
  hit_rate: number;
}

export interface ExplainabilityData {
  timestamp: string;
  analysis_date: string;
  latest_decision: LatestDecision | null;
  recent_decisions: Array<any>;
  signal_deep_dives: Record<string, SignalDeepDive>;
  top_sources_today: string[];
  decision_quality: Record<string, any>;
  freshness?: {
    status?: string;
    reason?: string;
    stale_source_file?: string;
    stale_analysis_date?: string;
  };
}

interface Props {
  data: ExplainabilityData | null | undefined;
}

// ── Helpers ─────────────────────────────────────────────────────────────

function formatPct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function formatConfidence(v: number): string {
  return `${(v * 100).toFixed(0)}%`;
}

export function formatSourceLabel(source: unknown): string {
  if (typeof source !== 'string' || source.trim() === '') {
    return 'Unknown signal';
  }
  return source.replace(/_/g, ' ');
}

export function normalizeContributionRows(rows: unknown, _role: 'driver' | 'opposer'): ContributionRow[] {
  if (!Array.isArray(rows)) return [];
  return rows.map((row) => {
    if (typeof row === 'string') {
      return {
        source: row,
        contribution: null,
        direction: 'unknown',
      };
    }
    if (row && typeof row === 'object') {
      const candidate = row as Record<string, unknown>;
      return {
        source: typeof candidate.source === 'string' && candidate.source.trim() !== ''
          ? candidate.source
          : 'Unknown signal',
        contribution: typeof candidate.contribution === 'number' ? candidate.contribution : null,
        direction: typeof candidate.direction === 'string' ? candidate.direction : 'unknown',
      };
    }
    return {
      source: 'Unknown signal',
      contribution: null,
      direction: 'unknown',
    };
  });
}

export function getExplainabilityEmptyState(data: ExplainabilityData | null | undefined): {
  title: string;
  detail: string;
} {
  if (data?.freshness?.status === 'unavailable') {
    const staleSource = data.freshness.stale_source_file;
    const staleDate = data.freshness.stale_analysis_date;
    const historical = staleSource
      ? ` The last historical report was ${staleSource}${staleDate ? ` from ${staleDate}` : ''}.`
      : '';
    return {
      title: 'No current explainability available',
      detail: `${data.decision_quality?.reason || 'Current signals data could not produce explainability.'}${historical}`,
    };
  }

  return {
    title: 'No explainability data available',
    detail: 'Current signals data has not produced an explainability report yet.',
  };
}

function regimeColor(r: string): string {
  switch (r) {
    case 'normal': return '#10b981';
    case 'high_vol': return '#f59e0b';
    case 'crisis': return '#ef4444';
    case 'recovery': return '#3b82f6';
    default: return '#6b7280';
  }
}

function directionEmoji(d: string): string {
  switch (d) {
    case 'bullish': return '🟢';
    case 'bearish': return '🔴';
    default: return '⚪';
  }
}

function signalBarColor(v: number): string {
  if (v > 0.15) return '#10b981';
  if (v > 0) return '#34d399';
  if (v > -0.15) return '#f87171';
  return '#ef4444';
}

// ── Component ───────────────────────────────────────────────────────────

export function PortfolioExplainabilityPanel({ data }: Props) {
  if (!data?.latest_decision) {
    const emptyState = getExplainabilityEmptyState(data);
    return (
      <div className="explainability-panel empty">
        <h3>Portfolio Explainability</h3>
        <p>{emptyState.title}</p>
        <small>{emptyState.detail}</small>
      </div>
    );
  }

  const ld = data.latest_decision;
  const drivers = normalizeContributionRows(ld.top_drivers, 'driver');
  const opposers = normalizeContributionRows(ld.top_opposers, 'opposer');
  const signals = ld.signals ?? [];

  return (
    <div className="explainability-panel">
      <div className="ex-header">
        <h3>Portfolio Explainability</h3>
        <span className="ex-version">v8.07</span>
      </div>

      {/* ── Latest Decision ─────────────────────────────────────── */}
      <div className="ex-decision-section">
        <h4>Latest Decision</h4>
        <div className="ex-decision-card" style={{ borderLeftColor: regimeColor(ld.regime) }}>
          <div className="ex-decision-top">
            <span className={`ex-action ex-action-${ld.action}`}>
              {ld.action.replace(/_/g, ' ').toUpperCase()}
            </span>
            <span className="ex-direction">
              {directionEmoji(ld.consensus_direction)} {ld.consensus_direction.toUpperCase()}
            </span>
            <span className="ex-confidence" style={{ color: ld.confidence > 0.6 ? '#10b981' : '#f59e0b' }}>
              {formatConfidence(ld.confidence)} confidence
            </span>
          </div>
          <div className="ex-decision-meta">
            <span className="ex-regime-badge" style={{ backgroundColor: regimeColor(ld.regime) }}>
              {ld.regime.toUpperCase()}
            </span>
            <span className="ex-period">{ld.period}</span>
            <span className="ex-agreement">
              {ld.agreement_ratio > 0.8 ? 'HIGH' : ld.agreement_ratio > 0.5 ? 'MODERATE' : 'LOW'} agreement
              ({formatPct(ld.agreement_ratio)})
            </span>
          </div>
          <div className="ex-reasoning">{ld.reasoning.split('\n')[0]}</div>
        </div>
      </div>

      {/* ── Decision Provenance ─────────────────────────────────── */}
      <div className="ex-provenance-section">
        <h4>Decision Provenance</h4>
        <div className="ex-provenance-grid">
          <div className="ex-prov-item">
            <label>Total Signals</label>
            <span>{ld.total_signals}</span>
          </div>
          <div className="ex-prov-item">
            <label>Agreement</label>
            <span>{formatPct(ld.agreement_ratio)}</span>
          </div>
          <div className="ex-prov-item">
            <label>Direction</label>
            <span>{directionEmoji(ld.consensus_direction)} {ld.consensus_direction}</span>
          </div>
          <div className="ex-prov-item">
            <label>Analysis Date</label>
            <span>{data.analysis_date}</span>
          </div>
        </div>
      </div>

      {/* ── Signal Contributions ─────────────────────────────────── */}
      {signals.length > 0 && (
        <div className="ex-signals-section">
          <h4>Signal Contributions</h4>
          <div className="ex-signal-bars">
            {signals.map((s, i) => (
              <div key={i} className="ex-signal-row">
                <span className="ex-signal-name">{formatSourceLabel(s.source)}</span>
                <div className="ex-signal-bar-container">
                  <div
                    className="ex-signal-bar"
                    style={{
                      width: `${Math.abs(s.value) * 200}%`,
                      backgroundColor: signalBarColor(s.value),
                      marginLeft: s.value < 0 ? 'auto' : '0',
                    }}
                  />
                </div>
                <span className="ex-signal-val" style={{ color: signalBarColor(s.value) }}>
                  {s.value > 0 ? '+' : ''}{s.value.toFixed(2)}
                </span>
                <span className="ex-signal-weight">({formatPct(s.weight)})</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Top Drivers & Opposers ────────────────────────────────── */}
      <div className="ex-drivers-opposers">
        {drivers.length > 0 && (
          <div className="ex-drivers">
            <h4>Top Drivers</h4>
            {drivers.map((d, i) => (
              <div key={i} className="ex-driver-row">
                <span className="ex-driver-name">{formatSourceLabel(d.source)}</span>
                {d.contribution != null && (
                  <span className="ex-driver-contribution" style={{ color: '#10b981' }}>
                    +{d.contribution.toFixed(2)}
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
        {opposers.length > 0 && (
          <div className="ex-opposers">
            <h4>Top Opposers</h4>
            {opposers.map((d, i) => (
              <div key={i} className="ex-opposer-row">
                <span className="ex-opposer-name">{formatSourceLabel(d.source)}</span>
                {d.contribution != null && (
                  <span className="ex-opposer-contribution" style={{ color: '#ef4444' }}>
                    {d.contribution.toFixed(2)}
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
        {drivers.length === 0 && opposers.length === 0 && (
          <div className="ex-no-drivers">
            <small>Full consensus — no opposing signals</small>
          </div>
        )}
      </div>

      {/* ── Signal Deep Dives ────────────────────────────────────── */}
      {data.signal_deep_dives && Object.keys(data.signal_deep_dives).length > 0 && (
        <div className="ex-deepdives-section">
          <h4>Signal Deep Dives</h4>
          <div className="ex-deepdives-list">
            {Object.entries(data.signal_deep_dives).slice(0, 5).map(([key, sd]) => (
              <div key={key} className="ex-deepdive-row">
                <div className="ex-dd-header">
                  <span className="ex-dd-name">{sd.display_name || key}</span>
                  <span className="ex-dd-category">{sd.category}</span>
                </div>
                <div className="ex-dd-metrics">
                  <span>Avg: {sd.avg_value?.toFixed(2) ?? 'N/A'}</span>
                  <span>Hit: {sd.hit_rate != null ? formatPct(sd.hit_rate) : 'N/A'}</span>
                  <span>Obs: {sd.total_observations ?? 0}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Footer ───────────────────────────────────────────────────── */}
      {data.timestamp && (
        <div className="ex-footer">
          <small>Generated: {new Date(data.timestamp).toLocaleString()}</small>
        </div>
      )}
    </div>
  );
}
