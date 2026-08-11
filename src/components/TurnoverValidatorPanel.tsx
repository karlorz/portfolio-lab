import React from 'react';

interface TurnoverSignalDiagnostics {
  periods: number;
  mean: number;
  std: number;
  sign_flip_rate: number;
  mag_vol: number;
  turnover_penalty: number;
  stability_score: number;
  marginal_score: number;
}

interface SyntheticBaseline {
  metadata?: { source_type?: string };
  diagnostics?: Partial<TurnoverSignalDiagnostics>;
}

interface TurnoverValidatorData {
  schema_version?: string;
  signals: Record<string, TurnoverSignalDiagnostics>;
  synthetic_baselines?: Record<string, SyntheticBaseline>;
  generated_at?: string;
}

interface TurnoverValidatorPanelProps {
  data: TurnoverValidatorData | null;
}

type UnknownRecord = Record<string, unknown>;

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function finiteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function normalizeDiagnostics(value: unknown): TurnoverSignalDiagnostics | null {
  if (!isRecord(value)) return null;
  const periods = finiteNumber(value.periods);
  const mean = finiteNumber(value.mean);
  const std = finiteNumber(value.std);
  const sign_flip_rate = finiteNumber(value.sign_flip_rate);
  const mag_vol = finiteNumber(value.mag_vol);
  const turnover_penalty = finiteNumber(value.turnover_penalty);
  const stability_score = finiteNumber(value.stability_score);
  const marginal_score = finiteNumber(value.marginal_score);
  if (
    periods === null ||
    mean === null ||
    std === null ||
    sign_flip_rate === null ||
    mag_vol === null ||
    turnover_penalty === null ||
    stability_score === null ||
    marginal_score === null
  ) {
    return null;
  }
  return {
    periods,
    mean,
    std,
    sign_flip_rate,
    mag_vol,
    turnover_penalty,
    stability_score,
    marginal_score,
  };
}

function toPartialDiagnostics(value: unknown): Partial<TurnoverSignalDiagnostics> | undefined {
  if (!isRecord(value)) return undefined;
  const out: Partial<TurnoverSignalDiagnostics> = {};
  for (const key of [
    'periods',
    'mean',
    'std',
    'sign_flip_rate',
    'mag_vol',
    'turnover_penalty',
    'stability_score',
    'marginal_score',
  ] as const) {
    const n = finiteNumber(value[key]);
    if (n !== null) out[key] = n;
  }
  return Object.keys(out).length > 0 ? out : undefined;
}

export function normalizeTurnoverValidatorData(value: unknown): TurnoverValidatorData | null {
  if (!isRecord(value)) return null;

  // Live producer shape (turnover-validator/v1): signals → per-source diagnostics.
  // Null only when the payload is genuinely empty (no usable signal sources).
  const signals: Record<string, TurnoverSignalDiagnostics> = {};
  if (isRecord(value.signals)) {
    for (const [source, diagnostics] of Object.entries(value.signals)) {
      const normalized = normalizeDiagnostics(diagnostics);
      if (normalized) signals[source] = normalized;
    }
  }
  if (Object.keys(signals).length === 0) return null;

  let synthetic_baselines: Record<string, SyntheticBaseline> | undefined;
  if (isRecord(value.synthetic_baselines)) {
    synthetic_baselines = {};
    for (const [name, baseline] of Object.entries(value.synthetic_baselines)) {
      if (!isRecord(baseline)) continue;
      synthetic_baselines[name] = {
        metadata: isRecord(baseline.metadata)
          ? {
              source_type:
                typeof baseline.metadata.source_type === 'string'
                  ? baseline.metadata.source_type
                  : undefined,
            }
          : undefined,
        diagnostics: toPartialDiagnostics(baseline.diagnostics),
      };
    }
  }

  return {
    schema_version: typeof value.schema_version === 'string' ? value.schema_version : undefined,
    signals,
    synthetic_baselines,
    generated_at: typeof value.generated_at === 'string' ? value.generated_at : undefined,
  };
}

function stabilityColor(score: number): string {
  return score >= 0.7 ? '#10b981' : score >= 0.5 ? '#f59e0b' : '#ef4444';
}

function formatPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function StabilityDot({ score }: { score: number }) {
  return (
    <span
      style={{
        display: 'inline-block',
        width: 10,
        height: 10,
        borderRadius: '50%',
        background: stabilityColor(score),
        marginRight: 6,
        verticalAlign: 'middle',
      }}
    />
  );
}

export function TurnoverValidatorPanel({ data }: TurnoverValidatorPanelProps) {
  const panelData = normalizeTurnoverValidatorData(data);

  if (!panelData) {
    return (
      <div className="panel">
        <h3>Turnover Validator</h3>
        <p className="muted">No turnover validator data available</p>
      </div>
    );
  }

  const sources = Object.entries(panelData.signals).sort(([a], [b]) => a.localeCompare(b));
  const baselines = panelData.synthetic_baselines
    ? Object.entries(panelData.synthetic_baselines)
    : [];

  return (
    <div className="panel">
      <h3>
        Turnover Validator
        {panelData.schema_version && (
          <span style={{ fontSize: 10, color: '#64748b', fontWeight: 400, marginLeft: 8 }}>
            {panelData.schema_version}
          </span>
        )}
      </h3>
      {panelData.generated_at && (
        <p className="muted" style={{ marginTop: 0, fontSize: 10 }}>
          generated {new Date(panelData.generated_at).toLocaleString()}
        </p>
      )}

      {/* Per-source diagnostics */}
      <div style={{ marginTop: 8 }}>
        <span className="label" style={{ display: 'block', marginBottom: 6 }}>
          Signal-Source Diagnostics
          <span style={{ color: '#64748b', fontWeight: 400, marginLeft: 6 }}>
            ({sources.length} sources)
          </span>
        </span>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ color: '#94a3b8', borderBottom: '1px solid #1e293b' }}>
                <th style={{ textAlign: 'left', padding: '3px 6px' }}>Source</th>
                <th style={{ textAlign: 'right', padding: '3px 6px' }}>Periods</th>
                <th style={{ textAlign: 'right', padding: '3px 6px' }}>Mean</th>
                <th style={{ textAlign: 'right', padding: '3px 6px' }}>Std</th>
                <th style={{ textAlign: 'right', padding: '3px 6px' }}>Sign-Flip</th>
                <th style={{ textAlign: 'right', padding: '3px 6px' }}>Mag/Vol</th>
                <th style={{ textAlign: 'right', padding: '3px 6px' }}>Penalty</th>
                <th style={{ textAlign: 'right', padding: '3px 6px' }}>Stability</th>
                <th style={{ textAlign: 'right', padding: '3px 6px' }}>Marginal</th>
              </tr>
            </thead>
            <tbody>
              {sources.map(([source, diag]) => (
                <tr key={source} style={{ borderBottom: '1px solid #0f172a' }}>
                  <td style={{ padding: '4px 6px', color: '#e2e8f0', whiteSpace: 'nowrap' }}>
                    <StabilityDot score={diag.stability_score} />
                    {source}
                  </td>
                  <td style={{ padding: '4px 6px', textAlign: 'right', fontFamily: 'monospace', color: '#94a3b8' }}>
                    {diag.periods}
                  </td>
                  <td style={{ padding: '4px 6px', textAlign: 'right', fontFamily: 'monospace', color: '#94a3b8' }}>
                    {diag.mean.toFixed(3)}
                  </td>
                  <td style={{ padding: '4px 6px', textAlign: 'right', fontFamily: 'monospace', color: '#94a3b8' }}>
                    {diag.std.toFixed(3)}
                  </td>
                  <td style={{ padding: '4px 6px', textAlign: 'right', fontFamily: 'monospace', color: '#94a3b8' }}>
                    {formatPct(diag.sign_flip_rate)}
                  </td>
                  <td style={{ padding: '4px 6px', textAlign: 'right', fontFamily: 'monospace', color: '#94a3b8' }}>
                    {diag.mag_vol.toFixed(3)}
                  </td>
                  <td style={{ padding: '4px 6px', textAlign: 'right', fontFamily: 'monospace', color: '#94a3b8' }}>
                    {diag.turnover_penalty.toFixed(3)}
                  </td>
                  <td
                    style={{
                      padding: '4px 6px',
                      textAlign: 'right',
                      fontFamily: 'monospace',
                      fontWeight: 600,
                      color: stabilityColor(diag.stability_score),
                    }}
                  >
                    {diag.stability_score.toFixed(3)}
                  </td>
                  <td style={{ padding: '4px 6px', textAlign: 'right', fontFamily: 'monospace', color: '#94a3b8' }}>
                    {diag.marginal_score.toFixed(3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Synthetic baselines disclosed separately */}
      {baselines.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <span className="label" style={{ display: 'block', marginBottom: 6 }}>
            Synthetic Baselines
            <span style={{ color: '#64748b', fontWeight: 400, marginLeft: 6 }}>disclosed separately from live signals</span>
          </span>
          <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ color: '#94a3b8', borderBottom: '1px solid #1e293b' }}>
                <th style={{ textAlign: 'left', padding: '3px 6px' }}>Baseline</th>
                <th style={{ textAlign: 'left', padding: '3px 6px' }}>Type</th>
                <th style={{ textAlign: 'right', padding: '3px 6px' }}>Periods</th>
                <th style={{ textAlign: 'right', padding: '3px 6px' }}>Mean</th>
                <th style={{ textAlign: 'right', padding: '3px 6px' }}>Stability</th>
              </tr>
            </thead>
            <tbody>
              {baselines.map(([name, baseline]) => (
                <tr key={name} style={{ borderBottom: '1px solid #0f172a' }}>
                  <td style={{ padding: '4px 6px', color: '#e2e8f0' }}>{name}</td>
                  <td style={{ padding: '4px 6px', color: '#94a3b8' }}>
                    {baseline.metadata?.source_type ?? 'unknown'}
                  </td>
                  <td style={{ padding: '4px 6px', textAlign: 'right', fontFamily: 'monospace', color: '#94a3b8' }}>
                    {baseline.diagnostics?.periods ?? '—'}
                  </td>
                  <td style={{ padding: '4px 6px', textAlign: 'right', fontFamily: 'monospace', color: '#94a3b8' }}>
                    {baseline.diagnostics?.mean != null ? baseline.diagnostics.mean.toFixed(3) : '—'}
                  </td>
                  <td
                    style={{
                      padding: '4px 6px',
                      textAlign: 'right',
                      fontFamily: 'monospace',
                      fontWeight: 600,
                      color:
                        baseline.diagnostics?.stability_score != null
                          ? stabilityColor(baseline.diagnostics.stability_score)
                          : '#94a3b8',
                    }}
                  >
                    {baseline.diagnostics?.stability_score != null
                      ? baseline.diagnostics.stability_score.toFixed(3)
                      : '—'}
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

export type { TurnoverValidatorData, TurnoverSignalDiagnostics, SyntheticBaseline };
