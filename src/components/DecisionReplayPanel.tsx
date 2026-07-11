import React, { useEffect, useMemo, useState } from 'react';
import { fetchDecisionRegistry } from '../data/decisionRegistry';
import type {
  DecisionRecordRow,
  DecisionRegistryData,
  PromotionEvaluationRow,
  PromotionRowDisclosure,
} from '../schemas/decision_registry';

function formatTs(ts: string): string {
  try {
    return new Date(ts).toLocaleString();
  } catch {
    return ts;
  }
}

function formatLabel(value: string | undefined): string {
  if (!value) {
    return '—';
  }
  const label = value.replaceAll('_', ' ');
  return label.charAt(0).toUpperCase() + label.slice(1);
}

function weightTable(weights: Record<string, number> | undefined): React.ReactNode {
  if (!weights || Object.keys(weights).length === 0) {
    return <span className="muted">—</span>;
  }
  return (
    <ul className="decision-weight-list">
      {Object.entries(weights)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([sym, w]) => (
          <li key={sym}>
            <code>{sym}</code> {(w * 100).toFixed(1)}%
          </li>
        ))}
    </ul>
  );
}

function formatMetricValue(name: string, value: number): string {
  if (name.includes('weight') || name.includes('delta') || name.includes('drift')) {
    return `${(value * 100).toFixed(2)}%`;
  }
  return Number.isInteger(value) ? String(value) : value.toFixed(4);
}

function metricsTable(metrics: Record<string, number> | undefined): React.ReactNode {
  if (!metrics || Object.keys(metrics).length === 0) {
    return <span className="muted">—</span>;
  }
  return (
    <table className="positions-table compact">
      <tbody>
        {Object.entries(metrics)
          .sort(([a], [b]) => a.localeCompare(b))
          .map(([name, value]) => (
            <tr key={name}>
              <td>{name}</td>
              <td>{formatMetricValue(name, value)}</td>
            </tr>
          ))}
      </tbody>
    </table>
  );
}

function formatBenchmarkWindow(
  window: Record<string, string | number | null> | undefined,
): string {
  if (!window || Object.keys(window).length === 0) {
    return '—';
  }
  return Object.entries(window)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([key, value]) => `${key}: ${value ?? '—'}`)
    .join(' · ');
}

function promotionLabel(
  promo: PromotionEvaluationRow | undefined,
  disclosure: PromotionRowDisclosure | undefined,
): React.ReactNode {
  if (!promo) {
    if (disclosure?.state === 'not_evaluated') {
      return (
        <div className="decision-promotion-disclosure">
          <strong>Not evaluated</strong>
          <br />
          <span>Promotion evaluation not published</span>
          <br />
          <small>{disclosure.message}</small>
        </div>
      );
    }
    return '—';
  }

  const semanticDisclosure = promo.semantic_disclosure;
  if (semanticDisclosure?.state === 'governance_blocked') {
    const details = [`governance: ${semanticDisclosure.governance_status}`];
    if (semanticDisclosure.provenance_status) {
      details.push(`provenance: ${semanticDisclosure.provenance_status}`);
    }

    return (
      <div className="decision-promotion-disclosure">
        <strong>Governance blocked</strong>
        <br />
        <span>Canonical recommendation: {promo.recommended_status}</span>
        <br />
        <span>Metric gate: {promo.metric_gate_status ?? '—'}</span>
        <br />
        <small>{details.join(' · ')}</small>
      </div>
    );
  }

  if (semanticDisclosure?.state === 'conflict') {
    const details = [`governance: ${semanticDisclosure.governance_status}`];
    if (semanticDisclosure.provenance_status) {
      details.push(`provenance: ${semanticDisclosure.provenance_status}`);
    }

    return (
      <div className="decision-promotion-disclosure">
        <strong>Semantic conflict</strong>
        <br />
        <span>Metric recommendation: {promo.recommended_status}</span>
        <br />
        <small>{details.join(' · ')}</small>
      </div>
    );
  }

  return `${promo.recommended_status}${promo.pass ? ' ✓' : ''}`;
}

export interface DecisionReplayPanelProps {
  /** Pre-fetched data (optional — panel fetches when omitted). */
  initialData?: DecisionRegistryData | null;
}

export function DecisionReplayPanel({ initialData }: DecisionReplayPanelProps) {
  const [data, setData] = useState<DecisionRegistryData | null>(initialData ?? null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(initialData === undefined);
  const [selectedId, setSelectedId] = useState<string | null>(
    initialData?.recent_decisions[0]?.decision_id ?? null,
  );

  useEffect(() => {
    if (initialData !== undefined) {
      setData(initialData);
      setLoading(false);
      return;
    }
    let cancelled = false;
    (async () => {
      setLoading(true);
      const result = await fetchDecisionRegistry();
      if (cancelled) return;
      setData(result.data);
      setError(result.error);
      setLoading(false);
      if (result.data?.recent_decisions[0]) {
        setSelectedId(result.data.recent_decisions[0].decision_id);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [initialData]);

  useEffect(() => {
    if (!selectedId && data?.recent_decisions.length) {
      setSelectedId(data.recent_decisions[0].decision_id);
    }
  }, [data, selectedId]);

  const selected: DecisionRecordRow | null = useMemo(() => {
    if (!data || !selectedId) return null;
    return data.recent_decisions.find((d) => d.decision_id === selectedId) ?? null;
  }, [data, selectedId]);

  const replayForSelected = useMemo(() => {
    if (!data || !selectedId) return null;
    return data.replay_summaries.find((r) => r.decision_id === selectedId) ?? null;
  }, [data, selectedId]);

  if (loading) {
    return (
      <section className="decision-replay-panel analytics-panel-group">
        <h3>Decision Replay</h3>
        <p className="muted">Loading decision registry…</p>
      </section>
    );
  }

  if (!data) {
    return (
      <section className="decision-replay-panel analytics-panel-group">
        <h3>Decision Replay</h3>
        <p className="analytics-empty">
          No decision registry artifact yet.
          {error ? ` (${error})` : ' Run dashboard generation to publish decision_registry.json.'}
        </p>
      </section>
    );
  }

  return (
    <section className="decision-replay-panel analytics-panel-group">
      <header className="decision-replay-header">
        <h3>Decision Replay</h3>
        <span className="muted">
          Updated {formatTs(data.generated_at)} · {data.counts.decisions} decisions ·{' '}
          {data.counts.experiments} experiments
        </span>
      </header>

      <div className="dashboard-grid dashboard-grid-two">
        <div className="stats-section">
          <h4>Recent decisions</h4>
          {data.recent_decisions.length === 0 ? (
            <p className="analytics-empty">No decisions recorded yet.</p>
          ) : (
            <div className="labs-table-scroll">
              <table className="positions-table">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Action</th>
                    <th>Regime</th>
                    <th>Gates</th>
                  </tr>
                </thead>
                <tbody>
                  {data.recent_decisions.slice(0, 25).map((row) => (
                    <tr
                      key={row.decision_id}
                      className={row.decision_id === selectedId ? 'selected-row' : ''}
                      onClick={() => setSelectedId(row.decision_id)}
                      style={{ cursor: 'pointer' }}
                    >
                      <td>{formatTs(row.timestamp_utc)}</td>
                      <td>
                        <strong>{row.action}</strong>
                      </td>
                      <td>{row.regime ?? '—'}</td>
                      <td>{(row.gates_triggered ?? []).join(', ') || '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="stats-section">
          <h4>Replay detail</h4>
          {!selected ? (
            <p className="analytics-empty">Select a decision row.</p>
          ) : (
            <>
              <p>{replayForSelected?.replay?.summary ?? selected.reason ?? selected.action}</p>
              <dl className="decision-replay-dl">
                <dt>Run</dt>
                <dd>
                  <code>{selected.run_id}</code>
                </dd>
                <dt>Role</dt>
                <dd>{formatLabel(selected.decision_role)}</dd>
                <dt>Source</dt>
                <dd>{formatLabel(selected.decision_source)}</dd>
                <dt>Controller</dt>
                <dd>{selected.controller_id ?? '—'}</dd>
                <dt>Git</dt>
                <dd>{selected.git_sha ?? '—'}</dd>
                <dt>Confidence</dt>
                <dd>
                  {selected.regime_confidence != null
                    ? selected.regime_confidence.toFixed(2)
                    : '—'}
                </dd>
              </dl>
              {selected.decision_role === 'shadow' ? (
                <>
                  <h5>Shadow evidence</h5>
                  <dl className="decision-replay-dl">
                    <dt>Live decision</dt>
                    <dd>{selected.live_decision_id ?? '—'}</dd>
                    <dt>Baseline controller</dt>
                    <dd>{selected.baseline_controller_id ?? '—'}</dd>
                    <dt>Promotion review</dt>
                    <dd>{formatLabel(selected.promotion_review_status)}</dd>
                    <dt>Benchmark window</dt>
                    <dd>{formatBenchmarkWindow(selected.benchmark_window)}</dd>
                  </dl>
                  <h5>Divergence metrics</h5>
                  {metricsTable(selected.divergence_metrics)}
                </>
              ) : null}
              <h5>Current vs target weights</h5>
              <div className="dashboard-grid dashboard-grid-two">
                <div>
                  <small>Current</small>
                  {weightTable(selected.current_weights)}
                </div>
                <div>
                  <small>Target</small>
                  {weightTable(selected.target_weights)}
                </div>
              </div>
              {replayForSelected?.replay?.top_signal_weights?.length ? (
                <>
                  <h5>Top signal weights</h5>
                  <table className="positions-table compact">
                    <tbody>
                      {replayForSelected.replay.top_signal_weights.map((row) => (
                        <tr key={row.signal}>
                          <td>{row.signal}</td>
                          <td>{row.weight.toFixed(3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              ) : null}
            </>
          )}
        </div>
      </div>

      {data.recent_experiments.length > 0 && (
        <div className="stats-section">
          <h4>Recent experiments (registry)</h4>
          {data.promotion_coverage.disclosure === 'partial_promotion_evaluation_coverage' && (
            <p className="muted decision-promotion-coverage">
              {`Partial coverage: ${data.promotion_coverage.evaluated_count}/${data.promotion_coverage.recent_experiment_count} evaluated`}
            </p>
          )}
          <div className="labs-table-scroll">
            <table className="positions-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Status</th>
                  <th>Sharpe</th>
                  <th>Promotion</th>
                </tr>
              </thead>
              <tbody>
                {data.recent_experiments.slice(0, 15).map((exp) => {
                  const promo = data.promotion_evaluations.find(
                    (p) => p.experiment_id === exp.experiment_id,
                  );
                  const sharpe = exp.metrics?.sharpe;
                  return (
                    <tr key={exp.experiment_id}>
                      <td>
                        <strong>{exp.experiment_id}</strong>
                      </td>
                      <td>{exp.promotion_status ?? 'candidate'}</td>
                      <td>{sharpe != null ? sharpe.toFixed(2) : '—'}</td>
                      <td>{promotionLabel(promo, exp.promotion_disclosure)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  );
}
