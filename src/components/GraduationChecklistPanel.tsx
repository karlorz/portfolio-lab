import React from 'react';

interface GraduationCriterion {
  id: string;
  label: string;
  passed: boolean;
  value: string;
  threshold: string;
}

interface PaperTradingSummary {
  start_date: string;
  initial_capital: number;
  current_value: number;
  days_elapsed: number;
  days_required: number;
}

export interface GraduationChecklistData {
  criteria: GraduationCriterion[];
  paper_trading: PaperTradingSummary;
  readiness_pct: number;
  eligible: boolean;
}

interface GraduationChecklistPanelProps {
  data: GraduationChecklistData | null;
}

const CRITERIA_ICONS: Record<string, string> = {
  'days_traded': 'D',
  'sharpe': 'S',
  'max_dd': 'M',
  'kill_switch': 'K',
  'signal_health': 'H',
  'correlation': 'C',
  'ic': 'I',
};

export function GraduationChecklistPanel({ data }: GraduationChecklistPanelProps) {
  if (!data) {
    return (
      <div className="panel">
        <h3>Graduation Checklist</h3>
        <p className="muted">No graduation checklist data available</p>
      </div>
    );
  }

  const criteria = data.criteria || [];
  const paper = data.paper_trading;
  const readinessPct = data.readiness_pct;
  const passedCount = criteria.filter((c) => c.passed).length;
  const totalCount = criteria.length;

  const formatCurrency = (val: number) =>
    val.toLocaleString('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 0 });

  const totalPnL = paper.current_value - paper.initial_capital;
  const pnlPct = ((totalPnL / paper.initial_capital) * 100);
  const pnlColor = totalPnL >= 0 ? '#10b981' : '#ef4444';
  const daysColor = paper.days_elapsed >= paper.days_required ? '#10b981' : '#f59e0b';

  return (
    <div className="panel">
      <h3>Graduation Checklist</h3>

      {/* Readiness progress bar */}
      <div style={{ marginBottom: 12 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
          <span className="label">Readiness</span>
          <span style={{
            fontSize: 13, fontWeight: 600, color: readinessPct >= 100 ? '#10b981' : readinessPct >= 60 ? '#f59e0b' : '#ef4444',
          }}>
            {passedCount}/{totalCount} ({readinessPct.toFixed(0)}%)
          </span>
        </div>
        <div style={{ height: 8, background: '#1e293b', borderRadius: 4, overflow: 'hidden' }}>
          <div style={{
            width: `${readinessPct}%`, height: '100%', borderRadius: 4,
            background: readinessPct >= 100
              ? 'linear-gradient(90deg, #10b981, #34d399)'
              : readinessPct >= 60
                ? 'linear-gradient(90deg, #f59e0b, #fbbf24)'
                : 'linear-gradient(90deg, #ef4444, #f87171)',
            transition: 'width 0.4s ease',
          }} />
        </div>
      </div>

      {/* Status badge */}
      {data.eligible && (
        <div style={{
          marginBottom: 12, padding: '6px 10px', borderRadius: 4,
          background: 'rgba(16, 185, 129, 0.12)', border: '1px solid rgba(16, 185, 129, 0.25)',
          textAlign: 'center',
        }}>
          <span style={{ fontSize: 12, fontWeight: 600, color: '#10b981' }}>
            Ready for production graduation
          </span>
        </div>
      )}

      {/* Paper Trading P&L Summary */}
      <div style={{ marginBottom: 12 }}>
        <span className="label" style={{ display: 'block', marginBottom: 6 }}>Paper Trading P&amp;L</span>
        <div className="panel-grid">
          <div className="metric">
            <span className="label">Start Date</span>
            <span className="value" style={{ fontSize: 13 }}>{paper.start_date}</span>
          </div>
          <div className="metric">
            <span className="label">Starting Capital</span>
            <span className="value">{formatCurrency(paper.initial_capital)}</span>
          </div>
          <div className="metric">
            <span className="label">Current Value</span>
            <span className="value">{formatCurrency(paper.current_value)}</span>
          </div>
          <div className="metric">
            <span className="label">Total P&amp;L</span>
            <span className="value" style={{ color: pnlColor }}>
              {totalPnL >= 0 ? '+' : ''}{formatCurrency(totalPnL)}
            </span>
          </div>
          <div className="metric">
            <span className="label">P&amp;L %</span>
            <span className="value" style={{ color: pnlColor }}>
              {pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(1)}%
            </span>
          </div>
          <div className="metric">
            <span className="label">Days Elapsed</span>
            <span className="value" style={{ color: daysColor }}>
              {paper.days_elapsed}/{paper.days_required}
            </span>
          </div>
        </div>
      </div>

      {/* Graduation Criteria */}
      <span className="label" style={{ display: 'block', marginBottom: 6 }}>
        Graduation Criteria
        <span style={{ color: '#64748b', fontWeight: 400, marginLeft: 6 }}>
          ({passedCount}/{totalCount} passed)
        </span>
      </span>

      <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ color: '#94a3b8', borderBottom: '1px solid #1e293b' }}>
            <th style={{ textAlign: 'left', padding: '3px 6px', width: 20 }} />
            <th style={{ textAlign: 'left', padding: '3px 6px' }}>Criterion</th>
            <th style={{ textAlign: 'right', padding: '3px 6px' }}>Value</th>
            <th style={{ textAlign: 'right', padding: '3px 6px' }}>Threshold</th>
          </tr>
        </thead>
        <tbody>
          {criteria.map((c) => (
            <tr
              key={c.id}
              style={{
                borderBottom: '1px solid #0f172a',
                opacity: c.passed ? 1 : 0.6,
              }}
            >
              <td style={{ padding: '4px 6px', textAlign: 'center' }}>
                <span style={{
                  display: 'inline-block', width: 16, height: 16, borderRadius: '50%',
                  lineHeight: '16px', textAlign: 'center', fontSize: 9, fontWeight: 700,
                  background: c.passed ? '#10b981' : '#334155',
                  color: c.passed ? '#fff' : '#64748b',
                }}>
                  {c.passed ? '✓' : CRITERIA_ICONS[c.id] || '?'}
                </span>
              </td>
              <td style={{ padding: '4px 6px', color: '#e2e8f0' }}>{c.label}</td>
              <td style={{
                padding: '4px 6px', textAlign: 'right',
                color: c.passed ? '#10b981' : '#ef4444',
              }}>
                {c.value}
              </td>
              <td style={{ padding: '4px 6px', textAlign: 'right', color: '#64748b' }}>
                {c.threshold}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Footer */}
      <div style={{ marginTop: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: 10, color: '#475569' }}>
          Paper trading started {paper.start_date}
        </span>
        <span style={{
          fontSize: 11, fontWeight: 600,
          color: data.eligible ? '#10b981' : '#f59e0b',
        }}>
          {data.eligible ? 'ELIGIBLE' : 'IN PROGRESS'}
        </span>
      </div>
    </div>
  );
}
