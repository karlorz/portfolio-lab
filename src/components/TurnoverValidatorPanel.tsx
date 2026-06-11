import React from 'react';

interface RebalanceEvent {
  date: string;
  turnover_pct: number;
  cost_bps: number;
  trigger: string; // "drift", "regime_change", "scheduled"
}

interface TurnoverValidatorData {
  current_turnover_pct: number;
  max_daily_turnover: number;
  max_monthly_turnover: number;
  max_annual_turnover: number;
  daily_budget_used: number;
  monthly_budget_used: number;
  annual_budget_used: number;
  recent_rebalances: RebalanceEvent[];
  cost_drag_bps: number;
}

interface TurnoverValidatorPanelProps {
  data: TurnoverValidatorData | null;
}

type UnknownRecord = Record<string, unknown>;

const TRIGGER_LABELS: Record<string, string> = {
  drift: 'Drift',
  regime_change: 'Regime Change',
  scheduled: 'Scheduled',
};

const TRIGGER_COLORS: Record<string, string> = {
  drift: '#f59e0b',
  regime_change: '#8b5cf6',
  scheduled: '#3b82f6',
};

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function finiteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function normalizeRebalanceEvents(value: unknown): RebalanceEvent[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter(isRecord)
    .map((event) => ({
      date: typeof event.date === 'string' ? event.date : 'unknown',
      turnover_pct: finiteNumber(event.turnover_pct) ?? 0,
      cost_bps: finiteNumber(event.cost_bps) ?? 0,
      trigger: typeof event.trigger === 'string' ? event.trigger : 'unknown',
    }));
}

export function normalizeTurnoverValidatorData(value: unknown): TurnoverValidatorData | null {
  if (!isRecord(value)) return null;

  const current_turnover_pct = finiteNumber(value.current_turnover_pct);
  const max_daily_turnover = finiteNumber(value.max_daily_turnover);
  const max_monthly_turnover = finiteNumber(value.max_monthly_turnover);
  const max_annual_turnover = finiteNumber(value.max_annual_turnover);
  const daily_budget_used = finiteNumber(value.daily_budget_used);
  const monthly_budget_used = finiteNumber(value.monthly_budget_used);
  const annual_budget_used = finiteNumber(value.annual_budget_used);
  const cost_drag_bps = finiteNumber(value.cost_drag_bps);

  if (
    current_turnover_pct === null ||
    max_daily_turnover === null ||
    max_monthly_turnover === null ||
    max_annual_turnover === null ||
    daily_budget_used === null ||
    monthly_budget_used === null ||
    annual_budget_used === null ||
    cost_drag_bps === null
  ) {
    return null;
  }

  return {
    current_turnover_pct,
    max_daily_turnover,
    max_monthly_turnover,
    max_annual_turnover,
    daily_budget_used,
    monthly_budget_used,
    annual_budget_used,
    recent_rebalances: normalizeRebalanceEvents(value.recent_rebalances),
    cost_drag_bps,
  };
}

function GaugeBar({ value, max, label }: { value: number; max: number; label: string }) {
  const pct = Math.min((value / max) * 100, 100);
  const color = value >= max ? '#ef4444' : value >= max * 0.9 ? '#f59e0b' : '#10b981';

  return (
    <div style={{ marginBottom: 4 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
        <span style={{ fontSize: 10, color: '#94a3b8' }}>{label}</span>
        <span style={{ fontSize: 11, fontFamily: 'monospace', color }}>
          {value.toFixed(2)}% / {max.toFixed(1)}%
        </span>
      </div>
      <div style={{ height: 8, background: '#1e293b', borderRadius: 4, overflow: 'hidden', position: 'relative' }}>
        <div style={{
          width: `${pct}%`, height: '100%', background: color, borderRadius: 4,
          transition: 'width 0.3s ease',
        }} />
      </div>
    </div>
  );
}

function BudgetBar({ used, label }: { used: number; label: string }) {
  const pct = Math.min(used * 100, 100);
  const remaining = Math.max((1 - used) * 100, 0);
  const color = used >= 1 ? '#ef4444' : used >= 0.85 ? '#f59e0b' : '#3b82f6';

  return (
    <div style={{ marginBottom: 4 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
        <span style={{ fontSize: 10, color: '#94a3b8' }}>{label}</span>
        <span style={{ fontSize: 11, fontFamily: 'monospace', color }}>
          {remaining.toFixed(0)}% remaining
        </span>
      </div>
      <div style={{ height: 8, background: '#1e293b', borderRadius: 4, overflow: 'hidden' }}>
        <div style={{
          width: `${pct}%`, height: '100%', background: color, borderRadius: 4,
          transition: 'width 0.3s ease',
        }} />
      </div>
    </div>
  );
}

function StatusDot({ passed }: { passed: boolean }) {
  return (
    <span style={{
      display: 'inline-block', width: 10, height: 10, borderRadius: '50%',
      background: passed ? '#10b981' : '#ef4444',
      marginRight: 6, verticalAlign: 'middle',
    }} />
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

  const dailyOk = panelData.daily_budget_used < 1;
  const monthlyOk = panelData.monthly_budget_used < 1;
  const annualOk = panelData.annual_budget_used < 1;
  const dailyPctFromMax = (panelData.current_turnover_pct / panelData.max_daily_turnover) * 100;
  const turnoverColor = panelData.current_turnover_pct >= panelData.max_daily_turnover
    ? '#ef4444' : panelData.current_turnover_pct >= panelData.max_daily_turnover * 0.9
    ? '#f59e0b' : '#10b981';
  const costColor = panelData.cost_drag_bps > 50
    ? '#ef4444' : panelData.cost_drag_bps > 25
    ? '#f59e0b' : '#10b981';

  return (
    <div className="panel">
      <h3>Turnover Validator</h3>

      {/* Summary metrics */}
      <div className="panel-grid">
        <div className="metric">
          <span className="label">Current Turnover</span>
          <span className="value" style={{ color: turnoverColor }}>
            {panelData.current_turnover_pct.toFixed(2)}%
          </span>
        </div>
        <div className="metric">
          <span className="label">Daily Max</span>
          <span className="value">{panelData.max_daily_turnover.toFixed(2)}%</span>
        </div>
        <div className="metric">
          <span className="label">Monthly Max</span>
          <span className="value">{panelData.max_monthly_turnover.toFixed(2)}%</span>
        </div>
        <div className="metric">
          <span className="label">Annual Max</span>
          <span className="value">{panelData.max_annual_turnover.toFixed(2)}%</span>
        </div>
      </div>

      {/* Current Turnover Gauge */}
      <div style={{ marginTop: 10 }}>
        <span className="label" style={{ display: 'block', marginBottom: 6 }}>
          Current Turnover <span style={{ color: '#64748b', fontWeight: 400 }}>vs Daily Max</span>
        </span>
        <div style={{
          position: 'relative', height: 14, background: '#1e293b', borderRadius: 7, overflow: 'hidden',
        }}>
          {/* Daily max threshold line */}
        <div style={{
          position: 'absolute', right: 0, top: -2, width: 2, height: 18,
          background: '#ef4444', opacity: 0.6, zIndex: 1,
        }} title={`Max: ${panelData.max_daily_turnover.toFixed(2)}%`} />
        <div style={{
          width: `${Math.min(dailyPctFromMax, 100)}%`, height: '100%',
          background: `linear-gradient(90deg, #10b981, ${turnoverColor})`,
            borderRadius: 7, transition: 'width 0.3s ease',
          }} />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 2 }}>
          <span style={{ fontSize: 9, color: '#64748b', fontFamily: 'monospace' }}>
            0%
        </span>
        <span style={{ fontSize: 9, color: turnoverColor, fontFamily: 'monospace', fontWeight: 600 }}>
          {panelData.current_turnover_pct.toFixed(2)}% used
        </span>
        <span style={{ fontSize: 9, color: '#64748b', fontFamily: 'monospace' }}>
          {panelData.max_daily_turnover.toFixed(1)}% max
        </span>
      </div>
      </div>

      {/* Turnover Budget */}
      <div style={{ marginTop: 12 }}>
        <span className="label" style={{ display: 'block', marginBottom: 6 }}>Turnover Budget</span>
        <BudgetBar used={panelData.daily_budget_used} label="Daily" />
        <BudgetBar used={panelData.monthly_budget_used} label="Monthly" />
        <BudgetBar used={panelData.annual_budget_used} label="Annual" />
      </div>

      {/* Constraint Status Table */}
      <div style={{ marginTop: 12 }}>
        <span className="label" style={{ display: 'block', marginBottom: 6 }}>Constraint Status</span>
        <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ color: '#94a3b8', borderBottom: '1px solid #1e293b' }}>
              <th style={{ textAlign: 'left', padding: '3px 6px' }}>Constraint</th>
              <th style={{ textAlign: 'right', padding: '3px 6px' }}>Limit</th>
              <th style={{ textAlign: 'right', padding: '3px 6px' }}>Used</th>
              <th style={{ textAlign: 'center', padding: '3px 6px' }}>Status</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: '1px solid #0f172a' }}>
              <td style={{ padding: '4px 6px', color: '#e2e8f0' }}>Max Daily</td>
              <td style={{ padding: '4px 6px', textAlign: 'right', fontFamily: 'monospace', color: '#94a3b8' }}>
                {panelData.max_daily_turnover.toFixed(2)}%
              </td>
              <td style={{
                padding: '4px 6px', textAlign: 'right', fontFamily: 'monospace',
                color: dailyOk ? '#10b981' : '#ef4444',
              }}>
                {(panelData.daily_budget_used * 100).toFixed(0)}%
              </td>
              <td style={{ padding: '4px 6px', textAlign: 'center' }}>
                <StatusDot passed={dailyOk} />
                <span style={{ color: dailyOk ? '#10b981' : '#ef4444', fontWeight: 600 }}>
                  {dailyOk ? 'PASS' : 'FAIL'}
                </span>
              </td>
            </tr>
            <tr style={{ borderBottom: '1px solid #0f172a' }}>
              <td style={{ padding: '4px 6px', color: '#e2e8f0' }}>Max Monthly</td>
              <td style={{ padding: '4px 6px', textAlign: 'right', fontFamily: 'monospace', color: '#94a3b8' }}>
                {panelData.max_monthly_turnover.toFixed(2)}%
              </td>
              <td style={{
                padding: '4px 6px', textAlign: 'right', fontFamily: 'monospace',
                color: monthlyOk ? '#10b981' : '#ef4444',
              }}>
                {(panelData.monthly_budget_used * 100).toFixed(0)}%
              </td>
              <td style={{ padding: '4px 6px', textAlign: 'center' }}>
                <StatusDot passed={monthlyOk} />
                <span style={{ color: monthlyOk ? '#10b981' : '#ef4444', fontWeight: 600 }}>
                  {monthlyOk ? 'PASS' : 'FAIL'}
                </span>
              </td>
            </tr>
            <tr>
              <td style={{ padding: '4px 6px', color: '#e2e8f0' }}>Max Annual</td>
              <td style={{ padding: '4px 6px', textAlign: 'right', fontFamily: 'monospace', color: '#94a3b8' }}>
                {panelData.max_annual_turnover.toFixed(2)}%
              </td>
              <td style={{
                padding: '4px 6px', textAlign: 'right', fontFamily: 'monospace',
                color: annualOk ? '#10b981' : '#ef4444',
              }}>
                {(panelData.annual_budget_used * 100).toFixed(0)}%
              </td>
              <td style={{ padding: '4px 6px', textAlign: 'center' }}>
                <StatusDot passed={annualOk} />
                <span style={{ color: annualOk ? '#10b981' : '#ef4444', fontWeight: 600 }}>
                  {annualOk ? 'PASS' : 'FAIL'}
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Recent Rebalancing Events */}
      {panelData.recent_rebalances.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <span className="label" style={{ display: 'block', marginBottom: 6 }}>
            Recent Rebalancing Events
            <span style={{ color: '#64748b', fontWeight: 400, marginLeft: 6 }}>
              (last {panelData.recent_rebalances.length})
            </span>
          </span>
          <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ color: '#94a3b8', borderBottom: '1px solid #1e293b' }}>
                <th style={{ textAlign: 'left', padding: '3px 6px' }}>Date</th>
                <th style={{ textAlign: 'right', padding: '3px 6px' }}>Turnover</th>
                <th style={{ textAlign: 'right', padding: '3px 6px' }}>Cost (bps)</th>
                <th style={{ textAlign: 'left', padding: '3px 6px' }}>Trigger</th>
              </tr>
            </thead>
            <tbody>
              {panelData.recent_rebalances.slice(0, 5).map((evt, i) => (
                <tr key={`${evt.date}-${i}`} style={{ borderBottom: i < Math.min(panelData.recent_rebalances.length, 5) - 1 ? '1px solid #0f172a' : 'none' }}>
                  <td style={{ padding: '4px 6px', color: '#e2e8f0', fontFamily: 'monospace' }}>
                    {evt.date}
                  </td>
                  <td style={{
                    padding: '4px 6px', textAlign: 'right', fontFamily: 'monospace',
                    color: evt.turnover_pct > 10 ? '#ef4444' : evt.turnover_pct > 5 ? '#f59e0b' : '#94a3b8',
                  }}>
                    {evt.turnover_pct.toFixed(2)}%
                  </td>
                  <td style={{
                    padding: '4px 6px', textAlign: 'right', fontFamily: 'monospace',
                    color: evt.cost_bps > 30 ? '#ef4444' : evt.cost_bps > 15 ? '#f59e0b' : '#94a3b8',
                  }}>
                    {evt.cost_bps.toFixed(1)}
                  </td>
                  <td style={{ padding: '4px 6px' }}>
                    <span style={{
                      fontSize: 10, fontWeight: 600, color: TRIGGER_COLORS[evt.trigger] || '#94a3b8',
                      background: `${TRIGGER_COLORS[evt.trigger] || '#94a3b8'}15`,
                      padding: '1px 6px', borderRadius: 3,
                    }}>
                      {TRIGGER_LABELS[evt.trigger] || evt.trigger}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* No rebalancing events placeholder */}
      {panelData.recent_rebalances.length === 0 && (
        <div style={{ marginTop: 12 }}>
          <span className="label" style={{ display: 'block', marginBottom: 4 }}>Recent Rebalancing Events</span>
          <p className="muted" style={{ margin: 0, fontSize: 11 }}>No rebalancing events recorded</p>
        </div>
      )}

      {/* Cost Impact */}
      <div style={{ marginTop: 12 }}>
        <span className="label" style={{ display: 'block', marginBottom: 6 }}>Cost Impact</span>
        <div style={{
          background: '#0f172a', borderRadius: 6, padding: '10px 12px',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <div>
            <span style={{ fontSize: 10, color: '#64748b', display: 'block', marginBottom: 2 }}>
              Turnover Cost Drag
            </span>
            <span style={{ fontSize: 12, color: '#94a3b8' }}>
              Estimated annualized return impact from transaction costs
            </span>
          </div>
          <div style={{ textAlign: 'right' }}>
            <span style={{
              fontSize: 20, fontWeight: 700, fontFamily: 'monospace', color: costColor,
            }}>
              {panelData.cost_drag_bps.toFixed(1)}
            </span>
            <span style={{ fontSize: 11, color: '#64748b', marginLeft: 2 }}>bps</span>
          </div>
        </div>
        {/* Severity indicator */}
        <div style={{
          marginTop: 6, display: 'flex', gap: 4, alignItems: 'center',
        }}>
          <div style={{
            flex: 1, height: 4, background: '#1e293b', borderRadius: 2, overflow: 'hidden', position: 'relative',
          }}>
            <div style={{
              width: `${Math.min((panelData.cost_drag_bps / 75) * 100, 100)}%`, height: '100%',
              background: costColor, borderRadius: 2, transition: 'width 0.3s ease',
            }} />
          </div>
          <span style={{
            fontSize: 9, fontWeight: 600, color: costColor, whiteSpace: 'nowrap',
          }}>
            {panelData.cost_drag_bps <= 15 ? 'Low' : panelData.cost_drag_bps <= 40 ? 'Moderate' : 'High'}
          </span>
        </div>
      </div>
    </div>
  );
}

export type { RebalanceEvent, TurnoverValidatorData };
