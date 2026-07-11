import React from 'react';

export interface MarlRuntimeStatusData {
  schema_version: 'marl-runtime-status/v1';
  available: boolean;
  timestamp: string | null;
  runtime: {
    version: string;
    device: string;
    agents_loaded: string[];
    signal_integrator_connected: boolean;
    checkpoint_loaded: boolean;
    inference_count: number;
    current_allocation: Record<string, number>;
    graph_metrics: Record<string, unknown>;
  };
  execution_role: {
    role: 'research_shadow_non_routed';
    routed: false;
    routed_by: null;
    live_authoritative: false;
    description: string;
  };
  error?: string;
}

interface MarlRuntimeStatusPanelProps {
  data: MarlRuntimeStatusData | null;
}

function formatRole(role: MarlRuntimeStatusData['execution_role']['role'] | undefined) {
  if (role === 'research_shadow_non_routed') return 'Research Shadow';
  return 'Unknown';
}

function formatAllocation(allocation: Record<string, number>) {
  const entries = Object.entries(allocation);
  if (entries.length === 0) return 'No allocation loaded';
  return entries
    .map(([symbol, weight]) => `${symbol} ${(weight * 100).toFixed(0)}%`)
    .join(' / ');
}

export function MarlRuntimeStatusPanel({ data }: MarlRuntimeStatusPanelProps) {
  if (!data) {
    return (
      <div className="panel">
        <h3>MARL Runtime Status</h3>
        <p className="muted">No MARL runtime status data available</p>
      </div>
    );
  }

  const runtime = data.runtime;

  return (
    <div className="panel">
      <h3>MARL Runtime Status</h3>

      <div className="panel-grid">
        <div className="metric">
          <span className="label">Role</span>
          <span className="value">{formatRole(data.execution_role.role)}</span>
        </div>
        <div className="metric">
          <span className="label">Routing</span>
          <span className="value">Not order-routed</span>
        </div>
        <div className="metric">
          <span className="label">Controller</span>
          <span className="value">{runtime.version}</span>
        </div>
        <div className="metric">
          <span className="label">Device</span>
          <span className="value">{runtime.device}</span>
        </div>
      </div>

      <p className="muted" style={{ marginTop: 10 }}>
        {data.execution_role.description}
      </p>
      <p className="muted" style={{ marginTop: 6 }}>
        Live orders still route through target_allocations.
      </p>

      {data.error && (
        <p className="warning" style={{ marginTop: 8 }}>
          Status collection error: {data.error}
        </p>
      )}

      <div style={{ marginTop: 10 }}>
        <span className="label" style={{ display: 'block', marginBottom: 6 }}>
          Controller Facts
        </span>
        <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
          <tbody>
            <tr style={{ borderBottom: '1px solid #0f172a' }}>
              <td style={{ padding: '2px 4px', color: '#94a3b8' }}>Agents</td>
              <td style={{ padding: '2px 4px', textAlign: 'right' }}>
                {runtime.agents_loaded.length > 0 ? runtime.agents_loaded.join(', ') : 'None'}
              </td>
            </tr>
            <tr style={{ borderBottom: '1px solid #0f172a' }}>
              <td style={{ padding: '2px 4px', color: '#94a3b8' }}>Signal Integrator</td>
              <td style={{ padding: '2px 4px', textAlign: 'right' }}>
                {runtime.signal_integrator_connected ? 'Connected' : 'Disconnected'}
              </td>
            </tr>
            <tr style={{ borderBottom: '1px solid #0f172a' }}>
              <td style={{ padding: '2px 4px', color: '#94a3b8' }}>Checkpoint</td>
              <td style={{ padding: '2px 4px', textAlign: 'right' }}>
                {runtime.checkpoint_loaded ? 'Loaded' : 'Not loaded'}
              </td>
            </tr>
            <tr style={{ borderBottom: '1px solid #0f172a' }}>
              <td style={{ padding: '2px 4px', color: '#94a3b8' }}>Inferences</td>
              <td style={{ padding: '2px 4px', textAlign: 'right' }}>
                {runtime.inference_count}
              </td>
            </tr>
            <tr>
              <td style={{ padding: '2px 4px', color: '#94a3b8' }}>Current Allocation</td>
              <td style={{ padding: '2px 4px', textAlign: 'right' }}>
                {formatAllocation(runtime.current_allocation)}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
