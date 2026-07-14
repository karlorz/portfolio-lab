import { describe, expect, it } from 'bun:test';
import { summarizeHealthOperations } from '../../src/components/healthOperations';
import type { HealthData } from '../../src/types/live';
import type { AllocationSurfaceRole } from '../../src/components/EnsembleVotingPanel';
import { formatAllocationSurfaceRoute } from '../../src/components/LiveDashboard';

const baseHealth = (): HealthData => ({
  system_status: 'critical',
  generated_at: '2026-07-12T12:00:00Z',
  cron_jobs: [
    { id: 'portfolio-lab-data', name: 'portfolio-lab-data', schedule: '5 * * * *', last_run: null, next_run: null, status: 'ok', state: 'scheduled' },
  ],
  scheduler_status: {
    status: 'ok',
    backends: {
      local: { backend: 'tasker', status: 'ok', source: 'data/cron_status.json', total_jobs: 15, failed_jobs: 0 },
    },
  },
  data_freshness: {
    SPY: { last_update: '2026-07-11', days_stale: 1, status: 'fresh' },
  },
  data_pipeline_slo: {
    schema_version: 'data-pipeline-slo/v1',
    status: 'critical',
    top_dimension: 'artifact',
    dimensions: {
      artifact: { status: 'critical', message: 'stale artifact' },
    },
  },
});

describe('kill-aware health operations summary', () => {
  it('prioritizes kill/halt over SLO primary cause in headerText', () => {
    const summary = summarizeHealthOperations({
      ...baseHealth(),
      kill_switch: {
        enabled: true,
        level: 'halt',
        status: 'critical',
        incident_id: '4d9e4f53-test',
        reason: 'max_drawdown',
      },
    });

    expect(summary.headline).toContain('kill/halt');
    expect(summary.headerText).toContain('kill/halt');
    expect(summary.topCauses[0]).toContain('kill/halt');
  });

  it('does not claim kill when kill_switch absent', () => {
    const summary = summarizeHealthOperations(baseHealth());
    expect(summary.headline).toContain('data pipeline');
    expect(summary.headerText).not.toContain('kill/halt');
  });
});

describe('formatAllocationSurfaceRoute under kill', () => {
  it('shows Blocked/halt when execution_blocked', () => {
    const role: AllocationSurfaceRole = {
      label: 'Target Allocation',
      role: 'execution_blocked',
      routed: true,
      routed_by: 'src.broker.order_router',
      description: 'blocked',
      execution_blocked: true,
      kill_switch_enabled: true,
      kill_switch_level: 'halt',
    };
    expect(formatAllocationSurfaceRoute(role)).toBe(
      'Blocked/halt (halt) via src.broker.order_router',
    );
  });

  it('shows Order-routed when not blocked', () => {
    const role: AllocationSurfaceRole = {
      label: 'Target Allocation',
      role: 'execution_routed',
      routed: true,
      routed_by: 'src.broker.order_router',
      description: 'routed',
    };
    expect(formatAllocationSurfaceRoute(role)).toBe(
      'Order-routed via src.broker.order_router',
    );
  });
});

describe('kill from alerts/broker when health omits kill_switch', () => {
  it('prioritizes kill/halt from alerts when health has no kill_switch', () => {
    const summary = summarizeHealthOperations(baseHealth(), {
      alerts: [
        {
          type: 'kill_switch',
          kill_switch_level: 'halt',
          title: 'PAPER Kill Switch Triggered',
          message: 'halted',
        },
      ],
    });
    expect(summary.headerText).toContain('kill/halt');
    expect(summary.topCauses[0]).toContain('kill/halt');
  });

  it('prioritizes kill/halt from broker when health has no kill_switch', () => {
    const summary = summarizeHealthOperations(baseHealth(), {
      broker: {
        kill_switch: true,
        kill_switch_level: 'halt',
        kill_switch_incident_id: '4d9e4f53-test',
      },
    });
    expect(summary.headerText).toContain('kill/halt');
    expect(summary.topCauses[0]).toContain('kill/halt');
  });
});
