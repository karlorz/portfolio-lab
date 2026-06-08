import { describe, expect, it } from 'bun:test';
import {
  LABS_DASHBOARD_ENDPOINTS,
  buildEmptyLabsDashboardData,
  fetchLabsDashboardData,
} from '../../src/data/labs';
import {
  LabsDashboardDataSchema,
  LabsProvenanceSchema,
  LabsRegistrySchema,
  LabsReplaySchema,
  LabsScorecardSchema,
  LabsValidationReportSchema,
  parseLabsJson,
} from '../../src/schemas/labs';

const generatedAt = '2026-06-08T12:00:00+00:00';

function validRegistry() {
  return {
    schema_version: 'labs-registry/v1',
    generated_at: generatedAt,
    experiments: [
      {
        experiment_id: 'gold-sweep',
        artifact_path: 'data/gold_allocation_sweep.json',
        status: 'validated',
        provenance_status: 'present',
        metrics: { sharpe: 0.95, cagr_pct: 10.4, max_drawdown_pct: -25.0 },
        baseline_deltas: { sharpe: 0.04, cagr_pct: 0.8 },
      },
    ],
  };
}

function validProvenance() {
  return {
    schema_version: 'experiment-manifest/v1',
    experiment_id: 'gold-sweep',
    generated_at: generatedAt,
    source_artifact_path: 'data/gold_allocation_sweep.json',
    command: 'python -m src.backtest.gold_allocation_sweep',
    module: 'src.backtest.gold_allocation_sweep',
    git: { commit: 'abc123', branch: 'main', dirty: false },
    config_snapshot: { min_gold_pct: 20 },
    environment: { PORTFOLIO_LAB_ENABLE_ML: '0' },
    input_file_hashes: { 'data/prices.json': 'a'.repeat(64) },
    freeze_manifest: {
      timestamp: generatedAt,
      config: {},
      file_hashes: {},
      file_count: 0,
    },
  };
}

function validScorecard() {
  return {
    schema_version: 'labs-scorecard/v1',
    experiment_id: 'gold-sweep',
    generated_at: generatedAt,
    status: 'promote',
    provenance_status: 'present',
    metrics: { sharpe: 0.95, cagr_pct: 10.4 },
    baseline_deltas: { sharpe: 0.04, max_drawdown_pct: 1.2 },
  };
}

function validReplay() {
  return {
    schema_version: 'labs-replay/v1',
    experiment_id: 'gold-sweep',
    generated_at: generatedAt,
    artifact_path: 'data/gold_allocation_sweep.json',
    status: 'passed',
    provenance_status: 'present',
    metrics: { rows_replayed: 109, max_abs_metric_delta: 0 },
    baseline_deltas: { sharpe: 0 },
  };
}

function validValidationReport() {
  return {
    results: [
      {
        path: 'data/gold_allocation_sweep.json',
        artifact_type: 'registry',
        schema_version: 'labs-registry/v1',
        valid: true,
        errors: [],
      },
    ],
  };
}

describe('Labs artifact schemas', () => {
  it('validates strict registry, provenance, scorecard, replay, and validation fixtures', () => {
    expect(LabsRegistrySchema.safeParse(validRegistry()).success).toBe(true);
    expect(LabsProvenanceSchema.safeParse(validProvenance()).success).toBe(true);
    expect(LabsScorecardSchema.safeParse(validScorecard()).success).toBe(true);
    expect(LabsReplaySchema.safeParse(validReplay()).success).toBe(true);
    expect(LabsValidationReportSchema.safeParse(validValidationReport()).success).toBe(true);
  });

  it('rejects malformed Labs fixture JSON with clear schema diagnostics', () => {
    const malformed = {
      ...validRegistry(),
      experiments: [
        {
          ...validRegistry().experiments[0],
          status: 'ship',
          metrics: { cagr_pct: '10.4' },
          extra: 'not allowed',
        },
      ],
    };

    const result = LabsRegistrySchema.safeParse(malformed);

    expect(result.success).toBe(false);
    if (!result.success) {
      const paths = result.error.issues.map((issue) => issue.path.join('.'));
      expect(paths).toContain('experiments.0.status');
      expect(paths).toContain('experiments.0.metrics.cagr_pct');
      expect(result.error.issues.some((issue) => issue.code === 'unrecognized_keys')).toBe(true);
    }
  });

  it('builds an empty disabled Labs state when artifacts are absent', () => {
    const empty = buildEmptyLabsDashboardData(['registry', 'scorecards']);

    expect(empty.available).toBe(false);
    expect(empty.registry).toBeNull();
    expect(empty.scorecards).toEqual([]);
    expect(empty.replays).toEqual([]);
    expect(empty.validation).toBeNull();
    expect(empty.missing).toEqual(['registry', 'scorecards']);
    expect(LabsDashboardDataSchema.safeParse(empty).success).toBe(true);
  });

  it('parses typed Labs JSON and returns diagnostics instead of passthrough data', () => {
    const parsed = parseLabsJson(validScorecard(), LabsScorecardSchema, 'scorecard');
    const malformed = parseLabsJson({ ...validScorecard(), status: 'ship' }, LabsScorecardSchema, 'scorecard');

    expect(parsed.data?.experiment_id).toBe('gold-sweep');
    expect(parsed.errors).toEqual([]);
    expect(malformed.data).toBeNull();
    expect(malformed.errors[0]).toContain('scorecard.status');
  });
});

describe('Labs dashboard data fetch helper', () => {
  it('returns a disabled empty state when Labs artifacts are missing', async () => {
    const fetcher = async () => new Response('', { status: 404 });

    const data = await fetchLabsDashboardData(fetcher);

    expect(data.available).toBe(false);
    expect(data.missing).toEqual([
      'registry',
      'scorecards',
      'replays',
      'validation',
    ]);
    expect(data.errors).toEqual([]);
  });

  it('fetches and validates the endpoint group without ad hoc dashboard state', async () => {
    const payloads: Record<string, unknown> = {
      [LABS_DASHBOARD_ENDPOINTS.registry]: validRegistry(),
      [LABS_DASHBOARD_ENDPOINTS.scorecards]: [validScorecard()],
      [LABS_DASHBOARD_ENDPOINTS.replays]: [validReplay()],
      [LABS_DASHBOARD_ENDPOINTS.validation]: validValidationReport(),
    };
    const fetcher = async (url: string) => new Response(JSON.stringify(payloads[url]), { status: 200 });

    const data = await fetchLabsDashboardData(fetcher);

    expect(data.available).toBe(true);
    expect(data.registry?.experiments[0].experiment_id).toBe('gold-sweep');
    expect(data.scorecards[0].status).toBe('promote');
    expect(data.replays[0].status).toBe('passed');
    expect(data.validation?.results[0].valid).toBe(true);
    expect(data.missing).toEqual([]);
    expect(data.errors).toEqual([]);
  });
});
