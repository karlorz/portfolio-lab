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
import { loadLabsFixture } from './labs-fixtures';

describe('Labs artifact schemas', () => {
  it('validates strict registry, provenance, scorecard, replay, and validation fixtures', () => {
    expect(LabsRegistrySchema.safeParse(loadLabsFixture('valid_registry')).success).toBe(true);
    expect(LabsProvenanceSchema.safeParse(loadLabsFixture('valid_provenance')).success).toBe(true);
    expect(LabsScorecardSchema.safeParse(loadLabsFixture('valid_scorecard')).success).toBe(true);
    expect(LabsReplaySchema.safeParse(loadLabsFixture('valid_replay_pass')).success).toBe(true);
    expect(LabsValidationReportSchema.safeParse(loadLabsFixture('validation_report')).success).toBe(true);
  });

  it('rejects malformed Labs fixture JSON with clear schema diagnostics', () => {
    const malformed = loadLabsFixture('invalid_missing_metrics');

    const result = LabsRegistrySchema.safeParse(malformed);

    expect(result.success).toBe(false);
    if (!result.success) {
      const paths = result.error.issues.map((issue) => issue.path.join('.'));
      expect(paths).toContain('experiments.0.metrics');
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
    const parsed = parseLabsJson(loadLabsFixture('valid_scorecard'), LabsScorecardSchema, 'scorecard');
    const malformed = parseLabsJson(loadLabsFixture('stale_schema'), LabsRegistrySchema, 'registry');

    expect(parsed.data?.experiment_id).toBe('gold-sweep');
    expect(parsed.errors).toEqual([]);
    expect(malformed.data).toBeNull();
    expect(malformed.errors[0]).toContain('registry.schema_version');
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
      [LABS_DASHBOARD_ENDPOINTS.registry]: loadLabsFixture('valid_registry'),
      [LABS_DASHBOARD_ENDPOINTS.scorecards]: [loadLabsFixture('valid_scorecard')],
      [LABS_DASHBOARD_ENDPOINTS.replays]: [loadLabsFixture('valid_replay_pass')],
      [LABS_DASHBOARD_ENDPOINTS.validation]: loadLabsFixture('validation_report'),
    };
    const fetcher = async (url: string) => new Response(JSON.stringify(payloads[url]), { status: 200 });

    const data = await fetchLabsDashboardData(fetcher);

    expect(data.available).toBe(true);
    expect(data.registry?.experiments[0].experiment_id).toBe('gold-sweep');
    expect(data.scorecards[0].status).toBe('promote');
    expect(data.replays[0].status).toBe('passed');
    expect(data.validation?.results[0].valid).toBe(false);
    expect(data.missing).toEqual([]);
    expect(data.errors).toEqual([]);
  });
});
