import { describe, expect, it } from 'bun:test';
import {
  LABS_FIXTURE_NAMES,
  buildLabsFixture,
  labsFixturePath,
  loadLabsFixture,
} from './labs-fixtures';
import {
  LabsProvenanceSchema,
  LabsRegistrySchema,
  LabsReplaySchema,
  LabsScorecardSchema,
  LabsValidationReportSchema,
} from '../../src/schemas/labs';

describe('shared Labs fixture pack', () => {
  it('exposes the expected compact fixture file set', async () => {
    expect(new Set(LABS_FIXTURE_NAMES)).toEqual(new Set([
      'valid_registry',
      'valid_provenance',
      'valid_scorecard',
      'valid_replay_pass',
      'valid_replay_fail',
      'validation_report',
      'invalid_missing_metrics',
      'invalid_mixed_units',
      'stale_schema',
      'dirty_provenance',
      'valid_experiment_diff',
      'valid_registry_with_envelope',
      'valid_validation_with_envelope',
    ]));

    for (const name of LABS_FIXTURE_NAMES) {
      const file = Bun.file(labsFixturePath(name));
      expect(await file.exists()).toBe(true);
      expect(file.size).toBeLessThan(4096);
    }
  });

  it('validates shared fixture JSON through Labs dashboard schemas', () => {
    expect(LabsRegistrySchema.safeParse(loadLabsFixture('valid_registry')).success).toBe(true);
    expect(LabsProvenanceSchema.safeParse(loadLabsFixture('valid_provenance')).success).toBe(true);
    expect(LabsScorecardSchema.safeParse(loadLabsFixture('valid_scorecard')).success).toBe(true);
    expect(LabsReplaySchema.safeParse(loadLabsFixture('valid_replay_pass')).success).toBe(true);
    expect(LabsReplaySchema.safeParse(loadLabsFixture('valid_replay_fail')).success).toBe(true);
    expect(LabsValidationReportSchema.safeParse(loadLabsFixture('validation_report')).success).toBe(true);
  });

  it('fixture builders produce stale, malformed, dirty, and replay-drift examples', () => {
    const stale = buildLabsFixture('registry', 'stale_schema');
    const malformed = buildLabsFixture('registry', 'missing_metrics');
    const mixedUnits = buildLabsFixture('scorecard', 'mixed_units');
    const dirty = buildLabsFixture('provenance', 'dirty');
    const drift = buildLabsFixture('replay', 'drift_fail');

    expect(stale.schema_version).toBe('labs-registry/v0');
    expect(malformed.experiments[0].metrics).toBeUndefined();
    expect(mixedUnits.metrics.cagr_pct).toBe(240);
    expect(dirty.git.dirty).toBe(true);
    expect(drift.status).toBe('failed');
    expect(drift.metrics.max_abs_metric_delta).toBeGreaterThan(0);
  });
});
