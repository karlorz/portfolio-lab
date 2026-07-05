import { describe, expect, it } from 'bun:test';
import {
  DECISION_REGISTRY_SCHEMA_VERSION,
  DecisionRegistrySchema,
  parseDecisionRegistryJson,
} from '../../src/schemas/decision_registry';

describe('decision_registry.json schema', () => {
  it('parses a minimal valid dashboard artifact', () => {
    const payload = {
      schema_version: DECISION_REGISTRY_SCHEMA_VERSION,
      generated_at: '2026-07-01T12:00:00+00:00',
      recent_decisions: [
        {
          decision_id: 'dec-1',
          timestamp_utc: '2026-07-01T12:00:00+00:00',
          run_id: 'dashboard-run',
          action: 'hold',
          regime: 'NORMAL',
          gates_triggered: [],
        },
      ],
      recent_experiments: [],
      replay_summaries: [
        { decision_id: 'dec-1', found: true, replay: { summary: 'Action hold', action: 'hold' } },
      ],
      promotion_evaluations: [],
      promotion_coverage: {
        scope: 'recent_experiments',
        recent_experiment_count: 0,
        evaluated_experiment_count: 0,
        unmatched_experiment_count: 0,
        unmatched_experiment_ids: [],
        disclosure: 'complete_promotion_evaluation_coverage',
      },
      counts: { decisions: 1, experiments: 0 },
    };

    const parsed = parseDecisionRegistryJson(payload);
    expect(parsed).not.toBeNull();
    expect(parsed?.recent_decisions[0].action).toBe('hold');
    expect(DecisionRegistrySchema.safeParse(payload).success).toBe(true);
  });

  it('rejects wrong schema version', () => {
    expect(
      parseDecisionRegistryJson({
        schema_version: 'other/v0',
        generated_at: 'x',
        recent_decisions: [],
        recent_experiments: [],
        replay_summaries: [],
        promotion_evaluations: [],
        counts: { decisions: 0, experiments: 0 },
      }),
    ).toBeNull();
  });

  it('requires explicit promotion coverage metadata', () => {
    const payloadWithoutCoverage = {
      schema_version: DECISION_REGISTRY_SCHEMA_VERSION,
      generated_at: '2026-07-01T12:00:00+00:00',
      recent_decisions: [],
      recent_experiments: [],
      replay_summaries: [],
      promotion_evaluations: [],
      counts: { decisions: 0, experiments: 0 },
    };

    expect(DecisionRegistrySchema.safeParse(payloadWithoutCoverage).success).toBe(false);
  });
});
