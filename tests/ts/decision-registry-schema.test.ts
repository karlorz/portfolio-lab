import { describe, expect, it } from 'bun:test';
import {
  DECISION_REGISTRY_SCHEMA_VERSION,
  DecisionRegistrySchema,
  parseDecisionRegistryJson,
} from '../../src/schemas/decision_registry';

const projectionFreshness = {
  status: 'current',
  ledger_head: {
    decision_id: 'dec-1',
    timestamp_utc: '2026-07-01T12:00:00+00:00',
  },
  projection_head: {
    decision_id: 'dec-1',
    timestamp_utc: '2026-07-01T12:00:00+00:00',
  },
  lag_decision_count: 0,
  lag_seconds: 0,
} as const;

describe('decision_registry.json schema', () => {
  it('parses a minimal valid dashboard artifact', () => {
    const payload = {
      schema_version: DECISION_REGISTRY_SCHEMA_VERSION,
      generated_at: '2026-07-01T12:00:00+00:00',
      projection_freshness: projectionFreshness,
      recent_decisions: [
        {
          decision_id: 'dec-1',
          timestamp_utc: '2026-07-01T12:00:00+00:00',
          run_id: 'dashboard-run',
          decision_role: 'live_executed',
          decision_source: 'dashboard_cycle',
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
        recent_experiment_count: 0,
        evaluated_count: 0,
        unmatched_count: 0,
        unmatched_experiment_ids: [],
        disclosure: 'complete_promotion_evaluation_coverage',
      },
      counts: { decisions: 1, experiments: 0 },
    };

    const parsed = parseDecisionRegistryJson(payload);
    expect(parsed).not.toBeNull();
    expect(parsed?.recent_decisions[0].action).toBe('hold');
    expect(parsed?.projection_freshness.status).toBe('current');
    expect(DecisionRegistrySchema.safeParse(payload).success).toBe(true);
  });

  it('rejects wrong schema version', () => {
    expect(
      parseDecisionRegistryJson({
        schema_version: 'other/v0',
        generated_at: 'x',
        projection_freshness: projectionFreshness,
        recent_decisions: [],
        recent_experiments: [],
        replay_summaries: [],
        promotion_evaluations: [],
        counts: { decisions: 0, experiments: 0 },
      }),
    ).toBeNull();
  });

  it('rejects invalid decision role and source labels', () => {
    const basePayload = {
      schema_version: DECISION_REGISTRY_SCHEMA_VERSION,
      generated_at: '2026-07-01T12:00:00+00:00',
      projection_freshness: projectionFreshness,
      recent_decisions: [
        {
          decision_id: 'dec-1',
          timestamp_utc: '2026-07-01T12:00:00+00:00',
          run_id: 'dashboard-run',
          decision_role: 'paper_suggestion',
          decision_source: 'spreadsheet',
          action: 'hold',
        },
      ],
      recent_experiments: [],
      replay_summaries: [],
      promotion_evaluations: [],
      counts: { decisions: 1, experiments: 0 },
    };

    expect(DecisionRegistrySchema.safeParse(basePayload).success).toBe(false);
  });

  it('preserves the shadow evidence contract fields and summary', () => {
    const payload = {
      schema_version: DECISION_REGISTRY_SCHEMA_VERSION,
      generated_at: '2026-07-01T12:00:00+00:00',
      projection_freshness: projectionFreshness,
      recent_decisions: [
        {
          decision_id: 'shadow-1',
          timestamp_utc: '2026-07-01T12:05:00+00:00',
          run_id: 'dashboard-shadow-run',
          decision_role: 'shadow',
          decision_source: 'dashboard_cycle',
          controller_id: 'ensemble_voter_shadow',
          live_decision_id: 'live-1',
          baseline_controller_id: 'regime_allocation_live',
          benchmark_window: { label: 'dashboard_cycle', observations: 1 },
          divergence_metrics: { max_weight_delta: 0.04, action_mismatch: 1 },
          promotion_review_status: 'pending_review',
          action: 'rebalance',
        },
      ],
      recent_experiments: [],
      replay_summaries: [
        {
          decision_id: 'shadow-1',
          found: true,
          replay: {
            summary: 'Action rebalance',
            action: 'rebalance',
            decision_role: 'shadow',
            controller_id: 'ensemble_voter_shadow',
            live_decision_id: 'live-1',
            baseline_controller_id: 'regime_allocation_live',
            benchmark_window: { label: 'dashboard_cycle', observations: 1 },
            divergence_metrics: { max_weight_delta: 0.04, action_mismatch: 1 },
            promotion_review_status: 'pending_review',
          },
        },
      ],
      promotion_evaluations: [],
      promotion_coverage: {
        recent_experiment_count: 0,
        evaluated_count: 0,
        unmatched_count: 0,
        unmatched_experiment_ids: [],
        disclosure: 'complete_promotion_evaluation_coverage',
      },
      shadow_evidence: {
        shadow_decision_count: 1,
        linked_shadow_decision_count: 1,
        controllers: ['ensemble_voter_shadow'],
        promotion_review_status_counts: { pending_review: 1 },
        latest_shadow_decisions: [
          {
            decision_id: 'shadow-1',
            controller_id: 'ensemble_voter_shadow',
            live_decision_id: 'live-1',
            baseline_controller_id: 'regime_allocation_live',
            benchmark_window: { label: 'dashboard_cycle', observations: 1 },
            divergence_metrics: { max_weight_delta: 0.04, action_mismatch: 1 },
            promotion_review_status: 'pending_review',
          },
        ],
      },
      counts: { decisions: 1, experiments: 0 },
    };

    const parsed = parseDecisionRegistryJson(payload);

    expect(parsed?.recent_decisions[0].controller_id).toBe('ensemble_voter_shadow');
    expect(parsed?.recent_decisions[0].live_decision_id).toBe('live-1');
    expect(parsed?.recent_decisions[0].promotion_review_status).toBe('pending_review');
    expect(parsed?.replay_summaries[0].replay?.divergence_metrics?.max_weight_delta).toBe(
      0.04,
    );
    expect(parsed?.shadow_evidence.shadow_decision_count).toBe(1);
    expect(parsed?.shadow_evidence.controllers).toContain('ensemble_voter_shadow');
  });

  it('rejects invalid shadow promotion review status labels', () => {
    const payload = {
      schema_version: DECISION_REGISTRY_SCHEMA_VERSION,
      generated_at: '2026-07-01T12:00:00+00:00',
      projection_freshness: projectionFreshness,
      recent_decisions: [
        {
          decision_id: 'shadow-1',
          timestamp_utc: '2026-07-01T12:05:00+00:00',
          run_id: 'dashboard-shadow-run',
          decision_role: 'shadow',
          decision_source: 'dashboard_cycle',
          controller_id: 'ensemble_voter_shadow',
          live_decision_id: 'live-1',
          promotion_review_status: 'maybe_promote',
          action: 'rebalance',
        },
      ],
      recent_experiments: [],
      replay_summaries: [],
      promotion_evaluations: [],
      promotion_coverage: {
        recent_experiment_count: 0,
        evaluated_count: 0,
        unmatched_count: 0,
        unmatched_experiment_ids: [],
        disclosure: 'complete_promotion_evaluation_coverage',
      },
      counts: { decisions: 1, experiments: 0 },
    };

    expect(DecisionRegistrySchema.safeParse(payload).success).toBe(false);
  });

  it('preserves governance-blocked metric gate disclosure fields', () => {
    const payload = {
      schema_version: DECISION_REGISTRY_SCHEMA_VERSION,
      generated_at: '2026-07-01T12:00:00+00:00',
      projection_freshness: projectionFreshness,
      recent_decisions: [],
      recent_experiments: [
        {
          experiment_id: 'artifact:watch-row',
          timestamp_utc: '2026-07-01T12:00:00+00:00',
          name: 'artifact:watch-row',
          metrics: { sharpe: 1.05 },
          promotion_status: 'candidate',
          artifacts: { provenance_status: 'missing' },
        },
      ],
      replay_summaries: [],
      promotion_evaluations: [
        {
          experiment_id: 'artifact:watch-row',
          recommended_status: 'candidate',
          metric_gate_status: 'promoted',
          metric_gate_pass: true,
          pass: false,
          failures: ['provenance_missing'],
          semantic_disclosure: {
            state: 'governance_blocked',
            recommendation_type: 'metric_gate',
            governance_status: 'candidate',
            provenance_status: 'missing',
            metric_gate_status: 'promoted',
            reasons: ['provenance_missing'],
          },
        },
      ],
      promotion_coverage: {
        recent_experiment_count: 1,
        evaluated_count: 1,
        unmatched_count: 0,
        unmatched_experiment_ids: [],
        disclosure: 'complete_promotion_evaluation_coverage',
      },
      counts: { decisions: 0, experiments: 1 },
    };

    const parsed = parseDecisionRegistryJson(payload);

    expect(parsed?.promotion_evaluations[0].recommended_status).toBe('candidate');
    expect(parsed?.promotion_evaluations[0].metric_gate_status).toBe('promoted');
    expect(parsed?.promotion_evaluations[0].semantic_disclosure?.state).toBe(
      'governance_blocked',
    );
    expect(parsed?.promotion_evaluations[0].semantic_disclosure?.reasons).toContain(
      'provenance_missing',
    );
  });

  it('preserves unmatched promotion coverage disclosure fields', () => {
    const payload = {
      schema_version: DECISION_REGISTRY_SCHEMA_VERSION,
      generated_at: '2026-07-01T12:00:00+00:00',
      projection_freshness: projectionFreshness,
      recent_decisions: [],
      recent_experiments: [
        {
          experiment_id: 'experiment-10',
          timestamp_utc: '2026-07-01T12:49:00+00:00',
          name: 'experiment-10',
          metrics: { sharpe: 1.1 },
          promotion_status: 'candidate',
          promotion_disclosure: {
            state: 'not_evaluated',
            reason: 'promotion_evaluation_limit',
            message:
              'No promotion evaluation was published for this experiment in the current snapshot.',
          },
        },
      ],
      replay_summaries: [],
      promotion_evaluations: [],
      promotion_coverage: {
        recent_experiment_count: 12,
        evaluated_count: 10,
        unmatched_count: 2,
        unmatched_experiment_ids: ['experiment-10', 'experiment-11'],
        disclosure: 'partial_promotion_evaluation_coverage',
      },
      counts: { decisions: 0, experiments: 12 },
    };

    const parsed = parseDecisionRegistryJson(payload);

    expect(parsed?.promotion_coverage.unmatched_experiment_ids).toContain('experiment-10');
    expect(parsed?.recent_experiments[0].promotion_disclosure?.state).toBe('not_evaluated');
  });
});
