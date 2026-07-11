import { z } from 'zod';

export const DECISION_REGISTRY_SCHEMA_VERSION = 'decision-registry/v1';

const WeightMapSchema = z.record(z.string(), z.number());
const FloatMapSchema = z.record(z.string(), z.number());
const BenchmarkWindowSchema = z.record(z.string(), z.union([z.string(), z.number(), z.null()]));
const DecisionRoleSchema = z.enum(['live_executed', 'shadow']);
const DecisionSourceSchema = z.enum(['dashboard_cycle', 'evaluator_cycle', 'manual']);
const DecisionHeadSchema = z.object({
  decision_id: z.string(),
  timestamp_utc: z.string(),
});
const ProjectionFreshnessSchema = z.object({
  status: z.enum(['current', 'projection_lagged']),
  ledger_head: DecisionHeadSchema.nullable(),
  projection_head: DecisionHeadSchema.nullable(),
  lag_decision_count: z.number().int().nonnegative(),
  lag_seconds: z.number().nonnegative().nullable(),
});
const DecisionPromotionReviewStatusSchema = z.enum([
  'not_applicable',
  'pending_review',
  'eligible_for_promotion',
  'promoted',
  'rejected',
  'archived',
]);

export const DecisionRecordSchema = z
  .object({
    decision_id: z.string(),
    timestamp_utc: z.string(),
    git_sha: z.string().nullable().optional(),
    run_id: z.string(),
    decision_role: DecisionRoleSchema,
    decision_source: DecisionSourceSchema,
    strategy_version: z.string().optional(),
    portfolio_value: z.number().nullable().optional(),
    current_weights: WeightMapSchema.optional(),
    target_weights: WeightMapSchema.optional(),
    action: z.string(),
    reason: z.string().optional(),
    regime: z.string().nullable().optional(),
    regime_confidence: z.number().nullable().optional(),
    signal_votes: FloatMapSchema.optional(),
    signal_weights: FloatMapSchema.optional(),
    risk_metrics: FloatMapSchema.optional(),
    gates_triggered: z.array(z.string()).optional(),
    data_snapshot_hash: z.string().nullable().optional(),
    freeze_manifest_hash: z.string().nullable().optional(),
    controller_id: z.string().nullable().optional(),
    live_decision_id: z.string().nullable().optional(),
    baseline_controller_id: z.string().nullable().optional(),
    benchmark_window: BenchmarkWindowSchema.optional(),
    divergence_metrics: FloatMapSchema.optional(),
    promotion_review_status: DecisionPromotionReviewStatusSchema.optional(),
    extras: z.record(z.string(), z.unknown()).optional(),
  })
  .passthrough();

export const ExperimentRecordSchema = z
  .object({
    experiment_id: z.string(),
    timestamp_utc: z.string(),
    name: z.string(),
    hypothesis: z.string().optional(),
    git_sha: z.string().nullable().optional(),
    metrics: FloatMapSchema.optional(),
    benchmark_metrics: FloatMapSchema.optional(),
    promotion_status: z
      .enum(['candidate', 'shadow', 'promoted', 'rejected', 'archived'])
      .optional(),
    rejection_reason: z.string().nullable().optional(),
    artifacts: z.record(z.string(), z.unknown()).optional(),
    tags: z.array(z.string()).optional(),
  })
  .passthrough();

export const ReplaySummarySchema = z.object({
  decision_id: z.string(),
  found: z.boolean(),
  replay: z
    .object({
      summary: z.string().optional(),
      action: z.string().optional(),
      reason: z.string().optional(),
      regime: z.string().nullable().optional(),
      regime_confidence: z.number().nullable().optional(),
      gates_triggered: z.array(z.string()).optional(),
      top_signal_weights: z
        .array(z.object({ signal: z.string(), weight: z.number() }))
        .optional(),
      weight_delta: WeightMapSchema.optional(),
      risk_metrics: FloatMapSchema.optional(),
      git_sha: z.string().nullable().optional(),
      run_id: z.string().optional(),
      decision_role: DecisionRoleSchema.optional(),
      decision_source: DecisionSourceSchema.optional(),
      controller_id: z.string().nullable().optional(),
      live_decision_id: z.string().nullable().optional(),
      baseline_controller_id: z.string().nullable().optional(),
      benchmark_window: BenchmarkWindowSchema.optional(),
      divergence_metrics: FloatMapSchema.optional(),
      promotion_review_status: DecisionPromotionReviewStatusSchema.optional(),
    })
    .nullable()
    .optional(),
});

export const PromotionSemanticDisclosureSchema = z.object({
  state: z.enum(['conflict', 'governance_blocked']),
  recommendation_type: z.literal('metric_gate'),
  governance_status: z.string(),
  provenance_status: z.string().nullable().optional(),
  metric_gate_status: z.string().optional(),
  reasons: z.array(z.string()),
});

export const PromotionRowDisclosureSchema = z.object({
  state: z.literal('not_evaluated'),
  reason: z.literal('promotion_evaluation_limit'),
  message: z.string(),
});

export const PromotionEvaluationSchema = z.object({
  experiment_id: z.string(),
  recommended_status: z.string(),
  metric_gate_status: z.string().optional(),
  metric_gate_pass: z.boolean().optional(),
  pass: z.boolean(),
  failures: z.array(z.string()),
  thresholds: z.record(z.string(), z.union([z.number(), z.boolean()])).optional(),
  semantic_disclosure: PromotionSemanticDisclosureSchema.optional(),
});

export const PromotionCoverageSchema = z.object({
  recent_experiment_count: z.number().int().nonnegative(),
  evaluated_count: z.number().int().nonnegative(),
  unmatched_count: z.number().int().nonnegative(),
  unmatched_experiment_ids: z.array(z.string()),
  disclosure: z.enum([
    'complete_promotion_evaluation_coverage',
    'partial_promotion_evaluation_coverage',
  ]),
});

export const ExperimentRecordWithPromotionDisclosureSchema = ExperimentRecordSchema.extend({
  promotion_disclosure: PromotionRowDisclosureSchema.optional(),
});

const ShadowEvidenceDecisionSchema = z.object({
  decision_id: z.string(),
  controller_id: z.string().nullable().optional(),
  live_decision_id: z.string().nullable().optional(),
  baseline_controller_id: z.string().nullable().optional(),
  promotion_review_status: DecisionPromotionReviewStatusSchema,
  divergence_metrics: FloatMapSchema.optional(),
  benchmark_window: BenchmarkWindowSchema.optional(),
});

export const ShadowEvidenceSchema = z.object({
  shadow_decision_count: z.number().int().nonnegative(),
  linked_shadow_decision_count: z.number().int().nonnegative(),
  controllers: z.array(z.string()),
  promotion_review_status_counts: z.record(z.string(), z.number().int().nonnegative()),
  latest_shadow_decisions: z.array(ShadowEvidenceDecisionSchema),
});

const EMPTY_SHADOW_EVIDENCE: z.infer<typeof ShadowEvidenceSchema> = {
  shadow_decision_count: 0,
  linked_shadow_decision_count: 0,
  controllers: [],
  promotion_review_status_counts: {},
  latest_shadow_decisions: [],
};

export const DecisionRegistrySchema = z.object({
  schema_version: z.literal(DECISION_REGISTRY_SCHEMA_VERSION),
  generated_at: z.string(),
  projection_freshness: ProjectionFreshnessSchema,
  recent_decisions: z.array(DecisionRecordSchema),
  recent_experiments: z.array(ExperimentRecordWithPromotionDisclosureSchema),
  replay_summaries: z.array(ReplaySummarySchema),
  promotion_evaluations: z.array(PromotionEvaluationSchema),
  promotion_coverage: PromotionCoverageSchema,
  shadow_evidence: ShadowEvidenceSchema.default(EMPTY_SHADOW_EVIDENCE),
  counts: z.object({
    decisions: z.number().int().nonnegative(),
    experiments: z.number().int().nonnegative(),
  }),
});

export type DecisionRegistryData = z.infer<typeof DecisionRegistrySchema>;
export type DecisionRecordRow = z.infer<typeof DecisionRecordSchema>;
export type ExperimentRecordRow = z.infer<typeof ExperimentRecordWithPromotionDisclosureSchema>;
export type PromotionEvaluationRow = z.infer<typeof PromotionEvaluationSchema>;
export type PromotionRowDisclosure = z.infer<typeof PromotionRowDisclosureSchema>;

export function parseDecisionRegistryJson(payload: unknown): DecisionRegistryData | null {
  const result = DecisionRegistrySchema.safeParse(payload);
  return result.success ? result.data : null;
}
