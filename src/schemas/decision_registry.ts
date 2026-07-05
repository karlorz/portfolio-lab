import { z } from 'zod';

export const DECISION_REGISTRY_SCHEMA_VERSION = 'decision-registry/v1';

const WeightMapSchema = z.record(z.string(), z.number());
const FloatMapSchema = z.record(z.string(), z.number());

export const DecisionRecordSchema = z
  .object({
    decision_id: z.string(),
    timestamp_utc: z.string(),
    git_sha: z.string().nullable().optional(),
    run_id: z.string(),
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
    })
    .nullable()
    .optional(),
});

export const PromotionEvaluationSchema = z.object({
  experiment_id: z.string(),
  recommended_status: z.string(),
  pass: z.boolean(),
  failures: z.array(z.string()),
  thresholds: z.record(z.string(), z.union([z.number(), z.boolean()])).optional(),
});

export const PromotionCoverageSchema = z.object({
  scope: z.literal('recent_experiments'),
  recent_experiment_count: z.number().int().nonnegative(),
  evaluated_experiment_count: z.number().int().nonnegative(),
  unmatched_experiment_count: z.number().int().nonnegative(),
  unmatched_experiment_ids: z.array(z.string()),
  disclosure: z.enum([
    'complete_promotion_evaluation_coverage',
    'partial_promotion_evaluation_coverage',
  ]),
});

export const DecisionRegistrySchema = z.object({
  schema_version: z.literal(DECISION_REGISTRY_SCHEMA_VERSION),
  generated_at: z.string(),
  recent_decisions: z.array(DecisionRecordSchema),
  recent_experiments: z.array(ExperimentRecordSchema),
  replay_summaries: z.array(ReplaySummarySchema),
  promotion_evaluations: z.array(PromotionEvaluationSchema),
  promotion_coverage: PromotionCoverageSchema,
  counts: z.object({
    decisions: z.number().int().nonnegative(),
    experiments: z.number().int().nonnegative(),
  }),
});

export type DecisionRegistryData = z.infer<typeof DecisionRegistrySchema>;
export type DecisionRecordRow = z.infer<typeof DecisionRecordSchema>;
export type ExperimentRecordRow = z.infer<typeof ExperimentRecordSchema>;

export function parseDecisionRegistryJson(payload: unknown): DecisionRegistryData | null {
  const result = DecisionRegistrySchema.safeParse(payload);
  return result.success ? result.data : null;
}
