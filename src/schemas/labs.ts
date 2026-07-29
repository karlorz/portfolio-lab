import { z } from 'zod';

export const LABS_REGISTRY_SCHEMA_VERSION = 'labs-registry/v1';
export const LABS_SCORECARD_SCHEMA_VERSION = 'labs-scorecard/v1';
export const LABS_REPLAY_SCHEMA_VERSION = 'labs-replay/v1';
export const LABS_PROVENANCE_SCHEMA_VERSION = 'experiment-manifest/v1';
export const LABS_VALIDATION_SCHEMA_VERSION = 'labs-validation/v1';
export const LABS_EXPERIMENT_DIFF_SCHEMA_VERSION = 'experiment-diff/v1';

const MetricMapSchema = z.record(z.string(), z.number());
const StringMapSchema = z.record(z.string(), z.string());
const UnknownMapSchema = z.record(z.string(), z.unknown());
const NonNegativeIntegerSchema = z.number().int().nonnegative();

export const LabsStatusSchema = z.enum([
  'candidate',
  'validated',
  'warning',
  'rejected',
  'archived',
]);

export const LabsScorecardStatusSchema = z.enum(['promote', 'watch', 'reject']);
export const LabsReplayStatusSchema = z.enum(['passed', 'failed', 'warning']);
export const LabsGovernanceStateSchema = z.enum(['clear', 'governance_blocked', 'rejected']);
export const LabsPromotionGovernanceSchema = z.object({
  recommended_status: z.string(),
  pass: z.boolean(),
  failures: z.array(z.string()),
  metric_gate_status: z.string(),
  metric_gate_pass: z.boolean(),
  governance_status: z.string(),
  provenance_status: z.string(),
}).strict();
export const LabsReplayFailureReasonSchema = z.enum([
  'safety_skip',
  'timeout',
  'validation_failure',
  'command_failure',
  'unexpected_error',
]);

export const LabsProvenanceStatusSchema = z.enum([
  'present',
  'embedded',
  'sidecar',
  'missing',
  'stale',
  'malformed',
  'unknown',
]);

export const LabsRegistryRowSchema = z.object({
  experiment_id: z.string(),
  artifact_path: z.string(),
  status: LabsStatusSchema,
  provenance_status: LabsProvenanceStatusSchema,
  metrics: MetricMapSchema,
  baseline_deltas: MetricMapSchema,
  governance_state: LabsGovernanceStateSchema.optional(),
  governance_reasons: z.array(z.string()).optional(),
  promotion_governance: LabsPromotionGovernanceSchema.optional(),
}).strict();

export const LabsRegistryWarningSchema = z.object({
  artifact_path: z.string(),
  error: z.string(),
}).strict();

export const LabsRegistrySchema = z.object({
  schema_version: z.literal(LABS_REGISTRY_SCHEMA_VERSION),
  generated_at: z.string(),
  experiments: z.array(LabsRegistryRowSchema),
  sources: z.array(z.string()).optional(),
  warnings: z.array(LabsRegistryWarningSchema).optional(),
}).strict();

export const LabsScorecardPolicyThresholdsSchema = z.object({
  min_promote_sharpe: z.number().nonnegative(),
  min_promote_sharpe_delta: z.number().nonnegative(),
  min_promote_dsr: z.number().nonnegative(),
  min_promote_wfe: z.number().nonnegative(),
}).strict();

export const LabsScorecardPolicySchema = z.object({
  version: z.string(),
  thresholds: LabsScorecardPolicyThresholdsSchema,
}).strict();

export const LabsProvenanceSchema = z.object({
  schema_version: z.literal(LABS_PROVENANCE_SCHEMA_VERSION),
  experiment_id: z.string(),
  generated_at: z.string(),
  source_artifact_path: z.string(),
  command: z.nullable(z.string()).optional(),
  module: z.nullable(z.string()).optional(),
  git: UnknownMapSchema,
  config_snapshot: UnknownMapSchema,
  environment: StringMapSchema,
  input_file_hashes: z.record(z.string(), z.nullable(z.string())),
  freeze_manifest: z.object({
    timestamp: z.nullable(z.string()).optional(),
    config: UnknownMapSchema,
    file_hashes: z.record(z.string(), z.string()),
    file_count: z.number(),
  }).strict(),
}).strict();

export const LabsScorecardSchema = z.object({
  schema_version: z.literal(LABS_SCORECARD_SCHEMA_VERSION),
  experiment_id: z.string(),
  generated_at: z.string(),
  status: LabsScorecardStatusSchema,
  provenance_status: LabsProvenanceStatusSchema,
  metrics: MetricMapSchema,
  baseline_deltas: MetricMapSchema,
  policy: LabsScorecardPolicySchema.optional(),
  governance_state: LabsGovernanceStateSchema.optional(),
  governance_reasons: z.array(z.string()).optional(),
  promotion_governance: LabsPromotionGovernanceSchema.optional(),
}).strict();

export const LabsReplaySchema = z.object({
  schema_version: z.literal(LABS_REPLAY_SCHEMA_VERSION),
  experiment_id: z.string(),
  generated_at: z.string(),
  artifact_path: z.string(),
  status: LabsReplayStatusSchema,
  provenance_status: LabsProvenanceStatusSchema,
  passed: z.boolean().optional(),
  command: z.nullable(z.string()).optional(),
  duration_seconds: z.number().optional(),
  metric_deltas: MetricMapSchema.optional(),
  metrics: MetricMapSchema,
  baseline_deltas: MetricMapSchema,
  failure_reason: LabsReplayFailureReasonSchema.optional(),
  error_type: z.string().optional(),
  error_message: z.string().optional(),
}).strict();

export const LabsValidationResultSchema = z.object({
  path: z.nullable(z.string()),
  artifact_type: z.enum(['registry', 'provenance', 'scorecard', 'replay', 'unknown']),
  schema_version: z.nullable(z.string()),
  valid: z.boolean(),
  errors: z.array(z.string()),
  omitted_error_count: NonNegativeIntegerSchema.optional(),
  experiment_id: z.string().optional(),
  artifact_path: z.string().optional(),
}).strict();

export const LabsValidationTruncationSchema = z.object({
  max_results: NonNegativeIntegerSchema,
  max_errors_per_result: NonNegativeIntegerSchema,
  total_result_count: NonNegativeIntegerSchema,
  returned_result_count: NonNegativeIntegerSchema,
  omitted_result_count: NonNegativeIntegerSchema,
  omitted_error_count: NonNegativeIntegerSchema,
}).strict();

export const LabsValidationReportSchema = z.object({
  schema_version: z.literal(LABS_VALIDATION_SCHEMA_VERSION),
  generated_at: z.string(),
  results: z.array(LabsValidationResultSchema),
  truncation: LabsValidationTruncationSchema.optional(),
}).strict();

export const LabsExperimentDiffSideSchema = z.object({
  label: z.string(),
  experiment_id: z.nullable(z.string()),
  artifact_path: z.nullable(z.string()),
  artifact_type: z.string(),
}).strict();

export const LabsExperimentMetricDeltaSchema = z.object({
  left: z.number(),
  right: z.number(),
  delta: z.number(),
}).strict();

export const LabsExperimentMissingMetricSchema = z.object({
  metric: z.string(),
  missing_from: z.array(z.enum(['left', 'right'])),
}).strict();

export const LabsExperimentConfigDiffSchema = z.object({
  left: z.unknown(),
  right: z.unknown(),
}).strict();

export const LabsExperimentProvenanceDiffSchema = z.object({
  left: z.string(),
  right: z.string(),
  changed: z.boolean(),
}).strict();

export const LabsExperimentDiffSchema = z.object({
  schema_version: z.literal(LABS_EXPERIMENT_DIFF_SCHEMA_VERSION),
  generated_at: z.string(),
  left: LabsExperimentDiffSideSchema,
  right: LabsExperimentDiffSideSchema,
  metric_deltas: z.record(z.string(), LabsExperimentMetricDeltaSchema),
  missing_metrics: z.array(LabsExperimentMissingMetricSchema),
  config_diffs: z.record(z.string(), LabsExperimentConfigDiffSchema),
  provenance: LabsExperimentProvenanceDiffSchema,
}).strict();

export const LabsEndpointKeySchema = z.enum(['registry', 'scorecards', 'replays', 'validation']);
export const LabsEndpointRenderStrategySchema = z.enum(['direct', 'summarize', 'paginate', 'missing']);

export const LabsPaginationPageSchema = z.object({
  page: z.number().int().positive(),
  path: z.string(),
  row_count: NonNegativeIntegerSchema.optional(),
}).strict();

export const LabsPaginationSchema = z.object({
  total_rows: NonNegativeIntegerSchema,
  page_size: z.number().int().positive(),
  page_count: NonNegativeIntegerSchema.optional(),
  pages: z.array(LabsPaginationPageSchema),
}).strict();

export const PublicDataSizeBudgetSchema = z.object({
  render_strategy: LabsEndpointRenderStrategySchema,
  status: z.string().optional(),
  size_bytes: z.nullable(z.number()).optional(),
  row_count: z.nullable(z.number()).optional(),
  requires_downsampling: z.boolean().optional(),
  requires_pagination: z.boolean().optional(),
}).passthrough();

export const PublicDataQualityIssueCountsSchema = z.object({
  duplicate_dates: NonNegativeIntegerSchema,
  empty_symbols: NonNegativeIntegerSchema,
  extreme_returns: NonNegativeIntegerSchema,
  internal_gaps: NonNegativeIntegerSchema,
  invalid_dates: NonNegativeIntegerSchema,
  invalid_prices: NonNegativeIntegerSchema,
  missing_required_keys: NonNegativeIntegerSchema,
  non_monotonic_rows: NonNegativeIntegerSchema,
  non_object_records: NonNegativeIntegerSchema,
  split_like_returns: NonNegativeIntegerSchema,
  stale_latest_dates: NonNegativeIntegerSchema,
  stale_latest_dates_within_tolerance: NonNegativeIntegerSchema.optional(),
  total: NonNegativeIntegerSchema,
}).strict();

export const PublicDataQualitySummarySchema = z.object({
  artifact: z.string(),
  schema_version: z.string().optional(),
  generated_at: z.string().optional(),
  status: z.enum(['ok', 'warn', 'fail', 'unavailable']),
  issue_counts: PublicDataQualityIssueCountsSchema.optional(),
}).strict();

export const PublicDataSourceMetadataSchema = z.object({
  provider: z.string().optional(),
  feed: z.string().optional(),
  source_mode: z.enum(['live', 'last_good', 'cached', 'stale_cached', 'synthetic']).optional(),
  status: z.enum(['success', 'degraded', 'failed', 'skipped']).optional(),
  fetched_at: z.nullable(z.string()).optional(),
  latest_observation: z.nullable(z.string()).optional(),
  row_count: z.nullable(z.number()).optional(),
  failure_reason: z.nullable(z.string()).optional(),
  fallback_reason: z.nullable(z.string()).optional(),
  data_quality: PublicDataQualitySummarySchema.optional(),
}).passthrough();

export const PublicDataIndexEntrySchema = z.object({
  filename: z.string(),
  path: z.string(),
  category: z.string(),
  schema_version: z.string(),
  status: z.enum(['present', 'missing']),
  validation_status: z.enum(['valid', 'invalid', 'missing', 'not_applicable']),
  validation_errors: z.array(z.string()),
  size_bytes: z.nullable(z.number()),
  size_budget: PublicDataSizeBudgetSchema,
  pagination: LabsPaginationSchema.optional(),
  source_manifest_path: z.string().optional(),
  source_metadata: PublicDataSourceMetadataSchema.optional(),
  sha256: z.nullable(z.string()).optional(),
  generated_at: z.nullable(z.string()).optional(),
}).passthrough();

export const PublicDataIndexSourceManifestSchema = z.object({
  path: z.string(),
  schema_version: z.string(),
  generated_at: z.string(),
  sha256: z.string().regex(/^[a-f0-9]{64}$/),
}).strict();

export const PublicDataIndexSchema = z.object({
  schema_version: z.string(),
  entries: z.array(PublicDataIndexEntrySchema),
  generated_at: z.string().optional(),
  source_manifest: PublicDataIndexSourceManifestSchema.optional(),
}).passthrough();

export const LabsEndpointStatusSchema = z.object({
  endpoint: LabsEndpointKeySchema,
  filename: z.string(),
  path: z.string(),
  status: z.enum(['present', 'missing']),
  validation_status: z.enum(['valid', 'invalid', 'missing', 'not_applicable']),
  validation_errors: z.array(z.string()),
  render_strategy: LabsEndpointRenderStrategySchema,
  size_bytes: z.nullable(z.number()),
  row_count: z.nullable(NonNegativeIntegerSchema).optional(),
  requires_downsampling: z.boolean().optional(),
  requires_pagination: z.boolean().optional(),
  summary_limited: z.boolean().optional(),
  size_budget_status: z.string().optional(),
  pagination: LabsPaginationSchema.optional(),
  selected_page: z.number().int().positive().optional(),
  generated_at: z.nullable(z.string()).optional(),
}).strict();

export const LabsDashboardDataSchema = z.object({
  available: z.boolean(),
  registry: z.nullable(LabsRegistrySchema),
  scorecards: z.array(LabsScorecardSchema),
  replays: z.array(LabsReplaySchema),
  validation: z.nullable(LabsValidationReportSchema),
  diffs: z.array(LabsExperimentDiffSchema).optional(),
  missing: z.array(LabsEndpointKeySchema),
  errors: z.array(z.string()),
  endpoint_status: z.array(LabsEndpointStatusSchema).optional(),
}).strict();

export interface LabsRegistryRow extends z.infer<typeof LabsRegistryRowSchema> {}
export interface LabsRegistryWarning extends z.infer<typeof LabsRegistryWarningSchema> {}
export interface LabsRegistryData extends z.infer<typeof LabsRegistrySchema> {}
export interface LabsProvenanceData extends z.infer<typeof LabsProvenanceSchema> {}
export interface LabsScorecardData extends z.infer<typeof LabsScorecardSchema> {}
export interface LabsScorecardPolicyThresholds extends z.infer<typeof LabsScorecardPolicyThresholdsSchema> {}
export interface LabsScorecardPolicy extends z.infer<typeof LabsScorecardPolicySchema> {}
export interface LabsReplayData extends z.infer<typeof LabsReplaySchema> {}
export interface LabsValidationResult extends z.infer<typeof LabsValidationResultSchema> {}
export interface LabsValidationTruncation extends z.infer<typeof LabsValidationTruncationSchema> {}
export interface LabsValidationReport extends z.infer<typeof LabsValidationReportSchema> {}
export interface LabsExperimentDiffSide extends z.infer<typeof LabsExperimentDiffSideSchema> {}
export interface LabsExperimentMetricDelta extends z.infer<typeof LabsExperimentMetricDeltaSchema> {}
export interface LabsExperimentMissingMetric extends z.infer<typeof LabsExperimentMissingMetricSchema> {}
export interface LabsExperimentConfigDiff extends z.infer<typeof LabsExperimentConfigDiffSchema> {}
export interface LabsExperimentProvenanceDiff extends z.infer<typeof LabsExperimentProvenanceDiffSchema> {}
export interface LabsExperimentDiffData extends z.infer<typeof LabsExperimentDiffSchema> {}
export type LabsEndpointKey = z.infer<typeof LabsEndpointKeySchema>;
export type LabsEndpointRenderStrategy = z.infer<typeof LabsEndpointRenderStrategySchema>;
export interface LabsPaginationPage extends z.infer<typeof LabsPaginationPageSchema> {}
export interface LabsPagination extends z.infer<typeof LabsPaginationSchema> {}
export interface PublicDataIndexEntry extends z.infer<typeof PublicDataIndexEntrySchema> {}
export interface PublicDataIndexData extends z.infer<typeof PublicDataIndexSchema> {}
export interface LabsEndpointStatus extends z.infer<typeof LabsEndpointStatusSchema> {}
export interface LabsDashboardData extends z.infer<typeof LabsDashboardDataSchema> {}

export interface LabsParseResult<T> {
  data: T | null;
  errors: string[];
}

function formatIssuePath(endpoint: string, path: PropertyKey[]): string {
  const suffix = path.length > 0 ? path.join('.') : '$';
  return `${endpoint}.${suffix}`;
}

export function parseLabsJson<T>(
  raw: unknown,
  schema: z.ZodType<T>,
  endpoint: string,
): LabsParseResult<T> {
  const result = schema.safeParse(raw);
  if (result.success) {
    return { data: result.data, errors: [] };
  }

  return {
    data: null,
    errors: result.error.issues.map((issue) => `${formatIssuePath(endpoint, issue.path)}: ${issue.message}`),
  };
}
