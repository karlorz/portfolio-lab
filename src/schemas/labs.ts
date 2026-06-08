import { z } from 'zod';

export const LABS_REGISTRY_SCHEMA_VERSION = 'labs-registry/v1';
export const LABS_SCORECARD_SCHEMA_VERSION = 'labs-scorecard/v1';
export const LABS_REPLAY_SCHEMA_VERSION = 'labs-replay/v1';
export const LABS_PROVENANCE_SCHEMA_VERSION = 'experiment-manifest/v1';

const MetricMapSchema = z.record(z.string(), z.number());
const StringMapSchema = z.record(z.string(), z.string());
const UnknownMapSchema = z.record(z.string(), z.unknown());

export const LabsStatusSchema = z.enum([
  'candidate',
  'validated',
  'warning',
  'rejected',
  'archived',
]);

export const LabsScorecardStatusSchema = z.enum(['promote', 'watch', 'reject']);
export const LabsReplayStatusSchema = z.enum(['passed', 'failed', 'warning']);

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
}).strict();

export const LabsRegistrySchema = z.object({
  schema_version: z.literal(LABS_REGISTRY_SCHEMA_VERSION),
  generated_at: z.string(),
  experiments: z.array(LabsRegistryRowSchema),
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
}).strict();

export const LabsReplaySchema = z.object({
  schema_version: z.literal(LABS_REPLAY_SCHEMA_VERSION),
  experiment_id: z.string(),
  generated_at: z.string(),
  artifact_path: z.string(),
  status: LabsReplayStatusSchema,
  provenance_status: LabsProvenanceStatusSchema,
  metrics: MetricMapSchema,
  baseline_deltas: MetricMapSchema,
}).strict();

export const LabsValidationResultSchema = z.object({
  path: z.nullable(z.string()),
  artifact_type: z.enum(['registry', 'provenance', 'scorecard', 'replay', 'unknown']),
  schema_version: z.nullable(z.string()),
  valid: z.boolean(),
  errors: z.array(z.string()),
}).strict();

export const LabsValidationReportSchema = z.object({
  results: z.array(LabsValidationResultSchema),
}).strict();

export const LabsEndpointKeySchema = z.enum(['registry', 'scorecards', 'replays', 'validation']);

export const LabsDashboardDataSchema = z.object({
  available: z.boolean(),
  registry: z.nullable(LabsRegistrySchema),
  scorecards: z.array(LabsScorecardSchema),
  replays: z.array(LabsReplaySchema),
  validation: z.nullable(LabsValidationReportSchema),
  missing: z.array(LabsEndpointKeySchema),
  errors: z.array(z.string()),
}).strict();

export interface LabsRegistryRow extends z.infer<typeof LabsRegistryRowSchema> {}
export interface LabsRegistryData extends z.infer<typeof LabsRegistrySchema> {}
export interface LabsProvenanceData extends z.infer<typeof LabsProvenanceSchema> {}
export interface LabsScorecardData extends z.infer<typeof LabsScorecardSchema> {}
export interface LabsReplayData extends z.infer<typeof LabsReplaySchema> {}
export interface LabsValidationResult extends z.infer<typeof LabsValidationResultSchema> {}
export interface LabsValidationReport extends z.infer<typeof LabsValidationReportSchema> {}
export type LabsEndpointKey = z.infer<typeof LabsEndpointKeySchema>;
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
