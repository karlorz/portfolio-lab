import { z } from 'zod';

export const TaskerRunStatusSchema = z.enum([
  'pending',
  'running',
  'success',
  'error',
  'timeout',
  'cancelled',
]);

export const TaskerRunSchema = z.object({
  run_id: z.string(),
  task_id: z.string(),
  command: z.array(z.string()),
  trigger: z.string(),
  retry_of: z.nullable(z.string()),
  status: TaskerRunStatusSchema,
  pid: z.nullable(z.number()),
  started_at: z.nullable(z.string()),
  finished_at: z.nullable(z.string()),
  duration_seconds: z.nullable(z.number()),
  exit_code: z.nullable(z.number()),
  error: z.nullable(z.string()),
  // Task 3A: named terminal cause (service_restart / operator_cancelled /
  // unplanned); additive, null for legacy rows.
  termination_cause: z.nullable(z.string()).optional(),
  termination_detail: z.nullable(z.string()).optional(),
  log_path: z.string(),
  created_at: z.string(),
  updated_at: z.string(),
});

export const TaskerTaskSchema = z.object({
  id: z.string(),
  label: z.string(),
  command: z.array(z.string()),
  schedule: z.nullable(z.string()),
  enabled: z.boolean(),
  manual_only: z.boolean(),
  timeout_seconds: z.number(),
  paused: z.boolean(),
  pause_reason: z.nullable(z.string()),
  last_status: z.nullable(z.string()),
  last_run_id: z.nullable(z.string()),
  last_finished_at: z.nullable(z.string()),
  last_duration_seconds: z.nullable(z.number()),
  failure_count: z.number(),
  consecutive_failures: z.number(),
});

export const TaskerStatusSchema = z.object({
  service: z.string(),
  backend: z.literal('tasker'),
  timestamp: z.string(),
  tasks: z.array(TaskerTaskSchema),
  recent_runs: z.array(TaskerRunSchema),
});

export const TaskerTasksResponseSchema = z.object({
  tasks: z.array(TaskerTaskSchema),
});

export type TaskerRunStatus = z.infer<typeof TaskerRunStatusSchema>;
export type TaskerRun = z.infer<typeof TaskerRunSchema>;
export type TaskerTask = z.infer<typeof TaskerTaskSchema>;
export type TaskerStatus = z.infer<typeof TaskerStatusSchema>;

export function parseTaskerStatus(raw: unknown): TaskerStatus | null {
  const parsed = TaskerStatusSchema.safeParse(raw);
  if (parsed.success) return parsed.data;
  return null;
}

export function parseTaskerTasks(raw: unknown): TaskerTask[] | null {
  const parsed = TaskerTasksResponseSchema.safeParse(raw);
  if (parsed.success) return parsed.data.tasks;
  return null;
}
