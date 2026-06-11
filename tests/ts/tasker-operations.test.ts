import { describe, expect, it } from 'bun:test';
import { summarizeTaskerOperations } from '../../src/components/TasksPanel';
import type { TaskerRun, TaskerStatus, TaskerTask } from '../../src/schemas/tasker';

const task = (overrides: Partial<TaskerTask>): TaskerTask => ({
  id: 'portfolio-lab-data',
  label: 'Data',
  command: ['make', 'data'],
  schedule: '5 * * * *',
  enabled: true,
  manual_only: false,
  timeout_seconds: 300,
  paused: false,
  pause_reason: null,
  last_status: 'success',
  last_run_id: 'run-data',
  last_finished_at: '2026-06-11T03:13:45Z',
  last_duration_seconds: 28,
  failure_count: 0,
  consecutive_failures: 0,
  ...overrides,
});

const run = (overrides: Partial<TaskerRun>): TaskerRun => ({
  run_id: 'run-data',
  task_id: 'portfolio-lab-data',
  command: ['make', 'data'],
  trigger: 'scheduled',
  retry_of: null,
  status: 'success',
  pid: 123,
  started_at: '2026-06-11T03:13:17Z',
  finished_at: '2026-06-11T03:13:45Z',
  duration_seconds: 28,
  exit_code: 0,
  error: null,
  log_path: '/root/projects/portfolio-lab/data/tasker_logs/run-data.log',
  created_at: '2026-06-11T03:13:17Z',
  updated_at: '2026-06-11T03:13:45Z',
  ...overrides,
});

describe('Tasker operations summary', () => {
  it('labels scheduled jobs separately from failures and exposes run trace fields', () => {
    const tasks = [
      task({}),
      task({ id: 'portfolio-lab-dashboard', label: 'Dashboard', command: ['make', 'dashboard'], schedule: '15 * * * *' }),
      task({ id: 'portfolio-lab-build', label: 'Build', command: ['make', 'build'], schedule: null, enabled: false, manual_only: true }),
    ];
    const status: TaskerStatus = {
      service: 'portfolio-lab-tasker',
      backend: 'tasker',
      timestamp: '2026-06-11T03:14:00Z',
      tasks,
      recent_runs: [run({})],
    };

    const summary = summarizeTaskerOperations(status, tasks);

    expect(summary.headline).toBe('3 registered tasks: 2 scheduled, 1 manual/paused, 0 failed, 0 running');
    expect(summary.failureLabel).toBe('No failed tasks');
    expect(summary.recentRunTraces[0]).toEqual({
      label: 'Data run run-data',
      taskId: 'portfolio-lab-data',
      runId: 'run-data',
      command: 'make data',
      trigger: 'scheduled',
      status: 'success',
      logPath: '/root/projects/portfolio-lab/data/tasker_logs/run-data.log',
    });
  });
});
