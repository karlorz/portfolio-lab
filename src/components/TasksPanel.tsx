import React, { useEffect, useMemo, useState } from 'react';
import { KeyRound, Pause, Play, RefreshCw, RotateCcw, Square, Unlock } from 'lucide-react';
import { parseTaskerStatus, parseTaskerTasks, type TaskerRun, type TaskerStatus, type TaskerTask } from '../schemas/tasker';

interface TasksPanelProps {
  refreshIntervalSeconds?: number;
}

type TaskerAction = 'run' | 'pause' | 'resume' | 'cancel' | 'retry';

async function fetchJson(path: string): Promise<unknown | null> {
  const response = await fetch(path);
  if (!response.ok) return null;
  const contentType = response.headers.get('content-type')?.toLowerCase() ?? '';
  if (!contentType.includes('application/json')) return null;
  return response.json();
}

function formatTime(isoString: string | null): string {
  if (!isoString) return 'Never';
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) return 'Unknown';
  return date.toLocaleString(undefined, {
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function formatDuration(seconds: number | null): string {
  if (seconds === null || Number.isNaN(seconds)) return '-';
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

function taskHealth(task: TaskerTask): 'ok' | 'paused' | 'warning' | 'error' | 'idle' {
  if (task.paused || task.manual_only || !task.enabled) return 'paused';
  if (task.last_status === 'error' || task.last_status === 'timeout') return 'error';
  if (task.consecutive_failures > 0 || task.last_status === 'cancelled') return 'warning';
  if (task.last_status === 'success') return 'ok';
  return 'idle';
}

interface TaskerRunTrace {
  label: string;
  taskId: string;
  runId: string;
  command: string;
  trigger: string;
  status: TaskerRun['status'];
  logPath: string;
}

interface TaskerOperationsSummary {
  headline: string;
  scheduledCount: number;
  manualPausedCount: number;
  failedCount: number;
  runningCount: number;
  failureLabel: string;
  recentRunTraces: TaskerRunTrace[];
}

export function summarizeTaskerOperations(
  status: TaskerStatus | null,
  tasks: TaskerTask[],
): TaskerOperationsSummary {
  const scheduledCount = tasks.filter((task) => task.enabled && !task.manual_only && !task.paused).length;
  const manualPausedCount = tasks.length - scheduledCount;
  const failedCount = tasks.filter((task) => task.last_status === 'error' || task.last_status === 'timeout').length;
  const runningCount = status?.recent_runs.filter((run) => run.status === 'running' || run.status === 'pending').length ?? 0;
  const taskLabels = new Map(tasks.map((task) => [task.id, task.label]));
  const recentRunTraces = (status?.recent_runs ?? []).map((run) => ({
    label: `${taskLabels.get(run.task_id) ?? run.task_id} run ${run.run_id}`,
    taskId: run.task_id,
    runId: run.run_id,
    command: run.command.join(' '),
    trigger: run.trigger,
    status: run.status,
    logPath: run.log_path,
  }));

  return {
    headline: `${tasks.length} registered tasks: ${scheduledCount} scheduled, ${manualPausedCount} manual/paused, ${failedCount} failed, ${runningCount} running`,
    scheduledCount,
    manualPausedCount,
    failedCount,
    runningCount,
    failureLabel: failedCount > 0 ? `${failedCount} failed tasks need attention` : 'No failed tasks',
    recentRunTraces,
  };
}

export function TasksPanel({ refreshIntervalSeconds = 15 }: TasksPanelProps) {
  const [status, setStatus] = useState<TaskerStatus | null>(null);
  const [tasks, setTasks] = useState<TaskerTask[]>([]);
  const [adminToken, setAdminToken] = useState('');
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const unlocked = adminToken.trim().length > 0;
  const selectedTask = useMemo(
    () => tasks.find((task) => task.id === selectedTaskId) ?? tasks[0] ?? null,
    [selectedTaskId, tasks]
  );
  const selectedRun = useMemo(
    () => status?.recent_runs.find((run) => run.run_id === selectedRunId) ?? status?.recent_runs[0] ?? null,
    [selectedRunId, status]
  );

  const loadTaskerData = async () => {
    try {
      const statusRaw = await fetchJson('/api/tasker/status');
      const tasksRaw = await fetchJson('/api/tasks');
      const parsedStatus = parseTaskerStatus(statusRaw);
      const parsedTasks = parseTaskerTasks(tasksRaw);

      if (parsedStatus) {
        setStatus(parsedStatus);
        setTasks(parsedTasks ?? parsedStatus.tasks);
        setError(null);
        return;
      }

      const fallbackRaw = await fetchJson('/data/tasker_status.json');
      const fallbackStatus = parseTaskerStatus(fallbackRaw);
      if (fallbackStatus) {
        setStatus(fallbackStatus);
        setTasks(fallbackStatus.tasks);
        setError(null);
        return;
      }

      setError('Tasker data unavailable');
    } catch {
      try {
        const fallbackRaw = await fetchJson('/data/tasker_status.json');
        const fallbackStatus = parseTaskerStatus(fallbackRaw);
        if (fallbackStatus) {
          setStatus(fallbackStatus);
          setTasks(fallbackStatus.tasks);
          setError(null);
          return;
        }
      } catch { /* render existing state */ }
      setError('Tasker data unavailable');
    }
  };

  useEffect(() => {
    loadTaskerData();
    const interval = setInterval(loadTaskerData, refreshIntervalSeconds * 1000);
    return () => clearInterval(interval);
  }, [refreshIntervalSeconds]);

  useEffect(() => {
    if (!selectedTaskId && tasks.length > 0) {
      setSelectedTaskId(tasks[0].id);
    }
  }, [selectedTaskId, tasks]);

  useEffect(() => {
    if (!selectedRunId && status?.recent_runs.length) {
      setSelectedRunId(status.recent_runs[0].run_id);
    }
  }, [selectedRunId, status]);

  const postAction = async (action: TaskerAction, taskId?: string, runId?: string) => {
    if (!unlocked) return;
    const endpoint = action === 'run'
      ? `/api/tasks/${taskId}/run`
      : action === 'pause'
        ? `/api/tasks/${taskId}/pause`
        : action === 'resume'
          ? `/api/tasks/${taskId}/resume`
          : action === 'cancel'
            ? `/api/runs/${runId}/cancel`
            : `/api/runs/${runId}/retry`;
    setBusyAction(`${action}:${taskId ?? runId ?? ''}`);
    setError(null);
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Tasker-Token': adminToken.trim(),
        },
        body: action === 'pause' ? JSON.stringify({ reason: 'dashboard' }) : undefined,
      });
      if (!response.ok) {
        setError(`Action failed: ${response.status}`);
        return;
      }
      await loadTaskerData();
    } catch {
      setError('Action failed');
    } finally {
      setBusyAction(null);
    }
  };

  const operationsSummary = summarizeTaskerOperations(status, tasks);

  return (
    <div className="tasks-panel">
      <div className="tasks-toolbar">
        <div className="tasks-summary-grid">
          <div className="task-summary-item">
            <label>Registered</label>
            <span>{tasks.length}</span>
          </div>
          <div className="task-summary-item">
            <label>Scheduled</label>
            <span>{operationsSummary.scheduledCount}</span>
          </div>
          <div className="task-summary-item">
            <label>Manual/Paused</label>
            <span>{operationsSummary.manualPausedCount}</span>
          </div>
          <div className="task-summary-item">
            <label>Failed</label>
            <span className={operationsSummary.failedCount > 0 ? 'negative' : 'positive'}>{operationsSummary.failedCount}</span>
          </div>
        </div>

        <div className="tasker-admin-controls">
          <div className={`tasker-unlock-state ${unlocked ? 'unlocked' : 'locked'}`}>
            {unlocked ? <Unlock size={16} /> : <KeyRound size={16} />}
            <span>{unlocked ? 'Unlocked' : 'Read only'}</span>
          </div>
          <input
            type="password"
            value={adminToken}
            onChange={(event) => setAdminToken(event.target.value)}
            placeholder="Admin token"
            aria-label="Tasker admin token"
          />
          <button type="button" className="icon-btn" onClick={loadTaskerData} title="Refresh">
            <RefreshCw size={16} />
          </button>
        </div>
      </div>

      {error && <div className="tasker-error">{error}</div>}

      <div className="tasker-operations-summary">
        <strong>{operationsSummary.headline}</strong>
        <span>{operationsSummary.failureLabel}</span>
      </div>

      <div className="tasks-console-layout">
        <div className="tasks-table-wrap">
          <table className="positions-table tasker-table">
            <thead>
              <tr>
                <th>Task</th>
                <th>Status</th>
                <th>Schedule</th>
                <th>Last Run</th>
                <th>Duration</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {tasks.map((task) => {
                const health = taskHealth(task);
                const busyKey = busyAction?.endsWith(task.id) ?? false;
                return (
                  <tr
                    key={task.id}
                    className={selectedTask?.id === task.id ? 'tasker-selected-row' : ''}
                    onClick={() => setSelectedTaskId(task.id)}
                  >
                    <td>
                      <strong>{task.label}</strong>
                      <small>{task.id}</small>
                    </td>
                    <td>
                      <span className={`tasker-status-pill status-${health}`}>
                        {task.paused ? 'paused' : task.last_status ?? 'idle'}
                      </span>
                    </td>
                    <td>{task.manual_only ? 'manual' : task.schedule ?? '-'}</td>
                    <td>{formatTime(task.last_finished_at)}</td>
                    <td>{formatDuration(task.last_duration_seconds)}</td>
                    <td>
                      <div className="tasker-actions">
                        <button
                          type="button"
                          className="icon-btn"
                          disabled={!unlocked || busyKey || task.paused}
                          onClick={(event) => {
                            event.stopPropagation();
                            postAction('run', task.id);
                          }}
                          title="Run"
                        >
                          <Play size={15} />
                        </button>
                        {task.paused ? (
                          <button
                            type="button"
                            className="icon-btn"
                            disabled={!unlocked || busyKey}
                            onClick={(event) => {
                              event.stopPropagation();
                              postAction('resume', task.id);
                            }}
                            title="Resume"
                          >
                            <Play size={15} />
                          </button>
                        ) : (
                          <button
                            type="button"
                            className="icon-btn"
                            disabled={!unlocked || busyKey}
                            onClick={(event) => {
                              event.stopPropagation();
                              postAction('pause', task.id);
                            }}
                            title="Pause"
                          >
                            <Pause size={15} />
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="tasker-side-panel">
          <div className="tasker-detail-block">
            <h3>{selectedTask?.label ?? 'Task'}</h3>
            <dl>
              <div>
                <dt>Command</dt>
                <dd>{selectedTask?.command.join(' ') ?? '-'}</dd>
              </div>
              <div>
                <dt>Schedule Type</dt>
                <dd>{selectedTask?.manual_only ? 'Manual only' : selectedTask?.schedule ? 'Scheduled' : 'Unscheduled'}</dd>
              </div>
              <div>
                <dt>Failures</dt>
                <dd>{selectedTask?.consecutive_failures ?? 0} consecutive</dd>
              </div>
              <div>
                <dt>Timeout</dt>
                <dd>{selectedTask ? `${selectedTask.timeout_seconds}s` : '-'}</dd>
              </div>
            </dl>
          </div>

          <div className="tasker-detail-block">
            <h3>Recent Runs</h3>
            <div className="tasker-run-list">
              {(status?.recent_runs ?? []).slice(0, 8).map((run) => (
                <button
                  type="button"
                  key={run.run_id}
                  className={`tasker-run-row ${selectedRun?.run_id === run.run_id ? 'active' : ''}`}
                  onClick={() => setSelectedRunId(run.run_id)}
                >
                  <span>{run.task_id.replace('portfolio-lab-', '')}</span>
                  <span className={`tasker-status-pill status-${run.status}`}>{run.status}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="tasker-detail-block">
            <h3>Run Trace</h3>
            {selectedRun ? (
              <>
                <dl>
                  <div>
                    <dt>Run</dt>
                    <dd>{selectedRun.run_id}</dd>
                  </div>
                  <div>
                    <dt>Task</dt>
                    <dd>{selectedRun.task_id}</dd>
                  </div>
                  <div>
                    <dt>Command</dt>
                    <dd>{selectedRun.command.join(' ')}</dd>
                  </div>
                  <div>
                    <dt>Trigger</dt>
                    <dd>{selectedRun.trigger}</dd>
                  </div>
                  <div>
                    <dt>Started</dt>
                    <dd>{formatTime(selectedRun.started_at)}</dd>
                  </div>
                  <div>
                    <dt>Exit</dt>
                    <dd>{selectedRun.exit_code ?? '-'}</dd>
                  </div>
                  <div>
                    <dt>Log</dt>
                    <dd>{selectedRun.log_path || '-'}</dd>
                  </div>
                </dl>
                <div className="tasker-actions">
                  <button
                    type="button"
                    className="icon-btn"
                    disabled={!unlocked || selectedRun.status !== 'running'}
                    onClick={() => postAction('cancel', undefined, selectedRun.run_id)}
                    title="Cancel"
                  >
                    <Square size={15} />
                  </button>
                  <button
                    type="button"
                    className="icon-btn"
                    disabled={!unlocked}
                    onClick={() => postAction('retry', undefined, selectedRun.run_id)}
                    title="Retry"
                  >
                    <RotateCcw size={15} />
                  </button>
                </div>
              </>
            ) : (
              <div className="tasker-empty-state">No runs</div>
            )}
          </div>
        </div>
      </div>

      <div className="tasker-footer">
        <span>Backend: {status?.backend ?? 'tasker'}</span>
        <span>Updated: {formatTime(status?.timestamp ?? null)}</span>
      </div>
    </div>
  );
}
