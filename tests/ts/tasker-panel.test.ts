import { describe, expect, it } from 'bun:test';
import { existsSync, readFileSync } from 'fs';

const liveDashboardSource = readFileSync('src/components/LiveDashboard.tsx', 'utf8');
const taskerSchemaPath = 'src/schemas/tasker.ts';
const tasksPanelPath = 'src/components/TasksPanel.tsx';

describe('Tasker dashboard console contract', () => {
  it('adds Tasks as a lazy top-level LiveDashboard tab', () => {
    expect(liveDashboardSource).toContain("const TasksPanel = lazy(() => import('./TasksPanel')");
    expect(liveDashboardSource).toContain("'tasks'");
    expect(liveDashboardSource).toContain("{ id: 'tasks', label: 'Tasks'");
    expect(liveDashboardSource).toMatch(/activeTab === 'tasks'[\s\S]*<TasksPanel \/>/);
    expect(liveDashboardSource).toMatch(/<PanelErrorBoundary name="Tasks">\s*<Suspense fallback=/);
  });

  it('defines Zod schemas for tasker API payloads', () => {
    expect(existsSync(taskerSchemaPath)).toBe(true);
    const schemaSource = readFileSync(taskerSchemaPath, 'utf8');

    expect(schemaSource).toContain('TaskerTaskSchema');
    expect(schemaSource).toContain('TaskerRunSchema');
    expect(schemaSource).toContain('TaskerStatusSchema');
    expect(schemaSource).toContain('parseTaskerStatus');
  });

  it('polls tasker API endpoints with static JSON fallback', () => {
    expect(existsSync(tasksPanelPath)).toBe(true);
    const panelSource = readFileSync(tasksPanelPath, 'utf8');

    expect(panelSource).toContain("fetchJson('/api/tasker/status')");
    expect(panelSource).toContain("fetchJson('/api/tasks')");
    expect(panelSource).toContain("fetchJson('/data/tasker_status.json')");
    expect(panelSource).toContain('setInterval(loadTaskerData, refreshIntervalSeconds * 1000)');
  });

  it('keeps mutations token-gated and scoped to registered task or run endpoints', () => {
    const panelSource = readFileSync(tasksPanelPath, 'utf8');

    expect(panelSource).toContain('X-Tasker-Token');
    expect(panelSource).toContain("method: 'POST'");
    expect(panelSource).toContain("`/api/tasks/${taskId}/run`");
    expect(panelSource).toContain("`/api/tasks/${taskId}/pause`");
    expect(panelSource).toContain("`/api/tasks/${taskId}/resume`");
    expect(panelSource).toContain("`/api/runs/${runId}/cancel`");
    expect(panelSource).toContain("`/api/runs/${runId}/retry`");
    expect(panelSource).not.toContain('/api/tasker/command');
  });
});
