export type WorkspaceId =
  | 'overview'
  | 'health'
  | 'rebalance'
  | 'risk'
  | 'performance'
  | 'analytics'
  | 'options'
  | 'auction'
  | 'labs'
  | 'decisions'
  | 'history'
  | 'tasks'
  | 'chat'
  | 'backtests';

export interface WorkspaceDestination {
  id: WorkspaceId;
  label: string;
  description: string;
}

export interface WorkspaceGroup {
  label: string;
  items: WorkspaceDestination[];
}

export const DEFAULT_WORKSPACE: WorkspaceId = 'overview';

export const WORKSPACE_GROUPS: WorkspaceGroup[] = [
  {
    label: 'Operations',
    items: [
      { id: 'overview', label: 'Overview', description: 'Authority, exceptions, and current posture' },
      { id: 'health', label: 'Health', description: 'Data freshness, jobs, and incidents' },
      { id: 'rebalance', label: 'Rebalance', description: 'Broker state and allocation drift' },
      { id: 'risk', label: 'Risk', description: 'Tail risk, entropy, and hedges' },
    ],
  },
  {
    label: 'Research',
    items: [
      { id: 'performance', label: 'Performance', description: 'Paper-book results and comparisons' },
      { id: 'analytics', label: 'Analytics', description: 'Signals, models, and explainability' },
      { id: 'options', label: 'Options', description: '0DTE and collar research' },
      { id: 'auction', label: 'Auction', description: 'Closing-auction signals' },
      { id: 'labs', label: 'Labs', description: 'Experimental research surfaces' },
      { id: 'decisions', label: 'Decisions', description: 'Decision replay and evidence' },
      { id: 'backtests', label: 'Backtests', description: 'Historical portfolio comparison' },
    ],
  },
  {
    label: 'System',
    items: [
      { id: 'history', label: 'History', description: 'Regime and state timeline' },
      { id: 'tasks', label: 'Tasks', description: 'Scheduled and operator work' },
      { id: 'chat', label: 'Chat', description: 'Research assistant' },
    ],
  },
];

export const WORKSPACES = WORKSPACE_GROUPS.flatMap((group) => group.items);

export const WORKSPACE_IDS = new Set<WorkspaceId>(
  WORKSPACES.map((item) => item.id),
);

export function isWorkspaceId(value: string | null): value is WorkspaceId {
  return value !== null && WORKSPACE_IDS.has(value as WorkspaceId);
}

export function workspaceDestination(id: WorkspaceId): WorkspaceDestination {
  return WORKSPACES.find((item) => item.id === id) ?? WORKSPACES[0]!;
}
