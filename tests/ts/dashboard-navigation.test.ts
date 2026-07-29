import { describe, expect, it } from 'bun:test';
import {
  DEFAULT_WORKSPACE,
  WORKSPACE_GROUPS,
  isWorkspaceId,
} from '../../src/components/control-plane/navigation';
import {
  parseWorkspace,
  serializeWorkspace,
} from '../../src/hooks/useWorkspaceLocation';

describe('operations cockpit navigation', () => {
  it('groups operations before research and system workspaces', () => {
    expect(WORKSPACE_GROUPS.map((group) => group.label)).toEqual(['Operations', 'Research', 'System']);
    expect(WORKSPACE_GROUPS[0]!.items.map((item) => item.id)).toEqual([
      'overview', 'health', 'rebalance', 'risk',
    ]);
    expect(WORKSPACE_GROUPS[1]!.items.some((item) => item.id === 'backtests')).toBe(true);
  });

  it('normalizes unknown locations to overview', () => {
    expect(parseWorkspace('?view=health')).toBe('health');
    expect(parseWorkspace('?view=retired')).toBe(DEFAULT_WORKSPACE);
    expect(parseWorkspace('')).toBe(DEFAULT_WORKSPACE);
    expect(isWorkspaceId('backtests')).toBe(true);
  });

  it('preserves unrelated query parameters while serializing workspace state', () => {
    expect(serializeWorkspace('?symbol=SPY&view=health', 'risk')).toBe('?symbol=SPY&view=risk');
    expect(serializeWorkspace('?symbol=SPY&view=health', 'overview')).toBe('?symbol=SPY');
  });
});
