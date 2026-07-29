import { useCallback, useEffect, useState } from 'react';
import {
  DEFAULT_WORKSPACE,
  isWorkspaceId,
  type WorkspaceId,
} from '../components/control-plane/navigation';

export function parseWorkspace(search: string): WorkspaceId {
  const value = new URLSearchParams(search).get('view');
  return isWorkspaceId(value) ? value : DEFAULT_WORKSPACE;
}
export function serializeWorkspace(search: string, workspace: WorkspaceId): string {
  const params = new URLSearchParams(search);
  if (workspace === DEFAULT_WORKSPACE) {
    params.delete('view');
  } else {
    params.set('view', workspace);
  }
  const serialized = params.toString();
  return serialized ? `?${serialized}` : '';
}

export function workspaceHref(workspace: WorkspaceId, search = window.location.search): string {
  return `${window.location.pathname}${serializeWorkspace(search, workspace)}${window.location.hash}`;
}

export function useWorkspaceLocation() {
  const [workspace, setWorkspace] = useState<WorkspaceId>(() => parseWorkspace(window.location.search));

  useEffect(() => {
    const handlePopState = () => setWorkspace(parseWorkspace(window.location.search));
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  const navigate = useCallback((next: WorkspaceId) => {
    const nextUrl = workspaceHref(next);
    window.history.pushState({ workspace: next }, '', nextUrl);
    setWorkspace(next);
    window.requestAnimationFrame(() => {
      document.getElementById('workspace-heading')?.focus({ preventScroll: true });
    });
  }, []);

  return { workspace, navigate };
}
