import React from 'react';
import { workspaceDestination, type WorkspaceId } from './navigation';

interface WorkspaceHeaderProps {
  workspace: WorkspaceId;
  refreshedAt?: string;
  actions?: React.ReactNode;
}
export function WorkspaceHeader({ workspace, refreshedAt, actions }: WorkspaceHeaderProps) {
  const destination = workspaceDestination(workspace);
  return (
    <header className="workspace-header">
      <div>
        <p className="control-eyebrow">{destination.description}</p>
        <h1 id="workspace-heading" tabIndex={-1}>{destination.label}</h1>
      </div>
      <div className="workspace-header-actions">
        {refreshedAt && <span>Refreshed {refreshedAt}</span>}
        {actions}
      </div>
    </header>
  );
}
