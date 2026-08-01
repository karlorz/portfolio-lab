import React from 'react';
import {
  WORKSPACE_GROUPS,
  type WorkspaceId,
} from './navigation';
import { workspaceHref } from '../../hooks/useWorkspaceLocation';

interface NavigationRailProps {
  id?: string;
  open?: boolean;
  active: WorkspaceId;
  onNavigate: (workspace: WorkspaceId) => void;
}

export function NavigationRail({ id, open = false, active, onNavigate }: NavigationRailProps) {
  return (
    <nav
      id={id}
      className={`navigation-rail${open ? ' navigation-rail-open' : ''}`}
      aria-label="Portfolio Lab workspaces"
    >
      {WORKSPACE_GROUPS.map((group) => (
        <section key={group.label} className="navigation-group">
          <h2>{group.label}</h2>
          <ul>
            {group.items.map((item) => (
              <li key={item.id}>
                <a
                  href={workspaceHref(item.id)}
                  aria-current={active === item.id ? 'page' : undefined}
                  title={item.description}
                  onClick={(event) => {
                    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
                    event.preventDefault();
                    onNavigate(item.id);
                  }}
                >
                  {item.label}
                </a>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </nav>
  );
}
