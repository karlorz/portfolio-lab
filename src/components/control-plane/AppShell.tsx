import React, { useCallback, useRef, useState } from 'react';
import { CommandPalette } from './CommandPalette';
import { NavigationRail } from './NavigationRail';
import { WorkspaceHeader } from './WorkspaceHeader';
import type { WorkspaceId } from './navigation';
import { useCommandPaletteShortcut } from '../../hooks/useKeyboardShortcuts';

interface AppShellProps {
  workspace: WorkspaceId;
  onNavigate: (workspace: WorkspaceId) => void;
  spine?: React.ReactNode;
  context?: React.ReactNode;
  refreshedAt?: string;
  children: React.ReactNode;
}

export function AppShell({ workspace, onNavigate, spine, context, refreshedAt, children }: AppShellProps) {
  const [paletteOpen, setPaletteOpen] = useState(false);
  const paletteInvoker = useRef<HTMLButtonElement>(null);
  const openPalette = useCallback(() => setPaletteOpen(true), []);
  useCommandPaletteShortcut(openPalette);

  const closePalette = () => {
    setPaletteOpen(false);
    window.requestAnimationFrame(() => paletteInvoker.current?.focus());
  };

  return (
    <div className="control-plane">
      <a className="skip-link" href="#workspace-main">Skip to workspace</a>
      <header className="control-plane-masthead">
        <a className="control-plane-brand" href="/" aria-label="Portfolio Lab overview">
          <span className="control-plane-brand-mark" aria-hidden="true">PL</span>
          <span>Portfolio Lab</span>
        </a>
        <button
          ref={paletteInvoker}
          className="command-palette-trigger"
          type="button"
          onClick={openPalette}
          aria-keyshortcuts="Control+K Meta+K"
        >
          <span className="command-palette-label">Navigate</span>
          <kbd>⌘ K</kbd>
        </button>
      </header>
      {spine}
      <div className={`control-plane-grid ${context ? 'control-plane-grid-with-context' : 'control-plane-grid-no-context'}`}>
        <NavigationRail active={workspace} onNavigate={onNavigate} />
        <main id="workspace-main" className="workspace-main">
          <WorkspaceHeader workspace={workspace} refreshedAt={refreshedAt} />
          {children}
        </main>
        {context ? context : null}
      </div>
      <CommandPalette open={paletteOpen} onClose={closePalette} onNavigate={onNavigate} />
    </div>
  );
}
