import React, { useEffect, useMemo, useRef, useState } from 'react';
import { WORKSPACES, type WorkspaceId } from './navigation';

interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
  onNavigate: (workspace: WorkspaceId) => void;
}

export function CommandPalette({ open, onClose, onNavigate }: CommandPaletteProps) {
  const [query, setQuery] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    setQuery('');
    inputRef.current?.focus();
  }, [open]);

  const results = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return WORKSPACES.filter(
      (item) => !needle || `${item.label} ${item.description}`.toLowerCase().includes(needle),
    );
  }, [query]);

  if (!open) return null;

  return (
    <div
      className="command-palette-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.currentTarget === event.target) onClose();
      }}
    >
      <section
        className="command-palette"
        role="dialog"
        aria-modal="true"
        aria-labelledby="command-palette-title"
        onKeyDown={(event) => {
          if (event.key === 'Escape') {
            event.preventDefault();
            onClose();
          }
          if (event.key === 'Tab') {
            const focusable = [inputRef.current, ...Array.from(event.currentTarget.querySelectorAll<HTMLButtonElement>('button'))]
              .filter(Boolean) as HTMLElement[];
            if (focusable.length === 0) return;
            const first = focusable[0]!;
            const last = focusable[focusable.length - 1]!;
            if (event.shiftKey && document.activeElement === first) {
              event.preventDefault();
              last.focus();
            } else if (!event.shiftKey && document.activeElement === last) {
              event.preventDefault();
              first.focus();
            }
          }
        }}
      >
        <div className="command-palette-heading">
          <h2 id="command-palette-title">Go to workspace</h2>
          <button type="button" onClick={onClose} aria-label="Close command palette">Esc</button>
        </div>
        <label htmlFor="workspace-command-search">Search workspaces</label>
        <input
          ref={inputRef}
          id="workspace-command-search"
          name="workspace-command-search"
          autoComplete="off"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Try “health”…"
        />
        <div className="command-palette-results">
          {results.length === 0 ? (
            <p className="control-empty">No matching workspace.</p>
          ) : results.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => {
                onNavigate(item.id);
                onClose();
              }}
            >
              <strong>{item.label}</strong>
              <span>{item.description}</span>
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}
