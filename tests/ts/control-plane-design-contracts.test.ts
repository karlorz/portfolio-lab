import { describe, expect, it } from 'bun:test';
import { readFileSync } from 'fs';

const read = (path: string) => readFileSync(path, 'utf8');
const components = [
  'AppShell',
  'NavigationRail',
  'WorkspaceHeader',
  'CommandPalette',
  'AllocationSpine',
  'AuthorityBadge',
  'StatusBadge',
  'MetricCard',
  'ActionCenter',
  'ContextRail',
  'DesignGuidePage',
];

describe('control-plane design contracts', () => {
  it('defines semantic tokens and interaction foundations', () => {
    const tokens = read('src/styles/tokens.css');
    const base = read('src/styles/base.css');
    for (const token of [
      '--surface-canvas',
      '--surface-panel',
      '--text-primary',
      '--border-subtle',
      '--status-warning',
      '--authority-accent',
      '--focus-ring',
    ]) {
      expect(tokens).toContain(token);
    }
    expect(base).toContain(':focus-visible');
    expect(base).toContain('prefers-reduced-motion');
    expect(base).toContain('font-variant-numeric: tabular-nums');
  });

  it('keeps raw palette literals out of control-plane components', () => {
    for (const component of components) {
      const source = read(`src/components/control-plane/${component}.tsx`);
      expect(source).not.toMatch(/#[0-9a-f]{3,8}\b/i);
      expect(source).not.toMatch(/\b(?:rgb|hsl)a?\(/i);
      expect(source).not.toContain('transition: all');
    }
  });

  it('loads token layers before compatibility styles', () => {
    const app = read('src/App.tsx');
    expect(app.indexOf("./styles/tokens.css")).toBeGreaterThanOrEqual(0);
    expect(app.indexOf("./styles/base.css")).toBeGreaterThan(app.indexOf("./styles/tokens.css"));
    expect(app.indexOf("./styles/control-plane.css")).toBeGreaterThan(app.indexOf("./styles/base.css"));
    expect(app.indexOf("./App.css")).toBeGreaterThan(app.indexOf("./styles/control-plane.css"));
  });

  it('shows every reusable control-plane composite in the living guide', () => {
    const guide = read('src/components/control-plane/DesignGuidePage.tsx');
    for (const component of [
      'AppErrorBoundary',
      'AppShell',
      'NavigationRail',
      'WorkspaceHeader',
      'CommandPalette',
      'AllocationSpine',
      'AuthorityBadge',
      'StatusBadge',
      'MetricCard',
      'ActionCenter',
      'ContextRail',
    ]) {
      expect(guide).toContain(`<${component}`);
    }
  });
});
