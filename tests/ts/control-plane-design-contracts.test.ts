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
  'OverflowRegion',
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
      'OverflowRegion',
    ]) {
      expect(guide).toContain(`<${component}`);
    }
  });

  it('makes the shell context track conditional and keeps mobile navigation before content', () => {
    const shell = read('src/components/control-plane/AppShell.tsx');
    const css = read('src/styles/control-plane.css');

    expect(shell).toContain('control-plane-grid-with-context');
    expect(shell).toContain('control-plane-grid-no-context');
    expect(shell).toContain('context ? context : null');
    expect(css).toContain('.control-plane-grid-no-context');
    expect(css).toContain('.control-plane-grid-with-context');
    expect(css).not.toContain('.workspace-main {\n    order: -1;');
  });

  it('uses one named container-query owner for dashboard reflow', () => {
    const dashboard = read('src/components/LiveDashboard.tsx');
    const css = read('src/styles/control-plane.css');
    const contextIndex = dashboard.indexOf('<ContextRail');
    const tabIndex = dashboard.indexOf('<div className="tab-content">');

    expect(css).toContain('container: live-dashboard / inline-size');
    expect(css).toContain('@container live-dashboard');
    expect(css).toContain('grid-template-areas: "main context"');
    expect(css).toContain('"context"');
    expect(css).toContain('"main"');
    expect(contextIndex).toBeGreaterThanOrEqual(0);
    expect(tabIndex).toBeGreaterThan(contextIndex);
    expect(dashboard).not.toContain('overview-mobile-actions');
  });

  it('wraps wide live tables in named keyboard-focusable overflow regions', () => {
    const overflow = read('src/components/control-plane/OverflowRegion.tsx');
    const dashboard = read('src/components/LiveDashboard.tsx');
    const css = read('src/styles/control-plane.css');

    expect(overflow).toContain('aria-label={label}');
    expect(overflow).toContain('tabIndex={0}');
    expect(css).toContain('.overflow-region');
    expect(css).toContain('overflow-x: auto');
    expect(css).toContain('.overflow-region:focus-visible');
    expect(dashboard).toContain('<OverflowRegion label="Current positions table">');
    expect(dashboard).toContain('<OverflowRegion label="Recent orders table">');
  });

  it('uses semantic regime tones and tabular numeric table regions', () => {
    const dashboard = read('src/components/LiveDashboard.tsx');
    const css = read('src/styles/control-plane.css');

    expect(dashboard).not.toMatch(/#[0-9a-f]{3,8}\b/i);
    expect(dashboard).toContain('regime-tone-');
    expect(dashboard).toContain('regime-text-');
    expect(css).toContain('font-variant-numeric: tabular-nums');
  });
});
