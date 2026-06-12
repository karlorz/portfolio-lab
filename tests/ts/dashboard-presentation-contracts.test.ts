import { describe, expect, it } from 'bun:test';
import { readFileSync } from 'fs';

function read(path: string): string {
  return readFileSync(path, 'utf8');
}

function cssRule(css: string, selector: string): string {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const match = css.match(new RegExp(`${escaped}\\s*\\{([\\s\\S]*?)\\}`));
  expect(match, `missing CSS rule for ${selector}`).not.toBeNull();
  return match?.[1] ?? '';
}

function expectRuleContains(css: string, selector: string, declarations: RegExp[]) {
  const rule = cssRule(css, selector);
  for (const declaration of declarations) {
    expect(rule).toMatch(declaration);
  }
}

function expectNoUtilityClassTokens(path: string) {
  const source = read(path);
  const classStrings = [...source.matchAll(/className=(?:"([^"]+)"|'([^']+)'|\{`([^`]+)`\})/g)]
    .map((match) => match[1] ?? match[2] ?? match[3] ?? '');
  const unsupported = classStrings
    .flatMap((classString) => classString.split(/\s+/).filter(Boolean))
    .filter((token) => !token.includes('${'))
    .filter((token) => [
      /^p-\d/,
      /^px-/,
      /^py-/,
      /^pt-/,
      /^mt-/,
      /^mb-/,
      /^grid$/,
      /^grid-cols-/,
      /^lg:grid-cols-/,
      /^gap-\d/,
      /^bg-gray-/,
      /^bg-white$/,
      /^text-gray-/,
      /^rounded-lg$/,
      /^shadow-sm$/,
      /^border-gray-/,
    ].some((pattern) => pattern.test(token)));

  expect(unsupported, `${path} still has unsupported utility-class presentation markup`).toEqual([]);
}

describe('dashboard presentation source contracts', () => {
  const css = read('src/App.css');
  const liveDashboard = read('src/components/LiveDashboard.tsx');

  it('uses supported local layout classes for dashboard tab groups', () => {
    expect(liveDashboard).toContain('className="dashboard-grid dashboard-grid-two analytics-panel-group"');
    expect(liveDashboard).toContain('className="dashboard-grid dashboard-grid-three analytics-panel-group"');
    expect(liveDashboard).toContain('className="dashboard-section-stack"');
    expect(liveDashboard).toContain('className="risk-primary-grid"');
    expect(liveDashboard).not.toContain('grid grid-cols-1');
    expect(liveDashboard).not.toContain('lg:grid-cols-');
    expect(liveDashboard).not.toContain('className="mt-4"');
  });

  it('wraps twelve dashboard tabs before they can create page-level overflow', () => {
    expectRuleContains(css, '.dashboard-tabs', [
      /flex-wrap:\s*wrap;/,
      /max-width:\s*100%;/,
      /overflow-x:\s*hidden;/,
    ]);
    expectRuleContains(css, '.dashboard-tabs .tab', [
      /flex:\s*1\s+1\s+96px;/,
      /min-width:\s*0;/,
    ]);
    expect(css).toMatch(/@media\s*\(max-width:\s*1100px\)[\s\S]*\.dashboard-tabs \.tab\s*\{[\s\S]*flex-basis:\s*calc\(25% - 4px\);/);
  });

  it('defines real responsive dashboard grids for analytics and risk tabs', () => {
    expectRuleContains(css, '.dashboard-section-stack', [
      /display:\s*flex;/,
      /flex-direction:\s*column;/,
      /gap:\s*16px;/,
    ]);
    expectRuleContains(css, '.dashboard-grid', [
      /display:\s*grid;/,
      /grid-template-columns:\s*repeat\(2,\s*minmax\(0,\s*1fr\)\);/,
    ]);
    expectRuleContains(css, '.dashboard-grid.dashboard-grid-three', [
      /grid-template-columns:\s*repeat\(3,\s*minmax\(0,\s*1fr\)\);/,
    ]);
    expectRuleContains(css, '.risk-primary-grid', [
      /display:\s*grid;/,
      /grid-template-columns:\s*repeat\(2,\s*minmax\(0,\s*1fr\)\);/,
    ]);
  });

  it('defines the rendered Performance summary and card class contract', () => {
    expect(liveDashboard).toContain('className="performance-summary"');
    expect(liveDashboard).toContain('className="perf-card"');
    expectRuleContains(css, '.performance-summary', [
      /display:\s*grid;/,
      /grid-template-columns:\s*repeat\(3,\s*minmax\(0,\s*1fr\)\);/,
      /gap:\s*20px;/,
    ]);
    expectRuleContains(css, '.perf-card', [
      /background:\s*#0f172a;/,
      /padding:\s*15px;/,
      /border-radius:\s*6px;/,
      /min-width:\s*0;/,
    ]);
    expectRuleContains(css, '.perf-card label', [
      /display:\s*block;/,
      /text-transform:\s*uppercase;/,
    ]);
    expectRuleContains(css, '.perf-card .value-display', [
      /display:\s*block;/,
      /font-weight:\s*600;/,
    ]);
    expect(css).toMatch(/@media\s*\(max-width:\s*768px\)[\s\S]*\.performance-summary\s*\{[\s\S]*grid-template-columns:\s*1fr;/);
  });

  it('moves risk presentation panels onto local dashboard CSS classes', () => {
    expectNoUtilityClassTokens('src/components/GarchCvarPanel.tsx');
    expectNoUtilityClassTokens('src/components/EntropyPanel.tsx');
    expectNoUtilityClassTokens('src/components/BondMomentumPanel.tsx');
    expectRuleContains(css, '.risk-card', [
      /background:\s*#1e293b;/,
      /padding:\s*16px;/,
      /border:\s*1px solid #334155;/,
    ]);
    expectRuleContains(css, '.risk-metric-grid', [/display:\s*grid;/]);
    expectRuleContains(css, '.risk-gauge-track', [/position:\s*relative;/]);
  });

  it('defines missing Rebalance and Options panel CSS from their semantic classes', () => {
    for (const selector of [
      '.rebalance-health-panel',
      '.rh-state-grid',
      '.rh-state-item',
      '.rh-history-row',
      '.zero-dte-panel',
      '.risk-summary',
      '.control-grid',
      '.panel',
      '.panel-grid',
      '.collar-viz',
    ]) {
      cssRule(css, selector);
    }
    expectRuleContains(css, '.rebalance-health-panel', [/background:\s*#1e293b;/]);
    expectRuleContains(css, '.rh-state-grid', [/grid-template-columns:\s*repeat\(2,\s*minmax\(0,\s*1fr\)\);/]);
    expectRuleContains(css, '.zero-dte-panel', [/background:\s*#1e293b;/]);
    expectRuleContains(css, '.control-grid', [/grid-template-columns:\s*repeat\(4,\s*minmax\(0,\s*1fr\)\);/]);
  });

  it('keeps ClosingAuctionPanel styles in the shared stylesheet', () => {
    const auctionSource = read('src/components/ClosingAuctionPanel.tsx');

    expect(auctionSource).not.toContain('<style>');
    for (const selector of [
      '.closing-auction-panel',
      '.auction-summary',
      '.signals-table',
      '.direction-badge',
      '.entry-window',
    ]) {
      cssRule(css, selector);
    }
  });
});
