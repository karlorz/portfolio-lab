import { describe, expect, it } from 'bun:test';
import { readFileSync } from 'fs';

function read(path: string): string {
  return readFileSync(path, 'utf8');
}

describe('dashboard delivery source contracts', () => {
  it('publishes a prices_compact.json artifact from the canonical fetch script', () => {
    const source = read('scripts/fetch-data.ts');

    expect(source).toContain("'prices_compact.json'");
    expect(source).toContain('writeJsonAtomic(pricesCompactPath, pricesCompactPayload)');
    expect(source).toContain('buildLastNBarsCompact');
  });

  it('syncs fetched prices into market.db before regenerating dashboard health', () => {
    const source = read('scripts/fetch-data.ts');
    const syncIndex = source.indexOf("runPythonModule('src.data.market_db_sync')");
    const generatorWrapperIndex = source.indexOf("runModule('src.dashboard.generator')");
    const generatorCallIndex = source.indexOf('await runDashboardGeneration()');

    expect(syncIndex).toBeGreaterThan(-1);
    expect(generatorWrapperIndex).toBeGreaterThan(-1);
    expect(generatorCallIndex).toBeGreaterThan(-1);
    expect(syncIndex).toBeLessThan(generatorCallIndex);
    expect(source).toContain("const PYTHON_RUNTIME = join(PROJECT_ROOT, 'scripts', 'python_runtime.sh');");
    expect(source).toContain("execFileSync(PYTHON_RUNTIME, ['-m', moduleName]");
    expect(source).not.toContain('python3 -m src.data.market_db_sync');
    expect(source).not.toContain('python3 -m src.dashboard.generator');
  });

  it('updates portfolio-lab-dashboard cron status after data-pipeline generator success', () => {
    const source = read('scripts/fetch-data.ts');

    expect(source).toContain('recordDashboardCronStatus');
    expect(source).toContain("'portfolio-lab-dashboard'");
    expect(source).toContain("join(PROJECT_ROOT, 'scripts', 'cron_update.py')");
    expect(source).toContain('triggered_by=${triggeredBy}');
    expect(source).toContain("triggeredBy: 'fetch_data'");
    // Status update must follow successful generator, not precede it
    const genIndex = source.indexOf("await runModule('src.dashboard.generator')");
    const statusIndex = source.indexOf('recordStatus({');
    expect(genIndex).toBeGreaterThan(-1);
    expect(statusIndex).toBeGreaterThan(genIndex);
  });

  it('fails the data job when any configured price symbol returns no rows', () => {
    const source = read('scripts/fetch-data.ts');

    expect(source).toContain('missingSymbols');
    expect(source).toMatch(/SYMBOLS\.filter\([^)]*priceData\[symbol\]/);
    expect(source).toMatch(/throw new Error\([^)]*missingSymbols/);
  });

  it('keeps mobile portfolio selector children from forcing document-level overflow', () => {
    const css = read('src/App.css');

    expect(css).toMatch(/\.portfolio-selector\s*\{[\s\S]*min-width:\s*0;/);
    expect(css).toMatch(/\.portfolio-categories\s*\{[\s\S]*min-width:\s*0;/);
    expect(css).toMatch(/\.portfolio-category\s*\{[\s\S]*min-width:\s*0;/);
    expect(css).toMatch(/\.toggle\s*\{[\s\S]*min-width:\s*0;/);
    expect(css).toMatch(/@media\s*\(max-width:\s*600px\)[\s\S]*\.portfolio-category\s*\{[\s\S]*align-items:\s*stretch;/);
  });

  it('clips live dashboard tables inside their own scroll containers on mobile', () => {
    const css = read('src/App.css');

    expect(css).toMatch(/\.live-dashboard\s*\{[\s\S]*min-width:\s*0;/);
    expect(css).toMatch(/\.positions-section,\s*\.orders-section\s*\{[\s\S]*overflow-x:\s*auto;/);
    expect(css).toMatch(/\.positions-table,\s*\.orders-table\s*\{[\s\S]*min-width:\s*520px;/);
  });

  it('contains wide dashboard tables in local scroll wrappers', () => {
    const css = read('src/App.css');

    expect(css).toMatch(/\.comparison-table\s*\{[\s\S]*max-width:\s*100%;[\s\S]*overflow-x:\s*auto;/);
    expect(css).toMatch(/\.overflow-x-auto\s*\{[\s\S]*max-width:\s*100%;[\s\S]*overflow-x:\s*auto;/);
  });

  it('wraps FIRE calculator segmented controls instead of widening mobile pages', () => {
    const css = read('src/App.css');
    const source = read('src/components/FIRECalculator.tsx');

    expect(source).toContain('className="fire-control-buttons"');
    expect(css).toMatch(/\.fire-control-buttons\s*\{[\s\S]*flex-wrap:\s*wrap;/);
  });

  it('removes hidden Recharts tooltips from layout so they cannot widen mobile pages', () => {
    const css = read('src/App.css');

    expect(css).toMatch(/\.recharts-tooltip-wrapper\[style\*=["']visibility:\s*hidden["']\]\s*\{[\s\S]*display:\s*none\s*!important;/);
  });

  it('sets separate cache policies for immutable assets and live dashboard data', () => {
    const caddy = read('Caddyfile');

    expect(caddy).toMatch(/handle\s+\/assets\/\*[\s\S]*Cache-Control\s+"public, max-age=31536000, immutable"/);
    expect(caddy).toMatch(/handle\s+\/data\/\*[\s\S]*Cache-Control\s+"no-cache"/);
  });

  it('registers decision_registry.json in the public data index contract', () => {
    const source = read('src/dashboard/public_data_index.py');

    expect(source).toContain('"decision_registry.json"');
    expect(source).toContain('DECISION_REGISTRY_SCHEMA_VERSION');
  });

  it('publishes decision_registry.json from dashboard generator run', () => {
    const source = read('src/dashboard/generator.py');

    expect(source).toContain('publish_decision_registry_json');
    expect(source).toContain('record_dashboard_cycle_decision');
  });
});
