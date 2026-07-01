import { describe, expect, it } from 'bun:test';
import { readFileSync } from 'fs';

const source = readFileSync('src/components/LiveDashboard.tsx', 'utf8');

const overviewStaticImports = [
  './YieldCurveIndicator',
  './BondAllocationPanel',
  './DurationOverlayPanel',
  './BehavioralSentimentPanel',
  './CryptoAllocationPanel',
  './CalendarSeasonalityPanel',
];

const lazyPanelImports = [
  { name: 'HealthPanel', path: './HealthPanel' },
  { name: 'RegimeTimeline', path: './RegimeTimeline' },
  { name: 'SPYComparisonChart', path: './SPYComparisonChart' },
  { name: 'RebalancePanel', path: './RebalancePanel' },
  { name: 'SmartRebalancePanel', path: './SmartRebalancePanel' },
  { name: 'BrokerPanel', path: './BrokerPanel' },
  { name: 'RebalanceHealthPanel', path: './RebalanceHealthPanel' },
  { name: 'UnderwaterChart', path: './AnalyticsCharts' },
  { name: 'RollingMetricsChart', path: './AnalyticsCharts' },
  { name: 'CrisisOverlay', path: './AnalyticsCharts' },
  { name: 'PortfolioExplainabilityPanel', path: './PortfolioExplainabilityPanel' },
  { name: 'EnsembleVotingPanel', path: './EnsembleVotingPanel' },
  { name: 'AlternativeDataPanel', path: './AlternativeDataPanel' },
  { name: 'FactorRotationPanel', path: './FactorRotationPanel' },
  { name: 'StackingEnsemblePanel', path: './StackingEnsemblePanel' },
  { name: 'ConvexityHarvestPanel', path: './ConvexityHarvestPanel' },
  { name: 'LLMSentimentPanel', path: './LLMSentimentPanel' },
  { name: 'SectorRotationPanel', path: './SectorRotationPanel' },
  { name: 'MLSignalsPanel', path: './MLSignalsPanel' },
  { name: 'FactorRotationDashboardPanel', path: './FactorRotationDashboardPanel' },
  { name: 'GraduationChecklistPanel', path: './GraduationChecklistPanel' },
  { name: 'AdaptiveSizingPanel', path: './AdaptiveSizingPanel' },
  { name: 'VixyHedgeSizingPanel', path: './VixyHedgeSizingPanel' },
  { name: 'HedgeSelectorPanel', path: './HedgeSelectorPanel' },
  { name: 'BlackLittermanMapperPanel', path: './BlackLittermanMapperPanel' },
  { name: 'TurnoverValidatorPanel', path: './TurnoverValidatorPanel' },
  { name: 'RegimeGatePanel', path: './RegimeGatePanel' },
  { name: 'TSMOMPanel', path: './TSMOMPanel' },
  { name: 'CrossAssetRVPanel', path: './CrossAssetRVPanel' },
  { name: 'ModelValidationPanel', path: './ModelValidationPanel' },
  { name: 'ZeroDTEPanel', path: './ZeroDTEPanel' },
  { name: 'CollarPanel', path: './CollarPanel' },
  { name: 'ClosingAuctionPanel', path: './ClosingAuctionPanel' },
  { name: 'GarchCvarPanel', path: './GarchCvarPanel' },
  { name: 'EntropyPanel', path: './EntropyPanel' },
  { name: 'VIXTermStructurePanel', path: './VIXTermStructurePanel' },
  { name: 'BondMomentumPanel', path: './BondMomentumPanel' },
  { name: 'KurtosisRegimePanel', path: './KurtosisRegimePanel' },
  { name: 'VolatilityParityPanel', path: './VolatilityParityPanel' },
  { name: 'LabsPanel', path: './LabsPanel' },
  { name: 'ChatPanel', path: './ChatPanel' },
  { name: 'TasksPanel', path: './TasksPanel' },
];

const lazyTabBoundaries = [
  'Health',
  'History',
  'Performance',
  'Rebalance',
  'Analytics',
  '0DTE Options',
  'Closing Auction',
  'Risk',
  'Labs',
  'Tasks',
  'Chat',
];

function hasStaticValueImport(importPath: string, importedName: string): boolean {
  const escapedPath = importPath.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const escapedName = importedName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const importPattern = new RegExp(`import\\s+(?!type)[^;]*\\b${escapedName}\\b[^;]*from ['"]${escapedPath}['"];`);
  return importPattern.test(source);
}

function sourceBetween(startMarker: string, endMarker: string): string {
  const start = source.indexOf(startMarker);
  const end = source.indexOf(endMarker, start + startMarker.length);
  expect(start).toBeGreaterThanOrEqual(0);
  expect(end).toBeGreaterThan(start);
  return source.slice(start, end);
}

describe('LiveDashboard lazy tab panel contract', () => {
  it('keeps Overview dependencies statically imported for first render', () => {
    for (const importPath of overviewStaticImports) {
      expect(source).toContain(`from '${importPath}'`);
    }
    expect(source).toMatch(/<PanelErrorBoundary name="Overview">\s*<div className="tab-panel overview-panel">/);
  });

  it('does not statically import non-overview tab panel values', () => {
    for (const panel of lazyPanelImports) {
      expect(hasStaticValueImport(panel.path, panel.name)).toBe(false);
    }
  });

  it('declares lazy dynamic imports for non-overview tab panel modules', () => {
    const uniqueImportPaths = [...new Set(lazyPanelImports.map((panel) => panel.path))];

    expect(source).toContain('lazy(() => import(');
    for (const importPath of uniqueImportPaths) {
      expect(source).toContain(`import('${importPath}')`);
    }
  });

  it('renders lazy tab contents under PanelErrorBoundary and Suspense', () => {
    for (const boundaryName of lazyTabBoundaries) {
      const escapedName = boundaryName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      expect(source).toMatch(new RegExp(`<PanelErrorBoundary name="${escapedName}">\\s*<Suspense fallback=`));
    }
  });

  it('keeps optional Analytics children isolated behind panel-level boundaries', () => {
    const analyticsSource = sourceBetween('{/* Analytics Tab */}', '{/* 0DTE Options Tab */}');
    const expectedBoundaries = [
      'Analytics/Explainability',
      'Analytics/Ensemble Voting',
      'Analytics/Graduation Checklist',
      'Analytics/Adaptive Sizing',
      'Analytics/VIXY Hedge Sizing',
      'Analytics/Black-Litterman',
      'Analytics/Turnover Validator',
    ];

    for (const boundaryName of expectedBoundaries) {
      expect(analyticsSource).toContain(`<PanelErrorBoundary name="${boundaryName}">`);
    }
  });

  it('wires Labs as a lazy tab that delegates endpoint loading to the Labs fetch helper', () => {
    const labsPanelSource = readFileSync('src/components/LabsPanel.tsx', 'utf8');

    expect(source).toContain("type IncidentTab");
    expect(source).toContain('type TabType = IncidentTab;');
    expect(source).toContain("{ id: 'labs', label: 'Labs'");
    expect(source).toContain("import('./LabsPanel')");
    expect(source).toMatch(/activeTab === 'labs'[\s\S]*<LabsPanel \/>/);
    expect(labsPanelSource).toContain("from '../data/labs'");
    expect(labsPanelSource).toContain('fetchLabsDashboardData');
  });

  it('keeps initial dashboard refresh scoped to core always-on endpoints', () => {
    const coreFetchSource = sourceBetween('const fetchCoreData = async () => {', 'const fetchOptionalDataForTab');
    const coreEndpoints = [
      '/data/signals.json',
      '/data/dashboard.json',
      '/data/alerts.json',
      '/data/stats.json',
      '/data/health.json',
      '/data/incidents.json',
    ];
    const optionalEndpoints = [
      '/data/analytics.json',
      '/data/rebalance_health.json',
      '/data/explainability/explainability_latest.json',
      '/data/graduation.json',
      '/data/adaptive_sizing.json',
      '/data/vixy_hedge.json',
      '/data/black_litterman.json',
      '/data/turnover_validator.json',
      '/data/regime_gate.json',
      '/data/tsmom.json',
      '/data/cross_asset_rv.json',
    ];

    for (const endpoint of coreEndpoints) {
      expect(coreFetchSource).toContain(endpoint);
    }
    for (const endpoint of optionalEndpoints) {
      expect(coreFetchSource).not.toContain(endpoint);
    }
  });

  it('guards the deprecated behavioral sentiment overview card behind complete live data', () => {
    const overviewSource = sourceBetween('{/* Signal Panels */}', '</div>\n          </div>\n        </PanelErrorBoundary>');

    expect(source).toContain('function isBehavioralSentimentData(value: unknown): value is BehavioralSentimentData');
    expect(source).toContain('const behavioralSentimentData = isBehavioralSentimentData(signals?.behavioral_sentiment)');
    expect(overviewSource).toContain('{behavioralSentimentData && (');
    expect(overviewSource).toContain('<BehavioralSentimentPanel');
  });

  it('loads optional endpoint groups from the active-tab fetch path only', () => {
    const optionalFetchSource = sourceBetween('const fetchOptionalDataForTab = async (tab: TabType', 'const portfolioValue');

    expect(optionalFetchSource).toContain("if (tab === 'rebalance')");
    expect(optionalFetchSource).toContain('/data/rebalance_health.json');
    expect(optionalFetchSource).toContain("if (tab === 'analytics')");
    expect(optionalFetchSource).toContain('/data/analytics.json');
    expect(optionalFetchSource).toContain('/data/explainability/explainability_latest.json');
    expect(optionalFetchSource).toContain('/data/graduation.json');
    expect(optionalFetchSource).toContain('/data/adaptive_sizing.json');
    expect(optionalFetchSource).toContain('/data/vixy_hedge.json');
    expect(optionalFetchSource).toContain('/data/black_litterman.json');
    expect(optionalFetchSource).toContain('/data/turnover_validator.json');
    expect(optionalFetchSource).toContain('/data/regime_gate.json');
    expect(optionalFetchSource).toContain('/data/tsmom.json');
    expect(optionalFetchSource).toContain('/data/cross_asset_rv.json');
  });

  it('schedules core refresh separately from active-tab optional refresh', () => {
    expect(source).toMatch(/useEffect\(\(\) => \{\s*fetchCoreData\(\);[\s\S]*setInterval\(fetchCoreData, refreshInterval \* 1000\)/);
    expect(source).toMatch(/useEffect\(\(\) => \{\s*fetchOptionalDataForTab\(activeTab\);[\s\S]*setInterval\(\(\) => fetchOptionalDataForTab\(activeTab, true\), refreshInterval \* 1000\)/);
  });

  it('guards overlapping core and optional refreshes with request generations', () => {
    const coreFetchSource = sourceBetween('const fetchCoreData = async () => {', 'const shouldFetchOptionalTab');
    const optionalFetchSource = sourceBetween('const fetchOptionalDataForTab = async (tab: TabType', 'const portfolioValue');

    expect(source).toContain('const coreFetchGeneration = useRef(0);');
    expect(source).toContain('const optionalFetchGenerations = useRef<Partial<Record<TabType, number>>>({});');
    expect(coreFetchSource).toContain('const requestGeneration = ++coreFetchGeneration.current;');
    expect(coreFetchSource).toContain('if (requestGeneration !== coreFetchGeneration.current) return;');
    expect(coreFetchSource).toContain('if (requestGeneration === coreFetchGeneration.current) setError(');
    expect(optionalFetchSource).toContain('optionalFetchGenerations.current[tab] = requestGeneration;');
    expect(optionalFetchSource).toContain('if (optionalFetchGenerations.current[tab] !== requestGeneration) return;');
  });

  it('renders the always-visible health header from the operations summary helper', () => {
    expect(source).toContain("from './healthOperations'");
    expect(source).toContain('const healthOperationsSummary = health ? summarizeHealthOperations(health) : null;');
    expect(source).toContain('{healthOperationsSummary?.headerText}');
    expect(source).not.toContain('System: {health.system_status}');
    expect(source).not.toContain('${health.cron_jobs.length} jobs');
  });
});
