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
  { name: 'ChatPanel', path: './ChatPanel' },
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
  'Chat',
];

function hasStaticValueImport(importPath: string, importedName: string): boolean {
  const escapedPath = importPath.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const escapedName = importedName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const importPattern = new RegExp(`import\\s+(?!type)[^;]*\\b${escapedName}\\b[^;]*from ['"]${escapedPath}['"];`);
  return importPattern.test(source);
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
});
