import React from 'react';
import type { PortfolioConfig } from '../backtest/engine';

interface PortfolioSelectorProps {
  portfolios: PortfolioConfig[];
  selected: string[];
  onToggle: (name: string) => void;
  colors: string[];
}

const CATEGORIES = [
  { label: 'Benchmarks', test: (n: string) => n.includes('(S&P 500)') || n.includes('(Nasdaq-100)') },
  { label: 'Traditional', test: (n: string) => n.startsWith('60/40') },
  { label: 'All-Weather', test: (n: string) => n.startsWith('All Weather') || n.startsWith('Golden Butterfly') },
  { label: '★ Winners', test: (n: string) => n.includes('★') || (n.includes('SPY/') && !n.includes('+')) },
  { label: 'Overlays', test: (n: string) => n.includes('+') },
  { label: 'International', test: (n: string) => n.includes('EFA') || n.includes('VXUS') },
];

export const PortfolioSelector: React.FC<PortfolioSelectorProps> = ({ portfolios, selected, onToggle, colors }) => {
  const uncategorized = new Set(portfolios.map(p => p.name));

  const categorized = CATEGORIES.map(cat => {
    const items = portfolios.filter(p => cat.test(p.name));
    items.forEach(p => uncategorized.delete(p.name));
    return { label: cat.label, portfolios: items };
  }).filter(c => c.portfolios.length > 0);

  // Add any remaining as "Other"
  const other = portfolios.filter(p => uncategorized.has(p.name));
  if (other.length > 0) categorized.push({ label: 'Other', portfolios: other });

  return (
    <div className="portfolio-selector">
      <h3>Select Portfolios to Compare</h3>
      <div className="portfolio-categories">
        {categorized.map(cat => (
          <div key={cat.label} className="portfolio-category">
            <div className="category-label">{cat.label}</div>
            <div className="portfolio-toggles">
              {cat.portfolios.map(p => {
                const i = portfolios.indexOf(p);
                return (
                  <label key={p.name} className="toggle">
                    <input
                      type="checkbox"
                      checked={selected.includes(p.name)}
                      onChange={() => onToggle(p.name)}
                    />
                    <span style={{ color: colors[i % colors.length] }}>{p.name}</span>
                    <small style={{ color: '#64748b', marginLeft: 4 }}>
                      ({Object.entries(p.allocation).map(([k, v]) => `${k} ${(v*100).toFixed(0)}%`).join(', ')})
                    </small>
                  </label>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
