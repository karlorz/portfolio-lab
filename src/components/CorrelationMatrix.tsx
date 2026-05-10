import React, { useMemo } from 'react';
import type { PriceData } from '../backtest/engine';

interface CorrelationMatrixProps {
  priceData: PriceData[];
}

export const CorrelationMatrix: React.FC<CorrelationMatrixProps> = ({ priceData }) => {
  const { symbols, matrix } = useMemo(() => {
    // Get daily returns per symbol
    const bySymbol = new Map<string, Map<string, number>>();
    for (const p of priceData) {
      if (!bySymbol.has(p.symbol)) bySymbol.set(p.symbol, new Map());
      bySymbol.get(p.symbol)!.set(p.date, p.price);
    }

    const symbols = Array.from(bySymbol.keys()).sort();
    const dates = new Set<string>();
    for (const symMap of bySymbol.values()) {
      for (const d of symMap.keys()) dates.add(d);
    }
    const sortedDates = Array.from(dates).sort();

    // Compute daily returns
    const returns = new Map<string, number[]>();
    for (const sym of symbols) {
      const symData = bySymbol.get(sym)!;
      const rets: number[] = [];
      for (let i = 1; i < sortedDates.length; i++) {
        const p0 = symData.get(sortedDates[i - 1]);
        const p1 = symData.get(sortedDates[i]);
        if (p0 && p1 && p0 > 0) rets.push((p1 - p0) / p0);
      }
      returns.set(sym, rets);
    }

    // Compute correlation matrix
    const n = symbols.length;
    const matrix: number[][] = [];

    for (let i = 0; i < n; i++) {
      matrix[i] = [];
      for (let j = 0; j < n; j++) {
        if (i === j) { matrix[i][j] = 1; continue; }
        const ri = returns.get(symbols[i])!;
        const rj = returns.get(symbols[j])!;
        const len = Math.min(ri.length, rj.length);
        if (len < 20) { matrix[i][j] = 0; continue; }

        const iSlice = ri.slice(ri.length - len);
        const jSlice = rj.slice(rj.length - len);

        const meanI = iSlice.reduce((a, b) => a + b, 0) / len;
        const meanJ = jSlice.reduce((a, b) => a + b, 0) / len;
        let cov = 0, varI = 0, varJ = 0;
        for (let k = 0; k < len; k++) {
          const di = iSlice[k] - meanI;
          const dj = jSlice[k] - meanJ;
          cov += di * dj;
          varI += di * di;
          varJ += dj * dj;
        }
        matrix[i][j] = varI && varJ ? cov / Math.sqrt(varI * varJ) : 0;
      }
    }

    return { symbols, matrix };
  }, [priceData]);

  // Focus on key symbols only (to keep matrix readable)
  const keySymbols = ['SPY', 'GLD', 'TLT', 'IEF', 'EFA', 'QQQ', 'VBR', 'DBC'];
  const focusSymbols = symbols.filter(s => keySymbols.includes(s));
  const focusIndices = focusSymbols.map(s => symbols.indexOf(s));

  const getColor = (corr: number): string => {
    if (corr >= 0.7) return '#10b981';
    if (corr >= 0.3) return '#22d3ee';
    if (corr >= -0.3) return '#334155';
    if (corr >= -0.7) return '#f59e0b';
    return '#ef4444';
  };

  return (
    <div className="chart-container">
      <h3>Asset Correlation Matrix (Daily Returns)</h3>
      <p style={{ color: '#64748b', fontSize: '0.85rem', marginBottom: 15 }}>
        Green = positive correlation, amber/red = negative, dark = low. Key diversifiers show negative correlation.
      </p>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={{ padding: '8px 12px', color: '#94a3b8', fontSize: '0.8rem' }}></th>
              {focusSymbols.map(s => (
                <th key={s} style={{ padding: '8px 10px', color: '#e2e8f0', fontSize: '0.8rem', fontWeight: 600 }}>{s}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {focusSymbols.map((rowSym, ri) => (
              <tr key={rowSym}>
                <td style={{ padding: '8px 12px', color: '#e2e8f0', fontSize: '0.8rem', fontWeight: 600, textAlign: 'right' }}>{rowSym}</td>
                {focusSymbols.map((colSym, ci) => {
                  const idx = focusIndices[ri];
                  const jdx = focusIndices[ci];
                  const corr = matrix[idx]?.[jdx] ?? 0;
                  return (
                    <td
                      key={colSym}
                      style={{
                        padding: '8px 10px',
                        textAlign: 'center',
                        fontSize: '0.8rem',
                        fontFamily: 'SF Mono, monospace',
                        background: getColor(corr) + '22',
                        color: getColor(corr),
                        fontWeight: ri === ci ? 800 : 400,
                        border: '1px solid #1e293b',
                      }}
                    >
                      {corr.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ display: 'flex', gap: 16, marginTop: 12, justifyContent: 'center', flexWrap: 'wrap' }}>
        <span style={{ color: '#10b981', fontSize: '0.75rem' }}>&#9632; High positive (0.7+)</span>
        <span style={{ color: '#22d3ee', fontSize: '0.75rem' }}>&#9632; Moderate (0.3-0.7)</span>
        <span style={{ color: '#334155', fontSize: '0.75rem' }}>&#9632; Low (-0.3 to 0.3)</span>
        <span style={{ color: '#f59e0b', fontSize: '0.75rem' }}>&#9632; Negative (-0.3 to -0.7)</span>
        <span style={{ color: '#ef4444', fontSize: '0.75rem' }}>&#9632; Strong negative (-0.7+)</span>
      </div>
    </div>
  );
};
