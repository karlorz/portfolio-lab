import { describe, expect, it } from 'bun:test';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { FactorRotationPanel } from '../../src/components/FactorRotationPanel';

describe('FactorRotationPanel production contract', () => {
  it('renders the production selected_factors allocation payload', () => {
    const html = renderToStaticMarkup(React.createElement(FactorRotationPanel, {
      data: {
        selected_factors: ['VLUE', 'VBR'],
        allocation: { VLUE: 0.27, VBR: 0.73 },
        signal_strength: 0.53,
        recommendation: 'Rotate to Value',
      },
    }));

    expect(html).toContain('Factor Rotation');
    expect(html).toContain('Advisory');
    expect(html).toContain('+0.53');
    expect(html).toContain('Rotate to Value');
    expect(html).toContain('VLUE');
    expect(html).toContain('VBR');
    expect(html).toContain('27%');
    expect(html).toContain('73%');
    expect(html).not.toContain('Q+M Score');
    expect(html).not.toContain('toFixed');
  });

  it('labels missing production numeric fields as unavailable', () => {
    const html = renderToStaticMarkup(React.createElement(FactorRotationPanel, {
      data: {
        selected_factors: ['QUAL'],
        recommendation: 'Hold quality sleeve',
      },
    }));

    expect(html).toContain('Unavailable');
    expect(html).toContain('Allocation unavailable');
    expect(html).toContain('QUAL');
    expect(html).not.toContain('NaN');
  });

  it('keeps malformed runtime numeric fields inside the advisory state', () => {
    const html = renderToStaticMarkup(React.createElement(FactorRotationPanel, {
      data: {
        selected_factors: ['VLUE', 'BAD'],
        allocation: { VLUE: 'not-a-number', BAD: null },
        signal_strength: 'bad',
        recommendation: '',
      },
    }));

    expect(html).toContain('Unavailable');
    expect(html).toContain('Recommendation unavailable');
    expect(html).toContain('advisory only');
    expect(html).not.toContain('NaN');
  });
});
