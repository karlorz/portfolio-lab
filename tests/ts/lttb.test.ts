import { describe, it, expect } from 'bun:test';
import { readFileSync } from 'fs';
import { downsampleLTTB, autoDownsample } from '../../src/utils/lttb';

/** Helper: generate a sine wave as DataPoint[] */
function sineWave(n: number, freq: number = 1): Array<{ date: number; value: number }> {
  return Array.from({ length: n }, (_, i) => ({
    date: i,
    value: Math.sin((2 * Math.PI * freq * i) / n),
  }));
}

/** Helper: monotonic linear data */
function linearData(n: number): Array<{ date: number; value: number }> {
  return Array.from({ length: n }, (_, i) => ({ date: i, value: i }));
}

describe('downsampleLTTB', () => {
  it('returns data as-is when length <= threshold', () => {
    const data = sineWave(10);
    const result = downsampleLTTB(data, 50, 'date', 'value');
    expect(result).toBe(data); // same reference, no copy
  });

  it('returns data as-is when threshold < 3', () => {
    const data = sineWave(100);
    const result = downsampleLTTB(data, 2, 'date', 'value');
    expect(result).toBe(data);
  });

  it('preserves first and last points', () => {
    const data = sineWave(1000);
    const result = downsampleLTTB(data, 100, 'date', 'value');
    expect(result[0]).toEqual(data[0]);
    expect(result[result.length - 1]).toEqual(data[data.length - 1]);
  });

  it('produces exactly threshold points', () => {
    const data = sineWave(2000);
    const result = downsampleLTTB(data, 100, 'date', 'value');
    expect(result.length).toBe(100);
  });

  it('preserves monotonicity for linear data', () => {
    const data = linearData(2000);
    const result = downsampleLTTB(data, 50, 'date', 'value');
    for (let i = 1; i < result.length; i++) {
      expect(result[i].value).toBeGreaterThan(result[i - 1].value);
    }
  });

  it('preserves overall shape of sine wave', () => {
    const data = sineWave(2000, 3);
    const result = downsampleLTTB(data, 200, 'date', 'value');
    // Find max and min in both — should be close to 1 and -1
    const origMax = Math.max(...data.map(d => d.value));
    const origMin = Math.min(...data.map(d => d.value));
    const sampMax = Math.max(...result.map(d => d.value));
    const sampMin = Math.min(...result.map(d => d.value));
    expect(sampMax).toBeCloseTo(origMax, 1);
    expect(sampMin).toBeCloseTo(origMin, 1);
  });

  it('handles NaN values gracefully', () => {
    const data: Array<{ date: number; value: number }> = Array.from({ length: 1000 }, (_, i) => ({
      date: i,
      value: i === 500 ? NaN : Math.sin((2 * Math.PI * i) / 1000),
    }));
    const result = downsampleLTTB(data, 100, 'date', 'value');
    // Should still produce output — NaN points get skipped in averages
    expect(result.length).toBeGreaterThan(0);
  });

  it('works with string date keys', () => {
    const data = Array.from({ length: 500 }, (_, i) => ({
      date: new Date(2020, 0, 1 + i).toISOString().split('T')[0],
      value: Math.sin((2 * Math.PI * i) / 500),
    }));
    const result = downsampleLTTB(data, 50, 'date', 'value');
    expect(result.length).toBe(50);
    expect(result[0].date).toBe(data[0].date);
    expect(result[result.length - 1].date).toBe(data[data.length - 1].date);
  });
});

describe('autoDownsample', () => {
  it('returns data as-is when below minSize', () => {
    const data = sineWave(500);
    const result = autoDownsample(data, 100, 'date', 'value', 1000);
    expect(result).toBe(data);
  });

  it('applies LTTB when above minSize', () => {
    const data = sineWave(2000);
    const result = autoDownsample(data, 100, 'date', 'value', 1000);
    expect(result.length).toBe(100);
    expect(result).not.toBe(data);
  });

  it('uses default parameters correctly', () => {
    // Default: threshold=500, xKey="date", yKey="value", minSize=1000
    const data = sineWave(1500);
    const result = autoDownsample(data);
    expect(result.length).toBe(500);
  });

  it('preserves first and last when applied', () => {
    const data = sineWave(2000);
    const result = autoDownsample(data, 100, 'date', 'value', 1000);
    expect(result[0]).toEqual(data[0]);
    expect(result[result.length - 1]).toEqual(data[data.length - 1]);
  });
});

describe('dashboard chart data transforms', () => {
  it('RollingMetricsChart does not merge windows with repeated find scans', () => {
    const source = readFileSync('src/components/AnalyticsCharts.tsx', 'utf8');
    const start = source.indexOf('export const RollingMetricsChart');
    const end = source.indexOf('interface CrisisPeriod');
    const rollingMetricsSource = source.slice(start, end);

    expect(rollingMetricsSource).not.toContain('.find(');
  });
});
