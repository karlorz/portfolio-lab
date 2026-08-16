import { describe, expect, it } from 'bun:test';
import { join } from 'path';
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import { createFredDiskCache } from '../../scripts/fetch-data';

type CacheEntry = {
  series_id: string;
  start_date: string;
  end_date: string;
  fetched_at: string;
  observations: { date: string; value: number }[];
};

function seedCache(cachePath: string, series: string[], endDates: string[]): void {
  const records: Record<string, CacheEntry> = {};
  for (const seriesId of series) {
    for (const endDate of endDates) {
      records[`${seriesId}:2005-01-01:${endDate}`] = {
        series_id: seriesId,
        start_date: '2005-01-01',
        end_date: endDate,
        fetched_at: '2026-08-14T00:00:00.000Z',
        observations: [],
      };
    }
  }
  writeFileSync(cachePath, JSON.stringify(records));
}

function endDatesFor(cachePath: string, seriesId: string): string[] {
  const records = JSON.parse(readFileSync(cachePath, 'utf8')) as Record<string, CacheEntry>;
  return Object.values(records)
    .filter((record) => record.series_id === seriesId)
    .map((record) => record.end_date)
    .sort();
}

const TODAY = { seriesId: 'S1', startDate: '2005-01-01', endDate: '2026-08-16' };

describe('fred cache eviction', () => {
  it('keeps only the 2 newest end-dates per series after set (oldest dropped, today present)', async () => {
    const tmp = mkdtempSync(join(tmpdir(), 'plab-fred-evict-'));
    try {
      const cachePath = join(tmp, 'fred.json');
      seedCache(
        cachePath,
        ['S1', 'S2', 'S3'],
        ['2026-08-10', '2026-08-11', '2026-08-12', '2026-08-13'],
      );
      const cache = createFredDiskCache(cachePath);
      await cache.set(TODAY, { fetched_at: '2026-08-16T10:00:00.000Z', observations: [] });
      expect(endDatesFor(cachePath, 'S1')).toEqual(['2026-08-13', '2026-08-16']);
      expect(endDatesFor(cachePath, 'S2')).toEqual(['2026-08-12', '2026-08-13']);
      expect(endDatesFor(cachePath, 'S3')).toEqual(['2026-08-12', '2026-08-13']);
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  });

  it('prunes series not in the new fetch down to their own newest-2 only', async () => {
    const tmp = mkdtempSync(join(tmpdir(), 'plab-fred-evict-'));
    try {
      const cachePath = join(tmp, 'fred.json');
      seedCache(cachePath, ['S2', 'S3'], ['2026-08-10', '2026-08-14', '2026-08-15', '2026-08-17']);
      const cache = createFredDiskCache(cachePath);
      await cache.set(TODAY, { fetched_at: '2026-08-16T10:00:00.000Z', observations: [] });
      // S2/S3 were untouched by the S1 fetch but were reduced to their own
      // newest 2 (cross-series independence: no S1 interference).
      expect(endDatesFor(cachePath, 'S2')).toEqual(['2026-08-15', '2026-08-17']);
      expect(endDatesFor(cachePath, 'S3')).toEqual(['2026-08-15', '2026-08-17']);
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  });

  it('malformed cache file: set no-throw and writes the fresh record', async () => {
    const tmp = mkdtempSync(join(tmpdir(), 'plab-fred-evict-'));
    try {
      const cachePath = join(tmp, 'fred.json');
      writeFileSync(cachePath, '{not valid json');
      const cache = createFredDiskCache(cachePath);
      await cache.set(TODAY, { fetched_at: '2026-08-16T10:00:00.000Z', observations: [] });
      expect(endDatesFor(cachePath, 'S1')).toEqual(['2026-08-16']);
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  });

  it('absent cache file: first set writes cleanly', async () => {
    const tmp = mkdtempSync(join(tmpdir(), 'plab-fred-evict-'));
    try {
      const cachePath = join(tmp, 'fred.json');
      expect(existsSync(cachePath)).toBe(false);
      const cache = createFredDiskCache(cachePath);
      await cache.set(TODAY, { fetched_at: '2026-08-16T10:00:00.000Z', observations: [] });
      expect(existsSync(cachePath)).toBe(true);
      expect(endDatesFor(cachePath, 'S1')).toEqual(['2026-08-16']);
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  });
});