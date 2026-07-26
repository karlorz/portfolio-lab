import { describe, expect, it } from 'bun:test';
import { join, resolve } from 'path';
import {
  chmodSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
  rmSync,
  existsSync,
  statSync,
} from 'fs';
import {
  resolvePublicDataDir,
  softMirrorMarketArtifactsToPrivate,
  PRIVATE_MARKET_SOFT_MIRROR_BASENAMES,
  writeJsonAtomic,
} from '../../scripts/fetch-data';

const PROJECT_ROOT = resolve(import.meta.dir, '../..');
const REPO_PUBLIC_DATA = join(PROJECT_ROOT, 'public', 'data');

describe('fetch-data PUBLIC_DATA_DIR resolution', () => {
  it('defaults to repo public/data when PUBLIC_DATA_DIR is unset and no live WWW', () => {
    const dir = resolvePublicDataDir({
      env: {},
      projectRoot: PROJECT_ROOT,
      livePublicDataDir: join(PROJECT_ROOT, 'no-such-live-www'),
    });
    expect(dir).toBe(REPO_PUBLIC_DATA);
  });

  it('prefers live WWW when PUBLIC_DATA_DIR is unset and live tree exists', () => {
    // Use a real existing directory as live SSOT for the test
    const live = join(PROJECT_ROOT, 'data');
    const dir = resolvePublicDataDir({
      env: {},
      projectRoot: PROJECT_ROOT,
      livePublicDataDir: live,
    });
    expect(dir).toBe(resolve(PROJECT_ROOT, live));
  });

  it('honors absolute PUBLIC_DATA_DIR from the environment', () => {
    const custom = '/var/www/portfolio-lab/data';
    const dir = resolvePublicDataDir({
      env: { PUBLIC_DATA_DIR: custom },
      projectRoot: PROJECT_ROOT,
      livePublicDataDir: join(PROJECT_ROOT, 'data'),
    });
    expect(dir).toBe(custom);
  });

  it('expands a relative PUBLIC_DATA_DIR against project root', () => {
    const dir = resolvePublicDataDir({
      env: { PUBLIC_DATA_DIR: 'tmp-public-data' },
      projectRoot: PROJECT_ROOT,
    });
    expect(dir).toBe(join(PROJECT_ROOT, 'tmp-public-data'));
  });

  it('PORTFOLIO_LAB_ALLOW_REPO_PUBLIC_DATA keeps checkout public/data', () => {
    const dir = resolvePublicDataDir({
      env: { PORTFOLIO_LAB_ALLOW_REPO_PUBLIC_DATA: '1' },
      projectRoot: PROJECT_ROOT,
      livePublicDataDir: join(PROJECT_ROOT, 'data'),
    });
    expect(dir).toBe(REPO_PUBLIC_DATA);
  });

  it('source contract: fetch-data resolves DATA_DIR via resolvePublicDataDir / PUBLIC_DATA_DIR', () => {
    const source = readFileSync('scripts/fetch-data.ts', 'utf8');
    expect(source).toContain('export function resolvePublicDataDir');
    expect(source).toContain('PUBLIC_DATA_DIR');
    expect(source).toMatch(/resolvePublicDataDir\s*\(/);
    expect(source).toContain('PORTFOLIO_LAB_LIVE_PUBLIC_DATA_DIR');
    // Must not hardcode the only write root as a static public/data join for main writes.
    expect(source).not.toMatch(
      /const DATA_DIR = join\(import\.meta\.dir, '\.\.', 'public', 'data'\)/,
    );
  });

  it('Batch HY: PRIVATE_MARKET_SOFT_MIRROR_BASENAMES includes prices/yields', () => {
    expect(PRIVATE_MARKET_SOFT_MIRROR_BASENAMES).toContain('prices.json');
    expect(PRIVATE_MARKET_SOFT_MIRROR_BASENAMES).toContain('yields.json');
    expect(PRIVATE_MARKET_SOFT_MIRROR_BASENAMES).toContain('prices_compact.json');
  });

  it('Batch HY: softMirrorMarketArtifactsToPrivate copies basenames @ 0644', () => {
    const tmpBase = join(PROJECT_ROOT, 'data', '.batch_hy_fetch_mirror_tmp');
    const pub = join(tmpBase, 'public');
    const priv = join(tmpBase, 'private');
    try {
      rmSync(tmpBase, { recursive: true, force: true });
      mkdirSync(pub, { recursive: true });
      mkdirSync(priv, { recursive: true });
      writeFileSync(join(pub, 'prices.json'), JSON.stringify({ SPY: [{ d: '2026-07-23', p: 1 }] }));
      writeFileSync(join(pub, 'yields.json'), JSON.stringify([{ date: '2026-07-23', DGS10: 4.2 }]));
      const report = softMirrorMarketArtifactsToPrivate({
        publicRoot: pub,
        privateRoot: priv,
        basenames: ['prices.json', 'yields.json'],
      });
      expect(report.errors).toEqual([]);
      expect(report.copied.sort()).toEqual(['prices.json', 'yields.json']);
      expect(readFileSync(join(priv, 'prices.json'), 'utf8')).toContain('SPY');
      expect((statSync(join(priv, 'prices.json')).mode & 0o777)).toBe(0o644);
    } finally {
      rmSync(tmpBase, { recursive: true, force: true });
    }
  });

  it('Batch HY: softMirror skips ephemeral plab-pytest public roots', () => {
    const tmpBase = join(PROJECT_ROOT, 'data', '.batch_hy_ephemeral_tmp');
    const pub = join(tmpBase, 'plab-pytest-public.hy', 'data');
    const priv = join(tmpBase, 'private');
    try {
      rmSync(tmpBase, { recursive: true, force: true });
      mkdirSync(pub, { recursive: true });
      mkdirSync(priv, { recursive: true });
      writeFileSync(join(pub, 'prices.json'), '{}');
      const report = softMirrorMarketArtifactsToPrivate({
        publicRoot: pub,
        privateRoot: priv,
        basenames: ['prices.json'],
      });
      expect(report.skipped).toContain('ephemeral-public-root');
      expect(report.copied).toEqual([]);
      expect(existsSync(join(priv, 'prices.json'))).toBe(false);
    } finally {
      rmSync(tmpBase, { recursive: true, force: true });
    }
  });

  it('Batch HY: writeJsonAtomic leaves mode 0644', async () => {
    const tmpBase = join(PROJECT_ROOT, 'data', '.batch_hy_atomic_tmp');
    const path = join(tmpBase, 'sample.json');
    try {
      rmSync(tmpBase, { recursive: true, force: true });
      mkdirSync(tmpBase, { recursive: true });
      await writeJsonAtomic(path, { ok: true });
      expect((statSync(path).mode & 0o777)).toBe(0o644);
    } finally {
      rmSync(tmpBase, { recursive: true, force: true });
    }
  });

  it('source contract: main soft-mirrors market artifacts to private data/', () => {
    const source = readFileSync('scripts/fetch-data.ts', 'utf8');
    expect(source).toContain('softMirrorMarketArtifactsToPrivate');
    expect(source).toContain('PRIVATE_MARKET_SOFT_MIRROR_BASENAMES');
    expect(source).toContain('chmodSync');
  });

  it('source contract: main rebuilds vix_term_structure.json from market.db after sync', () => {
    // Without this call the derived VIX history file freezes at the last
    // manual generation while market.db stays hourly-fresh; the term-structure
    // signal then reads stale levels (FILE_STALE_DAYS fallback never trips).
    const source = readFileSync('scripts/fetch-data.ts', 'utf8');
    expect(source).toContain('update_vix_term_structure');
    const syncIdx = source.indexOf("runPythonModule('src.data.market_db_sync')");
    const vixIdx = source.indexOf('update_vix_term_structure');
    const dashIdx = source.indexOf('await runDashboardGeneration()');
    expect(syncIdx).toBeGreaterThan(-1);
    expect(vixIdx).toBeGreaterThan(syncIdx);
    expect(dashIdx).toBeGreaterThan(vixIdx);
  });
});
