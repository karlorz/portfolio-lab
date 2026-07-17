import { describe, expect, it } from 'bun:test';
import { join, resolve } from 'path';
import { readFileSync } from 'fs';
import { resolvePublicDataDir } from '../../scripts/fetch-data';

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
});
