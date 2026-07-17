import { describe, expect, it } from 'bun:test';
import { join, resolve } from 'path';
import { readFileSync } from 'fs';
import { resolvePublicDataDir } from '../../scripts/fetch-data';

const PROJECT_ROOT = resolve(import.meta.dir, '../..');
const REPO_PUBLIC_DATA = join(PROJECT_ROOT, 'public', 'data');

describe('fetch-data PUBLIC_DATA_DIR resolution', () => {
  it('defaults to repo public/data when PUBLIC_DATA_DIR is unset', () => {
    const dir = resolvePublicDataDir({
      env: {},
      projectRoot: PROJECT_ROOT,
    });
    expect(dir).toBe(REPO_PUBLIC_DATA);
  });

  it('honors absolute PUBLIC_DATA_DIR from the environment', () => {
    const custom = '/var/www/portfolio-lab/data';
    const dir = resolvePublicDataDir({
      env: { PUBLIC_DATA_DIR: custom },
      projectRoot: PROJECT_ROOT,
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

  it('source contract: fetch-data resolves DATA_DIR via resolvePublicDataDir / PUBLIC_DATA_DIR', () => {
    const source = readFileSync('scripts/fetch-data.ts', 'utf8');
    expect(source).toContain('export function resolvePublicDataDir');
    expect(source).toContain('PUBLIC_DATA_DIR');
    expect(source).toMatch(/resolvePublicDataDir\s*\(/);
    // Must not hardcode the only write root as a static public/data join for main writes.
    expect(source).not.toMatch(
      /const DATA_DIR = join\(import\.meta\.dir, '\.\.', 'public', 'data'\)/,
    );
  });
});
