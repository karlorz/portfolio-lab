import { describe, expect, it } from 'bun:test';
import { execFileSync } from 'child_process';
import { existsSync, readFileSync } from 'fs';
import { resolve } from 'path';

const REPO_ROOT = resolve(import.meta.dir, '../..');

function gitLsFiles(pathspec: string): string[] {
  const out = execFileSync('git', ['ls-files', '--', pathspec], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
  });
  return out
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
}

describe('public/data runtime artifacts are not git-tracked', () => {
  it('keeps /public/data/ in .gitignore', () => {
    const gitignore = readFileSync(resolve(REPO_ROOT, '.gitignore'), 'utf8');
    expect(gitignore).toMatch(/^\/public\/data\/$/m);
  });

  it('does not track prices.json or historical.json', () => {
    const tracked = gitLsFiles('public/data/');
    expect(tracked).not.toContain('public/data/prices.json');
    expect(tracked).not.toContain('public/data/historical.json');
    // Guard: nothing under public/data should re-enter the index.
    expect(tracked).toEqual([]);
  });

  it('keeps local prices.json on disk for offline use when present', () => {
    // Not required in every clone (gitignore), but present on lab host.
    // Soft assert: if missing, skip — only fail if somehow a tracked copy
    // was required by the previous test (it must not be).
    if (existsSync(resolve(REPO_ROOT, 'public/data/prices.json'))) {
      expect(gitLsFiles('public/data/prices.json')).toEqual([]);
    }
  });
});
