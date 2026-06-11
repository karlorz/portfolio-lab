import { describe, expect, it } from 'bun:test';
import { readFileSync } from 'fs';

const source = readFileSync('vite.config.ts', 'utf8');

describe('Vite dev server API proxy', () => {
  it('proxies tasker API calls to the local Flask service', () => {
    expect(source).toContain('server:');
    expect(source).toContain("'/api'");
    expect(source).toContain("target: 'http://127.0.0.1:8000'");
    expect(source).toContain('changeOrigin: true');
  });
});
