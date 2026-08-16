import { chromium } from 'playwright';

const BASE = process.env.SMOKE_BASE || 'http://127.0.0.1:4173';
const results = [];
function check(name, ok) {
  results.push(`${ok ? 'PASS' : 'FAIL'} ${name}`);
  if (!ok) process.exitCode = 1;
}

const browser = await chromium.launch();
const page = await browser.newPage();
const errors = [];
page.on('pageerror', e => errors.push(String(e)));

// Backtests workspace renders and runs
await page.goto(`${BASE}/?view=backtests`, { waitUntil: 'domcontentloaded' });
await page.waitForSelector('.backtests-workspace', { timeout: 30000 });
check('workspace renders', true);
await page.waitForSelector('.portfolio-selector', { timeout: 15000 });
check('PortfolioSelector renders', true);
await page.waitForSelector('.metrics-grid', { timeout: 90000 });
check('backtest ran to results (metrics grid)', true);
const cards = await page.locator('.metric-card').count();
check(`metrics cards present (${cards})`, cards >= 1);
await page.waitForSelector('.chart-container svg', { timeout: 60000 });
check('EquityCurve chart renders', true);
await page.waitForSelector('.chart-row, .chart-loading', { timeout: 60000 });
await page.waitForSelector('.chart-row svg, .chart-row .chart-loading', { timeout: 60000 }).catch(() => {});
await page.waitForSelector('.backtests-footer', { timeout: 15000 });
const footerText = await page.locator('.backtests-footer').innerText();
check('footer has Highest Sharpe summary', footerText.includes('Highest Sharpe ratio'));
// PortfolioSelector toggle works
const toggles = page.locator('.portfolio-toggles .toggle');
const checkedBefore = await toggles.locator('input:checked, [aria-pressed="true"]').count();
await toggles.first().click();
await page.waitForTimeout(500);
const checkedAfter = await toggles.locator('input:checked, [aria-pressed="true"]').count();
check(`toggle toggled (${checkedBefore} -> ${checkedAfter})`, checkedBefore !== checkedAfter);
// Backtests workspace must NOT appear on the live view
await page.goto(`${BASE}/`, { waitUntil: 'domcontentloaded' });
await page.waitForTimeout(1500);
const wsOnLive = await page.locator('.backtests-workspace').count();
check('backtests workspace absent on live view', wsOnLive === 0);
// Lazy DesignGuidePage route
await page.goto(`${BASE}/design-guide`, { waitUntil: 'domcontentloaded' });
await page.waitForSelector('.design-guide-page', { timeout: 30000 });
check('design guide lazy route renders', true);
await page.waitForSelector('.allocation-spine', { timeout: 15000 });
check('interactive playground and allocation spine render', true);

console.log(results.join('\n'));
console.log('pageerrors:', errors.length ? errors.join(' | ') : 'none');
await browser.close();
process.exit(process.exitCode ?? 0);