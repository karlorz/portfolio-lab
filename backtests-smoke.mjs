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

// Item 19: Design Guide Interactive Playground User Interaction Smoke
const playground = page.locator('.design-guide-section').first();
const routingSelect = playground.locator('select').nth(0);
const killSelect = playground.locator('select').nth(1);
const regimeSelect = playground.locator('select').nth(2);
const toneSelect = playground.locator('select').nth(3);

// 1. Test Authority selection (Advisory)
await routingSelect.selectOption('false');
await page.waitForTimeout(200);
const authBadgeAdvisory = await playground.locator('.authority-badge').first().innerText();
check('playground routing select updates authority badge (advisory)', authBadgeAdvisory.toLowerCase().includes('not routed') || authBadgeAdvisory.toLowerCase().includes('advisory'));

// 2. Test Kill Switch selection (Halt)
await killSelect.selectOption('halt');
await page.waitForTimeout(200);
const authBadgeBlocked = await playground.locator('.authority-badge').first().innerText();
check('playground kill switch select triggers blocked authority badge', authBadgeBlocked.toLowerCase().includes('routing blocked') || authBadgeBlocked.toLowerCase().includes('blocked'));

// 3. Test Tone selection
await toneSelect.selectOption('critical');
await page.waitForTimeout(200);
const statusBadgeText = await playground.locator('.control-status').first().innerText();
check('playground status tone select updates badge to critical', statusBadgeText.includes('CRITICAL'));

// 4. Test Sliders
const spySlider = playground.locator('input[type="range"]').nth(0);
await spySlider.fill('70');
await page.waitForTimeout(200);
const sliderLabel = await playground.innerText();
check('playground SPY weight slider updates percentage label', sliderLabel.includes('SPY Weight: 70%'));

// 5. Test Command Palette interactive modal
const paletteBtn = playground.locator('button:has-text("Open Command Palette")');
await paletteBtn.click();
await page.waitForSelector('.command-palette, [role="dialog"], input[placeholder*="Search"]', { timeout: 5000 });
check('playground opens command palette modal', true);
// Close palette via Escape key
await page.keyboard.press('Escape');
await page.waitForTimeout(300);
const paletteCount = await page.locator('.command-palette, [role="dialog"]').count();
check('playground closes command palette on escape', paletteCount === 0);

console.log(results.join('\n'));
console.log('pageerrors:', errors.length ? errors.join(' | ') : 'none');
await browser.close();
process.exit(process.exitCode ?? 0);