import { test, expect } from '@playwright/test';

test.describe('Lohalloc GUI E2E', () => {
  test('page loads and captures console', async ({ page }) => {
    const consoleErrors: string[] = [];
    const consoleWarnings: string[] = [];

    page.on('console', (msg) => {
      if (msg.type() === 'error') consoleErrors.push(msg.text());
      if (msg.type() === 'warning') consoleWarnings.push(msg.text());
    });

    page.on('pageerror', (err) => {
      consoleErrors.push(`PAGE ERROR: ${err.message}`);
    });

    await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' });
    await page.waitForTimeout(3000);

    console.log('Console errors:', JSON.stringify(consoleErrors, null, 2));
    console.log('Console warnings:', JSON.stringify(consoleWarnings, null, 2));

    const appRoot = page.locator('[data-testid="app-root"]');
    await expect(appRoot).toBeVisible();

    const connText = await page.locator('[data-testid="app-root"]').textContent();
    console.log('Connection status:', connText?.includes('LINK UP') ? 'LINK UP' : 'LINK DN');

    await page.screenshot({ path: 'e2e-screenshot-initial.png', fullPage: true });
  });

  test('telemetry POST + WS stream populates visuals', async ({ page }) => {
    const consoleErrors: string[] = [];

    page.on('console', (msg) => {
      if (msg.type() === 'error') consoleErrors.push(msg.text());
    });

    await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' });
    await page.waitForTimeout(3000);

    const connTextBefore = await page.locator('[data-testid="app-root"]').textContent();
    console.log('Before POST - LINK:', connTextBefore?.includes('LINK UP') ? 'UP' : 'DN');

    const records = Array.from({ length: 20 }, (_, i) => ({
      timestamp: i + 1,
      op: i % 3 === 0 ? 'free' : 'alloc',
      size: [64, 128, 256, 512, 1024][i % 5],
      stack_hash: 100 + (i % 8),
      thread_id: 0,
      result_ptr: `0x${(0x1000 + i * 64).toString(16)}`,
      latency_ns: 50 + i * 10,
      fragmentation_pct: i * 0.5,
      backend: ['slab', 'buddy', 'system'][i % 3],
    }));

    const response = await page.evaluate(async (recs) => {
      const resp = await fetch('http://127.0.0.1:3000/api/telemetry', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(recs),
      });
      return { status: resp.status, body: await resp.text() };
    }, records);

    console.log('POST response:', JSON.stringify(response));
    await page.waitForTimeout(3000);

    const connTextAfter = await page.locator('[data-testid="app-root"]').textContent();
    console.log('After POST - LINK:', connTextAfter?.includes('LINK UP') ? 'UP' : 'DN');
    console.log('After POST, REC count:', connTextAfter?.match(/\d+\s+REC/)?.[0]);

    const telemetryPane = page.locator('[data-testid="telemetry-pane"]');
    const telemetryText = await telemetryPane.textContent();
    console.log('Telemetry pane text (first 200):', telemetryText?.substring(0, 200));

    await page.screenshot({ path: 'e2e-screenshot-after-post.png', fullPage: true });
  });

  test('simulation spawn works and shows events', async ({ page }) => {
    await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' });
    await page.waitForTimeout(3000);

    const simulateBtn = page.locator('button:has-text("SIMULATE")');
    await simulateBtn.click();
    await page.waitForTimeout(500);

    const lohallocOption = page.locator('button:has-text("LOHALLOC EXAMPLE")');
    await expect(lohallocOption).toBeVisible();
    await lohallocOption.click();
    await page.waitForTimeout(5000);

    const recText = await page.locator('[data-testid="app-root"]').textContent();
    console.log('After sim, REC count:', recText?.match(/\d+\s+REC/)?.[0]);
    console.log('After sim, LINK:', recText?.includes('LINK UP') ? 'UP' : 'DN');

    await page.screenshot({ path: 'e2e-screenshot-sim.png', fullPage: true });
  });
});
