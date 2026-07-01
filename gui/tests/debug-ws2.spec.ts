import { test } from '@playwright/test';

test('debug WS lifecycle', async ({ page }) => {
  const allLogs: string[] = [];
  page.on('console', (msg) => {
    allLogs.push(`[${msg.type()}] ${msg.text()}`);
  });

  await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' });
  
  // Wait 5 seconds for all StrictMode remounts to settle
  await page.waitForTimeout(5000);

  // Check WS readyState
  const wsState = await page.evaluate(() => {
    // Try to find the WS instance - it's stored in wsRef inside the hook
    // We can't access it directly, but we can check if we can post and receive
    return {
      // Check if there's a WS connection by looking at the DOM
      linkText: document.querySelector('[data-testid="app-root"]')?.textContent?.includes('LINK UP') ? 'UP' : 'DN',
    };
  });
  allLogs.push(`[TEST] After 5s: LINK=${wsState.linkText}`);

  // Post record
  const response = await page.evaluate(async () => {
    const record = {
      timestamp: Date.now(),
      op: 'alloc',
      size: 64,
      stack_hash: 42,
      thread_id: 0,
      result_ptr: '0x1000',
      latency_ns: 100,
      fragmentation_pct: 0.0,
      backend: 'slab'
    };
    const resp = await fetch('http://127.0.0.1:3000/api/telemetry', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify([record]),
    });
    return { status: resp.status, body: await resp.text() };
  });
  allLogs.push(`[TEST] POST: ${JSON.stringify(response)}`);

  // Wait for potential message delivery
  await page.waitForTimeout(5000);

  // Check again
  const finalState = await page.evaluate(() => {
    return {
      linkText: document.querySelector('[data-testid="app-root"]')?.textContent?.includes('LINK UP') ? 'UP' : 'DN',
      recText: document.querySelector('[data-testid="app-root"]')?.textContent?.match(/\d+\s+REC/)?.[0],
    };
  });
  allLogs.push(`[TEST] After POST: LINK=${finalState.linkText} REC=${finalState.recText}`);

  for (const log of allLogs) {
    console.log(log);
  }
});
