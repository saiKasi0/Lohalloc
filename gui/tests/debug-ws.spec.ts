import { test } from '@playwright/test';

test('debug WS onmessage', async ({ page }) => {
  const allLogs: string[] = [];
  page.on('console', (msg) => {
    allLogs.push(`[${msg.type()}] ${msg.text()}`);
  });
  page.on('pageerror', (err) => {
    allLogs.push(`[PAGE_ERROR] ${err.message}`);
  });

  await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(4000); // Wait for StrictMode remount + reconnect

  // Post a record
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

  allLogs.push(`[TEST] POST response: ${JSON.stringify(response)}`);
  await page.waitForTimeout(3000);

  // Print all console logs
  for (const log of allLogs) {
    console.log(log);
  }
});
