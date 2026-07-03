import { test, expect } from '@playwright/test';

/**
 * Full SIMULATE-button-matrix E2E coverage, against the REAL running stack
 * (lohalloc-server + Vite), not mocks. This is the regression guard for the
 * original blank-screen bug: clicking any preset must never unmount
 * `app-root`, and the simulation panel must open and show the spawn.
 *
 * Requires `cargo run -p lohalloc-server` running on :3000 separately (see
 * CLAUDE.md's two-terminal dev workflow) — `npm run dev` alone only starts
 * Vite. If the server isn't reachable, every test in this file is skipped
 * with an explanatory message rather than failing on a confusing timeout.
 */

const SERVER_URL = 'http://127.0.0.1:3000';

const KINDS: Array<{ kind: string; label: string }> = [
  { kind: 'lohalloc-example', label: 'LOHALLOC EXAMPLE' },
  { kind: 'long-running', label: 'LONG RUNNING' },
  { kind: 'stress-test', label: 'STRESS TEST' },
  { kind: 'high-churn', label: 'HIGH-FREQUENCY CHURN' },
  { kind: 'checkerboard', label: 'CHECKERBOARD FRAGMENTATION' },
  { kind: 'mixed-workload', label: 'MIXED WORKLOADS' },
];

let serverUp = false;

test.describe('SIMULATE dropdown — full button matrix', () => {
  test.beforeAll(async ({ request }) => {
    try {
      const res = await request.get(`${SERVER_URL}/health`, { timeout: 2000 });
      serverUp = res.ok();
    } catch {
      serverUp = false;
    }
  });

  test.beforeEach(() => {
    test.skip(
      !serverUp,
      `lohalloc-server not reachable at ${SERVER_URL} — start it with ` +
        `"cargo run -p lohalloc-server" before running e2e tests`,
    );
  });

  for (const { kind, label } of KINDS) {
    test(`clicking ${label} spawns ${kind} without blanking the dashboard`, async ({
      page,
    }) => {
      await page.goto('/');
      const appRoot = page.locator('[data-testid="app-root"]');
      await expect(appRoot).toBeVisible();

      await page.locator('button', { hasText: 'SIMULATE' }).first().click();
      await page.locator('button', { hasText: label }).click();

      // The blank-screen regression: app-root must stay mounted through the
      // resetState()+spawn cycle handleSpawn triggers, and the panel must
      // actually open — not just "the request didn't throw".
      await expect(appRoot).toBeVisible();
      await expect(page.getByText('[SIMULATIONS', { exact: false })).toBeVisible({
        timeout: 10000,
      });

      // Clean up so this kind's subprocess doesn't linger into the next
      // test case.
      const killAll = page.locator('[data-testid="kill-all-sims"]');
      if (await killAll.isVisible().catch(() => false)) {
        await killAll.click();
      }
    });
  }
});

test.describe('Freeze / Export controls', () => {
  test.beforeAll(async ({ request }) => {
    try {
      const res = await request.get(`${SERVER_URL}/health`, { timeout: 2000 });
      serverUp = res.ok();
    } catch {
      serverUp = false;
    }
  });

  test.beforeEach(() => {
    test.skip(
      !serverUp,
      `lohalloc-server not reachable at ${SERVER_URL} — start it with ` +
        `"cargo run -p lohalloc-server" before running e2e tests`,
    );
  });

  test('FREEZE flips the pane to CollapsedTopology without blanking, and does not auto-download', async ({
    page,
  }) => {
    await page.goto('/');
    const appRoot = page.locator('[data-testid="app-root"]');
    await expect(appRoot).toBeVisible();

    const freezeBtn = page.locator('[data-testid="freeze-btn"]');
    // A shared long-lived server may already be in inference mode from a
    // prior run — only exercise the transition if training mode is active.
    if (!(await freezeBtn.isVisible().catch(() => false))) {
      test.skip(true, 'server already in inference mode — nothing to freeze');
      return;
    }

    const downloadPromise = page
      .waitForEvent('download', { timeout: 2000 })
      .catch(() => null);

    await freezeBtn.click();

    await expect(appRoot).toBeVisible();
    await expect(page.locator('[data-testid="collapsed-topology"]')).toBeVisible({
      timeout: 10000,
    });
    await expect(freezeBtn).toBeHidden();

    // Strict state-freeze: clicking FREEZE must not trigger a file download
    // (that's what the separate EXPORT button is for).
    expect(await downloadPromise).toBeNull();
  });
});

test.describe('Perf graph accumulates points across a run (no rewind)', () => {
  test.beforeAll(async ({ request }) => {
    try {
      const res = await request.get(`${SERVER_URL}/health`, { timeout: 2000 });
      serverUp = res.ok();
    } catch {
      serverUp = false;
    }
  });

  test.beforeEach(() => {
    test.skip(
      !serverUp,
      `lohalloc-server not reachable at ${SERVER_URL} — start it with ` +
        `"cargo run -p lohalloc-server" before running e2e tests`,
    );
  });

  test('LATENCY panel point count grows over time instead of resetting toward 0', async ({
    page,
  }) => {
    await page.goto('/');
    await page.locator('button', { hasText: 'SIMULATE' }).first().click();
    await page.locator('button', { hasText: 'LOHALLOC EXAMPLE' }).click();

    const perfPane = page.locator('[data-testid="perf-pane"]');
    await expect(perfPane).toBeVisible();

    const readPointCount = async (): Promise<number> => {
      const text = await perfPane.locator('text=/\\d+ PT/').first().textContent();
      const match = text?.match(/(\d+)\s*PT/);
      return match ? parseInt(match[1], 10) : 0;
    };

    await page.waitForTimeout(2000);
    const first = await readPointCount();

    await page.waitForTimeout(3000);
    const second = await readPointCount();

    // This is the regression this fix targets: computePerfPoints used to
    // rebase to a shifting records[0], so the point count / graph could
    // visually collapse back toward 0 instead of growing.
    expect(second).toBeGreaterThanOrEqual(first);

    const killAll = page.locator('[data-testid="kill-all-sims"]');
    if (await killAll.isVisible().catch(() => false)) {
      await killAll.click();
    }
  });
});
