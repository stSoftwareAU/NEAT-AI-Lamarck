/**
 * Optional regenerator for Lamarck backprop golden fixtures (issue #2).
 *
 * Preferred CI path: commit expected.json produced by
 *   LAMARCK_REGEN_BACKPROP_FIXTURES=1 cargo test -p neat_ai_lamarck --test backprop_parity
 * which runs the neat-core `propagate_topological_loop` path (TS/WASM contract).
 *
 * This Deno script is a placeholder for refreshing goldens from sibling NEAT-AI
 * TypeScript when a dual-run harness is available:
 *
 *   deno run -A scripts/generate_backprop_parity_fixtures.ts
 *
 * Requires ../NEAT-AI checked out beside this repo.
 */

const FIXTURES = new URL("../lamarck/tests/fixtures/backprop/", import.meta.url);

async function main() {
  const neatAi = new URL("../../NEAT-AI/", import.meta.url);
  try {
    await Deno.stat(neatAi);
  } catch {
    console.error(
      "Sibling NEAT-AI not found. Use LAMARCK_REGEN_BACKPROP_FIXTURES=1 cargo test instead.",
    );
    Deno.exit(1);
  }

  console.log("NEAT-AI found at", neatAi.pathname);
  console.log("Fixture root:", FIXTURES.pathname);
  console.log(
    "Dual-run TS↔Rust golden export is not wired yet; regenerate with:",
  );
  console.log(
    "  LAMARCK_REGEN_BACKPROP_FIXTURES=1 cargo test -p neat_ai_lamarck --test backprop_parity",
  );
  console.log(
    "(Goldens exercise neat-core propagate_topological_loop — the TS/WASM behavioural port.)",
  );
}

if (import.meta.main) {
  await main();
}
