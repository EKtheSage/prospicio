// The built package as a JavaScript program uses it: `cargo xtask wasm`
// runs this under Node on target/wasm/pkg. The Rust tests in ../tests
// check the numbers against the parity rows; this checks what only
// JavaScript sees: the names, the argument and result types, thrown errors
// and the package's entry points.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..", "..", "..");
const pkg = process.env.PROSPICIO_WASM_PKG ?? join(root, "target", "wasm", "pkg");
const require = createRequire(import.meta.url);
const p = require(pkg);

/** GenIns as columns, from the repository's data file. */
function genins() {
  const rows = readFileSync(join(root, "validation", "data", "genins.csv"), "utf8")
    .split("\n")
    .filter((l) => l && !l.startsWith("#"))
    .slice(1)
    .map((l) => l.split(",").map(Number));
  return {
    origin: rows.map((r) => r[0]),
    development: rows.map((r) => r[1]),
    value: rows.map((r) => r[2]),
  };
}

/** A reference row's expected value. */
function reference(file, pick) {
  const lines = readFileSync(join(root, "validation", "reference", file), "utf8")
    .split("\n")
    .filter((l) => l && !l.startsWith("#"));
  const header = lines[0].split(",");
  for (const line of lines.slice(1)) {
    const row = Object.fromEntries(line.split(",").slice(0, header.length).map((v, i) => [header[i], v]));
    if (pick(row)) return Number(row.expected);
  }
  throw new Error(`no such row in ${file}`);
}

const near = (got, want, rel = 1e-9) =>
  assert.ok(Math.abs(got - want) <= rel * Math.abs(want), `got ${got}, want ${want}`);

test("the package names its version and entry points", () => {
  const manifest = JSON.parse(readFileSync(join(pkg, "package.json"), "utf8"));
  assert.equal(p.version(), manifest.version);
  for (const entry of [manifest.main, manifest.module, manifest.types, manifest.exports["./web"].default]) {
    assert.ok(readFileSync(join(pkg, entry)).length > 0, entry);
  }
});

test("Mack on GenIns matches R ChainLadder", () => {
  const tri = p.Triangle.fromLong(genins());
  assert.deepEqual([...tri.shape], [1, 1, 10, 10]);
  assert.deepEqual(tri.columns, ["value"]);
  assert.equal(tri.origins[0], "2001");
  const cl = p.chainLadder(tri, "value");
  const total = (row) => row.dataset === "genins" && row.method === "chain_ladder" && row.quantity === "total_reserve";
  near(cl.totals.reserve[0], reference("reserving_chainladder_r.csv", total));
  assert.ok(cl.byOrigin.reserve instanceof Float64Array);
  assert.equal(cl.byOrigin.origin.length, 10);
  const mack = p.mack(tri, "value", {});
  const se = (row) => row.dataset === "genins" && row.method === "mack" && row.quantity === "total_standard_error";
  near(mack.totals.standard_error[0], reference("reserving_chainladder_r.csv", se));
  // A constant tail raises every ultimate by its factor.
  const tailed = p.chainLadder(tri, "value", { tail: 1.05 });
  near(tailed.totals.ultimate[0], 1.05 * cl.totals.ultimate[0]);
  tri.free();
});

test("the bootstrap is reproducible by seed and typed arrays go in", () => {
  const g = genins();
  const tri = p.Triangle.fromLong({
    origin: Float64Array.from(g.origin),
    development: Uint32Array.from(g.development),
    values: { paid: Float64Array.from(g.value) },
  });
  const a = p.odpBootstrap(tri, "paid", { nSims: 300, seed: 3 });
  const b = p.odpBootstrap(tri, "paid", { nSims: 300, seed: 3 });
  const c = p.odpBootstrap(tri, "paid", { nSims: 300, seed: 4 });
  assert.deepEqual(a.reserves.dims, ["origin"]);
  assert.equal(a.reserves.nSims, 300);
  assert.deepEqual(a.reserves.total(), b.reserves.total());
  assert.notDeepEqual(a.reserves.total(), c.reserves.total());
  assert.ok(a.reserves.tvar(0.99) >= a.reserves.var(0.99));
  assert.equal(a.reserves.summary().origin.length, 10);
});

test("an error from the core is a thrown Error with its message", () => {
  const tri = p.Triangle.fromLong(genins());
  assert.throws(() => p.chainLadder(tri, "incurred"), (e) => e instanceof Error && /incurred/.test(e.message));
  assert.throws(() => p.odpBootstrap(tri, "value", { nSim: 500 }), /unknown field "nSim"/);
  assert.throws(() => p.Triangle.fromLong({ origin: [2020], development: [12] }), /value/);
  assert.throws(() => p.Dist.fromJson({ family: "lognormal" }), Error);
});

test("a severity, an aggregate, a tower and a price", () => {
  const sev = p.Dist.fromJson({
    format: "risk_rs.distribution", format_version: 1, family: "lognormal", meanlog: 4, sdlog: 1,
  });
  near(sev.mean(), Math.exp(4.5), 1e-14);
  assert.ok(sev.lev(100) < sev.mean() && sev.tvar(0.99) > sev.quantile(0.99));
  const n = p.Count.poisson(5);
  const agg = p.fft(n, sev);
  near(agg.mean(), 5 * sev.mean(), 1e-3);
  assert.ok(agg.probs() instanceof Float64Array && agg.aliasingError < 1e-9);
  const events = p.simulateEvents(n, sev, { nSims: 2000, seed: 1 });
  const tower = p.Tower.fromJson(p.Tower.fromJson(towerJson()).toJson());
  const ceded = tower.apply(events);
  assert.equal(ceded.nSims, 2000);
  assert.ok(ceded.components.length >= 3);
  const curve = p.oep(n, sev, [100, 500, 1000]);
  assert.ok(curve[0] > curve[1] && curve[1] > curve[2]);
  near(p.oep(n, sev, p.occurrencePml(n, sev, [0.01]))[0], 0.01, 1e-9);
  const assets = p.Distortion.fromName("tvar", 0.99);
  const quote = p.priceAtCostOfCapital(events.totals().total(), 0.1, assets);
  near(quote.returnOnCapital, 0.1, 1e-12);
  assert.ok(quote.premium > quote.expectedLoss && quote.assets > quote.premium);
  const wang = p.price(events.totals().total(), p.Distortion.fromName("wang", 0.3), assets);
  assert.ok(wang.premium > wang.expectedLoss);
});

/** A two-layer tower's saved document, written by the Rust core's `Tower::to_json`. */
function towerJson() {
  return readFileSync(join(here, "tower.json"), "utf8");
}
