#!/usr/bin/env node
/**
 * Compare the website's ft-param-estimator.js against PyTorch ground truth.
 *
 * Reads full_tests/param_truth.json (produced by param_count_check.py) and
 * runs FTParamEstimator.estimate() on each stored config, then reports the
 * relative drift of the total parameter count per config.
 *
 * Usage:
 *   node full_tests/param_estimate_check.mjs [path/to/ft-param-estimator.js] \
 *        [path/to/param_truth.json] [--tol 0.01]
 *
 * Exit code 1 if any config exceeds the tolerance (default 1%).
 */

import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2).filter(a => a !== '--');
const tolIdx = args.indexOf('--tol');
const TOL = tolIdx >= 0 ? parseFloat(args[tolIdx + 1]) : 0.01;
if (tolIdx >= 0) { args.splice(tolIdx, 2); }

const estPath = args[0] || path.resolve(here, '../../erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js');
const truthPath = args[1] || path.join(here, 'param_truth.json');

// Load the estimator (plain browser IIFE → eval into a sandbox namespace).
const src = readFileSync(estPath, 'utf8');
const FTParamEstimator = (() => {
  const module = {};
  const fn = new Function(`${src}; return FTParamEstimator;`);
  return fn();
})();

const truth = JSON.parse(readFileSync(truthPath, 'utf8'));
const rows = [];
let pass = 0, fail = 0, skip = 0;

for (const [name, entry] of Object.entries(truth.results)) {
  const cfg = entry.config;
  const gt = entry.cats.total;
  const est = FTParamEstimator.estimate(cfg);
  if (!est || typeof est.total !== 'number' || est.total < 0) {
    skip++;
    rows.push({ name, gt, est: null, drift: null });
    continue;
  }
  const drift = Math.abs(est.total - gt) / gt;
  if (drift <= TOL) pass++; else fail++;
  rows.push({ name, gt, est: est.total, drift, note: est.note });
}

// Report
const fmt = n => n == null ? '—' : n.toLocaleString('en-US');
console.log(`estimator: ${estPath}`);
console.log(`tolerance: ${(TOL * 100).toFixed(1)}%   pass: ${pass}  fail: ${fail}  skip: ${skip}\n`);
const bad = rows.filter(r => r.drift == null || r.drift > TOL).sort((a, b) => (b.drift ?? 9) - (a.drift ?? 9));
for (const r of bad.slice(0, 40)) {
  const pct = r.drift == null ? 'SKIP' : (r.drift * 100).toFixed(2) + '%';
  console.log(`FAIL ${pct.padStart(8)}  ${r.name}`);
  console.log(`     truth=${fmt(r.gt)}  est=${fmt(r.est)}`);
}
if (bad.length > 40) console.log(`... ${bad.length - 40} more`);
console.log(fail === 0 ? `\nALL PASS (${pass} within ${(TOL * 100).toFixed(1)}%)` : `\n${fail} configs exceed tolerance`);
process.exit(fail === 0 ? 0 : 1);
