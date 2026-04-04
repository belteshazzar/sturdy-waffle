'use strict';

const Brain                               = require('../src/brain/Brain');
const { booleanLogicSyllabus }            = require('../syllabi/booleanLogic');
const path                                = require('path');

// ── Helpers ───────────────────────────────────────────────────────────────────

function exprToString(expr) {
  if (expr.value !== undefined) return String(expr.value);
  return `${expr.op}(${expr.inputs.map(exprToString).join(', ')})`;
}

function hr(char = '─', width = 60) {
  return char.repeat(width);
}

// ── Main ──────────────────────────────────────────────────────────────────────

function main() {
  console.log(hr('═'));
  console.log('  Digital Brain Framework — Boolean Logic Demo');
  console.log(hr('═'));
  console.log();

  // ── Create a brand-new brain with zero knowledge ──────────────────────────
  const brain = new Brain({
    verbose:               true,
    defaultTargetAccuracy: 0.99,
    epochsPerRound:        300,
    maxEpochsTotal:        20000,
    maxMutations:          12,
  });

  const introBefore = brain.introspect();
  console.log(`Brain created.  Regions: ${introBefore.regionCount}  (no knowledge yet)\n`);

  // ── Progress logging ──────────────────────────────────────────────────────
  brain.on('training:progress', ({ domain, epoch, accuracy }) => {
    if (epoch % 1500 === 0) {
      process.stdout.write(
        `  [${domain}] epoch ${String(epoch).padStart(5)} | ` +
        `acc ${(accuracy * 100).toFixed(1).padStart(5)}%\r`
      );
    }
  });

  brain.on('lesson:complete', ({ domain, trained, accuracy, mutationCount, totalEpochs }) => {
    process.stdout.write('\n');   // clear progress line
    const status = trained ? '✓ learned' : '✗ partial';
    console.log(
      `  ${status}  ${domain.padEnd(14)} ` +
      `acc=${( accuracy * 100).toFixed(1).padStart(5)}%  ` +
      `mutations=${mutationCount}  epochs=${totalEpochs}`
    );
  });

  // ── Learn the full boolean logic syllabus ─────────────────────────────────
  console.log('Learning Boolean Logic Syllabus…');
  console.log(hr());
  brain.learnSyllabus(booleanLogicSyllabus);
  console.log(hr());
  console.log();

  // ── Basic gate tests ──────────────────────────────────────────────────────
  console.log('Basic Gate Tests');
  console.log(hr());

  const gateTests = [
    { expr: { op: 'AND',  inputs: [{ value: 0 }, { value: 0 }] }, expected: 0 },
    { expr: { op: 'AND',  inputs: [{ value: 1 }, { value: 1 }] }, expected: 1 },
    { expr: { op: 'OR',   inputs: [{ value: 0 }, { value: 0 }] }, expected: 0 },
    { expr: { op: 'OR',   inputs: [{ value: 1 }, { value: 0 }] }, expected: 1 },
    { expr: { op: 'NOT',  inputs: [{ value: 0 }]                }, expected: 1 },
    { expr: { op: 'NOT',  inputs: [{ value: 1 }]                }, expected: 0 },
    { expr: { op: 'XOR',  inputs: [{ value: 1 }, { value: 0 }] }, expected: 1 },
    { expr: { op: 'XOR',  inputs: [{ value: 1 }, { value: 1 }] }, expected: 0 },
    { expr: { op: 'NAND', inputs: [{ value: 1 }, { value: 1 }] }, expected: 0 },
    { expr: { op: 'NOR',  inputs: [{ value: 0 }, { value: 0 }] }, expected: 1 },
    { expr: { op: 'XNOR', inputs: [{ value: 0 }, { value: 0 }] }, expected: 1 },
    { expr: { op: 'XNOR', inputs: [{ value: 1 }, { value: 0 }] }, expected: 0 },
  ];

  let passed = 0;
  for (const { expr, expected } of gateTests) {
    const result  = brain.evaluate(expr);
    const ok      = result === expected;
    if (ok) passed++;
    console.log(`  ${ok ? '✓' : '✗'}  ${exprToString(expr).padEnd(16)} = ${result}  (expected ${expected})`);
  }
  console.log(hr());
  console.log(`  Basic: ${passed}/${gateTests.length} passed`);
  console.log();

  // ── Nested expression tests ───────────────────────────────────────────────
  console.log('Nested Expression Tests');
  console.log(hr());

  const nestedTests = [
    {
      // AND(OR(1,0), NOT(0))  →  AND(1, 1)  →  1
      expr: {
        op: 'AND', inputs: [
          { op: 'OR',  inputs: [{ value: 1 }, { value: 0 }] },
          { op: 'NOT', inputs: [{ value: 0 }]               },
        ],
      },
      expected: 1,
    },
    {
      // OR(AND(1,0), XOR(1,0))  →  OR(0, 1)  →  1
      expr: {
        op: 'OR', inputs: [
          { op: 'AND', inputs: [{ value: 1 }, { value: 0 }] },
          { op: 'XOR', inputs: [{ value: 1 }, { value: 0 }] },
        ],
      },
      expected: 1,
    },
    {
      // NOT(AND(1, 1))  →  NOT(1)  →  0
      expr: {
        op: 'NOT', inputs: [
          { op: 'AND', inputs: [{ value: 1 }, { value: 1 }] },
        ],
      },
      expected: 0,
    },
    {
      // AND(OR(0,0), NOT(1))  →  AND(0, 0)  →  0
      expr: {
        op: 'AND', inputs: [
          { op: 'OR',  inputs: [{ value: 0 }, { value: 0 }] },
          { op: 'NOT', inputs: [{ value: 1 }]               },
        ],
      },
      expected: 0,
    },
    {
      // XOR(AND(1,1), OR(0,0))  →  XOR(1, 0)  →  1
      expr: {
        op: 'XOR', inputs: [
          { op: 'AND', inputs: [{ value: 1 }, { value: 1 }] },
          { op: 'OR',  inputs: [{ value: 0 }, { value: 0 }] },
        ],
      },
      expected: 1,
    },
    {
      // 3-level: OR(AND(NOT(0), 1), XOR(1, 1))  →  OR(AND(1,1), 0)  →  OR(1,0)  →  1
      expr: {
        op: 'OR', inputs: [
          {
            op: 'AND', inputs: [
              { op: 'NOT', inputs: [{ value: 0 }] },
              { value: 1 },
            ],
          },
          { op: 'XOR', inputs: [{ value: 1 }, { value: 1 }] },
        ],
      },
      expected: 1,
    },
  ];

  let nestedPassed = 0;
  for (const { expr, expected } of nestedTests) {
    const result  = brain.evaluate(expr);
    const ok      = result === expected;
    if (ok) nestedPassed++;
    console.log(`  ${ok ? '✓' : '✗'}  ${exprToString(expr).padEnd(42)} = ${result}  (expected ${expected})`);
  }
  console.log(hr());
  console.log(`  Nested: ${nestedPassed}/${nestedTests.length} passed`);
  console.log();

  // ── Introspection ─────────────────────────────────────────────────────────
  console.log('Brain Introspection');
  console.log(hr());
  const info = brain.introspect();
  console.log(`  Version:      ${info.version}`);
  console.log(`  Created:      ${info.createdAt}`);
  console.log(`  Regions:      ${info.regionCount}`);
  console.log();
  console.log('  Domain                 Acc      Plasticity  Arch          Mutations');
  for (const [domain, r] of Object.entries(info.regions)) {
    console.log(
      `  ${domain.padEnd(22)} ${(r.accuracy * 100).toFixed(1).padStart(5)}%   ` +
      `${r.plasticity.toFixed(2).padStart(6)}      [${r.architecture}]`.padEnd(20) +
      `  ${r.mutationCount}`
    );
  }
  console.log();

  // ── Save / load round-trip ────────────────────────────────────────────────
  const savePath = path.join(__dirname, '../saves/boolean-logic-brain.json');
  console.log(`Saving brain state to: ${savePath}`);
  const saveInfo = brain.save(savePath);
  console.log(`  Saved ${saveInfo.regionCount} regions (${(saveInfo.sizeBytes / 1024).toFixed(1)} KB)\n`);

  console.log('Loading brain from saved state…');
  const loadedBrain = Brain.load(savePath);
  const loadedInfo  = loadedBrain.introspect();
  console.log(`  Loaded ${loadedInfo.regionCount} regions.`);

  const verify = brain.evaluate({ op: 'AND', inputs: [{ value: 1 }, { value: 1 }] });
  console.log(`  Verify AND(1,1) via loaded brain = ${verify} ${verify === 1 ? '✓' : '✗'}`);
  console.log();

  // ── Summary ───────────────────────────────────────────────────────────────
  const total = passed + nestedPassed;
  const max   = gateTests.length + nestedTests.length;
  console.log(hr('═'));
  console.log(`  Total: ${total}/${max} tests passed`);
  console.log(hr('═'));
}

main();
