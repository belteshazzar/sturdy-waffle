'use strict';

const Brain              = require('../src/brain/Brain');
const { fuzzySyllabus }  = require('../syllabi/fuzzy');
const path               = require('path');

// ── Helpers ───────────────────────────────────────────────────────────────────

function hr(char = '─', width = 60) {
  return char.repeat(width);
}

// ── Main ──────────────────────────────────────────────────────────────────────

function main() {
  console.log(hr('═'));
  console.log('  Digital Brain Framework — Fuzzy Logic Demo');
  console.log(hr('═'));
  console.log();
  console.log('  Fuzzy logic extends boolean reasoning from crisp binary');
  console.log('  {0, 1} truth values to graded membership degrees in [0, 1],');
  console.log('  enabling approximate and uncertain reasoning that naturally');
  console.log('  bridges the boolean and mathematics knowledge domains.');
  console.log();

  // ── Create a fresh brain ──────────────────────────────────────────────────
  const brain = new Brain({
    verbose:               true,
    defaultTargetAccuracy: 0.95,
    regressionTolerance:   0.05,
    epochsPerRound:        300,
    maxEpochsTotal:        30000,
    maxMutations:          15,
  });

  console.log(`Brain created.  Regions: ${brain.introspect().regionCount}  (no knowledge yet)\n`);

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
    process.stdout.write('\n');
    const status = trained ? '✓ learned' : '✗ partial';
    console.log(
      `  ${status}  ${domain.padEnd(14)} ` +
      `acc=${(accuracy * 100).toFixed(1).padStart(5)}%  ` +
      `mutations=${mutationCount}  epochs=${totalEpochs}`
    );
  });

  // ── Learn the full fuzzy syllabus ─────────────────────────────────────────
  console.log('Learning Fuzzy Logic Syllabus…');
  console.log(hr());
  brain.learnSyllabus(fuzzySyllabus);
  console.log(hr());
  console.log();

  // ── Prediction tests ──────────────────────────────────────────────────────
  console.log('Prediction Tests  (tolerance ±0.10 in raw units)');
  console.log(hr());

  const tests = [
    // Level 1 — linear complement
    { domain: 'fuzzy.NOT', inputs: [0.0],       expected: 1.0,  label: 'NOT(0.0)'       },
    { domain: 'fuzzy.NOT', inputs: [1.0],       expected: 0.0,  label: 'NOT(1.0)'       },
    { domain: 'fuzzy.NOT', inputs: [0.5],       expected: 0.5,  label: 'NOT(0.5)'       },
    { domain: 'fuzzy.NOT', inputs: [0.3],       expected: 0.7,  label: 'NOT(0.3)'       },
    // Level 2 — piecewise-linear AND/OR
    { domain: 'fuzzy.AND', inputs: [0.0, 0.0],  expected: 0.0,  label: 'AND(0.0, 0.0)'  },
    { domain: 'fuzzy.AND', inputs: [1.0, 1.0],  expected: 1.0,  label: 'AND(1.0, 1.0)'  },
    { domain: 'fuzzy.AND', inputs: [0.3, 0.7],  expected: 0.3,  label: 'AND(0.3, 0.7)'  },
    { domain: 'fuzzy.AND', inputs: [0.7, 0.3],  expected: 0.3,  label: 'AND(0.7, 0.3)'  },
    { domain: 'fuzzy.OR',  inputs: [0.0, 0.0],  expected: 0.0,  label: 'OR(0.0, 0.0)'   },
    { domain: 'fuzzy.OR',  inputs: [1.0, 1.0],  expected: 1.0,  label: 'OR(1.0, 1.0)'   },
    { domain: 'fuzzy.OR',  inputs: [0.3, 0.7],  expected: 0.7,  label: 'OR(0.3, 0.7)'   },
    // Level 3 — non-linear XOR and IMP
    { domain: 'fuzzy.XOR', inputs: [0.5, 0.5],  expected: 0.0,  label: 'XOR(0.5, 0.5)'  },
    { domain: 'fuzzy.XOR', inputs: [0.0, 1.0],  expected: 1.0,  label: 'XOR(0.0, 1.0)'  },
    { domain: 'fuzzy.XOR', inputs: [0.3, 0.7],  expected: 0.4,  label: 'XOR(0.3, 0.7)'  },
    { domain: 'fuzzy.IMP', inputs: [0.0, 0.0],  expected: 1.0,  label: 'IMP(0.0, 0.0)'  },
    { domain: 'fuzzy.IMP', inputs: [1.0, 0.0],  expected: 0.0,  label: 'IMP(1.0, 0.0)'  },
    { domain: 'fuzzy.IMP', inputs: [0.5, 0.5],  expected: 0.5,  label: 'IMP(0.5, 0.5)'  },
  ];

  let passed = 0;
  const tolerance = 0.10;
  for (const { domain, inputs, expected, label } of tests) {
    const result = brain.predict(inputs, domain)[0];
    const ok     = Math.abs(result - expected) <= tolerance;
    if (ok) passed++;
    console.log(
      `  ${ok ? '✓' : '✗'}  ${label.padEnd(20)} = ${result.toFixed(4).padStart(8)}  ` +
      `(expected ${String(expected).padStart(5)})`
    );
  }
  console.log(hr());
  console.log(`  Tests: ${passed}/${tests.length} passed\n`);

  // ── Expression-tree evaluation ────────────────────────────────────────────
  console.log('Expression Tree Tests');
  console.log(hr());

  // NOT(0.3) ≈ 0.7
  const exprNot = brain.evaluate({
    op: 'NOT', domain: 'fuzzy.NOT',
    inputs: [{ value: 0.3 }],
  });
  console.log(`  NOT(0.3)                       = ${exprNot.toFixed(4)}  (expected ≈ 0.7000)`);

  // AND(0.8, OR(0.3, 0.6))  →  AND(0.8, 0.6)  ≈  0.6
  const exprCompose = brain.evaluate({
    op: 'AND', domain: 'fuzzy.AND',
    inputs: [
      { value: 0.8 },
      { op: 'OR', domain: 'fuzzy.OR', inputs: [{ value: 0.3 }, { value: 0.6 }] },
    ],
  });
  console.log(`  AND(0.8, OR(0.3, 0.6))         = ${exprCompose.toFixed(4)}  (expected ≈ 0.6000)`);

  // IMP(NOT(0.2), 0.5)  →  IMP(0.8, 0.5) = max(1-0.8, 0.5) = max(0.2, 0.5) ≈ 0.5
  const exprImp = brain.evaluate({
    op: 'IMP', domain: 'fuzzy.IMP',
    inputs: [
      { op: 'NOT', domain: 'fuzzy.NOT', inputs: [{ value: 0.2 }] },
      { value: 0.5 },
    ],
  });
  console.log(`  IMP(NOT(0.2), 0.5)             = ${exprImp.toFixed(4)}  (expected ≈ 0.5000)`);

  console.log();

  // ── Introspection ─────────────────────────────────────────────────────────
  console.log('Brain Introspection');
  console.log(hr());
  const info = brain.introspect();
  console.log(`  Version:      ${info.version}`);
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
  const savePath = path.join(__dirname, '../saves/fuzzy-brain.json');
  console.log(`Saving brain state to: ${savePath}`);
  const saveInfo = brain.save(savePath);
  console.log(`  Saved ${saveInfo.regionCount} regions (${(saveInfo.sizeBytes / 1024).toFixed(1)} KB)\n`);

  console.log('Loading brain from saved state…');
  const loadedBrain  = Brain.load(savePath);
  const loadedInfo   = loadedBrain.introspect();
  const verifyResult = loadedBrain.predict([0.3], 'fuzzy.NOT')[0];
  console.log(`  Loaded ${loadedInfo.regionCount} regions.`);
  console.log(`  Verify NOT(0.3) via loaded brain = ${verifyResult.toFixed(4)}  ` +
    `${Math.abs(verifyResult - 0.7) <= 0.1 ? '✓' : '✗'}\n`);

  console.log(hr('═'));
  console.log(`  Total: ${passed}/${tests.length} tests passed`);
  console.log(hr('═'));
}

main();
