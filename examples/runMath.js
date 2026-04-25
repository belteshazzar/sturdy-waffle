'use strict';

const Brain            = require('../src/brain/Brain');
const { mathSyllabus } = require('../syllabi/math');
const path             = require('path');

// ── Helpers ───────────────────────────────────────────────────────────────────

function hr(char = '─', width = 60) {
  return char.repeat(width);
}

// ── Main ──────────────────────────────────────────────────────────────────────

function main() {
  console.log(hr('═'));
  console.log('  Digital Brain Framework — General Mathematics Demo');
  console.log(hr('═'));
  console.log();

  // ── Create a fresh brain ──────────────────────────────────────────────────
  const brain = new Brain({
    verbose:               true,
    defaultTargetAccuracy: 0.95,   // 95 % of samples within tolerance
    regressionTolerance:   0.05,   // ±5 % of the normalised [0,1] range
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

  // ── Learn the full math syllabus ──────────────────────────────────────────
  console.log('Learning General Mathematics Syllabus…');
  console.log(hr());
  brain.learnSyllabus(mathSyllabus);
  console.log(hr());
  console.log();

  // ── Prediction tests ──────────────────────────────────────────────────────
  console.log('Prediction Tests  (tolerance ±0.10 in raw units)');
  console.log(hr());

  const tests = [
    // Level 1 — linear
    { domain: 'math.ADD',  inputs: [0.3,  0.5],  expected: 0.8,                label: 'ADD(0.3, 0.5)'  },
    { domain: 'math.ADD',  inputs: [0.0,  1.0],  expected: 1.0,                label: 'ADD(0.0, 1.0)'  },
    { domain: 'math.SUB',  inputs: [0.8,  0.3],  expected: 0.5,                label: 'SUB(0.8, 0.3)'  },
    { domain: 'math.SUB',  inputs: [0.2,  0.6],  expected: -0.4,               label: 'SUB(0.2, 0.6)'  },
    // Level 2 — non-linear
    { domain: 'math.MUL',  inputs: [0.5,  0.6],  expected: 0.3,                label: 'MUL(0.5, 0.6)'  },
    { domain: 'math.MUL',  inputs: [0.0,  0.9],  expected: 0.0,                label: 'MUL(0.0, 0.9)'  },
    // Level 3 — curved / unbounded
    { domain: 'math.SQRT', inputs: [0.25],        expected: 0.5,                label: 'SQRT(0.25)'     },
    { domain: 'math.SQRT', inputs: [1.0],          expected: 1.0,                label: 'SQRT(1.0)'      },
    { domain: 'math.DIV',  inputs: [0.6,  0.5],  expected: 1.2,                label: 'DIV(0.6, 0.5)'  },
    { domain: 'math.DIV',  inputs: [0.3,  1.0],  expected: 0.3,                label: 'DIV(0.3, 1.0)'  },
    // Level 4 — periodic
    { domain: 'math.SIN',  inputs: [Math.PI / 2], expected: 1.0,               label: 'SIN(π/2)'       },
    { domain: 'math.SIN',  inputs: [Math.PI],      expected: 0.0,               label: 'SIN(π)'         },
    { domain: 'math.COS',  inputs: [0],             expected: 1.0,               label: 'COS(0)'         },
    { domain: 'math.COS',  inputs: [Math.PI],      expected: -1.0,              label: 'COS(π)'         },
  ];

  let passed = 0;
  const tolerance = 0.10;
  for (const { domain, inputs, expected, label } of tests) {
    const result = brain.predict(inputs, domain)[0];
    const ok     = Math.abs(result - expected) <= tolerance;
    if (ok) passed++;
    console.log(
      `  ${ok ? '✓' : '✗'}  ${label.padEnd(18)} = ${result.toFixed(4).padStart(8)}  ` +
      `(expected ${String(expected).padStart(6)})`
    );
  }
  console.log(hr());
  console.log(`  Tests: ${passed}/${tests.length} passed\n`);

  // ── Expression-tree evaluation ────────────────────────────────────────────
  console.log('Expression Tree Tests');
  console.log(hr());

  // ADD(0.3, 0.5) ≈ 0.8
  const exprAdd = brain.evaluate({
    op: 'ADD', domain: 'math.ADD',
    inputs: [{ value: 0.3 }, { value: 0.5 }],
  });
  console.log(`  ADD(0.3, 0.5)              = ${exprAdd.toFixed(4)}  (expected ≈ 0.8000)`);

  // MUL(ADD(0.2, 0.3), 0.5)  →  MUL(0.5, 0.5)  ≈  0.25
  const exprCompose = brain.evaluate({
    op: 'MUL', domain: 'math.MUL',
    inputs: [
      { op: 'ADD', domain: 'math.ADD', inputs: [{ value: 0.2 }, { value: 0.3 }] },
      { value: 0.5 },
    ],
  });
  console.log(`  MUL(ADD(0.2, 0.3), 0.5)   = ${exprCompose.toFixed(4)}  (expected ≈ 0.2500)`);

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
  const savePath = path.join(__dirname, '../saves/math-brain.json');
  console.log(`Saving brain state to: ${savePath}`);
  const saveInfo = brain.save(savePath);
  console.log(`  Saved ${saveInfo.regionCount} regions (${(saveInfo.sizeBytes / 1024).toFixed(1)} KB)\n`);

  console.log('Loading brain from saved state…');
  const loadedBrain  = Brain.load(savePath);
  const loadedInfo   = loadedBrain.introspect();
  const verifyResult = loadedBrain.predict([0.3, 0.5], 'math.ADD')[0];
  console.log(`  Loaded ${loadedInfo.regionCount} regions.`);
  console.log(`  Verify ADD(0.3, 0.5) via loaded brain = ${verifyResult.toFixed(4)}  ` +
    `${Math.abs(verifyResult - 0.8) <= 0.1 ? '✓' : '✗'}\n`);

  console.log(hr('═'));
  console.log(`  Total: ${passed}/${tests.length} tests passed`);
  console.log(hr('═'));
}

main();
