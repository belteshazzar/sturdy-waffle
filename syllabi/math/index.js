'use strict';

const Lesson   = require('../../src/learning/Lesson');
const Syllabus = require('../../src/learning/Syllabus');

// ── Sample-generation helpers ─────────────────────────────────────────────────

/**
 * Return `n` evenly-spaced values between `start` and `end` (inclusive).
 * @param {number} start
 * @param {number} end
 * @param {number} n
 * @returns {number[]}
 */
function linspace(start, end, n) {
  if (n <= 1) return [start];
  return Array.from({ length: n }, (_, i) => start + (i / (n - 1)) * (end - start));
}

/**
 * Build a set of 2-input training samples on a grid.
 * @param {function(number, number): number} fn
 * @param {number[]} aVals
 * @param {number[]} bVals
 * @returns {Array<{input: number[], output: number[]}>}
 */
function grid2D(fn, aVals, bVals) {
  const samples = [];
  for (const a of aVals) {
    for (const b of bVals) {
      samples.push({ input: [a, b], output: [fn(a, b)] });
    }
  }
  return samples;
}

/**
 * Build a set of 1-input training samples.
 * @param {function(number): number} fn
 * @param {number[]} xVals
 * @returns {Array<{input: number[], output: number[]}>}
 */
function samples1D(fn, xVals) {
  return xVals.map(x => ({ input: [x], output: [fn(x)] }));
}

// ── Shared network config for all regression lessons ─────────────────────────
// tanh hidden layers give better gradient flow for regression;
// linear output avoids saturation and lets the network reach any target value.
const REGRESSION_NET = { hiddenActivation: 'tanh', outputActivation: 'linear' };

// ── Level 1: Addition & Subtraction (linearly separable) ─────────────────────

const GRID_10 = linspace(0, 1, 10);   // 10 × 10 = 100 training samples

/**
 * Addition: output = a + b  (range [0, 2], normalised to [0, 1]).
 * This is a linear function — the easiest regression task.
 */
const addLesson = new Lesson({
  name:        'Addition',
  domain:      'math.ADD',
  description: 'Add two numbers in [0, 1]; result is in [0, 2].',
  trainingData: grid2D((a, b) => a + b, GRID_10, GRID_10),
  mode:         'regression',
  normalise:    { outputRange: [0, 2] },
  networkConfig: REGRESSION_NET,
  tags: ['math', 'arithmetic', 'level1'],
});

/**
 * Subtraction: output = a − b  (range [−1, 1], normalised to [0, 1]).
 */
const subLesson = new Lesson({
  name:        'Subtraction',
  domain:      'math.SUB',
  description: 'Subtract b from a where both are in [0, 1]; result is in [−1, 1].',
  trainingData: grid2D((a, b) => a - b, GRID_10, GRID_10),
  mode:         'regression',
  normalise:    { outputRange: [-1, 1] },
  networkConfig: REGRESSION_NET,
  tags: ['math', 'arithmetic', 'level1'],
});

// ── Level 2: Multiplication (non-linearly separable) ─────────────────────────

/**
 * Multiplication: output = a × b  (range [0, 1]).
 * Non-linear function — requires hidden-layer capacity analogous to XOR in
 * boolean logic.  No output normalisation needed since range is already [0, 1].
 */
const mulLesson = new Lesson({
  name:        'Multiplication',
  domain:      'math.MUL',
  description: 'Multiply two numbers in [0, 1]; result is in [0, 1].',
  trainingData: grid2D((a, b) => a * b, GRID_10, GRID_10),
  mode:         'regression',
  networkConfig: REGRESSION_NET,
  tags: ['math', 'arithmetic', 'level2'],
});

// ── Level 3: Division & Square Root (unbounded / curved outputs) ──────────────

// For division, keep b ≥ 0.1 to avoid division by zero.
// a ∈ [0, 1], b ∈ [0.1, 1]  →  a/b ∈ [0, 10].
const DIV_A_VALS = linspace(0, 1, 9);
const DIV_B_VALS = linspace(0.1, 1, 9);   // 9 × 9 = 81 samples

/**
 * Division: output = a / b  (b ∈ [0.1, 1], range [0, 10]).
 * Tests the normalisation layer: the raw output is unbounded relative to
 * the [0, 1] inputs.
 */
const divLesson = new Lesson({
  name:        'Division',
  domain:      'math.DIV',
  description: 'Divide a by b where a ∈ [0,1] and b ∈ [0.1,1]; result is in [0, 10].',
  trainingData: grid2D((a, b) => a / b, DIV_A_VALS, DIV_B_VALS),
  mode:         'regression',
  normalise:    { outputRange: [0, 10] },
  networkConfig: REGRESSION_NET,
  tags: ['math', 'arithmetic', 'level3'],
});

/**
 * Square root: output = √x  (x ∈ [0, 1], range [0, 1]).
 * Smooth but curved — no normalisation needed since range is already [0, 1].
 */
const sqrtLesson = new Lesson({
  name:        'Square Root',
  domain:      'math.SQRT',
  description: 'Square root of x in [0, 1]; result is in [0, 1].',
  trainingData: samples1D(x => Math.sqrt(x), linspace(0, 1, 51)),
  inputSize:    1,
  mode:         'regression',
  networkConfig: REGRESSION_NET,
  tags: ['math', 'arithmetic', 'level3'],
});

// ── Level 4: Trigonometry (periodic functions) ────────────────────────────────

const TWO_PI  = 2 * Math.PI;
const SIN_PTS = 64;   // 64 points over one full period

/**
 * Sine: output = sin(x)  (x ∈ [0, 2π], output ∈ [−1, 1]).
 * Periodic function — the hardest lesson in the syllabus.  Tests whether the
 * network can grow sufficient capacity via mutation to approximate periodicity.
 */
const sinLesson = new Lesson({
  name:        'Sine',
  domain:      'math.SIN',
  description: 'Compute sin(x) for x in [0, 2π]; result is in [−1, 1].',
  trainingData: samples1D(x => Math.sin(x), linspace(0, TWO_PI, SIN_PTS)),
  inputSize:    1,
  mode:         'regression',
  normalise:    { inputRange: [0, TWO_PI], outputRange: [-1, 1] },
  networkConfig: REGRESSION_NET,
  tags: ['math', 'trigonometry', 'level4'],
});

/**
 * Cosine: output = cos(x)  (x ∈ [0, 2π], output ∈ [−1, 1]).
 * Same difficulty as sine — included to show the brain can learn two
 * phase-shifted periodic functions as independent regions.
 */
const cosLesson = new Lesson({
  name:        'Cosine',
  domain:      'math.COS',
  description: 'Compute cos(x) for x in [0, 2π]; result is in [−1, 1].',
  trainingData: samples1D(x => Math.cos(x), linspace(0, TWO_PI, SIN_PTS)),
  inputSize:    1,
  mode:         'regression',
  normalise:    { inputRange: [0, TWO_PI], outputRange: [-1, 1] },
  networkConfig: REGRESSION_NET,
  tags: ['math', 'trigonometry', 'level4'],
});

// ── Syllabus ──────────────────────────────────────────────────────────────────

/**
 * A four-level progressive curriculum that teaches the brain general arithmetic
 * and trigonometry from zero, using the same spawn → train → consolidate
 * lifecycle as the boolean logic syllabus.
 *
 * Ordering (difficulty):
 *   Level 1 — ADD, SUB   : linear functions; trivially learnable
 *   Level 2 — MUL        : non-linear (degree 2); analogous to XOR
 *   Level 3 — DIV, SQRT  : non-linear with unbounded / curved outputs
 *   Level 4 — SIN, COS   : periodic; hardest — mutation typically needed
 */
const mathSyllabus = new Syllabus({
  name: 'General Mathematics',
  description:
    'A progressive curriculum teaching arithmetic and trigonometry from basic ' +
    'linear operations to periodic functions, using continuous regression.',
  lessons: [
    addLesson,
    subLesson,
    mulLesson,
    divLesson,
    sqrtLesson,
    sinLesson,
    cosLesson,
  ],
  tags: ['math', 'regression', 'arithmetic', 'trigonometry'],
});

module.exports = {
  mathSyllabus,
  lessons: {
    ADD:  addLesson,
    SUB:  subLesson,
    MUL:  mulLesson,
    DIV:  divLesson,
    SQRT: sqrtLesson,
    SIN:  sinLesson,
    COS:  cosLesson,
  },
};
