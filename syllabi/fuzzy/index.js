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
// tanh hidden layers give good gradient flow; linear output avoids saturation.
const REGRESSION_NET = { hiddenActivation: 'tanh', outputActivation: 'linear' };

// ── Grid points ───────────────────────────────────────────────────────────────

const GRID_11 = linspace(0, 1, 11);   // 11 × 11 = 121 training samples per 2-input op
const LINE_21 = linspace(0, 1, 21);   // 21 points for 1-input ops

// ── Lesson definitions ────────────────────────────────────────────────────────

/**
 * Fuzzy NOT (Zadeh complement): output = 1 − a
 *
 * The simplest fuzzy operation — a strict linear negation that maps
 * the membership degree a to its complement.  This is the continuous
 * analogue of boolean NOT and is perfectly linearly learnable.
 *
 * At the crisp boundaries: NOT(0) = 1, NOT(1) = 0.
 */
const notLesson = new Lesson({
  name:        'Fuzzy NOT',
  domain:      'fuzzy.NOT',
  description: 'Zadeh complement: output = 1 − a for a ∈ [0, 1].',
  trainingData: samples1D(a => 1 - a, LINE_21),
  inputSize:    1,
  mode:         'regression',
  networkConfig: REGRESSION_NET,
  tags: ['fuzzy', 'logic', 'complement', 'level1'],
});

/**
 * Fuzzy AND (minimum t-norm): output = min(a, b)
 *
 * The standard Zadeh intersection of fuzzy sets.  The output equals the
 * weakest (lowest) membership degree, representing the degree to which
 * both conditions hold simultaneously.  At crisp inputs this reduces
 * exactly to boolean AND.
 *
 * min is piecewise-linear — the network must learn a "valley" surface.
 */
const andLesson = new Lesson({
  name:        'Fuzzy AND',
  domain:      'fuzzy.AND',
  description: 'Minimum t-norm: output = min(a, b) for a, b ∈ [0, 1].',
  trainingData: grid2D((a, b) => Math.min(a, b), GRID_11, GRID_11),
  inputSize:    2,
  mode:         'regression',
  networkConfig: REGRESSION_NET,
  tags: ['fuzzy', 'logic', 'conjunction', 'level2'],
});

/**
 * Fuzzy OR (maximum t-conorm): output = max(a, b)
 *
 * The standard Zadeh union of fuzzy sets.  The output equals the
 * strongest (highest) membership degree, representing the degree to which
 * at least one condition holds.  At crisp inputs this reduces exactly to
 * boolean OR.
 *
 * max is piecewise-linear — the network must learn a "ridge" surface.
 */
const orLesson = new Lesson({
  name:        'Fuzzy OR',
  domain:      'fuzzy.OR',
  description: 'Maximum t-conorm: output = max(a, b) for a, b ∈ [0, 1].',
  trainingData: grid2D((a, b) => Math.max(a, b), GRID_11, GRID_11),
  inputSize:    2,
  mode:         'regression',
  networkConfig: REGRESSION_NET,
  tags: ['fuzzy', 'logic', 'disjunction', 'level2'],
});

/**
 * Fuzzy XOR (symmetric difference): output = |a − b|
 *
 * Measures the degree to which exactly one of the two conditions holds.
 * Analogous to boolean XOR: outputs 0 when both values are equal and 1
 * when they are maximally different.  The absolute-value surface is
 * non-linear and provides a useful challenge for hidden-layer capacity.
 */
const xorLesson = new Lesson({
  name:        'Fuzzy XOR',
  domain:      'fuzzy.XOR',
  description: 'Symmetric difference: output = |a − b| for a, b ∈ [0, 1].',
  trainingData: grid2D((a, b) => Math.abs(a - b), GRID_11, GRID_11),
  inputSize:    2,
  mode:         'regression',
  networkConfig: REGRESSION_NET,
  tags: ['fuzzy', 'logic', 'exclusive-or', 'level3'],
});

/**
 * Fuzzy implication (Kleene-Dienes): output = max(1 − a, b)
 *
 * Encodes the rule "IF a THEN b" in fuzzy terms.  When the antecedent a is
 * fully true (1) the result equals b; when a is fully false (0) the rule is
 * vacuously true (result = 1).  This is the foundation of fuzzy rule-based
 * inference systems and naturally bridges fuzzy logic with conditional
 * reasoning.  The surface combines a NOT and an OR, making it slightly
 * harder to learn than either component alone.
 */
const impLesson = new Lesson({
  name:        'Fuzzy IMP',
  domain:      'fuzzy.IMP',
  description: 'Kleene-Dienes implication: output = max(1 − a, b) for a, b ∈ [0, 1].',
  trainingData: grid2D((a, b) => Math.max(1 - a, b), GRID_11, GRID_11),
  inputSize:    2,
  mode:         'regression',
  networkConfig: REGRESSION_NET,
  tags: ['fuzzy', 'logic', 'implication', 'level3'],
});

// ── Syllabus ──────────────────────────────────────────────────────────────────

/**
 * A three-level progressive curriculum teaching fuzzy logic from the
 * simplest linear complement through piecewise-linear conjunctions and
 * disjunctions to the non-linear exclusive-or and implication operations.
 *
 * Fuzzy logic extends boolean logic from binary {0, 1} truth values to
 * graded membership degrees in [0, 1].  Every operation reduces exactly to
 * its boolean counterpart at the crisp boundaries, allowing a trained brain
 * to reason over uncertain or partial information in a way that naturally
 * generalises the knowledge it already holds in the boolean and math domains.
 *
 * Ordering (difficulty):
 *   Level 1 — NOT  : linear complement; trivially learnable
 *   Level 2 — AND  : piecewise-linear min surface
 *              OR   : piecewise-linear max surface
 *   Level 3 — XOR  : absolute-difference surface; non-linear
 *              IMP  : Kleene-Dienes implication; combines NOT + OR
 */
const fuzzySyllabus = new Syllabus({
  name: 'Fuzzy Logic',
  description:
    'A progressive curriculum teaching fuzzy (continuous) logic from the ' +
    'Zadeh complement through t-norms and implication, bridging boolean ' +
    'reasoning and continuous mathematics.',
  lessons: [
    notLesson,
    andLesson,
    orLesson,
    xorLesson,
    impLesson,
  ],
  tags: ['fuzzy', 'logic', 'continuous', 'approximate-reasoning'],
});

module.exports = {
  fuzzySyllabus,
  lessons: {
    NOT: notLesson,
    AND: andLesson,
    OR:  orLesson,
    XOR: xorLesson,
    IMP: impLesson,
  },
};
