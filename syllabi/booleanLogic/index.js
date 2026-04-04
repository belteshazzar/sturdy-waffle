'use strict';

const Lesson   = require('../../src/learning/Lesson');
const Syllabus = require('../../src/learning/Syllabus');

// ── Helper ────────────────────────────────────────────────────────────────────

/** Convenience to express boolean training samples clearly. */
const s = (inputs, output) => ({ input: inputs, output: [output] });

// ── Lesson definitions ────────────────────────────────────────────────────────

const andLesson = new Lesson({
  name:        'AND Gate',
  domain:      'boolean.AND',
  description: 'Output is 1 only when ALL inputs are 1.',
  trainingData: [
    s([0, 0], 0),
    s([0, 1], 0),
    s([1, 0], 0),
    s([1, 1], 1),
  ],
  tags: ['boolean', 'basic', 'conjunction'],
});

const orLesson = new Lesson({
  name:        'OR Gate',
  domain:      'boolean.OR',
  description: 'Output is 1 when ANY input is 1.',
  trainingData: [
    s([0, 0], 0),
    s([0, 1], 1),
    s([1, 0], 1),
    s([1, 1], 1),
  ],
  tags: ['boolean', 'basic', 'disjunction'],
});

const notLesson = new Lesson({
  name:        'NOT Gate',
  domain:      'boolean.NOT',
  description: 'Output is the complement of the single input.',
  trainingData: [
    s([0], 1),
    s([1], 0),
  ],
  inputSize:  1,
  outputSize: 1,
  tags: ['boolean', 'basic', 'negation'],
});

const xorLesson = new Lesson({
  name:        'XOR Gate',
  domain:      'boolean.XOR',
  description: 'Output is 1 when inputs DIFFER (exclusive or).',
  trainingData: [
    s([0, 0], 0),
    s([0, 1], 1),
    s([1, 0], 1),
    s([1, 1], 0),
  ],
  tags: ['boolean', 'basic', 'exclusive-or'],
});

const nandLesson = new Lesson({
  name:        'NAND Gate',
  domain:      'boolean.NAND',
  description: 'NOT AND — output is 0 only when all inputs are 1.',
  trainingData: [
    s([0, 0], 1),
    s([0, 1], 1),
    s([1, 0], 1),
    s([1, 1], 0),
  ],
  tags: ['boolean', 'basic', 'universal'],
});

const norLesson = new Lesson({
  name:        'NOR Gate',
  domain:      'boolean.NOR',
  description: 'NOT OR — output is 1 only when all inputs are 0.',
  trainingData: [
    s([0, 0], 1),
    s([0, 1], 0),
    s([1, 0], 0),
    s([1, 1], 0),
  ],
  tags: ['boolean', 'basic', 'universal'],
});

const xnorLesson = new Lesson({
  name:        'XNOR Gate',
  domain:      'boolean.XNOR',
  description: 'NOT XOR (equivalence) — output is 1 when inputs are EQUAL.',
  trainingData: [
    s([0, 0], 1),
    s([0, 1], 0),
    s([1, 0], 0),
    s([1, 1], 1),
  ],
  tags: ['boolean', 'basic', 'equivalence'],
});

// ── Syllabus ──────────────────────────────────────────────────────────────────

/**
 * A progressive curriculum that teaches boolean logic from the simplest gates
 * to the full set required for nested expression evaluation.
 *
 * Ordering:
 *   1. AND  – simple conjunction (linearly separable)
 *   2. OR   – simple disjunction (linearly separable)
 *   3. NOT  – single-input inversion (trivially learnable)
 *   4. XOR  – classic non-linearly separable problem; tests hidden layers
 *   5. NAND – universal gate; complement of AND
 *   6. NOR  – universal gate; complement of OR
 *   7. XNOR – equivalence; complement of XOR
 */
const booleanLogicSyllabus = new Syllabus({
  name: 'Boolean Logic Fundamentals',
  description:
    'A progressive curriculum teaching boolean logic from basic gates ' +
    'to the complete set needed for nested expression evaluation.',
  lessons: [
    andLesson,
    orLesson,
    notLesson,
    xorLesson,
    nandLesson,
    norLesson,
    xnorLesson,
  ],
  tags: ['boolean', 'logic', 'fundamentals'],
});

module.exports = {
  booleanLogicSyllabus,
  lessons: {
    AND:  andLesson,
    OR:   orLesson,
    NOT:  notLesson,
    XOR:  xorLesson,
    NAND: nandLesson,
    NOR:  norLesson,
    XNOR: xnorLesson,
  },
};
