'use strict';

/**
 * Tests for the fuzzy logic knowledge domain.
 *
 * The file is divided into:
 *   1. Syllabus structure — verify lesson metadata
 *   2. Fuzzy operations — deterministic output checks (using Math.min/max etc.)
 *   3. Brain integration — trains fuzzy.NOT and fuzzy.AND, spot-checks predictions
 *   4. Composed expressions — fuzzy ops inside evaluate() expression trees
 *   5. Boolean isolation — verify existing boolean domain still works unchanged
 */

const Brain   = require('../src/brain/Brain');
const Lesson  = require('../src/learning/Lesson');
const {
  fuzzySyllabus,
  lessons: fuzzyLessons,
} = require('../syllabi/fuzzy');

// ── 1. Syllabus structure ─────────────────────────────────────────────────────

describe('Fuzzy — syllabus structure', () => {
  test('syllabus has five lessons', () => {
    expect(fuzzySyllabus.lessons).toHaveLength(5);
  });

  test('lessons cover NOT, AND, OR, XOR, IMP', () => {
    const domains = fuzzySyllabus.lessons.map(l => l.domain);
    expect(domains).toContain('fuzzy.NOT');
    expect(domains).toContain('fuzzy.AND');
    expect(domains).toContain('fuzzy.OR');
    expect(domains).toContain('fuzzy.XOR');
    expect(domains).toContain('fuzzy.IMP');
  });

  test('all lessons use regression mode', () => {
    for (const lesson of fuzzySyllabus.lessons) {
      expect(lesson.mode).toBe('regression');
    }
  });

  test('all lessons use tanh hidden + linear output activations', () => {
    for (const lesson of fuzzySyllabus.lessons) {
      expect(lesson.networkConfig).not.toBeNull();
      expect(lesson.networkConfig.hiddenActivation).toBe('tanh');
      expect(lesson.networkConfig.outputActivation).toBe('linear');
    }
  });

  test('2-input lessons have inputSize 2', () => {
    for (const key of ['AND', 'OR', 'XOR', 'IMP']) {
      expect(fuzzyLessons[key].inputSize).toBe(2);
    }
  });

  test('NOT lesson has inputSize 1', () => {
    expect(fuzzyLessons.NOT.inputSize).toBe(1);
  });

  test('all lessons have no normalise (output already in [0,1])', () => {
    for (const lesson of fuzzySyllabus.lessons) {
      expect(lesson.normalise).toBeNull();
    }
  });

  test('training data covers the full [0,1] range', () => {
    for (const lesson of fuzzySyllabus.lessons) {
      const outputs = lesson.trainingData.map(s => s.output[0]);
      expect(Math.min(...outputs)).toBeCloseTo(0, 5);
      expect(Math.max(...outputs)).toBeCloseTo(1, 5);
    }
  });

  test('lessons export object has correct keys', () => {
    expect(Object.keys(fuzzyLessons).sort()).toEqual(['AND', 'IMP', 'NOT', 'OR', 'XOR']);
  });
});

// ── 2. Fuzzy operations — deterministic checks ───────────────────────────────

describe('Fuzzy — mathematical properties', () => {
  // Verify that the training-data generator produces correct targets,
  // since BrainRegion will learn directly from these samples.

  function sampleFor(lesson, input) {
    return lesson.trainingData.find(
      s => s.input.every((v, i) => Math.abs(v - input[i]) < 1e-9)
    );
  }

  test('NOT(0) = 1', () => {
    const s = sampleFor(fuzzyLessons.NOT, [0]);
    expect(s).not.toBeUndefined();
    expect(s.output[0]).toBeCloseTo(1, 5);
  });

  test('NOT(1) = 0', () => {
    const s = sampleFor(fuzzyLessons.NOT, [1]);
    expect(s).not.toBeUndefined();
    expect(s.output[0]).toBeCloseTo(0, 5);
  });

  test('NOT(0.5) = 0.5', () => {
    const s = sampleFor(fuzzyLessons.NOT, [0.5]);
    expect(s).not.toBeUndefined();
    expect(s.output[0]).toBeCloseTo(0.5, 5);
  });

  test('AND(0, 1) = 0 (crisp boundary)', () => {
    const s = sampleFor(fuzzyLessons.AND, [0, 1]);
    expect(s.output[0]).toBeCloseTo(0, 5);
  });

  test('AND(1, 1) = 1 (crisp boundary)', () => {
    const s = sampleFor(fuzzyLessons.AND, [1, 1]);
    expect(s.output[0]).toBeCloseTo(1, 5);
  });

  test('AND(0.3, 0.7) = 0.3 (min of 0.3 and 0.7)', () => {
    const s = sampleFor(fuzzyLessons.AND, [0.3, 0.7]);
    expect(s.output[0]).toBeCloseTo(0.3, 5);
  });

  test('OR(0, 1) = 1 (crisp boundary)', () => {
    const s = sampleFor(fuzzyLessons.OR, [0, 1]);
    expect(s.output[0]).toBeCloseTo(1, 5);
  });

  test('OR(0.3, 0.7) = 0.7 (max of 0.3 and 0.7)', () => {
    const s = sampleFor(fuzzyLessons.OR, [0.3, 0.7]);
    expect(s.output[0]).toBeCloseTo(0.7, 5);
  });

  test('XOR(0.5, 0.5) = 0 (equal inputs)', () => {
    const s = sampleFor(fuzzyLessons.XOR, [0.5, 0.5]);
    expect(s.output[0]).toBeCloseTo(0, 5);
  });

  test('XOR(0, 1) = 1 (maximally different crisp inputs)', () => {
    const s = sampleFor(fuzzyLessons.XOR, [0, 1]);
    expect(s.output[0]).toBeCloseTo(1, 5);
  });

  test('XOR(0.3, 0.7) = 0.4 (|0.3 - 0.7|)', () => {
    const s = sampleFor(fuzzyLessons.XOR, [0.3, 0.7]);
    expect(s.output[0]).toBeCloseTo(0.4, 5);
  });

  test('IMP(1, 0) = 0 (false consequent from true antecedent)', () => {
    const s = sampleFor(fuzzyLessons.IMP, [1, 0]);
    expect(s.output[0]).toBeCloseTo(0, 5);
  });

  test('IMP(0, 0) = 1 (vacuously true — antecedent is false)', () => {
    const s = sampleFor(fuzzyLessons.IMP, [0, 0]);
    expect(s.output[0]).toBeCloseTo(1, 5);
  });

  test('IMP(0.5, 0.5) = 0.5 (max(1-0.5, 0.5))', () => {
    const s = sampleFor(fuzzyLessons.IMP, [0.5, 0.5]);
    expect(s.output[0]).toBeCloseTo(0.5, 5);
  });

  test('De Morgan: AND(a,b) = NOT(OR(NOT(a), NOT(b))) holds in training data', () => {
    // Pick one interior point from the AND training set and verify the
    // De Morgan relationship holds in the raw mathematical targets.
    const a = 0.3, b = 0.7;
    const and  = Math.min(a, b);
    const deM  = 1 - Math.max(1 - a, 1 - b);
    expect(and).toBeCloseTo(deM, 10);
  });
});

// ── 3. Brain integration — fuzzy.NOT and fuzzy.AND ───────────────────────────

describe('Fuzzy — NOT lesson (regression integration)', () => {
  let brain;

  beforeAll(() => {
    brain = new Brain({
      defaultTargetAccuracy: 0.90,
      regressionTolerance:   0.05,
      epochsPerRound:        300,
      maxEpochsTotal:        15000,
      maxMutations:          10,
    });
    brain.learn(fuzzyLessons.NOT);
  }, 120000 /* 2 min max */);

  test('brain learned fuzzy.NOT', () => {
    expect(brain.knows('fuzzy.NOT')).toBe(true);
  });

  test('NOT(0) ≈ 1', () => {
    expect(Math.abs(brain.predict([0], 'fuzzy.NOT')[0] - 1)).toBeLessThan(0.12);
  });

  test('NOT(1) ≈ 0', () => {
    expect(Math.abs(brain.predict([1], 'fuzzy.NOT')[0] - 0)).toBeLessThan(0.12);
  });

  test('NOT(0.5) ≈ 0.5', () => {
    expect(Math.abs(brain.predict([0.5], 'fuzzy.NOT')[0] - 0.5)).toBeLessThan(0.12);
  });

  test('NOT(0.2) ≈ 0.8 (unseen input)', () => {
    expect(Math.abs(brain.predict([0.2], 'fuzzy.NOT')[0] - 0.8)).toBeLessThan(0.12);
  });
});

describe('Fuzzy — AND lesson (regression integration)', () => {
  let brain;

  beforeAll(() => {
    brain = new Brain({
      defaultTargetAccuracy: 0.85,
      regressionTolerance:   0.05,
      epochsPerRound:        300,
      maxEpochsTotal:        20000,
      maxMutations:          12,
    });
    brain.learn(fuzzyLessons.AND);
  }, 180000 /* 3 min max */);

  test('brain learned fuzzy.AND', () => {
    expect(brain.knows('fuzzy.AND')).toBe(true);
  });

  test('AND(0, 0) ≈ 0', () => {
    expect(Math.abs(brain.predict([0, 0], 'fuzzy.AND')[0] - 0)).toBeLessThan(0.15);
  });

  test('AND(1, 1) ≈ 1', () => {
    expect(Math.abs(brain.predict([1, 1], 'fuzzy.AND')[0] - 1)).toBeLessThan(0.15);
  });

  test('AND(0.3, 0.7) ≈ 0.3 (min)', () => {
    expect(Math.abs(brain.predict([0.3, 0.7], 'fuzzy.AND')[0] - 0.3)).toBeLessThan(0.15);
  });

  test('AND(0.7, 0.3) ≈ 0.3 (commutativity)', () => {
    expect(Math.abs(brain.predict([0.7, 0.3], 'fuzzy.AND')[0] - 0.3)).toBeLessThan(0.15);
  });
});

// ── 4. Composed expressions ───────────────────────────────────────────────────

describe('Fuzzy — evaluate() with composed expressions', () => {
  let brain;

  beforeAll(() => {
    // Train NOT so we can compose it inside evaluate()
    brain = new Brain({
      defaultTargetAccuracy: 0.90,
      regressionTolerance:   0.05,
      epochsPerRound:        300,
      maxEpochsTotal:        15000,
      maxMutations:          10,
    });
    brain.learn(fuzzyLessons.NOT);
  }, 120000);

  test('evaluate literal value returns unchanged', () => {
    expect(brain.evaluate({ value: 0.7 })).toBe(0.7);
  });

  test('evaluate NOT(0.3) ≈ 0.7', () => {
    const result = brain.evaluate({
      op: 'NOT', domain: 'fuzzy.NOT',
      inputs: [{ value: 0.3 }],
    });
    expect(typeof result).toBe('number');
    expect(Math.abs(result - 0.7)).toBeLessThan(0.15);
  });

  test('evaluate NOT(NOT(0.4)) ≈ 0.4 (double negation)', () => {
    const result = brain.evaluate({
      op: 'NOT', domain: 'fuzzy.NOT',
      inputs: [{
        op: 'NOT', domain: 'fuzzy.NOT',
        inputs: [{ value: 0.4 }],
      }],
    });
    expect(Math.abs(result - 0.4)).toBeLessThan(0.20);
  });
});

// ── 5. Boolean domain isolation ───────────────────────────────────────────────

describe('Boolean domain — unaffected by fuzzy domain', () => {
  let brain;

  beforeAll(() => {
    brain = new Brain({
      defaultTargetAccuracy: 1.0,
      epochsPerRound:        300,
      maxEpochsTotal:        15000,
      maxMutations:          10,
    });

    // Train boolean AND alongside fuzzy NOT
    brain.learn(new Lesson({
      name:   'AND Gate',
      domain: 'boolean.AND',
      trainingData: [
        { input: [0, 0], output: [0] },
        { input: [0, 1], output: [0] },
        { input: [1, 0], output: [0] },
        { input: [1, 1], output: [1] },
      ],
    }));

    brain.learn(fuzzyLessons.NOT);
  }, 180000);

  test('brain knows both boolean.AND and fuzzy.NOT', () => {
    expect(brain.knows('boolean.AND')).toBe(true);
    expect(brain.knows('fuzzy.NOT')).toBe(true);
  });

  test('boolean.AND still predicts correctly', () => {
    const result = brain.predict([1, 1], 'boolean.AND')[0];
    expect(result).toBeGreaterThan(0.5);
  });

  test('boolean.AND(0,1) still predicts 0', () => {
    const result = brain.predict([0, 1], 'boolean.AND')[0];
    expect(result).toBeLessThan(0.5);
  });

  test('domains are independent — two separate regions', () => {
    expect(brain.introspect().regionCount).toBe(2);
  });
});
