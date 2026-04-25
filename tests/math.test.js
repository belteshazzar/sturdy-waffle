'use strict';

/**
 * Tests for regression machinery and the math syllabus.
 *
 * The file is divided into:
 *   1. NeuralNetwork.regressionAccuracy — unit tests
 *   2. Lesson — unit tests for new fields (networkConfig, mode, normalise)
 *   3. BrainRegion — unit tests for normalisation helpers
 *   4. Brain.evaluate — support for explicit domain on expression nodes
 *   5. Math integration — trains ADD and MUL; spot-checks predictions
 */

const NeuralNetwork = require('../src/brain/NeuralNetwork');
const BrainRegion   = require('../src/brain/BrainRegion');
const Brain         = require('../src/brain/Brain');
const Lesson        = require('../src/learning/Lesson');

// ── 1. NeuralNetwork.regressionAccuracy ───────────────────────────────────────

describe('NeuralNetwork — regressionAccuracy', () => {
  test('returns a number in [0, 1]', () => {
    const nn   = new NeuralNetwork({ architecture: [1, 4, 1] });
    const data = [{ input: [0.5], output: [0.5] }];
    const acc  = nn.regressionAccuracy(data, 0.1);
    expect(acc).toBeGreaterThanOrEqual(0);
    expect(acc).toBeLessThanOrEqual(1);
  });

  test('returns 1.0 when all predictions are within tolerance', () => {
    const nn = new NeuralNetwork({
      architecture:    [1, 8, 1],
      learningRate:    0.3,
      outputActivation: 'linear',
    });
    const data = [{ input: [0.5], output: [0.5] }];
    nn.train(data, 3000);
    // After sufficient training, a single-sample regression should be within 0.05
    expect(nn.regressionAccuracy(data, 0.05)).toBe(1.0);
  });

  test('returns 0.0 when tolerance is impossibly small', () => {
    const nn   = new NeuralNetwork({ architecture: [1, 4, 1] });
    const data = [{ input: [0.5], output: [0.9] }];
    // Fresh untrained net predicts ~0.5; exact target 0.9 → outside 1e-9 tolerance
    const acc = nn.regressionAccuracy(data, 1e-9);
    expect(acc).toBe(0.0);
  });

  test('looser tolerance always gives acc >= stricter tolerance', () => {
    const nn   = new NeuralNetwork({ architecture: [2, 4, 1] });
    const data = [
      { input: [0, 0], output: [0.2] },
      { input: [1, 1], output: [0.8] },
    ];
    const strict = nn.regressionAccuracy(data, 0.01);
    const loose  = nn.regressionAccuracy(data, 0.50);
    expect(loose).toBeGreaterThanOrEqual(strict);
  });
});

// ── 2. Lesson — new fields ────────────────────────────────────────────────────

describe('Lesson — networkConfig, mode, normalise', () => {
  const baseOpts = {
    name:        'Test',
    domain:      'test.OP',
    trainingData: [{ input: [0.5], output: [0.5] }],
  };

  test('defaults: mode=classification, networkConfig=null, normalise=null', () => {
    const lesson = new Lesson(baseOpts);
    expect(lesson.mode).toBe('classification');
    expect(lesson.networkConfig).toBeNull();
    expect(lesson.normalise).toBeNull();
  });

  test('accepts mode regression', () => {
    const lesson = new Lesson({ ...baseOpts, mode: 'regression' });
    expect(lesson.mode).toBe('regression');
  });

  test('accepts networkConfig', () => {
    const cfg    = { hiddenActivation: 'tanh', outputActivation: 'linear' };
    const lesson = new Lesson({ ...baseOpts, networkConfig: cfg });
    expect(lesson.networkConfig).toEqual(cfg);
  });

  test('accepts normalise with inputRange and outputRange', () => {
    const lesson = new Lesson({
      ...baseOpts,
      normalise: { inputRange: [0, 10], outputRange: [-1, 1] },
    });
    expect(lesson.normalise.inputRange).toEqual([0, 10]);
    expect(lesson.normalise.outputRange).toEqual([-1, 1]);
  });

  test('accepts normalise with only outputRange', () => {
    const lesson = new Lesson({
      ...baseOpts,
      normalise: { outputRange: [0, 2] },
    });
    expect(lesson.normalise.outputRange).toEqual([0, 2]);
    expect(lesson.normalise.inputRange).toBeUndefined();
  });
});

// ── 3. BrainRegion — normalisation ───────────────────────────────────────────

describe('BrainRegion — normalisation helpers', () => {
  function makeRegion(normalise = null, mode = 'regression') {
    return new BrainRegion({
      domain: 'test.OP',
      lesson: new Lesson({
        name: 'test', domain: 'test.OP',
        trainingData: [{ input: [0], output: [0] }, { input: [10], output: [10] }],
        inputSize: 1, outputSize: 1,
        mode, normalise,
        networkConfig: { hiddenActivation: 'tanh', outputActivation: 'linear' },
      }),
    });
  }

  test('predict returns a vector', () => {
    const region = makeRegion();
    const result = region.predict([5]);
    expect(Array.isArray(result)).toBe(true);
    expect(result).toHaveLength(1);
  });

  test('input normalisation: value in range maps to [0,1] before network', () => {
    const region = makeRegion({ inputRange: [0, 10] });
    // _normalizeInput([5]) should produce [0.5]
    expect(region._normalizeInput([5])).toEqual([0.5]);
    expect(region._normalizeInput([0])).toEqual([0]);
    expect(region._normalizeInput([10])).toEqual([1]);
  });

  test('input normalisation: no inputRange → pass-through', () => {
    const region = makeRegion({ outputRange: [0, 10] });  // no inputRange
    expect(region._normalizeInput([7])).toEqual([7]);
  });

  test('output denormalisation: [0,1] value maps back to raw range', () => {
    const region = makeRegion({ outputRange: [0, 10] });
    expect(region._denormalizeOutput([0])).toEqual([0]);
    expect(region._denormalizeOutput([0.5])).toEqual([5]);
    expect(region._denormalizeOutput([1])).toEqual([10]);
  });

  test('output denormalisation: no outputRange → pass-through', () => {
    const region = makeRegion(null);
    expect(region._denormalizeOutput([0.7])).toEqual([0.7]);
  });

  test('output denormalisation: negative range [-1, 1]', () => {
    const region = makeRegion({ outputRange: [-1, 1] });
    expect(region._denormalizeOutput([0])[0]).toBeCloseTo(-1);
    expect(region._denormalizeOutput([0.5])[0]).toBeCloseTo(0);
    expect(region._denormalizeOutput([1])[0]).toBeCloseTo(1);
  });

  test('normalised training data is pre-computed', () => {
    const region = makeRegion({ inputRange: [0, 10], outputRange: [0, 10] });
    // Raw training: input [0], output [0]  → normalised: [0], [0]
    // Raw training: input [10], output [10] → normalised: [1], [1]
    expect(region._normTrainingData[0].input).toEqual([0]);
    expect(region._normTrainingData[0].output).toEqual([0]);
    expect(region._normTrainingData[1].input).toEqual([1]);
    expect(region._normTrainingData[1].output).toEqual([1]);
  });

  test('region uses tanh hidden activation from networkConfig', () => {
    const region = makeRegion();
    expect(region.network.hiddenActivation).toBe('tanh');
    expect(region.network.outputActivation).toBe('linear');
  });

  test('regression lessons start with higher hidden capacity than classification', () => {
    const sharedData = [
      { input: [0, 0], output: [0] },
      { input: [0, 1], output: [1] },
      { input: [1, 0], output: [1] },
      { input: [1, 1], output: [2] },
    ];

    const regressionRegion = new BrainRegion({
      domain: 'math.ADD',
      lesson: new Lesson({
        name: 'ADD',
        domain: 'math.ADD',
        trainingData: sharedData,
        mode: 'regression',
        normalise: { outputRange: [0, 2] },
        networkConfig: { hiddenActivation: 'tanh', outputActivation: 'linear' },
      }),
    });

    const classificationRegion = new BrainRegion({
      domain: 'boolean.OR',
      lesson: new Lesson({
        name: 'OR',
        domain: 'boolean.OR',
        trainingData: sharedData.map(({ input, output }) => ({
          input,
          output: [output[0] > 0 ? 1 : 0],
        })),
      }),
    });

    expect(regressionRegion.network.architecture[1])
      .toBeGreaterThan(classificationRegion.network.architecture[1]);
  });
});

// ── 4. Brain.evaluate — explicit domain on expression nodes ──────────────────

describe('Brain — evaluate with explicit domain', () => {
  function makeBoolBrain() {
    const brain = new Brain({
      defaultTargetAccuracy: 0.75,
      epochsPerRound:        200,
      maxEpochsTotal:        4000,
      maxMutations:          6,
    });
    const andLesson = new Lesson({
      name: 'AND', domain: 'boolean.AND',
      trainingData: [
        { input: [0, 0], output: [0] },
        { input: [0, 1], output: [0] },
        { input: [1, 0], output: [0] },
        { input: [1, 1], output: [1] },
      ],
    });
    brain.learn(andLesson);
    return brain;
  }

  test('existing boolean expressions still work (no domain field)', () => {
    const brain  = makeBoolBrain();
    const result = brain.evaluate({ op: 'AND', inputs: [{ value: 1 }, { value: 1 }] });
    expect([0, 1]).toContain(result);
  });

  test('evaluate with explicit domain field routes correctly', () => {
    const brain  = makeBoolBrain();
    const result = brain.evaluate({
      op: 'AND', domain: 'boolean.AND',
      inputs: [{ value: 1 }, { value: 1 }],
    });
    expect([0, 1]).toContain(result);
  });

  test('evaluate throws for unknown explicit domain', () => {
    const brain = new Brain();
    expect(() => brain.evaluate({
      op: 'ADD', domain: 'math.ADD',
      inputs: [{ value: 0.3 }, { value: 0.5 }],
    })).toThrow();
  });

  test('evaluate can resolve a unique non-boolean op without explicit domain', () => {
    const brain = new Brain({
      defaultTargetAccuracy: 0.7,
      regressionTolerance:   0.08,
      epochsPerRound:        200,
      maxEpochsTotal:        6000,
      maxMutations:          6,
    });

    const pts = [0, 0.25, 0.5, 0.75, 1];
    const addData = [];
    for (const a of pts) for (const b of pts) addData.push({ input: [a, b], output: [a + b] });

    brain.learn(new Lesson({
      name:         'ADD',
      domain:       'math.ADD',
      trainingData: addData,
      mode:         'regression',
      normalise:    { outputRange: [0, 2] },
      networkConfig: { hiddenActivation: 'tanh', outputActivation: 'linear' },
    }));

    const result = brain.evaluate({
      op: 'ADD',
      inputs: [{ value: 0.2 }, { value: 0.3 }],
    });
    expect(typeof result).toBe('number');
    expect(Math.abs(result - 0.5)).toBeLessThan(0.15);
  });

  test('evaluate requires explicit domain when multiple learned domains share an op', () => {
    const brain = new Brain({
      defaultTargetAccuracy: 0.6,
      regressionTolerance:   0.1,
      epochsPerRound:        100,
      maxEpochsTotal:        3000,
      maxMutations:          4,
    });

    const data = [
      { input: [0, 0], output: [0] },
      { input: [0, 1], output: [1] },
      { input: [1, 0], output: [1] },
      { input: [1, 1], output: [2] },
    ];

    const mkLesson = domain => new Lesson({
      name:         domain,
      domain,
      trainingData: data,
      mode:         'regression',
      normalise:    { outputRange: [0, 2] },
      networkConfig: { hiddenActivation: 'tanh', outputActivation: 'linear' },
    });

    brain.learn(mkLesson('math.ADD'));
    brain.learn(mkLesson('algebra.ADD'));

    expect(() => brain.evaluate({
      op: 'ADD',
      inputs: [{ value: 0.2 }, { value: 0.3 }],
    })).toThrow(/Unable to resolve domain/);
  });

  test('evaluate returns literal value unchanged', () => {
    expect(new Brain().evaluate({ value: 0.42 })).toBe(0.42);
  });
});

// ── 5. Math integration — ADD and MUL ────────────────────────────────────────

describe('Math — ADD lesson (regression integration)', () => {
  let brain;

  beforeAll(() => {
    brain = new Brain({
      defaultTargetAccuracy: 0.85,
      regressionTolerance:   0.05,
      epochsPerRound:        300,
      maxEpochsTotal:        20000,
      maxMutations:          10,
    });

    // Small 7×7 = 49-sample grid — enough to test generalisation
    const pts = Array.from({ length: 7 }, (_, i) => i / 6);
    const addData = [];
    for (const a of pts) for (const b of pts) addData.push({ input: [a, b], output: [a + b] });

    const addLesson = new Lesson({
      name:         'ADD',
      domain:       'math.ADD',
      trainingData: addData,
      mode:         'regression',
      normalise:    { outputRange: [0, 2] },
      networkConfig: { hiddenActivation: 'tanh', outputActivation: 'linear' },
    });

    brain.learn(addLesson);
  }, 300000 /* 5 min max */);

  test('brain learned ADD', () => {
    expect(brain.knows('math.ADD')).toBe(true);
  });

  test('ADD(0.0, 0.0) ≈ 0.0', () => {
    expect(Math.abs(brain.predict([0, 0], 'math.ADD')[0] - 0.0)).toBeLessThan(0.15);
  });

  test('ADD(0.5, 0.5) ≈ 1.0', () => {
    expect(Math.abs(brain.predict([0.5, 0.5], 'math.ADD')[0] - 1.0)).toBeLessThan(0.15);
  });

  test('ADD(1.0, 1.0) ≈ 2.0', () => {
    expect(Math.abs(brain.predict([1, 1], 'math.ADD')[0] - 2.0)).toBeLessThan(0.15);
  });

  test('ADD(0.3, 0.5) ≈ 0.8 (unseen input)', () => {
    expect(Math.abs(brain.predict([0.3, 0.5], 'math.ADD')[0] - 0.8)).toBeLessThan(0.15);
  });

  test('evaluate with explicit domain returns continuous value', () => {
    const result = brain.evaluate({
      op: 'ADD', domain: 'math.ADD',
      inputs: [{ value: 0.3 }, { value: 0.5 }],
    });
    expect(typeof result).toBe('number');
    expect(Math.abs(result - 0.8)).toBeLessThan(0.15);
  });

  test('composed expression: ADD(ADD(0.1, 0.2), 0.3) ≈ 0.6', () => {
    const result = brain.evaluate({
      op: 'ADD', domain: 'math.ADD',
      inputs: [
        { op: 'ADD', domain: 'math.ADD', inputs: [{ value: 0.1 }, { value: 0.2 }] },
        { value: 0.3 },
      ],
    });
    expect(Math.abs(result - 0.6)).toBeLessThan(0.25);   // wider tolerance for two hops
  });
});
