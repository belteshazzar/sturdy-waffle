'use strict';

const path   = require('path');
const fs     = require('fs');
const Brain  = require('../src/brain/Brain');
const Lesson = require('../src/learning/Lesson');

// ── Shared test lesson (fast: low accuracy target) ────────────────────────────

const andLesson = new Lesson({
  name:        'AND Gate',
  domain:      'boolean.AND',
  description: 'Test AND lesson',
  trainingData: [
    { input: [0, 0], output: [0] },
    { input: [0, 1], output: [0] },
    { input: [1, 0], output: [0] },
    { input: [1, 1], output: [1] },
  ],
});

function fastBrain(overrides = {}) {
  return new Brain({
    defaultTargetAccuracy: 0.75,
    epochsPerRound:        200,
    maxEpochsTotal:        4000,
    maxMutations:          6,
    ...overrides,
  });
}

// ── Construction ──────────────────────────────────────────────────────────────

describe('Brain — construction', () => {
  test('starts with no knowledge', () => {
    const brain = new Brain();
    expect(brain.regions.size).toBe(0);
  });

  test('introspect returns zero regions initially', () => {
    const info = new Brain().introspect();
    expect(info.regionCount).toBe(0);
    expect(info.domains).toEqual([]);
  });

  test('knows() returns false for unknown domain', () => {
    expect(new Brain().knows('boolean.AND')).toBe(false);
  });

  test('hasRegion() returns false for unknown domain', () => {
    expect(new Brain().hasRegion('boolean.AND')).toBe(false);
  });
});

// ── Learning ──────────────────────────────────────────────────────────────────

describe('Brain — learning', () => {
  test('spawns region for new domain', () => {
    const brain = fastBrain();
    brain.learn(andLesson);
    expect(brain.hasRegion('boolean.AND')).toBe(true);
  });

  test('learn returns result with accuracy and mutationCount', () => {
    const brain  = fastBrain();
    const result = brain.learn(andLesson);
    expect(typeof result.accuracy).toBe('number');
    expect(typeof result.mutationCount).toBe('number');
    expect(typeof result.totalEpochs).toBe('number');
  });

  test('calling learn twice on same domain does not spawn duplicate region', () => {
    const brain = fastBrain();
    brain.learn(andLesson);
    brain.learn(andLesson);
    expect(brain.regions.size).toBe(1);
  });

  test('introspect shows region after learning', () => {
    const brain = fastBrain();
    brain.learn(andLesson);
    const info = brain.introspect();
    expect(info.regionCount).toBe(1);
    expect(info.regions['boolean.AND']).toBeDefined();
    expect(info.regions['boolean.AND'].domain).toBe('boolean.AND');
  });
});

// ── Routing & inference ───────────────────────────────────────────────────────

describe('Brain — routing & inference', () => {
  test('predictBinary returns binary array after learning', () => {
    const brain  = fastBrain();
    brain.learn(andLesson);
    const result = brain.predictBinary([1, 1], 'boolean.AND');
    expect(result).toHaveLength(1);
    expect([0, 1]).toContain(result[0]);
  });

  test('predict throws for unknown domain', () => {
    expect(() => new Brain().predict([1, 1], 'boolean.AND')).toThrow();
  });

  test('predictBinary throws for unknown domain', () => {
    expect(() => new Brain().predictBinary([1, 1], 'boolean.AND')).toThrow();
  });
});

// ── evaluate (expression tree) ────────────────────────────────────────────────

describe('Brain — evaluate', () => {
  test('evaluates a literal value node', () => {
    expect(new Brain().evaluate({ value: 0 })).toBe(0);
    expect(new Brain().evaluate({ value: 1 })).toBe(1);
  });

  test('evaluates a nested expression once brain has learned gates', () => {
    const brain = fastBrain();
    brain.learn(andLesson);

    // Teach NOT too so nested test can work
    const notLesson = new Lesson({
      name: 'NOT', domain: 'boolean.NOT',
      trainingData: [{ input: [0], output: [1] }, { input: [1], output: [0] }],
    });
    brain.learn(notLesson);

    const expr = { op: 'AND', inputs: [{ value: 1 }, { op: 'NOT', inputs: [{ value: 0 }] }] };
    const result = brain.evaluate(expr);
    expect([0, 1]).toContain(result);
  });

  test('evaluate throws when op has no matching region', () => {
    const brain = new Brain();
    expect(() => brain.evaluate({ op: 'AND', inputs: [{ value: 1 }, { value: 1 }] })).toThrow();
  });

  test('evaluate throws for malformed node', () => {
    const brain = new Brain();
    expect(() => brain.evaluate({})).toThrow();
  });
});

// ── Persistence ───────────────────────────────────────────────────────────────

describe('Brain — persistence', () => {
  const savePath = path.join('/tmp', `brain-test-${Date.now()}.json`);

  afterAll(() => {
    if (fs.existsSync(savePath)) fs.unlinkSync(savePath);
  });

  test('save creates a JSON file', () => {
    const brain = fastBrain();
    brain.learn(andLesson);
    brain.save(savePath);
    expect(fs.existsSync(savePath)).toBe(true);
  });

  test('loaded brain has the same regions', () => {
    const brain = fastBrain();
    brain.learn(andLesson);
    brain.save(savePath);

    const loaded = Brain.load(savePath);
    expect(loaded.hasRegion('boolean.AND')).toBe(true);
    expect(loaded.regions.size).toBe(1);
  });

  test('loaded brain can predict', () => {
    const brain = fastBrain();
    brain.learn(andLesson);
    brain.save(savePath);

    const loaded = Brain.load(savePath);
    const result = loaded.predictBinary([0, 0], 'boolean.AND');
    expect(result).toHaveLength(1);
    expect([0, 1]).toContain(result[0]);
  });

  test('toJSON / fromJSON preserves config', () => {
    const brain = new Brain({ defaultTargetAccuracy: 0.88 });
    const json  = brain.toJSON();
    const brain2 = Brain.fromJSON(json);
    expect(brain2.config.defaultTargetAccuracy).toBe(0.88);
  });
});

// ── Events ────────────────────────────────────────────────────────────────────

describe('Brain — events', () => {
  test('emits lesson:unknown for new domain', () => {
    const brain  = fastBrain();
    const events = [];
    brain.on('lesson:unknown', e => events.push(e));
    brain.learn(andLesson);
    expect(events).toHaveLength(1);
    expect(events[0].domain).toBe('boolean.AND');
  });

  test('emits region:spawned for new domain', () => {
    const brain  = fastBrain();
    const events = [];
    brain.on('region:spawned', e => events.push(e));
    brain.learn(andLesson);
    expect(events).toHaveLength(1);
    expect(events[0].domain).toBe('boolean.AND');
  });

  test('does NOT emit lesson:unknown for already-known domain', () => {
    const brain  = fastBrain();
    brain.learn(andLesson);
    const events = [];
    brain.on('lesson:unknown', e => events.push(e));
    brain.learn(andLesson);   // second call — region already exists
    expect(events).toHaveLength(0);
  });
});
