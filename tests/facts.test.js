'use strict';

const path    = require('path');
const fs      = require('fs');
const Brain   = require('../src/brain/Brain');
const FactBase = require('../src/knowledge/FactBase');
const Lesson  = require('../src/learning/Lesson');

// ── Shared helpers ────────────────────────────────────────────────────────────

/**
 * A fast brain config — lower accuracy target keeps individual fact lessons
 * quick while still exercising the full learn → predict lifecycle.
 */
function fastBrain(overrides = {}) {
  return new Brain({
    defaultTargetAccuracy: 0.75,
    epochsPerRound:        200,
    maxEpochsTotal:        6000,
    maxMutations:          8,
    ...overrides,
  });
}

/**
 * Minimal FactBase used across multiple describe blocks.
 *
 * 4 subjects, 2 predicates — chosen so the classification patterns are
 * easy to learn (small network, low epoch budget).
 *
 *   Subjects: cat (0.0), bird (0.333), dog (0.667), bat (1.0)
 *
 *   canFly :  cat=F  bird=T  dog=F  bat=T   → [0, 1, 0, 1]  (alternating)
 *   hasFur :  cat=T  bird=F  dog=T  bat=T   → [1, 0, 1, 1]
 */
function makeSimpleFacts() {
  return new FactBase('Simple')
    .assert('cat',  'canFly',  false)
    .assert('bird', 'canFly',  true)
    .assert('dog',  'canFly',  false)
    .assert('bat',  'canFly',  true)
    .assert('cat',  'hasFur',  true)
    .assert('bird', 'hasFur',  false)
    .assert('dog',  'hasFur',  true)
    .assert('bat',  'hasFur',  true);
}

// ── FactBase — construction & fact management ─────────────────────────────────

describe('FactBase — construction', () => {
  test('creates an empty fact base with a name', () => {
    const fb = new FactBase('Test');
    expect(fb.name).toBe('Test');
    expect(fb.subjects).toHaveLength(0);
    expect(fb.predicates).toHaveLength(0);
    expect(fb.factCount).toBe(0);
  });

  test('defaults to name "Facts"', () => {
    expect(new FactBase().name).toBe('Facts');
  });
});

describe('FactBase — assert and retrieval', () => {
  test('assert adds subject and predicate to vocabulary', () => {
    const fb = new FactBase();
    fb.assert('bird', 'canFly', true);
    expect(fb.subjects).toContain('bird');
    expect(fb.predicates).toContain('canFly');
  });

  test('get returns 1 for true fact', () => {
    const fb = new FactBase();
    fb.assert('bird', 'canFly', true);
    expect(fb.get('bird', 'canFly')).toBe(1);
  });

  test('get returns 0 for false fact', () => {
    const fb = new FactBase();
    fb.assert('cat', 'canFly', false);
    expect(fb.get('cat', 'canFly')).toBe(0);
  });

  test('get returns null for unasserted fact', () => {
    const fb = new FactBase();
    fb.assert('bird', 'canFly', true);
    expect(fb.get('cat', 'canFly')).toBeNull();
  });

  test('has returns true only for asserted facts', () => {
    const fb = new FactBase();
    fb.assert('bird', 'canFly', true);
    expect(fb.has('bird', 'canFly')).toBe(true);
    expect(fb.has('cat',  'canFly')).toBe(false);
  });

  test('assert is chainable', () => {
    const fb = new FactBase();
    const result = fb.assert('a', 'p', true);
    expect(result).toBe(fb);
  });

  test('overwriting a fact replaces the value', () => {
    const fb = new FactBase();
    fb.assert('bird', 'canFly', true);
    fb.assert('bird', 'canFly', false);
    expect(fb.get('bird', 'canFly')).toBe(0);
  });

  test('assert defaults to true when value omitted', () => {
    const fb = new FactBase();
    fb.assert('bird', 'canFly');
    expect(fb.get('bird', 'canFly')).toBe(1);
  });

  test('throws for empty subject', () => {
    expect(() => new FactBase().assert('', 'canFly')).toThrow();
  });

  test('throws for empty predicate', () => {
    expect(() => new FactBase().assert('bird', '')).toThrow();
  });
});

// ── FactBase — encoding ───────────────────────────────────────────────────────

describe('FactBase — subject encoding', () => {
  test('single subject encodes to 0.5', () => {
    const fb = new FactBase();
    fb.assert('solo', 'p', true);
    expect(fb.encodeSubject('solo')).toBe(0.5);
  });

  test('first of two subjects encodes to 0', () => {
    const fb = new FactBase();
    fb.assert('a', 'p', true).assert('b', 'p', false);
    expect(fb.encodeSubject('a')).toBe(0);
  });

  test('second of two subjects encodes to 1', () => {
    const fb = new FactBase();
    fb.assert('a', 'p', true).assert('b', 'p', false);
    expect(fb.encodeSubject('b')).toBe(1);
  });

  test('four subjects are evenly spaced in [0, 1]', () => {
    const fb = makeSimpleFacts();
    expect(fb.encodeSubject('cat')).toBeCloseTo(0);
    expect(fb.encodeSubject('bird')).toBeCloseTo(1 / 3);
    expect(fb.encodeSubject('dog')).toBeCloseTo(2 / 3);
    expect(fb.encodeSubject('bat')).toBeCloseTo(1);
  });

  test('throws for unknown subject', () => {
    const fb = new FactBase();
    fb.assert('bird', 'canFly', true);
    expect(() => fb.encodeSubject('cat')).toThrow(/unknown subject/);
  });
});

// ── FactBase — toLessons ──────────────────────────────────────────────────────

describe('FactBase — toLessons', () => {
  test('generates one lesson per predicate', () => {
    const fb = makeSimpleFacts();
    const lessons = fb.toLessons();
    expect(lessons).toHaveLength(2);
  });

  test('lesson domains follow facts.<predicate> pattern', () => {
    const fb = makeSimpleFacts();
    const domains = fb.toLessons().map(l => l.domain);
    expect(domains).toContain('facts.canFly');
    expect(domains).toContain('facts.hasFur');
  });

  test('lessons use classification mode', () => {
    const fb = makeSimpleFacts();
    fb.toLessons().forEach(l => expect(l.mode).toBe('classification'));
  });

  test('each lesson has one training sample per subject', () => {
    const fb = makeSimpleFacts();   // 4 subjects
    fb.toLessons().forEach(l => expect(l.trainingData).toHaveLength(4));
  });

  test('training data inputs are normalised scalars in [0, 1]', () => {
    const fb = makeSimpleFacts();
    fb.toLessons().forEach(lesson => {
      lesson.trainingData.forEach(({ input }) => {
        expect(input).toHaveLength(1);
        expect(input[0]).toBeGreaterThanOrEqual(0);
        expect(input[0]).toBeLessThanOrEqual(1);
      });
    });
  });

  test('throws when no subjects have been asserted', () => {
    expect(() => new FactBase().toLessons()).toThrow();
  });
});

// ── FactBase — serialisation ──────────────────────────────────────────────────

describe('FactBase — serialisation', () => {
  test('toJSON / fromJSON round-trip preserves all fields', () => {
    const fb  = makeSimpleFacts();
    const fb2 = FactBase.fromJSON(fb.toJSON());
    expect(fb2.name).toBe(fb.name);
    expect(fb2.subjects).toEqual(fb.subjects);
    expect(fb2.predicates).toEqual(fb.predicates);
    expect(fb2.get('bird', 'canFly')).toBe(1);
    expect(fb2.get('cat',  'canFly')).toBe(0);
  });

  test('encoding is preserved after round-trip', () => {
    const fb  = makeSimpleFacts();
    const fb2 = FactBase.fromJSON(fb.toJSON());
    expect(fb2.encodeSubject('bird')).toBeCloseTo(fb.encodeSubject('bird'));
  });
});

// ── Brain — learnFacts & queryFact ────────────────────────────────────────────

describe('Brain — learnFacts', () => {
  test('learnFacts trains one region per predicate', () => {
    const brain = fastBrain();
    brain.learnFacts(makeSimpleFacts());
    expect(brain.hasRegion('facts.canFly')).toBe(true);
    expect(brain.hasRegion('facts.hasFur')).toBe(true);
  });

  test('learnFacts stores the factBase reference', () => {
    const brain = fastBrain();
    const fb    = makeSimpleFacts();
    brain.learnFacts(fb);
    expect(brain.factBase).toBe(fb);
  });

  test('learnFacts returns an array of results with predicate, domain, accuracy', () => {
    const brain   = fastBrain();
    const results = brain.learnFacts(makeSimpleFacts());
    expect(Array.isArray(results)).toBe(true);
    expect(results).toHaveLength(2);
    results.forEach(r => {
      expect(typeof r.predicate).toBe('string');
      expect(typeof r.domain).toBe('string');
      expect(typeof r.accuracy).toBe('number');
    });
  });
});

describe('Brain — queryFact', () => {
  test('queryFact returns 0 or 1', () => {
    const brain = fastBrain();
    brain.learnFacts(makeSimpleFacts());
    const result = brain.queryFact('bird', 'canFly');
    expect([0, 1]).toContain(result);
  });

  test('queryFact throws when no factBase is loaded', () => {
    expect(() => new Brain().queryFact('bird', 'canFly')).toThrow(/FactBase/);
  });

  test('queryFact throws for unknown subject', () => {
    const brain = fastBrain();
    brain.learnFacts(makeSimpleFacts());
    expect(() => brain.queryFact('dragon', 'canFly')).toThrow(/unknown subject/);
  });

  test('queryFact throws for unknown predicate (no region)', () => {
    const brain = fastBrain();
    brain.learnFacts(makeSimpleFacts());
    expect(() => brain.queryFact('bird', 'canBreathe')).toThrow();
  });
});

// ── Brain — evaluate with fact nodes ─────────────────────────────────────────

describe('Brain — evaluate with fact nodes', () => {
  test('evaluates a single fact node', () => {
    const brain = fastBrain();
    brain.learnFacts(makeSimpleFacts());
    const result = brain.evaluate({ fact: { subject: 'cat', predicate: 'canFly' } });
    expect([0, 1]).toContain(result);
  });

  test('fact node throws when no factBase is loaded', () => {
    expect(() =>
      new Brain().evaluate({ fact: { subject: 'bird', predicate: 'canFly' } })
    ).toThrow(/FactBase/);
  });

  test('evaluate throws for node with no value, fact, or op', () => {
    const brain = fastBrain();
    brain.learnFacts(makeSimpleFacts());
    expect(() => brain.evaluate({})).toThrow();
  });
});

// ── Brain — reasoning: fact + boolean logic ───────────────────────────────────

describe('Brain — reasoning with facts and boolean logic', () => {
  test('combine fact lookup with boolean NOT', () => {
    const brain = fastBrain();
    brain.learnFacts(makeSimpleFacts());

    // Teach NOT so we can use it in the expression tree
    brain.learn(new Lesson({
      name:  'NOT',
      domain: 'boolean.NOT',
      trainingData: [
        { input: [0], output: [1] },
        { input: [1], output: [0] },
      ],
    }));

    // NOT(canFly(cat)) — cat cannot fly, so NOT(0) should be 1
    const result = brain.evaluate({
      op:     'NOT',
      inputs: [{ fact: { subject: 'cat', predicate: 'canFly' } }],
    });
    expect([0, 1]).toContain(result);
  });

  test('combine two fact lookups with boolean AND', () => {
    const brain = fastBrain();
    brain.learnFacts(makeSimpleFacts());

    // Teach AND
    brain.learn(new Lesson({
      name:   'AND',
      domain: 'boolean.AND',
      trainingData: [
        { input: [0, 0], output: [0] },
        { input: [0, 1], output: [0] },
        { input: [1, 0], output: [0] },
        { input: [1, 1], output: [1] },
      ],
    }));

    // AND(canFly(bat), hasFur(bat)) — bat can fly AND has fur → should be 1
    const result = brain.evaluate({
      op:     'AND',
      inputs: [
        { fact: { subject: 'bat', predicate: 'canFly' } },
        { fact: { subject: 'bat', predicate: 'hasFur' } },
      ],
    });
    expect([0, 1]).toContain(result);
  });
});

// ── Brain — persistence with factBase ────────────────────────────────────────

describe('Brain — persistence with factBase', () => {
  const savePath = path.join('/tmp', `brain-facts-test-${Date.now()}.json`);

  afterAll(() => {
    if (fs.existsSync(savePath)) fs.unlinkSync(savePath);
  });

  test('toJSON includes factBase when one is loaded', () => {
    const brain = fastBrain();
    brain.learnFacts(makeSimpleFacts());
    const json = brain.toJSON();
    expect(json.factBase).toBeDefined();
    expect(json.factBase.subjects).toContain('bird');
  });

  test('toJSON factBase is null when no facts are loaded', () => {
    const json = new Brain().toJSON();
    expect(json.factBase).toBeNull();
  });

  test('fromJSON restores the factBase vocabulary', () => {
    const brain  = fastBrain();
    brain.learnFacts(makeSimpleFacts());
    const brain2 = Brain.fromJSON(brain.toJSON());
    expect(brain2.factBase).not.toBeNull();
    expect(brain2.factBase.subjects).toEqual(brain.factBase.subjects);
    expect(brain2.factBase.predicates).toEqual(brain.factBase.predicates);
  });

  test('save / load round-trip preserves factBase', () => {
    const brain = fastBrain();
    brain.learnFacts(makeSimpleFacts());
    brain.save(savePath);

    const loaded = Brain.load(savePath);
    expect(loaded.factBase).not.toBeNull();
    expect(loaded.factBase.get('bird', 'canFly')).toBe(1);
    expect(loaded.factBase.get('cat',  'canFly')).toBe(0);
  });

  test('loaded brain can still query facts', () => {
    const brain = fastBrain();
    brain.learnFacts(makeSimpleFacts());
    brain.save(savePath);

    const loaded = Brain.load(savePath);
    // The facts.* regions are restored; queryFact should not throw
    expect(() => loaded.queryFact('bird', 'canFly')).not.toThrow();
  });
});

// ── Brain — introspect includes factBase ─────────────────────────────────────

describe('Brain — introspect with factBase', () => {
  test('introspect returns null factBase when none is loaded', () => {
    expect(new Brain().introspect().factBase).toBeNull();
  });

  test('introspect returns factBase summary when one is loaded', () => {
    const brain = fastBrain();
    brain.learnFacts(makeSimpleFacts());
    const info = brain.introspect();
    expect(info.factBase).not.toBeNull();
    expect(info.factBase.subjectCount).toBe(4);
    expect(info.factBase.predicateCount).toBe(2);
    expect(info.factBase.subjects).toContain('bird');
    expect(info.factBase.predicates).toContain('canFly');
  });
});
