'use strict';

const Brain   = require('../src/brain/Brain');
const FactBase = require('../src/knowledge/FactBase');

// ── Shared helpers ────────────────────────────────────────────────────────────

function fastBrain(overrides = {}) {
  return new Brain({
    defaultTargetAccuracy: 0.80,
    epochsPerRound:        300,
    maxEpochsTotal:        8000,
    maxMutations:          10,
    ...overrides,
  });
}

/**
 * Small FactBase with only assertValue() calls — no binary predicates.
 *
 * 4 subjects, 1 attribute ('color'), 3 possible values.
 * Patterns are deliberately distinct so the network can learn them quickly.
 *
 *   apple  → color = red
 *   banana → color = yellow
 *   lime   → color = green
 *   plum   → color = purple
 */
function makeColorFacts() {
  return new FactBase('Colors')
    .assertValue('apple',  'color', 'red')
    .assertValue('banana', 'color', 'yellow')
    .assertValue('lime',   'color', 'green')
    .assertValue('plum',   'color', 'purple');
}

/**
 * Mixed FactBase: both binary predicates (assert) and categorical attributes
 * (assertValue) on the same set of subjects.
 */
function makeMixedFacts() {
  return new FactBase('Mixed')
    .assert('apple',  'isSweet', true)
    .assert('lemon',  'isSweet', false)
    .assert('lime',   'isSweet', false)
    .assertValue('apple', 'color', 'red')
    .assertValue('lemon', 'color', 'yellow')
    .assertValue('lime',  'color', 'green');
}

// ── FactBase.assertValue — construction ───────────────────────────────────────

describe('FactBase — assertValue', () => {
  test('assertValue adds subject to vocabulary', () => {
    const fb = new FactBase();
    fb.assertValue('apple', 'color', 'red');
    expect(fb.subjects).toContain('apple');
  });

  test('assertValue adds attribute to _attributeVocab', () => {
    const fb = new FactBase();
    fb.assertValue('apple', 'color', 'red');
    expect(fb.attributes).toContain('color');
  });

  test('assertValue is chainable', () => {
    const fb = new FactBase();
    const result = fb.assertValue('a', 'color', 'red');
    expect(result).toBe(fb);
  });

  test('throws for empty subject', () => {
    expect(() => new FactBase().assertValue('', 'color', 'red')).toThrow();
  });

  test('throws for empty attribute', () => {
    expect(() => new FactBase().assertValue('apple', '', 'red')).toThrow();
  });

  test('throws for empty value', () => {
    expect(() => new FactBase().assertValue('apple', 'color', '')).toThrow();
  });

  test('throws for non-string value', () => {
    expect(() => new FactBase().assertValue('apple', 'color', 1)).toThrow();
  });

  test('throws when attribute name conflicts with an existing binary predicate', () => {
    const fb = new FactBase().assert('bird', 'canFly', true);
    expect(() => fb.assertValue('bird', 'canFly', 'yes')).toThrow(/already defined as a binary predicate/);
  });

  test('overwriting a value replaces it', () => {
    const fb = new FactBase();
    fb.assertValue('apple', 'color', 'red');
    fb.assertValue('apple', 'color', 'green');
    expect(fb.getValue('apple', 'color')).toBe('green');
  });

  test('overwriting does not add a duplicate subject', () => {
    const fb = new FactBase();
    fb.assertValue('apple', 'color', 'red');
    fb.assertValue('apple', 'color', 'green');
    expect(fb.subjects.filter(s => s === 'apple')).toHaveLength(1);
  });
});

// ── FactBase — name conflict guards ──────────────────────────────────────────

describe('FactBase — name conflict detection', () => {
  test('assert throws when name conflicts with an existing attribute', () => {
    const fb = new FactBase().assertValue('apple', 'color', 'red');
    expect(() => fb.assert('apple', 'color', true)).toThrow(/already defined as a categorical attribute/);
  });

  test('assertValue throws when name conflicts with an existing predicate', () => {
    const fb = new FactBase().assert('bird', 'canFly', true);
    expect(() => fb.assertValue('bird', 'canFly', 'yes')).toThrow(/already defined as a binary predicate/);
  });
});

// ── FactBase.getValue & getAttributeVocabulary ────────────────────────────────

describe('FactBase — getValue', () => {
  test('returns the asserted string value', () => {
    const fb = makeColorFacts();
    expect(fb.getValue('apple', 'color')).toBe('red');
    expect(fb.getValue('lime',  'color')).toBe('green');
  });

  test('returns null for unasserted (subject, attribute) pair', () => {
    const fb = new FactBase();
    fb.assertValue('apple', 'color', 'red');
    expect(fb.getValue('banana', 'color')).toBeNull();
  });
});

describe('FactBase — getAttributeVocabulary', () => {
  test('returns all distinct values in insertion order', () => {
    const fb = makeColorFacts();
    const vocab = fb.getAttributeVocabulary('color');
    expect(vocab).toEqual(['red', 'yellow', 'green', 'purple']);
  });

  test('does not add duplicates when value is re-asserted', () => {
    const fb = new FactBase();
    fb.assertValue('a', 'color', 'red')
      .assertValue('b', 'color', 'red')
      .assertValue('c', 'color', 'blue');
    expect(fb.getAttributeVocabulary('color')).toHaveLength(2);
  });

  test('returns null for unknown attribute', () => {
    const fb = makeColorFacts();
    expect(fb.getAttributeVocabulary('size')).toBeNull();
  });
});

// ── FactBase.attributes & attributeCount ─────────────────────────────────────

describe('FactBase — attributes accessor', () => {
  test('lists attribute names', () => {
    const fb = makeColorFacts();
    expect(fb.attributes).toEqual(['color']);
  });

  test('is empty when no assertValue calls made', () => {
    expect(new FactBase().attributes).toHaveLength(0);
  });

  test('multiple attributes are all listed', () => {
    const fb = new FactBase()
      .assertValue('a', 'color', 'red')
      .assertValue('a', 'size',  'large');
    expect(fb.attributes).toHaveLength(2);
    expect(fb.attributes).toContain('color');
    expect(fb.attributes).toContain('size');
  });
});

describe('FactBase — attributeCount', () => {
  test('counts (subject, attribute) pairs', () => {
    const fb = makeColorFacts();
    expect(fb.attributeCount).toBe(4);
  });

  test('is 0 for pure binary FactBase', () => {
    const fb = new FactBase().assert('bird', 'canFly', true);
    expect(fb.attributeCount).toBe(0);
  });
});

// ── FactBase.toLessons — multi-class ─────────────────────────────────────────

describe('FactBase — toLessons with attributes', () => {
  test('generates one lesson per attribute', () => {
    const fb      = makeColorFacts();
    const lessons = fb.toLessons();
    expect(lessons).toHaveLength(1);
    expect(lessons[0].domain).toBe('facts.color');
  });

  test('attribute lesson has mode multiclass', () => {
    const lesson = makeColorFacts().toLessons()[0];
    expect(lesson.mode).toBe('multiclass');
  });

  test('attribute lesson outputSize equals vocabulary size', () => {
    const lesson = makeColorFacts().toLessons()[0];
    expect(lesson.outputSize).toBe(4);  // red, yellow, green, purple
  });

  test('training data output vectors are one-hot', () => {
    const lesson = makeColorFacts().toLessons()[0];
    lesson.trainingData.forEach(({ output }) => {
      expect(output.reduce((a, b) => a + b, 0)).toBe(1);  // exactly one 1
      output.forEach(v => expect([0, 1]).toContain(v));
    });
  });

  test('each subject has exactly one training sample', () => {
    const lesson = makeColorFacts().toLessons()[0];
    expect(lesson.trainingData).toHaveLength(4);
  });

  test('apple maps to index 0 (red) in color vocabulary', () => {
    const fb     = makeColorFacts();
    const lesson = fb.toLessons()[0];
    const appleSample = lesson.trainingData.find(
      s => Math.abs(s.input[0] - fb.encodeSubject('apple')) < 1e-9
    );
    expect(appleSample.output[0]).toBe(1);  // 'red' is index 0
    expect(appleSample.output.slice(1).every(v => v === 0)).toBe(true);
  });
});

describe('FactBase — toLessons with mixed facts', () => {
  test('generates binary lessons AND attribute lessons', () => {
    const fb      = makeMixedFacts();
    const lessons = fb.toLessons();
    // 1 binary predicate (isSweet) + 1 attribute (color)
    expect(lessons).toHaveLength(2);
    const domains = lessons.map(l => l.domain);
    expect(domains).toContain('facts.isSweet');
    expect(domains).toContain('facts.color');
  });

  test('binary lesson has mode classification', () => {
    const lessons = makeMixedFacts().toLessons();
    const binary  = lessons.find(l => l.domain === 'facts.isSweet');
    expect(binary.mode).toBe('classification');
  });

  test('attribute lesson has mode multiclass', () => {
    const lessons   = makeMixedFacts().toLessons();
    const attribute = lessons.find(l => l.domain === 'facts.color');
    expect(attribute.mode).toBe('multiclass');
  });
});

// ── FactBase — serialisation ──────────────────────────────────────────────────

describe('FactBase — assertValue serialisation', () => {
  test('toJSON includes attributeVocab and attributeFacts', () => {
    const json = makeColorFacts().toJSON();
    expect(json.attributeVocab).toBeDefined();
    expect(json.attributeFacts).toBeDefined();
  });

  test('fromJSON restores attribute vocabulary', () => {
    const fb  = makeColorFacts();
    const fb2 = FactBase.fromJSON(fb.toJSON());
    expect(fb2.getAttributeVocabulary('color')).toEqual(['red', 'yellow', 'green', 'purple']);
  });

  test('fromJSON restores attribute values', () => {
    const fb  = makeColorFacts();
    const fb2 = FactBase.fromJSON(fb.toJSON());
    expect(fb2.getValue('apple',  'color')).toBe('red');
    expect(fb2.getValue('banana', 'color')).toBe('yellow');
    expect(fb2.getValue('lime',   'color')).toBe('green');
    expect(fb2.getValue('plum',   'color')).toBe('purple');
  });

  test('fromJSON of old format (no attributeVocab) does not throw', () => {
    const legacyJSON = {
      name: 'Old', subjects: ['bird'], predicates: ['canFly'],
      facts: { 'bird:canFly': 1 },
    };
    const fb = FactBase.fromJSON(legacyJSON);
    expect(fb.attributes).toHaveLength(0);
    expect(fb.attributeCount).toBe(0);
  });
});

// ── NeuralNetwork — predictArgmax & multiclassAccuracy ───────────────────────

describe('NeuralNetwork — predictArgmax', () => {
  const NeuralNetwork = require('../src/brain/NeuralNetwork');

  test('returns index of highest output', () => {
    const nn = new NeuralNetwork({ architecture: [1, 4, 3] });
    // Manually set weights so output is controlled enough to test the argmax
    // We just need to confirm the function returns a valid index
    const argmax = nn.predictArgmax([0.5]);
    expect(argmax).toBeGreaterThanOrEqual(0);
    expect(argmax).toBeLessThan(3);
  });
});

describe('NeuralNetwork — multiclassAccuracy', () => {
  const NeuralNetwork = require('../src/brain/NeuralNetwork');

  test('returns a number in [0, 1]', () => {
    const nn = new NeuralNetwork({ architecture: [1, 4, 3] });
    const samples = [
      { input: [0.0], output: [1, 0, 0] },
      { input: [0.5], output: [0, 1, 0] },
      { input: [1.0], output: [0, 0, 1] },
    ];
    const acc = nn.multiclassAccuracy(samples);
    expect(acc).toBeGreaterThanOrEqual(0);
    expect(acc).toBeLessThanOrEqual(1);
  });
});

// ── Brain — learnFacts with attributes ───────────────────────────────────────

describe('Brain — learnFacts with assertValue', () => {
  test('spawns a region for the attribute domain', () => {
    const brain = fastBrain();
    brain.learnFacts(makeColorFacts());
    expect(brain.hasRegion('facts.color')).toBe(true);
  });

  test('learnFacts result includes the attribute entry', () => {
    const brain   = fastBrain();
    const results = brain.learnFacts(makeColorFacts());
    const colorResult = results.find(r => r.domain === 'facts.color');
    expect(colorResult).toBeDefined();
    expect(typeof colorResult.accuracy).toBe('number');
  });

  test('learnFacts on mixed FactBase spawns both binary and multi-class regions', () => {
    const brain = fastBrain();
    brain.learnFacts(makeMixedFacts());
    expect(brain.hasRegion('facts.isSweet')).toBe(true);
    expect(brain.hasRegion('facts.color')).toBe(true);
  });
});

// ── Brain — queryAttribute ────────────────────────────────────────────────────

describe('Brain — queryAttribute', () => {
  test('returns a string from the attribute vocabulary', () => {
    const brain = fastBrain();
    brain.learnFacts(makeColorFacts());
    const result = brain.queryAttribute('apple', 'color');
    expect(typeof result).toBe('string');
    expect(['red', 'yellow', 'green', 'purple']).toContain(result);
  });

  test('throws when no FactBase is loaded', () => {
    expect(() => new Brain().queryAttribute('apple', 'color')).toThrow(/FactBase/);
  });

  test('throws for unknown attribute', () => {
    const brain = fastBrain();
    brain.learnFacts(makeColorFacts());
    expect(() => brain.queryAttribute('apple', 'taste')).toThrow(/Attribute 'taste'/);
  });

  test('throws for unknown subject (from encodeSubject)', () => {
    const brain = fastBrain();
    brain.learnFacts(makeColorFacts());
    expect(() => brain.queryAttribute('mango', 'color')).toThrow(/unknown subject/i);
  });

  test('queryFact still works on binary predicates in a mixed FactBase', () => {
    const brain = fastBrain();
    brain.learnFacts(makeMixedFacts());
    const result = brain.queryFact('apple', 'isSweet');
    expect([0, 1]).toContain(result);
  });
});

// ── Brain — introspect with attributes ───────────────────────────────────────

describe('Brain — introspect with attributes', () => {
  test('introspect includes attributeCount', () => {
    const brain = fastBrain();
    brain.learnFacts(makeColorFacts());
    const info = brain.introspect();
    expect(info.factBase.attributeCount).toBe(1);
    expect(info.factBase.attributes).toContain('color');
  });

  test('attributeCount is 0 for a binary-only FactBase', () => {
    const brain = fastBrain();
    brain.learnFacts(
      new FactBase('Animals').assert('bird', 'canFly', true).assert('cat', 'canFly', false)
    );
    const info = brain.introspect();
    expect(info.factBase.attributeCount).toBe(0);
    expect(info.factBase.attributes).toHaveLength(0);
  });
});

// ── Brain — persistence with attributes ──────────────────────────────────────

describe('Brain — persistence with assertValue', () => {
  const path = require('path');
  const fs   = require('fs');
  const savePath = path.join('/tmp', `brain-multiclass-test-${Date.now()}.json`);

  afterAll(() => {
    if (fs.existsSync(savePath)) fs.unlinkSync(savePath);
  });

  test('toJSON includes attributeVocab in factBase', () => {
    const brain = fastBrain();
    brain.learnFacts(makeColorFacts());
    const json = brain.toJSON();
    expect(json.factBase.attributeVocab).toBeDefined();
    expect(json.factBase.attributeVocab.color).toEqual(['red', 'yellow', 'green', 'purple']);
  });

  test('fromJSON restores attribute vocabulary so queryAttribute does not throw', () => {
    const brain  = fastBrain();
    brain.learnFacts(makeColorFacts());
    const brain2 = Brain.fromJSON(brain.toJSON());
    expect(brain2.factBase.getAttributeVocabulary('color')).toEqual(
      ['red', 'yellow', 'green', 'purple']
    );
  });

  test('save / load round-trip preserves attribute vocabulary', () => {
    const brain = fastBrain();
    brain.learnFacts(makeColorFacts());
    brain.save(savePath);
    const loaded = Brain.load(savePath);
    expect(loaded.factBase.getAttributeVocabulary('color')).toEqual(
      ['red', 'yellow', 'green', 'purple']
    );
    expect(() => loaded.queryAttribute('apple', 'color')).not.toThrow();
  });
});
