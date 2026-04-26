'use strict';

const Brain   = require('../src/brain/Brain');
const { colorFacts,  colorSyllabus  } = require('../syllabi/facts/colorFacts');
const { shapeFacts,  shapeSyllabus  } = require('../syllabi/facts/shapeFacts');
const { sizeFacts,   sizeSyllabus   } = require('../syllabi/facts/sizeFacts');
const { worldFacts,  worldSyllabus  } = require('../syllabi/facts/worldFacts');
const { vehicleFacts, vehicleSyllabus } = require('../syllabi/facts/vehicleFacts');

// ── Shared helpers ────────────────────────────────────────────────────────────

/** Reduced accuracy target so individual lessons train quickly in tests. */
function fastBrain(overrides = {}) {
  return new Brain({
    defaultTargetAccuracy: 0.75,
    epochsPerRound:        200,
    maxEpochsTotal:        6000,
    maxMutations:          8,
    ...overrides,
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// colorFacts — structural tests (no neural network)
// ─────────────────────────────────────────────────────────────────────────────

describe('colorFacts — FactBase structure', () => {
  test('has eight subjects', () => {
    expect(colorFacts.subjects).toHaveLength(8);
  });

  test('has five predicates', () => {
    expect(colorFacts.predicates).toHaveLength(5);
    expect(colorFacts.predicates).toEqual(
      expect.arrayContaining(['isRed', 'isGreen', 'isYellow', 'isOrange', 'isPurple'])
    );
  });

  test('apple is red', () => {
    expect(colorFacts.get('apple', 'isRed')).toBe(1);
  });

  test('orange is not red', () => {
    expect(colorFacts.get('orange', 'isRed')).toBe(0);
  });

  test('banana is yellow', () => {
    expect(colorFacts.get('banana', 'isYellow')).toBe(1);
  });

  test('lime is green', () => {
    expect(colorFacts.get('lime', 'isGreen')).toBe(1);
  });

  test('grape is purple', () => {
    expect(colorFacts.get('grape', 'isPurple')).toBe(1);
  });

  test('strawberry is red', () => {
    expect(colorFacts.get('strawberry', 'isRed')).toBe(1);
  });

  test('plum is purple', () => {
    expect(colorFacts.get('plum', 'isPurple')).toBe(1);
  });
});

describe('colorSyllabus — structure', () => {
  test('syllabus has five lessons', () => {
    expect(colorSyllabus.lessons).toHaveLength(5);
  });

  test('lesson domains follow facts.<predicate> pattern', () => {
    const domains = colorSyllabus.lessons.map(l => l.domain);
    expect(domains).toEqual(
      expect.arrayContaining([
        'facts.isRed', 'facts.isGreen', 'facts.isYellow',
        'facts.isOrange', 'facts.isPurple',
      ])
    );
  });

  test('all lessons use classification mode', () => {
    colorSyllabus.lessons.forEach(l => expect(l.mode).toBe('classification'));
  });

  test('each lesson has one training sample per subject', () => {
    colorSyllabus.lessons.forEach(l => expect(l.trainingData).toHaveLength(8));
  });
});

describe('Brain — learnFacts with colorFacts', () => {
  test('trains one region per color predicate', () => {
    const brain = fastBrain();
    brain.learnFacts(colorFacts);
    ['isRed', 'isGreen', 'isYellow', 'isOrange', 'isPurple'].forEach(p => {
      expect(brain.hasRegion(`facts.${p}`)).toBe(true);
    });
  });

  test('queryFact returns 0 or 1 for color predicates', () => {
    const brain = fastBrain();
    brain.learnFacts(colorFacts);
    const result = brain.queryFact('apple', 'isRed');
    expect([0, 1]).toContain(result);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// shapeFacts — structural tests (no neural network)
// ─────────────────────────────────────────────────────────────────────────────

describe('shapeFacts — FactBase structure', () => {
  test('has eight subjects', () => {
    expect(shapeFacts.subjects).toHaveLength(8);
  });

  test('has four predicates', () => {
    expect(shapeFacts.predicates).toHaveLength(4);
    expect(shapeFacts.predicates).toEqual(
      expect.arrayContaining(['hasCorners', 'isRound', 'is3D', 'hasFlatFace'])
    );
  });

  test('circle has no corners', () => {
    expect(shapeFacts.get('circle', 'hasCorners')).toBe(0);
  });

  test('square has corners', () => {
    expect(shapeFacts.get('square', 'hasCorners')).toBe(1);
  });

  test('sphere is round', () => {
    expect(shapeFacts.get('sphere', 'isRound')).toBe(1);
  });

  test('cube is 3D', () => {
    expect(shapeFacts.get('cube', 'is3D')).toBe(1);
  });

  test('triangle is not 3D', () => {
    expect(shapeFacts.get('triangle', 'is3D')).toBe(0);
  });

  test('cylinder has a flat face', () => {
    expect(shapeFacts.get('cylinder', 'hasFlatFace')).toBe(1);
  });

  test('sphere has no flat face', () => {
    expect(shapeFacts.get('sphere', 'hasFlatFace')).toBe(0);
  });
});

describe('shapeSyllabus — structure', () => {
  test('syllabus has four lessons', () => {
    expect(shapeSyllabus.lessons).toHaveLength(4);
  });

  test('lesson domains follow facts.<predicate> pattern', () => {
    const domains = shapeSyllabus.lessons.map(l => l.domain);
    expect(domains).toEqual(
      expect.arrayContaining(['facts.hasCorners', 'facts.isRound', 'facts.is3D', 'facts.hasFlatFace'])
    );
  });

  test('all lessons use classification mode', () => {
    shapeSyllabus.lessons.forEach(l => expect(l.mode).toBe('classification'));
  });

  test('each lesson has one training sample per subject', () => {
    shapeSyllabus.lessons.forEach(l => expect(l.trainingData).toHaveLength(8));
  });
});

describe('Brain — learnFacts with shapeFacts', () => {
  test('trains one region per shape predicate', () => {
    const brain = fastBrain();
    brain.learnFacts(shapeFacts);
    ['hasCorners', 'isRound', 'is3D', 'hasFlatFace'].forEach(p => {
      expect(brain.hasRegion(`facts.${p}`)).toBe(true);
    });
  });

  test('queryFact returns 0 or 1 for shape predicates', () => {
    const brain = fastBrain();
    brain.learnFacts(shapeFacts);
    const result = brain.queryFact('cube', 'is3D');
    expect([0, 1]).toContain(result);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// sizeFacts — structural tests (no neural network)
// ─────────────────────────────────────────────────────────────────────────────

describe('sizeFacts — FactBase structure', () => {
  test('has nine subjects', () => {
    expect(sizeFacts.subjects).toHaveLength(9);
  });

  test('has three predicates', () => {
    expect(sizeFacts.predicates).toHaveLength(3);
    expect(sizeFacts.predicates).toEqual(
      expect.arrayContaining(['isSmall', 'isMedium', 'isLarge'])
    );
  });

  test('ant is small', () => {
    expect(sizeFacts.get('ant', 'isSmall')).toBe(1);
  });

  test('whale is large', () => {
    expect(sizeFacts.get('whale', 'isLarge')).toBe(1);
  });

  test('cat is medium', () => {
    expect(sizeFacts.get('cat', 'isMedium')).toBe(1);
  });

  test('elephant is not small', () => {
    expect(sizeFacts.get('elephant', 'isSmall')).toBe(0);
  });

  test('dog is not large', () => {
    expect(sizeFacts.get('dog', 'isLarge')).toBe(0);
  });
});

describe('sizeSyllabus — structure', () => {
  test('syllabus has three lessons', () => {
    expect(sizeSyllabus.lessons).toHaveLength(3);
  });

  test('lesson domains follow facts.<predicate> pattern', () => {
    const domains = sizeSyllabus.lessons.map(l => l.domain);
    expect(domains).toEqual(
      expect.arrayContaining(['facts.isSmall', 'facts.isMedium', 'facts.isLarge'])
    );
  });

  test('all lessons use classification mode', () => {
    sizeSyllabus.lessons.forEach(l => expect(l.mode).toBe('classification'));
  });

  test('each lesson has one training sample per subject', () => {
    sizeSyllabus.lessons.forEach(l => expect(l.trainingData).toHaveLength(9));
  });
});

describe('Brain — learnFacts with sizeFacts', () => {
  test('trains one region per size predicate', () => {
    const brain = fastBrain();
    brain.learnFacts(sizeFacts);
    ['isSmall', 'isMedium', 'isLarge'].forEach(p => {
      expect(brain.hasRegion(`facts.${p}`)).toBe(true);
    });
  });

  test('queryFact returns 0 or 1 for size predicates', () => {
    const brain = fastBrain();
    brain.learnFacts(sizeFacts);
    const result = brain.queryFact('whale', 'isLarge');
    expect([0, 1]).toContain(result);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// worldFacts — structural tests (no neural network)
// ─────────────────────────────────────────────────────────────────────────────

describe('worldFacts — FactBase structure', () => {
  test('has seven subjects', () => {
    expect(worldFacts.subjects).toHaveLength(7);
  });

  test('has four predicates', () => {
    expect(worldFacts.predicates).toHaveLength(4);
    expect(worldFacts.predicates).toEqual(
      expect.arrayContaining(['isAStar', 'isAPlanet', 'hasRings', 'hasMoons'])
    );
  });

  test('sun is a star', () => {
    expect(worldFacts.get('sun', 'isAStar')).toBe(1);
  });

  test('earth is a planet', () => {
    expect(worldFacts.get('earth', 'isAPlanet')).toBe(1);
  });

  test('saturn has rings', () => {
    expect(worldFacts.get('saturn', 'hasRings')).toBe(1);
  });

  test('mercury does not have rings', () => {
    expect(worldFacts.get('mercury', 'hasRings')).toBe(0);
  });

  test('earth has moons', () => {
    expect(worldFacts.get('earth', 'hasMoons')).toBe(1);
  });

  test('venus does not have moons', () => {
    expect(worldFacts.get('venus', 'hasMoons')).toBe(0);
  });

  test('sun is not a planet', () => {
    expect(worldFacts.get('sun', 'isAPlanet')).toBe(0);
  });
});

describe('worldSyllabus — structure', () => {
  test('syllabus has four lessons', () => {
    expect(worldSyllabus.lessons).toHaveLength(4);
  });

  test('lesson domains follow facts.<predicate> pattern', () => {
    const domains = worldSyllabus.lessons.map(l => l.domain);
    expect(domains).toEqual(
      expect.arrayContaining(['facts.isAStar', 'facts.isAPlanet', 'facts.hasRings', 'facts.hasMoons'])
    );
  });

  test('all lessons use classification mode', () => {
    worldSyllabus.lessons.forEach(l => expect(l.mode).toBe('classification'));
  });

  test('each lesson has one training sample per subject', () => {
    worldSyllabus.lessons.forEach(l => expect(l.trainingData).toHaveLength(7));
  });
});

describe('Brain — learnFacts with worldFacts', () => {
  test('trains one region per world predicate', () => {
    const brain = fastBrain();
    brain.learnFacts(worldFacts);
    ['isAStar', 'isAPlanet', 'hasRings', 'hasMoons'].forEach(p => {
      expect(brain.hasRegion(`facts.${p}`)).toBe(true);
    });
  });

  test('queryFact returns 0 or 1 for world predicates', () => {
    const brain = fastBrain();
    brain.learnFacts(worldFacts);
    const result = brain.queryFact('saturn', 'hasRings');
    expect([0, 1]).toContain(result);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// vehicleFacts — structural tests (no neural network)
// ─────────────────────────────────────────────────────────────────────────────

describe('vehicleFacts — FactBase structure', () => {
  test('has nine subjects', () => {
    expect(vehicleFacts.subjects).toHaveLength(9);
  });

  test('has four predicates', () => {
    expect(vehicleFacts.predicates).toHaveLength(4);
    expect(vehicleFacts.predicates).toEqual(
      expect.arrayContaining(['isLand', 'isWater', 'isAir', 'isHumanPowered'])
    );
  });

  test('car is land-based', () => {
    expect(vehicleFacts.get('car', 'isLand')).toBe(1);
  });

  test('airplane is airborne', () => {
    expect(vehicleFacts.get('airplane', 'isAir')).toBe(1);
  });

  test('submarine is water-based', () => {
    expect(vehicleFacts.get('submarine', 'isWater')).toBe(1);
  });

  test('bicycle is human powered', () => {
    expect(vehicleFacts.get('bicycle', 'isHumanPowered')).toBe(1);
  });

  test('train is not airborne', () => {
    expect(vehicleFacts.get('train', 'isAir')).toBe(0);
  });
});

describe('vehicleSyllabus — structure', () => {
  test('syllabus has four lessons', () => {
    expect(vehicleSyllabus.lessons).toHaveLength(4);
  });

  test('lesson domains follow facts.<predicate> pattern', () => {
    const domains = vehicleSyllabus.lessons.map(l => l.domain);
    expect(domains).toEqual(
      expect.arrayContaining(['facts.isLand', 'facts.isWater', 'facts.isAir', 'facts.isHumanPowered'])
    );
  });

  test('all lessons use classification mode', () => {
    vehicleSyllabus.lessons.forEach(l => expect(l.mode).toBe('classification'));
  });

  test('each lesson has one training sample per subject', () => {
    vehicleSyllabus.lessons.forEach(l => expect(l.trainingData).toHaveLength(9));
  });
});

describe('Brain — learnFacts with vehicleFacts', () => {
  test('trains one region per vehicle predicate', () => {
    const brain = fastBrain();
    brain.learnFacts(vehicleFacts);
    ['isLand', 'isWater', 'isAir', 'isHumanPowered'].forEach(p => {
      expect(brain.hasRegion(`facts.${p}`)).toBe(true);
    });
  });

  test('queryFact returns 0 or 1 for vehicle predicates', () => {
    const brain = fastBrain();
    brain.learnFacts(vehicleFacts);
    const result = brain.queryFact('sailboat', 'isWater');
    expect([0, 1]).toContain(result);
  });
});
