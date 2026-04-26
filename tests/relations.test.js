'use strict';

const Brain = require('../src/brain/Brain');
const FactBase = require('../src/knowledge/FactBase');

describe('FactBase relations', () => {
  test('brain can learn and query relations', () => {
    const fb = new FactBase('Relations');
    fb.assertRelation('parentOf', ['alice', 'bob'])
      .assertRelation('parentOf', ['alice', 'carol'])
      .assertRelation('siblingOf', ['bob', 'carol']);

    const brain = new Brain({ defaultTargetAccuracy: 0.7, epochsPerRound: 150, maxEpochsTotal: 3000 });
    brain.learnFacts(fb);

    expect(brain.queryRelation('parentOf', ['alice', 'bob'])).toBe(1);
  });
});
