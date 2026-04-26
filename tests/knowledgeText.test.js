'use strict';

const Brain = require('../src/brain/Brain');
const KnowledgeTextParser = require('../src/parsing/KnowledgeTextParser');

function fastBrain(overrides = {}) {
  return new Brain({
    defaultTargetAccuracy: 0.7,
    epochsPerRound: 150,
    maxEpochsTotal: 3000,
    maxMutations: 6,
    ...overrides,
  });
}

describe('KnowledgeTextParser', () => {
  test('parses facts, attributes, relations, and metadata', () => {
    const text = `
      fact: bird canFly; confidence=0.9; source=manual
      fact: penguin not canFly
      attribute: apple color = red
      attribute: mars radius = 3390; type=numeric
      relation: parentOf(alice,bob) = true; source=family
    `;
    const { statements } = KnowledgeTextParser.parse(text);
    expect(statements).toHaveLength(5);
    expect(statements[0]).toMatchObject({
      kind: 'fact',
      subject: 'bird',
      predicate: 'canFly',
      value: true,
    });
    expect(statements[0].meta.source).toBe('manual');
    expect(statements[1].value).toBe(false);
    expect(statements[3].value).toBe(3390);
    expect(statements[4]).toMatchObject({
      kind: 'relation',
      name: 'parentOf',
      args: ['alice', 'bob'],
    });
  });

  test('parses natural-language and tagged queries', () => {
    const text = `
      Is bird canFly?
      What is apple color?
      Is parentOf(alice,bob)?
      expr: AND(1,0)
    `;
    const { queries } = KnowledgeTextParser.parse(text);
    expect(queries).toHaveLength(4);
    expect(queries[0]).toMatchObject({ kind: 'fact', subject: 'bird', predicate: 'canFly' });
    expect(queries[1]).toMatchObject({ kind: 'attribute', subject: 'apple', attribute: 'color' });
    expect(queries[2]).toMatchObject({ kind: 'relation', name: 'parentOf', args: ['alice', 'bob'] });
    expect(queries[3]).toMatchObject({ kind: 'expression', mode: 'evaluate' });
  });
});

describe('Brain text ingestion', () => {
  test('learnText trains only affected domains and answers queries', () => {
    const brain = fastBrain();
    const result = brain.learnText(`
      fact: bird canFly
      fact: cat canFly = false
      attribute: bird color = red
      relation: parentOf(bird,cat) = true
    `);

    expect(result.trainedDomains).toEqual(
      expect.arrayContaining(['facts.canFly', 'facts.color', 'facts.parentOf'])
    );

    expect(brain.queryFact('bird', 'canFly')).toBe(1);
    expect(brain.queryAttribute('bird', 'color')).toBe('red');
    expect(brain.queryRelation('parentOf', ['bird', 'cat'])).toBe(1);

    const answers = brain.answerText(`
      Is bird canFly?
      What is bird color?
      Is parentOf(bird,cat)?
    `);
    expect(answers[0].value).toBe(1);
    expect(answers[1].value).toBe('red');
    expect(answers[2].value).toBe(1);
  });
});
