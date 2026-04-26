'use strict';

const Brain = require('../src/brain/Brain');
const Lesson = require('../src/learning/Lesson');

describe('Unified input pipeline', () => {
  const andLesson = new Lesson({
    name: 'AND',
    domain: 'boolean.AND',
    description: 'Learn AND truth table.',
    trainingData: [
      { input: [0, 0], output: [0] },
      { input: [0, 1], output: [0] },
      { input: [1, 0], output: [0] },
      { input: [1, 1], output: [1] },
    ],
    inputSize: 2,
    outputSize: 1,
    mode: 'classification',
  });

  test('normalizeInput handles knowledge statements and queries', () => {
    const brain = new Brain();
    const knowledge = brain.normalizeInput('fact: bird canFly');
    expect(knowledge.kind).toBe('knowledge');
    const query = brain.normalizeInput('Is bird canFly?');
    expect(query.kind).toBe('query');
  });

  test('processInput evaluates expression strings', () => {
    const brain = new Brain();
    brain.learn(andLesson);
    const result = brain.processInput('AND(1,0)').result;
    expect(result).toBe(0);
  });
});
