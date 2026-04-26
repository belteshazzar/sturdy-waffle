'use strict';

const Brain = require('../src/brain/Brain');
const Lesson = require('../src/learning/Lesson');
const FactBase = require('../src/knowledge/FactBase');
const ExpressionParser = require('../src/parsing/ExpressionParser');

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

  test('processInput handles tokens, knowledge, and query paths', () => {
    const brain = new Brain();
    brain.learn(andLesson);
    const tokens = ExpressionParser.expressionToTokens(
      ExpressionParser.parseExpression('AND(1,1)')
    );
    const tokenResult = brain.processInput(tokens, { execute: false });
    expect(tokenResult.normalized.kind).toBe('tokens');

    const factBase = new FactBase('Animals');
    factBase.assert('bird', 'canFly', true);
    brain.learnFacts(factBase);
    const queryResult = brain.processInput('Is bird canFly?');
    expect(queryResult.result[0].value).toBe(1);
  });
});
