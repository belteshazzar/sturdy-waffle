'use strict';

const Lesson = require('../../src/learning/Lesson');
const Syllabus = require('../../src/learning/Syllabus');
const ExpressionParser = require('../../src/parsing/ExpressionParser');
const FactBase = require('../../src/knowledge/FactBase');
const { TOKEN, VOCAB_SIZE } = require('../../src/decomposition/tokens');

const factBase = new FactBase('MixedFacts', { updatePolicy: 'overwrite' });
factBase.assert('bird', 'canFly', true);
factBase.assert('cat', 'canFly', false);
factBase.assertValue('apple', 'color', 'red');
factBase.assertValue('lime', 'color', 'green');
factBase.assertRelation('parentOf', ['alice', 'bob'], true);
factBase.assertRelation('parentOf', ['bob', 'charlie'], true);

const EXPRESSIONS = [
  'ADD(FACT(bird,canFly), 1)',
  'MUL(2, FACT(cat,canFly))',
  'ADD(REL(parentOf,alice,bob), REL(parentOf,bob,charlie))',
  'AND(FACT(bird,canFly), NOT(FACT(cat,canFly)))',
  'ADD(MUL(FACT(bird,canFly), 2), REL(parentOf,alice,bob))',
  'MUL(ADD(1, FACT(cat,canFly)), ADD(REL(parentOf,bob,charlie), 1))',
];

function resolveFact(node) {
  if (node.subject) {
    return factBase.get(node.subject, node.predicate) ?? 0;
  }
  if (node.name) {
    return factBase.getRelation(node.name, node.args) ?? 0;
  }
  return 0;
}

function evalExpression(node) {
  if (node.value !== undefined) return node.value;
  if (node.fact) return resolveFact(node.fact);
  if (node.relation) return resolveFact(node.relation);
  const args = node.inputs.map(evalExpression);
  switch (node.op) {
    case 'ADD': return args[0] + args[1];
    case 'SUB': return args[0] - args[1];
    case 'MUL': return args[0] * args[1];
    case 'DIV': return args[1] === 0 ? 0 : args[0] / args[1];
    case 'AND': return Math.min(args[0], args[1]);
    case 'OR': return Math.max(args[0], args[1]);
    case 'NOT': return 1 - args[0];
    default: throw new Error(`Unknown op ${node.op}`);
  }
}

function encodeTokens(tokens) {
  return tokens.map(tok => {
    const tokenId = tok && typeof tok === 'object' ? tok.token : tok;
    const value = tok && typeof tok === 'object'
      ? tok.value
      : tokenId === TOKEN.V0 ? 0 : tokenId === TOKEN.V1 ? 1 : 0;
    return [tokenId / Math.max(1, VOCAB_SIZE - 1), value];
  });
}

const trainingData = EXPRESSIONS.map(expr => {
  const tree = ExpressionParser.parseExpression(expr);
  const tokens = ExpressionParser.expressionToTokens(tree, { factResolver: resolveFact });
  return {
    input: encodeTokens(tokens),
    output: [evalExpression(tree)],
  };
});

const mixedExpressionLesson = new Lesson({
  name: 'Mixed Domain Expression Solving',
  domain: 'sequence.MIXED_EXPR',
  description: 'Solve mixed-domain expressions including facts and relations.',
  trainingData,
  inputSize: 2,
  outputSize: 1,
  mode: 'regression',
  normalise: { outputRange: [0, 4] },
  sequence: true,
  tags: ['expressions', 'mixed', 'sequence'],
});

const mixedExpressionSyllabus = new Syllabus({
  name: 'Mixed Domain Expressions',
  description: 'Train sequence reasoning across boolean, math, facts, and relations.',
  lessons: [mixedExpressionLesson],
  tags: ['expressions', 'mixed'],
});

module.exports = {
  mixedExpressionLesson,
  mixedExpressionSyllabus,
  mixedExpressionFactBase: factBase,
};
