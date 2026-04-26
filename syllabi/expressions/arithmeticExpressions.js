'use strict';

const Lesson = require('../../src/learning/Lesson');
const Syllabus = require('../../src/learning/Syllabus');
const ExpressionParser = require('../../src/parsing/ExpressionParser');
const { TOKEN, VOCAB_SIZE } = require('../../src/decomposition/tokens');

const EXPRESSIONS = [
  'ADD(1,0)',
  'SUB(1,0)',
  'MUL(1,1)',
  'DIV(1,1)',
  'ADD(MUL(1,1),1)',
  'ADD(SQRT(1),0)',
  'MUL(ADD(1,0),SUB(1,0))',
];

function evalExpression(node) {
  if (node.value !== undefined) return node.value;
  const args = node.inputs.map(evalExpression);
  switch (node.op) {
    case 'ADD': return args[0] + args[1];
    case 'SUB': return args[0] - args[1];
    case 'MUL': return args[0] * args[1];
    case 'DIV': return args[1] === 0 ? 0 : args[0] / args[1];
    case 'SQRT': return Math.sqrt(Math.max(0, args[0]));
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
  const tokens = ExpressionParser.expressionToTokens(tree);
  return {
    input: encodeTokens(tokens),
    output: [evalExpression(tree)],
  };
});

const arithmeticExpressionLesson = new Lesson({
  name: 'Arithmetic Expression Solving',
  domain: 'sequence.ARITH_EXPR',
  description: 'Solve short prefix arithmetic expressions from token sequences.',
  trainingData,
  inputSize: 2,
  outputSize: 1,
  mode: 'regression',
  normalise: { outputRange: [0, 2] },
  sequence: true,
  tags: ['expressions', 'math', 'sequence'],
});

const arithmeticExpressionSyllabus = new Syllabus({
  name: 'Arithmetic Expression Solving',
  description: 'Learn to solve small arithmetic expressions from token sequences.',
  lessons: [arithmeticExpressionLesson],
  tags: ['expressions', 'math'],
});

module.exports = {
  arithmeticExpressionLesson,
  arithmeticExpressionSyllabus,
};
