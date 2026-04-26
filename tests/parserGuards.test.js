'use strict';

const ExpressionParser = require('../src/parsing/ExpressionParser');
const KnowledgeTextParser = require('../src/parsing/KnowledgeTextParser');

describe('Parser guardrails', () => {
  test('ExpressionParser enforces max token count', () => {
    const longExpr = `ADD(${Array.from({ length: 20 }).map(() => '1').join(',')})`;
    expect(() => ExpressionParser.parseExpression(longExpr, { maxTokens: 5 }))
      .toThrow(/too many tokens/i);
  });

  test('KnowledgeTextParser enforces max lines', () => {
    const text = 'fact: a b\nfact: c d\nfact: e f';
    expect(() => KnowledgeTextParser.parse(text, { maxLines: 2 }))
      .toThrow(/max lines/i);
  });
});
