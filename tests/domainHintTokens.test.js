'use strict';

const ExpressionParser = require('../src/parsing/ExpressionParser');
const WorkingMemory = require('../src/decomposition/WorkingMemory');

describe('Domain hints in decomposition tokens', () => {
  test('propagates explicit domain into working memory', () => {
    const expr = ExpressionParser.parseExpression('fuzzy.AND(1,0)');
    const tokens = ExpressionParser.expressionToTokens(expr);
    const mem = new WorkingMemory();
    mem.load(tokens);
    expect(tokens[0].domain).toBe('fuzzy.AND');
    expect(mem.domains[0]).toBe('fuzzy.AND');
  });
});
