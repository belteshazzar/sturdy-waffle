'use strict';

const ExpressionParser = require('../src/parsing/ExpressionParser');
const { TOKEN } = require('../src/decomposition/tokens');

describe('ExpressionParser', () => {
  test('parses nested boolean expression', () => {
    const expr = ExpressionParser.parseExpression('AND(OR(1,0), NOT(0))');
    expect(expr.op).toBe('AND');
    expect(expr.inputs).toHaveLength(2);
  });

  test('parses domain-qualified operation', () => {
    const expr = ExpressionParser.parseExpression('math.ADD(1,0)');
    expect(expr.op).toBe('ADD');
    expect(expr.domain).toBe('math.ADD');
  });

  test('parses fact nodes', () => {
    const expr = ExpressionParser.parseExpression('FACT(apple, color)');
    expect(expr.fact).toBeDefined();
    expect(expr.fact.subject).toBe('apple');
    expect(expr.fact.predicate).toBe('color');
  });

  test('tokenizes numeric values with VALUE token', () => {
    const tokens = ExpressionParser.toTokenStream('ADD(2,1)');
    expect(tokens[0].token).toBe(TOKEN.ADD);
    expect(tokens[1].token).toBe(TOKEN.VALUE);
    expect(tokens[1].value).toBe(2);
  });
});
