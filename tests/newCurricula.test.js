'use strict';

const { arithmeticExpressionSyllabus } = require('../syllabi/expressions/arithmeticExpressions');
const { identitySyllabus } = require('../syllabi/math/identities');
const { fuzzyChainSyllabus } = require('../syllabi/fuzzy/chains');
const { relationSyllabus } = require('../syllabi/relations/relationalPuzzles');

describe('new curricula', () => {
  test('arithmetic expression syllabus uses sequence lesson', () => {
    expect(arithmeticExpressionSyllabus.lessons).toHaveLength(1);
    expect(arithmeticExpressionSyllabus.lessons[0].sequence).toBe(true);
  });

  test('identity syllabus contains algebraic lessons', () => {
    expect(identitySyllabus.lessons).toHaveLength(2);
  });

  test('fuzzy chain syllabus contains compositional lesson', () => {
    expect(fuzzyChainSyllabus.lessons).toHaveLength(1);
    expect(fuzzyChainSyllabus.lessons[0].domain).toBe('fuzzy.CHAIN');
  });

  test('relation syllabus contains relation lessons', () => {
    expect(relationSyllabus.lessons.length).toBeGreaterThan(0);
    expect(relationSyllabus.lessons[0].domain).toMatch(/^facts\./);
  });
});
