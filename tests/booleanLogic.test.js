'use strict';

/**
 * Integration tests for the Boolean Logic syllabus.
 *
 * The brain is trained once (in beforeAll) with a 99 % accuracy target and
 * then all gate truth-tables plus nested expressions are verified.
 *
 * Jest timeout is set to 300 s in package.json to accommodate the full
 * training run on slower CI machines.
 */

const Brain                    = require('../src/brain/Brain');
const { booleanLogicSyllabus } = require('../syllabi/booleanLogic');

// ── Shared trained brain ──────────────────────────────────────────────────────

let brain;

beforeAll(() => {
  brain = new Brain({
    defaultTargetAccuracy: 0.99,
    epochsPerRound:        400,
    maxEpochsTotal:        30000,
    maxMutations:          12,
  });
  brain.learnSyllabus(booleanLogicSyllabus);
});

// ── Syllabus coverage ─────────────────────────────────────────────────────────

describe('Boolean Logic — syllabus coverage', () => {
  test('brain has 7 regions after learning syllabus', () => {
    expect(brain.introspect().regionCount).toBe(7);
  });

  test.each(['AND', 'OR', 'NOT', 'XOR', 'NAND', 'NOR', 'XNOR'])(
    'domain boolean.%s exists', gate => {
      expect(brain.hasRegion(`boolean.${gate}`)).toBe(true);
    }
  );
});

// ── Gate truth tables ─────────────────────────────────────────────────────────

describe('Boolean Logic — AND truth table', () => {
  test.each([
    [[0, 0], 0],
    [[0, 1], 0],
    [[1, 0], 0],
    [[1, 1], 1],
  ])('AND(%j) = %i', (inputs, expected) => {
    expect(brain.predictBinary(inputs, 'boolean.AND')[0]).toBe(expected);
  });
});

describe('Boolean Logic — OR truth table', () => {
  test.each([
    [[0, 0], 0],
    [[0, 1], 1],
    [[1, 0], 1],
    [[1, 1], 1],
  ])('OR(%j) = %i', (inputs, expected) => {
    expect(brain.predictBinary(inputs, 'boolean.OR')[0]).toBe(expected);
  });
});

describe('Boolean Logic — NOT truth table', () => {
  test.each([
    [[0], 1],
    [[1], 0],
  ])('NOT(%j) = %i', (inputs, expected) => {
    expect(brain.predictBinary(inputs, 'boolean.NOT')[0]).toBe(expected);
  });
});

describe('Boolean Logic — XOR truth table', () => {
  test.each([
    [[0, 0], 0],
    [[0, 1], 1],
    [[1, 0], 1],
    [[1, 1], 0],
  ])('XOR(%j) = %i', (inputs, expected) => {
    expect(brain.predictBinary(inputs, 'boolean.XOR')[0]).toBe(expected);
  });
});

describe('Boolean Logic — NAND truth table', () => {
  test.each([
    [[0, 0], 1],
    [[0, 1], 1],
    [[1, 0], 1],
    [[1, 1], 0],
  ])('NAND(%j) = %i', (inputs, expected) => {
    expect(brain.predictBinary(inputs, 'boolean.NAND')[0]).toBe(expected);
  });
});

describe('Boolean Logic — NOR truth table', () => {
  test.each([
    [[0, 0], 1],
    [[0, 1], 0],
    [[1, 0], 0],
    [[1, 1], 0],
  ])('NOR(%j) = %i', (inputs, expected) => {
    expect(brain.predictBinary(inputs, 'boolean.NOR')[0]).toBe(expected);
  });
});

describe('Boolean Logic — XNOR truth table', () => {
  test.each([
    [[0, 0], 1],
    [[0, 1], 0],
    [[1, 0], 0],
    [[1, 1], 1],
  ])('XNOR(%j) = %i', (inputs, expected) => {
    expect(brain.predictBinary(inputs, 'boolean.XNOR')[0]).toBe(expected);
  });
});

// ── Nested expression evaluation ──────────────────────────────────────────────

describe('Boolean Logic — nested expression evaluation', () => {
  test('AND(OR(1,0), NOT(0)) = 1', () => {
    expect(brain.evaluate({
      op: 'AND', inputs: [
        { op: 'OR',  inputs: [{ value: 1 }, { value: 0 }] },
        { op: 'NOT', inputs: [{ value: 0 }]               },
      ],
    })).toBe(1);
  });

  test('OR(AND(1,0), XOR(1,0)) = 1', () => {
    expect(brain.evaluate({
      op: 'OR', inputs: [
        { op: 'AND', inputs: [{ value: 1 }, { value: 0 }] },
        { op: 'XOR', inputs: [{ value: 1 }, { value: 0 }] },
      ],
    })).toBe(1);
  });

  test('NOT(AND(1,1)) = 0', () => {
    expect(brain.evaluate({
      op: 'NOT', inputs: [
        { op: 'AND', inputs: [{ value: 1 }, { value: 1 }] },
      ],
    })).toBe(0);
  });

  test('XOR(AND(1,1), OR(0,0)) = 1', () => {
    expect(brain.evaluate({
      op: 'XOR', inputs: [
        { op: 'AND', inputs: [{ value: 1 }, { value: 1 }] },
        { op: 'OR',  inputs: [{ value: 0 }, { value: 0 }] },
      ],
    })).toBe(1);
  });

  test('three-level: OR(AND(NOT(0),1), XOR(1,1)) = 1', () => {
    expect(brain.evaluate({
      op: 'OR', inputs: [
        {
          op: 'AND', inputs: [
            { op: 'NOT', inputs: [{ value: 0 }] },
            { value: 1 },
          ],
        },
        { op: 'XOR', inputs: [{ value: 1 }, { value: 1 }] },
      ],
    })).toBe(1);
  });

  test('AND(OR(0,0), NOT(1)) = 0', () => {
    expect(brain.evaluate({
      op: 'AND', inputs: [
        { op: 'OR',  inputs: [{ value: 0 }, { value: 0 }] },
        { op: 'NOT', inputs: [{ value: 1 }]               },
      ],
    })).toBe(0);
  });
});
