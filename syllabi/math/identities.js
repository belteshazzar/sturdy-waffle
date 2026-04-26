'use strict';

const Lesson = require('../../src/learning/Lesson');
const Syllabus = require('../../src/learning/Syllabus');

const GRID = Array.from({ length: 9 }, (_, i) => i / 8);

const squareSumLesson = new Lesson({
  name: 'Square of Sum',
  domain: 'math.ID_SQUARE_SUM',
  description: 'Compute (a + b)^2 for a,b in [0,1].',
  trainingData: GRID.flatMap(a => GRID.map(b => ({
    input: [a, b],
    output: [(a + b) ** 2],
  }))),
  mode: 'regression',
  normalise: { outputRange: [0, 4] },
  networkConfig: { hiddenActivation: 'tanh', outputActivation: 'linear' },
  tags: ['math', 'identity'],
});

const differenceSquaresLesson = new Lesson({
  name: 'Difference of Squares',
  domain: 'math.ID_DIFF_SQUARES',
  description: 'Compute a^2 - b^2 for a,b in [0,1].',
  trainingData: GRID.flatMap(a => GRID.map(b => ({
    input: [a, b],
    output: [a ** 2 - b ** 2],
  }))),
  mode: 'regression',
  normalise: { outputRange: [-1, 1] },
  networkConfig: { hiddenActivation: 'tanh', outputActivation: 'linear' },
  tags: ['math', 'identity'],
});

const identitySyllabus = new Syllabus({
  name: 'Algebraic Identities',
  description: 'Regression lessons for common algebraic identities.',
  lessons: [squareSumLesson, differenceSquaresLesson],
  tags: ['math', 'identity'],
});

module.exports = {
  identitySyllabus,
  squareSumLesson,
  differenceSquaresLesson,
};
