'use strict';

const Lesson = require('../../src/learning/Lesson');
const Syllabus = require('../../src/learning/Syllabus');

const VALUES = [0, 0.25, 0.5, 0.75, 1];

const chainLesson = new Lesson({
  name: 'Fuzzy Logic Chain',
  domain: 'fuzzy.CHAIN',
  description: 'Compute AND(OR(a,b), NOT(c)) under fuzzy semantics.',
  trainingData: VALUES.flatMap(a => VALUES.flatMap(b => VALUES.map(c => ({
    input: [a, b, c],
    output: [Math.min(Math.max(a, b), 1 - c)],
  })))),
  mode: 'regression',
  networkConfig: { hiddenActivation: 'tanh', outputActivation: 'linear' },
  tags: ['fuzzy', 'compositional'],
});

const fuzzyChainSyllabus = new Syllabus({
  name: 'Fuzzy Logic Chains',
  description: 'Learn multi-step fuzzy logic compositions.',
  lessons: [chainLesson],
  tags: ['fuzzy', 'compositional'],
});

module.exports = {
  chainLesson,
  fuzzyChainSyllabus,
};
