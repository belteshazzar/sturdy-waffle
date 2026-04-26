'use strict';

const FactBase = require('../../src/knowledge/FactBase');
const Syllabus = require('../../src/learning/Syllabus');

const relationFacts = new FactBase('Family Relations');

relationFacts
  .assertRelation('parentOf', ['alice', 'bob'])
  .assertRelation('parentOf', ['alice', 'carol'])
  .assertRelation('parentOf', ['bob', 'dave'])
  .assertRelation('siblingOf', ['bob', 'carol'])
  .assertRelation('siblingOf', ['carol', 'bob']);

const relationSyllabus = new Syllabus({
  name: 'Relational Puzzles',
  description: 'Binary relations for small family puzzles.',
  lessons: relationFacts.toLessons(),
  tags: ['relations', 'facts'],
});

module.exports = {
  relationFacts,
  relationSyllabus,
};
