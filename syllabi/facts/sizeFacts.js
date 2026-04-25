'use strict';

const FactBase = require('../../src/knowledge/FactBase');
const Syllabus = require('../../src/learning/Syllabus');

// ── Relative size of animals ───────────────────────────────────────────────────
//
// Nine animals described by three relative-size predicates.  Size here is
// a rough bodily magnitude compared to a typical adult human.
//
// Predicates
// ──────────
//   isSmall  — noticeably smaller than a human hand (e.g. insects, small rodents)
//   isMedium — roughly cat-to-wolf sized (hand-sized up to human scale)
//   isLarge  — clearly larger than a human (e.g. horse, elephant, whale)
//
// Subjects (in vocabulary order — encoding is stable)
//   0  ant      → isSmall=T  isMedium=F  isLarge=F
//   1  bee      → isSmall=T  isMedium=F  isLarge=F
//   2  mouse    → isSmall=T  isMedium=F  isLarge=F
//   3  cat      → isSmall=F  isMedium=T  isLarge=F
//   4  dog      → isSmall=F  isMedium=T  isLarge=F
//   5  wolf     → isSmall=F  isMedium=T  isLarge=F
//   6  horse    → isSmall=F  isMedium=F  isLarge=T
//   7  elephant → isSmall=F  isMedium=F  isLarge=T
//   8  whale    → isSmall=F  isMedium=F  isLarge=T

const sizeFacts = new FactBase('Animal Sizes');

// isSmall
sizeFacts
  .assert('ant',      'isSmall', true)
  .assert('bee',      'isSmall', true)
  .assert('mouse',    'isSmall', true)
  .assert('cat',      'isSmall', false)
  .assert('dog',      'isSmall', false)
  .assert('wolf',     'isSmall', false)
  .assert('horse',    'isSmall', false)
  .assert('elephant', 'isSmall', false)
  .assert('whale',    'isSmall', false);

// isMedium
sizeFacts
  .assert('ant',      'isMedium', false)
  .assert('bee',      'isMedium', false)
  .assert('mouse',    'isMedium', false)
  .assert('cat',      'isMedium', true)
  .assert('dog',      'isMedium', true)
  .assert('wolf',     'isMedium', true)
  .assert('horse',    'isMedium', false)
  .assert('elephant', 'isMedium', false)
  .assert('whale',    'isMedium', false);

// isLarge
sizeFacts
  .assert('ant',      'isLarge', false)
  .assert('bee',      'isLarge', false)
  .assert('mouse',    'isLarge', false)
  .assert('cat',      'isLarge', false)
  .assert('dog',      'isLarge', false)
  .assert('wolf',     'isLarge', false)
  .assert('horse',    'isLarge', true)
  .assert('elephant', 'isLarge', true)
  .assert('whale',    'isLarge', true);

// ── Derive lessons and syllabus ───────────────────────────────────────────────

/**
 * One classification Lesson per size predicate (3 total).
 * Domain keys: facts.isSmall, facts.isMedium, facts.isLarge
 */
const lessons = sizeFacts.toLessons();

/**
 * A size-knowledge curriculum teaching the Brain relative size categories
 * for nine animals.
 *
 * After learning this syllabus the Brain can answer questions like:
 *   "Is an ant small?"       → brain.queryFact('ant',      'isSmall')  // → 1
 *   "Is a whale medium?"     → brain.queryFact('whale',    'isMedium') // → 0
 *   "Is an elephant large?"  → brain.queryFact('elephant', 'isLarge')  // → 1
 *   "Is a dog large?"        → brain.queryFact('dog',      'isLarge')  // → 0
 */
const sizeSyllabus = new Syllabus({
  name:        'Animal Sizes',
  description: 'Relative size facts for nine animals — isSmall, isMedium, isLarge.',
  lessons,
  tags: ['facts', 'knowledge', 'size', 'animals', 'declarative'],
});

module.exports = {
  sizeFacts,
  sizeSyllabus,
  lessons,
};
