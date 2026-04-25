'use strict';

const FactBase = require('../../src/knowledge/FactBase');
const Syllabus = require('../../src/learning/Syllabus');

// ── Fruit and food colors ─────────────────────────────────────────────────────
//
// Eight common fruits/foods with five color predicates.  Each predicate asks
// a binary question ("is this item primarily red?") which the Brain can learn
// and later answer via brain.queryFact(subject, predicate).
//
// Predicates
// ──────────
//   isRed    — the item is primarily red in color
//   isGreen  — the item is primarily green in color
//   isYellow — the item is primarily yellow in color
//   isOrange — the item is primarily orange in color
//   isPurple — the item is primarily purple/blue in color
//
// Subjects (in vocabulary order — encoding is stable)
//   0  apple      → isRed=T  isGreen=F  isYellow=F  isOrange=F  isPurple=F
//   1  orange     → isRed=F  isGreen=F  isYellow=F  isOrange=T  isPurple=F
//   2  banana     → isRed=F  isGreen=F  isYellow=T  isOrange=F  isPurple=F
//   3  grape      → isRed=F  isGreen=F  isYellow=F  isOrange=F  isPurple=T
//   4  strawberry → isRed=T  isGreen=F  isYellow=F  isOrange=F  isPurple=F
//   5  lemon      → isRed=F  isGreen=F  isYellow=T  isOrange=F  isPurple=F
//   6  lime       → isRed=F  isGreen=T  isYellow=F  isOrange=F  isPurple=F
//   7  plum       → isRed=F  isGreen=F  isYellow=F  isOrange=F  isPurple=T
//
// Example queries after brain.learnFacts(colorFacts):
//   brain.queryFact('apple',  'isRed')    // → 1  (yes, apples are red)
//   brain.queryFact('orange', 'isRed')    // → 0  (no, oranges are not red)
//   brain.queryFact('banana', 'isYellow') // → 1
//   brain.queryFact('lime',   'isGreen')  // → 1

const colorFacts = new FactBase('Fruit and Food Colors');

// isRed
colorFacts
  .assert('apple',      'isRed', true)
  .assert('orange',     'isRed', false)
  .assert('banana',     'isRed', false)
  .assert('grape',      'isRed', false)
  .assert('strawberry', 'isRed', true)
  .assert('lemon',      'isRed', false)
  .assert('lime',       'isRed', false)
  .assert('plum',       'isRed', false);

// isGreen
colorFacts
  .assert('apple',      'isGreen', false)
  .assert('orange',     'isGreen', false)
  .assert('banana',     'isGreen', false)
  .assert('grape',      'isGreen', false)
  .assert('strawberry', 'isGreen', false)
  .assert('lemon',      'isGreen', false)
  .assert('lime',       'isGreen', true)
  .assert('plum',       'isGreen', false);

// isYellow
colorFacts
  .assert('apple',      'isYellow', false)
  .assert('orange',     'isYellow', false)
  .assert('banana',     'isYellow', true)
  .assert('grape',      'isYellow', false)
  .assert('strawberry', 'isYellow', false)
  .assert('lemon',      'isYellow', true)
  .assert('lime',       'isYellow', false)
  .assert('plum',       'isYellow', false);

// isOrange
colorFacts
  .assert('apple',      'isOrange', false)
  .assert('orange',     'isOrange', true)
  .assert('banana',     'isOrange', false)
  .assert('grape',      'isOrange', false)
  .assert('strawberry', 'isOrange', false)
  .assert('lemon',      'isOrange', false)
  .assert('lime',       'isOrange', false)
  .assert('plum',       'isOrange', false);

// isPurple
colorFacts
  .assert('apple',      'isPurple', false)
  .assert('orange',     'isPurple', false)
  .assert('banana',     'isPurple', false)
  .assert('grape',      'isPurple', true)
  .assert('strawberry', 'isPurple', false)
  .assert('lemon',      'isPurple', false)
  .assert('lime',       'isPurple', false)
  .assert('plum',       'isPurple', true);

// ── Derive lessons and syllabus ───────────────────────────────────────────────

/**
 * One classification Lesson per color predicate (5 total).
 * Domain keys: facts.isRed, facts.isGreen, facts.isYellow, facts.isOrange,
 *              facts.isPurple
 */
const lessons = colorFacts.toLessons();

/**
 * A color-knowledge curriculum teaching the Brain to identify the primary
 * color of eight common fruits and foods.
 *
 * After learning this syllabus the Brain can answer questions like:
 *   "Is an orange red?"   → brain.queryFact('orange', 'isRed')    // → 0
 *   "Is an apple red?"    → brain.queryFact('apple',  'isRed')    // → 1
 *   "Is a banana yellow?" → brain.queryFact('banana', 'isYellow') // → 1
 *
 * For open-ended "what color is X?" queries, use a FactBase with
 * assertValue() and brain.queryAttribute() instead.
 */
const colorSyllabus = new Syllabus({
  name:        'Fruit and Food Colors',
  description: 'Color facts for eight fruits/foods — isRed, isGreen, isYellow, isOrange, isPurple.',
  lessons,
  tags: ['facts', 'knowledge', 'colors', 'food', 'declarative'],
});

module.exports = {
  colorFacts,
  colorSyllabus,
  lessons,
};
