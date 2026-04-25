'use strict';

const FactBase = require('../../src/knowledge/FactBase');
const Syllabus = require('../../src/learning/Syllabus');

// ── Animal kingdom facts ───────────────────────────────────────────────────────
//
// A small set of ground-truth facts about six animals across four predicates.
// These facts form the basis of a declarative knowledge base that the Brain can
// learn and then use to reason with — for example by combining fact lookups
// with boolean/fuzzy logic inside Brain.evaluate() expression trees.
//
// Predicates
// ──────────
//   canFly       — the animal is capable of sustained flight
//   hasFur       — the animal has a fur/hair coat
//   canSwim      — the animal can move through water
//   isWarmBlooded — the animal regulates its own body temperature
//
// Subjects (in vocabulary order — encoding is stable)
//   0  bird   → canFly=T  hasFur=F  canSwim=F  isWarmBlooded=T
//   1  bat    → canFly=T  hasFur=T  canSwim=F  isWarmBlooded=T
//   2  fish   → canFly=F  hasFur=F  canSwim=T  isWarmBlooded=F
//   3  frog   → canFly=F  hasFur=F  canSwim=T  isWarmBlooded=F
//   4  cat    → canFly=F  hasFur=T  canSwim=F  isWarmBlooded=T
//   5  penguin → canFly=F hasFur=F  canSwim=T  isWarmBlooded=T

const animalFacts = new FactBase('Animal Kingdom');

// canFly
animalFacts
  .assert('bird',    'canFly', true)
  .assert('bat',     'canFly', true)
  .assert('fish',    'canFly', false)
  .assert('frog',    'canFly', false)
  .assert('cat',     'canFly', false)
  .assert('penguin', 'canFly', false);

// hasFur
animalFacts
  .assert('bird',    'hasFur', false)
  .assert('bat',     'hasFur', true)
  .assert('fish',    'hasFur', false)
  .assert('frog',    'hasFur', false)
  .assert('cat',     'hasFur', true)
  .assert('penguin', 'hasFur', false);

// canSwim
animalFacts
  .assert('bird',    'canSwim', false)
  .assert('bat',     'canSwim', false)
  .assert('fish',    'canSwim', true)
  .assert('frog',    'canSwim', true)
  .assert('cat',     'canSwim', false)
  .assert('penguin', 'canSwim', true);

// isWarmBlooded
animalFacts
  .assert('bird',    'isWarmBlooded', true)
  .assert('bat',     'isWarmBlooded', true)
  .assert('fish',    'isWarmBlooded', false)
  .assert('frog',    'isWarmBlooded', false)
  .assert('cat',     'isWarmBlooded', true)
  .assert('penguin', 'isWarmBlooded', true);

// ── Derive lessons and syllabus ───────────────────────────────────────────────

/**
 * One classification Lesson per predicate (4 total).
 * Domain keys: facts.canFly, facts.hasFur, facts.canSwim, facts.isWarmBlooded.
 */
const lessons = animalFacts.toLessons();

/**
 * A simple declarative curriculum teaching the Brain to remember four animal
 * properties across six subjects.
 *
 * After learning this syllabus the Brain can:
 *   • Answer direct fact queries:
 *       brain.queryFact('bird', 'canFly')   // → 1
 *       brain.queryFact('fish', 'hasFur')   // → 0
 *
 *   • Perform simple reasoning by composing fact lookups with boolean operators
 *     inside Brain.evaluate() expression trees:
 *
 *       // "Does bat fly AND have fur?"
 *       brain.evaluate({
 *         op: 'AND',
 *         inputs: [
 *           { fact: { subject: 'bat', predicate: 'canFly'  } },
 *           { fact: { subject: 'bat', predicate: 'hasFur'  } },
 *         ],
 *       });  // → 1
 *
 *       // "Can penguin swim but NOT fly?"
 *       brain.evaluate({
 *         op: 'AND',
 *         inputs: [
 *           { fact: { subject: 'penguin', predicate: 'canSwim' } },
 *           { op:  'NOT',
 *             inputs: [{ fact: { subject: 'penguin', predicate: 'canFly' } }] },
 *         ],
 *       });  // → 1
 */
const animalSyllabus = new Syllabus({
  name:        'Animal Kingdom Facts',
  description: 'Declarative facts about six animals — canFly, hasFur, canSwim, isWarmBlooded.',
  lessons,
  tags: ['facts', 'knowledge', 'animals', 'declarative'],
});

module.exports = {
  animalFacts,
  animalSyllabus,
  lessons,
};
