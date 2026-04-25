'use strict';

const FactBase = require('../../src/knowledge/FactBase');
const Syllabus = require('../../src/learning/Syllabus');

// ── Solar system facts ─────────────────────────────────────────────────────────
//
// Seven solar-system bodies described by four astronomical predicates.
//
// Predicates
// ──────────
//   isAStar   — the body is a star (fuses hydrogen)
//   isAPlanet — the body is a planet orbiting the Sun
//   hasRings  — the body has a visible ring system
//   hasMoons  — the body has at least one natural satellite
//
// Subjects (in vocabulary order — encoding is stable)
//   0  sun     → isAStar=T  isAPlanet=F  hasRings=F  hasMoons=F
//   1  mercury → isAStar=F  isAPlanet=T  hasRings=F  hasMoons=F
//   2  venus   → isAStar=F  isAPlanet=T  hasRings=F  hasMoons=F
//   3  earth   → isAStar=F  isAPlanet=T  hasRings=F  hasMoons=T
//   4  mars    → isAStar=F  isAPlanet=T  hasRings=F  hasMoons=T
//   5  jupiter → isAStar=F  isAPlanet=T  hasRings=T  hasMoons=T
//   6  saturn  → isAStar=F  isAPlanet=T  hasRings=T  hasMoons=T

const worldFacts = new FactBase('Solar System');

// isAStar
worldFacts
  .assert('sun',     'isAStar', true)
  .assert('mercury', 'isAStar', false)
  .assert('venus',   'isAStar', false)
  .assert('earth',   'isAStar', false)
  .assert('mars',    'isAStar', false)
  .assert('jupiter', 'isAStar', false)
  .assert('saturn',  'isAStar', false);

// isAPlanet
worldFacts
  .assert('sun',     'isAPlanet', false)
  .assert('mercury', 'isAPlanet', true)
  .assert('venus',   'isAPlanet', true)
  .assert('earth',   'isAPlanet', true)
  .assert('mars',    'isAPlanet', true)
  .assert('jupiter', 'isAPlanet', true)
  .assert('saturn',  'isAPlanet', true);

// hasRings
worldFacts
  .assert('sun',     'hasRings', false)
  .assert('mercury', 'hasRings', false)
  .assert('venus',   'hasRings', false)
  .assert('earth',   'hasRings', false)
  .assert('mars',    'hasRings', false)
  .assert('jupiter', 'hasRings', true)
  .assert('saturn',  'hasRings', true);

// hasMoons
worldFacts
  .assert('sun',     'hasMoons', false)
  .assert('mercury', 'hasMoons', false)
  .assert('venus',   'hasMoons', false)
  .assert('earth',   'hasMoons', true)
  .assert('mars',    'hasMoons', true)
  .assert('jupiter', 'hasMoons', true)
  .assert('saturn',  'hasMoons', true);

// ── Derive lessons and syllabus ───────────────────────────────────────────────

/**
 * One classification Lesson per predicate (4 total).
 * Domain keys: facts.isAStar, facts.isAPlanet, facts.hasRings, facts.hasMoons
 */
const lessons = worldFacts.toLessons();

/**
 * An astronomy curriculum teaching the Brain basic facts about seven
 * solar-system bodies.
 *
 * After learning this syllabus the Brain can answer questions like:
 *   "Is the Sun a star?"      → brain.queryFact('sun',     'isAStar')   // → 1
 *   "Does Mercury have rings?"→ brain.queryFact('mercury', 'hasRings')  // → 0
 *   "Does Saturn have moons?" → brain.queryFact('saturn',  'hasMoons')  // → 1
 *   "Is Earth a planet?"      → brain.queryFact('earth',   'isAPlanet') // → 1
 */
const worldSyllabus = new Syllabus({
  name:        'Solar System Facts',
  description: 'Astronomical facts about seven solar-system bodies — isAStar, isAPlanet, hasRings, hasMoons.',
  lessons,
  tags: ['facts', 'knowledge', 'astronomy', 'solar-system', 'declarative'],
});

module.exports = {
  worldFacts,
  worldSyllabus,
  lessons,
};
