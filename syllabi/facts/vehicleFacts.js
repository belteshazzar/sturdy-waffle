'use strict';

const FactBase = require('../../src/knowledge/FactBase');
const Syllabus = require('../../src/learning/Syllabus');

// ── Vehicle categories ─────────────────────────────────────────────────────────
//
// Nine vehicles described by four transportation predicates. These facts let the
// Brain learn which modes of travel each vehicle belongs to.
//
// Predicates
// ──────────
//   isLand          — primarily travels on land
//   isWater         — primarily travels on water
//   isAir           — primarily travels through the air
//   isHumanPowered  — propelled directly by human effort
//
// Subjects (in vocabulary order — encoding is stable)
//   0  car        → isLand=T  isWater=F  isAir=F  isHumanPowered=F
//   1  bicycle    → isLand=T  isWater=F  isAir=F  isHumanPowered=T
//   2  motorcycle → isLand=T  isWater=F  isAir=F  isHumanPowered=F
//   3  bus        → isLand=T  isWater=F  isAir=F  isHumanPowered=F
//   4  train      → isLand=T  isWater=F  isAir=F  isHumanPowered=F
//   5  airplane   → isLand=F  isWater=F  isAir=T  isHumanPowered=F
//   6  helicopter → isLand=F  isWater=F  isAir=T  isHumanPowered=F
//   7  sailboat   → isLand=F  isWater=T  isAir=F  isHumanPowered=T
//   8  submarine  → isLand=F  isWater=T  isAir=F  isHumanPowered=F

const vehicleFacts = new FactBase('Vehicle Categories');

// isLand
vehicleFacts
  .assert('car',        'isLand', true)
  .assert('bicycle',    'isLand', true)
  .assert('motorcycle', 'isLand', true)
  .assert('bus',        'isLand', true)
  .assert('train',      'isLand', true)
  .assert('airplane',   'isLand', false)
  .assert('helicopter', 'isLand', false)
  .assert('sailboat',   'isLand', false)
  .assert('submarine',  'isLand', false);

// isWater
vehicleFacts
  .assert('car',        'isWater', false)
  .assert('bicycle',    'isWater', false)
  .assert('motorcycle', 'isWater', false)
  .assert('bus',        'isWater', false)
  .assert('train',      'isWater', false)
  .assert('airplane',   'isWater', false)
  .assert('helicopter', 'isWater', false)
  .assert('sailboat',   'isWater', true)
  .assert('submarine',  'isWater', true);

// isAir
vehicleFacts
  .assert('car',        'isAir', false)
  .assert('bicycle',    'isAir', false)
  .assert('motorcycle', 'isAir', false)
  .assert('bus',        'isAir', false)
  .assert('train',      'isAir', false)
  .assert('airplane',   'isAir', true)
  .assert('helicopter', 'isAir', true)
  .assert('sailboat',   'isAir', false)
  .assert('submarine',  'isAir', false);

// isHumanPowered
vehicleFacts
  .assert('car',        'isHumanPowered', false)
  .assert('bicycle',    'isHumanPowered', true)
  .assert('motorcycle', 'isHumanPowered', false)
  .assert('bus',        'isHumanPowered', false)
  .assert('train',      'isHumanPowered', false)
  .assert('airplane',   'isHumanPowered', false)
  .assert('helicopter', 'isHumanPowered', false)
  .assert('sailboat',   'isHumanPowered', true)
  .assert('submarine',  'isHumanPowered', false);

// ── Derive lessons and syllabus ───────────────────────────────────────────────

/**
 * One classification Lesson per predicate (4 total).
 * Domain keys: facts.isLand, facts.isWater, facts.isAir, facts.isHumanPowered.
 */
const lessons = vehicleFacts.toLessons();

/**
 * A transportation curriculum teaching the Brain which vehicles move through
 * land, water, or air, and which rely on direct human power.
 *
 * After learning this syllabus the Brain can answer questions like:
 *   "Is a bicycle human powered?" → brain.queryFact('bicycle', 'isHumanPowered') // → 1
 *   "Is a submarine airborne?"    → brain.queryFact('submarine', 'isAir')       // → 0
 *   "Is a sailboat water-based?"  → brain.queryFact('sailboat',  'isWater')     // → 1
 */
const vehicleSyllabus = new Syllabus({
  name:        'Vehicle Categories',
  description: 'Transportation facts for nine vehicles — isLand, isWater, isAir, isHumanPowered.',
  lessons,
  tags: ['facts', 'knowledge', 'vehicles', 'transport', 'declarative'],
});

module.exports = {
  vehicleFacts,
  vehicleSyllabus,
  lessons,
};
