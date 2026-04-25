'use strict';

const FactBase = require('../../src/knowledge/FactBase');
const Syllabus = require('../../src/learning/Syllabus');

// ── Geometric shape properties ─────────────────────────────────────────────────
//
// Eight common 2-D and 3-D shapes described by four structural predicates.
// The Brain can learn these facts and later answer binary questions such as
// "does a sphere have corners?" or "is a cube three-dimensional?".
//
// Predicates
// ──────────
//   hasCorners — the shape has sharp corners / vertices
//   isRound    — the shape has a circular or curved profile
//   is3D       — the shape is three-dimensional (a solid)
//   hasFlatFace — the shape has at least one flat face/side
//
// Subjects (in vocabulary order — encoding is stable)
//   0  circle    → hasCorners=F  isRound=T  is3D=F  hasFlatFace=F
//   1  square    → hasCorners=T  isRound=F  is3D=F  hasFlatFace=T
//   2  triangle  → hasCorners=T  isRound=F  is3D=F  hasFlatFace=T
//   3  rectangle → hasCorners=T  isRound=F  is3D=F  hasFlatFace=T
//   4  pentagon  → hasCorners=T  isRound=F  is3D=F  hasFlatFace=T
//   5  sphere    → hasCorners=F  isRound=T  is3D=T  hasFlatFace=F
//   6  cube      → hasCorners=T  isRound=F  is3D=T  hasFlatFace=T
//   7  cylinder  → hasCorners=F  isRound=T  is3D=T  hasFlatFace=T

const shapeFacts = new FactBase('Geometric Shapes');

// hasCorners
shapeFacts
  .assert('circle',    'hasCorners', false)
  .assert('square',    'hasCorners', true)
  .assert('triangle',  'hasCorners', true)
  .assert('rectangle', 'hasCorners', true)
  .assert('pentagon',  'hasCorners', true)
  .assert('sphere',    'hasCorners', false)
  .assert('cube',      'hasCorners', true)
  .assert('cylinder',  'hasCorners', false);

// isRound
shapeFacts
  .assert('circle',    'isRound', true)
  .assert('square',    'isRound', false)
  .assert('triangle',  'isRound', false)
  .assert('rectangle', 'isRound', false)
  .assert('pentagon',  'isRound', false)
  .assert('sphere',    'isRound', true)
  .assert('cube',      'isRound', false)
  .assert('cylinder',  'isRound', true);

// is3D
shapeFacts
  .assert('circle',    'is3D', false)
  .assert('square',    'is3D', false)
  .assert('triangle',  'is3D', false)
  .assert('rectangle', 'is3D', false)
  .assert('pentagon',  'is3D', false)
  .assert('sphere',    'is3D', true)
  .assert('cube',      'is3D', true)
  .assert('cylinder',  'is3D', true);

// hasFlatFace
shapeFacts
  .assert('circle',    'hasFlatFace', false)
  .assert('square',    'hasFlatFace', true)
  .assert('triangle',  'hasFlatFace', true)
  .assert('rectangle', 'hasFlatFace', true)
  .assert('pentagon',  'hasFlatFace', true)
  .assert('sphere',    'hasFlatFace', false)
  .assert('cube',      'hasFlatFace', true)
  .assert('cylinder',  'hasFlatFace', true);

// ── Derive lessons and syllabus ───────────────────────────────────────────────

/**
 * One classification Lesson per shape predicate (4 total).
 * Domain keys: facts.hasCorners, facts.isRound, facts.is3D, facts.hasFlatFace
 */
const lessons = shapeFacts.toLessons();

/**
 * A geometry curriculum teaching the Brain structural properties of common
 * 2-D and 3-D shapes.
 *
 * After learning this syllabus the Brain can answer questions like:
 *   "Does a circle have corners?" → brain.queryFact('circle',   'hasCorners') // → 0
 *   "Is a sphere round?"          → brain.queryFact('sphere',   'isRound')    // → 1
 *   "Is a cube 3D?"               → brain.queryFact('cube',     'is3D')       // → 1
 *   "Does a cylinder have a flat face?"
 *                                 → brain.queryFact('cylinder', 'hasFlatFace')// → 1
 */
const shapeSyllabus = new Syllabus({
  name:        'Geometric Shape Properties',
  description: 'Structural facts about eight shapes — hasCorners, isRound, is3D, hasFlatFace.',
  lessons,
  tags: ['facts', 'knowledge', 'geometry', 'shapes', 'declarative'],
});

module.exports = {
  shapeFacts,
  shapeSyllabus,
  lessons,
};
