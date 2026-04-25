'use strict';

/**
 * Digital Brain Framework
 *
 * Public API barrel — re-exports the core building blocks so that consumers
 * can import everything from the package root.
 *
 * Usage:
 *   const { Brain, Lesson, Syllabus } = require('sturdy-waffle');
 *   const { WorkingMemory, tokens }   = require('sturdy-waffle');
 *   const { FactBase }                = require('sturdy-waffle');
 */

const Brain        = require('./src/brain/Brain');
const BrainRegion  = require('./src/brain/BrainRegion');
const NeuralNetwork = require('./src/brain/NeuralNetwork');
const Router       = require('./src/routing/Router');
const StateManager = require('./src/persistence/StateManager');
const Lesson       = require('./src/learning/Lesson');
const Syllabus     = require('./src/learning/Syllabus');
const FactBase     = require('./src/knowledge/FactBase');

// Decomposition subsystem
const {
  WorkingMemory,
  DecompositionGraph,
  DecompositionController,
  ReplayBuffer,
  tokens,
  computeExpertTrace,
} = require('./src/decomposition');

module.exports = {
  Brain,
  BrainRegion,
  NeuralNetwork,
  Router,
  StateManager,
  Lesson,
  Syllabus,
  FactBase,
  // Decomposition
  WorkingMemory,
  DecompositionGraph,
  DecompositionController,
  ReplayBuffer,
  tokens,
  computeExpertTrace,
};
