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
const RecurrentNetwork = require('./src/brain/RecurrentNetwork');
const SequenceBrainRegion = require('./src/brain/SequenceBrainRegion');
const SharedEmbeddingBank = require('./src/brain/SharedEmbeddingBank');
const MetaLearner = require('./src/brain/MetaLearner');
const Router       = require('./src/routing/Router');
const StateManager = require('./src/persistence/StateManager');
const Lesson       = require('./src/learning/Lesson');
const Syllabus     = require('./src/learning/Syllabus');
const SelfSupervisedLearner = require('./src/learning/SelfSupervisedLearner');
const FactBase     = require('./src/knowledge/FactBase');
const MemorySystem = require('./src/memory/MemorySystem');
const WorldModel   = require('./src/world/WorldModel');
const EvaluationSuite = require('./src/evaluation/EvaluationSuite');
const ExpressionParser = require('./src/parsing/ExpressionParser');
const Ingestion = require('./src/ingestion');
const { COMPLEX_SKILLS } = require('./src/skills/ComplexSkills');

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
  RecurrentNetwork,
  SequenceBrainRegion,
  SharedEmbeddingBank,
  MetaLearner,
  Router,
  StateManager,
  Lesson,
  Syllabus,
  SelfSupervisedLearner,
  FactBase,
  MemorySystem,
  WorldModel,
  EvaluationSuite,
  ExpressionParser,
  Ingestion,
  COMPLEX_SKILLS,
  // Decomposition
  WorkingMemory,
  DecompositionGraph,
  DecompositionController,
  ReplayBuffer,
  tokens,
  computeExpertTrace,
};
