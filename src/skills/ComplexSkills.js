'use strict';

const COMPLEX_SKILLS = [
  {
    name: 'multiStepMath',
    description: 'Solve multi-step arithmetic expressions from token sequences.',
    inputs: ['prefix token sequence', 'numeric literals'],
    outputs: ['scalar numeric result'],
    metrics: ['compositional accuracy', 'regression tolerance'],
  },
  {
    name: 'relationalReasoning',
    description: 'Answer queries over multi-arity relations and attributes.',
    inputs: ['relation name', 'entity arguments'],
    outputs: ['boolean relation value'],
    metrics: ['relation accuracy', 'rule precision'],
  },
  {
    name: 'causalPrediction',
    description: 'Predict next states given state/action/context histories.',
    inputs: ['state vector', 'action vector', 'context vector'],
    outputs: ['next-state prediction'],
    metrics: ['prediction error', 'trajectory rollout accuracy'],
  },
  {
    name: 'fuzzyComposition',
    description: 'Compose fuzzy operators over continuous inputs.',
    inputs: ['continuous values in [0,1]'],
    outputs: ['continuous fuzzy output'],
    metrics: ['regression tolerance', 'compositional accuracy'],
  },
];

module.exports = {
  COMPLEX_SKILLS,
};
