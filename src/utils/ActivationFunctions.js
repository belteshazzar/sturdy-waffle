'use strict';

const sigmoid = x => 1 / (1 + Math.exp(-x));
const sigmoidPrime = x => { const s = sigmoid(x); return s * (1 - s); };

const relu = x => Math.max(0, x);
const reluPrime = x => (x > 0 ? 1 : 0);

const tanh = x => Math.tanh(x);
const tanhPrime = x => 1 - Math.tanh(x) ** 2;

const linear = x => x;
const linearPrime = () => 1;

const ACTIVATIONS = {
  sigmoid: { fn: sigmoid, derivative: sigmoidPrime },
  relu:    { fn: relu,    derivative: reluPrime    },
  tanh:    { fn: tanh,    derivative: tanhPrime    },
  linear:  { fn: linear,  derivative: linearPrime  },
};

function activate(x, name = 'sigmoid') {
  const entry = ACTIVATIONS[name];
  if (!entry) throw new Error(`Unknown activation function: ${name}`);
  return entry.fn(x);
}

function activatePrime(x, name = 'sigmoid') {
  const entry = ACTIVATIONS[name];
  if (!entry) throw new Error(`Unknown activation function: ${name}`);
  return entry.derivative(x);
}

module.exports = {
  sigmoid, sigmoidPrime,
  relu,    reluPrime,
  tanh,    tanhPrime,
  linear,  linearPrime,
  activate,
  activatePrime,
  ACTIVATIONS,
};
