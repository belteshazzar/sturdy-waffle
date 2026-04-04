'use strict';

const NeuralNetwork = require('../src/brain/NeuralNetwork');

// ── Architecture ──────────────────────────────────────────────────────────────

describe('NeuralNetwork — architecture', () => {
  test('stores the given architecture', () => {
    const nn = new NeuralNetwork({ architecture: [2, 4, 1] });
    expect(nn.architecture).toEqual([2, 4, 1]);
  });

  test('creates correct number of weight layers', () => {
    const nn = new NeuralNetwork({ architecture: [2, 4, 1] });
    expect(nn.layers).toHaveLength(2);
  });

  test('hidden-layer weight matrix has correct dimensions', () => {
    const nn = new NeuralNetwork({ architecture: [2, 4, 1] });
    expect(nn.layers[0].weights).toHaveLength(4);       // 4 output neurons
    expect(nn.layers[0].weights[0]).toHaveLength(2);    // 2 input neurons
  });

  test('output-layer weight matrix has correct dimensions', () => {
    const nn = new NeuralNetwork({ architecture: [2, 4, 1] });
    expect(nn.layers[1].weights).toHaveLength(1);       // 1 output neuron
    expect(nn.layers[1].weights[0]).toHaveLength(4);    // 4 input neurons
  });
});

// ── Forward pass ──────────────────────────────────────────────────────────────

describe('NeuralNetwork — forward pass', () => {
  test('predict returns vector of correct length', () => {
    const nn = new NeuralNetwork({ architecture: [2, 4, 1] });
    const out = nn.predict([0.5, 0.5]);
    expect(out).toHaveLength(1);
  });

  test('sigmoid output is in (0, 1)', () => {
    const nn = new NeuralNetwork({ architecture: [2, 4, 1] });
    const out = nn.predict([0.0, 1.0]);
    expect(out[0]).toBeGreaterThan(0);
    expect(out[0]).toBeLessThan(1);
  });

  test('predictBinary returns only 0 or 1', () => {
    const nn  = new NeuralNetwork({ architecture: [2, 4, 1] });
    const out = nn.predictBinary([0.5, 0.5]);
    expect([0, 1]).toContain(out[0]);
  });
});

// ── Learning ──────────────────────────────────────────────────────────────────

describe('NeuralNetwork — learning', () => {
  test('learns NOT gate (trivially separable)', () => {
    const nn   = new NeuralNetwork({ architecture: [1, 4, 1], learningRate: 0.3 });
    const data = [
      { input: [0], output: [1] },
      { input: [1], output: [0] },
    ];
    nn.train(data, 2000);
    expect(nn.accuracy(data)).toBe(1.0);
  });

  test('learns AND gate', () => {
    const nn   = new NeuralNetwork({ architecture: [2, 8, 1], learningRate: 0.3 });
    const data = [
      { input: [0, 0], output: [0] },
      { input: [0, 1], output: [0] },
      { input: [1, 0], output: [0] },
      { input: [1, 1], output: [1] },
    ];
    nn.train(data, 3000);
    expect(nn.accuracy(data)).toBeGreaterThanOrEqual(0.75);
  });

  test('train returns finalLoss and losses array', () => {
    const nn     = new NeuralNetwork({ architecture: [2, 4, 1] });
    const result = nn.train([{ input: [0, 0], output: [0] }], 10);
    expect(typeof result.finalLoss).toBe('number');
    expect(result.losses).toHaveLength(10);
  });

  test('loss decreases over training', () => {
    const nn   = new NeuralNetwork({ architecture: [2, 4, 1], learningRate: 0.3 });
    const data = [{ input: [0, 0], output: [0] }, { input: [1, 1], output: [1] }];
    const { losses } = nn.train(data, 500);
    // Average of last 50 epochs should be lower than first 50
    const first = losses.slice(0, 50).reduce((a, b) => a + b, 0) / 50;
    const last  = losses.slice(-50).reduce((a, b) => a + b, 0) / 50;
    expect(last).toBeLessThan(first);
  });
});

// ── Mutation ──────────────────────────────────────────────────────────────────

describe('NeuralNetwork — mutation', () => {
  test('addNeuronsToLayer increases hidden-layer size', () => {
    const nn = new NeuralNetwork({ architecture: [2, 4, 1] });
    const ok = nn.addNeuronsToLayer(0, 2);
    expect(ok).toBe(true);
    expect(nn.architecture[1]).toBe(6);
    expect(nn.layers[0].weights).toHaveLength(6);
    expect(nn.layers[1].weights[0]).toHaveLength(6);
  });

  test('addNeuronsToLayer does not modify output layer', () => {
    const nn = new NeuralNetwork({ architecture: [2, 4, 1] });
    const ok = nn.addNeuronsToLayer(1, 2);  // layer index 1 = output layer
    expect(ok).toBe(false);
    expect(nn.architecture[2]).toBe(1);
  });

  test('addHiddenLayer increases depth', () => {
    const nn = new NeuralNetwork({ architecture: [2, 4, 1] });
    nn.addHiddenLayer(3);
    expect(nn.architecture).toHaveLength(4);   // [2, 4, 3, 1]
    expect(nn.layers).toHaveLength(3);
  });

  test('network still produces valid output after addNeuronsToLayer', () => {
    const nn = new NeuralNetwork({ architecture: [2, 4, 1] });
    nn.addNeuronsToLayer(0, 3);
    const out = nn.predict([0.5, 0.5]);
    expect(out).toHaveLength(1);
    expect(out[0]).toBeGreaterThan(0);
    expect(out[0]).toBeLessThan(1);
  });

  test('network still produces valid output after addHiddenLayer', () => {
    const nn = new NeuralNetwork({ architecture: [2, 4, 1] });
    nn.addHiddenLayer(5);
    const out = nn.predict([1, 0]);
    expect(out).toHaveLength(1);
    expect(out[0]).toBeGreaterThan(0);
    expect(out[0]).toBeLessThan(1);
  });
});

// ── Serialisation ─────────────────────────────────────────────────────────────

describe('NeuralNetwork — serialisation', () => {
  test('toJSON / fromJSON round-trip preserves output', () => {
    const nn = new NeuralNetwork({ architecture: [2, 4, 1], learningRate: 0.2 });
    nn.train([
      { input: [0, 0], output: [0] },
      { input: [1, 1], output: [1] },
    ], 100);

    const json = nn.toJSON();
    const nn2  = NeuralNetwork.fromJSON(json);

    const input = [0.7, 0.3];
    expect(nn2.predict(input)[0]).toBeCloseTo(nn.predict(input)[0], 5);
  });

  test('serialised architecture matches', () => {
    const nn   = new NeuralNetwork({ architecture: [3, 5, 2] });
    const json = nn.toJSON();
    expect(json.architecture).toEqual([3, 5, 2]);
  });
});
