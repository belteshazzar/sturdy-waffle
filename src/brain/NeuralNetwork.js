'use strict';

const { activate, activatePrime } = require('../utils/ActivationFunctions');
const {
  createMatrix,
  createVector,
  xavierInit,
  matVecMul,
  vecAdd,
} = require('../utils/MathUtils');

/**
 * A feedforward neural network with backpropagation training and in-place
 * structural mutation (add neurons, add layers, reinitialise weights).
 *
 * Architecture is described as an array of layer sizes, e.g. [2, 8, 1] means
 * 2 inputs → 8 hidden neurons → 1 output.
 */
class NeuralNetwork {
  /**
   * @param {object} opts
   * @param {number[]} opts.architecture  Layer sizes [in, hidden…, out]
   * @param {number}   [opts.learningRate=0.1]
   * @param {string}   [opts.hiddenActivation='sigmoid']
   * @param {string}   [opts.outputActivation='sigmoid']
   */
  constructor({
    architecture,
    learningRate = 0.1,
    hiddenActivation = 'sigmoid',
    outputActivation = 'sigmoid',
  }) {
    this.architecture      = [...architecture];
    this.learningRate      = learningRate;
    this.hiddenActivation  = hiddenActivation;
    this.outputActivation  = outputActivation;
    this.layers            = [];
    this._initializeLayers();
  }

  // ── Initialisation ────────────────────────────────────────────────────────

  _initializeLayers() {
    this.layers = [];
    for (let l = 1; l < this.architecture.length; l++) {
      const inSize  = this.architecture[l - 1];
      const outSize = this.architecture[l];
      const isOut   = l === this.architecture.length - 1;

      this.layers.push({
        weights:    createMatrix(outSize, inSize, () => xavierInit(inSize, outSize)),
        biases:     createVector(outSize, () => 0),
        activation: isOut ? this.outputActivation : this.hiddenActivation,
        inputSize:  inSize,
        outputSize: outSize,
      });
    }
  }

  // ── Forward pass ──────────────────────────────────────────────────────────

  /**
   * Run a full forward pass.
   * @param {number[]} input
   * @returns {{ output: number[], activations: number[][], zValues: number[][] }}
   */
  forward(input) {
    let current = [...input];
    const activations = [current];
    const zValues     = [];

    for (const layer of this.layers) {
      const z = vecAdd(matVecMul(layer.weights, current), layer.biases);
      zValues.push(z);
      current = z.map(v => activate(v, layer.activation));
      activations.push(current);
    }

    return { output: current, activations, zValues };
  }

  /**
   * Return only the output vector (convenience wrapper around forward).
   * @param {number[]} input
   * @returns {number[]}
   */
  predict(input) {
    return this.forward(input).output;
  }

  /**
   * Predict and threshold output values to 0/1 booleans.
   * @param {number[]} input
   * @param {number}   [threshold=0.5]
   * @returns {number[]}
   */
  predictBinary(input, threshold = 0.5) {
    return this.predict(input).map(v => (v >= threshold ? 1 : 0));
  }

  // ── Backward pass (backpropagation) ───────────────────────────────────────

  /**
   * Perform one backpropagation step and return the MSE loss for this sample.
   * @param {number[]} input
   * @param {number[]} target
   * @returns {number} sample loss
   */
  backward(input, target) {
    const { activations, zValues } = this.forward(input);
    const numLayers = this.layers.length;
    const deltas    = new Array(numLayers);

    // Output-layer delta: δ = (a − y) · σ'(z)
    deltas[numLayers - 1] = activations[numLayers].map((a, i) =>
      (a - target[i]) * activatePrime(zValues[numLayers - 1][i], this.layers[numLayers - 1].activation)
    );

    // Hidden-layer deltas (backpropagate)
    for (let l = numLayers - 2; l >= 0; l--) {
      const nextLayer    = this.layers[l + 1];
      const currentLayer = this.layers[l];

      deltas[l] = zValues[l].map((z, j) => {
        const error = nextLayer.weights.reduce(
          (sum, row, i) => sum + row[j] * deltas[l + 1][i],
          0
        );
        return error * activatePrime(z, currentLayer.activation);
      });
    }

    // Apply gradient descent
    for (let l = 0; l < numLayers; l++) {
      const layer     = this.layers[l];
      const prevActs  = activations[l];

      for (let i = 0; i < layer.weights.length; i++) {
        for (let j = 0; j < layer.weights[i].length; j++) {
          layer.weights[i][j] -= this.learningRate * deltas[l][i] * prevActs[j];
        }
        layer.biases[i] -= this.learningRate * deltas[l][i];
      }
    }

    // Return MSE loss for this sample
    const output = activations[numLayers];
    return output.reduce((sum, a, i) => sum + 0.5 * (a - target[i]) ** 2, 0);
  }

  /**
   * Perform one backpropagation step like backward(), but additionally return
   * the gradient of the loss with respect to the **input** vector.
   *
   * This is used by DecompositionController to propagate gradients back into
   * the EmbeddingTable so that embeddings are updated jointly with the policy
   * network weights.
   *
   * @param {number[]} input
   * @param {number[]} target
   * @returns {{ loss: number, inputGrad: number[] }}
   */
  backwardWithInputGrad(input, target) {
    const { activations, zValues } = this.forward(input);
    const numLayers = this.layers.length;
    const deltas    = new Array(numLayers);

    // Output-layer delta: δ = (a − y) · σ'(z)
    deltas[numLayers - 1] = activations[numLayers].map((a, i) =>
      (a - target[i]) * activatePrime(zValues[numLayers - 1][i], this.layers[numLayers - 1].activation)
    );

    // Hidden-layer deltas
    for (let l = numLayers - 2; l >= 0; l--) {
      const nextLayer    = this.layers[l + 1];
      const currentLayer = this.layers[l];
      deltas[l] = zValues[l].map((z, j) => {
        const error = nextLayer.weights.reduce(
          (sum, row, i) => sum + row[j] * deltas[l + 1][i],
          0
        );
        return error * activatePrime(z, currentLayer.activation);
      });
    }

    // Apply gradient descent
    for (let l = 0; l < numLayers; l++) {
      const layer    = this.layers[l];
      const prevActs = activations[l];
      for (let i = 0; i < layer.weights.length; i++) {
        for (let j = 0; j < layer.weights[i].length; j++) {
          layer.weights[i][j] -= this.learningRate * deltas[l][i] * prevActs[j];
        }
        layer.biases[i] -= this.learningRate * deltas[l][i];
      }
    }

    // Compute input gradient: dL/dinput[j] = Σ_i (delta[0][i] * W[0][i][j])
    const firstLayer = this.layers[0];
    const inputGrad  = new Array(input.length).fill(0);
    for (let i = 0; i < firstLayer.weights.length; i++) {
      for (let j = 0; j < firstLayer.weights[i].length; j++) {
        inputGrad[j] += deltas[0][i] * firstLayer.weights[i][j];
      }
    }

    const output = activations[numLayers];
    const loss   = output.reduce((sum, a, i) => sum + 0.5 * (a - target[i]) ** 2, 0);
    return { loss, inputGrad };
  }

  // ── Training ──────────────────────────────────────────────────────────────

  /**
   * Train on a set of { input, output } samples for a fixed number of epochs.
   * Samples are shuffled each epoch by default.
   * @param {Array<{input: number[], output: number[]}>} samples
   * @param {number} [epochs=100]
   * @param {boolean} [shuffle=true]
   * @returns {{ finalLoss: number, losses: number[] }}
   */
  train(samples, epochs = 100, shuffle = true) {
    const losses = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
      let data = [...samples];

      if (shuffle) {
        for (let i = data.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [data[i], data[j]] = [data[j], data[i]];
        }
      }

      let epochLoss = 0;
      for (const sample of data) {
        epochLoss += this.backward(sample.input, sample.output);
      }
      losses.push(epochLoss / data.length);
    }

    return { finalLoss: losses[losses.length - 1], losses };
  }

  /**
   * Fraction of samples whose predicted binary output matches the target.
   * @param {Array<{input: number[], output: number[]}>} samples
   * @param {number} [threshold=0.5]
   * @returns {number}  0.0 – 1.0
   */
  accuracy(samples, threshold = 0.5) {
    let correct = 0;
    for (const sample of samples) {
      const predicted = this.predictBinary(sample.input, threshold);
      if (predicted.every((p, i) => p === sample.output[i])) correct++;
    }
    return correct / samples.length;
  }

  /**
   * Fraction of samples whose predicted continuous output is within `tolerance`
   * of every target value.  Designed for regression (non-binary) tasks where
   * exact equality is not meaningful.
   *
   * @param {Array<{input: number[], output: number[]}>} samples
   * @param {number} [tolerance=0.05]  Maximum allowed per-dimension absolute error
   * @returns {number}  0.0 – 1.0
   */
  regressionAccuracy(samples, tolerance = 0.05) {
    let correct = 0;
    for (const sample of samples) {
      const predicted = this.predict(sample.input);
      if (predicted.every((p, i) => Math.abs(p - sample.output[i]) <= tolerance)) correct++;
    }
    return correct / samples.length;
  }

  /**
   * Return the index of the highest-valued output neuron (argmax).
   * Used for multi-class classification where the output is a one-hot vector.
   *
   * @param {number[]} input
   * @returns {number}  Index of the predicted class
   */
  predictArgmax(input) {
    const output = this.predict(input);
    let maxIdx = 0;
    for (let i = 1; i < output.length; i++) {
      if (output[i] > output[maxIdx]) maxIdx = i;
    }
    return maxIdx;
  }

  /**
   * Fraction of samples whose predicted argmax matches the target argmax.
   * Designed for multi-class (one-hot encoded) classification tasks.
   *
   * @param {Array<{input: number[], output: number[]}>} samples
   * @returns {number}  0.0 – 1.0
   */
  multiclassAccuracy(samples) {
    let correct = 0;
    for (const sample of samples) {
      const predicted = this.predictArgmax(sample.input);
      // Target is one-hot; find the index of the 1
      let targetIdx = 0;
      for (let i = 1; i < sample.output.length; i++) {
        if (sample.output[i] > sample.output[targetIdx]) targetIdx = i;
      }
      if (predicted === targetIdx) correct++;
    }
    return correct / samples.length;
  }

  // ── Structural mutation ────────────────────────────────────────────────────

  /**
   * Grow a hidden layer by adding `count` new neurons.
   * Preserves all existing weights; new weights are Xavier-initialised.
   * The output layer is never mutated.
   *
   * @param {number} layerIndex  0-based index into this.layers (hidden only)
   * @param {number} [count=1]
   * @returns {boolean}  true when the mutation was applied
   */
  addNeuronsToLayer(layerIndex, count = 1) {
    // Refuse to mutate the output layer
    if (layerIndex < 0 || layerIndex >= this.layers.length - 1) return false;

    const layer     = this.layers[layerIndex];
    const nextLayer = this.layers[layerIndex + 1];
    const inSize    = layer.inputSize;
    const newOut    = layer.outputSize + count;

    // Extend current-layer weight matrix (add rows)
    for (let k = 0; k < count; k++) {
      layer.weights.push(
        Array.from({ length: inSize }, () => xavierInit(inSize, newOut))
      );
      layer.biases.push(0);
    }
    layer.outputSize = newOut;

    // Extend next-layer weight matrix (add columns to every row)
    for (let i = 0; i < nextLayer.weights.length; i++) {
      for (let k = 0; k < count; k++) {
        nextLayer.weights[i].push(xavierInit(newOut, nextLayer.outputSize));
      }
    }
    nextLayer.inputSize = newOut;

    this.architecture[layerIndex + 1] = newOut;
    return true;
  }

  /**
   * Insert a brand-new hidden layer before an existing layer.
   * The weights of the layer immediately following the insertion point are
   * re-initialised (the dimension changes).
   *
   * @param {number} size          Number of neurons in the new layer
   * @param {number} [insertAt=-1] Index in this.layers before which to insert;
   *                               -1 (default) inserts just before the output layer
   * @returns {boolean}
   */
  addHiddenLayer(size, insertAt = -1) {
    const targetIdx    = insertAt === -1 ? this.layers.length - 1 : insertAt;
    const prevSize     = targetIdx === 0
      ? this.architecture[0]
      : this.layers[targetIdx - 1].outputSize;
    const nextOutSize  = this.layers[targetIdx].outputSize;

    const newLayer = {
      weights:    createMatrix(size, prevSize, () => xavierInit(prevSize, size)),
      biases:     createVector(size, () => 0),
      activation: this.hiddenActivation,
      inputSize:  prevSize,
      outputSize: size,
    };

    // Re-initialise the split layer to accept `size` inputs
    this.layers[targetIdx].weights   = createMatrix(nextOutSize, size, () => xavierInit(size, nextOutSize));
    this.layers[targetIdx].inputSize = size;

    this.layers.splice(targetIdx, 0, newLayer);
    this.architecture.splice(targetIdx + 1, 0, size);
    return true;
  }

  /**
   * Re-initialise all weights while keeping the current architecture.
   * Useful as a "hard reset" mutation when the network appears stuck.
   */
  reinitialize() {
    this._initializeLayers();
  }

  // ── Serialisation ─────────────────────────────────────────────────────────

  toJSON() {
    return {
      architecture:     this.architecture,
      learningRate:     this.learningRate,
      hiddenActivation: this.hiddenActivation,
      outputActivation: this.outputActivation,
      layers: this.layers.map(l => ({
        weights:    l.weights,
        biases:     l.biases,
        activation: l.activation,
        inputSize:  l.inputSize,
        outputSize: l.outputSize,
      })),
    };
  }

  static fromJSON(data) {
    const nn = new NeuralNetwork({
      architecture:     data.architecture,
      learningRate:     data.learningRate,
      hiddenActivation: data.hiddenActivation,
      outputActivation: data.outputActivation,
    });
    nn.layers = data.layers.map(l => ({
      weights:    l.weights,
      biases:     l.biases,
      activation: l.activation,
      inputSize:  l.inputSize,
      outputSize: l.outputSize,
    }));
    return nn;
  }
}

module.exports = NeuralNetwork;
