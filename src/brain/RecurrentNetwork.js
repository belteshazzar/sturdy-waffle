'use strict';

const { xavierInit } = require('../utils/MathUtils');
const { activate, activatePrime } = require('../utils/ActivationFunctions');

class RecurrentNetwork {
  constructor({
    inputSize,
    hiddenSize = 16,
    outputSize,
    learningRate = 0.05,
    hiddenActivation = 'tanh',
    outputActivation = 'linear',
  } = {}) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;
    this.learningRate = learningRate;
    this.hiddenActivation = hiddenActivation;
    this.outputActivation = outputActivation;
    this.architecture = [inputSize, hiddenSize, outputSize];

    this.Wxh = Array.from({ length: hiddenSize }, () =>
      Array.from({ length: inputSize }, () => xavierInit(inputSize, hiddenSize))
    );
    this.Whh = Array.from({ length: hiddenSize }, () =>
      Array.from({ length: hiddenSize }, () => xavierInit(hiddenSize, hiddenSize))
    );
    this.Why = Array.from({ length: outputSize }, () =>
      Array.from({ length: hiddenSize }, () => xavierInit(hiddenSize, outputSize))
    );
    this.bh = new Array(hiddenSize).fill(0);
    this.by = new Array(outputSize).fill(0);
  }

  _matVec(matrix, vec) {
    return matrix.map(row => row.reduce((sum, w, j) => sum + w * vec[j], 0));
  }

  _vecAdd(a, b) {
    return a.map((v, i) => v + b[i]);
  }

  _vecMulScalar(vec, s) {
    return vec.map(v => v * s);
  }

  _transposeMul(matrix, vec) {
    const cols = matrix[0].length;
    const result = new Array(cols).fill(0);
    for (let i = 0; i < matrix.length; i++) {
      for (let j = 0; j < cols; j++) {
        result[j] += matrix[i][j] * vec[i];
      }
    }
    return result;
  }

  predict(sequence) {
    const { hiddenStates } = this._forward(sequence);
    const last = hiddenStates[hiddenStates.length - 1];
    const yLinear = this._vecAdd(this._matVec(this.Why, last), this.by);
    return yLinear.map(v => activate(v, this.outputActivation));
  }

  predictBinary(sequence, threshold = 0.5) {
    return this.predict(sequence).map(v => (v >= threshold ? 1 : 0));
  }

  predictArgmax(sequence) {
    const output = this.predict(sequence);
    let maxIdx = 0;
    for (let i = 1; i < output.length; i++) {
      if (output[i] > output[maxIdx]) maxIdx = i;
    }
    return maxIdx;
  }

  _forward(sequence) {
    const inputs = this._normalizeSequence(sequence);
    const hiddenStates = [];
    const hiddenLinears = [];
    let prev = new Array(this.hiddenSize).fill(0);
    for (const x of inputs) {
      const hLinear = this._vecAdd(
        this._vecAdd(this._matVec(this.Wxh, x), this._matVec(this.Whh, prev)),
        this.bh
      );
      const h = hLinear.map(v => activate(v, this.hiddenActivation));
      hiddenLinears.push(hLinear);
      hiddenStates.push(h);
      prev = h;
    }
    return { inputs, hiddenStates, hiddenLinears };
  }

  _normalizeSequence(sequence) {
    if (!Array.isArray(sequence)) return [];
    if (sequence.length === 0) return [];
    if (Array.isArray(sequence[0])) return sequence;
    return sequence.map(v => [v]);
  }

  train(samples, epochs = 20) {
    if (!samples || samples.length === 0) return { finalLoss: 0, losses: [] };
    const losses = [];
    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0;
      for (const sample of samples) {
        totalLoss += this._trainSample(sample);
      }
      losses.push(totalLoss / samples.length);
    }
    return { finalLoss: losses[losses.length - 1], losses };
  }

  _trainSample(sample) {
    const target = sample.output;
    const { inputs, hiddenStates, hiddenLinears } = this._forward(sample.input);
    const lastHidden = hiddenStates[hiddenStates.length - 1] || new Array(this.hiddenSize).fill(0);
    const yLinear = this._vecAdd(this._matVec(this.Why, lastHidden), this.by);
    const output = yLinear.map(v => activate(v, this.outputActivation));

    const dy = output.map((v, i) => (v - target[i]) * activatePrime(yLinear[i], this.outputActivation));

    // Gradients
    const dWhy = Array.from({ length: this.outputSize }, () => new Array(this.hiddenSize).fill(0));
    const dby = new Array(this.outputSize).fill(0);
    for (let i = 0; i < this.outputSize; i++) {
      for (let j = 0; j < this.hiddenSize; j++) {
        dWhy[i][j] += dy[i] * lastHidden[j];
      }
      dby[i] += dy[i];
    }

    let dh = this._transposeMul(this.Why, dy);

    const dWxh = Array.from({ length: this.hiddenSize }, () => new Array(this.inputSize).fill(0));
    const dWhh = Array.from({ length: this.hiddenSize }, () => new Array(this.hiddenSize).fill(0));
    const dbh = new Array(this.hiddenSize).fill(0);

    for (let t = hiddenStates.length - 1; t >= 0; t--) {
      const hLinear = hiddenLinears[t];
      const hPrev = t > 0 ? hiddenStates[t - 1] : new Array(this.hiddenSize).fill(0);
      const dhRaw = dh.map((v, i) => v * activatePrime(hLinear[i], this.hiddenActivation));
      const x = inputs[t];
      for (let i = 0; i < this.hiddenSize; i++) {
        for (let j = 0; j < this.inputSize; j++) {
          dWxh[i][j] += dhRaw[i] * x[j];
        }
        for (let j = 0; j < this.hiddenSize; j++) {
          dWhh[i][j] += dhRaw[i] * hPrev[j];
        }
        dbh[i] += dhRaw[i];
      }
      dh = this._transposeMul(this.Whh, dhRaw);
    }

    const lr = this.learningRate;
    for (let i = 0; i < this.hiddenSize; i++) {
      for (let j = 0; j < this.inputSize; j++) {
        this.Wxh[i][j] -= lr * dWxh[i][j];
      }
      for (let j = 0; j < this.hiddenSize; j++) {
        this.Whh[i][j] -= lr * dWhh[i][j];
      }
      this.bh[i] -= lr * dbh[i];
    }
    for (let i = 0; i < this.outputSize; i++) {
      for (let j = 0; j < this.hiddenSize; j++) {
        this.Why[i][j] -= lr * dWhy[i][j];
      }
      this.by[i] -= lr * dby[i];
    }

    const loss = output.reduce((acc, v, i) => acc + (v - target[i]) ** 2, 0) / output.length;
    return loss;
  }

  accuracy(samples, threshold = 0.5) {
    let correct = 0;
    for (const sample of samples) {
      const predicted = this.predictBinary(sample.input, threshold);
      if (predicted.every((p, i) => p === sample.output[i])) correct++;
    }
    return correct / samples.length;
  }

  regressionAccuracy(samples, tolerance = 0.05) {
    let correct = 0;
    for (const sample of samples) {
      const predicted = this.predict(sample.input);
      if (predicted.every((p, i) => Math.abs(p - sample.output[i]) <= tolerance)) correct++;
    }
    return correct / samples.length;
  }

  multiclassAccuracy(samples) {
    let correct = 0;
    for (const sample of samples) {
      const predicted = this.predictArgmax(sample.input);
      let targetIdx = 0;
      for (let i = 1; i < sample.output.length; i++) {
        if (sample.output[i] > sample.output[targetIdx]) targetIdx = i;
      }
      if (predicted === targetIdx) correct++;
    }
    return correct / samples.length;
  }

  hammingAccuracy(samples, threshold = 0.5) {
    let correctLabels = 0;
    let totalLabels = 0;
    for (const sample of samples) {
      const predicted = this.predictBinary(sample.input, threshold);
      for (let i = 0; i < predicted.length; i++) {
        if (predicted[i] === sample.output[i]) correctLabels++;
        totalLabels++;
      }
    }
    return totalLabels === 0 ? 0 : correctLabels / totalLabels;
  }

  toJSON() {
    return {
      inputSize: this.inputSize,
      hiddenSize: this.hiddenSize,
      outputSize: this.outputSize,
      learningRate: this.learningRate,
      hiddenActivation: this.hiddenActivation,
      outputActivation: this.outputActivation,
      Wxh: this.Wxh,
      Whh: this.Whh,
      Why: this.Why,
      bh: this.bh,
      by: this.by,
    };
  }

  static fromJSON(data) {
    const net = new RecurrentNetwork({
      inputSize: data.inputSize,
      hiddenSize: data.hiddenSize,
      outputSize: data.outputSize,
      learningRate: data.learningRate,
      hiddenActivation: data.hiddenActivation,
      outputActivation: data.outputActivation,
    });
    net.Wxh = data.Wxh;
    net.Whh = data.Whh;
    net.Why = data.Why;
    net.bh = data.bh;
    net.by = data.by;
    return net;
  }
}

module.exports = RecurrentNetwork;
