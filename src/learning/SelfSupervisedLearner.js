'use strict';

const NeuralNetwork = require('../brain/NeuralNetwork');

/**
 * SelfSupervisedLearner trains lightweight autoencoders per input size to
 * learn reconstruction-based representations without labels.
 */
class SelfSupervisedLearner {
  constructor({ hiddenScale = 2, learningRate = 0.15 } = {}) {
    this.hiddenScale = hiddenScale;
    this.learningRate = learningRate;
    this.autoencoders = new Map(); // inputSize -> NeuralNetwork
    this.lastLoss = null;
  }

  _ensureAutoencoder(inputSize) {
    if (!this.autoencoders.has(inputSize)) {
      const hidden = Math.max(4, inputSize * this.hiddenScale);
      const net = new NeuralNetwork({
        architecture:     [inputSize, hidden, inputSize],
        learningRate:     this.learningRate,
        hiddenActivation: 'tanh',
        outputActivation: 'linear',
      });
      this.autoencoders.set(inputSize, net);
    }
    return this.autoencoders.get(inputSize);
  }

  trainFromEpisodes(episodes, { epochs = 40 } = {}) {
    if (!episodes || episodes.length === 0) return { trained: false, loss: null };
    const grouped = new Map();
    for (const ep of episodes) {
      const size = ep.input.length;
      if (!grouped.has(size)) grouped.set(size, []);
      grouped.get(size).push({ input: ep.input, output: ep.input });
    }

    let totalLoss = 0;
    let groupCount = 0;
    for (const [size, samples] of grouped.entries()) {
      const net = this._ensureAutoencoder(size);
      const { finalLoss } = net.train(samples, epochs);
      totalLoss += finalLoss;
      groupCount++;
    }
    this.lastLoss = groupCount > 0 ? totalLoss / groupCount : null;
    return { trained: groupCount > 0, loss: this.lastLoss };
  }

  reconstruct(input) {
    const net = this.autoencoders.get(input.length);
    return net ? net.predict(input) : [...input];
  }

  getInfo() {
    return {
      autoencoderCount: this.autoencoders.size,
      inputSizes:       [...this.autoencoders.keys()],
      lastLoss:         this.lastLoss,
    };
  }

  toJSON() {
    return {
      hiddenScale:  this.hiddenScale,
      learningRate: this.learningRate,
      lastLoss:     this.lastLoss,
      autoencoders: [...this.autoencoders.entries()].map(([size, net]) => [size, net.toJSON()]),
    };
  }

  static fromJSON(data) {
    const learner = new SelfSupervisedLearner({
      hiddenScale:  data.hiddenScale,
      learningRate: data.learningRate,
    });
    learner.lastLoss = data.lastLoss || null;
    learner.autoencoders = new Map((data.autoencoders || []).map(([size, netData]) => [
      size,
      NeuralNetwork.fromJSON(netData),
    ]));
    return learner;
  }
}

module.exports = SelfSupervisedLearner;
