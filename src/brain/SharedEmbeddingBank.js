'use strict';

const { euclideanDistance } = require('../utils/MathUtils');

/**
 * SharedEmbeddingBank learns reusable input prototypes across domains so new
 * regions can start from a shared representation instead of raw inputs.
 */
class SharedEmbeddingBank {
  constructor({
    embeddingSize = 8,
    prototypeCount = null,
    learningRate = 0.2,
  } = {}) {
    const effectiveSize = prototypeCount ?? embeddingSize;
    this.embeddingSize  = effectiveSize;
    this.prototypeCount = effectiveSize;
    this.learningRate   = learningRate;
    this._banks         = new Map(); // inputSize -> { prototypes: number[][] }
  }

  _ensureBank(input) {
    const inputSize = input.length;
    if (!this._banks.has(inputSize)) {
      const prototypes = Array.from({ length: this.prototypeCount }, () =>
        input.map(v => v + (Math.random() - 0.5) * 0.1)
      );
      this._banks.set(inputSize, { prototypes });
    }
    return this._banks.get(inputSize);
  }

  embed(input) {
    const bank = this._ensureBank(input);
    return bank.prototypes.map(proto => 1 / (1 + euclideanDistance(proto, input)));
  }

  update(input) {
    const bank = this._ensureBank(input);
    let bestIdx = 0;
    let bestDist = Number.POSITIVE_INFINITY;
    for (let i = 0; i < bank.prototypes.length; i++) {
      const dist = euclideanDistance(bank.prototypes[i], input);
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = i;
      }
    }
    const proto = bank.prototypes[bestIdx];
    for (let i = 0; i < proto.length; i++) {
      proto[i] += this.learningRate * (input[i] - proto[i]);
    }
  }

  updateWithSamples(samples) {
    samples.forEach(sample => this.update(sample.input));
  }

  getEmbeddingSize(inputSize) {
    return this.prototypeCount;
  }

  getInfo() {
    const sizes = [...this._banks.keys()];
    return {
      embeddingSize:  this.embeddingSize,
      prototypeCount: this.prototypeCount,
      inputSizes:     sizes,
      bankCount:      sizes.length,
    };
  }

  toJSON() {
    return {
      embeddingSize:  this.embeddingSize,
      prototypeCount: this.prototypeCount,
      learningRate:   this.learningRate,
      banks:          [...this._banks.entries()],
    };
  }

  static fromJSON(data) {
    const bank = new SharedEmbeddingBank({
      embeddingSize:  data.embeddingSize,
      prototypeCount: data.prototypeCount,
      learningRate:   data.learningRate,
    });
    bank._banks = new Map(data.banks || []);
    return bank;
  }
}

module.exports = SharedEmbeddingBank;
