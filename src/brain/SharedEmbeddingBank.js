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
      const baseline = prototypes.map(row => [...row]);
      this._banks.set(inputSize, { prototypes, baseline });
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
      drift:          this.getDriftInfo(),
    };
  }

  getDriftInfo() {
    const perInputSize = {};
    let total = 0;
    let count = 0;
    for (const [size, bank] of this._banks.entries()) {
      if (!bank.baseline) continue;
      const distances = bank.prototypes.map((proto, idx) =>
        euclideanDistance(proto, bank.baseline[idx] || proto)
      );
      const avg = distances.reduce((a, b) => a + b, 0) / distances.length;
      perInputSize[size] = {
        average: avg,
        max: Math.max(...distances),
      };
      total += avg;
      count++;
    }
    return {
      average: count ? total / count : 0,
      perInputSize,
    };
  }

  toJSON() {
    return {
      embeddingSize:  this.embeddingSize,
      prototypeCount: this.prototypeCount,
      learningRate:   this.learningRate,
      banks:          [...this._banks.entries()].map(([size, bank]) => [
        size,
        {
          prototypes: bank.prototypes,
          baseline: bank.baseline,
        },
      ]),
    };
  }

  static fromJSON(data) {
    const bank = new SharedEmbeddingBank({
      embeddingSize:  data.embeddingSize,
      prototypeCount: data.prototypeCount,
      learningRate:   data.learningRate,
    });
    bank._banks = new Map((data.banks || []).map(([size, entry]) => [
      size,
      { prototypes: entry.prototypes, baseline: entry.baseline || entry.prototypes.map(row => [...row]) },
    ]));
    return bank;
  }
}

module.exports = SharedEmbeddingBank;
