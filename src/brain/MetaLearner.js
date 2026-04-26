'use strict';

/**
 * MetaLearner stores averaged network parameters per task signature so that
 * new regions can start from a few-shot-friendly initialization.
 */
class MetaLearner {
  constructor({ maxSnapshots = 32 } = {}) {
    this.maxSnapshots = maxSnapshots;
    this._templates = new Map(); // signature -> { count, weights }
  }

  _signature({ inputSize, outputSize, architecture, mode }) {
    const arch = architecture ? architecture.join('-') : `${inputSize}-?-${outputSize}`;
    return `${mode || 'classification'}:${arch}`;
  }

  registerRegion(region) {
    if (!region || !region.network) return;
    const signature = this._signature({
      inputSize:   region.network.architecture[0],
      outputSize:  region.network.architecture[region.network.architecture.length - 1],
      architecture: region.network.architecture,
      mode:        region.lesson ? region.lesson.mode : 'classification',
    });

    const snapshot = region.network.toJSON();
    if (!this._templates.has(signature)) {
      this._templates.set(signature, { count: 1, weights: snapshot });
      return;
    }

    const existing = this._templates.get(signature);
    existing.count += 1;
    existing.weights = this._averageNetwork(existing.weights, snapshot, existing.count);
    if (existing.count > this.maxSnapshots) {
      existing.count = this.maxSnapshots;
    }
  }

  getInitialWeights({ inputSize, outputSize, architecture, mode }) {
    const signature = this._signature({ inputSize, outputSize, architecture, mode });
    const template = this._templates.get(signature);
    if (!template) return null;
    return JSON.parse(JSON.stringify(template.weights));
  }

  _averageNetwork(base, next, count) {
    const averaged = JSON.parse(JSON.stringify(base));
    const countMinusOne = count - 1;
    averaged.layers = averaged.layers.map((layer, idx) => {
      const nextLayer = next.layers[idx];
      return {
        ...layer,
        weights: layer.weights.map((row, i) =>
          row.map((value, j) => (value * countMinusOne + nextLayer.weights[i][j]) / count)
        ),
        biases: layer.biases.map((value, i) =>
          (value * countMinusOne + nextLayer.biases[i]) / count
        ),
      };
    });
    return averaged;
  }

  getInfo() {
    return {
      templateCount: this._templates.size,
      signatures:    [...this._templates.keys()],
    };
  }

  toJSON() {
    return {
      maxSnapshots: this.maxSnapshots,
      templates:    [...this._templates.entries()],
    };
  }

  static fromJSON(data) {
    const learner = new MetaLearner({ maxSnapshots: data.maxSnapshots });
    learner._templates = new Map(data.templates || []);
    return learner;
  }
}

module.exports = MetaLearner;
