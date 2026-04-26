'use strict';

const NeuralNetwork = require('../brain/NeuralNetwork');

/**
 * LearnedRouter — a neural network that maps a slot embedding to the name of
 * the BrainRegion (domain) that should evaluate the corresponding operator.
 *
 * Neuroscience analogue: heteromodal association cortex that learns to route
 * sensory representations to the specialist cortical areas best equipped to
 * process them.  Rather than consulting a hard-coded dispatch table, the
 * router generalises from training examples to unseen operators and domains.
 *
 * Architecture
 * ────────────
 *  Input  : single operator slot embedding (length = embeddingDim)
 *  Hidden : one hidden layer of `hiddenSize` tanh neurons
 *  Output : softmax-like sigmoid over the registered domain classes
 *
 * Training signal
 * ───────────────
 *  For each operator token seen in the curriculum, the resolved domain is used
 *  as the one-hot target class.  After sufficient training the router produces
 *  the correct domain from embedding alone, making static routing unnecessary.
 *
 * Confidence threshold
 * ────────────────────
 *  When the maximum predicted score is below `confidenceThreshold` the caller
 *  should fall back to operator-name domain resolution to prevent misrouting at
 *  low confidence.
 */
class LearnedRouter {
  /**
   * @param {object}   opts
   * @param {number}   opts.embeddingDim          Dimension of operator embeddings
   * @param {string[]} opts.domains               Ordered list of domain strings to classify
   * @param {number}   [opts.hiddenSize=32]
   * @param {number}   [opts.learningRate=0.05]
   * @param {number}   [opts.confidenceThreshold=0.7]
   */
  constructor({ embeddingDim, domains, hiddenSize = 32, learningRate = 0.05, confidenceThreshold = 0.7 }) {
    if (!Array.isArray(domains) || domains.length === 0) {
      throw new Error('LearnedRouter requires a non-empty domains array');
    }

    this.embeddingDim         = embeddingDim;
    this.domains              = [...domains];
    this.confidenceThreshold  = confidenceThreshold;

    this.network = new NeuralNetwork({
      architecture:     [embeddingDim, hiddenSize, domains.length],
      learningRate,
      hiddenActivation: 'tanh',
      outputActivation: 'sigmoid',
    });
  }

  // ── Inference ─────────────────────────────────────────────────────────────

  /**
   * Predict the most likely domain for the given operator embedding.
   *
   * @param {number[]} opEmbedding  Length = embeddingDim
   * @returns {{ domain: string, confidence: number, aboveThreshold: boolean }}
   */
  route(opEmbedding) {
    const scores     = this.network.predict(opEmbedding);
    let   maxScore   = -Infinity;
    let   maxIdx     = 0;

    for (let i = 0; i < scores.length; i++) {
      if (scores[i] > maxScore) {
        maxScore = scores[i];
        maxIdx   = i;
      }
    }

    return {
      domain:         this.domains[maxIdx],
      confidence:     maxScore,
      aboveThreshold: maxScore >= this.confidenceThreshold,
    };
  }

  // ── Training ──────────────────────────────────────────────────────────────

  /**
   * Train the router to classify operator embeddings to domain indices.
   *
   * @param {Array<{ opEmbedding: number[], domainIndex: number }>} examples
   * @param {number} [epochs=30]
   */
  train(examples, epochs = 30) {
    if (examples.length === 0) return;
    const outputSize = this.domains.length;
    const samples    = examples.map(({ opEmbedding, domainIndex }) => {
      const target = new Array(outputSize).fill(0);
      if (domainIndex >= 0 && domainIndex < outputSize) {
        target[domainIndex] = 1;
      }
      return { input: opEmbedding, output: target };
    });
    this.network.train(samples, epochs);
  }

  // ── Serialisation ─────────────────────────────────────────────────────────

  toJSON() {
    return {
      embeddingDim:        this.embeddingDim,
      domains:             [...this.domains],
      confidenceThreshold: this.confidenceThreshold,
      network:             this.network.toJSON(),
    };
  }

  static fromJSON(data) {
    const r = new LearnedRouter({
      embeddingDim:        data.embeddingDim,
      domains:             data.domains,
      confidenceThreshold: data.confidenceThreshold,
    });
    r.network = NeuralNetwork.fromJSON(data.network);
    return r;
  }
}

module.exports = LearnedRouter;
