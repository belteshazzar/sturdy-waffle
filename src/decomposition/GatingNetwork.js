'use strict';

const NeuralNetwork = require('../brain/NeuralNetwork');

/**
 * GatingNetwork — a small neural network that learns which working-memory
 * positions are valid reduction candidates purely from learned slot embeddings.
 *
 * Neuroscience analogue: basal-ganglia go/no-go circuitry.  The striatum
 * evaluates each candidate action and gates whether it reaches the thalamus
 * (is selected for execution).  Here each slot's embedding is scored
 * independently to decide if it can be the start of a valid reduction.
 *
 * Architecture
 * ────────────
 *  Input  : single slot embedding (length = embeddingDim)
 *  Hidden : one hidden layer of `hiddenSize` neurons (tanh)
 *  Output : 1 sigmoid neuron — probability that this slot starts a valid reduction
 *
 * Training signal
 * ───────────────
 *  For every step in an imitation trace, every slot position that was in
 *  validReductions() at that moment gets target = 1; all others get target = 0.
 *  This trains the gating network to recognise reducible patterns from
 *  embeddings alone, without relying on the hard-coded ARITY table.
 *
 * Usage
 * ─────
 *  Once trained with sufficient accuracy, the gating network replaces the
 *  symbolic ARITY-based filter inside DecompositionController.selectAction().
 *  The hard-coded fallback (validReductions()) is still available when no
 *  gating network exists or when gating confidence is low.
 */
class GatingNetwork {
  /**
   * @param {object} opts
   * @param {number} opts.embeddingDim     Dimension of each slot embedding
   * @param {number} [opts.hiddenSize=16]  Neurons in the hidden layer
   * @param {number} [opts.learningRate=0.05]
   * @param {number} [opts.threshold=0.5]  Score threshold for "valid" classification
   */
  constructor({ embeddingDim, hiddenSize = 16, learningRate = 0.05, threshold = 0.5 }) {
    this.embeddingDim = embeddingDim;
    this.threshold    = threshold;

    this.network = new NeuralNetwork({
      architecture:     [embeddingDim, hiddenSize, 1],
      learningRate,
      hiddenActivation: 'tanh',
      outputActivation: 'sigmoid',
    });
  }

  // ── Inference ─────────────────────────────────────────────────────────────

  /**
   * Score a single slot embedding: returns the probability that this slot
   * starts a valid reduction.
   *
   * @param {number[]} slotEmbedding  Length = embeddingDim
   * @returns {number}  Probability in [0, 1]
   */
  scoreSlot(slotEmbedding) {
    return this.network.predict(slotEmbedding)[0];
  }

  /**
   * Predict which slot indices (among `slotCount`) are valid reduction starts,
   * using the provided embedding table to look up each slot's representation.
   *
   * Returns only indices whose score exceeds `this.threshold`.
   *
   * @param {number[]} slotTokens    Raw token IDs for each slot (length ≥ slotCount)
   * @param {number}   slotCount     Number of active slots
   * @param {object}   embeddingTable EmbeddingTable instance
   * @returns {number[]}  Predicted valid start indices
   */
  predictValidStarts(slotTokens, slotCount, embeddingTable) {
    const valid = [];
    for (let i = 0; i < slotCount; i++) {
      const emb   = embeddingTable.lookup(slotTokens[i]);
      const score = this.scoreSlot(emb);
      if (score >= this.threshold) valid.push(i);
    }
    return valid;
  }

  // ── Training ──────────────────────────────────────────────────────────────

  /**
   * Train the gating network on labelled (embedding, valid) examples.
   *
   * @param {Array<{ embedding: number[], valid: 0|1 }>} examples
   * @param {number} [epochs=20]
   */
  train(examples, epochs = 20) {
    if (examples.length === 0) return;
    const samples = examples.map(e => ({
      input:  e.embedding,
      output: [e.valid],
    }));
    this.network.train(samples, epochs);
  }

  // ── Serialisation ─────────────────────────────────────────────────────────

  toJSON() {
    return {
      embeddingDim: this.embeddingDim,
      threshold:    this.threshold,
      network:      this.network.toJSON(),
    };
  }

  static fromJSON(data) {
    const g = new GatingNetwork({
      embeddingDim: data.embeddingDim,
      threshold:    data.threshold,
    });
    g.network = NeuralNetwork.fromJSON(data.network);
    return g;
  }
}

module.exports = GatingNetwork;
