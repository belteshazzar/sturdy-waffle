'use strict';

/**
 * VQCodebook — a Vector Quantisation codebook for vocabulary-free operation
 * discovery (Phase 5).
 *
 * Neuroscience analogue: conceptual chunking and prototype formation in
 * category-selective regions of inferior temporal cortex.  Rather than using
 * hand-crafted token IDs, the system discovers its own discrete operation
 * vocabulary through experience: continuous encoder outputs are "snapped" to
 * the nearest prototype in the codebook, and the codebook entries are updated
 * via exponential moving-average to track the representations that matter.
 *
 * Algorithm (VQ-VAE inspired)
 * ───────────────────────────
 *  1. An external encoder maps raw inputs (character features, embeddings, …)
 *     to continuous vectors of size `dim`.
 *  2. quantize(z) finds the nearest codebook entry e_k and returns the
 *     straight-through copy of z with index k.
 *  3. During training:
 *       • EMA update: e_k ← α·e_k + (1−α)·z   (commitment)
 *       • The encoder is trained with a combined loss:
 *           policy_loss + β * ||sg(z) − e_k||²   (VQ loss)
 *         where sg(·) is the stop-gradient operator.
 *  4. After convergence the code indices play the role of token IDs, but they
 *     were discovered entirely from data — no fixed vocabulary is assumed.
 *
 * This module provides the codebook data structure and update rule.  Training
 * the encoder that feeds it is handled externally (e.g. via NeuralNetwork +
 * EmbeddingTable gradient combination).
 */
class VQCodebook {
  /**
   * @param {object} opts
   * @param {number} [opts.numCodes=16]       Number of codebook entries
   * @param {number} opts.dim                 Dimensionality of each code vector
   * @param {number} [opts.learningRate=0.05] EMA update rate α (0 < α ≤ 1)
   * @param {number} [opts.commitment=0.25]   β — commitment loss weight
   */
  constructor({ numCodes = 16, dim, learningRate = 0.05, commitment = 0.25 }) {
    if (!dim || dim <= 0) throw new Error('VQCodebook requires a positive dim');

    this.numCodes    = numCodes;
    this.dim         = dim;
    this.learningRate = learningRate;
    this.commitment  = commitment;

    // Initialise codebook with small random vectors
    this.codebook = Array.from({ length: numCodes }, () =>
      Array.from({ length: dim }, () => (Math.random() - 0.5) * 0.2)
    );

    // Usage counts — how many times each code has been the nearest neighbour
    this.usageCounts = new Array(numCodes).fill(0);
  }

  // ── Quantisation ──────────────────────────────────────────────────────────

  /**
   * Find the nearest codebook entry to `vector` and return the quantised
   * (snapped) representation along with the code index.
   *
   * The returned `quantized` array is a **copy** of the codebook entry —
   * safe to use as a straight-through estimate in gradient computation.
   *
   * @param {number[]} vector  Continuous input vector (length = this.dim)
   * @returns {{ codeIdx: number, quantized: number[], dist: number }}
   */
  quantize(vector) {
    let bestIdx  = 0;
    let bestDist = Infinity;

    for (let i = 0; i < this.numCodes; i++) {
      const dist = this._squaredDist(vector, this.codebook[i]);
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx  = i;
      }
    }

    return {
      codeIdx:   bestIdx,
      quantized: [...this.codebook[bestIdx]],
      dist:      bestDist,
    };
  }

  // ── Codebook update (EMA) ─────────────────────────────────────────────────

  /**
   * Update the codebook entry at `codeIdx` using exponential moving average
   * toward `vector`.
   *
   *   e ← (1 − α)·e + α·z
   *
   * @param {number} codeIdx
   * @param {number[]} vector  Encoder output to move toward (length = dim)
   */
  update(codeIdx, vector) {
    if (codeIdx < 0 || codeIdx >= this.numCodes) return;
    const α   = this.learningRate;
    const e   = this.codebook[codeIdx];
    for (let k = 0; k < this.dim; k++) {
      e[k] = (1 - α) * e[k] + α * vector[k];
    }
    this.usageCounts[codeIdx]++;
  }

  /**
   * Compute the VQ commitment loss term for a single (vector, codeIdx) pair.
   * Loss = β * ||sg(z) − e_k||²
   * (In practice the gradient of this w.r.t. z is zero due to stop-gradient,
   *  but the loss value is useful for monitoring.)
   *
   * @param {number[]} vector   Encoder output z
   * @param {number}   codeIdx  Nearest codebook index
   * @returns {number}
   */
  commitmentLoss(vector, codeIdx) {
    return this.commitment * this._squaredDist(vector, this.codebook[codeIdx]);
  }

  /**
   * Return usage statistics: how often each code has been the nearest
   * neighbour since construction (or last reset).
   *
   * @returns {{ usageCounts: number[], totalAssignments: number }}
   */
  getUsageStats() {
    const total = this.usageCounts.reduce((s, c) => s + c, 0);
    return { usageCounts: [...this.usageCounts], totalAssignments: total };
  }

  /** Reset usage counters (useful between training phases). */
  resetUsage() {
    this.usageCounts.fill(0);
  }

  // ── Utilities ─────────────────────────────────────────────────────────────

  _squaredDist(a, b) {
    let d = 0;
    for (let i = 0; i < this.dim; i++) {
      const diff = a[i] - b[i];
      d += diff * diff;
    }
    return d;
  }

  // ── Serialisation ─────────────────────────────────────────────────────────

  toJSON() {
    return {
      numCodes:     this.numCodes,
      dim:          this.dim,
      learningRate: this.learningRate,
      commitment:   this.commitment,
      codebook:     this.codebook.map(row => [...row]),
      usageCounts:  [...this.usageCounts],
    };
  }

  static fromJSON(data) {
    const vq = new VQCodebook({
      numCodes:     data.numCodes,
      dim:          data.dim,
      learningRate: data.learningRate,
      commitment:   data.commitment,
    });
    vq.codebook    = data.codebook.map(row => [...row]);
    vq.usageCounts = [...data.usageCounts];
    return vq;
  }
}

module.exports = VQCodebook;
