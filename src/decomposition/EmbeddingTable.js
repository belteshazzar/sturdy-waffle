'use strict';

/**
 * EmbeddingTable — a trainable dense lookup table that maps discrete token IDs
 * to continuous vector representations.
 *
 * Neuroscience analogue: distributed representations in cortical columns.
 * Rather than treating each token as orthogonal (one-hot), the embedding table
 * lets the network discover that semantically related operators (AND/NAND,
 * OR/NOR) occupy nearby regions of the representation space.
 *
 * The table is trained jointly with the policy network via backpropagation:
 * input-layer gradients are computed after each policy update and used to nudge
 * the embedding for each token that appeared in the current state.
 *
 * NULL tokens (id −1) always map to an all-zeros vector and are never updated.
 */
class EmbeddingTable {
  /**
   * @param {object} opts
   * @param {number} opts.vocabSize    Number of distinct (non-NULL) token IDs
   * @param {number} opts.dim          Embedding dimension
   * @param {number} [opts.learningRate=0.01]
   */
  constructor({ vocabSize, dim, learningRate = 0.01 }) {
    this.vocabSize    = vocabSize;
    this.dim          = dim;
    this.learningRate = learningRate;

    // Small random initialisation (uniform in [-0.1, 0.1])
    this.embeddings = Array.from({ length: vocabSize }, () =>
      Array.from({ length: dim }, () => (Math.random() - 0.5) * 0.2)
    );
  }

  // ── Lookup ────────────────────────────────────────────────────────────────

  /**
   * Return the embedding vector for `tokenId`.
   * NULL tokens (id < 0) and out-of-range IDs return an all-zeros vector.
   * The returned array is a copy — mutating it does not affect the table.
   *
   * @param {number} tokenId
   * @returns {number[]}  Length = this.dim
   */
  lookup(tokenId) {
    if (tokenId < 0 || tokenId >= this.vocabSize) {
      return new Array(this.dim).fill(0);
    }
    return [...this.embeddings[tokenId]];
  }

  // ── Gradient update ───────────────────────────────────────────────────────

  /**
   * Apply a gradient-descent step to the embedding for `tokenId`.
   *
   * @param {number}   tokenId    ID of the token whose embedding to update
   * @param {number[]} gradSlice  Gradient slice of length `this.dim`
   */
  update(tokenId, gradSlice) {
    if (tokenId < 0 || tokenId >= this.vocabSize) return;
    const emb = this.embeddings[tokenId];
    for (let k = 0; k < this.dim; k++) {
      emb[k] -= this.learningRate * gradSlice[k];
    }
  }

  // ── Serialisation ─────────────────────────────────────────────────────────

  toJSON() {
    return {
      vocabSize:    this.vocabSize,
      dim:          this.dim,
      learningRate: this.learningRate,
      embeddings:   this.embeddings.map(row => [...row]),
    };
  }

  static fromJSON(data) {
    const t = new EmbeddingTable({
      vocabSize:    data.vocabSize,
      dim:          data.dim,
      learningRate: data.learningRate,
    });
    t.embeddings = data.embeddings.map(row => [...row]);
    return t;
  }
}

module.exports = EmbeddingTable;
