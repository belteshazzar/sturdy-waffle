'use strict';

/**
 * ReplayBuffer — fixed-capacity circular store of problem-solving episode
 * traces, used for offline policy consolidation.
 *
 * Neuroscience analogue: hippocampal memory replay.  After each episode the
 * brain stores a compact trace (state → action → reward sequence) in the
 * hippocampus.  During offline replay training these traces are re-activated
 * to update the prefrontal cortex (DecompositionController) without
 * interfering with the specialist regions' consolidated knowledge —
 * preserving the stability/plasticity balance described in complementary
 * learning systems theory.
 *
 * The buffer is a fixed-size ring: once capacity is reached the oldest entry
 * is overwritten.  Positive (solved) traces can be sampled independently from
 * the full buffer to bias replay towards successful decomposition strategies.
 */
class ReplayBuffer {
  /**
   * @param {number} [capacity=1000]  Maximum number of stored episodes
   */
  constructor(capacity = 1000) {
    this.capacity  = capacity;
    this.buffer    = [];
    this._writeIdx = 0;   // ring-buffer write pointer
  }

  // ── Writing ───────────────────────────────────────────────────────────────

  /**
   * Add an episode trace to the buffer, overwriting the oldest entry when
   * capacity is exceeded.
   *
   * @param {{
   *   steps:  Array<{
   *     stateVec:    number[],
   *     chosenStart: number,
   *     op:          number,
   *     args:        number[],
   *     result:      number
   *   }>,
   *   reward: number,
   *   solved: boolean
   * }} trace
   */
  push(trace) {
    if (this.buffer.length < this.capacity) {
      this.buffer.push(trace);
    } else {
      this.buffer[this._writeIdx % this.capacity] = trace;
    }
    this._writeIdx++;
  }

  // ── Sampling ──────────────────────────────────────────────────────────────

  /**
   * Uniformly sample up to `batchSize` episodes from the full buffer.
   * @param {number} batchSize
   * @returns {Array}
   */
  sample(batchSize) {
    return this._sampleFrom(this.buffer, batchSize);
  }

  /**
   * Sample only from successfully solved episodes.
   * Returns an empty array when no successful traces have been stored yet.
   * @param {number} batchSize
   * @returns {Array}
   */
  samplePositive(batchSize) {
    const positive = this.buffer.filter(t => t.solved);
    return this._sampleFrom(positive, batchSize);
  }

  _sampleFrom(pool, batchSize) {
    const n      = Math.min(batchSize, pool.length);
    const chosen = new Set();
    while (chosen.size < n) {
      chosen.add(Math.floor(Math.random() * pool.length));
    }
    return Array.from(chosen).map(i => pool[i]);
  }

  // ── Stats ─────────────────────────────────────────────────────────────────

  get size()          { return this.buffer.length; }
  get positiveCount() { return this.buffer.filter(t => t.solved).length; }

  /** Plain-object summary for introspection. */
  getStats() {
    return {
      size:          this.size,
      capacity:      this.capacity,
      positiveCount: this.positiveCount,
      negativeCount: this.size - this.positiveCount,
    };
  }
}

module.exports = ReplayBuffer;
