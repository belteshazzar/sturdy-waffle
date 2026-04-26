'use strict';

const { euclideanDistance } = require('../utils/MathUtils');

/**
 * EpisodicMemory stores concrete experiences (input/output pairs) along with
 * metadata for replay, retrieval, and consolidation.
 */
class EpisodicMemory {
  /**
   * @param {object} [opts]
   * @param {number} [opts.capacity=2000]
   */
  constructor({ capacity = 2000 } = {}) {
    this.capacity = capacity;
    this.episodes = [];
    this._nextId  = 1;
  }

  addEpisode({ domain, input, output, tags = [], metadata = {} }) {
    const episode = {
      id:        this._nextId++,
      domain,
      input:     [...input],
      output:    [...output],
      tags:      [...tags],
      metadata:  { ...metadata },
      timestamp: Date.now(),
    };
    this.episodes.push(episode);
    if (this.episodes.length > this.capacity) {
      this.episodes.shift();
    }
    return episode;
  }

  /**
   * Return the closest episodes to a query input (same domain).
   */
  query({ domain, input, limit = 5 }) {
    const candidates = this.episodes.filter(ep => (domain ? ep.domain === domain : true));
    if (!input || candidates.length === 0) return candidates.slice(-limit);

    const scored = candidates.map(ep => ({
      episode: ep,
      distance: euclideanDistance(ep.input, input),
    }));
    scored.sort((a, b) => a.distance - b.distance);
    return scored.slice(0, limit).map(s => s.episode);
  }

  /**
   * Random sample for replay.
   */
  sample({ domain, limit = 10 } = {}) {
    const candidates = this.episodes.filter(ep => (domain ? ep.domain === domain : true));
    if (candidates.length <= limit) return [...candidates];
    const sample = [];
    for (let i = 0; i < limit; i++) {
      const idx = Math.floor(Math.random() * candidates.length);
      sample.push(candidates[idx]);
    }
    return sample;
  }

  getInfo() {
    const domains = new Set(this.episodes.map(ep => ep.domain));
    return {
      capacity:     this.capacity,
      count:        this.episodes.length,
      domainCount:  domains.size,
      domains:      [...domains],
      latestSample: this.episodes[this.episodes.length - 1] || null,
    };
  }

  toJSON() {
    return {
      capacity: this.capacity,
      episodes: this.episodes,
      nextId:   this._nextId,
    };
  }

  static fromJSON(data) {
    const mem = new EpisodicMemory({ capacity: data.capacity });
    mem.episodes = [...(data.episodes || [])];
    mem._nextId  = data.nextId || (mem.episodes.length + 1);
    return mem;
  }
}

module.exports = EpisodicMemory;
