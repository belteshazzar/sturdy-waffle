'use strict';

const EpisodicMemory = require('./EpisodicMemory');
const SemanticMemory = require('./SemanticMemory');

class MemorySystem {
  constructor({ episodicCapacity = 2000, semanticCapacity = 2000 } = {}) {
    this.episodic = new EpisodicMemory({ capacity: episodicCapacity });
    this.semantic = new SemanticMemory({ capacity: semanticCapacity });
    this.lastConsolidation = null;
  }

  recordLesson(lesson) {
    lesson.trainingData.forEach((sample, index) => {
      this.episodic.addEpisode({
        domain: lesson.domain,
        input:  sample.input,
        output: sample.output,
        tags:   lesson.tags || [],
        metadata: {
          lesson: lesson.name,
          sampleIndex: index,
        },
      });
    });
  }

  recordFactBase(factBase) {
    if (!factBase) return;
    for (const subject of factBase.subjects) {
      for (const predicate of factBase.predicates) {
        const value = factBase.get(subject, predicate);
        if (value !== null) {
          this.semantic.addFact({
            subject,
            predicate,
            value,
            confidence: 1,
            source: 'factBase',
          });
        }
      }
      for (const attribute of factBase.attributes) {
        const value = factBase.getValue(subject, attribute);
        if (value !== null) {
          this.semantic.addFact({
            subject,
            predicate: attribute,
            value,
            confidence: 1,
            source: 'factBaseAttribute',
          });
        }
      }
    }
    const relationFacts = factBase.getRelationFacts ? factBase.getRelationFacts() : [];
    relationFacts.forEach(({ relation, args, value }) => {
      this.semantic.addRelationFact({
        relation,
        args,
        value,
        confidence: 1,
        source: 'factBaseRelation',
      });
    });
  }

  consolidateEpisodes({ minSupport = 2 } = {}) {
    const grouped = new Map();
    for (const ep of this.episodic.episodes) {
      const serialize = value => {
        if (Array.isArray(value)) return `[${value.map(serialize).join('|')}]`;
        const num = Number(value);
        return Number.isNaN(num) ? String(value) : num.toFixed(3);
      };
      const signature = ep.signature || `${ep.domain}:${ep.input.map(serialize).join(',')}`;
      if (!grouped.has(signature)) {
        grouped.set(signature, { domain: ep.domain, input: ep.input, outputs: [], support: 0 });
      }
      const bucket = grouped.get(signature);
      bucket.outputs.push(ep.output);
      bucket.support++;
    }

    const concepts = [];
    for (const [signature, bucket] of grouped.entries()) {
      if (bucket.support < minSupport) continue;
      const outputLength = bucket.outputs[0].length;
      const meanOutput = new Array(outputLength).fill(0);
      for (const output of bucket.outputs) {
        for (let i = 0; i < outputLength; i++) meanOutput[i] += output[i];
      }
      for (let i = 0; i < outputLength; i++) meanOutput[i] /= bucket.support;
      const concept = this.semantic.addConcept({
        domain: bucket.domain,
        signature,
        prototypeInput: bucket.input,
        prototypeOutput: meanOutput,
        support: bucket.support,
        confidence: Math.min(1, bucket.support / (minSupport * 2)),
      });
      concepts.push(concept);
    }
    return concepts;
  }

  consolidate({ factBase, minSupport = 2, minConfidence = 0.8 } = {}) {
    const concepts = this.consolidateEpisodes({ minSupport });
    const rules = this.semantic.induceRulesFromFactBase(factBase, { minSupport, minConfidence });
    this.lastConsolidation = {
      concepts: concepts.length,
      rules: rules.length,
      timestamp: Date.now(),
    };
    return { concepts, rules };
  }

  getInfo() {
    return {
      episodic: this.episodic.getInfo(),
      semantic: this.semantic.getInfo(),
      consolidation: this.lastConsolidation,
    };
  }

  toJSON() {
    return {
      episodic: this.episodic.toJSON(),
      semantic: this.semantic.toJSON(),
      lastConsolidation: this.lastConsolidation,
    };
  }

  static fromJSON(data) {
    const mem = new MemorySystem();
    if (data && data.episodic) mem.episodic = EpisodicMemory.fromJSON(data.episodic);
    if (data && data.semantic) mem.semantic = SemanticMemory.fromJSON(data.semantic);
    mem.lastConsolidation = data.lastConsolidation || null;
    return mem;
  }
}

module.exports = MemorySystem;
