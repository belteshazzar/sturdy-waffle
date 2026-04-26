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
          const meta = factBase.getFactMeta ? factBase.getFactMeta(subject, predicate) : null;
          this.semantic.addFact({
            subject,
            predicate,
            value,
            confidence: meta?.confidence ?? 1,
            source: meta?.source || 'factBase',
          });
        }
      }
      for (const attribute of factBase.attributes) {
        const value = factBase.getValue(subject, attribute);
        if (value !== null) {
          const meta = factBase.getAttributeMeta ? factBase.getAttributeMeta(subject, attribute) : null;
          this.semantic.addFact({
            subject,
            predicate: attribute,
            value,
            confidence: meta?.confidence ?? 1,
            source: meta?.source || 'factBaseAttribute',
          });
        }
      }
    }
    const relationFacts = factBase.getRelationFacts ? factBase.getRelationFacts() : [];
    relationFacts.forEach(({ relation, args, value }) => {
      const meta = factBase.getRelationMeta ? factBase.getRelationMeta(relation, args) : null;
      this.semantic.addRelationFact({
        relation,
        args,
        value,
        confidence: meta?.confidence ?? 1,
        source: meta?.source || 'factBaseRelation',
      });
    });
  }

  recordTextStatements(statements, { defaultSource = 'text' } = {}) {
    if (!Array.isArray(statements) || statements.length === 0) return;
    statements.forEach((statement, index) => {
      const meta = statement.meta || {};
      const confidence = meta.confidence ?? 1;
      const source = meta.source || defaultSource;
      const line = statement.line || index + 1;
      if (statement.kind === 'fact') {
        this.semantic.addFact({
          subject: statement.subject,
          predicate: statement.predicate,
          value: statement.value ? 1 : 0,
          confidence,
          source,
        });
        this.episodic.addEpisode({
          domain: 'text.fact',
          input: [statement.subject, statement.predicate],
          output: [statement.value ? 1 : 0],
          tags: ['text'],
          metadata: { line, source, confidence },
        });
      }
      if (statement.kind === 'attribute') {
        this.semantic.addFact({
          subject: statement.subject,
          predicate: statement.attribute,
          value: statement.value,
          confidence,
          source,
        });
        this.episodic.addEpisode({
          domain: 'text.attribute',
          input: [statement.subject, statement.attribute],
          output: [statement.value],
          tags: ['text'],
          metadata: { line, source, confidence },
        });
      }
      if (statement.kind === 'relation') {
        this.semantic.addRelationFact({
          relation: statement.name,
          args: statement.args,
          value: statement.value ? 1 : 0,
          confidence,
          source,
        });
        this.episodic.addEpisode({
          domain: 'text.relation',
          input: [statement.name, ...statement.args],
          output: [statement.value ? 1 : 0],
          tags: ['text'],
          metadata: { line, source, confidence },
        });
      }
    });
  }

  recordDecompositionStep({ state, action, nextState, reward = null, predictionError = null } = {}) {
    if (!state || !nextState) return null;
    const input = Array.isArray(action) ? [...state, ...action] : [...state];
    const output = Array.isArray(nextState) ? nextState : [nextState];
    return this.episodic.addEpisode({
      domain: 'decomposition.step',
      input,
      output,
      tags: ['decomposition', 'planning'],
      metadata: {
        reward,
        predictionError,
      },
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

  validate({ strict = false } = {}) {
    const issues = [];
    if (!Array.isArray(this.episodic.episodes)) {
      issues.push('Episodic memory episodes is not an array');
    } else {
      this.episodic.episodes.forEach((ep, idx) => {
        if (!Array.isArray(ep.input)) issues.push(`Episode ${idx} input is not array`);
        if (!Array.isArray(ep.output)) issues.push(`Episode ${idx} output is not array`);
        if (strict) {
          const bad = (arr) => arr.some(v => typeof v !== 'number' || Number.isNaN(v));
          if (Array.isArray(ep.input) && bad(ep.input)) {
            issues.push(`Episode ${idx} input contains non-numeric values`);
          }
          if (Array.isArray(ep.output) && bad(ep.output)) {
            issues.push(`Episode ${idx} output contains non-numeric values`);
          }
        }
      });
    }
    if (!Array.isArray(this.semantic.facts)) {
      issues.push('Semantic memory facts is not an array');
    }
    if (!Array.isArray(this.semantic.rules)) {
      issues.push('Semantic memory rules is not an array');
    }
    if (!Array.isArray(this.semantic.relations)) {
      issues.push('Semantic memory relations is not an array');
    }
    if (!Array.isArray(this.semantic.concepts)) {
      issues.push('Semantic memory concepts is not an array');
    }
    return { valid: issues.length === 0, issues };
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
