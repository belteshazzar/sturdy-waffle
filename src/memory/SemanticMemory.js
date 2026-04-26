'use strict';

const { cosineSimilarity } = require('../utils/MathUtils');

/**
 * SemanticMemory stores compact facts, rules, and abstract concepts inferred
 * from episodic experiences.
 */
class SemanticMemory {
  constructor({ capacity = 2000 } = {}) {
    this.capacity = capacity;
    this.facts    = [];
    this.rules    = [];
    this.concepts = [];
    this._nextRuleId = 1;
  }

  addFact({ subject, predicate, value, confidence = 1, source = 'observation' }) {
    const existing = this.facts.find(f => f.subject === subject && f.predicate === predicate);
    const fact = {
      subject,
      predicate,
      value,
      confidence,
      source,
      updatedAt: Date.now(),
    };
    if (existing) {
      Object.assign(existing, fact);
      return existing;
    }
    this.facts.push(fact);
    this._trim();
    return fact;
  }

  addRule({ name, when, then, confidence = 0.5, support = 1, source = 'induction' }) {
    const rule = {
      name: name || `rule:${then.predicate}:${this._nextRuleId++}`,
      when: when || [],
      then,
      confidence,
      support,
      source,
      updatedAt: Date.now(),
    };
    this.rules.push(rule);
    this._trim();
    return rule;
  }

  addConcept({ domain, signature, prototypeInput, prototypeOutput, support, confidence }) {
    const concept = {
      domain,
      signature,
      prototypeInput,
      prototypeOutput,
      support,
      confidence,
      updatedAt: Date.now(),
    };
    this.concepts.push(concept);
    this._trim();
    return concept;
  }

  queryFacts({ subject, predicate } = {}) {
    return this.facts.filter(f =>
      (subject ? f.subject === subject : true) &&
      (predicate ? f.predicate === predicate : true)
    );
  }

  inferFromRules({ subject, predicate, factBase }) {
    const candidates = this.rules.filter(rule => rule.then.predicate === predicate);
    let best = null;
    for (const rule of candidates) {
      const matches = rule.when.every(cond => {
        if (!factBase) return false;
        if (cond.type === 'predicate') {
          const val = factBase.get(subject, cond.key);
          return val !== null && val === cond.value;
        }
        if (cond.type === 'attribute') {
          const val = factBase.getValue(subject, cond.key);
          return val !== null && val === cond.value;
        }
        return false;
      });
      if (matches && (!best || rule.confidence > best.confidence)) {
        best = rule;
      }
    }
    if (!best) return null;
    return {
      value:      best.then.value,
      confidence: best.confidence,
      source:     `rule:${best.name}`,
    };
  }

  inferByAnalogy({ subject, predicate, factBase }) {
    if (!factBase) return null;
    const subjects = factBase.subjects.filter(s => s !== subject);
    if (subjects.length === 0) return null;

    const targetVec = this._subjectVector(subject, factBase);
    let best = null;
    for (const other of subjects) {
      const vec = this._subjectVector(other, factBase);
      const similarity = cosineSimilarity(targetVec, vec);
      if (!best || similarity > best.similarity) {
        best = { subject: other, similarity };
      }
    }
    if (!best) return null;

    const predicted = factBase.get(best.subject, predicate);
    if (predicted === null) return null;
    return {
      value:      predicted,
      confidence: Math.max(0, Math.min(1, best.similarity)),
      source:     `analogy:${best.subject}`,
    };
  }

  induceRulesFromFactBase(factBase, { minSupport = 2, minConfidence = 0.8 } = {}) {
    if (!factBase) return [];
    const rules = [];
    const subjects = factBase.subjects;

    for (const predicate of factBase.predicates) {
      for (const attribute of factBase.attributes) {
        const vocab = factBase.getAttributeVocabulary(attribute) || [];
        for (const value of vocab) {
          let support = 0;
          let positives = 0;
          for (const subject of subjects) {
            const attrVal = factBase.getValue(subject, attribute);
            if (attrVal !== value) continue;
            support++;
            const predVal = factBase.get(subject, predicate);
            if (predVal === 1) positives++;
          }
          if (support >= minSupport) {
            const confidence = positives / support;
            if (confidence >= minConfidence || confidence <= (1 - minConfidence)) {
              const rule = this.addRule({
                when: [{ type: 'attribute', key: attribute, value }],
                then: { predicate, value: confidence >= 0.5 ? 1 : 0 },
                confidence: Math.max(confidence, 1 - confidence),
                support,
                source: 'factBaseInduction',
              });
              rules.push(rule);
            }
          }
        }
      }
    }
    return rules;
  }

  _subjectVector(subject, factBase) {
    const predicateVec = factBase.predicates.map(p => {
      const val = factBase.get(subject, p);
      return val === null ? 0.5 : val;
    });
    const attributeVec = factBase.attributes.flatMap(attr => {
      const vocab = factBase.getAttributeVocabulary(attr) || [];
      const val = factBase.getValue(subject, attr);
      return vocab.map(v => (v === val ? 1 : 0));
    });
    return [...predicateVec, ...attributeVec];
  }

  _trim() {
    while (this.facts.length + this.rules.length + this.concepts.length > this.capacity) {
      if (this.concepts.length > 0) {
        this.concepts.sort((a, b) => a.confidence - b.confidence);
        this.concepts.shift();
      } else if (this.rules.length > 0) {
        this.rules.sort((a, b) => a.confidence - b.confidence);
        this.rules.shift();
      } else {
        this.facts.sort((a, b) => a.confidence - b.confidence);
        this.facts.shift();
      }
    }
  }

  getInfo() {
    return {
      capacity:    this.capacity,
      factCount:   this.facts.length,
      ruleCount:   this.rules.length,
      conceptCount: this.concepts.length,
    };
  }

  toJSON() {
    return {
      capacity: this.capacity,
      facts:    this.facts,
      rules:    this.rules,
      concepts: this.concepts,
      nextRuleId: this._nextRuleId,
    };
  }

  static fromJSON(data) {
    const mem = new SemanticMemory({ capacity: data.capacity });
    mem.facts    = [...(data.facts || [])];
    mem.rules    = [...(data.rules || [])];
    mem.concepts = [...(data.concepts || [])];
    mem._nextRuleId = data.nextRuleId || (mem.rules.length + 1);
    return mem;
  }
}

module.exports = SemanticMemory;
