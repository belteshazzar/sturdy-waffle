'use strict';

const Lesson = require('../learning/Lesson');

/**
 * FactBase — a declarative knowledge store for ground-truth facts.
 *
 * A FactBase holds a set of (subject, predicate) → boolean triples and
 * provides the machinery to:
 *   1. Encode subjects as normalised scalars suitable for neural-network input.
 *   2. Generate one Lesson per predicate (domain: "facts.<predicate>") so that
 *      the Brain can train a specialist BrainRegion for each property.
 *   3. Serialise / deserialise the vocabulary (subjects + predicates + values)
 *      independently of the network weights, which are managed by BrainRegion.
 *
 * Encoding
 * ────────
 * Each subject is assigned an index in the order it was first asserted.
 * The index is normalised to [0, 1]: encodedValue = index / max(1, N−1),
 * where N is the total number of subjects.  A single-subject base uses 0.5.
 * This compact scalar representation lets a small classification network
 * memorise an arbitrary truth table over the subject vocabulary.
 *
 * Typical usage — binary predicates
 * ───────────────────────────────────
 *   const fb = new FactBase('Animals');
 *   fb.assert('bird',   'canFly',  true)
 *     .assert('cat',    'canFly',  false)
 *     .assert('bird',   'hasFur',  false)
 *     .assert('cat',    'hasFur',  true);
 *
 *   brain.learnFacts(fb);                      // trains facts.canFly, facts.hasFur
 *   brain.queryFact('bird', 'canFly');          // → 1
 *
 *   // Reasoning via evaluate() expression trees:
 *   brain.evaluate({
 *     op: 'AND',
 *     inputs: [
 *       { fact: { subject: 'bird', predicate: 'canFly' } },
 *       { op: 'NOT', inputs: [{ fact: { subject: 'bird', predicate: 'hasFur' } }] },
 *     ],
 *   });  // → 1  (birds can fly AND do NOT have fur)
 *
 * Typical usage — categorical attributes (multi-class)
 * ─────────────────────────────────────────────────────
 *   const fb = new FactBase('Fruits');
 *   fb.assertValue('apple',  'color', 'red')
 *     .assertValue('banana', 'color', 'yellow')
 *     .assertValue('lime',   'color', 'green');
 *
 *   brain.learnFacts(fb);                       // trains facts.color (3-output one-hot)
 *   brain.queryAttribute('apple', 'color');      // → 'red'
 */
class FactBase {
  /**
   * @param {string} [name='Facts']  Human-readable name for this knowledge base.
   */
  constructor(name = 'Facts', opts = {}) {
    let resolvedName = name;
    let resolvedOpts = opts;
    if (name && typeof name === 'object') {
      resolvedOpts = name;
      resolvedName = name.name || 'Facts';
    }
    this.name       = resolvedName;
    this.updatePolicy = resolvedOpts.updatePolicy || 'overwrite';
    this.subjects   = [];          // ordered subject vocabulary
    this._facts     = Object.create(null);   // `${subject}:${predicate}` → 0|1
    this._factMeta  = Object.create(null);   // `${subject}:${predicate}` → { confidence?, source? }
    this._predicates = [];         // ordered predicate vocabulary (insertion order)
    // Multi-class / categorical attributes
    this._attributeVocab = Object.create(null);  // attribute → string[] (ordered values)
    this._attributeFacts = Object.create(null);  // `${subject}:${attribute}` → string
    this._attributeTypes = Object.create(null);  // attribute → { type, range?, values? }
    this._attributeMeta = Object.create(null);   // `${subject}:${attribute}` → { confidence?, source? }
    this._relations = Object.create(null);       // relation → { arity, facts: { key: 0|1 } }
    this._relationMeta = Object.create(null);    // relation → { key: { confidence?, source? } }
  }

  // ── Fact management ───────────────────────────────────────────────────────

  /**
   * Assert a ground-truth fact.
   *
   * If the subject or predicate has not been seen before it is added to the
   * respective vocabulary.  Calling assert() a second time with the same
   * (subject, predicate) pair overwrites the previous value.
   *
   * @param {string}  subject    e.g. 'bird'
   * @param {string}  predicate  e.g. 'canFly'
   * @param {boolean} [value=true]
   * @returns {this}  Returns the FactBase for method chaining.
   */
  assert(subject, predicate, value = true, meta = {}) {
    if (typeof subject   !== 'string' || subject   === '') throw new Error('subject must be a non-empty string');
    if (typeof predicate !== 'string' || predicate === '') throw new Error('predicate must be a non-empty string');
    if (Object.prototype.hasOwnProperty.call(this._attributeVocab, predicate)) {
      throw new Error(
        `'${predicate}' is already defined as a categorical attribute. ` +
        'Predicate and attribute names must be distinct.'
      );
    }

    if (!this.subjects.includes(subject)) {
      this.subjects.push(subject);
    }
    if (!this._predicates.includes(predicate)) {
      this._predicates.push(predicate);
    }

    const key = `${subject}:${predicate}`;
    const existingMeta = this._factMeta[key];
    if (Object.prototype.hasOwnProperty.call(this._facts, key)) {
      if (!this._shouldOverwrite(existingMeta, meta)) {
        return this;
      }
    }
    this._facts[key] = value ? 1 : 0;
    this._storeFactMeta(subject, predicate, meta);
    return this;
  }

  /**
   * Assert a categorical (multi-class) attribute value for a subject.
   *
   * Unlike assert(), which stores boolean (0|1) facts, assertValue() stores a
   * string value drawn from an automatically-inferred finite vocabulary.
   * Calling assertValue() a second time with the same (subject, attribute) pair
   * overwrites the previous value.
   *
   * The Brain can learn these attributes via brain.learnFacts() and answer
   * open-ended "what is X?" questions via brain.queryAttribute(subject, attribute).
   *
   * Note: attribute names must not overlap with binary predicate names in the
   * same FactBase, since both share the `facts.<name>` domain namespace.
   *
   * @param {string} subject    e.g. 'apple'
   * @param {string} attribute  e.g. 'color'
   * @param {string} value      e.g. 'red'
   * @returns {this}  Returns the FactBase for method chaining.
   */
  assertValue(subject, attribute, value, meta = {}) {
    if (typeof subject   !== 'string' || subject   === '') throw new Error('subject must be a non-empty string');
    if (typeof attribute !== 'string' || attribute === '') throw new Error('attribute must be a non-empty string');
    const type = this._attributeTypes[attribute]?.type || 'categorical';
    if (type === 'categorical') {
      if (typeof value !== 'string' || value === '') {
        throw new Error('value must be a non-empty string');
      }
    }
    if (type === 'numeric') {
      if (typeof value !== 'number' || Number.isNaN(value)) {
        throw new Error('value must be a valid number');
      }
    }
    if (this._predicates.includes(attribute)) {
      throw new Error(
        `'${attribute}' is already defined as a binary predicate. ` +
        'Predicate and attribute names must be distinct.'
      );
    }

    if (!this.subjects.includes(subject)) {
      this.subjects.push(subject);
    }
    const key = `${subject}:${attribute}`;
    const existingMeta = this._attributeMeta[key];
    if (Object.prototype.hasOwnProperty.call(this._attributeFacts, key)) {
      if (!this._shouldOverwrite(existingMeta, meta)) {
        return this;
      }
    }
    if (type === 'categorical') {
      if (!Object.prototype.hasOwnProperty.call(this._attributeVocab, attribute)) {
        this._attributeVocab[attribute] = [];
      }
      if (!this._attributeVocab[attribute].includes(value)) {
        this._attributeVocab[attribute].push(value);
      }
      this._attributeFacts[key] = value;
    } else {
      this._attributeFacts[key] = value;
      const existing = this._attributeTypes[attribute];
      if (existing && existing.type === 'numeric') {
        const range = existing.range || [value, value];
        range[0] = Math.min(range[0], value);
        range[1] = Math.max(range[1], value);
        this._attributeTypes[attribute] = { ...existing, range };
      }
    }
    this._storeAttributeMeta(subject, attribute, meta);
    return this;
  }

  /**
   * Define an attribute's data type.
   * @param {string} attribute
   * @param {{ type: 'categorical'|'numeric'|'boolean', values?: string[], range?: number[] }} opts
   */
  defineAttribute(attribute, opts = {}) {
    if (!attribute) throw new Error('attribute must be provided');
    const type = opts.type || 'categorical';
    this._attributeTypes[attribute] = { type, values: opts.values, range: opts.range };
    if (type === 'categorical' && opts.values) {
      this._attributeVocab[attribute] = [...opts.values];
    }
    return this;
  }

  /**
   * Retrieve metadata for a (subject, predicate) fact pair.
   *
   * @param {string} subject
   * @param {string} predicate
   * @returns {{ confidence?: number, source?: string }|null}
   */
  getFactMeta(subject, predicate) {
    const key = `${subject}:${predicate}`;
    return this._factMeta[key] ? { ...this._factMeta[key] } : null;
  }

  /**
   * Retrieve metadata for a (subject, attribute) pair.
   *
   * @param {string} subject
   * @param {string} attribute
   * @returns {{ confidence?: number, source?: string }|null}
   */
  getAttributeMeta(subject, attribute) {
    const key = `${subject}:${attribute}`;
    return this._attributeMeta[key] ? { ...this._attributeMeta[key] } : null;
  }

  /**
   * Retrieve metadata for a relation fact.
   *
   * @param {string} relation
   * @param {string[]} args
   * @returns {{ confidence?: number, source?: string }|null}
   */
  getRelationMeta(relation, args) {
    const entry = this._relationMeta[relation];
    if (!entry) return null;
    const key = args.join('|');
    return entry[key] ? { ...entry[key] } : null;
  }

  /**
   * Retrieve the stored categorical value for a (subject, attribute) pair.
   * Returns null when no value has been asserted.
   *
   * @param {string} subject
   * @param {string} attribute
   * @returns {string|null}
   */
  getValue(subject, attribute) {
    const key = `${subject}:${attribute}`;
    return Object.prototype.hasOwnProperty.call(this._attributeFacts, key)
      ? this._attributeFacts[key]
      : null;
  }

  /**
   * Return the ordered list of distinct values observed for a given attribute.
   * The order matches the one-hot encoding used by toLessons().
   * Returns null when the attribute is unknown.
   *
   * @param {string} attribute
   * @returns {string[]|null}
   */
  getAttributeVocabulary(attribute) {
    return Object.prototype.hasOwnProperty.call(this._attributeVocab, attribute)
      ? [...this._attributeVocab[attribute]]
      : null;
  }

  getAttributeDefinition(attribute) {
    return this._attributeTypes[attribute] ? { ...this._attributeTypes[attribute] } : null;
  }

  // ── Relations ─────────────────────────────────────────────────────────────

  assertRelation(relation, args, value = true, meta = {}) {
    if (typeof relation !== 'string' || relation === '') throw new Error('relation must be a non-empty string');
    if (!Array.isArray(args) || args.length === 0) throw new Error('relation args must be a non-empty array');
    args.forEach(arg => {
      if (typeof arg !== 'string' || arg === '') throw new Error('relation args must be non-empty strings');
      if (!this.subjects.includes(arg)) this.subjects.push(arg);
    });
    if (!this._relations[relation]) {
      this._relations[relation] = { arity: args.length, facts: Object.create(null) };
    }
    if (this._relations[relation].arity !== args.length) {
      throw new Error(`Relation '${relation}' expects arity ${this._relations[relation].arity}`);
    }
    const key = args.join('|');
    const existingMeta = this._relationMeta[relation] ? this._relationMeta[relation][key] : null;
    if (Object.prototype.hasOwnProperty.call(this._relations[relation].facts, key)) {
      if (!this._shouldOverwrite(existingMeta, meta)) {
        return this;
      }
    }
    this._relations[relation].facts[key] = value ? 1 : 0;
    this._storeRelationMeta(relation, args, meta);
    return this;
  }

  getRelation(relation, args) {
    const entry = this._relations[relation];
    if (!entry) return null;
    const key = args.join('|');
    return Object.prototype.hasOwnProperty.call(entry.facts, key) ? entry.facts[key] : null;
  }

  hasRelation(relation, args) {
    const entry = this._relations[relation];
    if (!entry) return false;
    const key = args.join('|');
    return Object.prototype.hasOwnProperty.call(entry.facts, key);
  }

  get relations() {
    return Object.keys(this._relations);
  }

  getRelationFacts() {
    const facts = [];
    for (const [relation, entry] of Object.entries(this._relations)) {
      for (const [key, value] of Object.entries(entry.facts)) {
        facts.push({ relation, args: key.split('|'), value });
      }
    }
    return facts;
  }

  _storeFactMeta(subject, predicate, meta) {
    const normalized = FactBase._normalizeMeta(meta);
    if (!normalized) return;
    this._factMeta[`${subject}:${predicate}`] = normalized;
  }

  _storeAttributeMeta(subject, attribute, meta) {
    const normalized = FactBase._normalizeMeta(meta);
    if (!normalized) return;
    this._attributeMeta[`${subject}:${attribute}`] = normalized;
  }

  _storeRelationMeta(relation, args, meta) {
    const normalized = FactBase._normalizeMeta(meta);
    if (!normalized) return;
    if (!this._relationMeta[relation]) {
      this._relationMeta[relation] = Object.create(null);
    }
    this._relationMeta[relation][args.join('|')] = normalized;
  }

  _shouldOverwrite(existingMeta, incomingMeta) {
    const policy = incomingMeta?.policy || this.updatePolicy;
    if (!policy || policy === 'overwrite') return true;
    if (policy === 'confidence') {
      const existing = existingMeta?.confidence ?? 0;
      const incoming = incomingMeta?.confidence ?? 0;
      return incoming >= existing;
    }
    return true;
  }

  /**
   * Retrieve the stored truth value for a (subject, predicate) pair.
   * Returns null when the fact has not been explicitly asserted.
   *
   * @param {string} subject
   * @param {string} predicate
   * @returns {0|1|null}
   */
  get(subject, predicate) {
    const key = `${subject}:${predicate}`;
    return Object.prototype.hasOwnProperty.call(this._facts, key)
      ? this._facts[key]
      : null;
  }

  /**
   * True if the fact has been explicitly asserted (either true or false).
   * @param {string} subject
   * @param {string} predicate
   * @returns {boolean}
   */
  has(subject, predicate) {
    return Object.prototype.hasOwnProperty.call(this._facts, `${subject}:${predicate}`);
  }

  // ── Encoding ──────────────────────────────────────────────────────────────

  /**
   * Encode a subject name as a normalised scalar in [0, 1].
   *
   * The encoding is stable as long as the subject order does not change.
   * Adding new subjects at the end does not invalidate existing encodings.
   *
   * @param {string} subject
   * @returns {number}  Value in [0, 1].
   * @throws {Error}    When the subject is not in the vocabulary.
   */
  encodeSubject(subject) {
    const idx = this.subjects.indexOf(subject);
    if (idx < 0) throw new Error(`FactBase: unknown subject '${subject}'`);
    const n = this.subjects.length;
    return n === 1 ? 0.5 : idx / (n - 1);
  }

  /**
   * Decode a normalised subject scalar back to the nearest subject label.
   * @param {number} encoded
   * @returns {string|null}
   */
  decodeSubject(encoded) {
    const n = this.subjects.length;
    if (n === 0) return null;
    if (n === 1) return this.subjects[0];
    const idx = Math.min(n - 1, Math.max(0, Math.round(encoded * (n - 1))));
    return this.subjects[idx];
  }

  // ── Lesson generation ─────────────────────────────────────────────────────

  /**
   * Generate one classification Lesson per predicate.
   *
   * Each lesson:
   *   • domain:  `facts.<predicate>`
   *   • inputSize: 1  (the normalised subject encoding)
   *   • output: [0] or [1]
   *   • mode: 'classification'
   *
   * Subjects that have no explicit value for a predicate are treated as
   * false (output 0).
   *
   * @returns {Lesson[]}
   */
  toLessons() {
    if (this.subjects.length === 0) {
      throw new Error('FactBase has no subjects — assert at least one fact before calling toLessons()');
    }

    // ── Binary predicate lessons (one output node per predicate) ─────────────
    const binaryLessons = this._predicates.map(predicate => {
      const trainingData = this.subjects.map(subject => ({
        input:  [this.encodeSubject(subject)],
        output: [this.get(subject, predicate) ?? 0],
      }));

      return new Lesson({
        name:        `Fact: ${predicate}`,
        domain:      `facts.${predicate}`,
        description: `Learn which subjects satisfy the predicate '${predicate}'.`,
        trainingData,
        inputSize:   1,
        mode:        'classification',
      });
    });

    // ── Categorical attribute lessons (one-hot multi-class) ──────────────────
    // Each attribute produces one lesson whose output is a one-hot vector over
    // the attribute's value vocabulary.  Subjects with no asserted value for the
    // attribute receive an all-zeros output vector.
    const attributeLessons = Object.entries(this._attributeFacts).length === 0
      ? []
      : Object.entries(this._attributeVocab).map(([attribute, vocab]) => {
        if (!vocab || vocab.length === 0) return null;
        const definition = this._attributeTypes[attribute] || { type: 'categorical' };
        if (definition.type !== 'categorical') return null;
        const trainingData = this.subjects.map(subject => {
          const value  = this.getValue(subject, attribute);
          const output = vocab.map(v => (v === value ? 1 : 0));
          return {
            input:  [this.encodeSubject(subject)],
            output,
          };
        });

      return new Lesson({
        name:        `Attribute: ${attribute}`,
        domain:      `facts.${attribute}`,
        description: `Learn the '${attribute}' attribute for each subject (values: ${vocab.join(', ')}).`,
        trainingData,
        inputSize:   1,
        outputSize:  vocab.length,
        mode:        'multiclass',
      });
      }).filter(Boolean);

    const numericAttributeLessons = Object.entries(this._attributeFacts).length === 0
      ? []
      : Object.entries(this._attributeTypes)
        .filter(([, def]) => def.type === 'numeric')
        .map(([attribute, def]) => {
          const trainingData = this.subjects.map(subject => ({
            input: [this.encodeSubject(subject)],
            output: [this.getValue(subject, attribute) ?? 0],
          }));
          return new Lesson({
            name:        `Attribute: ${attribute}`,
            domain:      `facts.${attribute}`,
            description: `Learn numeric attribute '${attribute}'.`,
            trainingData,
            inputSize:   1,
            outputSize:  1,
            mode:        'regression',
            normalise:   def.range ? { outputRange: def.range } : null,
          });
        });

    const relationLessons = Object.entries(this._relations).map(([relation, rel]) => {
      const entities = [...this.subjects];
      const combos = this._relationCombinations(entities, rel.arity, 250);
      const trainingData = combos.map(args => ({
        input: args.map(arg => this.encodeSubject(arg)),
        output: [this.getRelation(relation, args) ?? 0],
      }));
      return new Lesson({
        name:        `Relation: ${relation}`,
        domain:      `facts.${relation}`,
        description: `Learn relation '${relation}' with arity ${rel.arity}.`,
        trainingData,
        inputSize:   rel.arity,
        mode:        'classification',
      });
    });

    return [...binaryLessons, ...attributeLessons, ...numericAttributeLessons, ...relationLessons];
  }

  _relationCombinations(items, arity, limit = 500) {
    const results = [];
    const build = (prefix) => {
      if (results.length >= limit) return;
      if (prefix.length === arity) {
        results.push(prefix);
        return;
      }
      for (const item of items) {
        build([...prefix, item]);
        if (results.length >= limit) return;
      }
    };
    build([]);
    return results;
  }

  // ── Accessors ─────────────────────────────────────────────────────────────

  /** Ordered list of predicate names. */
  get predicates() {
    return [...this._predicates];
  }

  /** Ordered list of categorical attribute names. */
  get attributes() {
    return [...new Set([
      ...Object.keys(this._attributeVocab),
      ...Object.keys(this._attributeTypes),
    ])];
  }

  /** Total number of asserted facts (subject × predicate pairs). */
  get factCount() {
    return Object.keys(this._facts).length;
  }

  /** Total number of asserted attribute values (subject × attribute pairs). */
  get attributeCount() {
    return Object.keys(this._attributeFacts).length;
  }

  // ── Serialisation ─────────────────────────────────────────────────────────

  toJSON() {
    return {
      name:           this.name,
      updatePolicy:   this.updatePolicy,
      subjects:       [...this.subjects],
      predicates:     [...this._predicates],
      facts:          { ...this._facts },
      factMeta:       { ...this._factMeta },
      attributeVocab: Object.fromEntries(
        Object.entries(this._attributeVocab).map(([k, v]) => [k, [...v]])
      ),
      attributeFacts: { ...this._attributeFacts },
      attributeMeta:  { ...this._attributeMeta },
      attributeTypes: Object.fromEntries(Object.entries(this._attributeTypes).map(([k, v]) => [k, { ...v }])),
      relations: Object.fromEntries(Object.entries(this._relations).map(([k, v]) => [
        k,
        { arity: v.arity, facts: { ...v.facts } },
      ])),
      relationMeta: Object.fromEntries(Object.entries(this._relationMeta).map(([k, v]) => [
        k,
        { ...v },
      ])),
    };
  }

  static fromJSON(data) {
    const fb         = new FactBase(data.name || 'Facts', { updatePolicy: data.updatePolicy });
    fb.subjects      = [...(data.subjects   || [])];
    fb._predicates   = [...(data.predicates || [])];
    fb._facts        = Object.assign(Object.create(null), data.facts || {});
    fb._factMeta     = Object.assign(Object.create(null), data.factMeta || {});
    // Restore categorical attributes (present in new format; absent in old saves)
    if (data.attributeVocab) {
      for (const [attr, vocab] of Object.entries(data.attributeVocab)) {
        fb._attributeVocab[attr] = [...vocab];
      }
    }
    if (data.attributeFacts) {
      Object.assign(fb._attributeFacts, data.attributeFacts);
    }
    if (data.attributeMeta) {
      Object.assign(fb._attributeMeta, data.attributeMeta);
    }
    if (data.attributeTypes) {
      for (const [attr, def] of Object.entries(data.attributeTypes)) {
        fb._attributeTypes[attr] = { ...def };
      }
    }
    if (data.relations) {
      for (const [rel, info] of Object.entries(data.relations)) {
        fb._relations[rel] = { arity: info.arity, facts: { ...info.facts } };
      }
    }
    if (data.relationMeta) {
      for (const [rel, meta] of Object.entries(data.relationMeta)) {
        fb._relationMeta[rel] = { ...meta };
      }
    }
    return fb;
  }

  /**
   * Normalize metadata objects containing optional confidence/source fields.
   * Returns null when no valid metadata is present.
   *
   * @param {{ confidence?: number, source?: string }} meta
   * @returns {{ confidence?: number, source?: string }|null}
   */
  static _normalizeMeta(meta) {
    if (!meta || typeof meta !== 'object') return null;
    const normalized = {};
    if (meta.confidence != null) {
      const num = Number(meta.confidence);
      if (!Number.isNaN(num)) {
        normalized.confidence = Math.max(0, Math.min(1, num));
      }
    }
    if (meta.source) normalized.source = meta.source;
    return Object.keys(normalized).length > 0 ? normalized : null;
  }
}

module.exports = FactBase;
