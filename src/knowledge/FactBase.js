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
 * Typical usage
 * ─────────────
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
 */
class FactBase {
  /**
   * @param {string} [name='Facts']  Human-readable name for this knowledge base.
   */
  constructor(name = 'Facts') {
    this.name       = name;
    this.subjects   = [];          // ordered subject vocabulary
    this._facts     = Object.create(null);   // `${subject}:${predicate}` → 0|1
    this._predicates = [];         // ordered predicate vocabulary (insertion order)
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
  assert(subject, predicate, value = true) {
    if (typeof subject   !== 'string' || subject   === '') throw new Error('subject must be a non-empty string');
    if (typeof predicate !== 'string' || predicate === '') throw new Error('predicate must be a non-empty string');

    if (!this.subjects.includes(subject)) {
      this.subjects.push(subject);
    }
    if (!this._predicates.includes(predicate)) {
      this._predicates.push(predicate);
    }

    this._facts[`${subject}:${predicate}`] = value ? 1 : 0;
    return this;
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

    return this._predicates.map(predicate => {
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
  }

  // ── Accessors ─────────────────────────────────────────────────────────────

  /** Ordered list of predicate names. */
  get predicates() {
    return [...this._predicates];
  }

  /** Total number of asserted facts (subject × predicate pairs). */
  get factCount() {
    return Object.keys(this._facts).length;
  }

  // ── Serialisation ─────────────────────────────────────────────────────────

  toJSON() {
    return {
      name:       this.name,
      subjects:   [...this.subjects],
      predicates: [...this._predicates],
      facts:      { ...this._facts },
    };
  }

  static fromJSON(data) {
    const fb         = new FactBase(data.name || 'Facts');
    fb.subjects      = [...(data.subjects   || [])];
    fb._predicates   = [...(data.predicates || [])];
    fb._facts        = Object.assign(Object.create(null), data.facts || {});
    return fb;
  }
}

module.exports = FactBase;
