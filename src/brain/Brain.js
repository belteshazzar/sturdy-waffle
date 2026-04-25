'use strict';

const { EventEmitter } = require('events');
const BrainRegion      = require('./BrainRegion');
const Router           = require('../routing/Router');

/**
 * Brain is the top-level adaptive intelligence container.
 *
 * It manages a dynamic collection of BrainRegions (one per knowledge domain),
 * a hierarchical Router that dispatches inputs to the correct region, and the
 * lifecycle around learning new lessons and recalling existing knowledge.
 *
 * Key responsibilities
 * ───────────────────
 *  • Detect when a new domain is presented and spawn a correctly-sized region
 *  • Drive the region's training loop via learn() / learnSyllabus()
 *  • Route inference requests to the appropriate region via predict()
 *  • Evaluate nested expression trees by recursively resolving sub-expressions
 *  • Provide full introspection so that visualisation layers can render state
 *  • Persist and restore complete state as JSON
 *
 * Events emitted
 * ──────────────
 *  lesson:unknown       { domain }
 *  region:spawned       { domain, architecture }
 *  lesson:start         { domain, lessonName }
 *  training:progress    { domain, epoch, accuracy, loss, architecture }
 *  mutation             { domain, mutationCount, type, architecture }
 *  region:consolidated  { domain, plasticity, accuracy }
 *  region:trained       { domain, trained, accuracy, mutationCount, totalEpochs, architecture }
 *  lesson:complete      { domain, lessonName, trained, accuracy, mutationCount, totalEpochs }
 *  syllabus:start       { name, lessonCount }
 *  syllabus:complete    { name, results }
 */
class Brain extends EventEmitter {
  /**
   * @param {object}  [config]
   * @param {number}  [config.defaultTargetAccuracy=0.99]
   * @param {number}  [config.maxMutations=12]
   * @param {number}  [config.epochsPerRound=500]
   * @param {number}  [config.maxEpochsTotal=30000]
   * @param {boolean} [config.verbose=false]  Print progress to stdout
   */
  constructor(config = {}) {
    super();

    this.config = {
      defaultTargetAccuracy: 0.99,
      maxMutations:          12,
      epochsPerRound:        500,
      maxEpochsTotal:        30000,
      regressionTolerance:   0.05,
      verbose:               false,
      ...config,
    };

    this.regions       = new Map();   // domain → BrainRegion
    this.router        = new Router();
    this.knowledgeTree = Object.create(null);
    this.createdAt     = new Date().toISOString();
    this.version       = '1.0.0';

    if (this.config.verbose) {
      this._setupVerboseLogging();
    }
  }

  // ── Verbose logging (optional) ────────────────────────────────────────────

  _setupVerboseLogging() {
    this.on('lesson:unknown', ({ domain }) =>
      console.log(`[Brain] Unknown domain '${domain}' — spawning new region`)
    );
    this.on('region:spawned', ({ domain, architecture }) =>
      console.log(`[Brain] Region '${domain}' spawned, arch=[${architecture}]`)
    );
    this.on('mutation', ({ domain, mutationCount, type }) =>
      console.log(`[Brain] ${domain} mutation #${mutationCount}: ${type}`)
    );
    this.on('region:trained', ({ domain, accuracy, mutationCount, totalEpochs }) =>
      console.log(
        `[Brain] Region '${domain}' trained — ` +
        `acc=${(accuracy * 100).toFixed(1)}%, mutations=${mutationCount}, epochs=${totalEpochs}`
      )
    );
  }

  // ── Region management ─────────────────────────────────────────────────────

  /**
   * Create and register a new BrainRegion for the given lesson.
   * The initial network size is set relative to the lesson's input space.
   * @private
   */
  _spawnRegion(lesson) {
    const regionConfig = {
      targetAccuracy:      this.config.defaultTargetAccuracy,
      maxMutations:        this.config.maxMutations,
      epochsPerRound:      this.config.epochsPerRound,
      maxEpochsTotal:      this.config.maxEpochsTotal,
      regressionTolerance: this.config.regressionTolerance,
    };

    const region = new BrainRegion({ domain: lesson.domain, lesson, config: regionConfig });

    // Bubble up region events to the Brain so callers can subscribe once
    region.on('mutation',          data => this.emit('mutation',             data));
    region.on('training:progress', data => this.emit('training:progress',   data));
    region.on('training:complete', data => this.emit('region:trained',      data));
    region.on('consolidated',      data => this.emit('region:consolidated', data));

    this.regions.set(lesson.domain, region);
    this.router.register(lesson.domain, region);

    this.emit('region:spawned', {
      domain:       lesson.domain,
      architecture: [...region.network.architecture],
    });

    return region;
  }

  // ── Learning ──────────────────────────────────────────────────────────────

  /**
   * Present a single Lesson to the Brain.  If no region exists for the
   * lesson's domain a new one is spawned automatically.
   *
   * @param {Lesson} lesson
   * @returns {{ trained: boolean, accuracy: number, mutationCount: number, totalEpochs: number }}
   */
  learn(lesson) {
    const { domain } = lesson;

    if (!this.regions.has(domain)) {
      this.emit('lesson:unknown', { domain });
      this._spawnRegion(lesson);
    }

    const region = this.regions.get(domain);
    this.emit('lesson:start', { domain, lessonName: lesson.name });

    const result = region.train();

    this._updateKnowledgeTree(domain);

    this.emit('lesson:complete', {
      domain,
      lessonName: lesson.name,
      ...result,
    });

    return result;
  }

  /**
   * Work through every lesson in a Syllabus in order.
   *
   * @param {Syllabus} syllabus
   * @returns {Array<{ lesson: string, domain: string, trained: boolean, accuracy: number }>}
   */
  learnSyllabus(syllabus) {
    this.emit('syllabus:start', { name: syllabus.name, lessonCount: syllabus.lessons.length });

    const results = syllabus.lessons.map(lesson => ({
      lesson: lesson.name,
      domain: lesson.domain,
      ...this.learn(lesson),
    }));

    this.emit('syllabus:complete', { name: syllabus.name, results });
    return results;
  }

  // ── Inference ─────────────────────────────────────────────────────────────

  /**
   * Return the raw (continuous) output of the region responsible for `domain`.
   * @param {number[]} input
   * @param {string}   domain
   * @returns {number[]}
   */
  predict(input, domain) {
    const region = this.router.route(domain);
    if (!region) throw new Error(`No brain region found for domain '${domain}'`);
    return region.predict(input);
  }

  /**
   * Return thresholded (0/1) binary output.
   * @param {number[]} input
   * @param {string}   domain
   * @param {number}   [threshold=0.5]
   * @returns {number[]}
   */
  predictBinary(input, domain, threshold = 0.5) {
    const region = this.router.route(domain);
    if (!region) throw new Error(`No brain region found for domain '${domain}'`);
    return region.predictBinary(input, threshold);
  }

  /**
   * Resolve which domain should handle an expression node.
   * Priority:
   *   1) Explicit node domain
   *   2) Unique learned domain whose last segment matches the op (e.g. "*.ADD")
   *   3) Legacy boolean.<OP> fallback
   * Returns null when resolution is ambiguous or no region exists.
   * @private
   */
  _resolveExpressionDomain(expression) {
    if (expression.domain) return expression.domain;

    const op = String(expression.op || '').toUpperCase();
    if (!op) return null;

    const matchingDomains = Array
      .from(this.regions.keys())
      .filter(domain => domain.split('.').pop() === op);

    if (matchingDomains.length === 1) {
      return matchingDomains[0];
    }

    const legacyBooleanDomain = `boolean.${op}`;
    if (this.router.hasRoute(legacyBooleanDomain)) {
      return legacyBooleanDomain;
    }

    return null;
  }

  /**
   * Recursively evaluate a nested expression tree using the appropriate brain
   * region for each node.
   *
   * Expression nodes take one of two forms:
   *   • Literal value:   { value: <number> }
   *   • Operation:       { op: 'AND', inputs: [expr, expr] }
   *                   or { op: 'ADD', domain: 'math.ADD', inputs: [expr, expr] }
   *
   * Domain resolution (in priority order):
   *   1. Use `expression.domain` when explicitly provided on the node.
   *   2. Fall back to `"boolean.<OP>"` for backwards compatibility with existing
   *      boolean expression trees that omit a domain.
   *
   * Output type:
   *   • Regions trained in 'classification' mode return a thresholded 0/1 value.
   *   • Regions trained in 'regression' mode return the raw continuous prediction,
   *     allowing continuous intermediate values to propagate through the tree.
   *
   * @param {object} expression
   * @returns {number}
   */
  evaluate(expression) {
    if (expression.value !== undefined) {
      return expression.value;
    }

    if (!expression.op) {
      throw new Error('Expression node must have either a "value" or an "op" property');
    }

    const evaluatedInputs = expression.inputs.map(e => this.evaluate(e));

    const domain = this._resolveExpressionDomain(expression);
    if (!domain) {
      throw new Error(
        `Unable to resolve domain for operation '${expression.op}'. ` +
        'Provide expression.domain explicitly or train exactly one matching domain.'
      );
    }

    const region = this.router.route(domain);
    if (!region) throw new Error(`No brain region found for domain '${domain}'`);

    // Regression regions return continuous values; classification regions return 0/1.
    if (region.lesson && region.lesson.mode === 'regression') {
      return region.predict(evaluatedInputs)[0];
    }
    return region.predictBinary(evaluatedInputs)[0];
  }

  // ── Knowledge checks ──────────────────────────────────────────────────────

  /** True if a region exists and has been successfully trained. */
  knows(domain) {
    return this.regions.has(domain) && this.regions.get(domain).trained;
  }

  /** True if any region (trained or not) exists for this domain. */
  hasRegion(domain) {
    return this.regions.has(domain);
  }

  // ── Knowledge tree ────────────────────────────────────────────────────────

  _updateKnowledgeTree(domain) {
    const parts = domain.split('.');
    let node = this.knowledgeTree;
    for (const part of parts) {
      // Guard against prototype-polluting keys such as __proto__ or constructor
      if (part === '__proto__' || part === 'constructor' || part === 'prototype') continue;
      if (!Object.prototype.hasOwnProperty.call(node, part)) {
        node[part] = Object.create(null);
      }
      node = node[part];
    }
    node._domain = domain;
  }

  // ── Introspection ─────────────────────────────────────────────────────────

  /**
   * Return a plain-object snapshot of the Brain's current state.
   * Safe to serialise to JSON and suitable for consumption by a visualisation layer.
   *
   * @returns {{
   *   version: string,
   *   createdAt: string,
   *   regionCount: number,
   *   domains: string[],
   *   knowledgeTree: object,
   *   routerTree: object,
   *   regions: object
   * }}
   */
  introspect() {
    const regions = {};
    for (const [domain, region] of this.regions.entries()) {
      regions[domain] = region.getInfo();
    }

    return {
      version:       this.version,
      createdAt:     this.createdAt,
      regionCount:   this.regions.size,
      domains:       Array.from(this.regions.keys()),
      knowledgeTree: this.knowledgeTree,
      routerTree:    this.router.getTreeStructure(),
      regions,
    };
  }

  // ── Persistence ───────────────────────────────────────────────────────────

  /**
   * Save the complete Brain state to a JSON file.
   * @param {string} filepath
   * @returns {{ saved: boolean, filepath: string, sizeBytes: number, regionCount: number }}
   */
  save(filepath) {
    const StateManager = require('../persistence/StateManager');
    return StateManager.save(this, filepath);
  }

  /**
   * Load a Brain from a previously saved JSON file.
   * @param {string} filepath
   * @returns {Brain}
   */
  static load(filepath) {
    const StateManager = require('../persistence/StateManager');
    return StateManager.load(filepath);
  }

  // ── Serialisation ─────────────────────────────────────────────────────────

  toJSON() {
    return {
      version:       this.version,
      createdAt:     this.createdAt,
      config:        this.config,
      knowledgeTree: this.knowledgeTree,
      regions: Array.from(this.regions.entries()).map(([domain, region]) => ({
        domain,
        region: region.toJSON(),
      })),
    };
  }

  static fromJSON(data) {
    const brain        = new Brain(data.config || {});
    brain.version      = data.version || '1.0.0';
    brain.createdAt    = data.createdAt || new Date().toISOString();
    brain.knowledgeTree = data.knowledgeTree || {};

    for (const { domain, region: regionData } of (data.regions || [])) {
      const region = BrainRegion.fromJSON(regionData);
      brain.regions.set(domain, region);
      brain.router.register(domain, region);
    }

    return brain;
  }
}

module.exports = Brain;
