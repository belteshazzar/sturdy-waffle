'use strict';

const { EventEmitter } = require('events');
const BrainRegion      = require('./BrainRegion');
const SequenceBrainRegion = require('./SequenceBrainRegion');
const Router           = require('../routing/Router');
const FactBase         = require('../knowledge/FactBase');
const MemorySystem     = require('../memory/MemorySystem');
const SharedEmbeddingBank = require('./SharedEmbeddingBank');
const MetaLearner      = require('./MetaLearner');
const SelfSupervisedLearner = require('../learning/SelfSupervisedLearner');
const WorldModel       = require('../world/WorldModel');
const ExpressionParser = require('../parsing/ExpressionParser');
const KnowledgeTextParser = require('../parsing/KnowledgeTextParser');
const { buildCapabilityMatrix, DEFAULT_TARGETS: DEFAULT_CAPABILITY_TARGETS } = require('../evaluation/CapabilityMatrix');

// Decomposition modules (loaded lazily to avoid circular deps at startup)
const {
  WorkingMemory,
  DecompositionGraph,
  DecompositionController,
  LearnedRouter,
  StringEncoder,
  computeExpertTrace,
  tokens: decompTokens,
} = require('../decomposition');

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
 *
 *  Decomposition events (emitted by solve() / trainDecomposition())
 *  decomposition:step      { step, op, args, result, memory }
 *  decomposition:complete  { tokens, solved, answer, steps, graph }
 *  decomposition:trained   { exampleCount, stageCount, replaySize }
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

    const defaults = {
      defaultTargetAccuracy: 0.99,
      maxMutations:          12,
      epochsPerRound:        500,
      maxEpochsTotal:        30000,
      regressionTolerance:   0.05,
      verbose:               false,
      sharedEmbedding: {
        enabled:        false,
        embeddingSize:  8,
        prototypeCount: 8,
        learningRate:   0.2,
      },
      memory: {
        episodicCapacity: 2000,
        semanticCapacity: 2000,
      },
      continualLearning: {
        replay:        { enabled: false, sampleCount: 32 },
        consolidation: { enabled: false, lambda: 0.015 },
      },
      metaLearning: {
        enabled:      false,
        maxSnapshots: 32,
      },
      selfSupervised: {
        enabled:     false,
        sampleCount: 64,
        epochs:      40,
      },
      worldModel: {
        enabled:        false,
        maxTransitions: 5000,
      },
      activeLearning: {
        uncertaintyWeight: 0.7,
        noveltyWeight:     0.3,
      },
      factUpdatePolicy: 'overwrite',
      inputLimits: {
        maxTokens:     512,
        maxDepth:      64,
        maxLines:      200,
        maxLineLength: 1000,
      },
      knowledgeConsolidation: {
        enabled:            true,
        onSyllabusComplete: true,
        onFactUpdate:       true,
        minSupport:         2,
        minConfidence:      0.8,
      },
      worldModelLoop: {
        enabled: true,
      },
      planning: {
        exploreWeight:     0.2,
        predictionWeight:  1,
        maxSteps:          32,
      },
      capabilityTargets: DEFAULT_CAPABILITY_TARGETS,
    };

    this.config = {
      ...defaults,
      ...config,
      sharedEmbedding: {
        ...defaults.sharedEmbedding,
        ...(config.sharedEmbedding || {}),
      },
      memory: {
        ...defaults.memory,
        ...(config.memory || {}),
      },
      continualLearning: {
        ...defaults.continualLearning,
        ...(config.continualLearning || {}),
        replay: {
          ...defaults.continualLearning.replay,
          ...((config.continualLearning && config.continualLearning.replay) || {}),
        },
        consolidation: {
          ...defaults.continualLearning.consolidation,
          ...((config.continualLearning && config.continualLearning.consolidation) || {}),
        },
      },
      metaLearning: {
        ...defaults.metaLearning,
        ...(config.metaLearning || {}),
      },
      selfSupervised: {
        ...defaults.selfSupervised,
        ...(config.selfSupervised || {}),
      },
      worldModel: {
        ...defaults.worldModel,
        ...(config.worldModel || {}),
      },
      activeLearning: {
        ...defaults.activeLearning,
        ...(config.activeLearning || {}),
      },
      inputLimits: {
        ...defaults.inputLimits,
        ...(config.inputLimits || {}),
      },
      knowledgeConsolidation: {
        ...defaults.knowledgeConsolidation,
        ...(config.knowledgeConsolidation || {}),
      },
      worldModelLoop: {
        ...defaults.worldModelLoop,
        ...(config.worldModelLoop || {}),
      },
      planning: {
        ...defaults.planning,
        ...(config.planning || {}),
      },
      capabilityTargets: {
        ...defaults.capabilityTargets,
        ...(config.capabilityTargets || {}),
      },
    };

    this.regions       = new Map();   // domain → BrainRegion
    this.router        = new Router();
    this.knowledgeTree = Object.create(null);
    this.createdAt     = new Date().toISOString();
    this.version       = '1.0.0';
    this.controller    = null;   // DecompositionController — null until initialised
    this.learnedRouter = null;   // LearnedRouter (Phase 3) — null until initialised
    this.stringEncoder = null;   // StringEncoder (Phase 4) — null until trained
    this.factBase      = null;   // FactBase — null until learnFacts() is called
    this.memory        = new MemorySystem(this.config.memory);
    this.sharedEmbedding = this.config.sharedEmbedding.enabled
      ? new SharedEmbeddingBank(this.config.sharedEmbedding)
      : null;
    this.metaLearner   = this.config.metaLearning.enabled
      ? new MetaLearner(this.config.metaLearning)
      : null;
    this.selfSupervisedLearner = this.config.selfSupervised.enabled
      ? new SelfSupervisedLearner(this.config.selfSupervised)
      : null;
    this.worldModel = this.config.worldModel.enabled
      ? new WorldModel(this.config.worldModel)
      : null;

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

    const RegionClass = lesson.sequence ? SequenceBrainRegion : BrainRegion;
    const region = new RegionClass({
      domain: lesson.domain,
      lesson,
      config: regionConfig,
      sharedEmbedding: this.sharedEmbedding,
      metaLearner: this.metaLearner,
    });

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

    if (this.config.continualLearning.replay.enabled) {
      const replay = this.memory.episodic.sample({
        domain,
        limit: this.config.continualLearning.replay.sampleCount,
      }).map(ep => ({ input: ep.input, output: ep.output }));
      region.setReplaySamples(replay);
    }
    region.setConsolidationConfig(this.config.continualLearning.consolidation);

    const result = region.train();

    this._updateKnowledgeTree(domain);
    this.memory.recordLesson(lesson);
    this._validateMemory('lesson');
    if (this.metaLearner && region.network && region.network.layers) {
      this.metaLearner.registerRegion(region);
    }

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
    this._maybeConsolidate('syllabus');
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

    if (expression.op === undefined || expression.op === null || expression.op === '') {
      return null;
    }
    if (typeof expression.op !== 'string') {
      return null;
    }
    const op = expression.op.toUpperCase();

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
   * Resolve a decomposition operator token to a trained domain.
   *
   * Uses the token's name (e.g. AND) and defers to _resolveExpressionDomain().
   * Returns null when the token is not an operator or when no matching domain
   * can be resolved from trained regions.
   *
   * Example:
   *   brain.resolveTokenDomain(decompTokens.TOKEN.AND) // → "boolean.AND"
   *
   * @param {number} opToken
   * @returns {string|null}
   */
  resolveTokenDomain(opToken) {
    if (!decompTokens.OPERATIONS.includes(opToken)) return null;
    const opName = decompTokens.TOKEN_NAMES[opToken];
    if (!opName) return null;
    return this._resolveExpressionDomain({ op: opName });
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
    * An additional terminal node form is supported for declarative fact lookups:
    *   • Fact query:     { fact: { subject: <string>, predicate: <string>, infer?: true } }
    *     Requires a FactBase to be loaded via learnFacts(). When `infer` is true
    *     the Brain consults semantic memory rules/analogies. When omitted or
    *     false, the Brain queries the trained fact region directly. Returns 0 or 1.
   *
   * @param {object} expression
   * @returns {number}
   */
  evaluate(expression) {
    // ── Fact lookup node ────────────────────────────────────────────────────
    if (expression.fact !== undefined) {
      if (!this.factBase) {
        throw new Error(
          'Expression contains a fact node but no FactBase is loaded. ' +
          'Call brain.learnFacts(factBase) first.'
        );
      }
      const { subject, predicate, infer } = expression.fact;
      if (infer) {
        const inferred = this.inferFact(subject, predicate);
        if (inferred.value === null) {
          throw new Error(`Unable to infer fact '${subject}:${predicate}'.`);
        }
        return inferred.value;
      }
      return this.queryFact(subject, predicate);
    }

    // ── Relation lookup node ────────────────────────────────────────────────
    if (expression.relation !== undefined) {
      if (!this.factBase) {
        throw new Error(
          'Expression contains a relation node but no FactBase is loaded. ' +
          'Call brain.learnFacts(factBase) first.'
        );
      }
      const { name, args, infer } = expression.relation;
      if (infer) {
        const inferred = this.inferRelation(name, args);
        if (inferred.value === null) {
          throw new Error(`Unable to infer relation '${name}(${args.join(',')})'.`);
        }
        return inferred.value;
      }
      return this.queryRelation(name, args);
    }

    if (expression.value !== undefined) {
      return expression.value;
    }

    if (!expression.op) {
      throw new Error('Expression node must have either a "value", "fact", or an "op" property');
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

  /**
   * Parse a human-readable expression string to a structured tree.
   */
  parseExpression(exprString, opts = {}) {
    const limits = this.config.inputLimits || {};
    return ExpressionParser.parseExpression(exprString, {
      maxTokens: opts.maxTokens ?? limits.maxTokens,
      maxDepth:  opts.maxDepth ?? limits.maxDepth,
    });
  }

  /**
   * Evaluate a raw expression string using the standard evaluate() pathway.
   */
  evaluateString(exprString, opts = {}) {
    return this.evaluate(this.parseExpression(exprString, opts));
  }

  /**
   * Convert an expression string to a decomposition-ready token stream.
   */
  tokenizeExpression(exprString, { resolveFacts = false, maxTokens, maxDepth } = {}) {
    const limits = this.config.inputLimits || {};
    const factResolver = resolveFacts ? this._createFactResolver() : null;
    return ExpressionParser.toTokenStream(exprString, {
      factResolver,
      maxTokens: maxTokens ?? limits.maxTokens,
      maxDepth:  maxDepth ?? limits.maxDepth,
    });
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

  // ── Declarative knowledge (FactBase) ──────────────────────────────────────

  /**
   * Teach the Brain a FactBase — a set of ground-truth (subject, predicate)
   * facts expressed as boolean values.
   *
   * For each predicate in the FactBase one classification Lesson is generated
   * (domain: "facts.<predicate>") and learned via the standard learn() pathway.
   * After this call the Brain can answer direct fact queries via queryFact() and
   * can evaluate expression trees that contain { fact: { subject, predicate } }
   * terminal nodes.
   *
   * The FactBase reference is stored on the Brain so that subject names can be
   * resolved to their encoded scalar values at inference time.
   *
   * @param {FactBase} factBase
   * @returns {Array<{ predicate: string, domain: string, trained: boolean, accuracy: number }>}
   */
  learnFacts(factBase) {
    if (factBase && this.config.factUpdatePolicy) {
      factBase.updatePolicy = this.config.factUpdatePolicy;
    }
    this.factBase = factBase;
    this.memory.recordFactBase(factBase);
    this._validateMemory('factBase');

    const results = factBase.toLessons().map(lesson => ({
      predicate: lesson.domain.split('.')[1],
      domain:    lesson.domain,
      ...this.learn(lesson),
    }));

    this.memory.semantic.induceRulesFromFactBase(factBase);
    this._maybeConsolidate('factUpdate');
    return results;
  }

  /**
   * Parse and ingest knowledge text, updating the FactBase and retraining only
   * the affected fact/attribute/relation domains.
   *
   * @param {string} text
   * @param {object} [opts]
   * @param {string} [opts.name='TextFacts']  FactBase name when creating new
   * @param {string} [opts.source='text']     Default source metadata
   * @param {boolean} [opts.retrain=true]     Whether to retrain affected domains
   * @returns {{ statements: object[], trainedDomains: string[], results: object[] }}
   */
  learnText(text, opts = {}) {
    const { name = 'TextFacts', source = 'text', retrain = true } = opts;
    const limits = this.config.inputLimits || {};
    const { statements } = KnowledgeTextParser.parse(text, {
      mode: 'statements',
      defaultSource: source,
      maxLines: limits.maxLines,
      maxLineLength: limits.maxLineLength,
    });
    if (statements.length === 0) {
      return { statements: [], trainedDomains: [], results: [] };
    }
    return this.learnStatements(statements, { name, source, retrain });
  }

  learnStatements(statements, opts = {}) {
    const { name = 'TextFacts', source = 'text', retrain = true } = opts;
    if (!Array.isArray(statements) || statements.length === 0) {
      return { statements: [], trainedDomains: [], results: [] };
    }

    if (!this.factBase) {
      this.factBase = new FactBase(name, { updatePolicy: this.config.factUpdatePolicy });
    }
    const affectedDomains = new Set();
    statements.forEach(statement => {
      const meta = { source, ...(statement.meta || {}) };
      if (statement.kind === 'fact') {
        this.factBase.assert(statement.subject, statement.predicate, statement.value, meta);
        affectedDomains.add(`facts.${statement.predicate}`);
      } else if (statement.kind === 'attribute') {
        if (statement.valueType === 'numeric') {
          this.factBase.defineAttribute(statement.attribute, { type: 'numeric' });
        }
        this.factBase.assertValue(statement.subject, statement.attribute, statement.value, meta);
        affectedDomains.add(`facts.${statement.attribute}`);
      } else if (statement.kind === 'relation') {
        this.factBase.assertRelation(statement.name, statement.args, statement.value, meta);
        affectedDomains.add(`facts.${statement.name}`);
      }
    });

    this.memory.recordTextStatements(statements, { defaultSource: source });
    this._validateMemory('textFacts');

    let results = [];
    if (retrain) {
      const lessons = this.factBase.toLessons().filter(lesson => affectedDomains.has(lesson.domain));
      results = lessons.map(lesson => ({
        domain: lesson.domain,
        ...this.learn(lesson),
      }));
    }
    this.memory.semantic.induceRulesFromFactBase(this.factBase);
    this._maybeConsolidate('factUpdate');
    return { statements, trainedDomains: [...affectedDomains], results };
  }

  /**
   * Query a single fact by name.
   *
   * Encodes the subject using the stored FactBase vocabulary and runs a binary
   * prediction through the "facts.<predicate>" BrainRegion.
   *
   * @param {string} subject    e.g. 'bird'
   * @param {string} predicate  e.g. 'canFly'
   * @returns {0|1}
   * @throws {Error}  When no FactBase is loaded or the predicate region is missing.
   */
  queryFact(subject, predicate) {
    if (!this.factBase) {
      throw new Error(
        'No FactBase is loaded. Call brain.learnFacts(factBase) first.'
      );
    }
    const encoded = this.factBase.encodeSubject(subject);
    const domain  = `facts.${predicate}`;
    return this.predictBinary([encoded], domain)[0];
  }

  /**
   * Query a categorical attribute and return the predicted string value.
   *
   * Encodes the subject using the stored FactBase vocabulary and runs the
   * "facts.<attribute>" multi-class BrainRegion, then decodes the argmax
   * index back to the corresponding label string.
   *
   * @param {string} subject    e.g. 'apple'
   * @param {string} attribute  e.g. 'color'
   * @returns {string}  The predicted value, e.g. 'red'
   * @throws {Error}  When no FactBase is loaded, the attribute is unknown,
   *                  or the corresponding region is missing.
   */
  queryAttribute(subject, attribute) {
    if (!this.factBase) {
      throw new Error(
        'No FactBase is loaded. Call brain.learnFacts(factBase) first.'
      );
    }

    const definition = this.factBase.getAttributeDefinition(attribute);
    const vocab = this.factBase.getAttributeVocabulary(attribute);
    if (!vocab && (!definition || definition.type !== 'numeric')) {
      throw new Error(
        `Attribute '${attribute}' is not defined in the loaded FactBase.`
      );
    }

    const encoded = this.factBase.encodeSubject(subject);
    const domain  = `facts.${attribute}`;
    const region  = this.router.route(domain);
    if (!region) {
      throw new Error(`No brain region found for domain '${domain}'.`);
    }

    if (definition && definition.type === 'numeric') {
      return region.predict([encoded])[0];
    }
    const argmax = region.predictArgmax([encoded]);
    return vocab[argmax];
  }

  /**
   * Query a multi-arity relation by name and argument list.
   *
   * @param {string} relation
   * @param {string[]} args
   * @returns {0|1}
   */
  queryRelation(relation, args) {
    if (!this.factBase) {
      throw new Error('No FactBase is loaded. Call brain.learnFacts(factBase) first.');
    }
    const value = this.factBase.getRelation(relation, args);
    if (value === null) {
      throw new Error(`No relation '${relation}' found for args [${args.join(', ')}].`);
    }
    return value;
  }

  /**
   * Parse text queries into structured query objects.
   *
   * @param {string} text
   * @returns {object[]}
   */
  parseTextQuery(text) {
    const limits = this.config.inputLimits || {};
    return KnowledgeTextParser.parse(text, {
      mode: 'queries',
      maxLines: limits.maxLines,
      maxLineLength: limits.maxLineLength,
    }).queries;
  }

  /**
   * Answer text queries using the loaded FactBase and expression evaluators.
   *
   * @param {string} text
   * @returns {Array<{ query: object, value: any, confidence?: number, source?: string }>}
   */
  answerText(text) {
    const queries = this.parseTextQuery(text);
    return this.answerQueries(queries);
  }

  answerQueries(queries) {
    return queries.map(query => {
      if (query.kind === 'fact') {
        if (query.infer) {
          const inferred = this.inferFact(query.subject, query.predicate);
          return { query, value: inferred.value, confidence: inferred.confidence, source: inferred.source };
        }
        return { query, value: this.queryFact(query.subject, query.predicate), confidence: 1, source: 'factBase' };
      }
      if (query.kind === 'attribute') {
        return { query, value: this.queryAttribute(query.subject, query.attribute), confidence: 1, source: 'factBase' };
      }
      if (query.kind === 'relation') {
        if (query.infer) {
          const inferred = this.inferRelation(query.name, query.args);
          return { query, value: inferred.value, confidence: inferred.confidence, source: inferred.source };
        }
        return { query, value: this.queryRelation(query.name, query.args), confidence: 1, source: 'factBase' };
      }
      if (query.kind === 'expression') {
        if (query.mode === 'solve') {
          const solved = this.solveString(query.expression);
          return { query, value: solved.answer, confidence: solved.solved ? 1 : 0, source: 'solveString' };
        }
        return { query, value: this.evaluateString(query.expression), confidence: 1, source: 'evaluateString' };
      }
      throw new Error(`Unknown query kind '${query.kind}'`);
    });
  }

  /**
   * Normalize heterogeneous input into a shared representation.
   *
   * Supported kinds:
   *  - expression: structured expression tree
   *  - tokens: decomposition token stream
   *  - knowledge: parsed knowledge statements
   *  - query: parsed knowledge/expression queries
   */
  normalizeInput(input, opts = {}) {
    if (input == null) {
      throw new Error('normalizeInput: input is required');
    }

    if (Array.isArray(input)) {
      return { kind: 'tokens', tokens: input };
    }

    if (typeof input === 'string') {
      const limits = this.config.inputLimits || {};
      let parsed = null;
      try {
        parsed = KnowledgeTextParser.parse(input, {
          mode: 'both',
          defaultSource: opts.source || 'text',
          maxLines: limits.maxLines,
          maxLineLength: limits.maxLineLength,
        });
      } catch (err) {
        if (opts.strictKnowledge) throw err;
        parsed = null;
      }
      if (parsed && (parsed.statements.length || parsed.queries.length)) {
        if ((opts.prefer === 'statements' || opts.mode === 'learn') && parsed.statements.length) {
          return { kind: 'knowledge', statements: parsed.statements, raw: input };
        }
        if (parsed.queries.length) {
          return { kind: 'query', queries: parsed.queries, raw: input };
        }
        return { kind: 'knowledge', statements: parsed.statements, raw: input };
      }
      return {
        kind: 'expression',
        expression: this.parseExpression(input, opts),
        raw: input,
      };
    }

    if (typeof input === 'object') {
      if (input.kind) return input;
      if (input.tokens) return { kind: 'tokens', tokens: input.tokens };
      if (input.expression) return { kind: 'expression', expression: input.expression };
      if (input.statements) return { kind: 'knowledge', statements: input.statements };
      if (input.queries) return { kind: 'query', queries: input.queries };
      if (input.op || input.value !== undefined || input.fact || input.relation) {
        return { kind: 'expression', expression: input };
      }
    }

    throw new Error('normalizeInput: unsupported input format');
  }

  /**
   * Unified entry point for text, expressions, tokens, and knowledge updates.
   */
  processInput(input, opts = {}) {
    const normalized = this.normalizeInput(input, opts);
    if (opts.execute === false) {
      return { normalized, result: null };
    }

    if (normalized.kind === 'expression') {
      const mode = opts.mode || 'evaluate';
      if (mode === 'solve') {
        const tokens = ExpressionParser.expressionToTokens(normalized.expression, {
          factResolver: this._createFactResolver(),
        });
        return { normalized, result: this.solve(tokens, opts.solveOptions || {}) };
      }
      return { normalized, result: this.evaluate(normalized.expression) };
    }

    if (normalized.kind === 'tokens') {
      return { normalized, result: this.solve(normalized.tokens, opts.solveOptions || {}) };
    }

    if (normalized.kind === 'knowledge') {
      if (opts.learn === false) {
        return { normalized, result: normalized.statements };
      }
      if (normalized.statements && normalized.statements.length) {
        return {
          normalized,
          result: this.learnStatements(normalized.statements, opts.learnOptions || {}),
        };
      }
      return {
        normalized,
        result: this.learnText(normalized.raw || '', opts.learnOptions || {}),
      };
    }

    if (normalized.kind === 'query') {
      return {
        normalized,
        result: normalized.queries ? this.answerQueries(normalized.queries) : this.answerText(normalized.raw || ''),
      };
    }

    throw new Error(`processInput: unsupported kind '${normalized.kind}'`);
  }

  // ── Reasoning & discovery ──────────────────────────────────────────────────

  /**
   * Infer a fact using semantic memory rules or analogies when explicit facts
   * are missing.
   */
  inferFact(subject, predicate) {
    if (!this.factBase) {
      throw new Error('No FactBase is loaded. Call brain.learnFacts(factBase) first.');
    }
    const explicit = this.factBase.get(subject, predicate);
    if (explicit !== null) {
      return { value: explicit, confidence: 1, source: 'factBase' };
    }
    const ruleInference = this.memory.semantic.inferFromRules({
      subject,
      predicate,
      factBase: this.factBase,
    });
    if (ruleInference) return ruleInference;
    const analogy = this.memory.semantic.inferByAnalogy({
      subject,
      predicate,
      factBase: this.factBase,
    });
    return analogy || { value: null, confidence: 0, source: 'unknown' };
  }

  /**
   * Infer a relation using semantic rules or analogies when explicit facts are missing.
   */
  inferRelation(relation, args) {
    if (!this.factBase) {
      throw new Error('No FactBase is loaded. Call brain.learnFacts(factBase) first.');
    }
    const explicit = this.factBase.getRelation(relation, args);
    if (explicit !== null) {
      return { value: explicit, confidence: 1, source: 'factBase' };
    }
    const ruleInference = this.memory.semantic.inferRelationFromRules({
      relation,
      args,
      factBase: this.factBase,
    });
    if (ruleInference) return ruleInference;
    return { value: null, confidence: 0, source: 'unknown' };
  }

  /**
   * Suggest informative training samples based on uncertainty + novelty.
   */
  suggestLessons(lessons, { limit = 5 } = {}) {
    const scored = [];
    lessons.forEach(lesson => {
      lesson.trainingData.forEach(sample => {
        const uncertainty = this._estimateUncertainty(lesson.domain, sample.input, lesson.mode);
        const novelty = this._estimateNovelty(lesson.domain, sample.input);
        const score = (
          this.config.activeLearning.uncertaintyWeight * uncertainty +
          this.config.activeLearning.noveltyWeight * novelty
        );
        scored.push({
          domain: lesson.domain,
          input: sample.input,
          output: sample.output,
          uncertainty,
          novelty,
          score,
        });
      });
    });
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, limit);
  }

  /**
   * Generate new lessons from episodic memory using uncertainty/novelty scores.
   */
  generateActiveLessons({ domains, limitPerDomain = 12, minScore = 0.4 } = {}) {
    const Lesson = require('../learning/Lesson');
    const episodes = this.memory.episodic.episodes;
    const domainSet = domains ? new Set(domains) : null;
    const grouped = new Map();
    episodes.forEach(ep => {
      if (domainSet && !domainSet.has(ep.domain)) return;
      if (!grouped.has(ep.domain)) grouped.set(ep.domain, []);
      grouped.get(ep.domain).push(ep);
    });

    const lessons = [];
    for (const [domain, eps] of grouped.entries()) {
      const scored = eps.map(ep => {
        const uncertainty = this._estimateUncertainty(domain, ep.input);
        const novelty = this._estimateNovelty(domain, ep.input);
        const score = this.config.activeLearning.uncertaintyWeight * uncertainty +
          this.config.activeLearning.noveltyWeight * novelty;
        return { ep, score };
      }).filter(entry => entry.score >= minScore);
      scored.sort((a, b) => b.score - a.score);
      const selected = scored.slice(0, limitPerDomain).map(entry => entry.ep);
      if (selected.length === 0) continue;
      const region = this.router.route(domain);
      const mode = region?.lesson?.mode || 'classification';
      lessons.push(new Lesson({
        name: `Auto Curriculum: ${domain}`,
        domain,
        description: 'Auto-generated lesson from episodic memory.',
        trainingData: selected.map(ep => ({ input: ep.input, output: ep.output })),
        inputSize: selected[0].input.length,
        outputSize: selected[0].output.length,
        mode,
        tags: ['auto', 'active-learning'],
        sequence: region?.lesson?.sequence || false,
      }));
    }
    return lessons;
  }

  _estimateUncertainty(domain, input, mode = 'classification') {
    const region = this.router.route(domain);
    if (!region) return 1;
    if (mode === 'multiclass') {
      const probs = region.predict(input);
      const max = Math.max(...probs);
      return 1 - max;
    }
    if (mode === 'regression') {
      return Math.min(1, 1 - region.accuracy);
    }
    const pred = region.predict(input)[0];
    return 1 - Math.abs(pred - 0.5) * 2;
  }

  _estimateNovelty(domain, input) {
    const nearest = this.memory.episodic.query({ domain, input, limit: 1 });
    if (!nearest.length) return 1;
    const flatten = value => (Array.isArray(value) ? value.flat(Infinity) : [value]);
    const flatInput = flatten(input);
    const flatNearest = flatten(nearest[0].input);
    const length = Math.max(1, flatInput.length);
    const distance = flatNearest.reduce((acc, v, i) => acc + Math.abs(v - (flatInput[i] || 0)), 0);
    return Math.min(1, distance / length);
  }

  _encodeAction(action, mem) {
    const maxToken = Math.max(1, decompTokens.VOCAB_SIZE - 1);
    const maxSlots = Math.max(1, (mem?.maxSlots || 16) - 1);
    const arity = decompTokens.ARITY[action.op] || 1;
    return [
      action.op / maxToken,
      action.start / maxSlots,
      arity / 3,
    ];
  }

  _predictionError(predicted, actual) {
    if (!predicted || !actual) return null;
    const n = Math.min(predicted.length, actual.length);
    if (n === 0) return null;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += Math.abs(predicted[i] - actual[i]);
    }
    return sum / n;
  }

  _createFactResolver() {
    return node => {
      if (node.subject) {
        return node.infer
          ? this.inferFact(node.subject, node.predicate).value
          : this.queryFact(node.subject, node.predicate);
      }
      if (node.name) {
        return node.infer
          ? this.inferRelation(node.name, node.args).value
          : this.queryRelation(node.name, node.args);
      }
      return null;
    };
  }

  // ── Decomposition controller ──────────────────────────────────────────────

  /**
   * Initialise a fresh DecompositionController if one does not already exist.
   *
   * The controller's working-memory dimensions must match the token vocabulary
   * used by WorkingMemory (maxSlots × VOCAB_SIZE input neurons, maxSlots output
   * neurons).  These defaults work for all Boolean-depth-≤-3 expressions.
   *
   * @param {object} [controllerConfig]  Forwarded to DecompositionController()
   * @returns {DecompositionController}
   */
  initController(controllerConfig = {}) {
    if (!this.controller) {
      this.controller = new DecompositionController(controllerConfig);
    }
    return this.controller;
  }

  /**
   * Initialise a LearnedRouter (Phase 3) for domain routing from embeddings.
   *
   * The router maps an operator's embedding to the BrainRegion domain that
   * should evaluate it.  It is trained during trainDecomposition() when the
   * controller is in embedding mode.
   *
   * Requires the controller to be in embedding mode (embeddingDim set).
   *
   * @param {object} [opts]
   * @param {string[]} [opts.domains]
   *   Override the domain list; when provided, auto-resolution from tokens is skipped.
   * @param {number[]} [opts.operatorTokens]  Operator tokens used for discovery (default: decompTokens.OPERATIONS)
   * @param {number}   [opts.confidenceThreshold=0.7]
   * @returns {LearnedRouter}
   */
  initLearnedRouter(opts = {}) {
    if (!this.controller || !this.controller.embeddingTable) {
      throw new Error(
        'initLearnedRouter() requires the controller to be in embedding mode. ' +
        'Call initController({ embeddingDim: <n> }) first.'
      );
    }
    const operatorTokens = opts.operatorTokens || decompTokens.OPERATIONS;
    const resolved = opts.domains
      ? opts.domains
      : operatorTokens.map(tok => this.resolveTokenDomain(tok)).filter(Boolean);
    const uniqueDomains = [...new Set(resolved)];
    if (uniqueDomains.length === 0) {
      throw new Error(
        'initLearnedRouter() could not resolve any operator domains. ' +
        'Provide opts.domains or train the relevant regions first.'
      );
    }
    this.learnedRouter = new LearnedRouter({
      embeddingDim:        this.controller.embeddingTable.dim,
      domains:             uniqueDomains,
      confidenceThreshold: opts.confidenceThreshold || 0.7,
    });
    return this.learnedRouter;
  }

  /**
   * Train the decomposition controller via imitation learning (supervised
   * warm-start) followed by optional replay consolidation.
   *
   * Phase 1 — Imitation:
   *   For every problem in every curriculum stage, the expert "leftmost-first"
   *   policy is run to generate a (stateVec, rawSlots, validStarts, targetStart)
   *   trace.  The controller is then trained on the aggregated trace via MSE.
   *   When the controller is in embedding mode, the EmbeddingTable is also
   *   updated jointly via backpropagation (Phase 1).
   *
    * Phase 2 — Replay:
    *   If the replay buffer already holds successful solve() episodes, a batch
    *   is sampled and used for additional supervised fine-tuning — reinforcing
    *   strategies that were empirically effective.
    *
    * Phase 3 — LearnedRouter (when learnedRouter is present):
    *   Trains the router to map operator embeddings to domain names using
    *   domains resolved from trained regions (or curriculum labels if supplied).
   *
   * Neuroscience analogue: imitation learning corresponds to procedural
   * memory encoding; replay consolidation corresponds to sleep-phase
   * hippocampal → cortical memory transfer.
   *
   * @param {DecompositionCurriculum} curriculum
   * @param {object} [opts]
   * @param {number} [opts.epochs=50]         Imitation training epochs
   * @param {number} [opts.replayBatch=32]    Replay sample size
   * @param {number} [opts.replayEpochs=5]    Replay fine-tuning epochs
   * @param {number} [opts.routerEpochs=30]   Learned-router training epochs
   * @returns {{ exampleCount: number, stageCount: number }}
   */
  trainDecomposition(curriculum, opts = {}) {
    const {
      epochs       = 50,
      replayBatch  = 32,
      replayEpochs = 5,
      routerEpochs = 30,
    } = opts;

    if (!this.controller) this.initController();

    const stages      = curriculum.getStages();
    const allExamples = [];

    for (const stage of stages) {
      for (const problem of stage.problems) {
        const { trace } = computeExpertTrace(
          problem.tokens,
          (opTok, args) => {
            // Prefer a trained specialist for evaluation fidelity; fall back to
            // the deterministic truth table (supplied by the curriculum).
            const domain = this.resolveTokenDomain(opTok);
            if (domain && this.router.hasRoute(domain)) {
              const region = this.router.route(domain);
              return (region.lesson && region.lesson.mode === 'regression')
                ? region.predict(args)[0]
                : region.predictBinary(args)[0];
            }
            return curriculum.evalOp
              ? curriculum.evalOp(opTok, args)
              : _fallbackEvalOp(opTok, args);
          }
        );
        for (const step of trace) {
          allExamples.push({
            stateVec:    step.stateVec,
            rawSlots:    step.rawSlots,
            rawValues:   step.rawValues,
            validStarts: step.validStarts,
            targetStart: step.chosenStart,
          });
        }
      }
    }

    this.controller.trainImitation(allExamples, epochs);

    // Phase 2 — replay consolidation
    if (this.controller.replayBuffer.size >= replayBatch) {
      this.controller.replayTrain(replayBatch, replayEpochs);
    }

    // Phase 3 — train LearnedRouter from embedding table
    if (this.learnedRouter && this.controller.embeddingTable) {
      const routerExamples = [];
      for (const stage of stages) {
        for (const problem of stage.problems) {
            for (const tok of problem.tokens) {
              const tokenId = (tok && typeof tok === 'object') ? tok.token : tok;
              const domain = this.resolveTokenDomain(tokenId);
              if (!domain) continue;
              const domainIdx = this.learnedRouter.domains.indexOf(domain);
              if (domainIdx < 0) continue;
              const opEmb = this.controller.embeddingTable.lookup(tokenId);
              routerExamples.push({ opEmbedding: opEmb, domainIndex: domainIdx });
            }
        }
      }
      if (routerExamples.length > 0) {
        this.learnedRouter.train(routerExamples, routerEpochs);
      }
    }

    const result = { exampleCount: allExamples.length, stageCount: stages.length };
    this.emit('decomposition:trained', {
      ...result,
      replaySize: this.controller.replayBuffer.size,
    });
    return result;
  }

  /**
   * Solve a flat-token problem using the learned decomposition controller.
   *
   * This is the **new evaluation pathway** that complements (and does not
   * replace) the legacy evaluate() method.  Rather than walking a pre-parsed
   * tree, the controller *learns* which sub-expression to reduce at each step.
   *
   * Pathway:
   *   1. Load tokens into WorkingMemory.
   *   2. At each step:
   *        a. Controller (PFC) selects which reduction to apply (basal-ganglia
   *           gating over valid candidates).
   *        b. The appropriate specialist BrainRegion executes the operation.
   *        c. The result is written back into WorkingMemory.
   *   3. Repeat until solved or budget exhausted.
   *   4. Store the episode trace in the replay buffer.
   *
   * The controller must be initialised (via initController() or
   * trainDecomposition()) before calling solve().  The relevant specialist
   * BrainRegions must also be trained.
   *
   * @param {number[]} tokens   Flat prefix-notation token sequence
   * @param {object}  [opts]
   * @param {boolean} [opts.forceExplore=false]  Force random action selection
   * @param {number}  [opts.maxSteps=32]         Iteration limit
   *
   * @returns {{
   *   answer: number | null,
   *   solved: boolean,
   *   steps:  number,
   *   graph:  object
   * }}
   */
  solve(tokens, opts = {}) {
    const { forceExplore = false, maxSteps = 32 } = opts;

    if (!this.controller) {
      throw new Error(
        'Brain has no decomposition controller. ' +
        'Call brain.initController() then brain.trainDecomposition() first.'
      );
    }

    const mem   = new WorkingMemory();
    const graph = new DecompositionGraph();
    mem.load(tokens);
    graph.init(tokens);

    const trace = [];
    let   steps = 0;

    while (!mem.isSolved() && steps < maxSteps) {
      const action = this.controller.selectAction(mem, forceExplore);

      if (!action) {
        this.emit('decomposition:stuck', { tokens, steps, memory: mem.slots.slice(0, mem.length) });
        break;
      }

      const stateVec = mem.toVector(this.controller.embeddingTable || null);

      // ── Phase 3: domain resolution ──────────────────────────────────────────
      // Try LearnedRouter first; fall back to operator-name resolution.
      let domain = null;
      const domainHint = action.domain || mem.domains[action.start] || null;
      if (domainHint) domain = domainHint;
      if (!domainHint && this.learnedRouter && this.controller.embeddingTable) {
        const opEmb  = this.controller.embeddingTable.lookup(action.op);
        const result = this.learnedRouter.route(opEmb);
        if (result.aboveThreshold) domain = result.domain;
      }
      if (!domain) domain = this.resolveTokenDomain(action.op);

      if (!domain) {
        const opName = decompTokens.TOKEN_NAMES[action.op] ?? String(action.op);
        throw new Error(
          `solve(): operator '${opName}' (token ${action.op}) could not be ` +
          'mapped to any trained specialist domain. Train the relevant syllabus ' +
          'and (optionally) a learned router first.'
        );
      }
      if (!this.router.hasRoute(domain)) {
        const opName = decompTokens.TOKEN_NAMES[action.op] ?? String(action.op);
        throw new Error(
          `solve(): operator '${opName}' (token ${action.op}) maps to ` +
          `domain '${domain}' with no trained specialist. ` +
          'Train the relevant syllabus and (optionally) a learned router first.'
        );
      }

      const region = this.router.route(domain);
      const result = (region.lesson && region.lesson.mode === 'regression')
        ? region.predict(action.args)[0]
        : region.predictBinary(action.args)[0];

      const actionVec = this._encodeAction(action, mem);
      const prediction = (this.worldModel && this.config.worldModelLoop.enabled && stateVec)
        ? this.worldModel.predict(stateVec, { action: actionVec })
        : null;

      trace.push({
        stateVec,
        rawSlots:    [...mem.slots],
        rawValues:   [...mem.values],
        chosenStart: action.start,
        op:          action.op,
        args:        [...action.args],
        result,
      });

      graph.addStep(action.start, action.op, action.args, result, domain);
      mem.reduce(action.start, decompTokens.ARITY[action.op], result);
      steps++;

      const nextStateVec = mem.toVector(this.controller.embeddingTable || null);
      const predictionError = prediction ? this._predictionError(prediction, nextStateVec) : null;
      if (this.worldModel && this.config.worldModelLoop.enabled && stateVec && nextStateVec) {
        this.worldModel.observe(stateVec, nextStateVec, { action: actionVec });
      }
      this.memory.recordDecompositionStep({
        state: stateVec,
        action: actionVec,
        nextState: nextStateVec,
        reward: mem.isSolved() ? 1 : 0,
        predictionError,
      });

      this.emit('decomposition:step', {
        step:   steps,
        op:     decompTokens.TOKEN_NAMES[action.op],
        args:   action.args,
        result,
        memory: mem.slots.slice(0, mem.length),
        values: mem.values.slice(0, mem.length),
        domain,
        predictionError,
      });
    }

    const solved = mem.isSolved();
    const answer = mem.answer();

    this.controller.storeTrace({ steps: trace, reward: solved ? 1 : -1, solved });

    const graphJSON = graph.toJSON();
    this.emit('decomposition:complete', { tokens, solved, answer, steps, graph: graphJSON });

    return { answer, solved, steps, graph: graphJSON };
  }

  // ── Phase 4: String encoder ───────────────────────────────────────────────

  /**
   * Train a StringEncoder (Phase 4) to convert human-readable expression
   * strings to token IDs using a character-level neural network.
   *
   * The training examples are extracted from the curriculum: for every problem
   * (which now carries a `string` field), each word in the string is paired
   * with the corresponding token ID from the `tokens` array.
   *
   * After training, `brain.solveString(expr)` accepts raw expression strings
   * without requiring callers to know the integer TOKEN vocabulary.
   *
   * @param {DecompositionCurriculum} curriculum
   * @param {object} [opts]
   * @param {number} [opts.epochs=40]  Training epochs for the encoder
   * @returns {StringEncoder}
   */
  trainStringEncoder(curriculum, opts = {}) {
    const { epochs = 40 } = opts;

    if (!this.stringEncoder) {
      this.stringEncoder = new StringEncoder();
    }

    const examples = [];
    for (const stage of curriculum.getStages()) {
      for (const problem of stage.problems) {
        if (!problem.string) continue;
        const words    = StringEncoder.splitWords(problem.string);
        const tokenIds = problem.tokens;
        const n        = Math.min(words.length, tokenIds.length);
        for (let i = 0; i < n; i++) {
          const token = tokenIds[i] && typeof tokenIds[i] === 'object'
            ? tokenIds[i].token
            : tokenIds[i];
          examples.push({ word: words[i], tokenId: token });
        }
      }
    }

    this.stringEncoder.train(examples, epochs);
    return this.stringEncoder;
  }

  /**
   * Solve a human-readable expression string.
   *
   * When a trained `stringEncoder` is present, the string is converted to
   * token IDs using the learned model.  Otherwise, a fast deterministic
   * parser (StringEncoder.toTokenIds) is used as a fallback.
   *
   * @param {string} exprString  e.g. "AND(OR(1,0), NOT(0))"
   * @param {object} [opts]      Forwarded to solve()
   * @returns {{ answer, solved, steps, graph }}
   */
  solveString(exprString, opts = {}) {
    let tokenIds;
    if (this.stringEncoder) {
      try {
        tokenIds = this.stringEncoder.encode(exprString);
      } catch (err) {
        tokenIds = null;
      }
    }
    if (!tokenIds) {
      const limits = this.config.inputLimits || {};
      tokenIds = ExpressionParser.toTokenStream(exprString, {
        factResolver: this._createFactResolver(),
        maxTokens: limits.maxTokens,
        maxDepth: limits.maxDepth,
      });
    }
    return this.solve(tokenIds, opts);
  }
 
  // ── Self-supervised learning & evaluation ──────────────────────────────────

  selfSupervise({ sampleCount, epochs } = {}) {
    const count = sampleCount || this.config.selfSupervised.sampleCount;
    const episodes = this.memory.episodic.sample({ limit: count });
    if (this.sharedEmbedding) {
      this.sharedEmbedding.updateWithSamples(episodes);
    }
    let selfSupervisedResult = null;
    if (this.selfSupervisedLearner) {
      selfSupervisedResult = this.selfSupervisedLearner.trainFromEpisodes(episodes, {
        epochs: epochs || this.config.selfSupervised.epochs,
      });
    }
    if (this.worldModel) {
      episodes.sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
      for (let i = 0; i < episodes.length - 1; i++) {
        this.worldModel.observe(episodes[i].input, episodes[i + 1].input);
      }
    }
    return {
      episodes: episodes.length,
      selfSupervised: selfSupervisedResult,
      worldModelUpdated: !!this.worldModel,
    };
  }

  /**
   * Pretrain shared embeddings and a sequence predictor from unlabeled token streams.
   */
  pretrainTokenModels({ tokenStreams = [], epochs = 40, window = 3 } = {}) {
    const Lesson = require('../learning/Lesson');
    const { VOCAB_SIZE } = decompTokens;
    const oneHot = (idx) => {
      const vec = new Array(VOCAB_SIZE).fill(0);
      if (idx >= 0 && idx < VOCAB_SIZE) vec[idx] = 1;
      return vec;
    };

    const autoEpisodes = [];
    const sequenceSamples = [];
    tokenStreams.forEach(stream => {
      for (let i = 0; i < stream.length; i++) {
        const tokenId = stream[i];
        autoEpisodes.push({ input: oneHot(tokenId) });
        if (i >= window) {
          const windowSlice = stream.slice(i - window, i);
          const input = windowSlice.map(tok => [tok / Math.max(1, VOCAB_SIZE - 1)]);
          const output = oneHot(tokenId);
          sequenceSamples.push({ input, output });
        }
      }
    });

    let autoResult = null;
    if (this.selfSupervisedLearner && autoEpisodes.length > 0) {
      autoResult = this.selfSupervisedLearner.trainFromEpisodes(autoEpisodes, { epochs });
    }
    if (this.sharedEmbedding && autoEpisodes.length > 0) {
      this.sharedEmbedding.updateWithSamples(autoEpisodes);
    }

    let sequenceResult = null;
    if (sequenceSamples.length > 0) {
      const lesson = new Lesson({
        name: 'Token Next Prediction',
        domain: 'sequence.NEXT_TOKEN',
        description: 'Predict the next token from a short history window.',
        trainingData: sequenceSamples,
        inputSize: 1,
        outputSize: VOCAB_SIZE,
        mode: 'multiclass',
        sequence: true,
        tags: ['self-supervised', 'sequence'],
      });
      sequenceResult = this.learn(lesson);
    }

    return {
      autoencoder: autoResult,
      sequencePredictor: sequenceResult,
      samples: sequenceSamples.length,
    };
  }

  selfLearn({ minSupport = 2, minConfidence = 0.8, promoteToFactBase = false } = {}) {
    const { concepts, rules } = this.memory.consolidate({
      factBase: this.factBase,
      minSupport,
      minConfidence,
    });
    const inferredFacts = [];
    if (this.factBase) {
      for (const rule of rules) {
        for (const subject of this.factBase.subjects) {
          if (this.factBase.get(subject, rule.then.predicate) !== null) continue;
          const inference = this.memory.semantic.inferFromRules({
            subject,
            predicate: rule.then.predicate,
            factBase: this.factBase,
          });
          if (inference && inference.value !== null) {
            const fact = this.memory.semantic.addFact({
              subject,
              predicate: rule.then.predicate,
              value: inference.value,
              confidence: inference.confidence,
              source: inference.source,
            });
            inferredFacts.push(fact);
            if (promoteToFactBase) {
              this.factBase.assert(subject, rule.then.predicate, inference.value === 1);
            }
          }
        }
      }
    }
    return { concepts, rules, facts: inferredFacts };
  }

  observeTransition(state, nextState, opts = {}) {
    if (!this.worldModel) return { observed: false };
    this.worldModel.observe(state, nextState, opts);
    return { observed: true };
  }

  predictNextState(state, opts = {}) {
    if (!this.worldModel) return null;
    return this.worldModel.predict(state, opts);
  }

  planTrajectory(state, { actions = [], context = null, steps = null } = {}) {
    if (!this.worldModel) return [];
    return this.worldModel.rollout(state, { actions, context, steps });
  }

  getCapabilityMatrix({ targets } = {}) {
    return buildCapabilityMatrix({ brain: this, targets: targets || this.config.capabilityTargets });
  }

  baselineReport({ syllabi = [], shots = 4 } = {}) {
    const EvaluationSuite = require('../evaluation/EvaluationSuite');
    const suite = new EvaluationSuite({
      brain: this,
      brainFactory: () => new Brain(this.config),
    });
    return suite.baseline({ syllabi, shots });
  }

  evaluateSuite({ syllabi, expressions, transferPairs, factBase, shots } = {}) {
    const EvaluationSuite = require('../evaluation/EvaluationSuite');
    const suite = new EvaluationSuite({
      brain: this,
      brainFactory: () => new Brain(this.config),
    });
    return suite.runAll({ syllabi, expressions, transferPairs, factBase, shots });
  }

  _maybeConsolidate(reason) {
    const policy = this.config.knowledgeConsolidation;
    if (!policy || !policy.enabled) return null;
    if (reason === 'syllabus' && !policy.onSyllabusComplete) return null;
    if (reason === 'factUpdate' && !policy.onFactUpdate) return null;
    const result = this.memory.consolidate({
      factBase: this.factBase,
      minSupport: policy.minSupport,
      minConfidence: policy.minConfidence,
    });
    this.emit('memory:consolidated', { reason, ...result });
    return result;
  }

  _validateMemory(context) {
    const report = this.memory.validate({ strict: false });
    if (!report.valid) {
      this.emit('memory:corrupt', { context, ...report });
    }
    return report;
  }

  planActObserve({ input, maxSteps, exploreWeight, predictionWeight } = {}) {
    const normalized = this.normalizeInput(input, { mode: 'solve' });
    const tokens = normalized.kind === 'tokens'
      ? normalized.tokens
      : ExpressionParser.expressionToTokens(normalized.expression, {
        factResolver: this._createFactResolver(),
      });

    if (!this.controller) {
      throw new Error(
        'planActObserve() requires a decomposition controller. ' +
        'Call brain.initController() then brain.trainDecomposition() first.'
      );
    }

    const mem = new WorkingMemory();
    mem.load(tokens);
    const trace = [];
    let steps = 0;
    const budget = maxSteps ?? this.config.planning.maxSteps;
    const explore = exploreWeight ?? this.config.planning.exploreWeight;
    const resolvedPredictionWeight = predictionWeight ?? this.config.planning.predictionWeight;

    while (!mem.isSolved() && steps < budget) {
      const candidates = mem.validReductions();
      if (candidates.length === 0) break;

      const stateVec = mem.toVector(this.controller.embeddingTable || null);
      let best = null;
      let bestScore = -Infinity;

      candidates.forEach(candidate => {
        const domainHint = candidate.domain || mem.domains[candidate.start] || null;
        const resolvedDomain = domainHint || this.resolveTokenDomain(candidate.op);
        const region = resolvedDomain ? this.router.route(resolvedDomain) : null;
        if (!region) return;
        const result = (region.lesson && region.lesson.mode === 'regression')
          ? region.predict(candidate.args)[0]
          : region.predictBinary(candidate.args)[0];

        const actionVec = this._encodeAction(candidate, mem);
        const simulated = mem.clone();
        simulated.reduce(candidate.start, decompTokens.ARITY[candidate.op], result);
        const nextVec = simulated.toVector(this.controller.embeddingTable || null);

        let predictionError = 1;
        if (this.worldModel && stateVec) {
          const predicted = this.worldModel.predict(stateVec, { action: actionVec });
          if (predicted) {
            predictionError = this._predictionError(predicted, nextVec);
          }
        }

        const solvedBonus = simulated.isSolved() ? 1 : 0;
        const novelty = this._estimateNovelty('decomposition.step', actionVec);
        const score = solvedBonus - resolvedPredictionWeight * predictionError + explore * novelty;
        if (score > bestScore) {
          bestScore = score;
          best = candidate;
        }
      });

      if (!best) {
        // Fallback: force exploration when no scored candidate is available.
        best = this.controller.selectAction(mem, true);
        if (!best) break;
      }

      const action = best;
      let domain = action.domain || mem.domains[action.start] || null;
      if (!domain) {
        domain = this.resolveTokenDomain(action.op);
      }
      const region = this.router.route(domain);
      if (!region) {
        throw new Error(`planActObserve(): no brain region found for domain '${domain}'`);
      }

      const result = (region.lesson && region.lesson.mode === 'regression')
        ? region.predict(action.args)[0]
        : region.predictBinary(action.args)[0];

      const actionVec = this._encodeAction(action, mem);
      const prediction = (this.worldModel && stateVec)
        ? this.worldModel.predict(stateVec, { action: actionVec })
        : null;

      mem.reduce(action.start, decompTokens.ARITY[action.op], result);
      const nextVec = mem.toVector(this.controller.embeddingTable || null);
      const predictionError = prediction ? this._predictionError(prediction, nextVec) : null;
      if (this.worldModel && this.config.worldModelLoop.enabled && stateVec && nextVec) {
        this.worldModel.observe(stateVec, nextVec, { action: actionVec });
      }
      this.memory.recordDecompositionStep({
        state: stateVec,
        action: actionVec,
        nextState: nextVec,
        reward: mem.isSolved() ? 1 : 0,
        predictionError,
      });

      trace.push({
        step: steps + 1,
        action,
        domain,
        result,
        predictionError,
      });
      steps++;
    }

    return {
      solved: mem.isSolved(),
      answer: mem.answer(),
      steps,
      trace,
    };
  }

  // ── Diagnostics ──────────────────────────────────────────────────────────

  getDiagnostics() {
    const routingConfidence = [];
    if (this.learnedRouter && this.controller && this.controller.embeddingTable) {
      for (const opTok of decompTokens.OPERATIONS) {
        const embedding = this.controller.embeddingTable.lookup(opTok);
        const routed = this.learnedRouter.route(embedding);
        routingConfidence.push({
          token: opTok,
          op: decompTokens.TOKEN_NAMES[opTok] || String(opTok),
          domain: routed.domain,
          confidence: routed.confidence,
          aboveThreshold: routed.aboveThreshold,
        });
      }
    }
    return {
      routingConfidence,
      embeddingDrift: this.sharedEmbedding ? this.sharedEmbedding.getDriftInfo() : null,
      consolidation: this.memory ? this.memory.lastConsolidation : null,
    };
  }

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
   *   regions: object,
   *   controller: object | null,
   *   learnedRouter: object | null,
   *   stringEncoder: object | null
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
      controller:    this.controller    ? this.controller.getInfo()    : null,
      learnedRouter: this.learnedRouter ? this.learnedRouter.toJSON()  : null,
      stringEncoder: this.stringEncoder ? this.stringEncoder.toJSON()  : null,
      sharedEmbedding: this.sharedEmbedding ? this.sharedEmbedding.getInfo() : null,
      metaLearner:      this.metaLearner ? this.metaLearner.getInfo() : null,
      selfSupervised:   this.selfSupervisedLearner ? this.selfSupervisedLearner.getInfo() : null,
      worldModel:       this.worldModel ? this.worldModel.getInfo() : null,
      memory:           this.memory.getInfo(),
      diagnostics:      this.getDiagnostics(),
      capabilityMatrix: this.getCapabilityMatrix(),
      capabilityGaps: Object.fromEntries(
        Object.entries(regions).map(([domain, info]) => [
          domain,
          Math.max(0, this.config.defaultTargetAccuracy - info.accuracy),
        ])
      ),
      factBase:      this.factBase ? {
        name:           this.factBase.name,
        subjectCount:   this.factBase.subjects.length,
        predicateCount: this.factBase.predicates.length,
        attributeCount: this.factBase.attributes.length,
        relationCount:  this.factBase.relations.length,
        subjects:       [...this.factBase.subjects],
        predicates:     this.factBase.predicates,
        attributes:     this.factBase.attributes,
        relations:      this.factBase.relations,
      } : null,
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
      controller:    this.controller    ? this.controller.toJSON()    : null,
      learnedRouter: this.learnedRouter ? this.learnedRouter.toJSON() : null,
      stringEncoder: this.stringEncoder ? this.stringEncoder.toJSON() : null,
      factBase:      this.factBase      ? this.factBase.toJSON()      : null,
      memory:        this.memory        ? this.memory.toJSON()        : null,
      sharedEmbedding: this.sharedEmbedding ? this.sharedEmbedding.toJSON() : null,
      metaLearner:     this.metaLearner ? this.metaLearner.toJSON() : null,
      selfSupervisedLearner: this.selfSupervisedLearner ? this.selfSupervisedLearner.toJSON() : null,
      worldModel:     this.worldModel ? this.worldModel.toJSON() : null,
    };
  }

  static fromJSON(data) {
    const brain         = new Brain(data.config || {});
    brain.version       = data.version    || '1.0.0';
    brain.createdAt     = data.createdAt  || new Date().toISOString();
    brain.knowledgeTree = data.knowledgeTree || {};
    if (data.sharedEmbedding) {
      brain.sharedEmbedding = SharedEmbeddingBank.fromJSON(data.sharedEmbedding);
    }

    for (const { domain, region: regionData } of (data.regions || [])) {
      const region = regionData.type === 'sequence'
        ? SequenceBrainRegion.fromJSON(regionData)
        : BrainRegion.fromJSON(regionData, { sharedEmbedding: brain.sharedEmbedding });
      brain.regions.set(domain, region);
      brain.router.register(domain, region);
    }

    if (data.controller) {
      brain.controller = DecompositionController.fromJSON(data.controller);
    }
    if (data.learnedRouter) {
      brain.learnedRouter = LearnedRouter.fromJSON(data.learnedRouter);
    }
    if (data.stringEncoder) {
      brain.stringEncoder = StringEncoder.fromJSON(data.stringEncoder);
    }
    if (data.factBase) {
      brain.factBase = FactBase.fromJSON(data.factBase);
    }
    if (data.memory) {
      brain.memory = MemorySystem.fromJSON(data.memory);
    }
    if (data.metaLearner) {
      brain.metaLearner = MetaLearner.fromJSON(data.metaLearner);
    }
    if (data.selfSupervisedLearner) {
      brain.selfSupervisedLearner = SelfSupervisedLearner.fromJSON(data.selfSupervisedLearner);
    }
    if (data.worldModel) {
      brain.worldModel = WorldModel.fromJSON(data.worldModel);
    }

    return brain;
  }
}

// ── Module-level helpers ──────────────────────────────────────────────────────

/**
 * Fallback Boolean truth-table evaluator used inside trainDecomposition() when
 * no trained specialist exists yet.  Kept outside the Brain class to avoid
 * polluting the public surface.
 * @private
 */
function _fallbackEvalOp(opTok, args) {
  const { TOKEN } = decompTokens;
  switch (opTok) {
    case TOKEN.AND:  return Math.min(args[0], args[1]);
    case TOKEN.OR:   return Math.max(args[0], args[1]);
    case TOKEN.NOT:  return 1 - args[0];
    case TOKEN.XOR:  return Math.abs(args[0] - args[1]);
    case TOKEN.NAND: return 1 - Math.min(args[0], args[1]);
    case TOKEN.NOR:  return 1 - Math.max(args[0], args[1]);
    case TOKEN.XNOR: return 1 - Math.abs(args[0] - args[1]);
    case TOKEN.IMP:  return Math.max(1 - args[0], args[1]);
    case TOKEN.ADD:  return args[0] + args[1];
    case TOKEN.SUB:  return args[0] - args[1];
    case TOKEN.MUL:  return args[0] * args[1];
    case TOKEN.DIV:  return args[1] === 0 ? 0 : args[0] / args[1];
    case TOKEN.SQRT: return Math.sqrt(Math.max(0, args[0]));
    default: throw new Error(`_fallbackEvalOp: unknown token ${opTok}`);
  }
}

module.exports = Brain;
