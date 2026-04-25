'use strict';

const { EventEmitter } = require('events');
const BrainRegion      = require('./BrainRegion');
const Router           = require('../routing/Router');

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
    this.controller    = null;   // DecompositionController — null until initialised
    this.learnedRouter = null;   // LearnedRouter (Phase 3) — null until initialised
    this.stringEncoder = null;   // StringEncoder (Phase 4) — null until trained

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
   * @param {string[]} [opts.domains]  Override the domain list (default: all TOKEN_DOMAIN values)
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
    const domains    = opts.domains || Object.values(decompTokens.TOKEN_DOMAIN);
    const uniqueDomains = [...new Set(domains)];
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
   *   Trains the router to map operator embeddings to domain names using the
   *   TOKEN_DOMAIN ground-truth labels from the curriculum.
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
            const domain = decompTokens.TOKEN_DOMAIN[opTok];
            if (domain && this.router.hasRoute(domain)) {
              return this.predictBinary(args, domain)[0];
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
            const domain = decompTokens.TOKEN_DOMAIN[tok];
            if (!domain) continue;
            const domainIdx = this.learnedRouter.domains.indexOf(domain);
            if (domainIdx < 0) continue;
            const opEmb = this.controller.embeddingTable.lookup(tok);
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

      // ── Phase 3: domain resolution ──────────────────────────────────────────
      // Try LearnedRouter first; fall back to hard-coded TOKEN_DOMAIN map.
      let domain = null;
      if (this.learnedRouter && this.controller.embeddingTable) {
        const opEmb  = this.controller.embeddingTable.lookup(action.op);
        const result = this.learnedRouter.route(opEmb);
        if (result.aboveThreshold) domain = result.domain;
      }
      if (!domain) domain = decompTokens.TOKEN_DOMAIN[action.op];

      if (!domain || !this.router.hasRoute(domain)) {
        throw new Error(
          `solve(): no trained specialist for domain '${domain}'. ` +
          'Train the Boolean logic syllabus first.'
        );
      }

      const result = this.predictBinary(action.args, domain)[0];

      trace.push({
        stateVec:    mem.toVector(this.controller.embeddingTable || null),
        rawSlots:    [...mem.slots],
        chosenStart: action.start,
        op:          action.op,
        args:        [...action.args],
        result,
      });

      graph.addStep(action.start, action.op, action.args, result);
      mem.reduce(action.start, decompTokens.ARITY[action.op], result);
      steps++;

      this.emit('decomposition:step', {
        step:   steps,
        op:     decompTokens.TOKEN_NAMES[action.op],
        args:   action.args,
        result,
        memory: mem.slots.slice(0, mem.length),
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
          examples.push({ word: words[i], tokenId: tokenIds[i] });
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
      tokenIds = this.stringEncoder.encode(exprString);
    } else {
      tokenIds = StringEncoder.toTokenIds(exprString);
    }
    return this.solve(tokenIds, opts);
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
    };
  }

  static fromJSON(data) {
    const brain         = new Brain(data.config || {});
    brain.version       = data.version    || '1.0.0';
    brain.createdAt     = data.createdAt  || new Date().toISOString();
    brain.knowledgeTree = data.knowledgeTree || {};

    for (const { domain, region: regionData } of (data.regions || [])) {
      const region = BrainRegion.fromJSON(regionData);
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
    case TOKEN.AND:  return args[0] & args[1];
    case TOKEN.OR:   return args[0] | args[1];
    case TOKEN.NOT:  return args[0] === 0 ? 1 : 0;
    case TOKEN.XOR:  return args[0] ^ args[1];
    case TOKEN.NAND: return (args[0] & args[1]) === 0 ? 1 : 0;
    case TOKEN.NOR:  return (args[0] | args[1]) === 0 ? 1 : 0;
    case TOKEN.XNOR: return (args[0] ^ args[1]) === 0 ? 1 : 0;
    default: throw new Error(`_fallbackEvalOp: unknown token ${opTok}`);
  }
}

module.exports = Brain;
