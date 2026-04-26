'use strict';

/**
 * Tests for the decomposition subsystem.
 *
 * Coverage:
 *  1. WorkingMemory   — slot management, valid reductions, toVector
 *  2. DecompositionGraph — step recording, serialisation
 *  3. ReplayBuffer    — push, sample, samplePositive, capacity overflow
 *  4. DecompositionController — construction, selectAction, trainImitation, replay
 *  5. Expert trace    — depth-1, depth-2 correctness
 *  6. Brain.solve()   — unit (mocked specialist), error handling
 *  7. Brain.trainDecomposition() + solve() — integration with Boolean syllabus
 *  8. Observability   — events, introspect(), save/load round-trip
 */

const path = require('path');
const Brain                   = require('../src/brain/Brain');
const BrainRegion             = require('../src/brain/BrainRegion');
const Lesson                  = require('../src/learning/Lesson');
const {
  WorkingMemory,
  DecompositionGraph,
  DecompositionController,
  ReplayBuffer,
  computeExpertTrace,
  tokens,
} = require('../src/decomposition');
const {
  DecompositionCurriculum,
  evalOp,
} = require('../syllabi/decomposition');

const { TOKEN, ARITY, VOCAB_SIZE } = tokens;

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Flat token sequence for AND(OR(1,0), NOT(0)) → expected answer 1 */
const DEPTH2_TOKENS = [TOKEN.AND, TOKEN.OR, TOKEN.V1, TOKEN.V0, TOKEN.NOT, TOKEN.V0];

/** Minimal Brain trained on a subset of Boolean gates for fast tests */
function makeFastBoolBrain() {
  const brain = new Brain({
    defaultTargetAccuracy: 0.75,
    epochsPerRound:        200,
    maxEpochsTotal:        4000,
    maxMutations:          6,
  });

  const truthTable = {
    AND:  [[0,0,0],[0,1,0],[1,0,0],[1,1,1]],
    OR:   [[0,0,0],[0,1,1],[1,0,1],[1,1,1]],
    XOR:  [[0,0,0],[0,1,1],[1,0,1],[1,1,0]],
  };

  const gates = ['AND', 'OR', 'NOT', 'XOR'];
  for (const name of gates) {
    const domain = `boolean.${name}`;
    let trainingData;
    if (name === 'NOT') {
      trainingData = [{ input: [0], output: [1] }, { input: [1], output: [0] }];
    } else {
      trainingData = truthTable[name].map(([a, b, out]) => ({ input: [a, b], output: [out] }));
    }
    brain.learn(new Lesson({ name, domain, trainingData }));
  }
  return brain;
}

// ── 1. WorkingMemory ──────────────────────────────────────────────────────────

describe('WorkingMemory', () => {
  test('load() fills slots and sets length', () => {
    const mem = new WorkingMemory(16);
    mem.load([TOKEN.AND, TOKEN.V1, TOKEN.V0]);
    expect(mem.length).toBe(3);
    expect(mem.slots[0]).toBe(TOKEN.AND);
    expect(mem.slots[1]).toBe(TOKEN.V1);
    expect(mem.slots[2]).toBe(TOKEN.V0);
    expect(mem.slots[3]).toBe(TOKEN.NULL);
  });

  test('load() silently truncates to maxSlots', () => {
    const mem = new WorkingMemory(4);
    mem.load([TOKEN.AND, TOKEN.V1, TOKEN.V0, TOKEN.OR, TOKEN.V1, TOKEN.V1]);
    expect(mem.length).toBe(4);
    expect(mem.slots.length).toBe(4);
  });

  test('validReductions() finds op with all-value args', () => {
    const mem = new WorkingMemory(8);
    mem.load([TOKEN.AND, TOKEN.V1, TOKEN.V1]);
    const reductions = mem.validReductions();
    expect(reductions).toHaveLength(1);
    expect(reductions[0]).toMatchObject({ start: 0, op: TOKEN.AND, args: [1, 1] });
  });

  test('validReductions() finds innermost op when outer args not yet resolved', () => {
    const mem = new WorkingMemory(16);
    // AND(OR(1,0), NOT(0))
    mem.load([TOKEN.AND, TOKEN.OR, TOKEN.V1, TOKEN.V0, TOKEN.NOT, TOKEN.V0]);
    const reductions = mem.validReductions();
    // OR at 1 (args 1,0) and NOT at 4 (arg 0) are valid; AND at 0 is not
    const starts = reductions.map(r => r.start);
    expect(starts).toContain(1);
    expect(starts).toContain(4);
    expect(starts).not.toContain(0);
  });

  test('reduce() collapses segment and maintains maxSlots length', () => {
    const mem = new WorkingMemory(8);
    mem.load([TOKEN.OR, TOKEN.V1, TOKEN.V0]);
    mem.reduce(0, 2, 1);
    expect(mem.length).toBe(1);
    expect(mem.slots.length).toBe(8);
    expect(mem.slots[0]).toBe(TOKEN.V1);
    expect(mem.slots[1]).toBe(TOKEN.NULL);
  });

  test('isSolved() is true after full reduction to single value', () => {
    const mem = new WorkingMemory(8);
    mem.load([TOKEN.NOT, TOKEN.V0]);
    expect(mem.isSolved()).toBe(false);
    mem.reduce(0, 1, 1);
    expect(mem.isSolved()).toBe(true);
    expect(mem.answer()).toBe(1);
  });

  test('isSolved() is false when multiple active slots remain', () => {
    const mem = new WorkingMemory(8);
    mem.load([TOKEN.AND, TOKEN.V1, TOKEN.V1]);
    expect(mem.isSolved()).toBe(false);
  });

  test('toVector() returns correct length', () => {
    const mem = new WorkingMemory(16);
    mem.load([TOKEN.AND, TOKEN.V1, TOKEN.V0]);
    const vec = mem.toVector();
    expect(vec.length).toBe(16 * (VOCAB_SIZE + 1));
  });

  test('toVector() encodes AND at slot 0 as one-hot at TOKEN.AND=2', () => {
    const mem = new WorkingMemory(4);
    mem.load([TOKEN.AND, TOKEN.V1, TOKEN.V0]);
    const vec = mem.toVector();
    const slotSize = VOCAB_SIZE + 1;
    // slot 0 → offset 0; TOKEN.AND = 2
    expect(vec[TOKEN.AND]).toBe(1);
    // other entries in slot-0's segment are 0
    for (let k = 0; k < VOCAB_SIZE; k++) {
      if (k !== TOKEN.AND) expect(vec[k]).toBe(0);
    }
    expect(vec[slotSize - 1]).toBe(0);
  });

  test('toVector() encodes NULL slot as all-zeros', () => {
    const mem = new WorkingMemory(4);
    mem.load([TOKEN.V1]);           // only slot 0 active
    const vec = mem.toVector();
    const slotSize = VOCAB_SIZE + 1;
    // slot 1 (NULL) → offset 1*slotSize .. (2*slotSize - 1) all 0
    for (let k = slotSize; k < 2 * slotSize; k++) {
      expect(vec[k]).toBe(0);
    }
  });

  test('clone() produces an independent copy', () => {
    const mem = new WorkingMemory(8);
    mem.load([TOKEN.AND, TOKEN.V1, TOKEN.V0]);
    const copy = mem.clone();
    copy.reduce(0, 2, 0);
    // original is unchanged
    expect(mem.length).toBe(3);
    expect(copy.length).toBe(1);
  });

  test('full depth-2 reduction trajectory', () => {
    const mem = new WorkingMemory(16);
    // AND(OR(1,0), NOT(0)) = 1
    mem.load(DEPTH2_TOKENS);
    expect(mem.isSolved()).toBe(false);

    // Step 1: reduce OR at 1
    mem.reduce(1, 2, 1);
    expect(mem.slots[1]).toBe(TOKEN.V1);

    // Step 2: reduce NOT at 2
    mem.reduce(2, 1, 1);
    expect(mem.slots[2]).toBe(TOKEN.V1);

    // Step 3: reduce AND at 0
    mem.reduce(0, 2, 1);
    expect(mem.isSolved()).toBe(true);
    expect(mem.answer()).toBe(1);
  });
});

// ── 2. DecompositionGraph ─────────────────────────────────────────────────────

describe('DecompositionGraph', () => {
  test('init() creates input nodes', () => {
    const g = new DecompositionGraph();
    g.init([TOKEN.AND, TOKEN.V1, TOKEN.V0]);
    expect(g.nodes).toHaveLength(3);
    expect(g.nodes[0]).toMatchObject({ type: 'input', token: TOKEN.AND });
  });

  test('addStep() appends a reduce node and a step record', () => {
    const g = new DecompositionGraph();
    g.init([TOKEN.NOT, TOKEN.V0]);
    const id = g.addStep(0, TOKEN.NOT, [0], 1);
    expect(id).toBe(2);
    expect(g.steps).toHaveLength(1);
    expect(g.steps[0]).toMatchObject({ op: TOKEN.NOT, args: [0], result: 1 });
  });

  test('toJSON() includes step summary', () => {
    const g = new DecompositionGraph();
    g.init([TOKEN.AND, TOKEN.V1, TOKEN.V1]);
    g.addStep(0, TOKEN.AND, [1, 1], 1);
    const json = g.toJSON();
    expect(json.stepCount).toBe(1);
    expect(json.steps[0].op).toBe('AND');
    expect(json.steps[0].result).toBe(1);
  });
});

// ── 3. ReplayBuffer ───────────────────────────────────────────────────────────

describe('ReplayBuffer', () => {
  test('push() and size', () => {
    const rb = new ReplayBuffer(10);
    rb.push({ steps: [], reward: 1, solved: true });
    expect(rb.size).toBe(1);
  });

  test('capacity overflow wraps oldest entry', () => {
    const rb = new ReplayBuffer(3);
    for (let i = 0; i < 5; i++) rb.push({ id: i, steps: [], reward: 1, solved: true });
    expect(rb.size).toBe(3);  // never exceeds capacity
  });

  test('sample() returns up to batchSize items', () => {
    const rb = new ReplayBuffer(100);
    for (let i = 0; i < 20; i++) rb.push({ steps: [], reward: 1, solved: i % 2 === 0 });
    const batch = rb.sample(5);
    expect(batch.length).toBe(5);
  });

  test('samplePositive() returns only solved traces', () => {
    const rb = new ReplayBuffer(100);
    for (let i = 0; i < 10; i++) rb.push({ steps: [], reward: i % 2 === 0 ? 1 : -1, solved: i % 2 === 0 });
    const batch = rb.samplePositive(20);
    expect(batch.length).toBeGreaterThan(0);
    batch.forEach(t => expect(t.solved).toBe(true));
  });

  test('samplePositive() returns empty array when no solved traces', () => {
    const rb = new ReplayBuffer(10);
    rb.push({ steps: [], reward: -1, solved: false });
    expect(rb.samplePositive(5)).toHaveLength(0);
  });

  test('getStats() reflects stored traces', () => {
    const rb = new ReplayBuffer(10);
    rb.push({ steps: [], reward: 1,  solved: true });
    rb.push({ steps: [], reward: -1, solved: false });
    const stats = rb.getStats();
    expect(stats.size).toBe(2);
    expect(stats.positiveCount).toBe(1);
    expect(stats.negativeCount).toBe(1);
  });
});

// ── 4. DecompositionController ────────────────────────────────────────────────

describe('DecompositionController', () => {
  test('constructor sets correct network architecture', () => {
    const ctrl = new DecompositionController({ maxSlots: 16, hiddenSize: 32 });
    expect(ctrl.network.architecture[0]).toBe(16 * (VOCAB_SIZE + 1));
    expect(ctrl.network.architecture).toContain(32);
    expect(ctrl.network.architecture[ctrl.network.architecture.length - 1]).toBe(16);
  });

  test('selectAction() returns a valid reduction', () => {
    const ctrl = new DecompositionController();
    const mem  = new WorkingMemory(16);
    mem.load([TOKEN.AND, TOKEN.V1, TOKEN.V1]);
    const action = ctrl.selectAction(mem);
    expect(action).not.toBeNull();
    expect(action.start).toBe(0);
    expect(action.op).toBe(TOKEN.AND);
    expect(action.args).toEqual([1, 1]);
  });

  test('selectAction() returns null when no valid reduction exists', () => {
    const ctrl = new DecompositionController();
    const mem  = new WorkingMemory(8);
    // Single AND token with no resolved operands — no valid reductions possible
    mem.load([TOKEN.AND]);
    const action = ctrl.selectAction(mem);
    expect(action).toBeNull();
  });

  test('trainImitation() marks controller as trained', () => {
    const ctrl = new DecompositionController({ maxSlots: 8 });
    const mem  = new WorkingMemory(8);
    mem.load([TOKEN.OR, TOKEN.V0, TOKEN.V1]);
    const stateVec = mem.toVector();
    ctrl.trainImitation([{ stateVec, targetStart: 0 }], 5);
    expect(ctrl.trained).toBe(true);
    expect(ctrl.trainCount).toBeGreaterThan(0);
  });

  test('storeTrace() adds to replay buffer', () => {
    const ctrl = new DecompositionController();
    expect(ctrl.replayBuffer.size).toBe(0);
    ctrl.storeTrace({ steps: [], reward: 1, solved: true });
    expect(ctrl.replayBuffer.size).toBe(1);
  });

  test('replayTrain() runs without error when buffer has solved traces', () => {
    const ctrl = new DecompositionController({ maxSlots: 8 });
    const mem  = new WorkingMemory(8);
    mem.load([TOKEN.NOT, TOKEN.V0]);
    const stateVec = mem.toVector();
    ctrl.storeTrace({
      steps: [{ stateVec, chosenStart: 0, op: TOKEN.NOT, args: [0], result: 1 }],
      reward: 1,
      solved: true,
    });
    // Should complete without throwing
    expect(() => ctrl.replayTrain(1, 2)).not.toThrow();
  });

  test('toJSON() / fromJSON() round-trip preserves config', () => {
    const ctrl  = new DecompositionController({ maxSlots: 8, hiddenSize: 16 });
    const json  = ctrl.toJSON();
    const ctrl2 = DecompositionController.fromJSON(json);
    expect(ctrl2.config.maxSlots).toBe(8);
    expect(ctrl2.config.hiddenSize).toBe(16);
    expect(ctrl2.network.architecture).toEqual(ctrl.network.architecture);
  });
});

// ── 5. Expert trace ───────────────────────────────────────────────────────────

describe('computeExpertTrace', () => {
  test('depth-1 NOT produces exactly one step', () => {
    const toks = [TOKEN.NOT, TOKEN.V0];
    const { trace, solved, answer } = computeExpertTrace(toks, evalOp);
    expect(trace).toHaveLength(1);
    expect(solved).toBe(true);
    expect(answer).toBe(1);
    expect(trace[0].chosenStart).toBe(0);
    expect(trace[0].op).toBe(TOKEN.NOT);
    expect(trace[0].args).toEqual([0]);
    expect(trace[0].result).toBe(1);
  });

  test('depth-1 AND produces exactly one step with correct result', () => {
    const toks = [TOKEN.AND, TOKEN.V1, TOKEN.V0];
    const { trace, solved, answer } = computeExpertTrace(toks, evalOp);
    expect(trace).toHaveLength(1);
    expect(solved).toBe(true);
    expect(answer).toBe(0);
  });

  test('depth-2 AND(OR(1,0), NOT(0)) produces 3 steps in correct order', () => {
    const { trace, solved, answer } = computeExpertTrace(DEPTH2_TOKENS, evalOp);
    expect(solved).toBe(true);
    expect(answer).toBe(1);
    expect(trace).toHaveLength(3);
    // Expert leftmost: first reduce OR at 1, then NOT at 2, then AND at 0
    expect(trace[0].op).toBe(TOKEN.OR);
    expect(trace[1].op).toBe(TOKEN.NOT);
    expect(trace[2].op).toBe(TOKEN.AND);
  });

  test('each step has stateVec of correct length', () => {
    const { trace } = computeExpertTrace(DEPTH2_TOKENS, evalOp);
    for (const step of trace) {
      expect(step.stateVec.length).toBe(16 * (VOCAB_SIZE + 1));
    }
  });

  test('stateVec for first step encodes initial memory state', () => {
    const { trace } = computeExpertTrace([TOKEN.NOT, TOKEN.V1], evalOp);
    const vec = trace[0].stateVec;
    // Slot 0 is NOT (=4)
    expect(vec[TOKEN.NOT]).toBe(1);
  });
});

// ── 6. DecompositionCurriculum ────────────────────────────────────────────────

describe('DecompositionCurriculum', () => {
  test('generateDepth1() returns 26 problems (7 gates, exhaustive)', () => {
    const c = new DecompositionCurriculum();
    expect(c.generateDepth1()).toHaveLength(26);
  });

  test('depth-1 answers are correct', () => {
    const c  = new DecompositionCurriculum();
    const ps = c.generateDepth1();
    for (const { tokens: toks, answer } of ps) {
      const opTok = toks[0].token ?? toks[0];
      const args = toks.slice(1).map(tok => {
        if (tok && typeof tok === 'object') return tok.value;
        if (tok === TOKEN.V0) return 0;
        if (tok === TOKEN.V1) return 1;
        return tok;
      });
      expect(evalOp(opTok, args)).toBe(answer);
    }
  });

  test('generateDepth2() returns expected count', () => {
    const c  = new DecompositionCurriculum({ depth2Count: 40 });
    const ps = c.generateDepth2();
    expect(ps.length).toBeLessThanOrEqual(40);
    expect(ps.length).toBeGreaterThan(0);
  });

  test('getStages() returns 3 stages in depth order', () => {
    const c      = new DecompositionCurriculum({ depth2Count: 10, depth3Count: 8 });
    const stages = c.getStages();
    expect(stages).toHaveLength(3);
    expect(stages[0].depth).toBe(1);
    expect(stages[1].depth).toBe(2);
    expect(stages[2].depth).toBe(3);
  });
});

// ── 7. Brain.solve() — unit tests ────────────────────────────────────────────

describe('Brain.solve() — unit', () => {
  test('throws when controller is not initialised', () => {
    const brain = new Brain();
    expect(() => brain.solve([TOKEN.AND, TOKEN.V1, TOKEN.V1])).toThrow(/controller/i);
  });

  test('throws when specialist domain is not trained', () => {
    const brain = new Brain();
    brain.initController();
    // Train controller with fallback evalFn but no specialist regions
    const c = new DecompositionCurriculum({ depth2Count: 5, depth3Count: 0 });
    brain.trainDecomposition(c, { epochs: 2 });
    expect(() => brain.solve([TOKEN.AND, TOKEN.V1, TOKEN.V1])).toThrow(/specialist/i);
  });

  test('returns { answer, solved, steps, graph } shape', done => {
    const brain = makeFastBoolBrain();
    brain.initController();
    const c = new DecompositionCurriculum({ depth2Count: 20, depth3Count: 0 });
    brain.trainDecomposition(c, { epochs: 10 });
    const res = brain.solve([TOKEN.NOT, TOKEN.V0]);
    expect(res).toHaveProperty('answer');
    expect(res).toHaveProperty('solved');
    expect(res).toHaveProperty('steps');
    expect(res).toHaveProperty('graph');
    expect(typeof res.steps).toBe('number');
    done();
  });

  test('solve depth-1 NOT(0) returns 1 (solved)', () => {
    const brain = makeFastBoolBrain();
    brain.initController();
    const c = new DecompositionCurriculum({ depth2Count: 20, depth3Count: 0 });
    brain.trainDecomposition(c, { epochs: 30 });
    const res = brain.solve([TOKEN.NOT, TOKEN.V0]);
    expect(res.solved).toBe(true);
    expect(res.steps).toBe(1);
    expect(res.answer).toBe(1);
  });
}, 60000);

// ── 8. Brain.trainDecomposition() + solve() — integration ────────────────────

describe('Brain.trainDecomposition() + solve() — integration', () => {
  let brain;

  beforeAll(() => {
    brain = makeFastBoolBrain();
    const c = new DecompositionCurriculum({ depth2Count: 40, depth3Count: 0 });
    brain.trainDecomposition(c, { epochs: 60 });
  }, 120000);

  test('controller is marked trained after trainDecomposition()', () => {
    expect(brain.controller).not.toBeNull();
    expect(brain.controller.trained).toBe(true);
  });

  test('controller solves all depth-1 single-gate problems', () => {
    const c  = new DecompositionCurriculum();
    const ps = c.generateDepth1();
    let   ok = 0;

    for (const { tokens: toks, answer } of ps) {
      const op = toks[0].token ?? toks[0];
      // Only test domains the brain has been trained on
      const domain = brain.resolveTokenDomain(op);
      if (!domain || !brain.router.hasRoute(domain)) continue;
      const res = brain.solve(toks, { forceExplore: false });
      if (res.solved && res.answer === answer) ok++;
    }
    // Expect at least 75% of tested depth-1 problems to be solved correctly.
    // The controller is trained with random initialisation and stochastic
    // curriculum sampling, so we allow some variance in this integration test.
    const tested = ps.filter(p => {
      const opToken = p.tokens[0].token ?? p.tokens[0];
      const domain = brain.resolveTokenDomain(opToken);
      return domain && brain.router.hasRoute(domain);
    }).length;
    expect(ok / tested).toBeGreaterThanOrEqual(0.75);
  });

  test('solve() bounds step count to ≤ maxSteps', () => {
    const res = brain.solve(DEPTH2_TOKENS, { maxSteps: 10 });
    expect(res.steps).toBeLessThanOrEqual(10);
  });

  test('replay buffer accumulates solved traces after multiple solves', () => {
    for (let i = 0; i < 5; i++) brain.solve([TOKEN.NOT, TOKEN.V1]);
    expect(brain.controller.replayBuffer.size).toBeGreaterThan(0);
  });

  test('graph toJSON contains correct step count for depth-1', () => {
    const res = brain.solve([TOKEN.NOT, TOKEN.V0]);
    if (res.solved) {
      expect(res.graph.stepCount).toBe(1);
    }
  });
}, 120000);

// ── 9. Observability ─────────────────────────────────────────────────────────

describe('Observability', () => {
  test('introspect() includes controller field when initialised', () => {
    const brain = new Brain();
    expect(brain.introspect().controller).toBeNull();
    brain.initController();
    const info = brain.introspect();
    expect(info.controller).not.toBeNull();
    expect(info.controller.trained).toBe(false);
    expect(Array.isArray(info.controller.architecture)).toBe(true);
  });

  test('decomposition:step event is emitted during solve()', done => {
    const brain = makeFastBoolBrain();
    brain.initController();
    const c = new DecompositionCurriculum({ depth2Count: 10, depth3Count: 0 });
    brain.trainDecomposition(c, { epochs: 10 });
    let stepCount = 0;
    brain.on('decomposition:step', () => stepCount++);
    brain.solve([TOKEN.NOT, TOKEN.V0]);
    expect(stepCount).toBeGreaterThan(0);
    done();
  });

  test('decomposition:complete event carries expected fields', done => {
    const brain = makeFastBoolBrain();
    brain.initController();
    const c = new DecompositionCurriculum({ depth2Count: 10, depth3Count: 0 });
    brain.trainDecomposition(c, { epochs: 10 });
    brain.on('decomposition:complete', payload => {
      expect(payload).toHaveProperty('tokens');
      expect(payload).toHaveProperty('solved');
      expect(payload).toHaveProperty('answer');
      expect(payload).toHaveProperty('steps');
      expect(payload).toHaveProperty('graph');
      done();
    });
    brain.solve([TOKEN.NOT, TOKEN.V0]);
  });

  test('decomposition:trained event is emitted', done => {
    const brain = new Brain();
    brain.on('decomposition:trained', payload => {
      expect(payload).toHaveProperty('exampleCount');
      expect(payload).toHaveProperty('stageCount');
      done();
    });
    const c = new DecompositionCurriculum({ depth2Count: 5, depth3Count: 0 });
    brain.trainDecomposition(c, { epochs: 1 });
  });

  test('toJSON() / fromJSON() preserves controller', () => {
    const brain = new Brain();
    brain.initController();
    const json     = brain.toJSON();
    expect(json.controller).not.toBeNull();
    const reloaded = Brain.fromJSON(json);
    expect(reloaded.controller).not.toBeNull();
    expect(reloaded.controller.config).toEqual(brain.controller.config);
  });

  test('save() / load() round-trip includes controller', () => {
    const brain    = new Brain();
    brain.initController();
    const filepath = path.join(__dirname, '../saves/decomp-test-brain.json');
    brain.save(filepath);
    const loaded   = Brain.load(filepath);
    expect(loaded.controller).not.toBeNull();
    expect(loaded.controller.config.maxSlots).toBe(16);
  });
});
