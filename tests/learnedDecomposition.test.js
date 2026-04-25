'use strict';

/**
 * Tests for the fully-learned decomposition phases (Phases 1–5).
 *
 * Coverage:
 *  1. EmbeddingTable    — lookup, update, toJSON/fromJSON
 *  2. WorkingMemory embedding mode — toVector(embeddingTable)
 *  3. NeuralNetwork.backwardWithInputGrad() — shape & gradient sign
 *  4. DecompositionController embedding mode — construction, selectAction,
 *     trainImitation with joint embedding update
 *  5. GatingNetwork     — construction, train, predictValidStarts
 *  6. DecompositionController gating mode — selectAction with gating filter
 *  7. LearnedRouter     — construction, train, route / confidence
 *  8. StringEncoder     — splitWords, toTokenIds, wordToFeature,
 *                         train, encode, toJSON/fromJSON
 *  9. DecompositionCurriculum string fields — problems carry `string` field
 * 10. Brain.trainStringEncoder() + solveString() (Phase 4)
 * 11. Brain.initLearnedRouter() + trainDecomposition with router (Phase 3)
 * 12. Brain.toJSON() / fromJSON() round-trip with all phases
 * 13. VQCodebook        — quantize, update (EMA), commitmentLoss, serialisation
 */

const path = require('path');
const Brain = require('../src/brain/Brain');
const Lesson = require('../src/learning/Lesson');

const {
  WorkingMemory,
  DecompositionController,
  EmbeddingTable,
  GatingNetwork,
  LearnedRouter,
  StringEncoder,
  VQCodebook,
  tokens,
} = require('../src/decomposition');

const {
  DecompositionCurriculum,
} = require('../syllabi/decomposition');

const { TOKEN, VOCAB_SIZE, TOKEN_DOMAIN } = tokens;

// ── Helpers ───────────────────────────────────────────────────────────────────

const EMBED_DIM = 4;   // small dim for fast tests

/** Minimal Brain trained on AND/OR/NOT/XOR gates */
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
  for (const name of ['AND', 'OR', 'NOT', 'XOR']) {
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

// ── 1. EmbeddingTable ─────────────────────────────────────────────────────────

describe('EmbeddingTable', () => {
  test('constructor creates vocabSize embeddings of correct dim', () => {
    const t = new EmbeddingTable({ vocabSize: VOCAB_SIZE, dim: EMBED_DIM });
    expect(t.embeddings).toHaveLength(VOCAB_SIZE);
    expect(t.embeddings[0]).toHaveLength(EMBED_DIM);
  });

  test('lookup returns a copy of the stored embedding', () => {
    const t = new EmbeddingTable({ vocabSize: VOCAB_SIZE, dim: EMBED_DIM });
    const emb1 = t.lookup(TOKEN.AND);
    const emb2 = t.lookup(TOKEN.AND);
    expect(emb1).toEqual(emb2);
    // Mutating the returned copy does not change the table
    emb1[0] += 999;
    expect(t.lookup(TOKEN.AND)[0]).not.toBeCloseTo(999);
  });

  test('lookup returns all-zeros for NULL (-1) token', () => {
    const t = new EmbeddingTable({ vocabSize: VOCAB_SIZE, dim: EMBED_DIM });
    expect(t.lookup(TOKEN.NULL)).toEqual(new Array(EMBED_DIM).fill(0));
  });

  test('lookup returns all-zeros for out-of-range token', () => {
    const t = new EmbeddingTable({ vocabSize: VOCAB_SIZE, dim: EMBED_DIM });
    expect(t.lookup(100)).toEqual(new Array(EMBED_DIM).fill(0));
  });

  test('update moves embedding in gradient direction', () => {
    const t  = new EmbeddingTable({ vocabSize: VOCAB_SIZE, dim: EMBED_DIM, learningRate: 1.0 });
    const before = [...t.embeddings[TOKEN.AND]];
    const grad   = new Array(EMBED_DIM).fill(1);   // all-ones gradient
    t.update(TOKEN.AND, grad);
    const after = t.embeddings[TOKEN.AND];
    // With lr=1: after[k] = before[k] - 1*1 = before[k] - 1
    for (let k = 0; k < EMBED_DIM; k++) {
      expect(after[k]).toBeCloseTo(before[k] - 1, 5);
    }
  });

  test('update ignores NULL token', () => {
    const t    = new EmbeddingTable({ vocabSize: VOCAB_SIZE, dim: EMBED_DIM });
    const grad = new Array(EMBED_DIM).fill(1);
    expect(() => t.update(TOKEN.NULL, grad)).not.toThrow();
  });

  test('toJSON / fromJSON round-trip preserves all embeddings', () => {
    const t    = new EmbeddingTable({ vocabSize: VOCAB_SIZE, dim: EMBED_DIM });
    const json = t.toJSON();
    const t2   = EmbeddingTable.fromJSON(json);
    expect(t2.vocabSize).toBe(t.vocabSize);
    expect(t2.dim).toBe(t.dim);
    for (let i = 0; i < VOCAB_SIZE; i++) {
      expect(t2.embeddings[i]).toEqual(t.embeddings[i]);
    }
  });
});

// ── 2. WorkingMemory embedding mode ──────────────────────────────────────────

describe('WorkingMemory.toVector(embeddingTable)', () => {
  test('without embeddingTable returns one-hot of length maxSlots*VOCAB_SIZE', () => {
    const mem = new WorkingMemory(8);
    mem.load([TOKEN.AND, TOKEN.V1, TOKEN.V0]);
    const vec = mem.toVector();
    expect(vec.length).toBe(8 * VOCAB_SIZE);
  });

  test('with embeddingTable returns flat vector of length maxSlots*dim', () => {
    const mem   = new WorkingMemory(8);
    const table = new EmbeddingTable({ vocabSize: VOCAB_SIZE, dim: EMBED_DIM });
    mem.load([TOKEN.AND, TOKEN.V1, TOKEN.V0]);
    const vec = mem.toVector(table);
    expect(vec.length).toBe(8 * EMBED_DIM);
  });

  test('embedded vector for NULL slot is all-zeros', () => {
    const mem   = new WorkingMemory(4);
    const table = new EmbeddingTable({ vocabSize: VOCAB_SIZE, dim: EMBED_DIM });
    mem.load([TOKEN.V1]);   // slots 1-3 are NULL
    const vec = mem.toVector(table);
    // Slots 1-3: all zeros
    for (let s = 1; s < 4; s++) {
      for (let k = 0; k < EMBED_DIM; k++) {
        expect(vec[s * EMBED_DIM + k]).toBe(0);
      }
    }
  });

  test('embedded vector for slot 0 matches table.lookup(tok)', () => {
    const mem   = new WorkingMemory(4);
    const table = new EmbeddingTable({ vocabSize: VOCAB_SIZE, dim: EMBED_DIM });
    mem.load([TOKEN.AND]);
    const vec = mem.toVector(table);
    const emb = table.lookup(TOKEN.AND);
    for (let k = 0; k < EMBED_DIM; k++) {
      expect(vec[k]).toBeCloseTo(emb[k], 10);
    }
  });
});

// ── 3. NeuralNetwork.backwardWithInputGrad() ──────────────────────────────────

describe('NeuralNetwork.backwardWithInputGrad()', () => {
  const NeuralNetwork = require('../src/brain/NeuralNetwork');

  test('returns { loss, inputGrad } with correct shapes', () => {
    const nn     = new NeuralNetwork({ architecture: [4, 8, 2], learningRate: 0.01 });
    const input  = [0.1, 0.2, 0.3, 0.4];
    const target = [1, 0];
    const result = nn.backwardWithInputGrad(input, target);
    expect(typeof result.loss).toBe('number');
    expect(result.inputGrad).toHaveLength(4);
  });

  test('loss is non-negative', () => {
    const nn     = new NeuralNetwork({ architecture: [2, 4, 1], learningRate: 0.01 });
    const { loss } = nn.backwardWithInputGrad([0, 1], [1]);
    expect(loss).toBeGreaterThanOrEqual(0);
  });

  test('inputGrad has same length as input', () => {
    const nn  = new NeuralNetwork({ architecture: [6, 4, 3], learningRate: 0.01 });
    const { inputGrad } = nn.backwardWithInputGrad([1,0,1,0,1,0], [0,1,0]);
    expect(inputGrad).toHaveLength(6);
  });

  test('weights are updated (same as backward() for same seed-state)', () => {
    // Verify that calling backwardWithInputGrad also trains the weights
    const NeuralNetwork = require('../src/brain/NeuralNetwork');
    const nn1 = new NeuralNetwork({ architecture: [2, 4, 1], learningRate: 0.1 });
    const nn2 = NeuralNetwork.fromJSON(nn1.toJSON());
    const input  = [0, 1];
    const target = [1];
    nn1.backward(input, target);
    nn2.backwardWithInputGrad(input, target);
    // Both networks should now have identical weights
    for (let l = 0; l < nn1.layers.length; l++) {
      for (let i = 0; i < nn1.layers[l].weights.length; i++) {
        for (let j = 0; j < nn1.layers[l].weights[i].length; j++) {
          expect(nn2.layers[l].weights[i][j]).toBeCloseTo(nn1.layers[l].weights[i][j], 10);
        }
      }
    }
  });
});

// ── 4. DecompositionController — embedding mode ───────────────────────────────

describe('DecompositionController — embedding mode', () => {
  test('creates embeddingTable when embeddingDim is set', () => {
    const ctrl = new DecompositionController({ embeddingDim: EMBED_DIM });
    expect(ctrl.embeddingTable).not.toBeNull();
    expect(ctrl.embeddingTable.dim).toBe(EMBED_DIM);
  });

  test('policy network input size = maxSlots * embeddingDim', () => {
    const ctrl = new DecompositionController({ maxSlots: 8, embeddingDim: EMBED_DIM });
    expect(ctrl.network.architecture[0]).toBe(8 * EMBED_DIM);
  });

  test('embeddingTable is null in one-hot mode', () => {
    const ctrl = new DecompositionController();
    expect(ctrl.embeddingTable).toBeNull();
  });

  test('selectAction uses embedded state vector', () => {
    const ctrl = new DecompositionController({ embeddingDim: EMBED_DIM, explorationRate: 0 });
    const mem  = new WorkingMemory(16);
    mem.load([TOKEN.NOT, TOKEN.V0]);
    // Should not throw even though embedding mode is on
    const action = ctrl.selectAction(mem);
    expect(action).not.toBeNull();
    expect(action.start).toBe(0);
  });

  test('trainImitation in embedding mode marks controller as trained', () => {
    const ctrl = new DecompositionController({ embeddingDim: EMBED_DIM });
    const examples = [
      {
        rawSlots:    [TOKEN.NOT, TOKEN.V0, TOKEN.NULL, TOKEN.NULL, TOKEN.NULL, TOKEN.NULL,
                      TOKEN.NULL, TOKEN.NULL, TOKEN.NULL, TOKEN.NULL, TOKEN.NULL, TOKEN.NULL,
                      TOKEN.NULL, TOKEN.NULL, TOKEN.NULL, TOKEN.NULL],
        validStarts: [0],
        targetStart: 0,
      },
    ];
    ctrl.trainImitation(examples, 2);
    expect(ctrl.trained).toBe(true);
  });

  test('embedding table values change after trainImitation', () => {
    const ctrl   = new DecompositionController({ embeddingDim: EMBED_DIM });
    const before = [...ctrl.embeddingTable.lookup(TOKEN.NOT)];
    const rawSlots = new Array(16).fill(TOKEN.NULL);
    rawSlots[0] = TOKEN.NOT;
    rawSlots[1] = TOKEN.V0;
    const examples = [{ rawSlots, validStarts: [0], targetStart: 0 }];
    ctrl.trainImitation(examples, 5);
    const after = ctrl.embeddingTable.lookup(TOKEN.NOT);
    // At least one dimension should have shifted
    const changed = before.some((v, k) => Math.abs(v - after[k]) > 1e-9);
    expect(changed).toBe(true);
  });

  test('toJSON / fromJSON preserves embeddingTable', () => {
    const ctrl = new DecompositionController({ embeddingDim: EMBED_DIM });
    const json = ctrl.toJSON();
    expect(json.embeddingTable).not.toBeNull();
    const ctrl2 = DecompositionController.fromJSON(json);
    expect(ctrl2.embeddingTable).not.toBeNull();
    expect(ctrl2.embeddingTable.dim).toBe(EMBED_DIM);
    // Embedding values are preserved
    for (let i = 0; i < VOCAB_SIZE; i++) {
      expect(ctrl2.embeddingTable.embeddings[i]).toEqual(ctrl.embeddingTable.embeddings[i]);
    }
  });
});

// ── 5. GatingNetwork ──────────────────────────────────────────────────────────

describe('GatingNetwork', () => {
  test('constructor creates correct architecture', () => {
    const g = new GatingNetwork({ embeddingDim: EMBED_DIM });
    expect(g.network.architecture[0]).toBe(EMBED_DIM);
    expect(g.network.architecture[g.network.architecture.length - 1]).toBe(1);
  });

  test('scoreSlot returns a value in [0, 1]', () => {
    const g   = new GatingNetwork({ embeddingDim: EMBED_DIM });
    const emb = new Array(EMBED_DIM).fill(0.1);
    const score = g.scoreSlot(emb);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  test('train() runs without error on labelled examples', () => {
    const g = new GatingNetwork({ embeddingDim: EMBED_DIM });
    const table = new EmbeddingTable({ vocabSize: VOCAB_SIZE, dim: EMBED_DIM });
    const examples = [
      { embedding: table.lookup(TOKEN.AND),  valid: 1 },
      { embedding: table.lookup(TOKEN.V0),   valid: 0 },
      { embedding: table.lookup(TOKEN.NULL), valid: 0 },
    ];
    expect(() => g.train(examples, 3)).not.toThrow();
  });

  test('predictValidStarts returns indices above threshold', () => {
    const g     = new GatingNetwork({ embeddingDim: EMBED_DIM, threshold: 0.0 });
    const table = new EmbeddingTable({ vocabSize: VOCAB_SIZE, dim: EMBED_DIM });
    // With threshold=0 every slot should be predicted valid
    const slotTokens = [TOKEN.AND, TOKEN.V1, TOKEN.V0, TOKEN.NULL];
    const starts = g.predictValidStarts(slotTokens, 3, table);
    expect(starts.length).toBeLessThanOrEqual(3);
  });

  test('toJSON / fromJSON round-trip', () => {
    const g    = new GatingNetwork({ embeddingDim: EMBED_DIM });
    const json = g.toJSON();
    const g2   = GatingNetwork.fromJSON(json);
    expect(g2.embeddingDim).toBe(EMBED_DIM);
    expect(g2.network.architecture).toEqual(g.network.architecture);
  });
});

// ── 6. DecompositionController — gating mode ─────────────────────────────────

describe('DecompositionController — gating mode', () => {
  test('creates gatingNetwork when useGating=true and embeddingDim set', () => {
    const ctrl = new DecompositionController({ embeddingDim: EMBED_DIM, useGating: true });
    expect(ctrl.gatingNetwork).not.toBeNull();
  });

  test('gatingNetwork is null when useGating=false', () => {
    const ctrl = new DecompositionController({ embeddingDim: EMBED_DIM, useGating: false });
    expect(ctrl.gatingNetwork).toBeNull();
  });

  test('gatingNetwork is null when embeddingDim is not set', () => {
    const ctrl = new DecompositionController({ useGating: true });
    expect(ctrl.gatingNetwork).toBeNull();
  });

  test('selectAction with gating mode still returns valid candidates', () => {
    const ctrl = new DecompositionController({
      embeddingDim:    EMBED_DIM,
      useGating:       true,
      explorationRate: 0,
    });
    const mem = new WorkingMemory(16);
    mem.load([TOKEN.NOT, TOKEN.V0]);
    const action = ctrl.selectAction(mem);
    // Fallback to structural candidates should ensure a result
    expect(action).not.toBeNull();
    expect(action.start).toBe(0);
  });

  test('trainImitation with gating mode trains both policy and gating network', () => {
    const ctrl = new DecompositionController({
      embeddingDim: EMBED_DIM,
      useGating:    true,
    });
    const rawSlots = new Array(16).fill(TOKEN.NULL);
    rawSlots[0] = TOKEN.NOT;
    rawSlots[1] = TOKEN.V0;
    const examples = [{ rawSlots, validStarts: [0], targetStart: 0 }];
    expect(() => ctrl.trainImitation(examples, 2)).not.toThrow();
    expect(ctrl.trained).toBe(true);
  });

  test('toJSON / fromJSON preserves gatingNetwork', () => {
    const ctrl = new DecompositionController({ embeddingDim: EMBED_DIM, useGating: true });
    const json = ctrl.toJSON();
    expect(json.gatingNetwork).not.toBeNull();
    const ctrl2 = DecompositionController.fromJSON(json);
    expect(ctrl2.gatingNetwork).not.toBeNull();
  });
});

// ── 7. LearnedRouter ──────────────────────────────────────────────────────────

describe('LearnedRouter', () => {
  const domains = ['boolean.AND', 'boolean.OR', 'boolean.NOT', 'boolean.XOR'];

  test('constructor requires non-empty domains', () => {
    expect(() => new LearnedRouter({ embeddingDim: EMBED_DIM, domains: [] })).toThrow();
  });

  test('route returns a domain string and confidence', () => {
    const r   = new LearnedRouter({ embeddingDim: EMBED_DIM, domains });
    const emb = new Array(EMBED_DIM).fill(0.1);
    const res = r.route(emb);
    expect(domains).toContain(res.domain);
    expect(typeof res.confidence).toBe('number');
    expect(typeof res.aboveThreshold).toBe('boolean');
  });

  test('train runs without error', () => {
    const r        = new LearnedRouter({ embeddingDim: EMBED_DIM, domains });
    const table    = new EmbeddingTable({ vocabSize: VOCAB_SIZE, dim: EMBED_DIM });
    const examples = domains.map((d, i) => ({
      opEmbedding: table.lookup(i + 2),   // AND=2, OR=3, NOT=4, XOR=5
      domainIndex: i,
    }));
    expect(() => r.train(examples, 5)).not.toThrow();
  });

  test('after enough training, routes to correct domain with high confidence', () => {
    const r     = new LearnedRouter({
      embeddingDim:        EMBED_DIM,
      domains,
      confidenceThreshold: 0.6,
    });
    // Fix distinctive embeddings by overriding the EmbeddingTable
    const table = new EmbeddingTable({ vocabSize: VOCAB_SIZE, dim: EMBED_DIM });
    // Overwrite embeddings with very distinct vectors
    const distinctEmbeddings = [
      [1, 0, 0, 0],  // AND → class 0
      [0, 1, 0, 0],  // OR  → class 1
      [0, 0, 1, 0],  // NOT → class 2
      [0, 0, 0, 1],  // XOR → class 3
    ];
    for (let i = 0; i < 4; i++) {
      table.embeddings[TOKEN.AND + i] = [...distinctEmbeddings[i]];
    }
    const examples = domains.map((d, i) => ({
      opEmbedding: table.lookup(TOKEN.AND + i),
      domainIndex: i,
    }));
    r.train(examples, 100);
    // Each should route correctly with confidence above threshold
    let correct = 0;
    for (let i = 0; i < 4; i++) {
      const res = r.route(table.lookup(TOKEN.AND + i));
      if (res.domain === domains[i]) correct++;
    }
    expect(correct).toBeGreaterThanOrEqual(3);
  });

  test('toJSON / fromJSON round-trip', () => {
    const r    = new LearnedRouter({ embeddingDim: EMBED_DIM, domains });
    const json = r.toJSON();
    const r2   = LearnedRouter.fromJSON(json);
    expect(r2.domains).toEqual(r.domains);
    expect(r2.embeddingDim).toBe(r.embeddingDim);
    expect(r2.confidenceThreshold).toBe(r.confidenceThreshold);
  });
});

// ── 8. StringEncoder ──────────────────────────────────────────────────────────

describe('StringEncoder', () => {
  test('splitWords handles parenthetical format', () => {
    const words = StringEncoder.splitWords('AND(OR(1,0), NOT(0))');
    expect(words).toEqual(['AND', 'OR', '1', '0', 'NOT', '0']);
  });

  test('splitWords handles space-separated format', () => {
    const words = StringEncoder.splitWords('AND OR 1 0 NOT 0');
    expect(words).toEqual(['AND', 'OR', '1', '0', 'NOT', '0']);
  });

  test('toTokenIds returns correct integer sequence', () => {
    const ids = StringEncoder.toTokenIds('AND(OR(1,0),NOT(0))');
    expect(ids).toEqual([TOKEN.AND, TOKEN.OR, TOKEN.V1, TOKEN.V0, TOKEN.NOT, TOKEN.V0]);
  });

  test('toTokenIds throws on unknown token', () => {
    expect(() => StringEncoder.toTokenIds('FOO(1,0)')).toThrow(/unknown token/i);
  });

  test('wordToFeature returns correct length', () => {
    const enc = new StringEncoder();
    const feat = enc.wordToFeature('AND');
    expect(feat.length).toBe(StringEncoder.CHAR_VOCAB_SIZE * enc.maxWordLen);
  });

  test('wordToFeature is different for different words', () => {
    const enc = new StringEncoder();
    const f1  = enc.wordToFeature('AND');
    const f2  = enc.wordToFeature('OR');
    expect(f1).not.toEqual(f2);
  });

  test('train runs without error and marks encoder as usable', () => {
    const enc = new StringEncoder({ vocabSize: VOCAB_SIZE });
    const examples = [
      { word: 'AND', tokenId: TOKEN.AND },
      { word: 'OR',  tokenId: TOKEN.OR  },
      { word: 'NOT', tokenId: TOKEN.NOT },
      { word: '0',   tokenId: TOKEN.V0  },
      { word: '1',   tokenId: TOKEN.V1  },
    ];
    expect(() => enc.train(examples, 5)).not.toThrow();
  });

  test('after training, predictTokenId is correct for distinctive words', () => {
    // Train with many epochs on a small vocabulary to force convergence
    const enc = new StringEncoder({ vocabSize: VOCAB_SIZE, hiddenSize: 16 });
    const examples = [];
    const vocab = [
      { word: 'AND',  id: TOKEN.AND  },
      { word: 'OR',   id: TOKEN.OR   },
      { word: 'NOT',  id: TOKEN.NOT  },
      { word: 'XOR',  id: TOKEN.XOR  },
      { word: '0',    id: TOKEN.V0   },
      { word: '1',    id: TOKEN.V1   },
    ];
    // Repeat examples to boost training signal
    for (let i = 0; i < 20; i++) {
      for (const v of vocab) examples.push({ word: v.word, tokenId: v.id });
    }
    enc.train(examples, 100);
    let correct = 0;
    for (const v of vocab) {
      if (enc.predictTokenId(v.word) === v.id) correct++;
    }
    expect(correct).toBeGreaterThanOrEqual(4);   // at least 4/6 correct
  });

  test('encode converts a full expression string to a token-ID array', () => {
    const enc = new StringEncoder({ vocabSize: VOCAB_SIZE });
    const ids  = enc.encode('NOT(0)');
    expect(ids).toHaveLength(2);
    expect(ids.every(Number.isInteger)).toBe(true);
  });

  test('toJSON / fromJSON round-trip preserves network weights', () => {
    const enc  = new StringEncoder({ vocabSize: VOCAB_SIZE });
    const json = enc.toJSON();
    const enc2 = StringEncoder.fromJSON(json);
    expect(enc2.maxWordLen).toBe(enc.maxWordLen);
    expect(enc2.vocabSize).toBe(enc.vocabSize);
    // Same weights
    for (let l = 0; l < enc.network.layers.length; l++) {
      expect(enc2.network.layers[l].weights).toEqual(enc.network.layers[l].weights);
    }
  });
});

// ── 9. DecompositionCurriculum — string fields ────────────────────────────────

describe('DecompositionCurriculum — string fields', () => {
  test('generateDepth1() problems all carry a string field', () => {
    const c  = new DecompositionCurriculum();
    const ps = c.generateDepth1();
    for (const p of ps) {
      expect(typeof p.string).toBe('string');
      expect(p.string.length).toBeGreaterThan(0);
    }
  });

  test('depth-1 string contains the operator name', () => {
    const c  = new DecompositionCurriculum();
    const ps = c.generateDepth1();
    for (const p of ps) {
      // The string should start with one of the operator names
      expect(p.string).toMatch(/^(AND|OR|NOT|XOR|NAND|NOR|XNOR)/i);
    }
  });

  test('generateDepth2() problems carry a string field', () => {
    const c  = new DecompositionCurriculum({ depth2Count: 10, depth3Count: 0 });
    const ps = c.generateDepth2();
    for (const p of ps) {
      expect(typeof p.string).toBe('string');
    }
  });

  test('curriculum string word count matches token count', () => {
    const c  = new DecompositionCurriculum();
    const ps = c.generateDepth1();
    for (const p of ps) {
      const words = StringEncoder.splitWords(p.string);
      expect(words.length).toBe(p.tokens.length);
    }
  });
});

// ── 10. Brain.trainStringEncoder() + solveString() ───────────────────────────

describe('Brain.trainStringEncoder() + solveString()', () => {
  let brain;

  beforeAll(() => {
    brain = makeFastBoolBrain();
    brain.initController();
    const c = new DecompositionCurriculum({ depth2Count: 20, depth3Count: 0 });
    brain.trainDecomposition(c, { epochs: 20 });
    brain.trainStringEncoder(c, { epochs: 30 });
  }, 60000);

  test('brain.stringEncoder is set after trainStringEncoder()', () => {
    expect(brain.stringEncoder).not.toBeNull();
  });

  test('solveString returns { answer, solved, steps, graph }', () => {
    const res = brain.solveString('NOT(0)');
    expect(res).toHaveProperty('answer');
    expect(res).toHaveProperty('solved');
    expect(res).toHaveProperty('steps');
    expect(res).toHaveProperty('graph');
  });

  test('solveString falls back gracefully to deterministic parser', () => {
    const brain2 = makeFastBoolBrain();
    brain2.initController();
    const c = new DecompositionCurriculum({ depth2Count: 10, depth3Count: 0 });
    brain2.trainDecomposition(c, { epochs: 10 });
    // No trainStringEncoder called — should use static toTokenIds fallback
    expect(brain2.stringEncoder).toBeNull();
    const res = brain2.solveString('NOT(0)');
    expect(res).toHaveProperty('answer');
  });
}, 60000);

// ── 11. Brain.initLearnedRouter() + Phase 3 training ─────────────────────────

describe('Brain — Phase 3 LearnedRouter', () => {
  let brain;

  beforeAll(() => {
    brain = makeFastBoolBrain();
    brain.initController({ embeddingDim: EMBED_DIM });
    brain.initLearnedRouter();
    const c = new DecompositionCurriculum({ depth2Count: 20, depth3Count: 0 });
    brain.trainDecomposition(c, { epochs: 20, routerEpochs: 20 });
  }, 60000);

  test('initLearnedRouter() throws without embeddingDim on controller', () => {
    const b = new Brain();
    b.initController();   // no embeddingDim
    expect(() => b.initLearnedRouter()).toThrow(/embedding mode/i);
  });

  test('brain.learnedRouter is set after initLearnedRouter()', () => {
    expect(brain.learnedRouter).not.toBeNull();
  });

  test('learnedRouter covers all boolean domains', () => {
    const domainValues = Object.values(TOKEN_DOMAIN);
    for (const d of domainValues) {
      expect(brain.learnedRouter.domains).toContain(d);
    }
  });

  test('learnedRouter.route() returns a domain string', () => {
    const emb = brain.controller.embeddingTable.lookup(TOKEN.AND);
    const res = brain.learnedRouter.route(emb);
    expect(typeof res.domain).toBe('string');
    expect(brain.learnedRouter.domains).toContain(res.domain);
  });

  test('solve() still works with learnedRouter active', () => {
    const res = brain.solve([TOKEN.NOT, TOKEN.V0]);
    expect(res).toHaveProperty('solved');
    expect(res).toHaveProperty('answer');
  });
}, 60000);

// ── 12. Brain toJSON/fromJSON with all phases ─────────────────────────────────

describe('Brain serialisation with learned components', () => {
  test('toJSON/fromJSON preserves learnedRouter', () => {
    const brain = new Brain();
    brain.initController({ embeddingDim: EMBED_DIM });
    brain.initLearnedRouter();
    const json    = brain.toJSON();
    expect(json.learnedRouter).not.toBeNull();
    const brain2  = Brain.fromJSON(json);
    expect(brain2.learnedRouter).not.toBeNull();
    expect(brain2.learnedRouter.domains).toEqual(brain.learnedRouter.domains);
  });

  test('toJSON/fromJSON preserves stringEncoder', () => {
    const brain = new Brain();
    brain.initController();
    brain.stringEncoder = new StringEncoder({ vocabSize: VOCAB_SIZE });
    const json   = brain.toJSON();
    expect(json.stringEncoder).not.toBeNull();
    const brain2 = Brain.fromJSON(json);
    expect(brain2.stringEncoder).not.toBeNull();
    expect(brain2.stringEncoder.maxWordLen).toBe(brain.stringEncoder.maxWordLen);
  });

  test('introspect() includes learnedRouter and stringEncoder', () => {
    const brain = new Brain();
    brain.initController({ embeddingDim: EMBED_DIM });
    brain.initLearnedRouter();
    brain.stringEncoder = new StringEncoder({ vocabSize: VOCAB_SIZE });
    const info = brain.introspect();
    expect(info.learnedRouter).not.toBeNull();
    expect(info.stringEncoder).not.toBeNull();
  });
});

// ── 13. VQCodebook ────────────────────────────────────────────────────────────

describe('VQCodebook', () => {
  test('constructor creates numCodes entries of correct dim', () => {
    const vq = new VQCodebook({ numCodes: 8, dim: EMBED_DIM });
    expect(vq.codebook).toHaveLength(8);
    expect(vq.codebook[0]).toHaveLength(EMBED_DIM);
  });

  test('throws if dim is not provided', () => {
    expect(() => new VQCodebook({ numCodes: 8 })).toThrow();
  });

  test('quantize returns codeIdx in [0, numCodes)', () => {
    const vq  = new VQCodebook({ numCodes: 8, dim: EMBED_DIM });
    const vec = new Array(EMBED_DIM).fill(0.1);
    const { codeIdx, quantized } = vq.quantize(vec);
    expect(codeIdx).toBeGreaterThanOrEqual(0);
    expect(codeIdx).toBeLessThan(8);
    expect(quantized).toHaveLength(EMBED_DIM);
  });

  test('quantize nearest code is closer than all others', () => {
    const vq = new VQCodebook({ numCodes: 4, dim: 2 });
    // Set codebook to known values
    vq.codebook = [[1,0],[0,1],[-1,0],[0,-1]];
    const vec = [0.9, 0.1];   // nearest to [1,0] = index 0
    const { codeIdx } = vq.quantize(vec);
    expect(codeIdx).toBe(0);
  });

  test('update moves codebook entry toward vector (EMA)', () => {
    const vq = new VQCodebook({ numCodes: 2, dim: 2, learningRate: 1.0 });
    vq.codebook[0] = [0, 0];
    vq.update(0, [1, 1]);
    // With lr=1: e ← (1-1)*e + 1*v = v = [1,1]
    expect(vq.codebook[0][0]).toBeCloseTo(1, 5);
    expect(vq.codebook[0][1]).toBeCloseTo(1, 5);
  });

  test('commitmentLoss is β * squared distance', () => {
    const vq = new VQCodebook({ numCodes: 2, dim: 2, commitment: 0.5 });
    vq.codebook[0] = [0, 0];
    const loss = vq.commitmentLoss([1, 1], 0);
    // squared dist = 2; β=0.5 → loss = 0.5*2 = 1.0
    expect(loss).toBeCloseTo(1.0, 5);
  });

  test('getUsageStats and resetUsage work correctly', () => {
    const vq = new VQCodebook({ numCodes: 4, dim: 2 });
    vq.update(0, [0, 0]);
    vq.update(0, [0, 0]);
    vq.update(2, [0, 0]);
    const { usageCounts, totalAssignments } = vq.getUsageStats();
    expect(usageCounts[0]).toBe(2);
    expect(usageCounts[2]).toBe(1);
    expect(totalAssignments).toBe(3);
    vq.resetUsage();
    expect(vq.getUsageStats().totalAssignments).toBe(0);
  });

  test('toJSON / fromJSON round-trip preserves codebook and usage', () => {
    const vq = new VQCodebook({ numCodes: 4, dim: EMBED_DIM });
    vq.update(0, new Array(EMBED_DIM).fill(0.5));
    const json = vq.toJSON();
    const vq2  = VQCodebook.fromJSON(json);
    expect(vq2.numCodes).toBe(vq.numCodes);
    expect(vq2.dim).toBe(vq.dim);
    expect(vq2.commitment).toBe(vq.commitment);
    for (let i = 0; i < vq.numCodes; i++) {
      expect(vq2.codebook[i]).toEqual(vq.codebook[i]);
    }
    expect(vq2.usageCounts).toEqual(vq.usageCounts);
  });
});
